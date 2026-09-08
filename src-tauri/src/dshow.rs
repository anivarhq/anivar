//! Cross-platform USB / integrated-camera capture via ffmpeg.
//!
//!   * Windows → DirectShow   (`-f dshow  -i video="<name>"`)
//!   * macOS   → AVFoundation (`-f avfoundation -i "<index>:none"`)
//!   * Linux   → Video4Linux2 (`-f v4l2 -i /dev/videoN`)
//!
//! (Module is historically named `dshow`; the Tauri command names are kept so the
//! frontend wiring is unchanged.) One ffmpeg per camera with TWO outputs (mature NVRs'
//! model): (A) NVR segments recorded DIRECTLY in-process — camera + mic demuxed from
//! one input on ONE clock, so A/V sync is inherent (the old separate pipe-recorder
//! stamped CFR video against a live mic → 86 ms/segment drift + constant offset) —
//! and (B) the HD MJPEG pipe → `frame_txs` (live `/stream`) + `fan_out_frame`
//! (HLS / inference) + drop-don't-queue detection. The record path keeps the
//! camera's native VFR timestamps (NO fps-filter duplication — every recorded frame
//! is a real frame; the old fps-normalize pad turned frame drops into judder).
//!
//! IMPORTANT (maintenance): only `camera_input_args` and `list_cameras_impl` are
//! OS-specific. EVERYTHING ELSE lives in the shared `spawn_capture`, which is compiled
//! on every OS — so a Windows `cargo check` verifies the bulk of this file. Keep the
//! per-OS helpers tiny (string building only) since they can't be compile-checked from
//! another OS.

use std::sync::Arc;
use base64::Engine as _;
use tauri::{State, Emitter};
use crate::AppState;
use crate::hw::hw_encoder_args;

/// List connected cameras as device identifiers — these are BOTH shown to the user and
/// passed back verbatim to `start_dshow_camera`.
#[tauri::command]
pub async fn list_dshow_cameras(state: State<'_, Arc<AppState>>) -> Result<Vec<String>, String> {
    let ffmpeg = crate::ensure_ffmpeg(&state.data_dir).await.map_err(|e| e.to_string())?;
    list_cameras_impl(&ffmpeg).await
}

/// Start a USB/integrated camera capture. `device_name` is the identifier from
/// `list_dshow_cameras` (Windows: device name · macOS: name or index · Linux: /dev path).
#[tauri::command]
pub async fn start_dshow_camera(
    state: State<'_, Arc<AppState>>,
    cam_id: u8,
    device_name: String,
) -> Result<(), String> {
    start_usb_capture(state, cam_id, device_name).await
}

/// Command body, extracted so the boot watchdog / footage-delete respawn can call it
/// with a `State` obtained from the app handle.
pub(crate) async fn start_usb_capture(
    state: State<'_, Arc<AppState>>,
    cam_id: u8,
    device_name: String,
) -> Result<(), String> {
    let cam = cam_id.min(15);
    // A REMOVED/disabled camera must never re-open its device, no matter who
    // asks (stale localStorage auto-start, retried UI call, old event handler).
    if !crate::cam_config::camera_start_allowed(&state.db, cam).await {
        tracing::info!("usb capture cam{cam}: refused — camera is removed/disabled");
        return Err("camera is disabled".into());
    }
    let ffmpeg = crate::ensure_ffmpeg(&state.data_dir).await.map_err(|e| e.to_string())?;
    // Windows dshow selects by NAME, but old camera_configs rows stored a numeric
    // index in device_id — boot then spawned `video=0`, dshow rejected it, and
    // the cam recorded NOTHING until the UI resolved the real name. Resolve
    // index → name here so headless boot works (macOS path already does this).
    let device_name = if cfg!(windows) && !device_name.is_empty()
        && device_name.chars().all(|c| c.is_ascii_digit())
    {
        let names = list_cameras_impl(&ffmpeg).await.unwrap_or_default();
        let idx: usize = device_name.parse().unwrap_or(0);
        match names.get(idx) {
            Some(n) => { tracing::info!("cam{cam}: resolved device index {device_name} -> {n}"); n.clone() }
            None => device_name,
        }
    } else { device_name };
    // Per-OS audio device for server-side YAMNet (None if none / unsupported). The shared
    // detector itself no-ops unless audio detection is enabled + the skill is installed.
    let audio_args = audio_input_args(&ffmpeg).await;
    spawn_capture(state, cam, format!("usb:{device_name}"), audio_args, &ffmpeg, &device_name).await
}

// ─── Shared capture pipeline (compiled on ALL platforms) ─────────────────────────
async fn spawn_capture(
    state: State<'_, Arc<AppState>>,
    cam: u8,
    key: String,
    audio_args: Option<Vec<String>>,
    ffmpeg: &std::path::Path,
    device_label: &str,
) -> Result<(), String> {
    // Serialize starts so two near-simultaneous calls (grid tile + focused view) can't
    // both pass the idempotency check before either records its capture_key — the TOCTOU
    // race that spawned duplicate ffmpegs. Held for the whole start.
    let _start_guard = state.capture_start_lock.lock().await;

    // IDEMPOTENT START: if this exact capture is already running, do NOTHING (avoids the
    // kill+respawn that gapped recording on every re-render). Only restart if the process
    // actually died or the device changed.
    if state.capture_keys.lock().await.get(&cam) == Some(&key) {
        let mut procs = state.rtsp_processes.lock().await;
        if let Some(child) = procs.get_mut(&cam) {
            if matches!(child.try_wait(), Ok(None)) {
                tracing::debug!("usb cam{}: already capturing {} — reusing", cam, device_label);
                return Ok(());
            }
        }
    }
    // Different device or dead process: tear the old one down, then respawn.
    crate::rtsp::stop_rtsp_relay(State::clone(&state), cam).await.ok();

    // ── Merged capture + record (mature NVRs model: one process, one clock) ──
    let nvr_on = state.settings.read().await.nvr_enabled;
    // Depth Map Anonymization: the capture ffmpeg must NOT record raw pixels.
    // It runs pipe-only; the app converts frames to colorized depth and feeds
    // a pipe recorder + every external consumer. Raw stays in RAM for local AI.
    let depth_on = crate::depth::is_anonymized(cam);
    let record_in_ffmpeg = nvr_on && !depth_on;
    if depth_on && !crate::depth::model_installed(&state.data_dir) {
        // FAIL-CLOSED: anonymization was promised; without the model we serve
        // and record NOTHING rather than silently leaking raw pixels.
        tracing::warn!("cam{cam}: depth anonymization ON but model NOT installed — feed/recording stay BLANK until the Depth skill is installed (fail-closed)");
    }
    let encoder = crate::nvr_pipes::assign_encoder(cam, &state.hw_encoder.read().unwrap().clone());
    let (av_input, audio_map) = camera_av_input_args(ffmpeg, device_label).await
        .unwrap_or((Vec::new(), None));
    let v_input = camera_input_args(ffmpeg, device_label).await?;
    if nvr_on {
        let _ = tokio::fs::create_dir_all(state.data_dir.join("nvr")).await;
        // The tee output also writes the live HLS playlist (same single encode).
        let _ = tokio::fs::create_dir_all(state.data_dir.join("hls")).await;
    }

    // `with_audio` selects the combined A/V input; false = video-only (fail-safe).
    let build_args = |with_audio: bool| -> Vec<String> {
        let mut a: Vec<String> = vec!["-hide_banner".into(), "-loglevel".into(), "error".into(), "-y".into()];
        let use_av = with_audio && audio_map.is_some() && !av_input.is_empty();
        // THREAD CAPS: with no caps this one process reached ~110 threads on a
        // 28-core box — the MJPEG decoder, the filter graph, AND the MJPEG pipe
        // encoder each default to threads≈cores. 720p30 needs a handful.
        a.extend(["-threads".into(), "4".into()]); // input option: decoder pool
        a.extend(if use_av { av_input.to_vec() }
                 else { v_input.to_vec() });
        a.extend([
            "-filter_threads".into(), "2".into(),
            "-filter_complex_threads".into(), "2".into(),
        ]);
        if record_in_ffmpeg {
            // Record path keeps NATIVE VFR timestamps (real frames only, zero
            // duplication); the pipe path alone gets fps-normalized cadence.
            a.extend([
                "-filter_complex".into(),
                "[0:v]scale=1280:720:force_original_aspect_ratio=decrease,split=2[rec][p0];[p0]fps=30[pipe]".into(),
            ]);
            // Output A: NVR segments, encoded in-process (one clock = A/V sync).
            a.extend(["-map".into(), "[rec]".into()]);
            if use_av {
                if let Some(am) = &audio_map { a.extend(["-map".into(), am.clone()]); }
            }
            a.extend(hw_encoder_args(&encoder));
            if use_av {
                a.extend([
                    "-c:a".into(), "aac".into(), "-b:a".into(), "96k".into(),
                    // Continuously pad/trim the mic against the shared clock —
                    // kills the 86 ms/segment audio-shorter-than-video drift.
                    "-af".into(), "aresample=async=1:first_pts=0".into(),
                ]);
            }
            // ONE encode → TWO muxers (tee): NVR segments + live HLS. This is
            // the mature NVRs "encode once" rule adapted for USB (raw) sources —
            // the old separate HLS ffmpeg re-encoded the MJPEG pipe, spending a
            // SECOND NVENC session per camera (2/cam × 16 cams = 32 vs the 8-12
            // session limit). Requires current_dir = data dir (relative paths).
            a.extend(crate::nvr_pipes::tee_segment_hls_spec(cam));
            // Output B: the HD MJPEG pipe (byte-identical consumer contract).
            a.extend(["-map".into(), "[pipe]".into()]);
        } else {
            a.extend(["-map".into(), "0:v".into(),
                "-vf".into(), "fps=30,scale=1280:720:force_original_aspect_ratio=decrease".into()]);
        }
        a.extend(["-f".into(), "image2pipe".into(), "-vcodec".into(), "mjpeg".into(),
                  "-q:v".into(), "3".into(),
                  "-threads".into(), "2".into(), // MJPEG encode pool (default ≈ cores)
                  "pipe:1".into()]);
        a
    };

    let data_dir_for_spawn = state.data_dir.clone();
    let spawn_one = move |args: Vec<String>| {
        crate::proc::tokio_cmd(ffmpeg)
            .args(&args)
            // tee output paths are RELATIVE to the data dir (a `C:` drive colon
            // inside tee's bracket options breaks its parser) — see
            // tee_segment_hls_spec. Harmless for the non-recording variant.
            .current_dir(&data_dir_for_spawn)
            .stdout(std::process::Stdio::piped())
            // stderr piped → WARN log below: recording errors are now diagnosable
            // (the old null'd stderr hid dshow "buffer full, dropping" for months).
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true)
            .spawn()
    };

    let want_audio = record_in_ffmpeg && audio_map.is_some();
    let mut child = spawn_one(build_args(true))
        .map_err(|e| format!("ffmpeg capture failed to start: {e}"))?;
    // FAIL-SAFE: combined A/V input dead at startup (mic busy/exclusive/invalid
    // device string) → respawn video-only so CAPTURE + RECORDING are never lost
    // to an audio problem (parity with the old recorder's mic fail-safe).
    if want_audio {
        tokio::time::sleep(std::time::Duration::from_millis(400)).await;
        if matches!(child.try_wait(), Ok(Some(_)) | Err(_)) {
            tracing::warn!("cam{cam}: A/V capture exited at startup — falling back to video-only capture");
            child = spawn_one(build_args(false))
                .map_err(|e| format!("ffmpeg capture fallback failed: {e}"))?;
        }
    }
    if let Some(stderr) = child.stderr.take() {
        tokio::spawn(async move {
            use tokio::io::{AsyncBufReadExt, BufReader};
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let l = line.trim();
                if !l.is_empty() { tracing::warn!("capture ffmpeg cam{cam}: {l}"); }
            }
        });
    }
    if record_in_ffmpeg {
        // Segments finalize+index with no legacy recorder alive.
        crate::nvr_pipes::ensure_postprocessor(&state.data_dir,
            state.app_handle.clone(), state.db.clone()).await;
        // Evict any legacy pipe recorder for this cam: recording is capture-owned
        // now. Also removes it from the legacy watchdog's key set (no thrash).
        if let Some(mut old) = state.nvr_processes.lock().await.remove(&cam) {
            let _ = old.kill().await;
            tracing::info!("cam{cam}: legacy pipe recorder retired (capture records directly)");
        }
        state.nvr_pipe_txs.lock().await.remove(&cam);
        // Evict any legacy HLS encoder too: the tee output above writes this
        // cam's m3u8 now — two writers on the same playlist corrupt it.
        if let Some(mut old) = state.hls_processes.lock().await.remove(&cam) {
            let _ = old.kill().await;
            tracing::info!("cam{cam}: legacy HLS encoder retired (tee writes HLS)");
        }
        state.hls_pipe_txs.lock().await.remove(&cam);
    } else if depth_on && nvr_on {
        // Recording continues under anonymization: a pipe recorder consumes the
        // DEPTH frames the worker emits (30fps CFR-paced, so its -framerate 30
        // stamping stays honest). Mic still recorded (visual anonymization).
        if let Some(a) = &audio_args { crate::nvr_pipes::set_nvr_mic(cam, Some(a.clone())); }
        if let Some(mut old) = state.nvr_processes.lock().await.remove(&cam) { let _ = old.kill().await; }
        state.nvr_pipe_txs.lock().await.remove(&cam);
        let enc2 = state.hw_encoder.read().unwrap().clone();
        let seg_mins = state.settings.read().await.nvr_segment_mins;
        match crate::spawn_nvr_pipe(cam, &state.data_dir, seg_mins, &enc2,
            state.app_handle.clone(), state.db.clone()).await
        {
            Ok((tx, child)) => {
                state.nvr_pipe_txs.lock().await.insert(cam, tx);
                state.nvr_processes.lock().await.insert(cam, child);
                tracing::info!("cam{cam}: depth-anonymized recording online (pipe recorder)");
            }
            Err(e) => tracing::warn!("cam{cam}: depth recorder spawn failed: {e}"),
        }
        // Depth cams keep the legacy HLS encoder (the depth worker feeds it
        // anonymized frames). Boot no longer spawns HLS for every cam, so a
        // runtime depth-toggle must bring it up here.
        if !state.hls_pipe_txs.lock().await.contains_key(&cam) {
            match crate::spawn_hls_pipe(cam, &state.data_dir, &enc2).await {
                Ok((tx, child)) => {
                    state.hls_pipe_txs.lock().await.insert(cam, tx);
                    state.hls_processes.lock().await.insert(cam, child);
                }
                Err(e) => tracing::warn!("cam{cam}: depth HLS spawn failed: {e}"),
            }
        }
    }

    // Depth worker: receives the newest RAW frame, emits colorized depth to every
    // external consumer at a steady cadence. Created per capture spawn; retires
    // itself when this reader drops the sender.
    let depth_raw_tx: Option<tokio::sync::watch::Sender<Option<Arc<Vec<u8>>>>> = if depth_on {
        let (tx, rx) = tokio::sync::watch::channel::<Option<Arc<Vec<u8>>>>(None);
        crate::depth::spawn_depth_worker(Arc::clone(&state), cam, rx);
        Some(tx)
    } else { None };

    let stdout = child.stdout.take().ok_or("no stdout")?;
    let state_arc = Arc::clone(&state);
    // DROP-DON'T-QUEUE backpressure (mature NVRs model). The previous design spawned a
    // process_frame_inner task per frame; under load they piled up (each holding a 720p
    // frame) → OOM, and the backlog delayed frames → slow-motion. Here at most ONE
    // detection runs at a time; excess frames are dropped, not queued. Recording stays
    // full-rate (cheap fan_out) with correct real-time timestamps.
    let busy = Arc::new(std::sync::atomic::AtomicBool::new(false));
    tokio::spawn(async move {
        use tokio::io::AsyncReadExt;
        // CHUNKED reader. The old loop read the pipe ONE BYTE per await -- ~6
        // million await-points/second at 30fps 720p. Under executor load those
        // yields starve, the pipe blocks, ffmpeg's rtbuffer fills, dshow silently
        // drops frames (stderr was null'd), and the fps filter papered it over
        // with duplicates -> the chronic "3 real fps" choppiness. Now: 64 KiB
        // reads + an O(n) resumable scan for JPEG SOI/EOI markers.
        let mut reader = stdout;
        let mut chunk = vec![0u8; 64 * 1024];
        let mut pending: Vec<u8> = Vec::with_capacity(1 << 20);
        let mut scan = 0usize; // resume position for the EOI scan
        // Ingest-rate telemetry: an underdelivering camera can never hide again.
        let mut win_start = std::time::Instant::now();
        let mut win_frames: u32 = 0;
        let mut first_report_done = false;
        // Motion/detection trigger rate gate (~7 fps): running the grayscale
        // diff on all 30 ingest fps buys nothing (mature NVRs' detect model runs
        // ~5 fps). The busy flag alone let a fast diff re-run every frame.
        let mut last_det = std::time::Instant::now() - std::time::Duration::from_secs(1);
        loop {
            let n = match reader.read(&mut chunk).await {
                Ok(0) | Err(_) => return,
                Ok(n) => n,
            };
            pending.extend_from_slice(&chunk[..n]);
            loop {
                // Align to SOI (FF D8): drop any leading garbage.
                match pending.windows(2).position(|w| w == [0xFF, 0xD8]) {
                    Some(0) => {}
                    Some(i) => { pending.drain(..i); scan = 0; }
                    None => {
                        let keep = pending.len().saturating_sub(1);
                        pending.drain(..keep);
                        scan = 0;
                        break;
                    }
                }
                // Find EOI (FF D9) after the SOI, resuming where the last scan ended.
                let from = scan.max(2);
                let eoi = pending[from.saturating_sub(1)..].windows(2)
                    .position(|w| w == [0xFF, 0xD9])
                    .map(|off| from.saturating_sub(1) + off);
                match eoi {
                    None => {
                        if pending.len() > 8 * 1024 * 1024 { pending.clear(); }
                        scan = pending.len();
                        break; // need more data
                    }
                    Some(j) => {
                        let frame: Vec<u8> = pending.drain(..j + 2).collect();
                        scan = 0;
                        if frame.len() < 100 { continue; }
                        win_frames += 1;
                        let el = win_start.elapsed().as_secs_f32();
                        if (!first_report_done && el >= 15.0) || el >= 60.0 {
                            let fps = win_frames as f32 / el;
                            if fps < 15.0 {
                                tracing::warn!("cam{cam} ingest only {fps:.1} fps -- camera/pipeline underdelivering");
                            } else {
                                tracing::info!("cam{cam} ingest {fps:.1} fps");
                            }
                            win_start = std::time::Instant::now();
                            win_frames = 0;
                            first_report_done = true;
                        }

                        let jpeg_arc = Arc::new(frame);
                        if let Some(tx) = &depth_raw_tx {
                            // ANONYMIZED: raw goes ONLY to in-RAM analysis — the
                            // YOLO queue (infer_tx) + motion below. The depth
                            // worker feeds every external consumer (frame_txs,
                            // recording, HLS, snapshots) with depth frames.
                            let _ = tx.send_replace(Some(Arc::clone(&jpeg_arc)));
                            state_arc.infer_queue.push(cam, Arc::clone(&jpeg_arc));
                        } else {
                        // FULL RATE (all cheap): live view + HLS + inference queue.
                        let _ = state_arc.frame_txs[cam as usize].send(Arc::clone(&jpeg_arc));
                        crate::inference_cmds::fan_out_frame(&state_arc, cam, &jpeg_arc).await;
                        }
                        // DETECTION (expensive) runs only if the previous frame is done
                        // AND at most ~7 fps (rate gate above).
                        if last_det.elapsed().as_millis() >= 140
                            && busy.compare_exchange(false, true,
                                std::sync::atomic::Ordering::AcqRel,
                                std::sync::atomic::Ordering::Acquire).is_ok()
                        {
                            last_det = std::time::Instant::now();
                            let b64 = base64::engine::general_purpose::STANDARD.encode(jpeg_arc.as_slice());
                            let s2 = Arc::clone(&state_arc);
                            let busy2 = Arc::clone(&busy);
                            tokio::spawn(async move {
                                let _ = crate::process_frame_inner(&s2, b64, cam, 0, false).await;
                                busy2.store(false, std::sync::atomic::Ordering::Release);
                            });
                        }
                    }
                }
            }
        }
    });

    state.rtsp_processes.lock().await.insert(cam, child);
    state.capture_keys.lock().await.insert(cam, key);
    state.app_handle.emit("rtsp:started",
        serde_json::json!({ "cam_id": cam, "url": format!("usb:{device_label}") })).ok();
    tracing::info!("USB capture started for cam{}: {}", cam, device_label);

    // Server-side audio for this USB camera — started only on a FRESH capture (the
    // idempotent reuse path returns above), tied to the video lifecycle. Two consumers
    // of the (shared-mode) mic, both proven to coexist: the RECORDER (segments → audible
    // continuous player + clips) and the DETECTOR (YAMNet events + clip-mux ring buffer).
    if let Some(aargs) = audio_args {
        // Recording audio is CAPTURE-OWNED now (muxed in-process, one clock).
        // This concurrent open of the shared-mode mic only feeds YAMNet.
        crate::audio_cmds::spawn_audio_detection(&state, cam, ffmpeg, aargs, true).await;
    }
    Ok(())
}

// ─── Per-OS: COMBINED camera+mic input (one process = one clock = A/V sync) ───
// Returns (input args, audio -map spec). None audio -> caller records video-only.

#[cfg(windows)]
async fn camera_av_input_args(ffmpeg: &std::path::Path, device: &str)
    -> Option<(Vec<String>, Option<String>)> {
    let mic = audio_device_name(ffmpeg).await?;
    Some((vec![
        "-f".into(), "dshow".into(), "-rtbufsize".into(), "50M".into(),
        "-framerate".into(), "30".into(),
        "-i".into(), format!("video={device}:audio={mic}"),
    ], Some("0:a:0".into())))
}

#[cfg(target_os = "macos")]
async fn camera_av_input_args(ffmpeg: &std::path::Path, device: &str)
    -> Option<(Vec<String>, Option<String>)> {
    // AVFoundation combined "video_index:audio_index"; audio 0 = default mic
    // (the ":0" convention audio_input_args already relies on).
    let index = if !device.is_empty() && device.chars().all(|c| c.is_ascii_digit()) {
        device.to_string()
    } else {
        let names = list_cameras_impl(ffmpeg).await.unwrap_or_default();
        names.iter().position(|n| n == device).map(|i| i.to_string()).unwrap_or_else(|| "0".into())
    };
    Some((vec![
        "-f".into(), "avfoundation".into(),
        "-framerate".into(), "30".into(),
        "-i".into(), format!("{index}:0"),
    ], Some("0:a:0".into())))
}

#[cfg(target_os = "linux")]
async fn camera_av_input_args(_ffmpeg: &std::path::Path, device: &str)
    -> Option<(Vec<String>, Option<String>)> {
    // v4l2 carries no audio; a second alsa input in the SAME process still shares
    // the process wallclock base, and record-time aresample heals residual drift.
    let path = if device.starts_with("/dev/") { device.to_string() } else { "/dev/video0".to_string() };
    Some((vec![
        "-f".into(), "v4l2".into(), "-i".into(), path,
        "-f".into(), "alsa".into(), "-thread_queue_size".into(), "1024".into(),
        "-i".into(), "default".into(),
    ], Some("1:a:0".into())))
}

#[cfg(not(any(windows, target_os = "macos", target_os = "linux")))]
async fn camera_av_input_args(_ffmpeg: &std::path::Path, _device: &str)
    -> Option<(Vec<String>, Option<String>)> { None }

// ─── Per-OS: ffmpeg AUDIO input args for server-side YAMNet (None = no mic) ───────

#[cfg(windows)]
pub(crate) async fn audio_device_name(ffmpeg: &std::path::Path) -> Option<String> {
    // dshow needs an explicit audio device NAME (no "default"); take the first one listed.
    let out = crate::proc::tokio_cmd(ffmpeg)
        .args(["-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"])
        .output().await.ok()?;
    let text = String::from_utf8_lossy(&out.stderr);
    for line in text.lines() {
        if line.contains("(audio)") {
            if let (Some(a), Some(b)) = (line.find('"'), line.rfind('"')) {
                if b > a + 1 { return Some(line[a + 1..b].to_string()); }
            }
        }
    }
    None
}

#[cfg(windows)]
pub(crate) async fn audio_input_args(ffmpeg: &std::path::Path) -> Option<Vec<String>> {
    let mic = audio_device_name(ffmpeg).await?;
    Some(vec!["-f".into(), "dshow".into(), "-i".into(), format!("audio={mic}")])
}

#[cfg(target_os = "macos")]
pub(crate) async fn audio_input_args(_ffmpeg: &std::path::Path) -> Option<Vec<String>> {
    // AVFoundation "video:audio" → "none:0" = no video, default mic (audio index 0).
    Some(vec!["-f".into(), "avfoundation".into(), "-i".into(), "none:0".into()])
}

#[cfg(target_os = "linux")]
pub(crate) async fn audio_input_args(_ffmpeg: &std::path::Path) -> Option<Vec<String>> {
    // ALSA "default" (routes through PulseAudio/PipeWire when present).
    Some(vec!["-f".into(), "alsa".into(), "-i".into(), "default".into()])
}

#[cfg(not(any(windows, target_os = "macos", target_os = "linux")))]
pub(crate) async fn audio_input_args(_ffmpeg: &std::path::Path) -> Option<Vec<String>> { None }

// ─── Per-OS: ffmpeg input args (the ONLY OS-specific capture code) ────────────────

#[cfg(windows)]
async fn camera_input_args(_ffmpeg: &std::path::Path, device: &str) -> Result<Vec<String>, String> {
    // `-rtbufsize` absorbs USB bursts so dshow doesn't drop frames.
    Ok(vec![
        "-f".into(), "dshow".into(), "-rtbufsize".into(), "50M".into(),
        // Ask the camera for its native 30 fps explicitly — without this some
        // drivers negotiate lower modes and the whole chain inherits the cap.
        "-framerate".into(), "30".into(),
        "-i".into(), format!("video={device}"),
    ])
}

#[cfg(target_os = "macos")]
async fn camera_input_args(ffmpeg: &std::path::Path, device: &str) -> Result<Vec<String>, String> {
    // AVFoundation selects by INDEX. Accept a numeric index directly, else resolve the
    // device NAME → its position in the video-device list. Syntax is "video:audio";
    // ":none" = video only (audio is a separate server-side tap, like the RTSP path).
    let index = if !device.is_empty() && device.chars().all(|c| c.is_ascii_digit()) {
        device.to_string()
    } else {
        let names = list_cameras_impl(ffmpeg).await.unwrap_or_default();
        names.iter().position(|n| n == device).map(|i| i.to_string()).unwrap_or_else(|| "0".into())
    };
    Ok(vec![
        "-f".into(), "avfoundation".into(),
        "-framerate".into(), "30".into(),     // default is 30000/1001; explicit 30 is widely supported
        "-i".into(), format!("{index}:none"),
    ])
}

#[cfg(target_os = "linux")]
async fn camera_input_args(_ffmpeg: &std::path::Path, device: &str) -> Result<Vec<String>, String> {
    // `device` is a /dev/videoN path. We deliberately DON'T force `-input_format mjpeg`:
    // not all cams support it and it would hard-fail; letting ffmpeg negotiate works on
    // every cam, and the scale filter normalizes whatever resolution it picks.
    let path = if device.starts_with("/dev/") { device.to_string() } else { "/dev/video0".to_string() };
    Ok(vec!["-f".into(), "v4l2".into(), "-i".into(), path])
}

#[cfg(not(any(windows, target_os = "macos", target_os = "linux")))]
async fn camera_input_args(_ffmpeg: &std::path::Path, _device: &str) -> Result<Vec<String>, String> {
    Err("USB camera capture is not supported on this platform".into())
}

// ─── Per-OS: device enumeration ──────────────────────────────────────────────────

#[cfg(windows)]
async fn list_cameras_impl(ffmpeg: &std::path::Path) -> Result<Vec<String>, String> {
    let out = crate::proc::tokio_cmd(ffmpeg)
        .args(["-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"])
        .output().await.map_err(|e| e.to_string())?;
    // dshow prints the device list to stderr; video lines look like: "Integrated Camera" (video)
    let text = String::from_utf8_lossy(&out.stderr);
    let mut names = Vec::new();
    for line in text.lines() {
        if line.contains("(video)") {
            if let (Some(a), Some(b)) = (line.find('"'), line.rfind('"')) {
                if b > a + 1 { names.push(line[a + 1..b].to_string()); }
            }
        }
    }
    names.dedup();
    Ok(names)
}

#[cfg(target_os = "macos")]
async fn list_cameras_impl(ffmpeg: &std::path::Path) -> Result<Vec<String>, String> {
    let out = crate::proc::tokio_cmd(ffmpeg)
        .args(["-hide_banner", "-list_devices", "true", "-f", "avfoundation", "-i", "dummy"])
        .output().await.map_err(|e| e.to_string())?;
    // Output (stderr):
    //   [AVFoundation indev @ ..] AVFoundation video devices:
    //   [AVFoundation indev @ ..] [0] FaceTime HD Camera
    //   [AVFoundation indev @ ..] AVFoundation audio devices:
    //   [AVFoundation indev @ ..] [0] Built-in Microphone
    // Take entries in the VIDEO section, returning the name (index order == list order).
    let text = String::from_utf8_lossy(&out.stderr);
    let mut names = Vec::new();
    let mut in_audio = false;
    for line in text.lines() {
        if line.contains("audio devices") { in_audio = true; continue; }
        if line.contains("video devices") { in_audio = false; continue; }
        if in_audio { continue; }
        // Match the "] [N] Name" tail.
        if let Some(p) = line.find("] [") {
            if let Some(c) = line[p + 3..].find("] ") {
                let name = line[p + 3 + c + 2..].trim().to_string();
                if !name.is_empty() { names.push(name); }
            }
        }
    }
    names.dedup();
    Ok(names)
}

#[cfg(target_os = "linux")]
async fn list_cameras_impl(_ffmpeg: &std::path::Path) -> Result<Vec<String>, String> {
    // Scan /dev/video*. Many UVC cams expose several nodes (capture + metadata); we list
    // them all (sorted) and let the user/start pick. /dev/video0 is the usual capture node.
    let mut devs: Vec<String> = Vec::new();
    if let Ok(mut rd) = tokio::fs::read_dir("/dev").await {
        while let Ok(Some(e)) = rd.next_entry().await {
            let n = e.file_name().to_string_lossy().to_string();
            // Require a trailing number (videoN) — skips a bare "video" or odd nodes.
            if n.starts_with("video") && n.len() > 5 && n[5..].chars().all(|c| c.is_ascii_digit()) {
                devs.push(format!("/dev/{n}"));
            }
        }
    }
    devs.sort();
    Ok(devs)
}

#[cfg(not(any(windows, target_os = "macos", target_os = "linux")))]
async fn list_cameras_impl(_ffmpeg: &std::path::Path) -> Result<Vec<String>, String> {
    Ok(vec![])
}
