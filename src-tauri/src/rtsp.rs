//! RTSP relay — pulls an external RTSP/MJPEG stream with ffmpeg and feeds JPEG frames into the existing motion-detection pipeline so a network camera looks identical to a browser-attached one.

use std::sync::Arc;

use base64::Engine as _;
use tauri::{Emitter, State};

use crate::{AppState, ensure_ffmpeg, process_frame_inner};


/// Scheme-aware ffmpeg INPUT flags (the bit that goes before `-i <url>`).
///
/// This is the difference between a working and a broken relay: `-reconnect*`
/// are HTTP-protocol options — passing them to an rtmp:// or srt:// input makes
/// ffmpeg abort ("Option reconnect not found"). So each scheme gets only the
/// flags it actually understands:
///   • rtsp://            → `-rtsp_transport tcp|udp` + socket `-timeout`
///   • http:// https://   → reconnect family (covers MJPEG, HLS, HTTP-FLV)
///   • rtmp:// srt:// …    → nothing (ffmpeg's own defaults; never bogus options)
pub(crate) fn relay_input_flags(url: &str, transport: &str) -> Vec<String> {
    if url.starts_with("rtsp://") {
        let t = if transport.eq_ignore_ascii_case("udp") { "udp" } else { "tcp" };
        return vec!["-rtsp_transport".into(), t.into(), "-timeout".into(), "5000000".into()];
    }
    if url.starts_with("http://") || url.starts_with("https://") {
        return vec![
            "-reconnect".into(), "1".into(),
            "-reconnect_streamed".into(), "1".into(),
            "-reconnect_delay_max".into(), "5".into(),
        ];
    }
    Vec::new()
}

/// Look up a camera's saved RTSP transport ("tcp" | "udp"), defaulting to tcp.
async fn cam_transport(state: &AppState, cam: u8) -> String {
    sqlx::query_scalar::<_, String>("SELECT transport FROM camera_configs WHERE cam_id=?")
        .bind(cam as i64)
        .fetch_optional(&state.db).await
        .ok().flatten()
        .filter(|t| !t.is_empty())
        .unwrap_or_else(|| "tcp".into())
}

/// Optional low-res DETECT sub-stream URL (mature NVRs' model). None = detect on
/// the main stream, downscaled in ffmpeg.
async fn cam_detect_url(state: &AppState, cam: u8) -> Option<String> {
    sqlx::query_scalar::<_, String>("SELECT detect_url FROM camera_configs WHERE cam_id=?")
        .bind(cam as i64)
        .fetch_optional(&state.db).await
        .ok().flatten()
        .filter(|u| !u.trim().is_empty())
}

/// Pull an RTSP/MJPEG stream with ffmpeg and pipe JPEG frames into the existing
/// motion-detection + streaming pipeline. No browser camera needed.
#[tauri::command]
pub async fn start_rtsp_relay(
    state: State<'_, Arc<AppState>>,
    cam_id: u8,
    url: String,
) -> Result<(), String> {
    let cam = cam_id.min(15);
    // A REMOVED/disabled camera never re-opens its stream (see camera_start_allowed).
    if !crate::cam_config::camera_start_allowed(&state.db, cam).await {
        tracing::info!("rtsp relay cam{cam}: refused — camera is removed/disabled");
        return Err("camera is disabled".into());
    }
    let key = format!("rtsp:{url}");
    // Serialize starts (see capture_start_lock) so dual mounts/re-renders for one IP
    // camera can't race past the idempotency check and spawn duplicate relays.
    let _start_guard = state.capture_start_lock.lock().await;
    // IDEMPOTENT START — same stream already relaying + alive ⇒ NO-OP (no respawn gap).
    if state.capture_keys.lock().await.get(&cam) == Some(&key) {
        let mut procs = state.rtsp_processes.lock().await;
        if let Some(child) = procs.get_mut(&cam) {
            if matches!(child.try_wait(), Ok(None)) {
                tracing::debug!("rtsp cam{}: already relaying {} — reusing", cam, url);
                return Ok(());
            }
        }
    }
    stop_rtsp_relay(State::clone(&state), cam).await.ok();

    // The camera's saved transport (tcp default | udp) drives every ffmpeg we spawn
    // for it — detection relay, recorder, and audio tap — so a UDP-only camera works
    // end-to-end, not just in the connection test. Scheme-aware flags via
    // `relay_input_flags` keep auto-reconnect on HTTP without breaking rtmp/srt.
    let transport = cam_transport(&state, cam).await;

    // Hardware-decode the detection stream when the GPU can (mature NVRs' #1 CPU win):
    // `-hwaccel <x>` before `-i` moves H.264/H.265 decode off the CPU. Only the
    // detection path uses this — recording is `-c:v copy` (no decode at all).
    let hw_dec = state.hw_decoder.read().unwrap().clone();
    let input_args = |url: &str| -> Vec<String> {
        let mut a: Vec<String> = vec!["-hide_banner".into(), "-loglevel".into(), "error".into()];
        a.extend(crate::hw::hw_decode_args(&hw_dec));
        a.extend(relay_input_flags(url, &transport));
        a
    };

    // ffmpeg: decode stream → output individual JPEG frames at 10fps to stdout
    let ffmpeg_bin = ensure_ffmpeg(&state.data_dir).await.map_err(|e| e.to_string())?;
    // Detect on the camera's LOW-RES SUB-STREAM when one is configured
    // (mature NVRs' #1 efficiency recommendation — decoding the substream instead
    // of the main stream cuts detect decode cost ~10x). Recorder + audio tap
    // stay on the main URL.
    let detect_src = cam_detect_url(&state, cam).await.unwrap_or_else(|| url.clone());
    if detect_src != url {
        tracing::info!("rtsp cam{cam}: detection uses sub-stream {detect_src}");
    }
    let mut det_args = input_args(&detect_src);
    det_args.extend([
        "-i", detect_src.as_str(),
        // 5 fps low-res detection (mature NVRs' recommended detect rate): at 16
        // cameras this bounds worst-case inference demand to 16×5=80 f/s and
        // keeps 16 relay decodes cheap. Live view rides the copy-HLS below —
        // this MJPEG feed is detection + grid-fallback only.
        "-vf", "fps=5,scale=640:-2",
        "-f", "image2pipe", "-vcodec", "mjpeg",
        "-q:v", "5",
        "pipe:1",
    ].map(String::from));
    let mut child = crate::proc::tokio_cmd(&ffmpeg_bin)
        .args(&det_args)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| format!("ffmpeg not found: {e}. Install ffmpeg to use RTSP relay."))?;

    let stdout = child.stdout.take().ok_or("no stdout")?;
    let state_arc = Arc::clone(&state);

    // DROP-DON'T-QUEUE backpressure: at most one frame in flight (see dshow.rs for the
    // OOM/slow-motion this prevents). Excess frames are dropped, not queued.
    let busy = Arc::new(std::sync::atomic::AtomicBool::new(false));
    // Spawn a reader task that parses the JPEG stream and feeds process_frame
    tokio::spawn(async move {
        use tokio::io::AsyncReadExt;
        let mut reader = tokio::io::BufReader::new(stdout);
        let mut buf = Vec::<u8>::with_capacity(256 * 1024);
        loop {
            // Read JPEG: starts with FF D8, ends with FF D9
            buf.clear();
            // Find SOI marker
            let mut b = [0u8; 1];
            loop {
                if reader.read_exact(&mut b).await.is_err() { return; }
                if b[0] == 0xFF {
                    let mut b2 = [0u8; 1];
                    if reader.read_exact(&mut b2).await.is_err() { return; }
                    if b2[0] == 0xD8 { buf.extend_from_slice(&[0xFF, 0xD8]); break; }
                }
            }
            // Read until EOI marker
            loop {
                if reader.read_exact(&mut b).await.is_err() { break; }
                buf.push(b[0]);
                if buf.len() >= 2 && buf[buf.len()-2] == 0xFF && buf[buf.len()-1] == 0xD9 { break; }
                if buf.len() > 4 * 1024 * 1024 { break; } // safety limit 4MB
            }
            if buf.len() < 100 { continue; }
            let jpeg_arc = Arc::new(buf.clone());
            // FULL RATE (all cheap): live view + recording + HLS + inference queue.
            let _ = state_arc.frame_txs[cam as usize].send(Arc::clone(&jpeg_arc));
            crate::inference_cmds::fan_out_frame(&state_arc, cam, &jpeg_arc).await;
            // DETECTION (expensive) runs drop-don't-queue so tasks can't pile up (OOM)
            // or lag behind (slow-motion). Recording above is unaffected by its speed.
            if busy.compare_exchange(false, true,
                std::sync::atomic::Ordering::AcqRel,
                std::sync::atomic::Ordering::Acquire).is_ok()
            {
                let b64 = base64::engine::general_purpose::STANDARD.encode(&buf);
                let s2 = Arc::clone(&state_arc);
                let busy2 = Arc::clone(&busy);
                tokio::spawn(async move {
                    let _ = process_frame_inner(&s2, b64, cam, 0, false).await;
                    busy2.store(false, std::sync::atomic::Ordering::Release);
                });
            }
        }
    });

    state.rtsp_processes.lock().await.insert(cam, child);
    state.capture_keys.lock().await.insert(cam, key);
    tracing::info!("RTSP relay started for cam{}: {}", cam, url);

    // go2rtc: restream this camera over WebRTC for sub-second live view.
    // Best-effort and fully decoupled — on any failure the frontend's
    // WebRTC → HLS → MJPEG ladder keeps the live tile working.
    if url.starts_with("rtsp://") {
        let data_dir = state.data_dir.clone();
        let url_g = url.clone();
        tokio::spawn(async move {
            if let Err(e) = crate::go2rtc::ensure_running(&data_dir).await {
                tracing::info!("go2rtc not available ({e}) — cam{cam} live view stays on HLS");
                return;
            }
            crate::go2rtc::register_stream(cam, &url_g).await;
        });
    }

    // Zero-reencoding NVR: record directly from RTSP with -c:v copy (no quality loss, minimal CPU).
    // This runs in parallel with the detection relay — same source, two outputs.
    // Mature NVRs' key insight: decode once for detection, copy-stream for recording.
    //
    // The SAME process also writes the live HLS playlist as a second COPY output
    // (go2rtc's restream model: the camera's own H.264 serves record AND live) —
    // so an RTSP camera costs ZERO encode sessions. The old path re-encoded the
    // relay's MJPEG into HLS: 1 NVENC per cam, ×16 cams over the session limit.
    let settings = state.settings.read().await.clone();
    if settings.nvr_enabled {
        // Segments finalize via the global scanner. (Previously never started on
        // the RTSP path — its .tmp.mp4 segments were never finalized/indexed.)
        crate::nvr_pipes::ensure_postprocessor(&state.data_dir,
            state.app_handle.clone(), state.db.clone()).await;
        // Evict any legacy HLS encoder for this cam — the recorder owns the
        // playlist now (two writers corrupt it).
        if let Some(mut old) = state.hls_processes.lock().await.remove(&cam) { let _ = old.kill().await; }
        state.hls_pipe_txs.lock().await.remove(&cam);

        let nvr_dir = state.data_dir.join("nvr");
        let hls_dir = state.data_dir.join("hls");
        tokio::fs::create_dir_all(&nvr_dir).await.ok();
        tokio::fs::create_dir_all(&hls_dir).await.ok();
        // 10-second segments — the same contract as USB cams (short segments =
        // clips available seconds after an event; the old 60s+ here made fresh
        // RTSP clips lag a minute behind).
        let pattern = nvr_dir.join(format!("cam{}_rtsp_%Y%m%d_%H%M%S.tmp.mp4", cam));
        let is_rtsp2 = url.starts_with("rtsp://");
        let mut rargs: Vec<String> = vec!["-hide_banner".into(), "-loglevel".into(), "error".into(), "-y".into()];
        rargs.extend(relay_input_flags(&url, &transport));
        rargs.extend(["-i".into(), url.clone()]);
        if is_rtsp2 {
            // Output 1 — segments: zero-re-encode copy (full quality, low CPU).
            rargs.extend(["-map", "0:v:0", "-map", "0:a?", "-c:v", "copy", "-c:a", "copy"].map(String::from));
        } else {
            // Non-RTSP (MJPEG / HLS / RTMP / …) → re-encode to a clean H.264 MP4 the
            // clip/player pipeline can use (copy of fragmented/MJPEG sources breaks it).
            rargs.extend(["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-threads", "2", "-an"].map(String::from));
        }
        rargs.extend([
            "-f", "segment", "-segment_time", "10", "-segment_format", "mp4",
            "-segment_time_delta", "0.05", "-strftime", "1", "-reset_timestamps", "1",
        ].map(String::from));
        rargs.push(pattern.to_string_lossy().to_string());
        if is_rtsp2 {
            // Output 2 — live HLS, ALSO copy (0 encode sessions). Segment length
            // follows the camera's keyframe cadence (copy can only cut on
            // keyframes); hls.js falls back to the MJPEG stream for codecs the
            // WebView can't decode (e.g. some H.265 cams) — existing behavior.
            let m3u8 = hls_dir.join(format!("cam{cam}.m3u8"));
            let ts   = hls_dir.join(format!("cam{cam}_%04d.ts"));
            rargs.extend(["-map".into(), "0:v:0".into(), "-c:v".into(), "copy".into(), "-an".into(),
                "-f".into(), "hls".into(), "-hls_time".into(), "2".into(),
                "-hls_list_size".into(), "10".into(),
                "-hls_flags".into(), "delete_segments+append_list+discont_start".into(),
                "-hls_segment_filename".into(), ts.to_string_lossy().into_owned(),
                m3u8.to_string_lossy().into_owned()]);
        }
        let rec = crate::proc::tokio_cmd(&ffmpeg_bin)
            .args(&rargs)
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true)
            .spawn();
        match rec {
            Ok(mut child) => {
                if let Some(stderr) = child.stderr.take() {
                    tokio::spawn(async move {
                        use tokio::io::{AsyncBufReadExt, BufReader};
                        let mut lines = BufReader::new(stderr).lines();
                        while let Ok(Some(line)) = lines.next_line().await {
                            let l = line.trim();
                            if !l.is_empty() { tracing::warn!("RTSP recorder cam{cam}: {l}"); }
                        }
                    });
                }
                // Held in nvr_processes so stop_rtsp_relay kills it and the
                // watchdog sees it die (previously an orphan local — a dead
                // RTSP recorder silently stopped recording forever).
                if let Some(mut old) = state.nvr_processes.lock().await.insert(cam, child) {
                    let _ = old.kill().await;
                }
                tracing::info!("RTSP copy recorder+HLS started for cam{cam}");
            }
            Err(e) => tracing::warn!("RTSP recorder cam{cam} failed to start: {e}"),
        }
    }

    // ── Audio event detection (YAMNet) ───────────────────────────────────────
    // Tap the stream's audio server-side → 16 kHz mono PCM → YAMNet → fire an event for a
    // loud, confident "listened" class. Shared with the USB capture path; the helper
    // checks the toggle + `audio_yamnet` skill itself and no-ops otherwise.
    // record_audio = false: the RTSP recorder already copies the camera's own audio
    // into its segments, so the clip exporter carries it through — no buffering needed.
    let mut audio_in = relay_input_flags(&url, &transport);
    audio_in.push("-i".into());
    audio_in.push(url.clone());
    crate::audio_cmds::spawn_audio_detection(&state, cam, &ffmpeg_bin, audio_in, false).await;

    state.app_handle.emit("rtsp:started", serde_json::json!({ "cam_id": cam, "url": url })).ok();
    Ok(())
}


#[tauri::command]
pub async fn stop_rtsp_relay(state: State<'_, Arc<AppState>>, cam_id: u8) -> Result<(), String> {
    stop_capture_for_cam(state.inner(), cam_id).await;
    Ok(())
}

/// Stop EVERY capture process for one camera — the single teardown path.
///
/// USB/dshow captures live in the same `rtsp_processes` map as RTSP relays (see
/// `dshow::spawn_capture`), so this covers both kinds. Callable outside the
/// command layer so config changes can tear a camera down too: removing a
/// camera used to only flip `enabled=0` in the DB, leaving its ffmpeg holding
/// the webcam (LED on) until the whole app exited.
pub(crate) async fn stop_capture_for_cam(state: &Arc<AppState>, cam_id: u8) {
    if let Some(mut child) = state.rtsp_processes.lock().await.remove(&cam_id) {
        child.kill().await.ok();
        state.app_handle.emit("rtsp:stopped", serde_json::json!({ "cam_id": cam_id })).ok();
        tracing::info!("capture stopped for cam{}", cam_id);
    }
    if let Some(mut child) = state.audio_processes.lock().await.remove(&cam_id) {
        child.kill().await.ok();
    }
    // Cancel a native (nokhwa) capture task if this cam ran on that path.
    if let Some(task) = state.capture_handles.lock().await.remove(&cam_id) {
        task.cancel.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    // Kill the RTSP copy-recorder (+ merged HLS output) if this cam has one. Only
    // for rtsp-keyed cams: for legacy pipe cams nvr_processes holds the pipe
    // recorder, whose lifecycle the NVR watchdog owns.
    let is_rtsp_cam = state.capture_keys.lock().await.get(&cam_id)
        .map(|k| k.starts_with("rtsp:")).unwrap_or(false);
    if is_rtsp_cam {
        if let Some(mut child) = state.nvr_processes.lock().await.remove(&cam_id) {
            child.kill().await.ok();
        }
    }
    // Close any open audio event so it doesn't linger as perpetually in-progress.
    crate::audio_cmds::close_audio_for_cam(state, cam_id).await;
    // Clear the recorder's mic registration so a watchdog respawn while the camera is
    // stopped goes back to video-only (the next fresh capture re-registers it).
    crate::nvr_pipes::set_nvr_mic(cam_id, None);
    // Drop the idempotency marker so a later start re-spawns cleanly.
    state.capture_keys.lock().await.remove(&cam_id);
}


// ── Connection test ──────────────────────────────────────────────────────────

/// What a `probe_stream` test learned about a camera URL. `ok=false` carries a
/// human-readable `error`; the rest is populated only on success.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct StreamProbe {
    pub ok:          bool,
    pub codec:       String,  // h264 / hevc / mjpeg …
    pub width:       u32,
    pub height:      u32,
    pub fps:         f32,
    pub has_audio:   bool,
    pub audio_codec: String,
    pub error:       String,
}

fn parse_rational_fps(s: &str) -> f32 {
    let mut it = s.split('/');
    let num: f32 = it.next().and_then(|n| n.trim().parse().ok()).unwrap_or(0.0);
    let den: f32 = it.next().and_then(|n| n.trim().parse().ok()).unwrap_or(1.0);
    if den > 0.0 { num / den } else { 0.0 }
}

/// Turn ffprobe's stderr into a one-line, user-actionable message. Covers the
/// failure modes IP cameras actually hit: bad credentials, wrong port/path,
/// unreachable host, timeouts, and unrecognized formats.
fn humanize_probe_error(stderr: &str) -> String {
    let low = stderr.to_lowercase();
    if low.contains("401") || low.contains("unauthorized") {
        return "Authentication failed — check the username and password.".into();
    }
    if low.contains("403") || low.contains("forbidden") {
        return "Access forbidden (403) — the account may lack streaming permission.".into();
    }
    if low.contains("connection refused") {
        return "Connection refused — wrong port, or RTSP isn't enabled on the camera.".into();
    }
    if low.contains("no route to host") || low.contains("network is unreachable") || low.contains("host is unreachable") {
        return "Host unreachable — check the IP address and that you're on the same network.".into();
    }
    if low.contains("timed out") || low.contains("timeout") || low.contains("etimedout") {
        return "Timed out — the camera didn't respond (check IP, port, and firewall).".into();
    }
    if low.contains("404") || low.contains("not found") {
        return "Stream path not found (404) — check the URL path / channel.".into();
    }
    if low.contains("invalid data") || low.contains("could not find codec") || low.contains("does not contain") {
        return "Connected, but the stream format wasn't recognized.".into();
    }
    let last = stderr.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or("could not open the stream");
    format!("Could not open the stream: {}", last.trim())
}

/// Probe a camera URL with ffprobe to validate it BEFORE the user commits the
/// camera — reports codec / resolution / fps / audio so the UI can show "✓ 1080p
/// H.264" or a precise error. `transport` ("tcp" default | "udp") applies to
/// rtsp:// only. Never errors on a bad camera — returns `ok=false` with `error`.
#[tauri::command]
pub async fn probe_stream(
    state: State<'_, Arc<AppState>>,
    url: String,
    transport: Option<String>,
) -> Result<StreamProbe, String> {
    let url = url.trim().to_string();
    if url.is_empty() {
        return Ok(StreamProbe { ok: false, error: "Enter a stream URL first.".into(), ..Default::default() });
    }

    let ffmpeg_bin = ensure_ffmpeg(&state.data_dir).await.map_err(|e| e.to_string())?;
    let ffprobe_bin = ffmpeg_bin.parent()
        .map(|p| p.join(if cfg!(windows) { "ffprobe.exe" } else { "ffprobe" }))
        .unwrap_or_else(|| std::path::PathBuf::from("ffprobe"));

    let mut args: Vec<String> = vec!["-hide_banner".into(), "-v".into(), "error".into()];
    if url.starts_with("rtsp://") {
        let t = if transport.as_deref() == Some("udp") { "udp" } else { "tcp" };
        args.extend(["-rtsp_transport".into(), t.to_string(), "-timeout".into(), "8000000".into()]);
    }
    args.extend([
        "-i".into(), url.clone(),
        "-show_entries".into(), "stream=codec_type,codec_name,width,height,avg_frame_rate".into(),
        "-of".into(), "json".into(),
    ]);

    let mut cmd = crate::proc::tokio_cmd(&ffprobe_bin);
    cmd.args(&args).kill_on_drop(true);
    // window suppression: proc::tokio_cmd already sets DETACHED_PROCESS (see proc.rs)

    // Hard wall-clock cap: a wedged/unreachable camera must never hang the test.
    // kill_on_drop ensures the ffprobe child dies when this future is dropped.
    let out = match tokio::time::timeout(std::time::Duration::from_secs(12), cmd.output()).await {
        Ok(Ok(o))  => o,
        Ok(Err(e)) => return Ok(StreamProbe { ok: false, error: format!("Couldn't run the probe: {e}"), ..Default::default() }),
        Err(_)     => return Ok(StreamProbe { ok: false, error: "Timed out connecting to the stream (check URL, network, and credentials).".into(), ..Default::default() }),
    };

    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Ok(StreamProbe { ok: false, error: humanize_probe_error(&stderr), ..Default::default() });
    }

    let json: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap_or_else(|_| serde_json::json!({}));
    let mut probe = StreamProbe { ok: true, ..Default::default() };
    for s in json["streams"].as_array().cloned().unwrap_or_default() {
        match s["codec_type"].as_str() {
            Some("video") if probe.codec.is_empty() => {
                probe.codec  = s["codec_name"].as_str().unwrap_or("").to_string();
                probe.width  = s["width"].as_u64().unwrap_or(0) as u32;
                probe.height = s["height"].as_u64().unwrap_or(0) as u32;
                probe.fps    = parse_rational_fps(s["avg_frame_rate"].as_str().unwrap_or("0/1"));
            }
            Some("audio") => {
                probe.has_audio = true;
                if probe.audio_codec.is_empty() {
                    probe.audio_codec = s["codec_name"].as_str().unwrap_or("").to_string();
                }
            }
            _ => {}
        }
    }
    if probe.codec.is_empty() {
        return Ok(StreamProbe { ok: false, error: "Connected, but no video stream was found at this URL.".into(), ..Default::default() });
    }
    Ok(probe)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rtsp_uses_transport_and_timeout() {
        assert_eq!(
            relay_input_flags("rtsp://cam/stream", "tcp"),
            vec!["-rtsp_transport", "tcp", "-timeout", "5000000"]
        );
        assert_eq!(
            relay_input_flags("rtsp://cam/stream", "udp"),
            vec!["-rtsp_transport", "udp", "-timeout", "5000000"]
        );
        // Unknown/empty transport must fall back to tcp, never a bogus value.
        assert_eq!(relay_input_flags("rtsp://cam", "")[1], "tcp");
        assert_eq!(relay_input_flags("rtsp://cam", "bogus")[1], "tcp");
    }

    #[test]
    fn http_gets_reconnect_family() {
        // MJPEG, HLS and HTTP-FLV all ride the http(s) reconnect path.
        for url in ["http://cam/video", "https://cam/index.m3u8"] {
            let f = relay_input_flags(url, "tcp");
            assert!(f.contains(&"-reconnect".to_string()), "{url} should reconnect");
            assert!(!f.contains(&"-rtsp_transport".to_string()), "{url} must not get rtsp flags");
        }
    }

    #[test]
    fn rtmp_and_srt_get_no_protocol_options() {
        // The bug guard: -reconnect on rtmp/srt makes ffmpeg abort. These must be bare.
        assert!(relay_input_flags("rtmp://cam/live", "tcp").is_empty());
        assert!(relay_input_flags("srt://cam:9000", "tcp").is_empty());
    }
}
