//! Rust-side NVR encoding pipeline: feeds JPEG frames through ffmpeg subprocess pipes for both continuous H.264 MP4 segments and a live HLS playlist.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use chrono::Utc;
use tokio::io::AsyncWriteExt;
use sqlx::SqlitePool;
use tauri::Emitter;

use crate::ensure_ffmpeg;
use crate::hw::hw_encoder_args;
use chrono::TimeZone as _;

// ─── Per-camera mic registry for audio-in-recordings ─────────────────────────
//
// USB/native cameras record from a silent JPEG frame pipe. To put audio in their
// NVR SEGMENTS (so both the continuous player AND event clips have sound), the
// capture layer (dshow.rs) registers the local mic's ffmpeg INPUT args here; the
// recorder reads them so respawns (watchdog/stall) preserve audio without any
// signature change. RTSP cams never register (their copy-recorder already carries
// the camera's own audio). Mature NVRs' proven model: a separate audio source muxed
// with `-map 0:v -map 1:a` (+genpts to keep the two live inputs aligned).
static NVR_MICS: OnceLock<Mutex<HashMap<u8, Vec<String>>>> = OnceLock::new();
fn nvr_mics() -> &'static Mutex<HashMap<u8, Vec<String>>> {
    NVR_MICS.get_or_init(|| Mutex::new(HashMap::new()))
}
/// Register (or clear) the mic input args used when (re)spawning this cam's recorder.
pub(crate) fn set_nvr_mic(cam: u8, args: Option<Vec<String>>) {
    let mut m = nvr_mics().lock().unwrap_or_else(|e| e.into_inner());
    match args { Some(a) => { m.insert(cam, a); } None => { m.remove(&cam); } }
}
/// Is a mic already registered for this cam? (drives "restart the recorder or not")
/// Unused since audio capture moved into the merged capture+record path; kept as
/// the single truthful answer to "does this camera have a mic".
#[allow(dead_code)]
pub(crate) fn has_nvr_mic(cam: u8) -> bool {
    nvr_mics().lock().unwrap_or_else(|e| e.into_inner()).contains_key(&cam)
}
fn get_nvr_mic(cam: u8) -> Option<Vec<String>> {
    nvr_mics().lock().unwrap_or_else(|e| e.into_inner()).get(&cam).cloned()
}

// ─── Per-camera SOFTWARE-encoder fallback ────────────────────────────────────
//
// The hardware encoder (nvenc/qsv/amf) can fail to init or exit mid-run — most
// often NVENC session-limit / GPU contention now that DirectML inference is pinned
// to the SAME discrete GPU that also does the encode. That silently killed recording
// (0 segments, watchdog flap). After repeated hardware-encoder deaths the watchdog
// flags the cam here; the next respawn uses libx264 (CPU), which always works — so
// recording self-heals to reliable software encoding instead of flapping forever.
static NVR_SW_ENCODER: OnceLock<Mutex<std::collections::HashSet<u8>>> = OnceLock::new();
fn nvr_sw_set() -> &'static Mutex<std::collections::HashSet<u8>> {
    NVR_SW_ENCODER.get_or_init(|| Mutex::new(std::collections::HashSet::new()))
}
/// Force this cam's recorder onto the software encoder (libx264) from now on.
pub(crate) fn force_software_encoder(cam: u8) {
    nvr_sw_set().lock().unwrap_or_else(|e| e.into_inner()).insert(cam);
    release_nvenc(cam); // frees its session slot for other cams
}
pub(crate) fn wants_software_encoder(cam: u8) -> bool {
    nvr_sw_set().lock().unwrap_or_else(|e| e.into_inner()).contains(&cam)
}

// ─── NVENC session budget (deliberate assignment, not death-triggered) ───────
//
// Consumer GeForce caps CONCURRENT NVENC sessions process-wide (measured 12 on
// this RTX 5060 / driver 610.62; older drivers allow 8 — we budget the floor).
// Before this, every camera silently grabbed sessions until ffmpeg died with
// "OpenEncodeSessionEx failed" and the watchdog downgraded it to CPU after two
// deaths. Now cams are ASSIGNED an encoder up front: NVENC while under budget,
// then Intel QSV (separate silicon, no NVENC limit), then libx264.
const NVENC_SESSION_BUDGET: usize = 8;
static NVENC_CAMS: OnceLock<Mutex<std::collections::HashSet<u8>>> = OnceLock::new();
fn nvenc_set() -> &'static Mutex<std::collections::HashSet<u8>> {
    NVENC_CAMS.get_or_init(|| Mutex::new(std::collections::HashSet::new()))
}
static QSV_AVAILABLE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// Record (from boot's encoder probe) whether Intel QSV exists as an overflow lane.
pub(crate) fn set_qsv_available(avail: bool) {
    QSV_AVAILABLE.store(avail, std::sync::atomic::Ordering::Relaxed);
}
/// Pick the encoder this camera should actually use. `best` is the globally
/// detected encoder (state.hw_encoder). Idempotent per cam — a respawn keeps
/// its existing NVENC slot rather than counting itself twice.
pub(crate) fn assign_encoder(cam: u8, best: &str) -> String {
    if wants_software_encoder(cam) { return "libx264".into(); }
    if best != "h264_nvenc" { return best.to_string(); }
    let mut set = nvenc_set().lock().unwrap_or_else(|e| e.into_inner());
    if set.contains(&cam) || set.len() < NVENC_SESSION_BUDGET {
        set.insert(cam);
        return "h264_nvenc".into();
    }
    if QSV_AVAILABLE.load(std::sync::atomic::Ordering::Relaxed) {
        tracing::info!("encoder budget: cam{cam} over NVENC budget ({NVENC_SESSION_BUDGET}) — assigning h264_qsv (iGPU)");
        return "h264_qsv".into();
    }
    tracing::info!("encoder budget: cam{cam} over NVENC budget ({NVENC_SESSION_BUDGET}), no QSV — assigning libx264");
    "libx264".into()
}
/// Free a camera's NVENC slot (call when its capture/recorder stops for good).
pub(crate) fn release_nvenc(cam: u8) {
    nvenc_set().lock().unwrap_or_else(|e| e.into_inner()).remove(&cam);
}

/// SINGLE SOURCE OF TRUTH for NVR segment timestamps. ffmpeg writes segment filenames in
/// LOCAL time (`-strftime`), but the DB (and every event) stores UTC — so this is the one
/// place the local→UTC conversion happens. Parse "YYYYMMDD" + "HHMMSS" as a LOCAL
/// wall-clock time → UTC rfc3339. Returns None for a malformed or DST-ambiguous/skipped
/// local time so callers can fall back to "now".
pub(crate) fn segment_local_name_to_utc(date_part: &str, time_part: &str) -> Option<String> {
    if date_part.len() != 8 || time_part.len() != 6 { return None; }
    chrono::NaiveDateTime::parse_from_str(&format!("{date_part}{time_part}"), "%Y%m%d%H%M%S")
        .ok()
        .and_then(|n| chrono::Local.from_local_datetime(&n).single())
        .map(|t| t.with_timezone(&Utc).to_rfc3339())
}

//
// Frames flow: run_capture_loop / process_frame → jpeg_arc → pipe channel
//              → ffmpeg stdin → H.264 MP4 segments (NVR) / HLS (stream)
//
// Zero browser involvement: no MediaRecorder, no IPC round-trip, no UI thread.

/// standard NVR pipeline:
///   1. Record JPEG frames → ffmpeg → .tmp.mp4 segments (no special movflags)
///   2. When a segment completes, post-process with -movflags +faststart → .mp4
///   3. list_nvr_recordings only exposes .mp4 (post-processed, seekable) files
///   4. .tmp.mp4 files are cleaned up on startup (orphans from killed sessions)
///
/// This mirrors mature NVRs exactly: record to cache, migrate with faststart.
/// Output-side args for NVR segment recording — the SINGLE definition of the
/// segment contract (gop/keyframes/10s cuts/tmp naming), shared by the legacy
/// pipe recorder and the merged capture+record ffmpeg so they can never drift.
pub(crate) fn segment_output_args(cam_id: u8, nvr_dir: &std::path::Path) -> Vec<String> {
    let pattern = nvr_dir.join(format!("cam{}_%Y%m%d_%H%M%S.tmp.mp4", cam_id));
    vec![
        "-g".into(), "30".into(),
        "-force_key_frames".into(), "expr:gte(t,n_forced*1)".into(),
        "-f".into(), "segment".into(),
        "-segment_time".into(), "10".into(),
        "-segment_format".into(), "mp4".into(),
        "-reset_timestamps".into(), "1".into(),
        "-segment_time_delta".into(), "0.05".into(),
        "-strftime".into(), "1".into(),
        pattern.to_string_lossy().into_owned(),
    ]
}

/// tee-muxer output spec: ONE encode feeds BOTH the 10-s NVR segments and the
/// live HLS playlist — halving encode sessions vs the old separate HLS ffmpeg
/// (2 NVENC/cam → 1). Paths are RELATIVE ("nvr/…", "hls/…") because tee's
/// bracket parser treats `:` as an option separator — a `C:\` drive letter
/// inside `hls_segment_filename=` breaks it. Callers MUST set the ffmpeg
/// process's current_dir to the data dir. `onfail=ignore` on the HLS slave:
/// a live-view hiccup must never kill RECORDING.
pub(crate) fn tee_segment_hls_spec(cam_id: u8) -> Vec<String> {
    let seg = format!(
        "[f=segment:segment_time=10:segment_format=mp4:reset_timestamps=1:\
segment_time_delta=0.05:strftime=1]nvr/cam{cam_id}_%Y%m%d_%H%M%S.tmp.mp4");
    // discont_start: a respawned capture continues the previous playlist
    // (append_list) with a fresh encoder timeline — without the DISCONTINUITY
    // tag the player hits the PTS jump, errors, and re-attaches in a loop
    // (visible as live-view flicker after every capture respawn).
    let hls = format!(
        "[f=hls:onfail=ignore:hls_time=1:hls_list_size=10:\
hls_flags=delete_segments+append_list+discont_start:hls_segment_filename=hls/cam{cam_id}_%04d.ts]\
hls/cam{cam_id}.m3u8");
    vec![
        // Keyframe cadence (was inside segment_output_args): ~1s keyframes so
        // clips/seeks don't open on black, and segment cuts stay on keyframes.
        "-g".into(), "30".into(),
        "-force_key_frames".into(), "expr:gte(t,n_forced*1)".into(),
        // tee slaves get the encoder's GLOBAL extradata; the TS slave converts
        // to in-stream headers automatically (auto-bsf). Required for mp4+ts mix.
        "-flags".into(), "+global_header".into(),
        "-f".into(), "tee".into(),
        format!("{seg}|{hls}"),
    ]
}

/// Spawn THE segment post-processor exactly once per app lifetime — one scanner
/// for the WHOLE nvr dir, finalizing every camera's segments. The old design ran
/// one scanner per camera, each re-reading the entire shared directory every 2 s
/// (16 cams = 8 full scans/sec of everyone's files) — and cameras whose spawn
/// path never called it (RTSP copy-recorder) got their .tmp.mp4 never finalized.
pub(crate) async fn ensure_postprocessor(
    data_dir: &std::path::Path,
    app_handle: tauri::AppHandle,
    db: SqlitePool,
) {
    static RUNNING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if RUNNING.swap(true, std::sync::atomic::Ordering::SeqCst) {
        return; // already running
    }
    let ffmpeg = match ensure_ffmpeg(data_dir).await {
        Ok(f) => f,
        Err(e) => { tracing::warn!("postprocessor: no ffmpeg: {e}"); RUNNING.store(false, std::sync::atomic::Ordering::SeqCst); return; }
    };
    let nvr_dir = data_dir.join("nvr");
    tokio::spawn(async move {
        postprocess_nvr_segments(ffmpeg, nvr_dir, app_handle, db).await;
    });
}

pub(crate) async fn spawn_nvr_pipe(
    cam_id: u8,
    data_dir: &std::path::Path,
    segment_mins: u32,
    encoder: &str,
    app_handle: tauri::AppHandle,
    db: SqlitePool,
) -> anyhow::Result<(tokio::sync::mpsc::Sender<Arc<Vec<u8>>>, tokio::process::Child)> {
    let ffmpeg   = ensure_ffmpeg(data_dir).await?;
    let nvr_dir  = data_dir.join("nvr");
    tokio::fs::create_dir_all(&nvr_dir).await?;
    // Segment length: 10 SECONDS, matching mature NVRs. Short segments are the key to
    // CLIP AVAILABILITY — the segment covering a just-happened event rotates +
    // finalizes within seconds, so its clip plays on the first click instead of
    // only after the (long) current segment eventually rotates. The earlier perf
    // worry (44k files) was actually caused by the UNBOUNDED concat, which is now
    // bounded (`/nvr-concat` window), so short segments are safe again. seek
    // precision is unaffected (server-side `inpoint` seeks within the first seg).
    // `nvr_segment_mins` is no longer used for length — kept in settings for UI.
    let _ = segment_mins;
    let seg_secs = 10u64;
    let pattern  = nvr_dir.join(format!("cam{}_%Y%m%d_%H%M%S.tmp.mp4", cam_id));
    // Deliberate per-cam encoder assignment: NVENC within the session budget,
    // QSV/x264 overflow, libx264 if this cam's hardware encoder proved unstable.
    let encoder = assign_encoder(cam_id, encoder);
    let enc_args = hw_encoder_args(&encoder);
    let app_handle2 = app_handle.clone();
    let db2 = db.clone();

    // Mature NVRs' approach: timestamps come from the source (camera wall clock).
    // For RTSP they use -c:v copy which preserves camera timestamps.
    // For our JPEG pipe we use -use_wallclock_as_timestamps 1 — each frame is
    // stamped with the actual system clock time it arrived, not the declared
    // framerate. This fixes sped-up video when our actual send rate drops below
    // the declared -framerate due to IPC skip-if-busy frame dropping.
    // CRITICAL for clean clip/seek playback: the `segment` muxer can only cut at a
    // KEYFRAME, AND `-c copy` event clips/seeks can only START at a keyframe — so
    // any clip sliced mid-GOP shows BLACK until the next keyframe. With one
    // keyframe per 10s segment that meant up to ~10s of "no footage" at the front
    // of every event clip (and a mangled tail). Fix: force a keyframe every
    // KEYFRAME_SECS (~1s) like mature NVRs/IP cameras. Segment boundaries still land on
    // a keyframe because `seg_secs` is a whole multiple of KEYFRAME_SECS, so the
    // on-time segment cut is preserved. Cost: modestly larger files.
    const KEYFRAME_SECS: u64 = 1;
    let gop = (30 * KEYFRAME_SECS).to_string();
    let kf_expr = format!("expr:gte(t,n_forced*{})", KEYFRAME_SECS);
    let pattern_s = pattern.to_string_lossy().into_owned();

    // Build the ffmpeg args, optionally muxing a separate mic input (input #1) into
    // the segments. With NO mic this is byte-identical to the long-standing
    // video-only recorder (zero regression for RTSP / audio-off cams).
    let make_args = |mic: &Option<Vec<String>>| -> Vec<String> {
        let mut a: Vec<String> = vec!["-hide_banner".into(), "-loglevel".into(), "error".into(), "-y".into()];
        // +genpts keeps the two independent live inputs (frame pipe + mic) aligned.
        if mic.is_some() { a.extend(["-fflags".into(), "+genpts".into()]); }
        a.extend([
            "-use_wallclock_as_timestamps".into(), "1".into(),
            "-f".into(), "mjpeg".into(), "-framerate".into(), "30".into(),
            "-i".into(), "pipe:0".into(),
        ]);
        if let Some(m) = mic {
            // Big queue so a momentary frame-pipe stall can't drop/block mic packets.
            a.extend(["-thread_queue_size".into(), "1024".into()]);
            a.extend(m.iter().cloned());
            a.extend(["-map".into(), "0:v:0".into(), "-map".into(), "1:a:0".into()]);
        }
        a.extend(enc_args.clone());
        if mic.is_some() { a.extend(["-c:a".into(), "aac".into(), "-b:a".into(), "96k".into()]); }
        a.extend(segment_output_args(cam_id, &nvr_dir));
        a
    };
    let _ = (&gop, &kf_expr, &pattern_s, seg_secs); // superseded by segment_output_args

    let spawn_one = |args: Vec<String>| {
        crate::proc::tokio_cmd(&ffmpeg)
            .args(&args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::null())
            // CAPTURE stderr (was black-holed): when a recorder exits mid-session the
            // reason lived only in ffmpeg's stderr, so the watchdog just saw "pipe died"
            // with no cause. Now every ffmpeg error line is logged, so a flapping NVR
            // pipe is diagnosable instead of silent.
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true)
            .spawn()
    };

    let mic = get_nvr_mic(cam_id);
    let mut child = spawn_one(make_args(&mic))
        .map_err(|e| anyhow::anyhow!("Failed to start ffmpeg NVR ({encoder}): {e}"))?;
    // FAIL-SAFE: if the mic-augmented recorder dies immediately (mic busy/invalid),
    // drop the mic and respawn video-only so RECORDING IS NEVER LOST to an audio issue.
    if mic.is_some() {
        tokio::time::sleep(std::time::Duration::from_millis(350)).await;
        if matches!(child.try_wait(), Ok(Some(_)) | Err(_)) {
            tracing::warn!("NVR cam{}: audio-muxed recorder exited at startup — falling back to video-only", cam_id);
            set_nvr_mic(cam_id, None);
            child = spawn_one(make_args(&None))
                .map_err(|e| anyhow::anyhow!("Failed to start ffmpeg NVR fallback ({encoder}): {e}"))?;
        }
    }

    let stdin = child.stdin.take().ok_or_else(|| anyhow::anyhow!("ffmpeg stdin unavailable"))?;
    // Drain + log the recorder's own error output so a pipe that exits tells us WHY
    // (mic/device error, encoder failure, bad input …) instead of dying silently.
    if let Some(stderr) = child.stderr.take() {
        let cam = cam_id;
        tokio::spawn(async move {
            use tokio::io::{AsyncBufReadExt, BufReader};
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let l = line.trim();
                if !l.is_empty() { tracing::warn!("NVR ffmpeg cam{}: {}", cam, l); }
            }
        });
    }
    // Bounded (~3 s of frames): a stalled encoder drops frames instead of
    // buffering RAM without limit. Producers use try_send (see capture.rs).
    let (tx, rx) = tokio::sync::mpsc::channel::<Arc<Vec<u8>>>(90);
    tokio::spawn(pipe_frames_to_ffmpeg(rx, stdin));

    // Segment post-processor (global single scanner — see ensure_postprocessor).
    ensure_postprocessor(data_dir, app_handle2, db2).await;

    Ok((tx, child))
}


/// Does this finalized MP4 contain an audio stream? One quick `ffmpeg -i` probe
/// (stderr parse — we ship ffmpeg, not ffprobe). Powers the per-segment
/// `has_audio` flag that playback uses to decide whether to map audio.
async fn file_has_audio(ffmpeg: &std::path::Path, file: &std::path::Path) -> bool {
    let out = crate::proc::tokio_cmd(ffmpeg)
        .args(["-hide_banner", "-i", &file.to_string_lossy()])
        .output().await;
    match out {
        Ok(o) => String::from_utf8_lossy(&o.stderr).contains(" Audio:"),
        Err(_) => false,
    }
}

/// Watches the NVR directory for completed .tmp.mp4 segments and converts them
/// to seekable faststart .mp4 files.  A segment is "complete" when it has not
/// been modified for at least segment_secs + 30 seconds (meaning ffmpeg has
/// rotated to the next file).
/// standard segment post-processor.
/// Converts completed .tmp.mp4 → faststart .mp4 then indexes them in nvr_segments DB.
/// The DB is the single source of truth — list_nvr_recordings and nvr_concat both
/// query it instead of scanning the filesystem.
pub(crate) async fn postprocess_nvr_segments(
    ffmpeg: std::path::PathBuf,
    nvr_dir: std::path::PathBuf,
    app_handle: tauri::AppHandle,
    db: SqlitePool,
) {
    // A segment is "done" once ffmpeg has ROTATED away from it — detectable as
    // the .tmp.mp4 not being modified for a short window. ffmpeg writes the
    // active segment continuously, so a few seconds of no-modification reliably
    // means it's closed. We keep this SMALL (not segment_secs*2) so a just-ended
    // segment is finalized + indexed within seconds — that's what makes a fresh
    // event's clip available almost immediately instead of minutes later, and
    // keeps the DB the up-to-date single source of truth.
    let stale_threshold = std::time::Duration::from_secs(8);
    // Poll every 2 seconds for low finalize latency.
    let mut interval = tokio::time::interval(std::time::Duration::from_secs(2));
    // Track how many times each file has failed so we can delete corrupt ones.
    let mut fail_counts: std::collections::HashMap<String, u8> = std::collections::HashMap::new();
    loop {
        interval.tick().await;
        let mut entries = match tokio::fs::read_dir(&nvr_dir).await {
            Ok(e) => e, Err(_) => continue,
        };
        while let Ok(Some(entry)) = entries.next_entry().await {
            let path = entry.path();
            let fname = match path.file_name().and_then(|n| n.to_str()) {
                Some(f) => f.to_string(), None => continue,
            };
            if !fname.starts_with("cam") || !fname.ends_with(".tmp.mp4") { continue; }
            // Already given up on this file (locked corrupt) — don't re-attempt
            // every 5s; that's what jammed the pipeline.
            if fail_counts.get(&fname).copied().unwrap_or(0) >= 100 { continue; }

            // Capture .tmp.mp4 mtime BEFORE postprocessing — this is when ffmpeg
            // stopped writing the file (segment ended naturally). Used as ended_at.
            let tmp_meta = match path.metadata() {
                Ok(m) => m, Err(_) => continue,
            };
            let tmp_modified = match tmp_meta.modified() {
                Ok(t) => t, Err(_) => continue,
            };
            let age = std::time::SystemTime::now().duration_since(tmp_modified).unwrap_or_default();
            if age < stale_threshold { continue; }

            let final_name = fname.replace(".tmp.mp4", ".mp4");
            let final_path = nvr_dir.join(&final_name);

            if final_path.exists() {
                let _ = tokio::fs::remove_file(&path).await;
                continue;
            }

            // Run +faststart conversion
            let result = crate::proc::tokio_cmd(&ffmpeg)
                .args([
                    "-hide_banner", "-loglevel", "error", "-y",
                    "-i", &path.to_string_lossy(),
                    "-c", "copy",
                    "-movflags", "+faststart",
                    &final_path.to_string_lossy(),
                ])
                .status().await;

            if !result.map(|s| s.success()).unwrap_or(false) {
                let count = fail_counts.entry(fname.clone()).or_insert(0);
                *count += 1;
                if *count >= 3 {
                    // Corrupt (moov atom missing, ffmpeg killed mid-write). Get it
                    // OUT of the way so it stops jamming the post-processor. Try a
                    // delete first; if that fails (Windows file lock / still-open
                    // handle), RENAME it to .corrupt — which the loop ignores
                    // (it only scans .tmp.mp4). This is the fix for the infinite
                    // "deleting corrupt segment" loop that blocked new segments
                    // (and made today show no recordings).
                    if tokio::fs::remove_file(&path).await.is_err() {
                        let corrupt = nvr_dir.join(format!("{}.corrupt", fname));
                        if tokio::fs::rename(&path, &corrupt).await.is_err() {
                            // Still stuck (locked): cap the counter high so we stop
                            // retrying this file every 5s and move on to others.
                            *count = 250;
                            tracing::warn!("NVR: cannot remove/rename corrupt {} (locked) — skipping", fname);
                            continue;
                        }
                        tracing::warn!("NVR: quarantined corrupt segment {} (delete failed)", fname);
                    } else {
                        tracing::warn!("NVR: deleted corrupt segment {} after 3 failed attempts", fname);
                    }
                    fail_counts.remove(&fname);
                } else {
                    tracing::warn!("NVR postprocess attempt {}/3 failed for {}", count, fname);
                }
                continue;
            }
            fail_counts.remove(&fname);
            let _ = tokio::fs::remove_file(&path).await;

            // ── Index the segment in the DB (mature NVRs' approach) ─────────────────
            // Parse started_at from the filename (local time → UTC)
            let stem = final_name.trim_end_matches(".mp4");
            let all_parts: Vec<&str> = stem.split('_').collect();
            let started_at_rfc = if all_parts.len() >= 3 {
                segment_local_name_to_utc(all_parts[all_parts.len() - 2], all_parts[all_parts.len() - 1])
                    .unwrap_or_else(|| Utc::now().to_rfc3339())
            } else {
                Utc::now().to_rfc3339()
            };

            // ended_at = .tmp.mp4 mtime (when ffmpeg finished writing the segment)
            let ended_at_rfc = chrono::DateTime::<Utc>::from_timestamp(
                tmp_modified.duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs() as i64,
                0,
            ).unwrap_or_else(Utc::now).to_rfc3339();

            // Exact duration = ended_at - started_at (accurate to the second)
            let start_unix = chrono::DateTime::parse_from_rfc3339(&started_at_rfc)
                .map(|t| t.timestamp()).unwrap_or(0);
            let end_unix = chrono::DateTime::parse_from_rfc3339(&ended_at_rfc)
                .map(|t| t.timestamp()).unwrap_or(0);
            let duration_secs = (end_unix - start_unix).max(1) as f64;

            let size_bytes = tokio::fs::metadata(&final_path).await
                .map(|m| m.len()).unwrap_or(0) as i64;

            let cam_num: u8 = all_parts.first()
                .and_then(|s| s.trim_start_matches("cam").parse().ok())
                .unwrap_or(0);

            let seg_id = uuid::Uuid::new_v4().to_string();
            let path_str = final_path.to_string_lossy().to_string();
            let filename = final_name.clone();
            let has_audio = file_has_audio(&ffmpeg, &final_path).await;

            sqlx::query(
                "INSERT OR IGNORE INTO nvr_segments(id,cam_id,path,started_at,ended_at,duration_secs,size_bytes,has_audio) VALUES(?,?,?,?,?,?,?,?)"
            )
            .bind(&seg_id).bind(cam_num as i64).bind(&path_str)
            .bind(&started_at_rfc).bind(&ended_at_rfc)
            .bind(duration_secs).bind(size_bytes).bind(has_audio as i64)
            .execute(&db).await.ok();

            tracing::info!("NVR: indexed {} ({:.0}s)", filename, duration_secs);
            app_handle.emit("nvr:segment-saved", serde_json::json!({ "filename": filename })).ok();
        }
    }
}

/// Delete orphaned .tmp.mp4 files left by a previously killed session.
/// Called once on startup before NVR pipes are spawned.
///
/// CRITICAL: only delete files that are genuinely STALE. The frontend can race
/// ahead of boot and start an external capture (dshow USB / RTSP relay) *before*
/// this runs — its in-progress segment is a live `.tmp.mp4` being written right
/// now. A true orphan from a dead session hasn't been touched in many seconds;
/// an active segment is modified continuously. Deleting an active one truncated
/// the recording (the "something is deleting my footage" bug for dshow cams).
pub(crate) async fn cleanup_orphaned_nvr_temps(nvr_dir: &std::path::Path) {
    const STALE_SECS: u64 = 30; // a live segment is written every frame; 30s of silence ⇒ dead session
    let mut entries = match tokio::fs::read_dir(nvr_dir).await { Ok(e) => e, Err(_) => return };
    while let Ok(Some(entry)) = entries.next_entry().await {
        let path = entry.path();
        if path.to_str().map(|p| p.ends_with(".tmp.mp4")).unwrap_or(false) {
            // Skip anything modified recently — it belongs to a capture that's
            // already running (raced ahead of boot), not a previous session.
            let fresh = path.metadata().ok()
                .and_then(|m| m.modified().ok())
                .and_then(|t| t.elapsed().ok())
                .map(|age| age.as_secs() < STALE_SECS)
                .unwrap_or(false);
            if fresh { continue; }
            let _ = tokio::fs::remove_file(&path).await;
            tracing::info!("NVR: removed orphaned {:?}", path.file_name().unwrap_or_default());
        }
    }
}

/// Scan the nvr directory and insert any .mp4 files that are on disk but not yet
/// indexed in nvr_segments. Called on startup so all recordings are visible immediately.
pub(crate) async fn reindex_existing_nvr_segments(nvr_dir: &std::path::Path, db: &SqlitePool) {
    let mut entries = match tokio::fs::read_dir(nvr_dir).await { Ok(e) => e, Err(_) => return };
    let mut indexed = 0u32;
    while let Ok(Some(entry)) = entries.next_entry().await {
        let path = entry.path();
        let fname = match path.file_name().and_then(|n| n.to_str()) {
            Some(f) => f.to_string(), None => continue,
        };
        if !fname.ends_with(".mp4") || fname.ends_with(".tmp.mp4") { continue; }

        // Skip if already indexed
        let exists: bool = sqlx::query_scalar("SELECT COUNT(*) > 0 FROM nvr_segments WHERE path=?")
            .bind(path.to_string_lossy().as_ref())
            .fetch_one(db).await.unwrap_or(false);
        if exists { continue; }

        // Parse cam_id + started_at from filename
        let stem = fname.trim_end_matches(".mp4");
        let all_parts: Vec<&str> = stem.split('_').collect();
        if all_parts.len() < 3 { continue; }
        let cam_num: u8 = all_parts[0].trim_start_matches("cam").parse().unwrap_or(0);
        let date_p = all_parts[all_parts.len() - 2];
        let time_p = all_parts[all_parts.len() - 1];
        if date_p.len() != 8 || time_p.len() != 6 { continue; }

        let started_at_rfc = segment_local_name_to_utc(date_p, time_p)
            .unwrap_or_else(|| Utc::now().to_rfc3339());

        let meta = tokio::fs::metadata(&path).await;
        let size_bytes = meta.as_ref().map(|m| m.len()).unwrap_or(0) as i64;
        let mtime_secs: Option<u64> = meta.as_ref().ok()
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs());
        let start_unix = chrono::DateTime::parse_from_rfc3339(&started_at_rfc)
            .map(|t| t.timestamp() as u64).unwrap_or(0);
        let duration_secs = mtime_secs
            .map(|mt| mt.saturating_sub(start_unix).saturating_sub(15).max(10))
            .unwrap_or(60) as f64;
        let ended_unix = start_unix + duration_secs as u64;
        let ended_at_rfc = chrono::DateTime::<Utc>::from_timestamp(ended_unix as i64, 0)
            .unwrap_or_else(Utc::now).to_rfc3339();

        let seg_id = uuid::Uuid::new_v4().to_string();
        sqlx::query(
            "INSERT OR IGNORE INTO nvr_segments(id,cam_id,path,started_at,ended_at,duration_secs,size_bytes) VALUES(?,?,?,?,?,?,?)"
        )
        .bind(&seg_id).bind(cam_num as i64)
        .bind(path.to_string_lossy().as_ref())
        .bind(&started_at_rfc).bind(&ended_at_rfc)
        .bind(duration_secs).bind(size_bytes)
        .execute(db).await.ok();
        indexed += 1;
    }
    if indexed > 0 {
        tracing::info!("NVR: re-indexed {} existing segments on startup", indexed);
    }
}

/// Prune nvr_segments DB rows whose file no longer exists on disk. Now that the
/// DB is the single source of truth for list_nvr_recordings (no filesystem
/// fallback), a stale row would surface a segment the player can't load. Runs
/// once at startup, after reindex, so the listed archive always matches disk.
pub(crate) async fn prune_orphaned_nvr_segments(db: &SqlitePool) {
    let paths: Vec<(String, String)> = sqlx::query_as("SELECT id, path FROM nvr_segments")
        .fetch_all(db).await.unwrap_or_default();
    let mut removed = 0u32;
    for (id, path) in paths {
        if !tokio::fs::try_exists(&path).await.unwrap_or(false) {
            sqlx::query("DELETE FROM nvr_segments WHERE id=?")
                .bind(&id).execute(db).await.ok();
            removed += 1;
        }
    }
    if removed > 0 {
        tracing::info!("NVR: pruned {} orphaned segment rows (file missing)", removed);
    }
}

/// Spawn an ffmpeg subprocess that generates an HLS stream from JPEG frames.
/// Produces a .m3u8 playlist + .ts segments served at /hls/camN.m3u8.
pub(crate) async fn spawn_hls_pipe(
    cam_id: u8,
    data_dir: &std::path::Path,
    encoder: &str,
) -> anyhow::Result<(tokio::sync::mpsc::Sender<Arc<Vec<u8>>>, tokio::process::Child)> {
    let ffmpeg   = ensure_ffmpeg(data_dir).await?;
    let hls_dir  = data_dir.join("hls");
    tokio::fs::create_dir_all(&hls_dir).await?;
    let m3u8     = hls_dir.join(format!("cam{}.m3u8", cam_id));
    let segments = hls_dir.join(format!("cam{}_%04d.ts", cam_id));
    // Budget-aware: this legacy pipe (browser/depth cams only now) is a real
    // encode session and must count against the NVENC budget like any other.
    let encoder  = assign_encoder(cam_id, encoder);
    let enc_args = hw_encoder_args(&encoder);
    let seg_str  = segments.to_string_lossy().into_owned();
    let m3u8_str = m3u8.to_string_lossy().into_owned();

    let mut args: Vec<String> = vec![
        "-hide_banner".into(), "-loglevel".into(), "error".into(), "-y".into(),
        // NOTE: for the mjpeg demuxer this flag is effectively ignored —
        // `-framerate` below does the (CFR) stamping. Kept for parity with the
        // recorder input; the REAL live-choppiness fix was raising the capture
        // chain from the old fps=15 cap to 30 (see dshow.rs spawn_capture).
        "-use_wallclock_as_timestamps".into(), "1".into(),
        "-f".into(), "mjpeg".into(), "-framerate".into(), "30".into(),
        "-i".into(), "pipe:0".into(),
    ];
    args.extend(enc_args);
    args.extend([
        // 1s keyframe interval + 1s segments + a short list = near-live latency (~2-3s)
        // for the live monitor, while `delete_segments` keeps the on-disk set bounded.
        "-g".into(), "30".into(),
        "-hls_time".into(), "1".into(),
        // 10s window (was 4): a 4-deep playlist left the player ~2s of margin —
        // any scheduling hiccup underran the buffer and stuttered the live view.
        "-hls_list_size".into(), "10".into(),
        "-hls_flags".into(), "delete_segments+append_list+discont_start".into(),
        "-hls_segment_filename".into(), seg_str,
        m3u8_str,
    ]);

    let mut child = crate::proc::tokio_cmd(&ffmpeg)
        .args(&args)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| anyhow::anyhow!("Failed to start ffmpeg HLS ({encoder}): {e}"))?;

    let stdin = child.stdin.take().ok_or_else(|| anyhow::anyhow!("ffmpeg stdin unavailable"))?;
    // Log the HLS ffmpeg's own errors — it runs but silently produces no playlist when
    // the encoder or input is unhappy; without this the failure is invisible.
    if let Some(stderr) = child.stderr.take() {
        tokio::spawn(async move {
            use tokio::io::{AsyncBufReadExt, BufReader};
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let l = line.trim();
                if !l.is_empty() { tracing::warn!("HLS ffmpeg cam{}: {}", cam_id, l); }
            }
        });
    }
    // Bounded (~3 s of frames): a stalled encoder drops frames instead of
    // buffering RAM without limit. Producers use try_send (see capture.rs).
    let (tx, rx) = tokio::sync::mpsc::channel::<Arc<Vec<u8>>>(90);
    tokio::spawn(pipe_frames_to_ffmpeg(rx, stdin));
    Ok((tx, child))
}

/// Drain a channel of JPEG frames and write them to an ffmpeg stdin pipe.
pub(crate) async fn pipe_frames_to_ffmpeg(
    mut rx: tokio::sync::mpsc::Receiver<Arc<Vec<u8>>>,
    mut stdin: tokio::process::ChildStdin,
) {
        while let Some(frame) = rx.recv().await {
        if stdin.write_all(&frame).await.is_err() { break; }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // `segment_local_name_to_utc` is the ONE place ffmpeg's LOCAL-time segment filename
    // (`-strftime`) becomes the UTC the DB stores. A bug here silently misplaces every
    // clip/timeline (we lived that). These tests are TZ-independent.

    #[test]
    fn rejects_malformed_filenames() {
        assert!(segment_local_name_to_utc("2026062", "224035").is_none());  // 7-char date
        assert!(segment_local_name_to_utc("20260629", "22403").is_none());  // 5-char time
        assert!(segment_local_name_to_utc("notadate", "224035").is_none()); // non-numeric
        assert!(segment_local_name_to_utc("", "").is_none());
    }

    #[test]
    fn round_trips_local_wall_clock() {
        // Filename is LOCAL wall-clock; the result is UTC. Converting back to local must
        // reproduce the exact digits — in ANY timezone (no DST gap at :40:35).
        let utc = segment_local_name_to_utc("20260629", "224035").expect("valid");
        let local = chrono::DateTime::parse_from_rfc3339(&utc)
            .expect("rfc3339")
            .with_timezone(&chrono::Local);
        assert_eq!(local.format("%Y%m%d%H%M%S").to_string(), "20260629224035");
    }

    #[test]
    fn emits_utc_offset_zero() {
        let utc = segment_local_name_to_utc("20260101", "000000").expect("valid");
        let dt = chrono::DateTime::parse_from_rfc3339(&utc).expect("rfc3339");
        assert_eq!(dt.offset().local_minus_utc(), 0, "stored timestamp must be UTC");
    }
}
