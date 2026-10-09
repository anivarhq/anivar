//! Rust-side NVR encoding pipeline: feeds JPEG frames through ffmpeg subprocess pipes for both continuous H.264 MP4 segments and a live HLS playlist.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use chrono::Utc;
use tokio::io::AsyncWriteExt;
use sqlx::SqlitePool;
use tauri::Emitter;

use crate::ensure_ffmpeg;
use crate::hw::hw_encoder_args;

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
/// wall-clock time → UTC rfc3339. Returns None for a malformed name.
///
/// `written_unix` is when the file was last written (its mtime). It settles the hour
/// that happens twice when clocks go back: 01:30 occurs at both 00:30 and 01:30 UTC in
/// London on the last Sunday of October. This used to return None there, the caller
/// fell back to "now", and every segment of that hour got the wrong start and a 1 s
/// length.
pub(crate) fn segment_local_name_to_utc(date_part: &str, time_part: &str, written_unix: Option<i64>)
    -> Option<String>
{
    local_name_to_utc(&chrono::Local, date_part, time_part, written_unix).map(|t| t.to_rfc3339())
}

fn local_name_to_utc<Tz: chrono::TimeZone>(tz: &Tz, date_part: &str, time_part: &str, written_unix: Option<i64>)
    -> Option<chrono::DateTime<Utc>>
{
    if date_part.len() != 8 || time_part.len() != 6 { return None; }
    let naive = chrono::NaiveDateTime::parse_from_str(&format!("{date_part}{time_part}"), "%Y%m%d%H%M%S").ok()?;
    match tz.from_local_datetime(&naive) {
        chrono::LocalResult::Single(t) => Some(t.with_timezone(&Utc)),
        // The segment STARTED before the file was last written, so the right
        // occurrence is the latest one at or before the mtime (2 s of slack for
        // clock rounding). With no mtime, take the earlier one.
        chrono::LocalResult::Ambiguous(a, b) => {
            let (a, b) = (a.with_timezone(&Utc), b.with_timezone(&Utc));
            let (early, late) = if a <= b { (a, b) } else { (b, a) };
            Some(match written_unix {
                Some(w) if late.timestamp() <= w + 2 => late,
                _ => early,
            })
        }
        // A skipped wall-clock time (clocks went forward). ffmpeg's clock never
        // shows one, so a name like this is not one of ours.
        chrono::LocalResult::None => None,
    }
}

//
// Frames flow: run_capture_loop / process_frame → jpeg_arc → pipe channel
//              → ffmpeg stdin → H.264 MP4 segments (NVR) / HLS (stream)
//
// Zero browser involvement: no MediaRecorder, no IPC round-trip, no UI thread.

/// standard NVR pipeline:
///   1. Record → ffmpeg → `nvr/incoming/*.tmp.mp4` 10-second segments
///   2. A finished segment (rotated, `moov` written) is renamed into `nvr/` and
///      indexed in `nvr_segments` (`postprocess_nvr_segments`)
///   3. Everything reads the DB, never the directory
///
/// Output-side args for NVR segment recording — the SINGLE definition of the
/// segment contract (gop/keyframes/10s cuts/tmp naming), shared by the legacy
/// pipe recorder and the merged capture+record ffmpeg so they can never drift.
pub(crate) fn segment_output_args(cam_id: u8, nvr_dir: &std::path::Path) -> Vec<String> {
    let pattern = incoming_dir(nvr_dir).join(format!("cam{}_%Y%m%d_%H%M%S.tmp.mp4", cam_id));
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
segment_time_delta=0.05:strftime=1]nvr/incoming/cam{cam_id}_%Y%m%d_%H%M%S.tmp.mp4");
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
    let nvr_dir = data_dir.join("nvr");
    let _ = tokio::fs::create_dir_all(incoming_dir(&nvr_dir)).await;
    tokio::spawn(async move {
        postprocess_nvr_segments(nvr_dir, app_handle, db).await;
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
    tokio::fs::create_dir_all(incoming_dir(&nvr_dir)).await?;
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


/// The body of an MP4's `moov` box, wherever it is (front with faststart, back
/// without). `None` when the file isn't a parseable MP4 or has no `moov` yet (a
/// segment still being written, or one killed before ffmpeg wrote it). The
/// header questions below all start here, so none of them needs an ffmpeg.
pub(crate) fn read_moov<R: std::io::Read + std::io::Seek>(r: &mut R) -> Option<Vec<u8>> {
    use std::io::SeekFrom;
    let len = r.seek(SeekFrom::End(0)).ok()?;
    let mut pos = 0u64;
    while pos + 8 <= len {
        r.seek(SeekFrom::Start(pos)).ok()?;
        let mut hdr = [0u8; 8];
        r.read_exact(&mut hdr).ok()?;
        let kind: [u8; 4] = hdr[4..8].try_into().ok()?;
        let mut size = u32::from_be_bytes(hdr[..4].try_into().ok()?) as u64;
        let mut hdr_len = 8u64;
        if size == 1 {
            let mut big = [0u8; 8];
            r.read_exact(&mut big).ok()?;
            size = u64::from_be_bytes(big);
            hdr_len = 16;
        } else if size == 0 {
            size = len - pos; // box runs to end of file
        }
        if size < hdr_len { return None; }
        if &kind == b"moov" {
            let body = (size - hdr_len).min(16 << 20) as usize;
            let mut buf = vec![0u8; body];
            r.read_exact(&mut buf).ok()?;
            return Some(buf);
        }
        pos = pos.checked_add(size)?;
    }
    None
}

/// Does this MP4 carry an audio track? A `hdlr` whose handler type is `soun`.
/// Same answer as `ffmpeg -i` (checked on real segments, a video-only copy and
/// the oldest archive segment) without a process per segment.
pub(crate) fn mp4_has_audio<R: std::io::Read + std::io::Seek>(r: &mut R) -> Option<bool> {
    read_moov(r).map(|m| moov_has_audio(&m))
}

fn moov_has_audio(moov: &[u8]) -> bool {
    // hdlr box: size, "hdlr", version+flags, pre_defined, handler_type.
    (0..moov.len().saturating_sub(15)).any(|i|
        &moov[i..i + 4] == b"hdlr" && &moov[i + 12..i + 16] == b"soun")
}

/// The movie's duration in seconds, from `mvhd` (a direct child of `moov`).
pub(crate) fn mp4_duration(moov: &[u8]) -> Option<f64> {
    let mut pos = 0usize;
    while pos + 8 <= moov.len() {
        let size = u32::from_be_bytes(moov[pos..pos + 4].try_into().ok()?) as usize;
        if size < 8 { return None; }
        if &moov[pos + 4..pos + 8] == b"mvhd" {
            let b = moov.get(pos + 8..pos + size)?;
            // version 0: 32-bit times; version 1: 64-bit. Both: timescale, then duration.
            let (scale, dur) = if b.first()? == &1 {
                (u32::from_be_bytes(b.get(20..24)?.try_into().ok()?) as f64,
                 u64::from_be_bytes(b.get(24..32)?.try_into().ok()?) as f64)
            } else {
                (u32::from_be_bytes(b.get(12..16)?.try_into().ok()?) as f64,
                 u32::from_be_bytes(b.get(16..20)?.try_into().ok()?) as f64)
            };
            return (scale > 0.0).then(|| dur / scale);
        }
        pos += size;
    }
    None
}

/// Is the video H.264 in a 4:4:4 or 4:2:2 profile, which WebView2 can't decode?
/// The `avcC` record's profile byte: 244 (High 4:4:4 Predictive), 122 (High
/// 4:2:2) or 44 (CAVLC 4:4:4). `stsd`, which holds `avcC`, comes before the
/// sample tables in `stbl`, so the first match is the real one.
pub(crate) fn mp4_needs_420(moov: &[u8]) -> bool {
    moov.windows(4).position(|w| w == b"avcC")
        .and_then(|i| moov.get(i + 5))
        .is_some_and(|p| matches!(p, 44 | 122 | 244))
}

/// Where recorders write the segment in progress. Finished segments are RENAMED
/// out into `nvr/`, so the post-processor lists a handful of files every 2 s,
/// never the archive (about 60k files per camera at 7-day retention).
pub(crate) fn incoming_dir(nvr_dir: &std::path::Path) -> std::path::PathBuf {
    nvr_dir.join("incoming")
}

/// Move a finished segment (ffmpeg has written its `moov`) out of `incoming/`
/// into `nvr/`, returning its new path and the `moov`. `Ok(None)`: no `moov`,
/// so the file can't be read (yet, or ever if ffmpeg was killed writing it).
fn finish_segment(nvr_dir: &std::path::Path, tmp: &std::path::Path)
    -> std::io::Result<Option<(std::path::PathBuf, Vec<u8>)>>
{
    let Some(moov) = std::fs::File::open(tmp).ok().and_then(|mut f| read_moov(&mut f)) else {
        return Ok(None);
    };
    let name = tmp.file_name().unwrap_or_default().to_string_lossy().replace(".tmp.mp4", ".mp4");
    let dest = nvr_dir.join(name);
    std::fs::rename(tmp, &dest)?;
    Ok(Some((dest, moov)))
}

/// Finalises segments as they finish. A segment is finished once ffmpeg has
/// rotated away from it (no writes for a few seconds) and has written its
/// `moov`. Finalising is a rename into `nvr/` plus a row in `nvr_segments`: the
/// DB is the single source of truth, and every reader of these files is ffmpeg,
/// which doesn't need faststart. (Each segment used to be remuxed with
/// `-movflags +faststart`: a second ffmpeg per segment per camera, ~170 ms of CPU.)
pub(crate) async fn postprocess_nvr_segments(
    nvr_dir: std::path::PathBuf,
    app_handle: tauri::AppHandle,
    db: SqlitePool,
) {
    // ffmpeg writes the active segment continuously, so a few seconds without a
    // write reliably means it has rotated. Kept SMALL so a just-ended segment is
    // indexed within seconds: that's what makes a fresh event's clip available.
    let stale_threshold = std::time::Duration::from_secs(8);
    let mut interval = tokio::time::interval(std::time::Duration::from_secs(2));
    // Polls an idle segment has gone without a moov; it's discarded after three.
    let mut fail_counts: std::collections::HashMap<String, u8> = std::collections::HashMap::new();
    loop {
        interval.tick().await;
        for final_name in finalise_incoming(&nvr_dir, &db, stale_threshold, &mut fail_counts).await {
            app_handle.emit("nvr:segment-saved", serde_json::json!({ "filename": final_name })).ok();
        }
    }
}

/// One pass over `incoming/`: finalises and indexes every segment that has gone
/// `stale` without a write, and returns the finished file names.
pub(crate) async fn finalise_incoming(
    nvr_dir: &std::path::Path,
    db: &SqlitePool,
    stale: std::time::Duration,
    fail_counts: &mut std::collections::HashMap<String, u8>,
) -> Vec<String> {
    let incoming = incoming_dir(nvr_dir);
    let mut indexed = Vec::new();
    let Ok(mut entries) = tokio::fs::read_dir(&incoming).await else { return indexed };
    while let Ok(Some(entry)) = entries.next_entry().await {
        let path = entry.path();
        let fname = match path.file_name().and_then(|n| n.to_str()) {
            Some(f) => f.to_string(), None => continue,
        };
        if !fname.starts_with("cam") || !fname.ends_with(".tmp.mp4") { continue; }
        // Already given up on this file (locked and unreadable).
        if fail_counts.get(&fname).copied().unwrap_or(0) >= 100 { continue; }

        // When ffmpeg stopped writing: the segment's end, and what resolves
        // the repeated hour at a DST change (see local_name_to_utc).
        let tmp_modified = match path.metadata().and_then(|m| m.modified()) {
            Ok(t) => t, Err(_) => continue,
        };
        let age = std::time::SystemTime::now().duration_since(tmp_modified).unwrap_or_default();
        if age < stale { continue; }

        let (nd, p) = (nvr_dir.to_path_buf(), path.clone());
        let finished = match tokio::task::spawn_blocking(move || finish_segment(&nd, &p)).await {
            Ok(r) => r,
            Err(_) => continue,
        };
        let (final_path, moov) = match finished {
            Ok(Some(done)) => done,
            Err(e) => { tracing::warn!("NVR: couldn't finalise {fname}: {e} — retrying"); continue; }
            Ok(None) => {
            // No moov: ffmpeg was killed mid-segment and the file can't be read.
            let count = fail_counts.entry(fname.clone()).or_insert(0);
            *count += 1;
            if *count >= 3 {
                // Delete it, or if Windows won't (a handle still open), move it
                // aside as .corrupt, which this loop ignores, so it can never
                // jam indexing.
                if tokio::fs::remove_file(&path).await.is_err() {
                    let corrupt = incoming.join(format!("{fname}.corrupt"));
                    if tokio::fs::rename(&path, &corrupt).await.is_err() {
                        *count = 250;
                        tracing::warn!("NVR: cannot remove/rename unreadable {fname} (locked) — skipping");
                        continue;
                    }
                    tracing::warn!("NVR: quarantined unreadable segment {fname} (delete failed)");
                } else {
                    tracing::warn!("NVR: deleted unreadable segment {fname} (no moov)");
                }
                fail_counts.remove(&fname);
            }
            continue;
            }
        };
        fail_counts.remove(&fname);
        let final_name = fname.replace(".tmp.mp4", ".mp4");

        // ── Index the segment in the DB ──────────────────────────────────
        // started_at from the filename (local time → UTC)
        let stem = final_name.trim_end_matches(".mp4");
        let all_parts: Vec<&str> = stem.split('_').collect();
        let written_unix = tmp_modified.duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64).unwrap_or(0);
        let started_at_rfc = if all_parts.len() >= 3 {
            segment_local_name_to_utc(all_parts[all_parts.len() - 2], all_parts[all_parts.len() - 1],
                Some(written_unix))
                .unwrap_or_else(|| Utc::now().to_rfc3339())
        } else {
            Utc::now().to_rfc3339()
        };
        let start_unix = chrono::DateTime::parse_from_rfc3339(&started_at_rfc)
            .map(|t| t.timestamp()).unwrap_or(0);
        // Length from the header; the file's write time only as a fallback.
        let duration_secs = mp4_duration(&moov)
            .unwrap_or(((written_unix - start_unix).max(1)) as f64);
        let ended_at_rfc = chrono::DateTime::<Utc>::from_timestamp(
            start_unix + duration_secs.round() as i64, 0,
        ).unwrap_or_else(Utc::now).to_rfc3339();

        let size_bytes = tokio::fs::metadata(&final_path).await
            .map(|m| m.len()).unwrap_or(0) as i64;
        let cam_num: u8 = all_parts.first()
            .and_then(|s| s.trim_start_matches("cam").parse().ok())
            .unwrap_or(0);
        let seg_id = uuid::Uuid::new_v4().to_string();
        let path_str = final_path.to_string_lossy().to_string();

        sqlx::query(
            "INSERT OR IGNORE INTO nvr_segments(id,cam_id,path,started_at,ended_at,duration_secs,size_bytes,has_audio) VALUES(?,?,?,?,?,?,?,?)"
        )
        .bind(&seg_id).bind(cam_num as i64).bind(&path_str)
        .bind(&started_at_rfc).bind(&ended_at_rfc)
        .bind(duration_secs).bind(size_bytes).bind(moov_has_audio(&moov) as i64)
        .execute(db).await.ok();

        tracing::info!("NVR: indexed {} ({:.1}s)", final_name, duration_secs);
        indexed.push(final_name);
    }
    indexed
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
        // A segment in progress from before `incoming/` existed: the post-processor
        // finishes it there (or discards it if it has no moov).
        if fname.ends_with(".tmp.mp4") {
            let _ = tokio::fs::rename(&path, incoming_dir(nvr_dir).join(&fname)).await;
            continue;
        }
        if !fname.ends_with(".mp4") { continue; }

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

        let meta = tokio::fs::metadata(&path).await;
        let size_bytes = meta.as_ref().map(|m| m.len()).unwrap_or(0) as i64;
        let mtime_secs: Option<u64> = meta.as_ref().ok()
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs());
        let started_at_rfc = segment_local_name_to_utc(date_p, time_p, mtime_secs.map(|s| s as i64))
            .unwrap_or_else(|| Utc::now().to_rfc3339());
        let start_unix = chrono::DateTime::parse_from_rfc3339(&started_at_rfc)
            .map(|t| t.timestamp() as u64).unwrap_or(0);
        // The header says how long the segment is. The old guess (file time minus
        // the start, minus 15 s) could over-declare a segment's length, and before
        // a gap that made hls.js jump forward to escape the hole.
        let p = path.clone();
        let moov = tokio::task::spawn_blocking(move ||
            std::fs::File::open(&p).ok().and_then(|mut f| read_moov(&mut f)))
            .await.ok().flatten();
        let duration_secs = moov.as_deref().and_then(mp4_duration).unwrap_or_else(|| mtime_secs
            .map(|mt| mt.saturating_sub(start_unix).saturating_sub(15).max(10))
            .unwrap_or(60) as f64);
        let ended_unix = start_unix + duration_secs.round() as u64;
        let ended_at_rfc = chrono::DateTime::<Utc>::from_timestamp(ended_unix as i64, 0)
            .unwrap_or_else(Utc::now).to_rfc3339();

        // has_audio was left out here, so it defaulted to 0: every segment
        // recovered after a DB reset played silent, and exports dropped audio.
        let has_audio = moov.as_deref().is_some_and(moov_has_audio);

        let seg_id = uuid::Uuid::new_v4().to_string();
        sqlx::query(
            "INSERT OR IGNORE INTO nvr_segments(id,cam_id,path,started_at,ended_at,duration_secs,size_bytes,has_audio) VALUES(?,?,?,?,?,?,?,?)"
        )
        .bind(&seg_id).bind(cam_num as i64)
        .bind(path.to_string_lossy().as_ref())
        .bind(&started_at_rfc).bind(&ended_at_rfc)
        .bind(duration_secs).bind(size_bytes).bind(has_audio as i64)
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

    fn mp4_box(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let mut v = ((8 + body.len()) as u32).to_be_bytes().to_vec();
        v.extend_from_slice(kind);
        v.extend_from_slice(body);
        v
    }
    /// ftyp + moov{trak{mdia{hdlr <handler>}}}, optionally with a big mdat first.
    fn mp4_with(handlers: &[&[u8; 4]], mdat_first: usize) -> Vec<u8> {
        let traks: Vec<u8> = handlers.iter().flat_map(|h| {
            let mut hdlr = vec![0u8; 8];            // version+flags, pre_defined
            hdlr.extend_from_slice(*h);              // handler_type
            hdlr.extend_from_slice(&[0u8; 13]);      // reserved + empty name
            mp4_box(b"trak", &mp4_box(b"mdia", &mp4_box(b"hdlr", &hdlr)))
        }).collect();
        let mut f = mp4_box(b"ftyp", b"isom\0\0\0\0isom");
        if mdat_first > 0 { f.extend(mp4_box(b"mdat", &vec![0u8; mdat_first])); }
        f.extend(mp4_box(b"moov", &traks));
        f
    }

    #[test]
    fn a_finished_segment_moves_out_and_a_killed_one_stays_put() {
        let dir = std::env::temp_dir().join(format!("anivar-finish-{}", uuid::Uuid::new_v4()));
        let incoming = incoming_dir(&dir);
        std::fs::create_dir_all(&incoming).unwrap();
        let finished = incoming.join("cam0_20261005_101500.tmp.mp4");
        let killed   = incoming.join("cam0_20261005_101510.tmp.mp4");
        std::fs::write(&finished, mp4_with(&[b"vide"], 1000)).unwrap();
        std::fs::write(&killed, mp4_box(b"ftyp", b"isom\0\0\0\0isom")).unwrap(); // no moov

        let (dest, moov) = finish_segment(&dir, &finished).unwrap().expect("has a moov");
        assert_eq!(dest, dir.join("cam0_20261005_101500.mp4"));
        assert!(dest.exists() && !finished.exists() && !moov.is_empty());

        assert!(finish_segment(&dir, &killed).unwrap().is_none(), "no moov: not finished");
        assert!(killed.exists(), "left for the post-processor to discard");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn mp4_duration_reads_mvhd_in_both_versions() {
        let mvhd = |version: u8, scale: u32, dur: u64| {
            let mut b = vec![version, 0, 0, 0];
            if version == 1 {
                b.extend([0u8; 16]);                       // creation + modification
                b.extend(scale.to_be_bytes());
                b.extend(dur.to_be_bytes());
            } else {
                b.extend([0u8; 8]);
                b.extend(scale.to_be_bytes());
                b.extend((dur as u32).to_be_bytes());
            }
            b.extend([0u8; 80]);                           // rate, volume, matrix, …
            mp4_box(b"mvhd", &b)
        };
        assert_eq!(mp4_duration(&mvhd(0, 1000, 10_040)), Some(10.04));
        let mut moov = mp4_box(b"trak", b"");                 // mvhd needn't be first
        moov.extend(mvhd(1, 90_000, 903_600));
        assert_eq!(mp4_duration(&moov), Some(10.04));
        assert_eq!(mp4_duration(&mp4_box(b"trak", b"")), None);
    }

    #[test]
    fn only_a_4_4_4_or_4_2_2_profile_needs_a_transcode() {
        // avcC: configurationVersion, then AVCProfileIndication.
        let avcc = |profile: u8| mp4_box(b"stsd", &mp4_box(b"avcC", &[1, profile, 0, 31]));
        assert!(mp4_needs_420(&avcc(244)), "High 4:4:4 Predictive (legacy yuvj444p)");
        assert!(mp4_needs_420(&avcc(122)), "High 4:2:2");
        assert!(!mp4_needs_420(&avcc(100)), "High: what every camera and recorder writes");
        assert!(!mp4_needs_420(&avcc(66)), "Baseline");
        assert!(!mp4_needs_420(b"no avcC at all (H.265, or not video)"));
    }

    #[test]
    fn mp4_has_audio_reads_the_moov_handlers() {
        use std::io::Cursor;
        let has = |b: Vec<u8>| mp4_has_audio(&mut Cursor::new(b));
        assert_eq!(has(mp4_with(&[b"vide", b"soun"], 0)), Some(true), "video + audio");
        assert_eq!(has(mp4_with(&[b"vide"], 0)), Some(false), "video only");
        assert_eq!(has(mp4_with(&[b"vide", b"soun"], 50_000)), Some(true), "moov after mdat");
        // "soun" inside the media data must not count: only moov is scanned.
        let mut decoy = mp4_box(b"ftyp", b"isom");
        decoy.extend(mp4_box(b"mdat", b"hdlr\0\0\0\0\0\0\0\0soun"));
        decoy.extend(mp4_box(b"moov", &mp4_box(b"trak", b"")));
        assert_eq!(has(decoy), Some(false), "mdat bytes are not handlers");
        // 64-bit box size (size field == 1): a 24-byte mdat between ftyp and moov.
        let base = mp4_with(&[b"soun"], 0);
        let (ftyp, moov) = base.split_at(20); // ftyp box is 8 + 12 bytes
        let mut big = ftyp.to_vec();
        big.extend(1u32.to_be_bytes()); big.extend(b"mdat");
        big.extend(24u64.to_be_bytes()); big.extend([0u8; 8]);
        big.extend_from_slice(moov);
        assert_eq!(has(big), Some(true), "largesize box skipped correctly");
        assert_eq!(has(b"not an mp4 at all".to_vec()), None);
        assert_eq!(has(Vec::new()), None);
    }

    // `segment_local_name_to_utc` is the ONE place ffmpeg's LOCAL-time segment filename
    // (`-strftime`) becomes the UTC the DB stores. A bug here silently misplaces every
    // clip/timeline (we lived that). These tests are TZ-independent.

    #[test]
    fn rejects_malformed_filenames() {
        assert!(segment_local_name_to_utc("2026062", "224035", None).is_none());  // 7-char date
        assert!(segment_local_name_to_utc("20260629", "22403", None).is_none());  // 5-char time
        assert!(segment_local_name_to_utc("notadate", "224035", None).is_none()); // non-numeric
        assert!(segment_local_name_to_utc("", "", None).is_none());
    }

    #[test]
    fn round_trips_local_wall_clock() {
        // Filename is LOCAL wall-clock; the result is UTC. Converting back to local must
        // reproduce the exact digits — in ANY timezone (no DST gap at :40:35).
        let utc = segment_local_name_to_utc("20260629", "224035", None).expect("valid");
        let local = chrono::DateTime::parse_from_rfc3339(&utc)
            .expect("rfc3339")
            .with_timezone(&chrono::Local);
        assert_eq!(local.format("%Y%m%d%H%M%S").to_string(), "20260629224035");
    }

    #[test]
    fn emits_utc_offset_zero() {
        let utc = segment_local_name_to_utc("20260101", "000000", None).expect("valid");
        let dt = chrono::DateTime::parse_from_rfc3339(&utc).expect("rfc3339");
        assert_eq!(dt.offset().local_minus_utc(), 0, "stored timestamp must be UTC");
    }

    /// Clocks go back in London on 2026-10-25: 01:00–02:00 local happens twice,
    /// at 00:xx UTC (BST) and again at 01:xx UTC (GMT). The file's mtime decides.
    #[test]
    fn the_repeated_hour_is_resolved_by_when_the_file_was_written() {
        let tz = chrono_tz::Europe::London;
        let first  = chrono::DateTime::parse_from_rfc3339("2026-10-25T00:30:00Z").unwrap().timestamp();
        let second = chrono::DateTime::parse_from_rfc3339("2026-10-25T01:30:00Z").unwrap().timestamp();
        // Written 10 s after each start.
        let a = local_name_to_utc(&tz, "20261025", "013000", Some(first + 10)).unwrap();
        let b = local_name_to_utc(&tz, "20261025", "013000", Some(second + 10)).unwrap();
        assert_eq!(a.timestamp(), first,  "first 01:30 (BST)");
        assert_eq!(b.timestamp(), second, "second 01:30 (GMT)");
        // No mtime: the earlier occurrence, never "now".
        assert_eq!(local_name_to_utc(&tz, "20261025", "013000", None).unwrap().timestamp(), first);
        // An ordinary time is unaffected by the hint.
        let noon = local_name_to_utc(&tz, "20261025", "120000", Some(0)).unwrap();
        assert_eq!(noon.to_rfc3339(), "2026-10-25T12:00:00+00:00");
        // The skipped hour in spring (clocks go forward 2026-03-29 01:00) can't be one of ours.
        assert!(local_name_to_utc(&tz, "20260329", "013000", None).is_none());
    }
}
