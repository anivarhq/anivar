//! Audio command surface: the shared audio-event lifecycle (used by both the RTSP
//! server-side tap in `rtsp.rs` and the USB/native tap in `dshow.rs`) + the
//! `analyze_audio_window` command the WebView calls with mic PCM.
//!
//! Audio events follow mature NVRs' SUSTAINED model: when a listened sound is heard we
//! OPEN an event (`ended_at` NULL); while it keeps being heard the event stays open;
//! once the sound has not been heard for [`AUDIO_MAX_NOT_HEARD_SECS`] we CLOSE it
//! (`ended_at` = last-heard time). A continuous alarm is therefore ONE coherent event
//! whose clip spans the whole sound — not a burst of 1-second blips. The event reuses
//! `motion_events`, so it flows through Review / timeline / alerts / search unchanged.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use tauri::{Emitter, State};

use crate::AppState;

/// Mature NVRs' `max_not_heard` (default 30 s): a sustained audio event ends once its
/// sound has been silent for this long. Reaping runs every detection window (~1 s).
const AUDIO_MAX_NOT_HEARD_SECS: u64 = 30;

/// A non-critical sound must be heard in this many confident ~1 s windows before
/// the user is ALERTED. The event itself still opens on the first window (so the
/// clip covers the sound from its start and the Audio tab shows every detection)
/// — only the notification waits. Single-window YAMNet blips (one transient
/// misread as bark/speech) were pinging Telegram instantly; multi-window
/// confirmation is the standard debounce in acoustic event detection.
const AUDIO_CONFIRM_WINDOWS: u32 = 2;

/// Impulse sounds that may never produce a second confident window — these alert
/// on the FIRST window (as critical), same as the high-pitch register.
const IMPULSE_CLASSES: &[&str] = &["gunshot", "gun shot", "explosion"];

fn is_impulse_class(name: &str) -> bool {
    let n = name.to_lowercase();
    IMPULSE_CLASSES.iter().any(|c| n.contains(c))
}

/// Ignore windows quieter than this RMS — mature NVRs' default `min_volume` is 500
/// on the s16 sample scale, ≈ 500/32768 ≈ 0.0153 in float RMS.
const MIN_VOLUME_RMS: f32 = 0.0153;

/// The currently-OPEN audio event for a camera (absent until a sound is heard).
struct OpenAudio {
    event_id:   String,
    started:    chrono::DateTime<chrono::Utc>,
    last_heard: Instant,
    peak_score: f32,
    label:      String,
    /// Any window of this event contained a high-pitch class ≥ threshold.
    high_pitch: bool,
    /// Loudest window so far (dBFS).
    loud_db:    f32,
    /// Top classes of the loudest window (for the card's class chips).
    top:        Vec<(String, f32)>,
    /// Confident windows heard so far (drives alert confirmation).
    hits:       u32,
    /// The user has been notified about this event (alert at most once per event).
    alerted:    bool,
}

fn open_audio() -> &'static Mutex<HashMap<u8, OpenAudio>> {
    static M: OnceLock<Mutex<HashMap<u8, OpenAudio>>> = OnceLock::new();
    M.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Lock the open-event map, recovering from a poisoned mutex instead of panicking
/// (the critical sections never await, so a poisoned guard is always safe to reuse).
fn lock_open() -> std::sync::MutexGuard<'static, HashMap<u8, OpenAudio>> {
    open_audio().lock().unwrap_or_else(|e| e.into_inner())
}

// ─── Rolling audio buffer (USB/native cams) ──────────────────────────────────
//
// USB cameras record video as a JPEG frame pipe (no audio track), so their NVR
// segments are silent. Rather than rearchitect the tuned capture/recording
// pipeline — or write+prune a second set of on-disk audio segments — we keep a
// small BOUNDED in-RAM ring of the last few minutes of 16 kHz mono PCM per camera
// and mux the matching window into the event clip at export time. No disk, no
// retention, RAM-capped (~8 MB/cam), and strictly additive: if anything here
// fails, the video-only clip is returned unchanged. RTSP cams DON'T fill the ring
// (their segments already carry audio), so they're never double-tracked.

/// How much recent audio to retain. Must exceed the longest clip window
/// (pre-buffer + capped event ≤120 s + post-buffer) plus the close-time pre-warm
/// delay, so a just-closed audio event's whole window is still buffered.
const RING_SECS: f64 = 240.0;

/// One ~1 s window: (wall-clock unix secs at capture, raw s16le bytes).
type AudioWin = (f64, Vec<u8>);

fn audio_ring() -> &'static Mutex<HashMap<u8, std::collections::VecDeque<AudioWin>>> {
    static M: OnceLock<Mutex<HashMap<u8, std::collections::VecDeque<AudioWin>>>> = OnceLock::new();
    M.get_or_init(|| Mutex::new(HashMap::new()))
}

fn push_ring(cam: u8, ts: f64, bytes: Vec<u8>) {
    let mut map = audio_ring().lock().unwrap_or_else(|e| e.into_inner());
    let dq = map.entry(cam).or_default();
    dq.push_back((ts, bytes));
    let cutoff = ts - RING_SECS;
    while matches!(dq.front(), Some((t, _)) if *t < cutoff) { dq.pop_front(); }
}

/// Concatenate the buffered PCM overlapping `[start, end]` (unix secs). Returns the
/// raw s16le bytes + the wall-clock start of the FIRST included window (so the
/// caller can offset it to line up with the video).
fn extract_ring(cam: u8, start: f64, end: f64) -> Option<(Vec<u8>, f64)> {
    let map = audio_ring().lock().unwrap_or_else(|e| e.into_inner());
    let dq = map.get(&cam)?;
    let mut out = Vec::new();
    let mut first: Option<f64> = None;
    for (t, b) in dq.iter() {
        // Each window covers ~[t-1, t]; include if it overlaps [start, end].
        if *t < start - 1.0 { continue; }
        if *t > end + 1.0 { break; }
        if first.is_none() { first = Some(*t - 1.0); }
        out.extend_from_slice(b);
    }
    if out.len() < 16_000 { return None; } // < ~0.5 s of audio — not worth muxing
    Some((out, first?))
}

/// Does this file already contain an audio stream? Used to keep the ring mux a pure
/// fallback (don't double-track clips whose segments already carry audio).
pub(crate) async fn clip_has_audio(ffmpeg: &std::path::Path, clip: &std::path::Path) -> bool {
    match crate::proc::tokio_cmd(ffmpeg)
        .args(["-hide_banner", "-i", &clip.to_string_lossy()])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .output().await
    {
        Ok(o) => String::from_utf8_lossy(&o.stderr).contains("Audio:"),
        Err(_) => false,
    }
}

/// Best-effort: mux the buffered audio for `[start_secs, end_secs]` into an already
/// exported (video-only) clip, IN PLACE. Returns true if audio was added. Never
/// fails the clip — on any error the original video clip is left untouched. Only
/// does anything when the ring has audio for this camera (i.e. a USB cam with audio
/// detection on); RTSP clips already carry their own audio and are skipped.
pub(crate) async fn mux_ring_audio(
    data_dir: &std::path::Path,
    cam: u8,
    start_secs: f64,
    end_secs: f64,
    clip_path: &std::path::Path,
) -> bool {
    let ffmpeg = match crate::ensure_ffmpeg(data_dir).await { Ok(f) => f, Err(_) => return false };
    // FALLBACK ONLY: if the clip already carries audio (USB segments now mux the mic,
    // or an RTSP camera's own audio), leave it alone — no needless re-encode/double-track.
    if clip_has_audio(&ffmpeg, clip_path).await { return false; }
    let (pcm, first_ts) = match extract_ring(cam, start_secs, end_secs) { Some(x) => x, None => return false };

    let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos()).unwrap_or(0);
    let tmp_pcm = data_dir.join(format!("clipaudio_cam{}_{}.pcm", cam, stamp));
    let tmp_out = data_dir.join(format!("clipmux_cam{}_{}.mp4", cam, stamp));
    if tokio::fs::write(&tmp_pcm, &pcm).await.is_err() { return false; }

    // Line the PCM up with the video: the first buffered window starts `offset` secs
    // into the requested window. Clamp ≥0 (a tiny lead-in is harmless; negative would
    // drop samples). Re-encode to AAC (mature NVRs `preset-record-*-audio-aac` parity).
    let offset = (first_ts - start_secs).max(0.0);
    let ok = crate::proc::tokio_cmd(&ffmpeg)
        .args([
            "-hide_banner", "-loglevel", "error", "-y",
            "-i", &clip_path.to_string_lossy(),
            "-itsoffset", &format!("{:.3}", offset),
            "-f", "s16le", "-ar", "16000", "-ac", "1", "-i", &tmp_pcm.to_string_lossy(),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-shortest",
            "-movflags", "+faststart",
            &tmp_out.to_string_lossy(),
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .kill_on_drop(true)
        .status().await
        .map(|s| s.success()).unwrap_or(false);
    let _ = tokio::fs::remove_file(&tmp_pcm).await;

    let big = tokio::fs::metadata(&tmp_out).await.map(|m| m.len() > 4096).unwrap_or(false);
    if ok && big {
        // Replace the video-only clip with the muxed one (remove-then-rename: Windows
        // rename won't overwrite an existing file).
        let _ = tokio::fs::remove_file(clip_path).await;
        if tokio::fs::rename(&tmp_out, clip_path).await.is_ok() {
            tracing::info!("AUDIO: muxed buffered audio into clip cam{} ({} KB pcm)", cam, pcm.len() / 1024);
            return true;
        }
    }
    let _ = tokio::fs::remove_file(&tmp_out).await;
    false
}

/// A listened sound was heard on `cam_id`. Opens a new sustained audio event or
/// extends the open one (refreshing its last-heard time + peak label). Debounced by
/// design: while an event is open, repeated hits never create duplicate events.
pub(crate) async fn note_audio_hit(
    state: &Arc<AppState>,
    cam_id: u8,
    sound: &str,
    score: f32,
    top: &[(String, f32)],
    rms: f32,
    threshold: f32,
) {
    // Per-window derived signals — YAMNet used PROPERLY: keep the top classes
    // (not just the fired label), flag high-pitch semantically, keep loudness.
    let ldb = 20.0 * rms.max(1e-5).log10();
    let win_hp = top.iter().any(|(n, s)| *s >= threshold && crate::audio::is_high_pitch_class(n));

    /// Serialize the audio metadata blob (motion_events.audio_meta — its own
    /// column; `attributes` belongs to the visual pipeline and clobbers guests).
    fn audio_attrs(top: &[(String, f32)], hp: bool) -> String {
        let classes: Vec<serde_json::Value> = top.iter().take(3)
            .map(|(l, s)| serde_json::json!({"l": l, "s": (s * 100.0).round() / 100.0}))
            .collect();
        serde_json::json!({"classes": classes, "high_pitch": hp}).to_string()
    }

    // Immediate-alert sounds: the high-pitch register (scream/alarm/glass) and
    // single-shot impulses (gunshot) — waiting for a second window could miss them.
    let win_critical = win_hp || is_impulse_class(sound);

    // Decide start-vs-extend under the lock; do all DB / alert work OUTSIDE it (never
    // hold a std Mutex across an await).
    enum Act {
        Start,
        Update {
            event_id: String,
            new_peak: bool,
            db_write: bool,
            attrs: String,
            loud_db: f32,
            // (alert_type, label, peak_score) when this window CONFIRMS the alert.
            alert: Option<(&'static str, String, f32)>,
        },
        None,
    }
    let act = {
        let mut map = lock_open();
        match map.get_mut(&cam_id) {
            Some(o) => {
                o.last_heard = Instant::now();
                o.hits += 1;
                let became_hp = win_hp && !o.high_pitch;
                o.high_pitch |= win_hp;
                o.loud_db = o.loud_db.max(ldb);
                let new_peak = score > o.peak_score;
                if new_peak {
                    o.peak_score = score;
                    o.label = sound.to_string();
                    o.top = top.iter().take(3).cloned().collect();
                }
                // Alert once the sound is CONFIRMED (2nd confident window) — or
                // immediately if this window turned the event critical.
                let alert = if !o.alerted && (win_critical || o.hits >= AUDIO_CONFIRM_WINDOWS) {
                    o.alerted = true;
                    let atype = if o.high_pitch || is_impulse_class(&o.label) { "audio_alarm" } else { "audio" };
                    Some((atype, o.label.clone(), o.peak_score))
                } else { None };
                let db_write = new_peak || became_hp;
                if db_write || alert.is_some() {
                    Act::Update {
                        event_id: o.event_id.clone(),
                        new_peak,
                        db_write,
                        attrs: audio_attrs(&o.top, o.high_pitch),
                        loud_db: o.loud_db,
                        alert,
                    }
                } else {
                    Act::None
                }
            }
            None => Act::Start,
        }
    };

    match act {
        Act::Start => {
            let id  = uuid::Uuid::new_v4().to_string();
            let now = chrono::Utc::now();
            // Card thumbnail = the moment's frame (depth-anonymization-safe:
            // latest_frames holds depth frames when that mode is on).
            let thumb: Option<String> = state.latest_frames.read().await.get(&cam_id)
                .map(|f| base64::Engine::encode(&base64::engine::general_purpose::STANDARD, f));
            let attrs = audio_attrs(&top.iter().take(3).cloned().collect::<Vec<_>>(), win_hp);
            // OPEN event (ended_at NULL = in-progress, mature NVRs model).
            let _ = sqlx::query(
                "INSERT INTO motion_events(id, started_at, ended_at, duration_secs, peak_score, event_category, dominant_label, cam_id, thumbnail, audio_meta, loudness_db)
                 VALUES(?,?,NULL,NULL,?,'audio',?,?,?,?,?)"
            ).bind(&id).bind(now.to_rfc3339()).bind(score).bind(sound).bind(cam_id as i64)
             .bind(&thumb).bind(&attrs).bind(ldb as f64)
             .execute(&state.db).await;
            crate::timeline::log(&state.db, &id, cam_id as i64,
                crate::timeline::class::AUDIO, Some("audio"), Some(sound), Some(score)).await;
            crate::review_segments::upsert_review_segment(&state.db, &id).await;
            lock_open().insert(cam_id, OpenAudio {
                event_id: id, started: now, last_heard: Instant::now(),
                peak_score: score, label: sound.to_string(),
                high_pitch: win_hp, loud_db: ldb,
                top: top.iter().take(3).cloned().collect(),
                hits: 1, alerted: win_critical,
            });
            tracing::info!("AUDIO: cam{} → '{}' ({:.2}) [start] hp={} {:.0}dB", cam_id, sound, score, win_hp, ldb);
            state.app_handle.emit("agent:analyzed", ()).ok();
            // Critical sounds (high-pitch register + impulses) notify immediately
            // and rank critical (survive quiet hours). Ordinary sounds wait for a
            // 2nd confident window (see Act::Update) — the event exists either
            // way; only the ping is debounced. The mute/threshold gate lives in
            // dispatch_intelligence_alert.
            if win_critical {
                let summary = format!("Audio: {} detected (confidence {:.0}%)", sound, score * 100.0);
                crate::agent::dispatch_intelligence_alert(state, "audio_alarm", &summary, cam_id, None).await;
            }
        }
        Act::Update { event_id, new_peak, db_write, attrs, loud_db, alert } => {
            if new_peak {
                let _ = sqlx::query(
                    "UPDATE motion_events SET peak_score=?, dominant_label=?, audio_meta=?, loudness_db=? WHERE id=?")
                    .bind(score).bind(sound).bind(&attrs).bind(loud_db as f64).bind(&event_id)
                    .execute(&state.db).await;
            } else if db_write {
                // high-pitch flag flipped mid-event without a new peak.
                let _ = sqlx::query(
                    "UPDATE motion_events SET audio_meta=?, loudness_db=? WHERE id=?")
                    .bind(&attrs).bind(loud_db as f64).bind(&event_id)
                    .execute(&state.db).await;
            }
            if let Some((atype, label, peak)) = alert {
                let summary = format!("Audio: {} detected (confidence {:.0}%)", label, peak * 100.0);
                crate::agent::dispatch_intelligence_alert(state, atype, &summary, cam_id, None).await;
            }
        }
        Act::None => {}
    }
}

/// Close any audio events whose sound has not been heard for [`AUDIO_MAX_NOT_HEARD_SECS`].
/// Cheap; call once per detection window so events also close during silence.
pub(crate) async fn reap_audio_events(state: &Arc<AppState>) {
    let now = Instant::now();
    let stale: Vec<(u8, OpenAudio)> = {
        let mut map = lock_open();
        let keys: Vec<u8> = map.iter()
            .filter(|(_, o)| now.duration_since(o.last_heard).as_secs() >= AUDIO_MAX_NOT_HEARD_SECS)
            .map(|(k, _)| *k).collect();
        keys.into_iter().filter_map(|k| map.remove(&k).map(|o| (k, o))).collect()
    };
    for (cam, o) in stale { finalize_audio_event(state, cam, o).await; }
}

/// Force-close the open audio event for a camera (called when its capture stops, so a
/// sound that was playing at teardown doesn't leave a perpetual open event).
pub(crate) async fn close_audio_for_cam(state: &Arc<AppState>, cam_id: u8) {
    let o = lock_open().remove(&cam_id);
    if let Some(o) = o { finalize_audio_event(state, cam_id, o).await; }
}

async fn finalize_audio_event(state: &Arc<AppState>, cam: u8, o: OpenAudio) {
    // ended_at ≈ when the sound was last heard (now − silence-elapsed), clamped so the
    // duration is always ≥ 1 s and the clip window stays valid.
    let silence = Instant::now().saturating_duration_since(o.last_heard);
    let ended = (chrono::Utc::now() - chrono::Duration::from_std(silence).unwrap_or_default())
        .max(o.started + chrono::Duration::seconds(1));
    let dur = ((ended - o.started).num_milliseconds() as f64 / 1000.0).max(1.0);
    let _ = sqlx::query("UPDATE motion_events SET ended_at=?, duration_secs=? WHERE id=?")
        .bind(ended.to_rfc3339()).bind(dur).bind(&o.event_id).execute(&state.db).await;
    crate::timeline::log_at(&state.db, &o.event_id, cam as i64, &ended.to_rfc3339(),
        crate::timeline::class::GONE, Some("audio"), Some(&o.label), Some(o.peak_score)).await;
    crate::review_segments::upsert_review_segment(&state.db, &o.event_id).await;
    tracing::info!("AUDIO: cam{} → '{}' ({:.2}) [end · {:.1}s]", cam, o.label, o.peak_score, dur);
    // Same signal Start emits — the Audio tab (and every other event surface)
    // refreshes on it, so the card's duration/clip-readiness appear live.
    state.app_handle.emit("agent:analyzed", ()).ok();
    // Pre-warm the (now audible) clip once the post-buffer footage has finalized.
    let st = state.clone();
    let eid = o.event_id.clone();
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_secs(14)).await;
        crate::agent::clip_export::ensure_event_clip(&st, &eid).await;
    });
}

/// Close any audio events left OPEN by a previous session (crash / hard-stop) for this
/// camera, so they don't linger as perpetual "in-progress" rows. A fresh detector means
/// nothing is actually ongoing, so every still-open audio event here is an orphan.
async fn close_orphan_audio_events(state: &Arc<AppState>, cam_id: u8) {
    let _ = sqlx::query(
        "UPDATE motion_events SET ended_at=started_at, duration_secs=1.0
         WHERE cam_id=? AND event_category='audio' AND ended_at IS NULL"
    ).bind(cam_id as i64).execute(&state.db).await;
}

/// Spawn a SERVER-SIDE audio-event detector for a camera. `input_args` are the per-source
/// ffmpeg INPUT args (RTSP: `-rtsp_transport tcp -i <url>`; USB: the OS capture device);
/// this appends the common `-vn -ac 1 -ar 16000 -f s16le` downmix, then runs YAMNet on 1 s
/// windows → opens/extends a sustained audio event for a loud, confident "listened" class.
/// Shared by the RTSP relay and the USB capture so there's ONE audio pipeline. Best-effort:
/// no-ops if audio detection is off, the skill isn't installed, or ffmpeg can't open audio.
/// Throttle for the "hearing X" trace log (once per ~10s across all cams).
static LAST_AUDIO_TRACE: std::sync::Mutex<Option<std::time::Instant>> = std::sync::Mutex::new(None);

pub(crate) async fn spawn_audio_detection(
    state: &Arc<AppState>,
    cam: u8,
    ffmpeg: &std::path::Path,
    input_args: Vec<String>,
    // USB/native cams record video as a silent JPEG pipe → buffer their audio so the
    // event clip can be muxed audible. RTSP cams pass false (segments already carry
    // their camera's audio; buffering would double-track it).
    record_audio: bool,
) {
    let (enabled, listen_raw, threshold) = {
        let s = state.settings.read().await;
        (s.audio_detection, s.audio_listen.clone(), s.audio_threshold)
    };
    if !enabled { return; }
    // RECORDING (buffer for clip muxing) is decoupled from DETECTION: the mic is
    // captured whenever audio is enabled, even if the YAMNet model is missing/broken,
    // so the user still HEARS audio in clips. DETECTION additionally needs a valid
    // model (`is_installed` now size-validates it) + a non-empty listen list.
    let model_ok = crate::audio::is_installed(&state.data_dir);
    let listen: Vec<String> = listen_raw.split(',')
        .map(|s| s.trim().to_lowercase()).filter(|s| !s.is_empty()).collect();
    let detect = model_ok && !listen.is_empty();
    // Nothing to do at all (no recording wanted AND can't detect) → don't open the mic.
    if !record_audio && !detect { return; }

    // A fresh detector means no sound is actually ongoing — clear any open audio event
    // left behind by a crash so it doesn't show as perpetually "in progress".
    close_orphan_audio_events(state, cam).await;

    let mut args: Vec<String> = vec!["-hide_banner".into(), "-loglevel".into(), "error".into()];
    args.extend(input_args);
    for a in ["-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "pipe:1"] { args.push(a.to_string()); }

    let child = crate::proc::tokio_cmd(ffmpeg)
        .args(&args)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .kill_on_drop(true)
        .spawn().ok();
    let mut child = match child { Some(c) => c, None => return };
    let stdout = match child.stdout.take() { Some(s) => s, None => return };
    state.audio_processes.lock().await.insert(cam, child);
    let st = Arc::clone(state);
    tokio::spawn(async move {
        use tokio::io::AsyncReadExt;
        let mut reader = tokio::io::BufReader::new(stdout);
        const WIN: usize = 16_000; // 1 s @ 16 kHz
        let mut bytes = vec![0u8; WIN * 2];
        loop {
            if reader.read_exact(&mut bytes).await.is_err() { break; }
            // Buffer this window for clip muxing (USB only) — record continuously,
            // including silence, so the exported clip has unbroken sound. `ts` is the
            // wall-clock END of the ~1 s window (read_exact blocks until it's full).
            if record_audio {
                let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs_f64()).unwrap_or(0.0);
                push_ring(cam, ts, bytes.clone());
            }
            // Close any sustained event that's gone quiet — runs every window, even
            // during silence (a silent mic still yields ~1 s of zero samples).
            reap_audio_events(&st).await;
            // DETECTION (only with a valid model + listen list). Recording above runs
            // regardless, so a broken/absent model never silences the clips.
            if !detect { continue; }
            let pcm: Vec<f32> = bytes.chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0).collect();
            let loud = crate::audio::rms(&pcm);
            if loud < MIN_VOLUME_RMS { continue; } // silence gate (mature NVRs min_volume parity)
            let data_dir = st.data_dir.clone();
            let results = tokio::task::spawn_blocking(move || {
                crate::audio::with_detector(&data_dir, |y| y.detect(&pcm))
            }).await.ok().flatten().unwrap_or_default();
            // VISIBILITY: log the top classification (throttled) even when it does
            // NOT match the listen list — without this, a healthy detector that hears
            // "Speech 0.6" while the user waits for events is indistinguishable from a
            // dead one. This is what makes "audio isn't triggering" diagnosable.
            if let Some((top_name, top_score)) = results.first() {
                let now = std::time::Instant::now();
                let mut last = LAST_AUDIO_TRACE.lock().unwrap_or_else(|e| e.into_inner());
                if last.map_or(true, |t| now.duration_since(t).as_secs() >= 10) {
                    *last = Some(now);
                    tracing::info!("audio cam{}: hearing '{}' ({:.2}) — listen list: {:?}",
                        cam, top_name, top_score, listen);
                }
            }
            // Keep YAMNet's top classes (results are score-ordered) for the
            // event's class chips + the semantic high-pitch flag.
            let top3: Vec<(String, f32)> = results.iter().take(3).cloned().collect();
            let hit = results.iter().find(|(name, score)| {
                *score >= threshold && {
                    let n = name.to_lowercase();
                    listen.iter().any(|w| n.contains(w))
                }
            }).cloned();
            if let Some((sound, score)) = hit {
                note_audio_hit(&st, cam, &sound, score, &top3, loud, threshold).await;
            }
        }
        // Capture ended (camera stopped / stream dropped): close any open event.
        close_audio_for_cam(&st, cam).await;
        tracing::info!("audio detector stopped for cam{}", cam);
    });
}

/// Analyse one mono 16 kHz PCM window streamed from the WebView (USB/webcam cameras,
/// whose audio isn't available to the server-side ffmpeg tap). Runs the same YAMNet +
/// sustained-event path as the server tap. No-ops unless audio detection is on + the
/// skill is installed.
#[tauri::command]
pub async fn analyze_audio_window(
    state: State<'_, Arc<AppState>>,
    cam_id: u8,
    pcm: Vec<f32>,
) -> Result<(), String> {
    let (enabled, listen, threshold) = {
        let s = state.settings.read().await;
        (s.audio_detection, s.audio_listen.clone(), s.audio_threshold)
    };
    if !enabled || !crate::audio::is_installed(&state.data_dir) { return Ok(()); }
    // Reap on every window so events close even when the WebView keeps streaming silence.
    reap_audio_events(state.inner()).await;
    let loud = crate::audio::rms(&pcm);
    if loud < MIN_VOLUME_RMS { return Ok(()); }

    let listen_v: Vec<String> = listen.split(',')
        .map(|s| s.trim().to_lowercase()).filter(|s| !s.is_empty()).collect();
    let data_dir = state.data_dir.clone();
    let results = tokio::task::spawn_blocking(move || {
        crate::audio::with_detector(&data_dir, |y| y.detect(&pcm))
    }).await.ok().flatten().unwrap_or_default();

    let top3: Vec<(String, f32)> = results.iter().take(3).cloned().collect();
    let hit = results.iter().find(|(name, score)| {
        *score >= threshold && {
            let n = name.to_lowercase();
            listen_v.iter().any(|w| n.contains(w))
        }
    }).cloned();
    if let Some((sound, score)) = hit {
        note_audio_hit(state.inner(), cam_id.min(15), &sound, score, &top3, loud, threshold).await;
    }
    Ok(())
}
