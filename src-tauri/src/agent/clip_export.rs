//! v14: server-side event-clip exporter — the single source of the clip the
//! agent SENDS (Telegram, share links, mobile MJPEG).
//!
//! Replaces the legacy browser MediaRecorder (`save_clip_blob`) that recorded
//! the live view from motion-onset until 15 s after motion stopped — unbounded,
//! so a busy scene produced a 30-minute `.webm` that choked the WebView2
//! renderer and arrived in Telegram as an un-previewable document (VP9 ≠ H.264).
//!
//! Instead we slice the continuous **NVR** recording to the event window (the
//! exact bounds the in-app player uses, via [`crate::footage::event_clip_window`]),
//! cap the length, and remux to a real H.264 MP4 with the moov atom up front
//! (`-movflags +faststart`) — so `sendVideo` renders an inline-playable preview.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, OnceLock};
use crate::AppState;

/// Per-event generation gate. Concurrent callers for the SAME event compute the
/// identical output path `clip_{id}.mp4`; without serialization they run ffmpeg on
/// that one file simultaneously and interleave it into a CORRUPT, unplayable clip
/// (the observed event-clip corruption). Only the first caller generates; the rest
/// await the same lock and then reuse the freshly-cached result.
static INFLIGHT: OnceLock<Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>> = OnceLock::new();

fn inflight_gate(event_id: &str) -> Arc<tokio::sync::Mutex<()>> {
    let map = INFLIGHT.get_or_init(Default::default);
    let mut g = map.lock().unwrap();
    g.entry(event_id.to_string()).or_default().clone()
}

/// Clip paths whose audio state is already settled this boot — either the cache
/// carries audio, or its window is genuinely silent (nothing to regenerate).
/// Without this, the silent-cache probe (an ffmpeg spawn) ran on EVERY HTTP
/// range request the <video> element makes, which visibly lagged playback.
/// Segments are immutable once written, so a settled verdict never changes.
static CLIP_AUDIO_SETTLED: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();

fn audio_settled(p: &str) -> bool {
    CLIP_AUDIO_SETTLED.get_or_init(Default::default).lock().unwrap().contains(p)
}
fn mark_audio_settled(p: &str) {
    CLIP_AUDIO_SETTLED.get_or_init(Default::default).lock().unwrap().insert(p.to_string());
}

/// Clip paths whose STRUCTURAL integrity is confirmed this boot. Clips corrupted by
/// the old concurrent-write race decode with H.264 NAL errors and won't play; we
/// verify a cached clip ONCE and remember the verdict, because re-decoding on every
/// range request would lag playback (same reasoning as [`CLIP_AUDIO_SETTLED`]).
static CLIP_VERIFIED: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();

fn clip_verified(p: &str) -> bool {
    CLIP_VERIFIED.get_or_init(Default::default).lock().unwrap().contains(p)
}
fn mark_clip_verified(p: &str) {
    CLIP_VERIFIED.get_or_init(Default::default).lock().unwrap().insert(p.to_string());
}

/// Decode the first few seconds; a clip mangled by concurrent writers emits H.264
/// bitstream errors ("Invalid NAL unit size", "Error splitting the input into NAL
/// units"). Bounded to 3 s so the once-per-boot check stays cheap. Conservative: on
/// any failure to even run ffmpeg we report "not corrupt" so a good clip is never
/// nuked on our own tooling failure.
async fn clip_looks_corrupt(ffmpeg: &std::path::Path, clip: &std::path::Path) -> bool {
    let out = crate::proc::tokio_cmd(ffmpeg)
        .args([
            "-hide_banner", "-v", "error",
            "-t", "3",
            "-i", &clip.to_string_lossy(),
            "-f", "null", "-",
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        .output().await;
    match out {
        // A clean H.264 clip decodes with EMPTY stderr at `-v error`. Any decode-level
        // message (NAL/reference/concealing errors) or a non-zero exit means corrupt.
        Ok(o) => !o.status.success() || !o.stderr.is_empty(),
        Err(_) => false,
    }
}

/// True if a cached clip is safe to serve. Structural health is proven ONCE per boot
/// per file (`CLIP_VERIFIED`); a corrupt clip returns false so the caller regenerates.
async fn clip_is_healthy(state: &Arc<AppState>, p: &str) -> bool {
    if clip_verified(p) { return true; }
    let Ok(ff) = crate::ensure_ffmpeg(&state.data_dir).await else {
        return true; // can't probe → don't discard a possibly-good clip
    };
    if clip_looks_corrupt(&ff, std::path::Path::new(p)).await {
        tracing::warn!("cached clip is corrupt (decode errors) — regenerating: {p}");
        false
    } else {
        mark_clip_verified(p);
        true
    }
}

/// Hard ceiling on a SENT clip's length (seconds). The in-app player streams the
/// full event; only the exported file we upload/share is capped, so a pathological
/// never-closing event can never again produce a 30-minute upload.
pub(crate) const MAX_SENT_CLIP_SECS: f64 = 120.0;

/// Minimum size (bytes) for a clip file to count as real. Anything at/below this is a
/// failed/empty generation (footage not recorded yet) — never cache or serve it.
pub(crate) const MIN_CLIP_BYTES: u64 = 4096;

/// Delete an event's exported clip file(s). Holds the per-event generation gate,
/// so a delete can never race an in-flight export that would re-publish the file
/// microseconds later. Removes BOTH the path the row pointed at and the
/// deterministic `clip_{id}.mp4` — an OPEN event's clip is generated but
/// deliberately never cached in `clip_path`, so that copy used to survive the
/// event that owned it and sit in the data dir until the next manual purge.
///
/// Files outside the data dir are left alone (defence against a poisoned
/// `clip_path`); a locked file (Windows: player still holding it) is left for
/// `purge_orphaned_clips`, which sweeps anything the DB no longer references.
pub(crate) async fn delete_event_clip(state: &Arc<AppState>, event_id: &str, clip_path: Option<String>) {
    let gate = inflight_gate(event_id);
    let _permit = gate.lock().await;
    if let Some(p) = clip_path.filter(|p| !p.is_empty()) {
        let path = std::path::PathBuf::from(&p);
        if path.starts_with(&state.data_dir) {
            if let Err(e) = tokio::fs::remove_file(&path).await {
                if path.exists() { tracing::warn!("clip {p} could not be deleted ({e}) — left for the orphan sweep"); }
            }
        }
    }
    let _ = tokio::fs::remove_file(state.data_dir.join(format!("clip_{event_id}.mp4"))).await;
    // Drop the per-event gate entry — the event is gone, nothing will ask again.
    if let Some(map) = INFLIGHT.get() {
        map.lock().unwrap_or_else(|p| p.into_inner()).remove(event_id);
    }
}

/// Returns the path to the event's bounded MP4 clip, generating it from the NVR
/// recording if it isn't already on disk. Idempotent and cached in
/// `motion_events.clip_path`. Returns `None` when the event is unknown or has no
/// NVR footage in range (callers fall back to a snapshot).
pub(crate) async fn ensure_event_clip(state: &Arc<AppState>, event_id: &str) -> Option<String> {
    // Cached? Reuse only a NON-EMPTY file on disk. A 0-byte/tiny file means a prior
    // generation ran before the footage existed (eager close-time gen) and wrote junk;
    // reusing it by path alone is exactly what served "no footage". Require a real size.
    let existing: Option<String> = sqlx::query_scalar(
        "SELECT clip_path FROM motion_events WHERE id=?"
    ).bind(event_id).fetch_optional(&state.db).await.ok().flatten().flatten();
    if let Some(p) = existing {
        if !p.is_empty() {
            let big_enough = tokio::fs::metadata(&p).await.map(|m| m.len() > MIN_CLIP_BYTES).unwrap_or(false);
            // A big-enough file can still be CORRUPT (old concurrent-write race). Prove
            // it decodes before serving; if not, fall through to the delete+regenerate
            // path below. Verified once per boot, so playback isn't slowed.
            if big_enough && clip_is_healthy(state, &p).await {
                // Fast path: verdict already settled this boot — serve immediately.
                // The probe below spawns ffmpeg, and this function runs on every
                // range request during playback; probing each one lags the video.
                if audio_settled(&p) { return Some(p); }
                // Recordings gained audio (2026-07): a clip cached from the silent
                // era plays mute forever even though its window is audible now.
                // Regenerate ONCE in that case; audible or genuinely-silent-window
                // caches are served as-is.
                let ff = crate::ensure_ffmpeg(&state.data_dir).await.ok();
                let silent_cache = match &ff {
                    Some(ff) => !crate::audio_cmds::clip_has_audio(ff, std::path::Path::new(&p)).await,
                    None => false,
                };
                if silent_cache {
                    let (pre0, post0) = {
                        let g = state.settings.read().await;
                        (g.record_pre_buffer_secs as i64, g.record_post_buffer_secs as i64)
                    };
                    if let Some((cam0, s0, e0, _open)) =
                        crate::footage::event_clip_window(&state.db, pre0, post0, event_id).await
                    {
                        if crate::nvr_stream::window_all_audio(&state.db, cam0, s0, Some(e0)).await {
                            tracing::info!("clip {}: silent cache but window is audible — regenerating with audio", event_id);
                            let _ = tokio::fs::remove_file(&p).await;
                            let _ = sqlx::query("UPDATE motion_events SET clip_path=NULL WHERE id=?")
                                .bind(event_id).execute(&state.db).await;
                        } else {
                            // Window is genuinely silent — the cache is as good as it
                            // gets. Segments are immutable, so remember that forever.
                            mark_audio_settled(&p);
                            return Some(p);
                        }
                    } else {
                        mark_audio_settled(&p);
                        return Some(p);
                    }
                } else {
                    // Cache carries audio — final. Never probe this path again.
                    mark_audio_settled(&p);
                    return Some(p);
                }
            }
            // Stale/empty cache — drop the junk file + the pointer, then regenerate.
            let _ = tokio::fs::remove_file(&p).await;
            let _ = sqlx::query("UPDATE motion_events SET clip_path=NULL WHERE id=?")
                .bind(event_id).execute(&state.db).await;
        }
    }

    // Cache miss: SERIALIZE generation for this event. Close-time prewarm, alert
    // clip-attach and playback/Telegram requests all fire near-simultaneously and
    // target the same `clip_{id}.mp4`; concurrent ffmpeg writers corrupt it. Hold the
    // per-event gate across generate + mux so exactly one runs.
    let gate = inflight_gate(event_id);
    let _permit = gate.lock().await;
    // Re-check: whoever held the gate before us may have JUST produced the clip.
    let just_made: Option<String> = sqlx::query_scalar(
        "SELECT clip_path FROM motion_events WHERE id=?"
    ).bind(event_id).fetch_optional(&state.db).await.ok().flatten().flatten();
    if let Some(p) = just_made {
        if !p.is_empty()
            && tokio::fs::metadata(&p).await.map(|m| m.len() > MIN_CLIP_BYTES).unwrap_or(false)
        {
            return Some(p);
        }
    }

    // Resolve the event window (anchored + padded), then cap the SENT length.
    let (pre, post) = {
        let g = state.settings.read().await;
        (g.record_pre_buffer_secs as i64, g.record_post_buffer_secs as i64)
    };
    let (cam_id, start_secs, end_raw, is_open) =
        crate::footage::event_clip_window(&state.db, pre, post, event_id).await?;
    let end_secs = end_raw.min(start_secs + MAX_SENT_CLIP_SECS);

    // Export to a deterministic per-event file (removed when the event is deleted,
    // see agent_data_cmds — it deletes the file referenced by clip_path).
    let out = state.data_dir.join(format!("clip_{}.mp4", event_id));
    let ok = crate::nvr_stream::concat_window_to_file(
        &state.db, &state.data_dir, cam_id, start_secs, end_secs, &out,
    ).await;
    // Only accept + cache a REAL clip. A failed/empty generation (e.g. footage for the
    // post-buffer not yet recorded) produces a 0-byte file — never persist it as
    // clip_path, or it'd be served forever as "no footage". A later on-demand open
    // regenerates once the footage exists.
    let size = tokio::fs::metadata(&out).await.map(|m| m.len()).unwrap_or(0);
    if !ok || size <= MIN_CLIP_BYTES {
        let _ = tokio::fs::remove_file(&out).await;
        return None;
    }

    // Best-effort: for USB/native cams (silent JPEG-pipe recording), mux the buffered
    // mic audio for this window into the clip so audio events are AUDIBLE. No-ops for
    // RTSP cams (their clip already carries the camera's audio) and never fails the clip.
    crate::audio_cmds::mux_ring_audio(&state.data_dir, cam_id, start_secs, end_secs, &out).await;

    let out_str = out.to_string_lossy().to_string();
    // Freshly written by the single-flight, atomic-rename exporter to a unique temp
    // then published — it is structurally sound by construction. Trust it so the
    // once-per-boot integrity probe never re-decodes our own clean output.
    mark_clip_verified(&out_str);
    // Only CACHE (persist clip_path) for a CLOSED event — its footage is final. An OPEN
    // (in-progress) event's clip is still growing; caching it would freeze the partial
    // and serve it forever as the "final" clip. We still return the freshly-generated
    // file (so Telegram/share of an in-progress event works), just don't persist it —
    // the close-time pre-warm regenerates + caches the complete clip once the event ends.
    if !is_open {
        sqlx::query("UPDATE motion_events SET clip_path=? WHERE id=?")
            .bind(&out_str).bind(event_id)
            .execute(&state.db).await.ok();
        // Freshly generated with the audio-aware exporter — its audio state is
        // final by construction; don't waste a probe on the first playback.
        mark_audio_settled(&out_str);
    }
    Some(out_str)
}
