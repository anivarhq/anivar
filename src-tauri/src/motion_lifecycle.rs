//! Unified motion-event lifecycle (v9).
//!
//! Pre-v9 there were two state machines: one in [`crate::capture`] (native USB
//! / nokhwa cameras) and one in [`crate::inference_cmds::process_frame_inner`]
//! (browser / MJPEG / mobile-pushed frames). v8 only patched the former; the
//! latter still had the v7 buggy "any single noisy frame resets the close
//! timer" lifecycle, hardcoded 15-second post-buffer, etc. Browser-camera
//! users saw the v7 "event stuck on" bug as if v8 had never shipped.
//!
//! This module is the single source of truth for opening / sustaining /
//! closing motion events. Both call sites collapse into `tick_motion_event`.
//!
//! The lifecycle is mature NVRs-shaped when YOLO is active:
//!   * Motion alone never opens an event when YOLO is running — YOLO must
//!     confirm a tracked class within the post-buffer window. (edge-AI NVRs
//!     and mature NVRs behave this way by construction.)
//!   * An event stays alive as long as YOLO keeps seeing a tracked class,
//!     even when raw motion stops (parked car, sitting person).
//!   * The event closes when EITHER tracked-object presence has been gone
//!     for `detect_max_disappeared_frames` AND stillness has held, OR YOLO
//!     is off / not installed and stillness alone holds.
//!
//! When YOLO is off, we fall back to v8 hysteresis-only behaviour.

use std::sync::Arc;
use std::time::Instant;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as B64;
use chrono::Utc;
use uuid::Uuid;

use crate::AppState;

/// Result of one lifecycle tick. The caller wires up clip recording /
/// agent-clip-analysis based on these signals — the lifecycle module itself
/// never spawns those side-effects so it stays cheap to call per-frame.
pub struct EventLifecycleResult {
    /// `Some(event_id)` while an event is open. `None` between events.
    pub event_id:        Option<String>,
    /// Hysteresis-confirmed motion signal (NOT raw single-frame motion).
    /// Drives the UI MOTION chip and `setMotionDetected` on the frontend.
    pub motion_detected: bool,
    /// `Some(event_id)` on the tick where an event just closed (or was
    /// dropped because YOLO never confirmed). Lets the caller flush the
    /// detection buffer / stop recording / dispatch the final clip-analysis.
    pub just_closed:     Option<String>,
    /// `true` if the event close was a "drop" because YOLO was active but
    /// never confirmed a tracked class. Caller can skip clip-analysis since
    /// the row has been removed from the DB.
    pub just_dropped:    bool,
    /// `Some(event_id)` on the tick where an event just OPENED. The caller
    /// uses this to fire its kind-specific notification (ntfy push for the
    /// browser path). The lifecycle module itself
    /// never side-effects beyond DB writes — keeps it cheap to call.
    pub just_opened:     Option<String>,
}

/// Snapshot of all settings the lifecycle reads — kept as a struct so callers
/// (browser/native) don't each have to know the field list.
pub struct LifecycleSettings {
    pub sensitivity:                 f32,
    pub motion_min_frames:           u32,
    pub motion_open_score_mult:      f32,
    pub record_post_buffer_secs:     f32,
    /// Hard cap on a single event's wall-clock length. `0` = unlimited. Force-closes
    /// runaway events (persistent noise, stuck detections) so nothing records for
    /// minutes with nobody acting.
    pub record_max_event_secs:       u32,
    pub require_object_to_open_event: bool,
    pub detect_max_disappeared_frames: u32,
    /// Whether YOLO is currently active for this host. When true, raw motion
    /// alone can't open an event AND object-absence is required for close.
    /// When false we fall back to v8 hysteresis-only behaviour.
    pub yolo_active: bool,
}

impl LifecycleSettings {
    pub async fn snapshot(state: &Arc<AppState>) -> Self {
        let s = state.settings.read().await;
        // YOLO is "active" if the inference status RwLock says "ready". The
        // status struct is populated by `run_inference_loop` on every state
        // change. Falling back to !yolo_variant.is_empty() would over-report
        // (the variant is set even before the model loads).
        let yolo_active = {
            let inf = state.inference_status.read().await;
            inf.state == "ready"
        };
        Self {
            sensitivity:                 s.sensitivity,
            motion_min_frames:           s.motion_min_frames.max(1),
            motion_open_score_mult:      s.motion_open_score_mult.max(1.0),
            record_post_buffer_secs:     s.record_post_buffer_secs as f32,
            record_max_event_secs:       s.record_max_event_secs,
            require_object_to_open_event: s.require_object_to_open_event,
            detect_max_disappeared_frames: s.detect_max_disappeared_frames.max(1),
            yolo_active,
        }
    }
}

/// One lifecycle tick. Call once per frame from both the native capture loop
/// and the per-frame command. `motion_score` is the raw frame-diff score
/// (the caller computes it via `motion::compute_motion_masked`). `jpeg_bytes`
/// is used only on the open path to seed the event's thumbnail.
///
/// `approx_fps` is the caller's approximate frame rate. It's used to convert
/// `record_post_buffer_secs` into a stillness-streak target. Pass `20` for
/// the native loop (50ms budget) and `15` for the browser path (typical).
pub async fn tick_motion_event(
    state: &Arc<AppState>,
    cam_id: u8,
    motion_score: f32,
    jpeg_bytes: &[u8],
    settings: &LifecycleSettings,
    approx_fps: f32,
) -> EventLifecycleResult {
    // Stillness-streak target. Floor at 1 to avoid divide-by-zero / instant
    // close on a misconfigured host.
    let still_frames_to_close = ((settings.record_post_buffer_secs * approx_fps).round() as i64).max(1) as u32;

    let mut cs_map = state.cam_states.lock().await;
    let cs = cs_map.entry(cam_id).or_default();

    // ── Temporal hysteresis ─────────────────────────────────────────────
    // Symmetric this time (v9 fix): the stillness-streak is reset on the
    // hysteresis-confirmed `sustained` signal, not on raw single-frame
    // motion. A single noisy frame in the close window can no longer
    // restart the close timer — that was v8's residual bug.
    let raw_motion = motion_score >= settings.sensitivity;
    if raw_motion {
        cs.motion_streak = cs.motion_streak.saturating_add(1);
    } else {
        cs.motion_streak = 0;
    }
    let sustained = cs.motion_streak >= settings.motion_min_frames;
    if sustained {
        cs.stillness_streak = 0;
    } else {
        cs.stillness_streak = cs.stillness_streak.saturating_add(1);
    }
    // OPEN requires the higher bar (sensitivity * mult). Once open, regular
    // sensitivity sustains the streak.
    let open_now = sustained && motion_score >= settings.sensitivity * settings.motion_open_score_mult;

    let event_id: Option<String>;
    let mut just_closed: Option<String> = None;
    let mut just_opened: Option<String> = None;
    let mut just_dropped: bool = false;

    // Hard max-duration cap (monotonic clock — immune to wall-clock skew). When an
    // open event exceeds the cap we DON'T let it keep sustaining; it falls through to
    // the close path and force-closes, then the next moving frame reopens a fresh
    // event. Continuous activity thus splits into ≤cap chunks that the review-segment
    // GAP-merge rejoins into one item. `0` = unlimited.
    let over_max = settings.record_max_event_secs > 0
        && cs.event_opened_at
            .map(|t| t.elapsed().as_secs() >= settings.record_max_event_secs as u64)
            .unwrap_or(false);

    // ── Open / sustain ─────────────────────────────────────────────────
    let should_open  = open_now  && cs.motion_active.is_none();
    let should_sustain = sustained && cs.motion_active.is_some() && !over_max;
    if should_open || should_sustain {
        cs.last_motion_at = Some(Instant::now());
        if motion_score > cs.motion_peak { cs.motion_peak = motion_score; }

        if cs.motion_active.is_none() {
            let id  = Uuid::new_v4().to_string();
            let now = Utc::now().to_rfc3339();
            let thumb = B64.encode(jpeg_bytes);
            let id_for_insert = id.clone();
            let ms = motion_score;
            let st = state.clone();
            tokio::spawn(async move {
                // Depth-anonymized cams: the stored thumbnail must be the DEPTH
                // frame (latest_frames holds depth in that mode), never raw.
                let thumb = if crate::depth::is_anonymized(cam_id) {
                    st.latest_frames.read().await.get(&cam_id)
                        .map(|f| B64.encode(f))
                        .unwrap_or_default()
                } else { thumb };
                // Bind cam_id — without it every motion event defaulted to camera 0
                // and vanished from every other camera's timeline / Review feed.
                sqlx::query("INSERT INTO motion_events(id,started_at,peak_score,thumbnail,clip_path,cam_id) VALUES(?,?,?,?,NULL,?)")
                    .bind(&id_for_insert).bind(&now).bind(ms).bind(&thumb).bind(cam_id as i64)
                    .execute(&st.db).await.ok();
            });

            cs.motion_active = Some(id.clone());
            cs.detection_buffer.clear();
            cs.event_opened_at = Some(Instant::now());
            cs.event_object_confirmed = false;
            cs.last_object_seen_at = None;
            cs.object_absence_frames = 0;
            cs.last_analysis_at = None;
            cs.last_burst_at = None;
            cs.classes_seen.clear();
            cs.last_dominant = None;
            cs.crowd_alerted_count = 0;
            just_opened = Some(id.clone());
            event_id = Some(id);
        } else {
            let id  = cs.motion_active.clone().unwrap();
            let pk  = cs.motion_peak;
            let db  = state.db.clone();
            let id_for_update = id.clone();
            tokio::spawn(async move {
                sqlx::query("UPDATE motion_events SET peak_score=? WHERE id=?")
                    .bind(pk).bind(&id_for_update).execute(&db).await.ok();
            });
            event_id = Some(id);
        }
    } else {
        // ── Close path ─────────────────────────────────────────────────
        // FRIGATE MODEL: when YOLO is active, require BOTH object-absence
        // AND stillness. A parked car / sitting person keeps the event
        // alive because YOLO keeps writing `last_object_seen_at` in the
        // inference loop, holding `object_absence_frames` at 0.
        //
        // When YOLO is off, fall back to v8 hysteresis-only behaviour:
        // stillness alone closes.
        // `over_max` (computed above) force-closes through this SAME path — so the
        // clip + review segment + state reset all fire — guaranteeing no runaway.
        let close_signal = over_max || if settings.yolo_active {
            cs.object_absence_frames >= settings.detect_max_disappeared_frames
                && cs.stillness_streak >= still_frames_to_close
        } else {
            cs.stillness_streak >= still_frames_to_close
        };

        if close_signal {
            if let Some(id) = cs.motion_active.take() {
                // Only DROP an unconfirmed event if the user EXPLICITLY opted into
                // `require_object_to_open_event`. Previously a YOLO-active event
                // with no confirmed class was always deleted — which silently lost
                // most real motion when YOLO was slow/missed/mid-load. Default now:
                // KEEP it as a plain motion event (category stays "other").
                let drop_unconfirmed =
                    settings.require_object_to_open_event && !cs.event_object_confirmed;

                let det_json = if !cs.detection_buffer.is_empty() {
                    let j = serde_json::to_string(&cs.detection_buffer).unwrap_or_default();
                    cs.detection_buffer.clear();
                    Some(j)
                } else { None };

                let id_for_spawn = id.clone();
                let db = state.db.clone();
                if drop_unconfirmed {
                    tracing::info!(
                        "Motion event {} dropped — YOLO never confirmed a tracked class.",
                        &id_for_spawn[..8.min(id_for_spawn.len())]
                    );
                    let id_for_delete = id_for_spawn.clone();
                    tokio::spawn(async move {
                        let _ = sqlx::query("DELETE FROM motion_events WHERE id=?")
                            .bind(&id_for_delete).execute(&db).await;
                    });
                    just_dropped = true;
                } else {
                    let now = Utc::now().to_rfc3339();
                    // v26: belt-and-suspenders. If a row's wall-clock duration
                    // turns out to be < 1 s for any reason (timing race, slow
                    // SQLite, system clock jump), DELETE it instead of UPDATE.
                    // The motion_min_frames debounce should already prevent
                    // this, but if it ever slips through we don't want junk
                    // empty rows showing up in the events strip.
                    let state_for_clip = state.clone();
                    let post_buffer = settings.record_post_buffer_secs;
                    tokio::spawn(async move {
                        let row: Option<(f64,)> = sqlx::query_as(
                            "SELECT (julianday(?) - julianday(started_at)) * 86400 \
                             FROM motion_events WHERE id=?",
                        )
                        .bind(&now).bind(&id_for_spawn)
                        .fetch_optional(&db).await.unwrap_or(None);
                        let dur = row.map(|r| r.0).unwrap_or(0.0);
                        if dur < 0.4 {
                            let _ = sqlx::query("DELETE FROM motion_events WHERE id=?")
                                .bind(&id_for_spawn).execute(&db).await;
                            tracing::info!(
                                "Motion event {} dropped — duration {:.2}s < 0.4s threshold.",
                                &id_for_spawn[..8.min(id_for_spawn.len())], dur,
                            );
                            return;
                        }
                        sqlx::query("UPDATE motion_events SET ended_at=?, duration_secs=? WHERE id=?")
                            .bind(&now).bind(dur).bind(&id_for_spawn)
                            .execute(&db).await.ok();
                        if let Some(dj) = det_json {
                            sqlx::query("UPDATE motion_events SET detections=? WHERE id=?")
                                .bind(dj).bind(&id_for_spawn).execute(&db).await.ok();
                        }
                        // Group this closed event into its server-side review segment
                        // (mature NVRs review-item parity) — the canonical grouping the Review
                        // feed + NVR timeline read. Idempotent; re-runs after AI analysis
                        // to refine severity/labels once classification lands. Footage-
                        // independent, so do it immediately.
                        crate::review_segments::upsert_review_segment(&state_for_clip.db, &id_for_spawn).await;
                        // Pre-warm the bounded H.264 clip — but ONLY AFTER the post-buffer
                        // footage (ended_at + post) has been recorded AND the covering
                        // segment finalized/indexed. The old eager gen ran here at close,
                        // BEFORE that footage existed → ffmpeg wrote a 0-byte clip that got
                        // cached + served forever as "no footage". On-demand opens regenerate
                        // regardless, so this is just a best-effort warm-up for the play badge.
                        let st = state_for_clip.clone();
                        let eid = id_for_spawn.clone();
                        let delay = (post_buffer as u64).saturating_add(12);
                        tokio::spawn(async move {
                            tokio::time::sleep(std::time::Duration::from_secs(delay)).await;
                            crate::agent::clip_export::ensure_event_clip(&st, &eid).await;
                        });
                    });
                }
                just_closed = Some(id);

                cs.motion_peak = 0.0;
                cs.last_motion_at = None;
                cs.event_opened_at = None;
                cs.event_object_confirmed = false;
                cs.last_object_seen_at = None;
                cs.object_absence_frames = 0;
                cs.last_analysis_at = None;
                cs.last_burst_at = None;
                cs.classes_seen.clear();
                cs.crowd_alerted_count = 0;
            }
            event_id = None;
        } else {
            event_id = cs.motion_active.clone();
        }
    }

    EventLifecycleResult {
        event_id,
        motion_detected: sustained,
        just_closed,
        just_dropped,
        just_opened,
    }
}
