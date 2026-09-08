//! Event lifecycle timeline (mature NVRs `Timeline` table parity).
//!
//! Records the notable moments WITHIN a motion event — object appeared, entered
//! a zone, became stationary, recognised face, plate read, line crossed, sped,
//! audio detected, fall, gone — as rows in `event_timeline`. This gives a
//! queryable, human-readable "what happened, when" record per event, distinct
//! from the single end-of-event `ai_summary`. It is rendered as a Review detail
//! timeline, folded into the agent/VLM context, and embedded for search.
//!
//! Every write is best-effort: a failed insert NEVER disrupts the detection or
//! clip pipeline, mirroring the other `motion_events` writes.

use serde::Serialize;
use sqlx::SqlitePool;

use crate::state::AppState;

/// `class_type` vocabulary (kept as constants so call sites can't typo a value
/// the UI then fails to render). Mirrors mature NVRs' Timeline class types plus our
/// own analytics events.
pub mod class {
    pub const APPEARED: &str = "appeared";
    pub const ENTERED_ZONE: &str = "entered_zone";
    pub const RECOGNIZED: &str = "recognized";
    pub const LPR: &str = "lpr";
    pub const ATTRIBUTE: &str = "attribute";
    pub const SPEED: &str = "speed";
    pub const CROSSING: &str = "crossing";
    pub const AUDIO: &str = "audio";
    pub const GONE: &str = "gone";
}

/// A single lifecycle entry returned to the UI.
#[derive(Debug, Clone, Serialize)]
pub struct TimelineEntry {
    pub ts: String,
    pub class_type: String,
    pub label: Option<String>,
    pub value: Option<String>,
    pub score: Option<f32>,
}

/// Append a lifecycle entry stamped *now*. Fire-and-forget — failures are
/// swallowed so the caller (an inference frame, a clip analysis, an alert firer)
/// is never blocked.
pub async fn log(
    db: &SqlitePool,
    event_id: &str,
    cam_id: i64,
    class_type: &str,
    label: Option<&str>,
    value: Option<&str>,
    score: Option<f32>,
) {
    log_at(db, event_id, cam_id, &chrono::Utc::now().to_rfc3339(), class_type, label, value, score).await;
}

/// Append a lifecycle entry with an explicit timestamp — used for `appeared`
/// (first_object_at) and `gone` (ended_at) so the timeline's endpoints reflect
/// the real event boundaries rather than the analysis time.
#[allow(clippy::too_many_arguments)]
pub async fn log_at(
    db: &SqlitePool,
    event_id: &str,
    cam_id: i64,
    ts: &str,
    class_type: &str,
    label: Option<&str>,
    value: Option<&str>,
    score: Option<f32>,
) {
    let _ = sqlx::query(
        "INSERT INTO event_timeline(event_id, cam_id, ts, class_type, label, value, score) \
         VALUES(?,?,?,?,?,?,?)",
    )
    .bind(event_id)
    .bind(cam_id)
    .bind(ts)
    .bind(class_type)
    .bind(label)
    .bind(value)
    .bind(score)
    .execute(db)
    .await;
}

/// Fetch the ordered lifecycle of an event for the Review detail panel.
#[tauri::command]
pub async fn get_event_timeline(
    state: tauri::State<'_, AppState>,
    event_id: String,
) -> Result<Vec<TimelineEntry>, String> {
    let rows: Vec<(String, String, Option<String>, Option<String>, Option<f32>)> = sqlx::query_as(
        "SELECT ts, class_type, label, value, score FROM event_timeline \
         WHERE event_id=? ORDER BY id ASC",
    )
    .bind(&event_id)
    .fetch_all(&state.db)
    .await
    .map_err(|e| e.to_string())?;

    Ok(rows
        .into_iter()
        .map(|(ts, class_type, label, value, score)| TimelineEntry {
            ts,
            class_type,
            label,
            value,
            score,
        })
        .collect())
}
