//! Event bookmarks (saved/favorites pattern).
//!
//! One row per bookmarked `motion_event`. Powers the bookmark toggle on every
//! Review card + the "Bookmarks" tab. `list_bookmarked_events` (which returns
//! the full `MotionEvent` rows for the tab) lives in `nvr_recording.rs` next to
//! the shared `MotionEvent` mapping.

use std::sync::Arc;

use tauri::State;

use crate::AppState;

/// Bookmark an event. cam_id is derived from the event row so the saved list can
/// filter by camera. Idempotent (INSERT OR IGNORE keyed on event_id).
#[tauri::command]
pub async fn add_bookmark(
    state: State<'_, Arc<AppState>>,
    event_id: String,
) -> Result<(), String> {
    sqlx::query(
        "INSERT OR IGNORE INTO event_bookmarks(event_id, cam_id, created_at)
         SELECT id, cam_id, datetime('now') FROM motion_events WHERE id=?"
    )
    .bind(&event_id)
    .execute(&state.db).await.map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
pub async fn remove_bookmark(
    state: State<'_, Arc<AppState>>,
    event_id: String,
) -> Result<(), String> {
    sqlx::query("DELETE FROM event_bookmarks WHERE event_id=?")
        .bind(&event_id).execute(&state.db).await.map_err(|e| e.to_string())?;
    Ok(())
}

/// The set of bookmarked event ids — drives the active/inactive state of the
/// card toggle and the "Bookmarks" tab count. Lightweight (ids only).
#[tauri::command]
pub async fn list_bookmark_ids(
    state: State<'_, Arc<AppState>>,
) -> Result<Vec<String>, String> {
    let rows: Vec<(String,)> = sqlx::query_as("SELECT event_id FROM event_bookmarks")
        .fetch_all(&state.db).await.map_err(|e| e.to_string())?;
    Ok(rows.into_iter().map(|(id,)| id).collect())
}
