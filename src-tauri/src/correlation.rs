//! Face sightings — "person X seen on camera Y at T". Where people move and what
//! they did now lives in person tracks and visits (person_track.rs,
//! people_search.rs); the name-keyed correlation and duration-guess anomaly
//! commands that used to live here had no remaining callers.

use std::sync::Arc;

use tauri::State;

use crate::AppState;


/// Insert one face sighting. Shared by the live `record_face_sighting` command
/// AND the server-side event recognition path (`agent::clip`), so cross-camera
/// context is populated even when NO camera window is open — previously only the
/// frontend recorded sightings, leaving the headless path's cross-camera block empty.
pub async fn insert_face_sighting(
    db: &sqlx::SqlitePool,
    person_name: &str,
    person_id: Option<&str>,
    camera_id: i64,
    event_id: Option<&str>,
    confidence: f32,
    method: &str,
) -> anyhow::Result<()> {
    let id = uuid::Uuid::new_v4().to_string();
    let now = chrono::Utc::now().to_rfc3339();
    sqlx::query(
        "INSERT INTO face_sightings(id, person_name, person_id, camera_id, event_id, seen_at, confidence, method)
         VALUES(?,?,?,?,?,?,?,?)"
    )
    .bind(&id).bind(person_name).bind(person_id).bind(camera_id)
    .bind(event_id).bind(&now).bind(confidence).bind(method)
    .execute(db).await?;
    Ok(())
}

#[tauri::command]
pub async fn record_face_sighting(
    state: State<'_, Arc<AppState>>,
    person_name: String,
    camera_id: i64,
    event_id: Option<String>,
    confidence: f32,
) -> Result<(), String> {
    // Frontend-originated sighting: resolve the id from the name best-effort
    // (the command's callers predate the id contract).
    let pid: Option<String> = sqlx::query_scalar("SELECT id FROM known_persons WHERE name=? LIMIT 1")
        .bind(&person_name).fetch_optional(&state.db).await.ok().flatten();
    insert_face_sighting(&state.db, &person_name, pid.as_deref(), camera_id, event_id.as_deref(), confidence, "manual")
        .await.map_err(|e| e.to_string())
}

