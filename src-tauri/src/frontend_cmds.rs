//! Tauri commands called by the JS frontend for scene-object reporting, clip blob/frame round-trips, and Guardian chat / Ollama model pulls.

use std::sync::Arc;

use tauri::State;

use crate::{AppState, SceneObject};


/// Called by the frontend every ~5 s with the AI model's current detections.
/// Rust stores these so the Guardian agent can track object presence/absence.
#[tauri::command]
pub async fn update_scene_objects(
    state: State<'_, Arc<AppState>>,
    cam_id: u8,
    objects: Vec<SceneObject>,
) -> Result<(), String> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    state.scene_objects.write().await.insert(cam_id, objects);
    state.scene_last_update.write().await.insert(cam_id, now);
    Ok(())
}

// `stream_chat_with_agent` was DELETED. It was a THIRD tag parser with its own
// five-line system prompt and its own hand-rolled [REMEMBER:] loop, it never
// called `parse_tags`, and it had no callers. Streaming now comes from
// `retrieve::answer` via the same `guardian:chat-token` event.

