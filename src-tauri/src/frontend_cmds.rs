//! Tauri commands called by the JS frontend for scene-object reporting, clip blob/frame round-trips, and Guardian chat / Ollama model pulls.

use std::sync::Arc;

use base64::{engine::general_purpose::STANDARD as B64, Engine};
use tauri::{Emitter, State};

use crate::{AppState, SceneObject, agent};


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

/// Reads a saved clip file and returns its frames as base64-encoded JPEGs.
/// Returns every 2nd frame (~5 fps) to keep the IPC payload manageable.
/// Returns empty list if the clip is still being recorded or file is missing.
#[tauri::command]
pub async fn read_clip_frames(
    state: State<'_, Arc<AppState>>,
    event_id: String,
) -> Result<Vec<String>, String> {
    let row: Option<(Option<String>,)> =
        sqlx::query_as("SELECT clip_path FROM motion_events WHERE id=?")
            .bind(&event_id)
            .fetch_optional(&state.db)
            .await
            .map_err(|e| e.to_string())?;

    let path = match row.and_then(|(p,)| p) {
        Some(p) if !p.is_empty() => p,
        _ => return Ok(vec![]),
    };

    let data = match tokio::fs::read(&path).await {
        Ok(d) => d,
        Err(_) => return Ok(vec![]),
    };

    // Parse: each frame is [4-byte big-endian length][JPEG bytes]
    let mut frames: Vec<String> = Vec::new();
    let mut cursor = 0usize;
    let mut i = 0usize;
    while cursor + 4 <= data.len() {
        let len = u32::from_be_bytes([
            data[cursor], data[cursor + 1], data[cursor + 2], data[cursor + 3],
        ]) as usize;
        cursor += 4;
        if cursor + len > data.len() { break; }
        if i % 2 == 0 { // every 2nd frame → ~5 fps from a 10-fps recording
            frames.push(B64.encode(&data[cursor..cursor + len]));
        }
        cursor += len;
        i += 1;
        if frames.len() >= 600 { break; } // cap at 600 frames (~2 min)
    }
    Ok(frames)
}

/// Receives a completed WebM/MP4 clip blob from the browser MediaRecorder,
/// saves it to disk, updates the motion_events DB row, and triggers LLM analysis.
#[tauri::command]
pub async fn save_clip_blob(
    state: State<'_, Arc<AppState>>,
    event_id: String,
    blob_b64: String,
    mime_type: String,
) -> Result<(), String> {
    let bytes = B64.decode(blob_b64.trim()).map_err(|e| format!("base64 decode: {e}"))?;
    let ext = if mime_type.contains("mp4") { "mp4" } else { "webm" };
    let path = state.data_dir.join(format!("{}.{}", event_id, ext));
    tokio::fs::write(&path, &bytes).await.map_err(|e| format!("write clip: {e}"))?;
    let path_str = path.to_string_lossy().to_string();
    sqlx::query("UPDATE motion_events SET clip_path=? WHERE id=?")
        .bind(&path_str).bind(&event_id)
        .execute(&state.db).await.map_err(|e| e.to_string())?;
    tracing::info!("Saved {} clip: {} ({} KB)", ext, &event_id[..8.min(event_id.len())], bytes.len() / 1024);
    // Trigger LLM analysis in background
    let s2  = Arc::clone(&state);
    let id2 = event_id.clone();
    tokio::spawn(async move { agent::analyze_event_clip(s2, id2).await; });
    state.app_handle.emit("agent:analyzed", ()).ok();
    Ok(())
}

// `stream_chat_with_agent` was DELETED. It was a THIRD tag parser with its own
// five-line system prompt and its own hand-rolled [REMEMBER:] loop, it never
// called `parse_tags`, and it had no callers. Streaming now comes from
// `retrieve::answer` via the same `guardian:chat-token` event.

