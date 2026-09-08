//! Settings I/O, storage stats, motion-event listing/keep-alive, detection/AI-summary persistence.

use std::sync::Arc;
use std::time::Instant;

use tauri::{Emitter, State};

use crate::{
    AppState, MotionEvent, Settings, StorageInfo,
    crypto::encrypt_settings_secrets,
    db::save_settings_to_db,
};


#[tauri::command]
pub async fn get_storage_info(state: State<'_, Arc<AppState>>) -> Result<StorageInfo, String> {
    // Collect all clip paths tracked in the DB (async — cheap).
    let tracked: std::collections::HashSet<String> =
        sqlx::query_scalar::<_, Option<String>>("SELECT clip_path FROM motion_events")
            .fetch_all(&state.db).await.unwrap_or_default()
            .into_iter().flatten().collect();

    let row: Option<(i64, Option<String>, Option<String>)> =
        sqlx::query_as("SELECT COUNT(*), MIN(started_at), MAX(started_at) FROM motion_events")
            .fetch_optional(&state.db).await.map_err(|e| e.to_string())?;
    let (event_count, oldest_event, newest_event) = row
        .map(|(c, o, n)| (c as u64, o, n))
        .unwrap_or((0, None, None));

    // The filesystem walk can be tens of thousands of files (NVR segments). Run
    // it on a blocking thread so it NEVER stalls the async runtime — that was the
    // cause of the Storage panel spinning forever on large archives.
    let data_dir = state.data_dir.clone();
    let (total_bytes, clip_count, orphaned_clips, nvr_bytes, nvr_count) =
        tokio::task::spawn_blocking(move || {
            let mut total_bytes = 0u64;
            let mut clip_count  = 0u64;
            let mut orphaned    = 0u64;
            if let Ok(entries) = std::fs::read_dir(&data_dir) {
                for entry in entries.flatten() {
                    let Ok(meta) = entry.metadata() else { continue };
                    if !meta.is_file() { continue; }
                    total_bytes += meta.len();
                    let path = entry.path();
                    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                        if matches!(ext, "mp4" | "mkv" | "avi" | "ts" | "bin") {
                            clip_count += 1;
                            if !tracked.contains(&path.to_string_lossy().to_string()) {
                                orphaned += 1;
                            }
                        }
                    }
                }
            }
            let mut nvr_bytes = 0u64; let mut nvr_count = 0u64;
            if let Ok(entries) = std::fs::read_dir(data_dir.join("nvr")) {
                for e in entries.flatten() {
                    if let Ok(m) = e.metadata() {
                        if m.is_file() { nvr_bytes += m.len(); nvr_count += 1; }
                    }
                }
            }
            (total_bytes, clip_count, orphaned, nvr_bytes, nvr_count)
        }).await.map_err(|e| e.to_string())?;

    Ok(StorageInfo {
        total_bytes: total_bytes + nvr_bytes,
        clip_count, event_count, orphaned_clips, oldest_event, newest_event,
        nvr_bytes, nvr_count,
    })
}

#[tauri::command]
pub async fn get_settings(state: State<'_, Arc<AppState>>) -> Result<Settings, String> {
    // Return in-memory settings — always decrypted (encryption only happens at DB write)
    Ok(state.settings.read().await.clone())
}

/// Single settings-write path used by BOTH the app's `save_settings` command and
/// the Telegram `/menu` (`agent::dispatch::persist_settings`). Encrypts secrets
/// for the DB, keeps the in-memory copy decrypted, and emits `settings:updated`
/// so the frontend (App.tsx) refreshes live — keeping the app's Alerts panel and
/// the Telegram /menu in lockstep no matter which side made the change.
pub(crate) async fn apply_settings_update(state: &Arc<AppState>, new_s: Settings) {
    // Keep the GPU-inference switch in sync (applies to models loaded after this;
    // already-cached sessions keep their device until the app restarts).
    crate::inference::set_gpu_inference(new_s.inference_device != "cpu");
    // Snapshot the OLD audio + depth-anonymization config before overwriting.
    let (old_audio, old_depth, old_relaunch) = {
        let g = state.settings.read().await;
        ((g.audio_detection, g.audio_listen.clone(), g.audio_threshold),
         g.depth_anonymize.clone(),
         g.relaunch_after_crash)
    };
    let mut encrypted = new_s.clone();
    encrypt_settings_secrets(&state.master_key, &mut encrypted);
    let _ = save_settings_to_db(&state.db, &encrypted).await;
    *state.settings.write().await = new_s.clone();
    let _ = state.app_handle.emit("settings:updated", &new_s);

    // Audio toggled or retuned → (re)apply to running cameras WITHOUT a manual restart.
    let new_audio = (new_s.audio_detection, new_s.audio_listen.clone(), new_s.audio_threshold);
    if old_audio != new_audio {
        let st = state.clone();
        let on = new_s.audio_detection;
        tokio::spawn(async move { reapply_audio(&st, on).await; });
    }

    // Depth Anonymization toggled → refresh the registry and respawn the affected
    // USB captures (the pipeline SHAPE changes: record-in-ffmpeg vs depth pipe).
    crate::depth::refresh_anon_cams(&new_s.depth_anonymize);
    // Depth-model variant pick (takes effect on next session load / restart).
    crate::depth::set_depth_variant(&new_s.depth_model);
    // On-device model tier. Takes effect immediately — `set_tier` unloads the old
    // weights when the path changes, so the next answer comes from the model the
    // user just picked rather than the one still memory-mapped.
    crate::agent::local_llm::set_tier(&state.data_dir, &new_s.local_llm_tier);
    if old_depth != new_s.depth_anonymize {
        let st = state.clone();
        tokio::spawn(async move { reapply_depth_anonymize(&st).await; });
    }

    // Crash auto-restart toggled → (un)register the keep-alive Scheduled Task.
    if old_relaunch != new_s.relaunch_after_crash {
        let enable = new_s.relaunch_after_crash;
        tokio::spawn(async move {
            if let Err(e) = crate::system_cmds::set_keepalive_task(enable).await {
                tracing::warn!("keepalive task update failed: {e}");
            }
        });
    }
}

/// Respawn every running USB capture whose depth-anonymization state may have
/// changed. Same kill+restart pattern as the boot capture watchdog: remove the
/// capture_key (defeats the idempotency skip), kill the process, restart. The
/// per-cam pipe recorder is killed too — spawn_capture rebuilds the right shape.
pub(crate) async fn reapply_depth_anonymize(state: &Arc<AppState>) {
    use tauri::Manager;
    let usb: Vec<(u8, String)> = state.capture_keys.lock().await.iter()
        .filter_map(|(c, k)| k.strip_prefix("usb:").map(|d| (*c, d.to_string())))
        .collect();
    for (cam, device) in usb {
        if let Some(mut child) = state.rtsp_processes.lock().await.remove(&cam) {
            let _ = child.kill().await;
        }
        state.capture_keys.lock().await.remove(&cam);
        if let Some(mut child) = state.nvr_processes.lock().await.remove(&cam) {
            let _ = child.kill().await;
        }
        state.nvr_pipe_txs.lock().await.remove(&cam);
        let st = state.app_handle.state::<Arc<AppState>>();
        match crate::dshow::start_usb_capture(st, cam, device).await {
            Ok(()) => tracing::info!("cam{cam}: capture respawned for depth-anonymization change"),
            Err(e) => tracing::warn!("cam{cam}: depth-anonymization respawn failed: {e}"),
        }
    }
}

/// (Re)apply audio to every CURRENTLY-RUNNING camera after an audio-settings change, so
/// toggling takes effect immediately (mature NVRs restart the affected per-camera ffmpeg on
/// a config reload — same idea). USB cams mux the mic into their NVR segments; RTSP cams
/// only (re)start the detector (their recorder already carries the camera's own audio).
async fn reapply_audio(state: &Arc<AppState>, audio_on: bool) {
    let ffmpeg = match crate::ensure_ffmpeg(&state.data_dir).await { Ok(f) => f, Err(_) => return };
    let active: Vec<(u8, String)> =
        state.capture_keys.lock().await.iter().map(|(c, k)| (*c, k.clone())).collect();
    for (cam, key) in active {
        // Clean restart: stop any running detector first.
        if let Some(mut c) = state.audio_processes.lock().await.remove(&cam) { let _ = c.kill().await; }
        if !audio_on {
            crate::audio_cmds::close_audio_for_cam(state, cam).await; // finalize any open event
        }
        if key.starts_with("usb:") {
            // Detector only (ring buffer + YAMNet). Recording audio is CAPTURE-OWNED
            // (muxed in-process) and independent of the detection toggle.
            if audio_on {
                if let Some(mic) = crate::dshow::audio_input_args(&ffmpeg).await {
                    crate::audio_cmds::spawn_audio_detection(state, cam, &ffmpeg, mic, true).await;
                }
            }
        } else if let Some(url) = key.strip_prefix("rtsp:") {
            if audio_on {
                crate::audio_cmds::spawn_audio_detection(
                    state, cam, &ffmpeg,
                    vec!["-rtsp_transport".into(), "tcp".into(), "-i".into(), url.to_string()],
                    false,
                ).await;
            }
        }
    }
    tracing::info!("audio re-applied (on={audio_on}) to running cameras");
}

#[tauri::command]
pub async fn save_settings(state: State<'_, Arc<AppState>>, settings: Settings) -> Result<(), String> {
    apply_settings_update(state.inner(), settings).await;
    Ok(())
}

#[tauri::command]
pub async fn get_motion_events(state: State<'_, Arc<AppState>>, limit: Option<i64>) -> Result<Vec<MotionEvent>, String> {
    let capped_limit = limit.unwrap_or(50).min(500);
    let rows: Vec<(String, String, Option<String>, Option<f64>, f64, Option<String>, Option<String>, Option<String>, Option<String>, Option<String>, Option<String>, Option<String>, Option<String>, Option<String>)> =
        // PAYLOAD: ship a tiny '@thumb' PRESENCE MARKER instead of the ~50KB inline
        // base64 thumbnail — 500 events went from megabytes of JSON over the WebView2
        // IPC bridge (re-parsed on every detection refetch) to a few KB. The frontend
        // renders markers via GET /footage/:id/thumbnail (HTTP-cached, evictable).
        sqlx::query_as("SELECT id, started_at, ended_at, duration_secs, peak_score, clip_path, CASE WHEN thumbnail IS NOT NULL AND thumbnail <> '' THEN '@thumb' END AS thumbnail, detections, ai_summary, event_category, recognized_plate, dominant_label, sub_label, first_object_at FROM motion_events ORDER BY started_at DESC LIMIT ?")
            .bind(capped_limit).fetch_all(&state.db).await.map_err(|e| e.to_string())?;
    Ok(rows.into_iter().map(|(id, started_at, ended_at, duration_secs, peak_score, clip_path, thumbnail, detections, ai_summary, event_category, recognized_plate, dominant_label, sub_label, first_object_at)| {
        MotionEvent {
            id, started_at, ended_at, duration_secs, peak_score: peak_score as f32,
            clip_path, thumbnail, detections, ai_summary, cam_id: None,
            event_category, recognized_plate, dominant_label, sub_label, first_object_at,
            top_speed_kmh: None,
        }
    }).collect())
}

/// Called by the frontend whenever AI detects a person/animal in frame.
/// Resets last_motion_at so the event and recording stay open as long as
/// the subject is visible, even if they are completely still.
#[tauri::command]
pub async fn keep_alive_event(
    state: State<'_, Arc<AppState>>,
    cam_id: Option<u8>,
) -> Result<(), String> {
    let cam = cam_id.unwrap_or(0).min(15);
    let mut cs_map = state.cam_states.lock().await;
    if let Some(cs) = cs_map.get_mut(&cam) {
        if cs.motion_active.is_some() {
            cs.last_motion_at = Some(Instant::now());
        }
    }
    Ok(())
}

#[tauri::command]
pub async fn store_detections(
    state: State<'_, Arc<AppState>>,
    event_id: String,
    detections: String,
) -> Result<(), String> {
    sqlx::query("UPDATE motion_events SET detections=? WHERE id=?")
        .bind(&detections).bind(&event_id)
        .execute(&state.db).await.map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
pub async fn store_ai_summary(
    state: State<'_, Arc<AppState>>,
    event_id: String,
    summary: String,
) -> Result<(), String> {
    sqlx::query("UPDATE motion_events SET ai_summary=? WHERE id=?")
        .bind(&summary).bind(&event_id)
        .execute(&state.db).await.map_err(|e| e.to_string())?;
    Ok(())
}
