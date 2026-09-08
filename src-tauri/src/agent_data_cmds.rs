//! Tauri commands backing the assistant's data surface: event search/explore, alert
//! conditions, memory files, crowd reports, and motion-event maintenance.

use std::sync::Arc;

use tauri::{Manager, State};

use crate::AppState;


#[tauri::command]
pub async fn search_clips(
    state: State<'_, Arc<AppState>>,
    query: String,
) -> Result<Vec<serde_json::Value>, String> {
    Ok(crate::agent::search_clips(&state, &query).await)
}

#[tauri::command]
pub async fn list_alert_conditions(state: State<'_, Arc<AppState>>) -> Result<Vec<crate::agent::AlertCondition>, String> {
    Ok(crate::agent::list_alert_conditions(&state.db).await)
}

#[tauri::command]
pub async fn create_alert_condition(
    state: State<'_, Arc<AppState>>,
    name: String, condition: String,
    channels: String, min_risk: String,
) -> Result<crate::agent::AlertCondition, String> {
    Ok(crate::agent::create_alert_condition(&state.db, &name, &condition, &channels, &min_risk).await)
}

#[tauri::command]
pub async fn delete_alert_condition(state: State<'_, Arc<AppState>>, id: String) -> Result<(), String> {
    crate::agent::delete_alert_condition(&state.db, &id).await;
    Ok(())
}

#[tauri::command]
pub async fn toggle_alert_condition(
    state: State<'_, Arc<AppState>>, id: String, enabled: bool,
) -> Result<(), String> {
    crate::agent::toggle_alert_condition(&state.db, &id, enabled).await;
    Ok(())
}

#[tauri::command]
pub async fn read_memory_file(
    state: State<'_, Arc<AppState>>, category: String,
) -> Result<String, String> {
    Ok(crate::agent::read_memory_file(&state.db, &category).await)
}

#[tauri::command]
pub async fn write_memory_file(
    state: State<'_, Arc<AppState>>, category: String, content: String,
) -> Result<(), String> {
    crate::agent::write_memory_file(&state.db, &category, &content).await;
    Ok(())
}

#[tauri::command]
pub async fn read_all_memory_files(state: State<'_, Arc<AppState>>) -> Result<String, String> {
    Ok(crate::agent::read_all_memory_files(&state.db).await)
}

/// Assistant: crowd counting — called by frontend with person count per frame.
/// Fires an alert when count exceeds the configured threshold.
#[tauri::command]
pub async fn report_crowd_count(
    state: State<'_, Arc<AppState>>,
    cam_id: u8,
    count: u32,
    event_id: Option<String>,
) -> Result<(), String> {
    let s = state.settings.read().await.clone();
    if !s.crowd_detection { return Ok(()); }
    let threshold = s.crowd_threshold;
    drop(s);

    if count < threshold { return Ok(()); }

    // Sustained-crowd gate: over-threshold reports must keep arriving (≤3 s gaps)
    // for ≥5 s before a crowd is real — a single frame of YOLO double-boxing or a
    // group briefly walking past is not a crowd. The streak self-resets via the
    // gap timeout (the frontend only reports frames with 2+ people).
    const CROWD_SUSTAIN_SECS: u64 = 5;
    const CROWD_GAP_SECS: u64 = 3;
    let now = std::time::Instant::now();
    let mut cs_map = state.cam_states.lock().await;
    let cs = cs_map.entry(cam_id).or_default();
    let since = match cs.crowd_over {
        Some((since, last)) if now.duration_since(last).as_secs() <= CROWD_GAP_SECS => since,
        _ => now,
    };
    cs.crowd_over = Some((since, now));
    if now.duration_since(since).as_secs() < CROWD_SUSTAIN_SECS { return Ok(()); }
    // Only alert once per crowd event (suppress repeats)
    if cs.crowd_alerted_count >= count { return Ok(()); }
    cs.crowd_alerted_count = count;
    drop(cs_map);

    let summary = format!(
        "{} people detected simultaneously on cam{} (threshold: {}). {}",
        count, cam_id + 1, threshold,
        event_id.as_deref().map(|_| "Recording in progress.").unwrap_or("")
    );
    tracing::warn!("[on-device assistants] Crowd alert: {}", summary);
    let state2 = Arc::clone(&*state);
    tauri::async_runtime::spawn(async move {
        crate::agent::dispatch_intelligence_alert(&state2, "crowd", &summary, cam_id, None).await;
    });
    Ok(())
}

#[tauri::command]
pub async fn delete_motion_event(state: State<'_, Arc<AppState>>, id: String) -> Result<(), String> {
    delete_events(state, vec![id]).await
}

/// Delete events and everything that belongs to them: exported clip files, the
/// row, every satellite row, the bookmark, and the review group's stale member
/// list. A Review card is a GROUP of events, so the UI deletes all its members
/// in one call — deleting them one at a time re-aggregated the group in between
/// and briefly resurrected a card the user had just dismissed.
///
/// Unknown ids are ignored (already deleted elsewhere / double click), so the
/// command is idempotent. Face + People data is never touched (hard rule).
#[tauri::command]
pub async fn delete_events(state: State<'_, Arc<AppState>>, ids: Vec<String>) -> Result<(), String> {
    if ids.is_empty() { return Ok(()); }
    // Resolve clip paths BEFORE the rows go away.
    let ph = vec!["?"; ids.len()].join(",");
    let sql = format!("SELECT id, clip_path FROM motion_events WHERE id IN ({ph})");
    let mut q = sqlx::query_as::<_, (String, Option<String>)>(&sql);
    for id in &ids { q = q.bind(id); }
    let rows = q.fetch_all(&state.db).await.map_err(|e| e.to_string())?;

    // One deletion path: rows + all satellites + review-group repair.
    crate::nvr_recording::delete_events_and_satellites(&state.db, &ids).await;
    crate::nvr_recording::prune_empty_review_segments(&state.db).await;

    // Files last: with the rows gone, a clip that can't be unlinked right now
    // (Windows keeps a handle while the player is open) is a plain orphan that
    // `purge_orphaned_clips` sweeps — never a card pointing at nothing.
    let st = state.inner().clone();
    for (id, clip) in rows {
        crate::agent::clip_export::delete_event_clip(&st, &id, clip).await;
    }
    Ok(())
}

/// Stop every running NVR pipe so the ffmpeg processes RELEASE their open file
/// handles — on Windows an actively-written segment can't be deleted, which is
/// why "delete all footage" silently failed while recording was live. Returns
/// the cam ids that were stopped so the caller can respawn them afterwards.
async fn stop_all_nvr_pipes(state: &Arc<AppState>) -> Vec<u8> {
    let cams: Vec<u8> = {
        let mut set: std::collections::BTreeSet<u8> = std::collections::BTreeSet::new();
        set.extend(state.nvr_pipe_txs.lock().await.keys().copied());
        set.extend(state.nvr_processes.lock().await.keys().copied());
        set.into_iter().collect()
    };
    state.nvr_pipe_txs.lock().await.clear();
    {
        let mut procs = state.nvr_processes.lock().await;
        for (_, mut child) in procs.drain() { child.kill().await.ok(); }
    }
    // USB captures RECORD DIRECTLY now — they hold the live .tmp.mp4 Windows
    // lock, so they must be stopped too or deletion silently fails.
    let usb: Vec<(u8, String)> = state.capture_keys.lock().await.iter()
        .filter_map(|(c, k)| k.strip_prefix("usb:").map(|d| (*c, d.to_string())))
        .collect();
    for (cam, _) in &usb {
        if let Some(mut child) = state.rtsp_processes.lock().await.remove(cam) {
            child.kill().await.ok();
        }
        state.capture_keys.lock().await.remove(cam);
    }
    *state.stopped_usb_captures.lock().await = usb;
    // Give the OS a moment to release the file handles after the kill.
    tokio::time::sleep(std::time::Duration::from_millis(400)).await;
    cams
}

/// Respawn the NVR pipes for the given cams (after a stop+delete).
async fn respawn_nvr_pipes(state: &Arc<AppState>, cams: &[u8]) {
    // USB captures restart regardless of nvr_enabled — they also serve live view.
    let usb: Vec<(u8, String)> = std::mem::take(&mut *state.stopped_usb_captures.lock().await);
    for (cam, device) in usb {
        let st = state.app_handle.state::<Arc<AppState>>();
        if let Err(e) = crate::dshow::start_usb_capture(st, cam, device).await {
            tracing::warn!("respawn: usb capture cam{cam} failed: {e}");
        }
    }
    if cams.is_empty() { return; }
    if !state.settings.read().await.nvr_enabled { return; }
    let seg_mins = state.settings.read().await.nvr_segment_mins;
    let enc = state.hw_encoder.read().unwrap().clone();
    for &cam in cams {
        if let Ok((tx, child)) = crate::spawn_nvr_pipe(
            cam, &state.data_dir, seg_mins, &enc,
            state.app_handle.clone(), state.db.clone()).await
        {
            state.nvr_pipe_txs.lock().await.insert(cam, tx);
            state.nvr_processes.lock().await.insert(cam, child);
        }
    }
}

#[tauri::command]
pub async fn clear_nvr_recordings(state: State<'_, Arc<AppState>>) -> Result<u64, String> {
    // 1. Stop recording so ffmpeg releases its file locks.
    let cams = stop_all_nvr_pipes(&state).await;

    // 2. DB is the source of truth: delete the files the DB knows about first.
    let paths: Vec<String> = sqlx::query_scalar("SELECT path FROM nvr_segments")
        .fetch_all(&state.db).await.unwrap_or_default();
    let nvr_dir = state.data_dir.join("nvr");
    let mut removed = 0u64;
    for p in &paths {
        if tokio::fs::remove_file(p).await.is_ok() { removed += 1; }
    }
    // 3. Sweep any stragglers on disk not tracked in the DB (orphans / .tmp).
    let nvr_dir2 = nvr_dir.clone();
    let swept = tokio::task::spawn_blocking(move || -> u64 {
        let Ok(entries) = std::fs::read_dir(&nvr_dir2) else { return 0; };
        let mut c = 0u64;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() && std::fs::remove_file(&path).is_ok() { c += 1; }
        }
        c
    }).await.unwrap_or(0);

    // 4. Clear the table (DB single source of truth).
    sqlx::query("DELETE FROM nvr_segments").execute(&state.db).await.map_err(|e| e.to_string())?;

    // 5. Resume recording.
    respawn_nvr_pipes(&state, &cams).await;
    Ok(removed + swept)
}

#[tauri::command]
pub async fn clear_all_events(state: State<'_, Arc<AppState>>) -> Result<(), String> {
    // 1. Stop recording so ffmpeg releases its file locks (otherwise the live
    //    segment can't be deleted on Windows and footage "won't delete").
    let cams = stop_all_nvr_pipes(&state).await;

    // 2. DB is the source of truth — delete the clip files it references, then
    //    the NVR segment files it references.
    let clip_paths: Vec<String> = sqlx::query_scalar::<_, Option<String>>(
        "SELECT clip_path FROM motion_events")
        .fetch_all(&state.db).await.unwrap_or_default()
        .into_iter().flatten().collect();
    for p in &clip_paths { tokio::fs::remove_file(p).await.ok(); }

    let seg_paths: Vec<String> = sqlx::query_scalar("SELECT path FROM nvr_segments")
        .fetch_all(&state.db).await.unwrap_or_default();
    for p in &seg_paths { tokio::fs::remove_file(p).await.ok(); }

    // 3. Wipe both tables + EVERY event satellite (DB single source of truth).
    //    Leaving review_segments behind rendered GHOST cards in Review for
    //    events that no longer existed — the feed's day view reads groups, not
    //    raw events. Face/People tables are never touched (hard rule).
    sqlx::query("DELETE FROM motion_events").execute(&state.db).await.map_err(|e| e.to_string())?;
    sqlx::query("DELETE FROM nvr_segments").execute(&state.db).await.ok();
    for t in ["review_segments", "event_embeddings", "event_timeline", "agent_alerts", "event_bookmarks"] {
        let _ = sqlx::query(&format!("DELETE FROM {t}")).execute(&state.db).await;
    }
    sqlx::query("VACUUM").execute(&state.db).await.ok();

    // 4. Sweep any stragglers on disk (orphan clips in root + any file in nvr/).
    delete_all_clips_from_dir(&state.data_dir).await;
    let nvr_dir = state.data_dir.join("nvr");
    tokio::task::spawn_blocking(move || {
        if let Ok(entries) = std::fs::read_dir(&nvr_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() { std::fs::remove_file(&path).ok(); }
            }
        }
    }).await.ok();

    // 5. Resume recording.
    respawn_nvr_pipes(&state, &cams).await;
    Ok(())
}

pub(crate) fn is_clip_file(ext: &str) -> bool {
    matches!(ext, "mp4" | "mkv" | "avi" | "ts" | "bin" | "webm")
}

pub(crate) async fn delete_all_clips_from_dir(dir: &std::path::Path) {
    let dir = dir.to_path_buf();
    tokio::task::spawn_blocking(move || {
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() {
                    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                        if is_clip_file(ext) {
                            std::fs::remove_file(&path).ok();
                        }
                    }
                }
            }
        }
    }).await.ok();
}

/// Result summary for a ranged delete — shown back to the user.
#[derive(serde::Serialize)]
pub struct DeleteResult {
    pub segments_deleted: u64,
    pub events_deleted: u64,
}

/// Enterprise storage management: delete all footage (NVR segments + motion
/// events/clips) whose start time falls in [from, to] (RFC3339 UTC strings).
/// DB is the single source of truth — we select the affected rows, delete their
/// files, then delete the rows. Lock-safe: stops the NVR pipes first so the live
/// segment file can actually be removed, then respawns recording.
#[tauri::command]
pub async fn delete_footage_in_range(
    state: State<'_, Arc<AppState>>,
    from: String,
    to: String,
) -> Result<DeleteResult, String> {
    // Stop recording so ffmpeg releases handles (Windows can't delete open files).
    let cams = stop_all_nvr_pipes(&state).await;

    // ── NVR segments in range ──
    let seg_paths: Vec<String> = sqlx::query_scalar(
        "SELECT path FROM nvr_segments WHERE started_at >= ? AND started_at <= ?")
        .bind(&from).bind(&to)
        .fetch_all(&state.db).await.unwrap_or_default();
    let mut segments_deleted = 0u64;
    for p in &seg_paths {
        if tokio::fs::remove_file(p).await.is_ok() { segments_deleted += 1; }
    }
    sqlx::query("DELETE FROM nvr_segments WHERE started_at >= ? AND started_at <= ?")
        .bind(&from).bind(&to)
        .execute(&state.db).await.map_err(|e| e.to_string())?;

    // ── Motion events (+ their clip files) in range ──
    let clip_paths: Vec<Option<String>> = sqlx::query_scalar(
        "SELECT clip_path FROM motion_events WHERE started_at >= ? AND started_at <= ?")
        .bind(&from).bind(&to)
        .fetch_all(&state.db).await.unwrap_or_default();
    for p in clip_paths.into_iter().flatten() {
        tokio::fs::remove_file(&p).await.ok();
    }
    // Delete through the ONE path that also removes every satellite row
    // (embeddings/timeline/alerts) — a bare `DELETE FROM motion_events` left
    // review groups rendering ghost cards for footage that no longer existed.
    let ids: Vec<String> = sqlx::query_scalar(
        "SELECT id FROM motion_events WHERE started_at >= ? AND started_at <= ?")
        .bind(&from).bind(&to)
        .fetch_all(&state.db).await.unwrap_or_default();
    let events_deleted = ids.len() as u64;
    crate::nvr_recording::delete_events_and_satellites(&state.db, &ids).await;
    crate::nvr_recording::prune_empty_review_segments(&state.db).await;

    sqlx::query("VACUUM").execute(&state.db).await.ok();

    // Resume recording.
    respawn_nvr_pipes(&state, &cams).await;
    Ok(DeleteResult { segments_deleted, events_deleted })
}

#[tauri::command]
pub async fn purge_orphaned_clips(state: State<'_, Arc<AppState>>) -> Result<u64, String> {
    let tracked: std::collections::HashSet<String> =
        sqlx::query_scalar::<_, Option<String>>("SELECT clip_path FROM motion_events")
            .fetch_all(&state.db).await.map_err(|e| e.to_string())?
            .into_iter().flatten().collect();

    let data_dir = state.data_dir.clone();
    let removed = tokio::task::spawn_blocking(move || -> u64 {
        let Ok(entries) = std::fs::read_dir(&data_dir) else { return 0; };
        let mut count = 0u64;
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() { continue; }
            let Some(ext) = path.extension().and_then(|e| e.to_str()) else { continue; };
            if is_clip_file(ext) {
                let key = path.to_string_lossy().to_string();
                if !tracked.contains(&key) && std::fs::remove_file(&path).is_ok() {
                    count += 1;
                }
            }
        }
        count
    }).await.map_err(|e| e.to_string())?;

    Ok(removed)
}

