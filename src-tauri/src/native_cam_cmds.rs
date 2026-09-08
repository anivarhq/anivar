//! Native camera management Tauri commands: list/start/stop devices, browser-camera reporting, active-camera query.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use tauri::State;

use crate::{
    AppState, BrowserCameraInfo, CameraInventory, CaptureTask, NativeCameraDevice,
    run_capture_loop,
};


/// Return which camera IDs have sent a frame in the last 30 seconds (i.e. are actively streaming).
#[tauri::command]
pub async fn get_active_cameras(state: State<'_, Arc<AppState>>) -> Result<Vec<u8>, String> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();
    let last_upd = state.scene_last_update.read().await;
    let mut active: Vec<u8> = last_upd.iter()
        .filter(|(_, t)| now.saturating_sub(**t) < 30)
        .map(|(c, _)| *c)
        .collect();
    // Also include cameras that have frames but haven't sent scene updates yet
    let frames = state.latest_frames.read().await;
    for cam_id in frames.keys() {
        if !active.contains(cam_id) { active.push(*cam_id); }
    }
    active.sort_unstable();
    Ok(active)
}

/// Mask the password in a stream URL for display (rtsp://user:••••@host…).
fn mask_stream_url(url: &str) -> String {
    if let Some(scheme_end) = url.find("://") {
        let after = &url[scheme_end + 3..];
        if let Some(at) = after.find('@') {
            let creds = &after[..at];
            if let Some(colon) = creds.find(':') {
                let user = &creds[..colon];
                return format!("{}://{}:••••@{}", &url[..scheme_end], user, &after[at + 1..]);
            }
        }
    }
    url.to_string()
}

/// Detailed per-camera telemetry for the Device Info panel — identity, recording
/// footprint, and live health, all from data we actually have (config + runtime
/// state + the nvr_segments ledger). Live stream stats (codec/res/fps) are fetched
/// separately by the UI via `probe_stream` so this stays instant.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct CameraTelemetry {
    pub cam_id:              u8,
    pub name:                String,
    pub brand:               String,
    pub source_type:         String,
    pub source_url_masked:   String,
    pub device_id:           String,
    pub transport:           String,
    pub online:              bool,
    pub last_frame_secs:     Option<u64>,
    pub nvr_enabled:         bool,
    pub recording:           bool,
    pub segments_total:      i64,
    pub bytes_total:         i64,
    pub duration_total_secs: f64,
    pub segments_today:      i64,
    pub bytes_today:         i64,
    pub oldest_at:           Option<String>,
    pub newest_at:           Option<String>,
}

#[tauri::command]
pub async fn get_camera_telemetry(
    state: State<'_, Arc<AppState>>,
    cam_id: u8,
) -> Result<CameraTelemetry, String> {
    // Config row (defaults if the slot was never saved).
    let cfg: Option<(String, String, String, String, String, String)> = sqlx::query_as(
        "SELECT name, source_type, source_url, device_id, transport, brand FROM camera_configs WHERE cam_id=?",
    ).bind(cam_id as i64).fetch_optional(&state.db).await.map_err(|e| e.to_string())?;
    let (name, source_type, source_url, device_id, transport, brand) = cfg.unwrap_or_else(|| (
        format!("Camera {}", cam_id + 1), "native".into(), String::new(), String::new(), "tcp".into(), String::new(),
    ));

    // Live health — same signal as get_active_cameras (frame in the last 30 s).
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();
    let last_frame_secs = state.scene_last_update.read().await.get(&cam_id).map(|t| now.saturating_sub(*t));
    let has_frame = state.latest_frames.read().await.contains_key(&cam_id);
    let online = has_frame || last_frame_secs.map(|s| s < 30).unwrap_or(false);
    let nvr_enabled = state.settings.read().await.nvr_enabled;

    // Recording footprint (whole history).
    let (segments_total, bytes_total, duration_total, oldest_at, newest_at):
        (i64, i64, Option<f64>, Option<String>, Option<String>) = sqlx::query_as(
        "SELECT COUNT(*), COALESCE(SUM(size_bytes),0), SUM(duration_secs), MIN(started_at), MAX(started_at)
           FROM nvr_segments WHERE cam_id=?",
    ).bind(cam_id as i64).fetch_one(&state.db).await.map_err(|e| e.to_string())?;

    // Today (local calendar day → UTC RFC3339 so it matches the stored format and
    // dodges the 'T'-vs-space lexicographic-compare trap).
    let today_start = {
        use chrono::{Local, TimeZone, Utc};
        Local::now().date_naive().and_hms_opt(0, 0, 0)
            .and_then(|m| Local.from_local_datetime(&m).single())
            .map(|dt| dt.with_timezone(&Utc).to_rfc3339())
            .unwrap_or_default()
    };
    let (segments_today, bytes_today): (i64, i64) = sqlx::query_as(
        "SELECT COUNT(*), COALESCE(SUM(size_bytes),0) FROM nvr_segments WHERE cam_id=? AND started_at >= ?",
    ).bind(cam_id as i64).bind(&today_start).fetch_one(&state.db).await.map_err(|e| e.to_string())?;

    Ok(CameraTelemetry {
        cam_id, name, brand, source_type,
        source_url_masked: mask_stream_url(&source_url),
        device_id, transport,
        online, last_frame_secs,
        nvr_enabled, recording: nvr_enabled && online,
        segments_total, bytes_total,
        duration_total_secs: duration_total.unwrap_or(0.0),
        segments_today, bytes_today,
        oldest_at, newest_at,
    })
}

/// Return everything the app knows about available cameras (native + network + browser).
#[tauri::command]
pub async fn get_camera_inventory(state: State<'_, Arc<AppState>>) -> Result<CameraInventory, String> {
    Ok(state.camera_inventory.read().await.clone())
}

/// Frontend calls this to report browser-enumerated camera devices (getUserMedia list).
#[tauri::command]
pub async fn report_browser_cameras(
    state: State<'_, Arc<AppState>>,
    cameras: Vec<BrowserCameraInfo>,
) -> Result<(), String> {
    state.camera_inventory.write().await.browser = cameras;
    Ok(())
}

/// List all physical cameras available on this machine via nokhwa.
#[tauri::command]
pub async fn list_native_cameras(state: State<'_, Arc<AppState>>) -> Result<Vec<NativeCameraDevice>, String> {
    #[cfg(any(windows, target_os = "macos", target_os = "linux"))]
    {
        // Try without spawn_blocking first, as query should be fast
        let res = {
            use nokhwa::utils::ApiBackend;
            nokhwa::query(ApiBackend::Auto)
                .map(|cams| {
                    cams.into_iter().map(|info| {
                        let index = match info.index() {
                            nokhwa::utils::CameraIndex::Index(n) => *n,
                            _ => 0,
                        };
                        NativeCameraDevice {
                            index,
                            name: info.human_name().to_string(),
                            description: info.description().to_string(),
                        }
                    }).collect::<Vec<_>>()
                })
                .map_err(|e| e.to_string())
        }; match res {
            Ok(result) => {
                state.camera_inventory.write().await.native = result.clone();
                Ok(result)
            }
            Err(_) => {
                let result: Vec<NativeCameraDevice> = tokio::task::spawn_blocking(|| {
                    use nokhwa::utils::ApiBackend;
                    nokhwa::query(ApiBackend::Auto)
                        .map(|cams| {
                            cams.into_iter().map(|info| {
                                let index = match info.index() {
                                    nokhwa::utils::CameraIndex::Index(n) => *n,
                                    _ => 0,
                                };
                                NativeCameraDevice {
                                    index,
                                    name: info.human_name().to_string(),
                                    description: info.description().to_string(),
                                }
                            }).collect()
                        })
                        .map_err(|e| e.to_string())
                })
                .await
                .map_err(|e| format!("Spawn blocking failed: {}", e))??;
                state.camera_inventory.write().await.native = result.clone();
                Ok(result)
            }
        }
    }
    #[cfg(not(any(windows, target_os = "macos", target_os = "linux")))]
    { Ok(vec![]) }
}

/// Start capturing from a physical camera in a dedicated Rust background task.
/// `cam_id` is the Anivar slot (0-3); `device_index` is the nokhwa device index.
#[tauri::command]
pub async fn start_native_camera(
    state: State<'_, Arc<AppState>>,
    cam_id: u8,
    device_index: u32,
) -> Result<(), String> {
    #[cfg(any(windows, target_os = "macos", target_os = "linux"))]
    {
        let cam_id = cam_id.min(15);
        // A REMOVED/disabled camera never re-opens its device (see camera_start_allowed).
        if !crate::cam_config::camera_start_allowed(&state.db, cam_id).await {
            tracing::info!("native capture cam{cam_id}: refused — camera is removed/disabled");
            return Err("camera is disabled".into());
        }
        let cancel_flag = Arc::new(AtomicBool::new(false));

        // ATOMIC dedupe + reserve (single lock scope — no race). Two CameraView
        // instances (grid + focused view) both call this nearly simultaneously;
        // opening the same USB device twice makes the two nokhwa handles preempt each
        // other (0xC00D3EA3) → no frames, no recording. Here: if the slot is already
        // capturing this device, no-op; else cancel any stale capture and RESERVE the
        // slot immediately so a concurrent call sees it and dedupes.
        {
            let mut handles = state.capture_handles.lock().await;
            if let Some(existing) = handles.get(&cam_id) {
                if existing.device_index == device_index && !existing.cancel.load(Ordering::Relaxed) {
                    return Ok(()); // already running this device — duplicate start
                }
                existing.cancel.store(true, Ordering::Relaxed); // stop a stale/other-device capture
            }
            handles.insert(cam_id, CaptureTask { cancel: Arc::clone(&cancel_flag), device_index });
        }

        // Let the OS thread release the device handle before the new loop opens it,
        // so the re-open doesn't race the old close and get "device preempted".
        tokio::time::sleep(std::time::Duration::from_millis(400)).await;

        let state_arc = Arc::clone(&state);
        // nokhwa FALLBACK path (dshow failed): boot no longer pre-spawns pipe
        // recorders for native cams (the dshow capture records directly), so
        // this path must self-provision its legacy recorder or footage stops.
        {
            let s2 = Arc::clone(&state_arc);
            tokio::spawn(async move {
                let settings = s2.settings.read().await.clone();
                if settings.nvr_enabled && !s2.nvr_pipe_txs.lock().await.contains_key(&cam_id) {
                    let enc = s2.hw_encoder.read().unwrap().clone();
                    match crate::spawn_nvr_pipe(cam_id, &s2.data_dir, settings.nvr_segment_mins,
                        &enc, s2.app_handle.clone(), s2.db.clone()).await
                    {
                        Ok((tx, child)) => {
                            s2.nvr_pipe_txs.lock().await.insert(cam_id, tx);
                            s2.nvr_processes.lock().await.insert(cam_id, child);
                            tracing::info!("nokhwa fallback cam{cam_id}: legacy pipe recorder self-provisioned");
                        }
                        Err(e) => tracing::warn!("nokhwa fallback cam{cam_id}: recorder spawn failed: {e}"),
                    }
                }
            });
        }
        tokio::spawn(run_capture_loop(cam_id, device_index, cancel_flag, state_arc));

        Ok(())
    }
    #[cfg(not(any(windows, target_os = "macos", target_os = "linux")))]
    { Err("Native capture not supported on this platform".into()) }
}

/// Stop a running native-capture task for the given camera slot.
#[tauri::command]
pub async fn stop_native_camera(
    state: State<'_, Arc<AppState>>,
    cam_id: u8,
) -> Result<(), String> {
    let cam_id = cam_id.min(15);
    let mut handles = state.capture_handles.lock().await;
    if let Some(task) = handles.remove(&cam_id) {
        task.cancel.store(true, Ordering::Relaxed);
    }
    if cam_id == 0 {
        *state.camera_active.write().await = false;
        state.camera_state_tx.send(false).ok();
    }
    Ok(())
}

/// Stop all running native-capture tasks (called on app exit or "stop all").
#[tauri::command]
pub async fn stop_all_native_cameras(state: State<'_, Arc<AppState>>) -> Result<(), String> {
    let mut handles = state.capture_handles.lock().await;
    for (_, task) in handles.drain() {
        task.cancel.store(true, Ordering::Relaxed);
    }
    *state.camera_active.write().await = false;
    state.camera_state_tx.send(false).ok();
    Ok(())
}
