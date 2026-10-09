//! Multi-camera configuration commands (`get_camera_configs`, `set_camera_config`, `get_active_cameras`).

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tauri::{Emitter, State};

use crate::AppState;


fn default_transport() -> String { "tcp".into() }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraConfig {
    pub cam_id:      u8,
    pub name:        String,
    pub source_type: String,  // "browser" | "rtsp" | "mjpeg" | "native"
    pub source_url:  String,
    pub device_id:   String,
    pub enabled:     bool,
    /// RTSP transport ("tcp" default | "udp"). Drives the live relay, recorder and
    /// audio tap — not just the connection test. Ignored for non-RTSP sources.
    #[serde(default = "default_transport")]
    pub transport:   String,
    /// Optional make/brand the user picked or typed (e.g. "Reolink", "Lorex").
    /// Shown in Device Info; falls back to a live ONVIF manufacturer when present.
    #[serde(default)]
    pub brand:       String,
    /// Optional LOW-RES sub-stream URL for detection (mature NVRs' model: detect on
    /// the camera's substream, record the main stream). Empty = detect on the
    /// main stream, downscaled in ffmpeg (works, but decodes the full stream).
    #[serde(default)]
    pub detect_url:  String,
}

/// May this camera slot open its device RIGHT NOW?
///
/// THE authority for "is this camera on", checked inside every capture-start
/// path (USB/dshow, native, RTSP relay). Without it the backend trusted its
/// callers, so a stale frontend auto-start (CameraView restores from
/// `localStorage.cam_source_N`) re-opened the webcam for a camera the user had
/// REMOVED — light back on, minutes after boot, with `enabled=0` in the DB.
///
/// A MISSING row means "not configured yet" and is allowed: `set_camera_config`
/// always writes the row (enabled=1) before anything starts a capture, so the
/// add-camera and onboarding flows are unaffected. Only an explicit `enabled=0`
/// refuses. Same rule the capture watchdog already used to decide respawns.
pub(crate) async fn camera_start_allowed(db: &sqlx::SqlitePool, cam_id: u8) -> bool {
    sqlx::query_scalar::<_, i64>("SELECT enabled FROM camera_configs WHERE cam_id=?")
        .bind(cam_id as i64)
        .fetch_optional(db).await
        .ok().flatten()
        .map(|e| e != 0)
        .unwrap_or(true)
}

/// Depth anonymization belongs to a USB camera, not to its slot. A removed
/// camera's flag stayed on the slot, and the next camera added there (often a
/// network camera, which can't be anonymized) showed "Anonymized" while its live
/// view, snapshots and share links were raw. Keeps flags only for enabled USB
/// cameras; runs on every camera save and at boot.
pub(crate) async fn drop_stale_anonymize(state: &Arc<AppState>) {
    let usb: Vec<i64> = sqlx::query_scalar(
        "SELECT cam_id FROM camera_configs WHERE enabled=1 AND source_type='native'"
    ).fetch_all(&state.db).await.unwrap_or_default();
    let mut s = state.settings.read().await.clone();
    let flags: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(&s.depth_anonymize).unwrap_or_default();
    let kept = keep_usb_flags(&flags, &usb);
    if kept.len() == flags.len() { return; }
    tracing::info!("depth anonymization: cleared {} slot(s) with no USB camera", flags.len() - kept.len());
    s.depth_anonymize = serde_json::Value::Object(kept).to_string();
    crate::events_cmds::apply_settings_update(state, s).await;
}

fn keep_usb_flags(flags: &serde_json::Map<String, serde_json::Value>, usb: &[i64])
    -> serde_json::Map<String, serde_json::Value>
{
    flags.iter()
        .filter(|(cam, on)| on.as_bool() == Some(true) && cam.parse::<i64>().is_ok_and(|c| usb.contains(&c)))
        .map(|(cam, on)| (cam.clone(), on.clone()))
        .collect()
}

/// Camera URLs carry the camera's login (`rtsp://user:pass@host/…`), so they're
/// stored encrypted with the app's `.master_key`; every reader opens them here.
/// A value this key can't open (the key file was replaced) is returned as it is:
/// saving the camera again keeps it, and restoring the key file brings it back.
pub(crate) fn open_url(key: &[u8; 32], stored: &str) -> String {
    crate::crypto::try_decrypt(key, stored).unwrap_or_else(|_| stored.to_string())
}

/// Encrypts camera URLs saved before they were encrypted. Runs at boot;
/// `encrypt_secret` leaves an already-encrypted value alone.
pub(crate) async fn seal_plain_urls(state: &AppState) {
    let rows: Vec<(i64, String, String)> = sqlx::query_as(
        "SELECT cam_id, source_url, detect_url FROM camera_configs
         WHERE (source_url <> '' AND source_url NOT LIKE 'enc:%')
            OR (detect_url <> '' AND detect_url NOT LIKE 'enc:%')"
    ).fetch_all(&state.db).await.unwrap_or_default();
    if rows.is_empty() { return; }
    let Ok(mut conn) = state.db.acquire().await else { return };
    // Zeroes the plaintext's old bytes instead of leaving them in the file's free space.
    let _ = sqlx::query("PRAGMA secure_delete=ON").execute(&mut *conn).await;
    for (cam, url, det) in &rows {
        let _ = sqlx::query("UPDATE camera_configs SET source_url=?, detect_url=? WHERE cam_id=?")
            .bind(crate::crypto::encrypt_secret(&state.master_key, url))
            .bind(crate::crypto::encrypt_secret(&state.master_key, det))
            .bind(cam).execute(&mut *conn).await;
    }
    let _ = sqlx::query("PRAGMA secure_delete=OFF").execute(&mut *conn).await;
    tracing::info!("encrypted the saved URLs of {} camera(s)", rows.len());
}

/// Return configurations for all 16 camera slots.
#[tauri::command]
pub async fn get_camera_configs(state: State<'_, Arc<AppState>>) -> Result<Vec<CameraConfig>, String> {
    let rows: Vec<(i64, String, String, String, String, i64, String, String, String)> =
        sqlx::query_as("SELECT cam_id,name,source_type,source_url,device_id,enabled,transport,brand,detect_url FROM camera_configs ORDER BY cam_id ASC")
            .fetch_all(&state.db).await.map_err(|e| e.to_string())?;
    let mut configs: Vec<CameraConfig> = rows.into_iter().map(|(id, name, st, url, dev, en, tr, brand, det)| CameraConfig {
        cam_id: id as u8, name, source_type: st, source_url: open_url(&state.master_key, &url),
        device_id: dev, enabled: en != 0,
        transport: if tr.is_empty() { "tcp".into() } else { tr },
        brand,
        detect_url: open_url(&state.master_key, &det),
    }).collect();
    // Fill in defaults for any slots not yet in DB
    for id in 0u8..16 {
        if !configs.iter().any(|c| c.cam_id == id) {
            configs.push(CameraConfig {
                cam_id: id,
                name: format!("Camera {}", id + 1),
                source_type: "browser".into(),
                source_url: String::new(),
                device_id: String::new(),
                enabled: false,
                transport: "tcp".into(),
                brand: String::new(),
                detect_url: String::new(),
            });
        }
    }
    configs.sort_by_key(|c| c.cam_id);
    Ok(configs)
}

/// Save or update the configuration for one camera slot.
#[tauri::command]
pub async fn set_camera_config(
    state: State<'_, Arc<AppState>>,
    config: CameraConfig,
) -> Result<(), String> {
    let transport = if config.transport.is_empty() { "tcp" } else { config.transport.as_str() };
    sqlx::query(
        "INSERT OR REPLACE INTO camera_configs(cam_id,name,source_type,source_url,device_id,enabled,transport,brand,detect_url)
         VALUES(?,?,?,?,?,?,?,?,?)"
    ).bind(config.cam_id as i64).bind(&config.name)
     .bind(&config.source_type).bind(crate::crypto::encrypt_secret(&state.master_key, &config.source_url))
     .bind(&config.device_id).bind(config.enabled as i64).bind(transport).bind(&config.brand)
     .bind(crate::crypto::encrypt_secret(&state.master_key, &config.detect_url))
     .execute(&state.db).await.map_err(|e| e.to_string())?;

    // A camera that's been DISABLED/REMOVED must stop capturing NOW. Writing
    // `enabled=0` alone left its ffmpeg alive holding the device — the webcam
    // LED stayed on for a camera the UI no longer showed, until the whole app
    // exited. Done here (not in the frontend) so every caller — Live remove,
    // Settings, onboarding — tears down through one path.
    if !config.enabled {
        crate::rtsp::stop_capture_for_cam(state.inner(), config.cam_id.min(15)).await;
    }
    drop_stale_anonymize(state.inner()).await;

    state.app_handle.emit("cameras:updated", ()).ok();
    // Home Assistant learns cameras from MQTT discovery, sent on connect:
    // reconnect so an added, renamed or removed camera shows up there too.
    let st = state.inner().clone();
    tokio::spawn(async move { crate::mqtt::restart(&st).await; });
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn only_an_enabled_usb_camera_keeps_its_anonymize_flag() {
        let flags: serde_json::Map<String, serde_json::Value> =
            serde_json::from_str(r#"{"0": true, "1": true, "2": false, "x": true}"#).unwrap();
        // Slot 0 now holds a network camera; slot 1 is still the USB camera.
        let kept = super::keep_usb_flags(&flags, &[1]);
        assert_eq!(serde_json::Value::Object(kept).to_string(), r#"{"1":true}"#);
    }

    #[test]
    fn a_camera_url_is_stored_encrypted_and_reads_back() {
        let key = [7u8; 32];
        let url = "rtsp://admin:s3cret@192.168.1.20:554/stream1";
        let stored = crate::crypto::encrypt_secret(&key, url);
        assert!(!stored.contains("s3cret"));
        assert_eq!(super::open_url(&key, &stored), url);
        assert_eq!(super::open_url(&key, url), url, "a URL saved before encryption still reads");
        assert_eq!(super::open_url(&[8u8; 32], &stored), stored, "a key that can't open it leaves it as stored");
    }
}
