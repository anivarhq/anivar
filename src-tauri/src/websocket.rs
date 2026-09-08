//! Bidirectional MJPEG WebSocket endpoint — desktop publisher pushes frames, mobile subscribers receive them. Handles per-IP rate limiting, session bookkeeping, kick-by-id.

use std::sync::Arc;

use axum::body::Body;
use axum::extract::{
    ws::{Message, WebSocket, WebSocketUpgrade}, State as AxumState,
};
use axum::http::{Request, StatusCode};
use axum::response::{IntoResponse, Response};
use tauri::{Emitter, Manager};
use tokio::sync::broadcast;
use uuid::Uuid;

use crate::{AppState, ClientSession, StreamState, constant_time_eq};


pub(crate) async fn ws_stream(
    ws: WebSocketUpgrade,
    AxumState(s): AxumState<StreamState>,
    req: Request<Body>,
) -> Response {
    // Defense-in-depth: verify token explicitly before committing to the WS upgrade.
    // The middleware already checked, but WebSocket upgrades are high-value targets.
    let current_token = s.auth_token.read().await.clone();
    let query = req.uri().query().unwrap_or("");
    let token_ok = query.split('&').any(|kv| {
        let mut it = kv.splitn(2, '=');
        let k = it.next().unwrap_or("");
        let v = it.next().unwrap_or("");
        k == "token" && constant_time_eq(v.as_bytes(), current_token.as_bytes())
    }) || req.headers()
        .get("Authorization")
        .and_then(|v| v.to_str().ok())
        .map(|v| {
            let expected = format!("Bearer {}", current_token);
            constant_time_eq(v.as_bytes(), expected.as_bytes())
        })
        .unwrap_or(false);

    if !token_ok {
        return (StatusCode::UNAUTHORIZED, "Unauthorized").into_response();
    }

    let query = req.uri().query().unwrap_or("");
    let cam_id = query.split('&')
        .find_map(|kv| { let mut it = kv.splitn(2,'='); if it.next()? == "cam" { it.next()?.parse::<usize>().ok() } else { None } })
        .unwrap_or(0).min(15);
    // Optional per-connection quality override (0 = use server default)
    let quality_override: Option<u8> = query.split('&')
        .find_map(|kv| { let mut it = kv.splitn(2,'='); if it.next()? == "quality" { it.next()?.parse::<u8>().ok() } else { None } });

    // Client IP: prefer X-Forwarded-For (Cloudflare/reverse proxy) over direct connection
    let client_ip = req.headers()
        .get("x-forwarded-for")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.split(',').next())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "local".to_string());

    ws.on_upgrade(move |socket| handle_ws(socket, s, cam_id, quality_override, client_ip))
}

pub(crate) async fn handle_ws(mut socket: WebSocket, s: StreamState, cam_id: usize, quality_override: Option<u8>, client_ip: String) {
    let mut frame_rx = s.frame_txs[cam_id].subscribe();
    let mut cam_rx = s.camera_state_tx.subscribe();
    let mut revoke_rx = s.revoke_rx.clone();
    let mut bytes_sent: u64 = 0;

    // Assign a stable UUID for this connection — used for targeted disconnect
    let session_id = Uuid::new_v4().to_string();
    let connected_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Kick channel — sending () forces this handler to close the socket
    let (kick_tx, mut kick_rx) = tokio::sync::oneshot::channel::<()>();

    // Register session in shared maps
    s.client_sessions.write().await.insert(session_id.clone(), ClientSession {
        id: session_id.clone(),
        ip: client_ip,
        connected_at,
        cam_id,
    });
    s.kick_txs.lock().await.insert(session_id.clone(), kick_tx);

    // Track connected viewers — emit event to desktop UI
    let count = s.connected_clients.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    emit_client_list(&s, count).await;

    // Send current camera state immediately on connect
    let initial_active = *s.camera_active.read().await;

    // If this is the first viewer and no camera is running, nudge the desktop user
    if count == 1 && !initial_active {
        s.app_handle.emit("viewer:no_camera", serde_json::json!({
            "message": "Phone connected — start a camera on the Live tab to stream video"
        })).ok();
    }
    let init_msg = format!(r#"{{"type":"camera_state","active":{}}}"#, initial_active);
    if socket.send(Message::Text(init_msg)).await.is_err() {
        cleanup_session(&s, &session_id).await;
        return;
    }

    // Send camera info so mobile can build smart camera selector with names
    if let Some(app_state) = s.app_handle.try_state::<Arc<AppState>>() {
        let db  = &app_state.db;
        let settings = app_state.settings.read().await;
        let cam_name_0 = settings.camera_name.clone();
        drop(settings);

        // Load named configs from DB
        let rows: Vec<(i64, String, i64)> = sqlx::query_as(
            "SELECT cam_id, name, enabled FROM camera_configs WHERE enabled=1 ORDER BY cam_id ASC"
        ).fetch_all(db).await.unwrap_or_default();

        let mut cams: Vec<serde_json::Value> = rows.into_iter().map(|(id, name, _)| {
            let display_name = if !name.is_empty() { name }
                else if id == 0 && !cam_name_0.is_empty() { cam_name_0.clone() }
                else { format!("Camera {}", id + 1) };
            serde_json::json!({ "id": id, "name": display_name })
        }).collect();

        // Always include cam 0 as fallback
        if cams.is_empty() {
            cams.push(serde_json::json!({
                "id": 0,
                "name": if cam_name_0.is_empty() { "Camera 1".to_string() } else { cam_name_0 }
            }));
        }

        let msg = serde_json::json!({ "type": "cameras_info", "cameras": cams }).to_string();
        socket.send(Message::Text(msg)).await.ok();
    }

    loop {
        tokio::select! {
            // biased: poll in declaration order when multiple branches are ready.
            // This guarantees camera-state messages and incoming commands are never
            // starved behind a flood of video frames on slow tunnel connections.
            biased;

            // Kicked by desktop operator → close cleanly
            _ = &mut kick_rx => {
                socket.send(Message::Close(Some(axum::extract::ws::CloseFrame {
                    code: 4403,
                    reason: std::borrow::Cow::Borrowed("disconnected_by_operator"),
                }))).await.ok();
                break;
            }

            // Token revoked → close with 4401 so the client knows to re-pair
            _ = revoke_rx.changed() => {
                socket.send(Message::Close(Some(axum::extract::ws::CloseFrame {
                    code: 4401,
                    reason: std::borrow::Cow::Borrowed("token_revoked"),
                }))).await.ok();
                break;
            }

            // Camera state change → phone  (highest priority)
            result = cam_rx.recv() => {
                match result {
                    Ok(active) => {
                        let msg = format!(r#"{{"type":"camera_state","active":{}}}"#, active);
                        if socket.send(Message::Text(msg)).await.is_err() { break; }
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                    Err(broadcast::error::RecvError::Lagged(_)) => {}
                }
            }

            // Phone → control command  (second priority — ack immediately)
            result = socket.recv() => {
                match result {
                    Some(Ok(Message::Text(text))) => {
                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) {
                            match v["type"].as_str() {
                                Some("start_camera") => {
                                    // Ack immediately so the phone shows "Starting…" without delay
                                    socket.send(Message::Text(
                                        r#"{"type":"cmd_ack","cmd":"start_camera"}"#.to_string()
                                    )).await.ok();
                                    s.app_handle.emit("remote:start_camera", ()).ok();
                                }
                                Some("stop_camera") => {
                                    socket.send(Message::Text(
                                        r#"{"type":"cmd_ack","cmd":"stop_camera"}"#.to_string()
                                    )).await.ok();
                                    s.app_handle.emit("remote:stop_camera", ()).ok();
                                }
                                Some("ping") => {
                                    socket.send(Message::Text(r#"{"type":"pong"}"#.to_string())).await.ok();
                                }
                                _ => {}
                            }
                        }
                    }
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Err(_)) => break,
                    _ => {}
                }
            }

            // Video frame → phone  (lowest priority — dropped when connection is saturated)
            result = frame_rx.recv() => {
                match result {
                    Ok(frame) => {
                        let payload = if let Some(q) = quality_override {
                            // Re-encode at requested quality (only when different from broadcast quality)
                            let server_q = s.app_handle
                                .try_state::<Arc<AppState>>()
                                .and_then(|st| Some(st.settings.try_read().ok()?.stream_quality))
                                .unwrap_or(60);
                            if q != server_q && q > 0 && q <= 100 {
                                // Decode and re-encode at client quality
                                if let Ok(img) = image::load_from_memory(&frame) {
                                    let mut buf = Vec::with_capacity(frame.len());
                                    let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, q);
                                    if enc.encode_image(&img).is_ok() {
                                        std::sync::Arc::new(buf)
                                    } else { frame.clone() }
                                } else { frame.clone() }
                            } else { frame.clone() }
                        } else { frame.clone() };

                        let len = payload.len() as u64;
                        if socket.send(Message::Binary(payload.as_ref().clone())).await.is_err() {
                            break;
                        }
                        bytes_sent += len;
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                    Err(broadcast::error::RecvError::Lagged(_)) => {} // skip stale frames
                }
            }
        }
    }
    tracing::debug!("WS cam{} closed — sent {:.1}MB", cam_id, bytes_sent as f64 / 1_048_576.0);
    cleanup_session(&s, &session_id).await;
}

pub(crate) async fn cleanup_session(s: &StreamState, session_id: &str) {
    s.client_sessions.write().await.remove(session_id);
    s.kick_txs.lock().await.remove(session_id);
    let count = s.connected_clients.fetch_sub(1, std::sync::atomic::Ordering::Relaxed).saturating_sub(1);
    emit_client_list(s, count).await;
}

pub(crate) async fn emit_client_list(s: &StreamState, count: usize) {
    let sessions = s.client_sessions.read().await;
    let clients: Vec<&ClientSession> = sessions.values().collect();
    s.app_handle.emit("stream:clients", serde_json::json!({
        "count": count,
        "clients": clients,
    })).ok();
}
