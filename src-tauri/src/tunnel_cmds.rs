//! Local stream-info command.
//!
//! This file used to own the cloudflared tunnel lifecycle
//! (`start_tunnel` / `stop_tunnel` / `get_tunnel_status`). Cloudflare was
//! REMOVED entirely (2026-07-28): remote access is Tailscale Funnel, which
//! `share_cmds` reaches through `tailscale::ensure_public_url`. The tunnel
//! commands had no frontend caller left, and trycloudflare quick tunnels were
//! never ToS-compliant for continuous video anyway.

use std::sync::Arc;

use tauri::State;

use crate::{AppState, StreamInfo};

/// v11 minimal stream-info — local host + port + token. Used by the desktop
/// UI so CameraView can build `<img src="http://localhost:{port}/snapshot…">`
/// URLs without crashing into port 0. The pre-v11 version returned QR data,
/// IPv6, and a public IP for the mobile pairing flow — all gone with the
/// remote-access cut.
#[tauri::command]
pub async fn get_stream_info(state: State<'_, Arc<AppState>>) -> Result<StreamInfo, String> {
    let port  = state.settings.read().await.stream_port;
    let token = state.auth_token.read().await.clone();
    let base  = format!("http://127.0.0.1:{port}");
    Ok(StreamInfo {
        local_ip:       "127.0.0.1".to_string(),
        public_ip:      None,
        ipv6:           None,
        port,
        url:            base.clone(),
        url_with_token: format!("{base}/?token={token}"),
        auth_token:     token,
        qr_data_url:    String::new(),
    })
}
