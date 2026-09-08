//! HTTP server boot — token auth middleware, security headers, axum `Router` wiring, port-firewall opening, listener spawn.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use axum::{
    body::Body,
    extract::State as AxumState,
    http::{Request, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    routing::get,
    Router,
};
use tokio::sync::{broadcast, watch, Mutex, RwLock};
use tower_http::cors::CorsLayer;

use crate::{
    ClientSession, SignalRoom, StreamState, constant_time_eq,
};
use crate::footage::{footage_clip, footage_list, footage_stream, footage_thumbnail, ping};
use crate::hls::hls_serve;
use crate::http_handlers::{cam_proxy, login_with_password, mjpeg_stream, redeem, share_clip, share_live, snapshot, webrtc_whep};
use crate::nvr_stream::{nvr_concat_stream, nvr_export_stream, nvr_seek_stream, nvr_stream};
/// Token auth middleware with per-IP rate limiting on failures.
pub(crate) async fn require_token(
    AxumState(ss): AxumState<StreamState>,
    req: Request<Body>,
    next: Next,
) -> Response {
    let path = req.uri().path();

    // Public paths that bypass auth entirely.
    // "/" serves only the static PWA shell HTML — no sensitive data.
    // Sensitive resources (/stream, /ws, /snapshot) remain fully protected below.
    // OPTIONS preflight requests must bypass auth — the CORS layer handles them.
    if req.method() == axum::http::Method::OPTIONS {
        return next.run(req).await;
    }

    if path == "/login"
        // v11 share routes carry their own HMAC token + HttpOnly cookie auth,
        // they don't use the desktop auth_token at all.
        || path == "/redeem"
        || path.starts_with("/clips/")
        || path.starts_with("/live/")
        {
        return next.run(req).await;
    }

    // Extract caller IP for rate limiting (ConnectInfo not available behind tunnel,
    // so fall back to X-Forwarded-For then a fixed sentinel)
    let ip = req.headers()
        .get("x-forwarded-for")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.split(',').next())
        .unwrap_or("unknown")
        .trim()
        .to_string();

    // Rate-limit check: max 15 failed attempts per IP per 60 seconds
    {
        let mut map = ss.failed_auth.lock().await;
        let now = Instant::now();
        let window = std::time::Duration::from_secs(60);
        let attempts = map.entry(ip.clone()).or_default();
        attempts.retain(|t| now.duration_since(*t) < window);
        if attempts.len() >= 15 {
            return (StatusCode::TOO_MANY_REQUESTS, "Too many failed attempts — try again later").into_response();
        }
    }

    let query = req.uri().query().unwrap_or("");
    let current_token = ss.auth_token.read().await.clone();

    let token_in_query = query.split('&').any(|kv| {
        let mut it = kv.splitn(2, '=');
        let k = it.next().unwrap_or("");
        let v = it.next().unwrap_or("");
        k == "token" && constant_time_eq(v.as_bytes(), current_token.as_bytes())
    });

    let token_in_header = req
        .headers()
        .get("Authorization")
        .and_then(|v| v.to_str().ok())
        .map(|v| {
            let expected = format!("Bearer {}", current_token);
            constant_time_eq(v.as_bytes(), expected.as_bytes())
        })
        .unwrap_or(false);

    if token_in_query || token_in_header {
        // Success — clear any recorded failures for this IP
        ss.failed_auth.lock().await.remove(&ip);
        next.run(req).await
    } else {
        // Record the failure
        ss.failed_auth.lock().await
            .entry(ip)
            .or_default()
            .push(Instant::now());
        (StatusCode::UNAUTHORIZED, "Access denied — use the full link from Anivar").into_response()
    }
}

pub(crate) async fn security_headers(req: axum::extract::Request, next: axum::middleware::Next) -> impl axum::response::IntoResponse {
    let mut resp = next.run(req).await;
    let h = resp.headers_mut();
    use axum::http::HeaderValue as HV;
    h.insert("X-Content-Type-Options",  HV::from_static("nosniff"));
    h.insert("X-Frame-Options",         HV::from_static("DENY"));
    h.insert("X-XSS-Protection",        HV::from_static("1; mode=block"));
    h.insert("Referrer-Policy",         HV::from_static("strict-origin-when-cross-origin"));
    h.insert("Permissions-Policy",      HV::from_static("geolocation=()"));
    h.insert("Content-Security-Policy", HV::from_static(
        "default-src 'self' data: blob:; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob: http: https:; connect-src 'self' ws: wss: http: https:; media-src 'self' blob:;"
    ));
    resp
}

pub fn start_http_server(
    frame_txs: Arc<Vec<broadcast::Sender<Arc<Vec<u8>>>>>,
    port: u16,
    auth_token: Arc<RwLock<String>>,
    app_handle: tauri::AppHandle,
    camera_active: Arc<RwLock<bool>>,
    camera_state_tx: broadcast::Sender<bool>,
    db: sqlx::SqlitePool,
    data_dir: PathBuf,
    revoke_rx: watch::Receiver<u64>,
    client_sessions: Arc<tokio::sync::RwLock<HashMap<String, ClientSession>>>,
    kick_txs: Arc<tokio::sync::Mutex<HashMap<String, tokio::sync::oneshot::Sender<()>>>>,
    lan_access: bool,
) {
    let ss = StreamState {
        frame_txs,
        camera_state_tx,
        camera_active,
        auth_token,
        app_handle,
        connected_clients: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        failed_auth: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
        signal_room: Arc::new(Mutex::new(SignalRoom { host: None, viewer: None })),
        db,
        data_dir,
        revoke_rx,
        client_sessions,
        kick_txs,
    };

    // Allow the Capacitor mobile app (capacitor://localhost, https://localhost) and
    // the local browser (http://localhost during dev) to reach the API.
    // The auth middleware still protects all sensitive endpoints.
    use tower_http::cors::AllowOrigin;
    use axum::http::HeaderValue;
    let cors = CorsLayer::new()
        .allow_origin(AllowOrigin::predicate(|origin: &HeaderValue, _req| {
            let s = origin.to_str().unwrap_or("");
            // Allow any localhost origin (covers Vite dev :5174, Tauri, Capacitor)
            s.starts_with("http://localhost")
                || s.starts_with("https://localhost")
                || s.starts_with("tauri://localhost")
                || s.starts_with("capacitor://localhost")
        }))
        .allow_methods([
            axum::http::Method::GET,
            axum::http::Method::POST,
            axum::http::Method::OPTIONS,
        ])
        .allow_headers([
            axum::http::header::AUTHORIZATION,
            axum::http::header::CONTENT_TYPE,
        ]);

    // v11: the embedded HTTP server is purely a backing for the tunnel-based
    // share routes (live/clip) added later in Step 3c. The old mobile-PWA
    // viewer, MJPEG `/stream`, WebSocket camera relay, libp2p signaling, and
    // PWA manifest/service-worker routes are gone — replaced by a one-shot
    // /redeem flow.
    // SHORT request/response routes get a hard 30s timeout so one wedged
    // handler (dead camera probe, stuck DB call) can never hold a connection
    // hostage. Streaming routes (MJPEG live, HLS, VOD, clip ranges) are
    // long-lived BY DESIGN and must not sit under a timeout.
    let short_routes = Router::new()
        .route("/ping", get(ping))
        .route("/login", axum::routing::post(login_with_password))
        .route("/snapshot", get(snapshot))
        .route("/footage", get(footage_list))
        .route("/clip-start", get(crate::footage::clip_start_meta))
        .route("/footage/:id/thumbnail", get(footage_thumbnail))
        // People-section crops (URL-served, Chromium-cacheable — replaces the
        // inline-base64 IPC payloads that froze the People tab).
        .route("/face/:id/crop", get(crate::footage::face_crop))
        .route("/body/:id/crop", get(crate::footage::body_crop))
        // WHEP SDP exchange → go2rtc (sub-second WebRTC live view; media flows
        // peer-to-peer after this one authed request/response).
        .route("/webrtc/:cam", axum::routing::post(webrtc_whep))
        // v11 share routes — public to the tunnel; HMAC token + HttpOnly cookie auth.
        .route("/redeem", get(redeem))
        .layer(tower_http::timeout::TimeoutLayer::new(std::time::Duration::from_secs(30)));
    let app = Router::new()
        .route("/cam-proxy", get(cam_proxy))
        .route("/stream", get(mjpeg_stream))
        .route("/footage/:id/clip", get(footage_clip))
        .route("/footage/:id/stream", get(footage_stream))
        .route("/nvr-stream",  get(nvr_stream))
        .route("/nvr-seek",    get(nvr_seek_stream))
        .route("/nvr-concat",  get(nvr_concat_stream))
        // v13: bounded export endpoint — same shape as /nvr-concat but takes
        // an `end` param and tags the response Content-Disposition: attachment.
        .route("/nvr-export",  get(nvr_export_stream))
        // HLS VOD playback (mature NVRs model): playlist over indexed segments +
        // per-segment copy-remux to TS. The player stitches seams — fixes the
        // pause/play stutter of glued copy-concat streams.
        .route("/nvr-vod/:cam/playlist.m3u8", get(crate::nvr_vod::nvr_vod_playlist))
        .route("/nvr-vod/seg/:file", get(crate::nvr_vod::nvr_vod_segment))
        .route("/nvr-preview/:file", get(crate::nvr_preview::nvr_preview_file))
        .route("/hls/:file", get(hls_serve))
        .route("/clips/:event_id", get(share_clip))
        .route("/live/:cam_id",    get(share_live))
        .merge(short_routes)
        .layer(axum::middleware::from_fn_with_state(ss.clone(), require_token))
        .layer(axum::middleware::from_fn(security_headers))
        .layer(cors)
        .with_state(ss);

    // v11: the libp2p WS firewall opener is gone with the p2p module. The
    // embedded HTTP server now serves only the share routes behind the on-
    // demand Tailscale Funnel — there's no LAN-direct path to open the
    // Windows firewall for.

    tokio::spawn(async move {
        // LAN access (default): reachable from phones on the network. Off →
        // loopback only, so nothing on the network can even connect.
        let ip: [u8; 4] = if lan_access { [0, 0, 0, 0] } else { [127, 0, 0, 1] };
        let addr = SocketAddr::from((ip, port));

        // ALSO listen on the IPv6 side. WebView2 (and curl) resolve `localhost`
        // to ::1 FIRST; with no IPv6 listener that SYN goes nowhere and the
        // client sits out its Happy-Eyeballs fallback delay (~300ms in
        // Chromium) before retrying 127.0.0.1 — a hidden per-connection tax
        // that visibly stuttered clip playback (the media demuxer opens fresh
        // connections for sparse A/V range reads). Best-effort: if IPv6 is
        // disabled on the machine, the IPv4 listener still carries everything.
        let v6 = if lan_access { std::net::Ipv6Addr::UNSPECIFIED } else { std::net::Ipv6Addr::LOCALHOST };
        let addr6 = SocketAddr::from((v6, port));
        let app6 = app.clone();
        tokio::spawn(async move {
            match loop_bind_reuseaddr(addr6).await {
                Some(l6) => {
                    tracing::info!("Stream server listening on {} (IPv6)", addr6);
                    if let Err(e) = axum::serve(l6, app6).await {
                        tracing::error!("IPv6 stream server error: {}", e);
                    }
                }
                None => tracing::warn!(
                    "IPv6 bind {} failed — localhost clients will pay the Happy-Eyeballs fallback delay", addr6
                ),
            }
        });
        // RESILIENT BIND. The previous code used a plain TcpListener::bind with no
        // SO_REUSEADDR and gave up on the first error — so reopening the app while
        // the prior instance's /stream sockets were still in TIME_WAIT failed with
        // WSAEADDRINUSE (10048) and the stream server never came up → BLANK camera
        // (and dead clips/snapshots) even though capture was fine. Fix: set
        // SO_REUSEADDR and retry for a short window so a fast close→reopen recovers.
        let listener = loop_bind_reuseaddr(addr).await;
        match listener {
            Some(listener) => {
                tracing::info!("Stream server listening on {}", addr);
                if let Err(e) = axum::serve(listener, app).await {
                    tracing::error!("Stream server error: {}", e);
                }
            }
            None => {
                tracing::error!("Failed to bind port {} after retries — stream server DOWN", addr);
            }
        }
    });
}

/// Bind `addr` with SO_REUSEADDR set, retrying briefly while a prior instance's
/// sockets drain from TIME_WAIT. Returns a tokio listener ready for axum::serve.
async fn loop_bind_reuseaddr(addr: SocketAddr) -> Option<tokio::net::TcpListener> {
    use socket2::{Domain, Protocol, Socket, Type};
    for attempt in 0..20u32 {
        let try_bind = || -> std::io::Result<tokio::net::TcpListener> {
            let domain = if addr.is_ipv6() { Domain::IPV6 } else { Domain::IPV4 };
            let sock = Socket::new(domain, Type::STREAM, Some(Protocol::TCP))?;
            if addr.is_ipv6() {
                // Keep this socket v6-only so it can coexist with the separate
                // IPv4 listener on the same port (no dual-stack overlap).
                sock.set_only_v6(true)?;
            }
            sock.set_reuse_address(true)?;     // rebind over TIME_WAIT
            sock.set_nonblocking(true)?;
            sock.bind(&addr.into())?;
            sock.listen(1024)?;
            tokio::net::TcpListener::from_std(std::net::TcpListener::from(sock))
        };
        match try_bind() {
            Ok(l) => return Some(l),
            Err(e) => {
                if attempt == 0 {
                    tracing::warn!("Port {} busy ({}); retrying while it frees…", addr, e);
                }
                tokio::time::sleep(std::time::Duration::from_millis(750)).await;
            }
        }
    }
    None
}
