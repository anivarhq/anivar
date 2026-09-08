//! v11 HTTP handlers for the embedded Axum server.
//!
//! After the v11 channels-first pivot the server's job shrank to the
//! routes the on-demand share flow actually uses:
//!
//!   • `/cam-proxy`   — MJPEG/HTTP camera proxy used by the desktop UI
//!                      itself (bypasses Tauri's CSP `img-src` restriction).
//!   • `/login`       — password → token exchange for hosts that opted in
//!                      to password auth on the desktop UI.
//!   • `/snapshot`    — one-frame JPEG for the local UI's preview screen.
//!
//! The MJPEG live stream / mobile PWA viewer / libp2p discovery / PWA
//! manifest+sw / public-IP probe were all part of the deleted remote-access
//! scaffolding; the share-link routes (`/redeem`, `/clips/*`, `/live/*`)
//! land in v11 Step 3c and live in a new module rather than here.

use std::sync::Arc;

use axum::body::Body;
use axum::extract::State as AxumState;
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use tauri::Manager;

use crate::{AppState, StreamState, constant_time_eq};


/// Camera proxy — fetches a remote MJPEG / HTTP camera URL and streams it
/// back through the same origin as the Tauri UI so it can be rendered in
/// `<img>` / `<video>` tags without tripping the WebView CSP.
///
/// This bypasses Tauri's CSP which restricts img-src to localhost only.
/// Usage: GET /cam-proxy?url=http://192.168.1.123:8080/video&token=AUTH
/// True when an address is on a private/local network — the ONLY destinations
/// `cam_proxy` may reach. IPv4: RFC1918 + loopback + link-local. IPv6:
/// loopback + ULA (fc00::/7) + link-local (fe80::/10).
fn is_private_ip(ip: std::net::IpAddr) -> bool {
    match ip {
        std::net::IpAddr::V4(v4) => v4.is_private() || v4.is_loopback() || v4.is_link_local(),
        std::net::IpAddr::V6(v6) => v6.is_loopback()
            || (v6.segments()[0] & 0xfe00) == 0xfc00   // ULA fc00::/7
            || (v6.segments()[0] & 0xffc0) == 0xfe80,  // link-local fe80::/10
    }
}

/// SSRF guard for cam-proxy: the endpoint exists to relay LAN camera streams,
/// so the destination must be http(s) to a PRIVATE address (or localhost /
/// `.local` mDNS). Hostnames are resolved and EVERY resolved address must be
/// private — a name with any public A/AAAA record is rejected (blocks
/// DNS-rebinding a "camera" hostname to the public internet). Without this, a
/// leaked token + the public Funnel turned the NVR into an open proxy.
async fn cam_url_is_local(raw: &str) -> bool {
    let Ok(parsed) = reqwest::Url::parse(raw) else { return false };
    if !matches!(parsed.scheme(), "http" | "https") { return false; }
    // Own the host string so no borrow of `parsed` crosses the resolver await.
    let bare: String = match parsed.host_str() {
        Some(h) => h.trim_start_matches('[').trim_end_matches(']').to_string(),
        None => return false,
    };
    if let Ok(ip) = bare.parse::<std::net::IpAddr>() {
        return is_private_ip(ip);
    }
    let hl = bare.to_ascii_lowercase();
    if hl == "localhost" || hl.ends_with(".local") { return true; } // mDNS = LAN by construction
    let port = parsed.port_or_known_default().unwrap_or(80);
    // Owned (String, u16): the resolver future must not borrow locals.
    match tokio::net::lookup_host((bare.clone(), port)).await {
        Ok(addrs) => {
            let mut any = false;
            for a in addrs {
                any = true;
                if !is_private_ip(a.ip()) { return false; }
            }
            any
        }
        Err(_) => false,
    }
}

/// Downscale + recompress a broadcast JPEG for the SHARED live link. The source
/// frames are 1280×720 q3 (~60 KB) — required for snapshots/enrollment, but far
/// too heavy for MJPEG (no inter-frame compression) over a phone/tunnel link:
/// they queue in TCP and replay in SLOW-MOTION. 640×360 @ q45 is ~12–18 KB, so
/// ~12 fps fits in ~1.5 Mbit/s and the stream stays LIVE. Returns None on any
/// decode/encode error — the caller falls back to the original bytes so a bad
/// transcode can never blank the stream.
fn downscale_jpeg_for_wan(jpeg: &[u8]) -> Option<Vec<u8>> {
    let img = image::load_from_memory(jpeg).ok()?;
    // Triangle = fast bilinear; fine for a small live glance. Only shrinks
    // (mature NVRs' WAN-live sizing); already-small frames pass through unchanged.
    let scaled = if img.width() > 640 || img.height() > 360 {
        img.resize(640, 360, image::imageops::FilterType::Triangle)
    } else { img };
    let rgb = scaled.to_rgb8();
    let mut out = Vec::with_capacity(20 * 1024);
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 45)
        .encode_image(&image::DynamicImage::ImageRgb8(rgb)).ok()?;
    Some(out)
}

pub(crate) async fn cam_proxy(
    AxumState(s): AxumState<StreamState>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Response {
    // Auth check
    let token = params.get("token").map(|t| t.as_str()).unwrap_or("");
    let valid  = constant_time_eq(token.as_bytes(), s.auth_token.read().await.as_bytes());
    if !valid {
        return (StatusCode::UNAUTHORIZED, "Unauthorized").into_response();
    }

    let cam_url = match params.get("url") {
        Some(u) if !u.is_empty() => u.clone(),
        _ => return (StatusCode::BAD_REQUEST, "Missing url parameter").into_response(),
    };
    // SSRF guard — LAN cameras only (see cam_url_is_local).
    if !cam_url_is_local(&cam_url).await {
        return (StatusCode::FORBIDDEN,
            "cam-proxy only reaches local-network cameras (private IPs, localhost, or .local names)"
        ).into_response();
    }
    let cam_user = params.get("user").cloned().unwrap_or_default();
    let cam_pass = params.get("pass").cloned().unwrap_or_default();

    // Fetch the remote MJPEG stream and pipe it through.
    // Use connect_timeout only — NOT a read/response timeout.
    // MJPEG streams are long-lived multipart HTTP responses; any response
    // timeout would kill the stream after N seconds.
    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(10))
        .pool_idle_timeout(None)
        .build()
        .unwrap_or_default();

    let mut req = client.get(&cam_url);
    if !cam_user.is_empty() {
        req = req.basic_auth(&cam_user, if cam_pass.is_empty() { None } else { Some(&cam_pass) });
    }

    match req.send().await {
        Ok(resp) => {
            let status = resp.status();
            let ct     = resp.headers().get(reqwest::header::CONTENT_TYPE)
                            .and_then(|v| v.to_str().ok())
                            .unwrap_or("application/octet-stream")
                            .to_string();
            use futures::StreamExt;
            let stream = resp.bytes_stream().map(|r| r.map_err(std::io::Error::other));
            let body   = Body::from_stream(stream);
            axum::response::Response::builder()
                .status(status.as_u16())
                .header("Content-Type", ct)
                .header("Cache-Control", "no-store")
                .header("Access-Control-Allow-Origin", "*")
                .header("Cross-Origin-Resource-Policy", "cross-origin")
                .header("Timing-Allow-Origin", "*")
                .body(body)
                .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
        }
        Err(e) => (StatusCode::BAD_GATEWAY, format!("Cannot connect to camera: {e}")).into_response(),
    }
}

/// Login with username + password — returns the persistent auth token.
/// Hosts that didn't configure password auth get 403 (no fallback path now;
/// pre-v11 the alternative was QR pairing through the mobile PWA, which is
/// gone).
pub(crate) async fn login_with_password(
    AxumState(s): AxumState<StreamState>,
    axum::Json(body): axum::Json<serde_json::Value>,
) -> Response {
    let username = body.get("username").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
    let password = body.get("password").and_then(|v| v.as_str()).unwrap_or("").to_string();

    let app_state = match s.app_handle.try_state::<Arc<AppState>>() {
        Some(st) => st,
        None => return (StatusCode::INTERNAL_SERVER_ERROR, "Server error").into_response(),
    };
    let settings = app_state.settings.read().await;

    if settings.auth_username.is_empty() || settings.auth_password_hash.is_empty() {
        return (StatusCode::FORBIDDEN,
            axum::Json(serde_json::json!({ "error": "Password login not configured." }))
        ).into_response();
    }

    if !constant_time_eq(username.as_bytes(), settings.auth_username.as_bytes()) {
        return (StatusCode::UNAUTHORIZED,
            axum::Json(serde_json::json!({ "error": "Invalid username or password." }))
        ).into_response();
    }

    let provided_hash = sha256_hex(password.as_bytes());
    if !constant_time_eq(provided_hash.as_bytes(), settings.auth_password_hash.as_bytes()) {
        return (StatusCode::UNAUTHORIZED,
            axum::Json(serde_json::json!({ "error": "Invalid username or password." }))
        ).into_response();
    }

    let token = app_state.auth_token.read().await.clone();
    drop(settings);

    (StatusCode::OK,
        axum::Json(serde_json::json!({ "token": token, "device_name": "Anivar" }))
    ).into_response()
}

pub(crate) fn sha256_hex(data: &[u8]) -> String {
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("{:x}", hasher.finalize())
}

/// WHEP SDP exchange proxy → go2rtc (sub-second WebRTC live view). The WebView
/// posts its SDP offer here (token-authed, same origin as every other stream
/// route); we forward to go2rtc's loopback API and return the SDP answer.
/// Media then flows peer-to-peer — this route only does the handshake.
pub(crate) async fn webrtc_whep(
    axum::extract::Path(cam): axum::extract::Path<u8>,
    body: String,
) -> Response {
    match crate::go2rtc::whep_exchange(cam, body).await {
        Ok(answer) => (
            StatusCode::CREATED,
            [(axum::http::header::CONTENT_TYPE, "application/sdp")],
            answer,
        ).into_response(),
        Err(e) => (StatusCode::SERVICE_UNAVAILABLE, e.to_string()).into_response(),
    }
}

/// Snapshot — single JPEG of the requested camera (`?cam=N`, default 0) for the
/// local preview UI (e.g. the mask/zone editor background).
pub(crate) async fn snapshot(
    AxumState(s): AxumState<StreamState>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Response {
    let cam: usize = params.get("cam").and_then(|v| v.parse().ok()).unwrap_or(0);
    let cam = cam.min(s.frame_txs.len().saturating_sub(1));
    let mut rx = s.frame_txs[cam].subscribe();
    match rx.recv().await {
        Ok(frame) => {
            let mut headers = HeaderMap::new();
            headers.insert("Content-Type", HeaderValue::from_static("image/jpeg"));
            headers.insert("Cache-Control", HeaderValue::from_static("no-cache"));
            (StatusCode::OK, headers, Body::from(frame.as_ref().clone())).into_response()
        }
        _ => StatusCode::SERVICE_UNAVAILABLE.into_response(),
    }
}

/// GET /stream?cam=N — LOCAL continuous multipart MJPEG of a camera's broadcast
/// frames. NATIVE (nokhwa, server-side) cameras publish to `frame_txs[cam]`; the
/// browser USB path renders getUserMedia directly, so this endpoint is what gives a
/// native-source camera a live picture in the app. Token-protected by the route
/// middleware (same as /snapshot). Seeds with the latest cached frame for an instant
/// first picture, and keep-alives on an idle camera so the connection never hangs.
pub(crate) async fn mjpeg_stream(
    AxumState(s): AxumState<StreamState>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Response {
    let cam: usize = params.get("cam").and_then(|v| v.parse().ok()).unwrap_or(0);
    if cam >= s.frame_txs.len() { return StatusCode::BAD_REQUEST.into_response(); }
    let mut rx = s.frame_txs[cam].subscribe();
    let seed_frame: Option<Vec<u8>> = match s.app_handle.try_state::<Arc<AppState>>() {
        Some(st) => st.latest_frames.read().await.get(&(cam as u8)).cloned(),
        None => None,
    };
    let boundary = "frame";
    let content_type = format!("multipart/x-mixed-replace; boundary={boundary}");
    let boundary_owned = boundary.to_string();
    let stream = async_stream::try_stream! {
        let mut last: Option<Vec<u8>> = seed_frame.clone();
        if let Some(frame) = &seed_frame {
            let header = format!("--{boundary_owned}\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n", frame.len());
            yield axum::body::Bytes::from(header.into_bytes());
            yield axum::body::Bytes::from(frame.clone());
            yield axum::body::Bytes::from_static(b"\r\n");
        }
        loop {
            match tokio::time::timeout(Duration::from_secs(10), rx.recv()).await {
                Ok(Ok(frame)) => {
                    let header = format!("--{boundary_owned}\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n", frame.len());
                    yield axum::body::Bytes::from(header.into_bytes());
                    yield axum::body::Bytes::from(frame.as_ref().clone());
                    yield axum::body::Bytes::from_static(b"\r\n");
                    last = Some(frame.as_ref().clone());
                }
                Ok(Err(tokio::sync::broadcast::error::RecvError::Lagged(_))) => continue,
                Ok(Err(_)) => break,
                Err(_) => {
                    if let Some(frame) = &last {
                        let header = format!("--{boundary_owned}\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n", frame.len());
                        yield axum::body::Bytes::from(header.into_bytes());
                        yield axum::body::Bytes::from(frame.clone());
                        yield axum::body::Bytes::from_static(b"\r\n");
                    }
                }
            }
        }
    };
    let body = Body::from_stream(
        Box::pin(stream) as std::pin::Pin<Box<dyn futures::Stream<Item = Result<axum::body::Bytes, std::io::Error>> + Send>>
    );
    axum::response::Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", content_type)
        .header("Cache-Control", "no-cache")
        .body(body)
        .unwrap()
}

// ── v11 share-link routes ───────────────────────────────────────────────────
//
// Public-facing routes served via the on-demand Tailscale Funnel. Each route
// is HMAC-token-gated AND rate-limited per remote IP. The /redeem step also
// sets an HttpOnly cookie so the underlying /clips/* and /live/* URLs never
// expose the share token in browser history or referrers.
//
// The HttpOnly cookie carries a fresh per-cookie token that's a one-way hash
// of the share-token. It binds the session to the resource (cam_id / event_id)
// without echoing the original share-token in the URL bar.

use axum::extract::Path;
use std::time::Duration;

use crate::share_security::{verify_share_token, SharePayload};

/// Static once-per-process rate limiters. Reasonably sized for personal use;
/// adjust if you front this with a CDN.
static REDEEM_LIMITER: std::sync::OnceLock<crate::share_security::RateLimiter> = std::sync::OnceLock::new();
static CLIP_LIMITER:   std::sync::OnceLock<crate::share_security::RateLimiter> = std::sync::OnceLock::new();
static LIVE_LIMITER:   std::sync::OnceLock<crate::share_security::RateLimiter> = std::sync::OnceLock::new();

fn redeem_limiter() -> &'static crate::share_security::RateLimiter {
    REDEEM_LIMITER.get_or_init(|| crate::share_security::RateLimiter::new(10, Duration::from_secs(60)))
}
fn clip_limiter() -> &'static crate::share_security::RateLimiter {
    CLIP_LIMITER.get_or_init(|| crate::share_security::RateLimiter::new(60, Duration::from_secs(60)))
}
fn live_limiter() -> &'static crate::share_security::RateLimiter {
    LIVE_LIMITER.get_or_init(|| crate::share_security::RateLimiter::new(120, Duration::from_secs(60)))
}

/// Helper: best-effort client IP extraction. Proxies put the real IP in
/// `cf-connecting-ip`; X-Forwarded-For is the standard fallback.
fn client_ip(headers: &HeaderMap) -> String {
    headers.get("cf-connecting-ip")
        .or_else(|| headers.get("x-forwarded-for"))
        .and_then(|v| v.to_str().ok())
        .map(|s| s.split(',').next().unwrap_or("").trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Derive the HttpOnly cookie value from the share token + resource. The
/// recipient's browser sends this back as proof they redeemed the link;
/// the underlying /clips/* and /live/* routes verify it without ever seeing
/// the original share-token again.
fn cookie_for(master_key: &[u8; 32], share_token: &str, kind: &str, resource_id: &str) -> String {
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    hasher.update(master_key);
    hasher.update(b"\x00cookie\x00");
    hasher.update(share_token.as_bytes());
    hasher.update(b"\x00");
    hasher.update(kind.as_bytes());
    hasher.update(b"\x00");
    hasher.update(resource_id.as_bytes());
    let digest = hasher.finalize();
    B64URL.encode(&digest[..24])
}

use base64::Engine as _;
use base64::engine::general_purpose::URL_SAFE_NO_PAD as B64URL;

/// Viewer-facing error page for the share routes. These URLs are opened by
/// NON-technical recipients on phones — raw strings like "invalid or expired
/// token" (the exact text users reported) explain nothing. One tiny
/// self-contained page, styled inline, that says what happened and what to do.
fn share_error_page(status: StatusCode, title: &str, hint: &str) -> Response {
    let html = format!(
        "<!doctype html><html><head><meta charset=\"utf-8\">\
         <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\
         <title>Anivar</title></head>\
         <body style=\"margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;\
           background:#0b0a10;color:#e8e6f0;font-family:system-ui,-apple-system,Segoe UI,sans-serif\">\
         <div style=\"max-width:420px;padding:36px 28px;text-align:center\">\
           <div style=\"font-size:40px;margin-bottom:14px\">\u{1F4F7}</div>\
           <div style=\"font-size:17px;font-weight:700;margin-bottom:10px\">{title}</div>\
           <div style=\"font-size:13.5px;line-height:1.65;color:#a7a3b8\">{hint}</div>\
         </div></body></html>"
    );
    (
        status,
        [("content-type", "text/html; charset=utf-8"),
         ("cache-control", "no-store, no-cache, must-revalidate")],
        html,
    ).into_response()
}

fn expired_link_page() -> Response {
    share_error_page(
        StatusCode::FORBIDDEN,
        "This link has expired or was revoked",
        "Share links are private and short-lived on purpose. \
         Ask the person who sent it for a fresh one — in Anivar that's one tap \
         (\u{1F4E1} Live or \u{1F517} Share).",
    )
}

/// GET /redeem?t=...  ── verify the share-token, set an HttpOnly cookie, then
/// 302 the browser to the actual content URL. Token never re-appears in the
/// browser history past this hop.
pub(crate) async fn redeem(
    AxumState(s): AxumState<StreamState>,
    headers: HeaderMap,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Response {
    let ip = client_ip(&headers);
    let (ok, retry_after) = redeem_limiter().check(&ip);
    if !ok {
        return share_error_page(
            StatusCode::TOO_MANY_REQUESTS,
            "Too many attempts",
            &format!("Please wait about {retry_after} seconds and open the link again."),
        );
    }

    let app_state = match s.app_handle.try_state::<Arc<AppState>>() {
        Some(st) => st,
        None => return share_error_page(StatusCode::INTERNAL_SERVER_ERROR,
            "Something went wrong", "The camera app hit an internal error — try the link again in a moment."),
    };

    let token = match params.get("t") {
        Some(v) if !v.is_empty() => v.clone(),
        _ => return share_error_page(StatusCode::BAD_REQUEST,
            "This link is incomplete",
            "Part of the link got cut off. Ask the sender to share it again, and open the whole link."),
    };

    let generation = *app_state.share_generation.read().await;
    let payload: SharePayload = match verify_share_token(&app_state.master_key, generation, &token) {
        Some(p) => p,
        None => return expired_link_page(),
    };

    let cookie_value = cookie_for(&app_state.master_key, &token, &payload.kind, &payload.resource_id);
    let target = match payload.kind.as_str() {
        "clip" => format!("/clips/{}.mp4", payload.resource_id),
        "live" => format!("/live/{}.mjpeg",  payload.resource_id),
        _      => return expired_link_page(),
    };
    // SameSite=Lax (NOT Strict): the link is opened from an EXTERNAL app
    // (Telegram/email), so the redeem→resource navigation chain has a cross-site
    // initiator. A Strict cookie is withheld on the redirected /live or /clips
    // request → "missing or invalid session cookie". Lax is still sent on
    // top-level GET navigations (this redirect), while blocking cross-site
    // subresource/POST abuse. Secure (HTTPS tunnel) + HttpOnly + the HMAC token
    // (generation + expiry) keep it safe.
    let max_age = (payload.expires_at - chrono::Utc::now().timestamp()).max(60);
    let cookie = format!(
        "sc_share={value}; Path=/; Max-Age={max_age}; HttpOnly; Secure; SameSite=Lax",
        value = cookie_value,
    );

    axum::response::Response::builder()
        .status(StatusCode::SEE_OTHER)
        .header("location", target)
        .header("set-cookie", cookie)
        .header("cache-control", "no-store, no-cache, must-revalidate")
        .header("referrer-policy", "no-referrer")
        .body(Body::empty())
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
}

/// Check the sc_share cookie against the requested resource. Returns the
/// share-token's derived cookie value the request CLAIMS to have, plus
/// whether it matched the one we'd derive for this `(kind, resource_id)`.
fn check_share_cookie(headers: &HeaderMap, expected: &str) -> bool {
    let cookies = headers.get("cookie").and_then(|v| v.to_str().ok()).unwrap_or("");
    for part in cookies.split(';') {
        let part = part.trim();
        if let Some(value) = part.strip_prefix("sc_share=") {
            return crate::constant_time_eq(value.as_bytes(), expected.as_bytes());
        }
    }
    false
}

/// Helper: re-derive the expected cookie for a `(kind, resource_id)` pair by
/// scanning the active-share list for a matching outstanding share. We don't
/// know which exact share_token the cookie came from, but any valid token
/// against the current generation yields the same cookie when fed into
/// `cookie_for`, so we re-mint each candidate's expected cookie and compare.
async fn expected_cookie_for(
    app_state: &Arc<AppState>,
    kind: &str,
    resource_id: &str,
) -> Option<String> {
    let generation = *app_state.share_generation.read().await;
    let shares = app_state.active_shares.read().await.clone();
    let now = chrono::Utc::now().timestamp();
    for entry in shares {
        if entry.kind != kind || entry.resource_id != resource_id { continue; }
        if entry.expires_at != 0 && entry.expires_at <= now { continue; }
        // Reconstruct the share-token under the current generation, then the
        // cookie. The constant_time_eq inside check_share_cookie does the
        // comparison; we just need to produce the candidate string.
        let payload = SharePayload {
            kind: entry.kind.clone(),
            resource_id: entry.resource_id.clone(),
            expires_at: entry.expires_at,
        };
        let token = crate::share_security::sign_share_token(&app_state.master_key, generation, &payload);
        let cookie = cookie_for(&app_state.master_key, &token, &payload.kind, &payload.resource_id);
        return Some(cookie);
    }
    None
}

/// GET /clips/:event_id.mp4 (cookie-gated)
pub(crate) async fn share_clip(
    AxumState(s): AxumState<StreamState>,
    headers: HeaderMap,
    Path(event_id_with_ext): Path<String>,
) -> Response {
    let ip = client_ip(&headers);
    let (ok, retry_after) = clip_limiter().check(&ip);
    if !ok {
        return share_error_page(
            StatusCode::TOO_MANY_REQUESTS,
            "Too many attempts",
            &format!("Please wait about {retry_after} seconds and open the link again."),
        );
    }

    let event_id = event_id_with_ext.trim_end_matches(".mp4").to_string();
    let app_state = match s.app_handle.try_state::<Arc<AppState>>() {
        Some(st) => st,
        None => return share_error_page(StatusCode::INTERNAL_SERVER_ERROR,
            "Something went wrong", "The camera app hit an internal error — try the link again in a moment."),
    };

    let expected = match expected_cookie_for(&app_state, "clip", &event_id).await {
        Some(c) => c,
        None => return expired_link_page(),
    };
    if !check_share_cookie(&headers, &expected) {
        return share_error_page(StatusCode::UNAUTHORIZED,
            "Couldn't verify this link",
            "Open the original link from the message again (not a copy of this page's address). \
             If it keeps happening, open it in your regular browser instead of the in-app preview.");
    }

    // v14: the bounded H.264 NVR clip (sliced from the continuous recording,
    // generated on demand if the close-time export hasn't run). Replaces the old
    // browser-recorded .webm so the shared link plays inline.
    let path = match crate::agent::clip_export::ensure_event_clip(app_state.inner(), &event_id).await {
        Some(p) => p,
        None => return share_error_page(StatusCode::NOT_FOUND,
            "This clip isn't available",
            "The recording may still be processing or has been removed. Ask for a fresh link."),
    };

    // RANGE-SERVED (same responder as the in-app player): streamed body with
    // Accept-Ranges so phones can actually SEEK, instant first byte, and no
    // whole-file RAM copy per view (the old `fs::read` buffered up to 50 MB
    // per request and supported no ranges — the shared player felt weak).
    let mut resp = crate::footage::serve_file_range(&path, &headers, 0, None).await;
    resp.headers_mut().insert("Referrer-Policy", HeaderValue::from_static("no-referrer"));
    resp
}

/// GET /live/:cam_id.mjpeg (cookie-gated, multipart MJPEG stream)
pub(crate) async fn share_live(
    AxumState(s): AxumState<StreamState>,
    headers: HeaderMap,
    Path(cam_with_ext): Path<String>,
) -> Response {
    let ip = client_ip(&headers);
    let (ok, retry_after) = live_limiter().check(&ip);
    if !ok {
        return share_error_page(
            StatusCode::TOO_MANY_REQUESTS,
            "Too many attempts",
            &format!("Please wait about {retry_after} seconds and open the link again."),
        );
    }

    let cam_str = cam_with_ext.trim_end_matches(".mjpeg").to_string();
    let cam_id: usize = match cam_str.parse() {
        Ok(n) if n < s.frame_txs.len() => n,
        _ => return share_error_page(StatusCode::BAD_REQUEST,
            "This camera doesn't exist",
            "The link points at a camera that isn't set up. Ask for a fresh link."),
    };
    let app_state = match s.app_handle.try_state::<Arc<AppState>>() {
        Some(st) => st,
        None => return share_error_page(StatusCode::INTERNAL_SERVER_ERROR,
            "Something went wrong", "The camera app hit an internal error — try the link again in a moment."),
    };

    let expected = match expected_cookie_for(&app_state, "live", &cam_str).await {
        Some(c) => c,
        None => return expired_link_page(),
    };
    if !check_share_cookie(&headers, &expected) {
        return share_error_page(StatusCode::UNAUTHORIZED,
            "Couldn't verify this link",
            "Open the original link from the message again (not a copy of this page's address). \
             If it keeps happening, open it in your regular browser instead of the in-app preview.");
    }

    let mut rx = s.frame_txs[cam_id].subscribe();
    let boundary = "frame";
    let content_type = format!("multipart/x-mixed-replace; boundary={boundary}");

    // Seed with the most recent cached frame so the recipient sees content
    // INSTANTLY instead of a blank/never-loading page when the live capture is
    // momentarily idle (e.g. the user hasn't opened the live view yet). The
    // same frame doubles as a keep-alive payload below.
    let seed_frame: Option<Vec<u8>> =
        app_state.latest_frames.read().await.get(&(cam_id as u8)).cloned();

    let boundary_owned = boundary.to_string();
    // Transcode a source JPEG to the small WAN frame OFF the tokio executor
    // (JPEG decode/encode is CPU-blocking). Falls back to the original bytes if
    // the transcode fails — a bad frame must never blank the stream.
    async fn to_wan(bytes: Vec<u8>) -> Vec<u8> {
        tokio::task::spawn_blocking(move || downscale_jpeg_for_wan(&bytes).unwrap_or(bytes))
            .await.unwrap_or_default()
    }
    let stream = async_stream::try_stream! {
        use tokio::sync::broadcast::error::{RecvError, TryRecvError};
        // `last` holds the last SMALL (already-transcoded) frame — the keepalive
        // re-sends it without paying the transcode again.
        let mut last: Option<Vec<u8>> = None;
        // Seed with the freshest cached frame (transcoded) so the recipient sees
        // content INSTANTLY instead of a blank page.
        if let Some(sf) = seed_frame {
            let small = to_wan(sf).await;
            let header = format!(
                "--{boundary_owned}\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n",
                small.len()
            );
            yield axum::body::Bytes::from(header.into_bytes());
            yield axum::body::Bytes::from(small.clone());
            yield axum::body::Bytes::from_static(b"\r\n");
            last = Some(small);
        }
        // ~12 fps cap. The REAL slow-motion fix is the transcode above (small
        // frames fit a phone's bandwidth so nothing queues); the cap keeps the
        // outbound bitrate low, and drop-to-latest guards momentary spikes.
        let min_interval = Duration::from_millis(83);
        let mut last_sent = tokio::time::Instant::now()
            .checked_sub(min_interval).unwrap_or_else(tokio::time::Instant::now);
        // Collapse any backlog to the single freshest frame — keeps the stream
        // LIVE instead of replaying stale queued frames.
        macro_rules! drain_to_latest { ($frame:ident) => {
            loop {
                match rx.try_recv() {
                    Ok(f) => $frame = f,
                    Err(TryRecvError::Lagged(_)) => continue,
                    Err(_) => break,
                }
            }
        }}
        loop {
            // Time-box the wait so an idle camera doesn't leave the connection
            // hanging silently — on timeout we re-push the last frame as a
            // keep-alive, which keeps the stream (and any CF proxy) alive.
            match tokio::time::timeout(Duration::from_secs(10), rx.recv()).await {
                Ok(Ok(mut frame)) => {
                    drain_to_latest!(frame);
                    let since = last_sent.elapsed();
                    if since < min_interval {
                        tokio::time::sleep(min_interval - since).await;
                        drain_to_latest!(frame);
                    }
                    // Downscale 1280×720 q3 (~60 KB) → 640×360 q45 (~15 KB) so the
                    // MJPEG stream fits a phone/tunnel link and stays real-time.
                    let small = to_wan(frame.as_ref().clone()).await;
                    let header = format!(
                        "--{boundary_owned}\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n",
                        small.len()
                    );
                    yield axum::body::Bytes::from(header.into_bytes());
                    yield axum::body::Bytes::from(small.clone());
                    yield axum::body::Bytes::from_static(b"\r\n");
                    last = Some(small);
                    last_sent = tokio::time::Instant::now();
                }
                Ok(Err(RecvError::Lagged(_))) => continue,
                Ok(Err(_)) => break,
                Err(_) => {
                    // recv timed out — keep-alive with the last (small) frame.
                    if let Some(frame) = &last {
                        let header = format!(
                            "--{boundary_owned}\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n",
                            frame.len()
                        );
                        yield axum::body::Bytes::from(header.into_bytes());
                        yield axum::body::Bytes::from(frame.clone());
                        yield axum::body::Bytes::from_static(b"\r\n");
                    }
                }
            }
        }
    };
    // Pin and box so axum's Body::from_stream is happy with the
    // type-erased `dyn Stream + Send` shape.
    let body = Body::from_stream(
        Box::pin(stream) as std::pin::Pin<Box<dyn futures::Stream<Item = Result<axum::body::Bytes, std::io::Error>> + Send>>
    );

    axum::response::Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", content_type)
        .header("Cache-Control", "no-store, no-cache, must-revalidate")
        .header("Referrer-Policy", "no-referrer")
        .body(body)
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
}

#[cfg(test)]
mod ssrf_tests {
    use super::*;

    #[test]
    fn private_ranges_allowed_public_rejected() {
        let ok = ["192.168.1.10", "10.0.0.5", "172.16.4.2", "127.0.0.1", "169.254.1.1", "::1", "fe80::1", "fd00::2"];
        for ip in ok {
            assert!(is_private_ip(ip.parse().unwrap()), "{ip} should be private");
        }
        let bad = ["8.8.8.8", "1.1.1.1", "142.250.65.78", "172.32.0.1", "2607:f8b0::1"];
        for ip in bad {
            assert!(!is_private_ip(ip.parse().unwrap()), "{ip} should be public");
        }
    }

    #[tokio::test]
    async fn cam_url_guard_literal_and_scheme() {
        assert!(cam_url_is_local("http://192.168.1.44/video.mjpg").await);
        assert!(cam_url_is_local("http://[fe80::1]:8080/stream").await);
        assert!(cam_url_is_local("http://localhost:8081/cam").await);
        assert!(cam_url_is_local("http://frontdoor.local/mjpeg").await);
        assert!(!cam_url_is_local("http://8.8.8.8/anything").await);
        assert!(!cam_url_is_local("https://example.com/steal").await);   // resolves public
        assert!(!cam_url_is_local("ftp://192.168.1.44/x").await);        // scheme
        assert!(!cam_url_is_local("file:///etc/passwd").await);          // scheme
        assert!(!cam_url_is_local("not a url").await);
    }
}
