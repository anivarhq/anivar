//! axum HTTP handlers for recorded-footage download endpoints + ping + ntfy topic.

use std::collections::HashMap;

use axum::{
    body::{Body, Bytes}, extract::State as AxumState,
    http::{HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
};
use base64::{engine::general_purpose::STANDARD as B64, Engine};
use tauri::Manager;
use tokio::io::{AsyncReadExt, AsyncSeekExt};

use crate::StreamState;


/// Simple auth health-check — phone uses this to distinguish "token revoked" from
/// plain network errors (HTTP 401 means revoked; anything else is network).
pub(crate) async fn ping(_: AxumState<StreamState>) -> StatusCode {
    StatusCode::OK
}



/// List motion events (metadata only, no thumbnail blobs — keeps payload small).
pub(crate) async fn footage_list(
    AxumState(s): AxumState<StreamState>,
    axum::extract::Query(params): axum::extract::Query<HashMap<String, String>>,
) -> Response {
    let limit: i64  = params.get("limit").and_then(|v| v.parse().ok()).unwrap_or(20).min(100);
    let offset: i64 = params.get("offset").and_then(|v| v.parse().ok()).unwrap_or(0).max(0);

    type Row = (String, String, Option<String>, Option<f64>, f64, bool);
    let rows: Vec<Row> = sqlx::query_as(
        "SELECT id, started_at, ended_at, duration_secs, peak_score,
                (clip_path IS NOT NULL) AS has_clip
         FROM motion_events ORDER BY started_at DESC LIMIT ? OFFSET ?"
    )
    .bind(limit).bind(offset)
    .fetch_all(&s.db).await.unwrap_or_default();

    let events: Vec<serde_json::Value> = rows.into_iter().map(
        |(id, started_at, ended_at, duration_secs, peak_score, has_clip)| {
            serde_json::json!({
                "id": id,
                "started_at": started_at,
                "ended_at": ended_at,
                "duration_secs": duration_secs,
                "peak_score": peak_score,
                "has_clip": has_clip,
            })
        }
    ).collect();

    (
        StatusCode::OK,
        [(axum::http::header::CONTENT_TYPE, "application/json"),
         (axum::http::header::CACHE_CONTROL, "no-store")],
        serde_json::to_string(&events).unwrap_or_default(),
    ).into_response()
}

/// Serve the JPEG thumbnail for a single event.
pub(crate) async fn footage_thumbnail(
    axum::extract::Path(id): axum::extract::Path<String>,
    AxumState(s): AxumState<StreamState>,
) -> Response {
    // Validate: UUIDs are hex + hyphens only — reject anything else to prevent injection
    if !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') || id.len() > 40 {
        return StatusCode::BAD_REQUEST.into_response();
    }
    let row: Option<(Option<String>,)> =
        sqlx::query_as("SELECT thumbnail FROM motion_events WHERE id=?")
            .bind(&id).fetch_optional(&s.db).await.unwrap_or(None);

    match row {
        Some((Some(b64),)) => match B64.decode(b64.trim()) {
            Ok(jpeg) => {
                let mut headers = HeaderMap::new();
                headers.insert("Content-Type",  HeaderValue::from_static("image/jpeg"));
                headers.insert("Cache-Control", HeaderValue::from_static("max-age=3600, immutable"));
                (StatusCode::OK, headers, Body::from(jpeg)).into_response()
            }
            Err(_) => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
        },
        _ => StatusCode::NOT_FOUND.into_response(),
    }
}

/// Serve a FACE crop by face_embeddings id — the URL-served twin of the old
/// inline-base64 payloads (Chromium caches + evicts these like normal images;
/// data-URIs pinned megabytes in the DOM and shipped over IPC every refresh).
pub(crate) async fn face_crop(
    axum::extract::Path(id): axum::extract::Path<String>,
    AxumState(s): AxumState<StreamState>,
) -> Response {
    if !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') || id.len() > 48 {
        return StatusCode::BAD_REQUEST.into_response();
    }
    let row: Option<(Option<String>,)> =
        sqlx::query_as("SELECT thumbnail_b64 FROM face_embeddings WHERE id=?")
            .bind(&id).fetch_optional(&s.db).await.unwrap_or(None);
    match row {
        Some((Some(v),)) => {
            let bytes = crate::blobstore::resolve_bytes(&s.data_dir, &v);
            if bytes.is_empty() { return StatusCode::NOT_FOUND.into_response(); }
            let mut headers = HeaderMap::new();
            headers.insert("Content-Type",  HeaderValue::from_static("image/jpeg"));
            // A face crop for a given id never changes → immutable.
            headers.insert("Cache-Control", HeaderValue::from_static("max-age=3600, immutable"));
            (StatusCode::OK, headers, Body::from(bytes)).into_response()
        }
        _ => StatusCode::NOT_FOUND.into_response(),
    }
}

/// Serve a BODY-track crop by track id: latest stored body crop, else the
/// latest linked event thumbnail (same preference order the Tracked list used
/// when it inlined these). Short cache — a track's latest crop advances.
pub(crate) async fn body_crop(
    axum::extract::Path(track_id): axum::extract::Path<String>,
    AxumState(s): AxumState<StreamState>,
) -> Response {
    if !track_id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') || track_id.len() > 64 {
        return StatusCode::BAD_REQUEST.into_response();
    }
    let own: Option<String> = sqlx::query_scalar(
        "SELECT thumbnail_b64 FROM body_embeddings
          WHERE person_id = ? AND thumbnail_b64 IS NOT NULL
          ORDER BY seen_at DESC LIMIT 1"
    ).bind(&track_id).fetch_optional(&s.db).await.ok().flatten();
    let v = match own {
        Some(v) => Some(v),
        None => sqlx::query_scalar(
            "SELECT m.thumbnail FROM body_embeddings b
              JOIN motion_events m ON m.id = b.event_id
             WHERE b.person_id = ? AND b.event_id IS NOT NULL AND m.thumbnail IS NOT NULL
             ORDER BY b.seen_at DESC LIMIT 1"
        ).bind(&track_id).fetch_optional(&s.db).await.ok().flatten(),
    };
    match v {
        Some(v) => {
            let bytes = crate::blobstore::resolve_bytes(&s.data_dir, &v);
            if bytes.is_empty() { return StatusCode::NOT_FOUND.into_response(); }
            let mut headers = HeaderMap::new();
            headers.insert("Content-Type",  HeaderValue::from_static("image/jpeg"));
            headers.insert("Cache-Control", HeaderValue::from_static("max-age=300"));
            (StatusCode::OK, headers, Body::from(bytes)).into_response()
        }
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

/// GET /clip-start — the TRUE wall-clock start of what playback will show, so
/// the player needle/scrubber can match the video exactly (no ffmpeg spawned).
///   ?event=<id>            → event clips: exact window start when the event is
///                            CLOSED (the cached clip is re-encoded frame-accurate);
///                            keyframe-snapped when still OPEN (copy-concat).
///   ?cam=<n>&start=<secs>  → history seeks: keyframe-snapped concat start.
pub(crate) async fn clip_start_meta(
    axum::extract::Query(q): axum::extract::Query<std::collections::HashMap<String, String>>,
    AxumState(s): AxumState<StreamState>,
) -> Response {
    let json = |ms: i64| {
        (StatusCode::OK, [("Content-Type", "application/json")],
         format!("{{\"start_ms\":{ms}}}")).into_response()
    };
    if let Some(id) = q.get("event") {
        if !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') || id.len() > 40 {
            return StatusCode::BAD_REQUEST.into_response();
        }
        // Same pre/post buffers footage_clip uses, so the window matches exactly.
        let (pre, post) = match s.app_handle.try_state::<std::sync::Arc<crate::AppState>>() {
            Some(st) => {
                let g = st.settings.read().await;
                (g.record_pre_buffer_secs as i64, g.record_post_buffer_secs as i64)
            }
            None => (3, 10),
        };
        let Some((cam_id, start_secs, _end, is_open)) = event_clip_window(&s.db, pre, post, id).await else {
            return StatusCode::NOT_FOUND.into_response();
        };
        let ms = if is_open {
            crate::nvr_stream::snapped_playback_start_ms(&s.db, cam_id, start_secs).await
        } else {
            (start_secs * 1000.0) as i64 // cached clip is re-encoded frame-accurate
        };
        return json(ms);
    }
    let cam: u8 = q.get("cam").and_then(|v| v.parse().ok()).unwrap_or(0);
    let Some(start): Option<f64> = q.get("start").and_then(|v| v.parse().ok()) else {
        return StatusCode::BAD_REQUEST.into_response();
    };
    json(crate::nvr_stream::snapped_playback_start_ms(&s.db, cam, start).await)
}

/// v14: compute the NVR clip window for an event — anchor at the first confirmed
/// object (trims the empty motion-only lead-in), padded by the pre/post capture
/// buffers. Shared by `footage_clip` (streams the full event in-app) and
/// `agent::clip_export::ensure_event_clip` (caps + exports the file the agent
/// SENDS). `pre`/`post` are `record_pre_buffer_secs` / `record_post_buffer_secs`.
/// Returns `(cam_id, start_secs, end_secs)` or `None` if the event is unknown.
pub(crate) async fn event_clip_window(
    db: &sqlx::SqlitePool,
    pre: i64,
    post: i64,
    event_id: &str,
) -> Option<(u8, f64, f64, bool)> {  // (cam_id, start_secs, end_secs, is_open/in-progress)
    let row: Option<(i64, String, Option<String>, Option<String>)> = sqlx::query_as(
        "SELECT cam_id, started_at, ended_at, first_object_at FROM motion_events WHERE id=?"
    ).bind(event_id).fetch_optional(db).await.unwrap_or(None);
    let (cam_id, started_at_raw, ended_at_raw, first_object_raw) = row?;

    let started_ts = chrono::DateTime::parse_from_rfc3339(&started_at_raw).ok()?.timestamp();
    // Anchor to the first-confirmed object when present (skip the motion-only
    // lead-in); fall back to motion onset for motion-only events.
    let anchor_ts = first_object_raw.as_deref()
        .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
        .map(|dt| dt.timestamp())
        .unwrap_or(started_ts);
    // IN-PROGRESS (mature NVRs model): an OPEN event (ended_at NULL) plays everything
    // RECORDED SO FAR — its requested end is NOW (clamped to finalized footage below),
    // not a fixed `started+30` snippet. A CLOSED event ends at its real ended_at.
    let now_ts = chrono::Utc::now().timestamp();
    let is_open = ended_at_raw.is_none();
    let ended_ts = ended_at_raw.as_deref()
        .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
        .map(|t| t.timestamp())
        .unwrap_or(now_ts);

    let mut start_secs = (anchor_ts - pre).max(0) as f64;
    // FRONT CLAMP: never start the window before the first recorded segment for this
    // camera — otherwise the pre-buffer reaches into unrecorded time and renders as a
    // black "no footage" lead-in (symmetric to the end cap below). When the event sits
    // near the recording start (or its pre-roll was pruned), begin at the real footage.
    {
        let first_seg: Option<(String,)> = sqlx::query_as(
            "SELECT started_at FROM nvr_segments WHERE cam_id=? ORDER BY started_at ASC LIMIT 1"
        ).bind(cam_id).fetch_optional(db).await.unwrap_or(None);
        if let Some((first_raw,)) = first_seg {
            if let Ok(first_dt) = chrono::DateTime::parse_from_rfc3339(&first_raw) {
                let first_start = first_dt.timestamp() as f64;
                if start_secs < first_start { start_secs = first_start; }
            }
        }
    }
    // Requested end = event end + post-buffer, but NEVER past real footage — else
    // ffmpeg pads a black/frozen tail. Cap to (a) now (no future footage) and
    // (b) the real end of the last recorded segment covering the window.
    let mut end_ts = (ended_ts + post).min(now_ts);
    {
        // Last (and previous) segment starting at/before the requested end. The gap
        // between the two = the typical segment length → the last segment's real end.
        let end_probe = chrono::DateTime::<chrono::Utc>::from_timestamp(end_ts + 2, 0)
            .unwrap_or_else(chrono::Utc::now).to_rfc3339();
        let last2: Vec<(String,)> = sqlx::query_as(
            "SELECT started_at FROM nvr_segments WHERE cam_id=? AND started_at <= ? ORDER BY started_at DESC LIMIT 2"
        ).bind(cam_id).bind(&end_probe).fetch_all(db).await.unwrap_or_default();
        if let Some((last_raw,)) = last2.first() {
            if let Ok(last_dt) = chrono::DateTime::parse_from_rfc3339(last_raw) {
                let last_start = last_dt.timestamp();
                let seg_len = last2.get(1)
                    .and_then(|(prev,)| chrono::DateTime::parse_from_rfc3339(prev).ok())
                    .map(|p| last_start - p.timestamp())
                    .filter(|d| *d > 0 && *d < 300)   // sane segment length
                    .unwrap_or(15);                    // default ~12-15s segments
                let seg_real_end = last_start + seg_len;
                end_ts = end_ts.min(seg_real_end);
            }
        }
    }
    // NOT-READY GUARD: if no FINALIZED footage covers the window yet (an event that
    // started seconds ago — its 10s segment is still being written and isn't indexed),
    // there is genuinely nothing to play. Return None so the caller 404s and the player
    // shows a brief loading spinner + retries, instead of building a 262-byte EMPTY clip
    // from a window whose -ss seeks past the only available segment. Once the first
    // segment finalizes (~10-15s later) the next request succeeds and grows over time.
    if (end_ts as f64) <= start_secs + 1.0 {
        tracing::info!(
            "event_clip_window {}: not ready (open={}) — no finalized footage yet",
            &event_id[..8.min(event_id.len())], is_open
        );
        return None;
    }
    let end_secs = end_ts as f64;
    Some((cam_id as u8, start_secs, end_secs, is_open))
}

/// v12: serve the event's clip as a virtual slice of the continuous NVR
/// recording — mature NVRs' `GET /events/{id}/clip.mp4` model.
///
/// The pre-v12 per-event clip writer wrote JPEG-frame `.bin` files that the
/// browser couldn't decode. We deleted that writer. The clip is now an
/// ffmpeg concat of NVR segments between `started_at - pre_capture_secs` and
/// `ended_at + post_capture_secs`. If continuous NVR isn't enabled for this
/// cam, there are no segments → return 404 with a clear log.
pub(crate) async fn footage_clip(
    axum::extract::Path(id): axum::extract::Path<String>,
    AxumState(s): AxumState<StreamState>,
    axum::extract::Query(params): axum::extract::Query<HashMap<String, String>>,
    _req_headers: HeaderMap,
) -> Response {
    if !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') || id.len() > 40 {
        return StatusCode::BAD_REQUEST.into_response();
    }
    let download = params.get("download")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    // Pull the pre/post-capture pad from settings (StreamState carries the
    // app_handle the AppState is attached to). The bound math lives in
    // `event_clip_window`, shared with the clip exporter the agent SENDS.
    let app_state = s.app_handle.try_state::<std::sync::Arc<crate::AppState>>();
    let (pre, post) = match app_state {
        Some(st) => {
            let g = st.settings.read().await;
            (g.record_pre_buffer_secs as i64, g.record_post_buffer_secs as i64)
        }
        None => (3, 10),
    };

    let (cam_id, start_secs, end_secs, is_open) = match event_clip_window(&s.db, pre, post, &id).await {
        Some(w) => w,
        None    => return StatusCode::NOT_FOUND.into_response(),
    };
    // True first-frame wall-clock of the clip — lets the player anchor the timeline
    // needle exactly (clip frame 0 == start_secs), instead of guessing the event time.
    let start_ms = (start_secs * 1000.0) as i64;

    // IN-PROGRESS events: NEVER serve/keep a cached file — the footage is still growing,
    // so a cached partial would be frozen + later served as the "final" clip. Stream the
    // play-so-far window FRESH each request (mature NVRs in-progress model). Only a CLOSED
    // event uses the cached, re-encoded file below.
    //
    // PREFER (closed) the cached, RE-ENCODED clip file (frame-accurate, faststart, no
    // black edges) and serve it with HTTP range support so seeking/±10s is instant. The
    // on-the-fly `-c copy` concat is the fallback (it can show black at a mid-GOP start).
    if !is_open {
    if let Some(st) = s.app_handle.try_state::<std::sync::Arc<crate::AppState>>() {
        if let Some(clip_path) = crate::agent::clip_export::ensure_event_clip(&st, &id).await {
            // Path-safety: the file must live under data_dir.
            let safe = std::path::PathBuf::from(&clip_path).canonicalize().ok()
                .zip(st.data_dir.canonicalize().ok())
                .map(|(c, d)| c.starts_with(&d))
                .unwrap_or(false);
            if safe {
                let dl_name = download.then(|| {
                    let id_short: String = id.chars().take(8).collect();
                    format!("event_{}.mp4", id_short)
                });
                return serve_file_range(&clip_path, &_req_headers, start_ms, dl_name).await;
            }
        }
    }
    } // end `if !is_open` — closed events serve the cached file; open events fall through

    // Fallback (closed, no cache) / IN-PROGRESS (open): on-the-fly bounded concat of the
    // play-so-far window — always fresh, nothing stale cached.
    // Reencode: an OPEN event streams straight into the <video> element, so the
    // audio must be ONE continuous AAC stream (copy would stutter at seams).
    let mut resp = crate::nvr_stream::stream_concat_window(
        &s, cam_id, start_secs, Some(end_secs), crate::nvr_stream::ConcatAudio::Reencode).await;
    if let Ok(hv) = HeaderValue::from_str(&start_ms.to_string()) {
        resp.headers_mut().insert("X-Clip-Start-Ms", hv);
    }
    if download {
        let id_short: String = id.chars().take(8).collect();
        let suggested = format!("event_{}.mp4", id_short);
        if let Ok(hv) = HeaderValue::from_str(&format!("attachment; filename=\"{}\"", suggested)) {
            resp.headers_mut().insert("Content-Disposition", hv);
        }
    }
    resp
}

/// Serve a local MP4 file with HTTP range support (seekable, instant start) plus the
/// `X-Clip-Start-Ms` true-start header. Mirrors the NVR file responder in
/// `nvr_stream`. `download_name` adds a `Content-Disposition: attachment` when set.
/// pub(crate): also the share-link clip responder (`http_handlers::share_clip`) —
/// which previously read the WHOLE file into memory with no Range support, so
/// shared-link viewers couldn't seek and every view spiked RAM by the clip size.
pub(crate) async fn serve_file_range(
    path: &str,
    req_headers: &HeaderMap,
    start_ms: i64,
    download_name: Option<String>,
) -> Response {
    let mut file = match tokio::fs::File::open(path).await {
        Ok(f) => f,
        Err(_) => return StatusCode::NOT_FOUND.into_response(),
    };
    let file_size = match file.metadata().await {
        Ok(m) => m.len(),
        Err(_) => return StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    };
    let range = req_headers.get("range")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.strip_prefix("bytes="))
        .and_then(|s| {
            let mut parts = s.splitn(2, '-');
            let start: u64 = parts.next()?.parse().ok()?;
            let end: u64 = parts.next()
                .and_then(|e| if e.is_empty() { None } else { e.parse().ok() })
                .unwrap_or(file_size.saturating_sub(1));
            if start > end || end >= file_size { None } else { Some((start, end)) }
        });

    let mut headers = HeaderMap::new();
    headers.insert("Content-Type", HeaderValue::from_static("video/mp4"));
    headers.insert("Accept-Ranges", HeaderValue::from_static("bytes"));
    headers.insert("Cache-Control", HeaderValue::from_static("no-cache"));
    if let Ok(hv) = HeaderValue::from_str(&start_ms.to_string()) {
        headers.insert("X-Clip-Start-Ms", hv);
    }
    if let Some(name) = download_name {
        if let Ok(hv) = HeaderValue::from_str(&format!("attachment; filename=\"{}\"", name)) {
            headers.insert("Content-Disposition", hv);
        }
    }

    match range {
        Some((start, end)) => {
            file.seek(std::io::SeekFrom::Start(start)).await.ok();
            let length = end - start + 1;
            let stream = tokio_util::io::ReaderStream::new(file.take(length));
            headers.insert("Content-Length", HeaderValue::from_str(&length.to_string()).unwrap());
            headers.insert("Content-Range", HeaderValue::from_str(&format!("bytes {start}-{end}/{file_size}")).unwrap());
            (StatusCode::PARTIAL_CONTENT, headers, Body::from_stream(stream)).into_response()
        }
        None => {
            let stream = tokio_util::io::ReaderStream::new(file);
            headers.insert("Content-Length", HeaderValue::from_str(&file_size.to_string()).unwrap());
            (StatusCode::OK, headers, Body::from_stream(stream)).into_response()
        }
    }
}

/// Replay a recorded clip as an MJPEG stream (for mobile `<img>` playback).
/// Streams at ~15 fps. Returns 404 when clip isn't ready yet.
pub(crate) async fn footage_stream(
    axum::extract::Path(id): axum::extract::Path<String>,
    AxumState(s): AxumState<StreamState>,
) -> Response {
    if !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') || id.len() > 40 {
        return StatusCode::BAD_REQUEST.into_response();
    }
    let row: Option<(Option<String>,)> =
        sqlx::query_as("SELECT clip_path FROM motion_events WHERE id=?")
            .bind(&id).fetch_optional(&s.db).await.unwrap_or(None);
    let clip_path = match row { Some((Some(p),)) => p, _ => return StatusCode::NOT_FOUND.into_response() };

    let abs_clip = match std::path::PathBuf::from(&clip_path).canonicalize() {
        Ok(p) => p, Err(_) => return StatusCode::NOT_FOUND.into_response(),
    };
    let abs_data = match s.data_dir.canonicalize() {
        Ok(p) => p, Err(_) => return StatusCode::NOT_FOUND.into_response(),
    };
    if !abs_clip.starts_with(&abs_data) { return StatusCode::FORBIDDEN.into_response(); }

    let stream = async_stream::stream! {
        let mut file = match tokio::fs::File::open(&abs_clip).await {
            Ok(f) => f, Err(_) => return,
        };
        let mut len_buf = [0u8; 4];
        loop {
            if file.read_exact(&mut len_buf).await.is_err() { break; }
            let frame_len = u32::from_be_bytes(len_buf) as usize;
            if frame_len == 0 || frame_len > 10 * 1024 * 1024 { break; }
            let mut jpeg = vec![0u8; frame_len];
            if file.read_exact(&mut jpeg).await.is_err() { break; }
            let header = format!(
                "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n",
                frame_len
            );
            let mut chunk = header.into_bytes();
            chunk.extend_from_slice(&jpeg);
            chunk.extend_from_slice(b"\r\n");
            yield Ok::<_, std::convert::Infallible>(Bytes::from(chunk));
            tokio::time::sleep(tokio::time::Duration::from_millis(66)).await;
        }
    };

    let body = Body::from_stream(stream);
    let mut headers = HeaderMap::new();
    headers.insert("Content-Type",  HeaderValue::from_static("multipart/x-mixed-replace; boundary=frame"));
    headers.insert("Cache-Control", HeaderValue::from_static("no-cache, no-store"));
    headers.insert("Connection",    HeaderValue::from_static("close"));
    (StatusCode::OK, headers, body).into_response()
}
