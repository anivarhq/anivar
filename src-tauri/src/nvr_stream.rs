//! Axum HTTP handlers for streaming the recorded NVR archive — supports HTTP Range, seek, and multi-segment concatenation.

use std::collections::HashMap;

use axum::{
    body::Body, extract::State as AxumState,
    http::{HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
};
use chrono::Utc;
use tokio::io::{AsyncReadExt, AsyncSeekExt};

use crate::{StreamState, ensure_ffmpeg};


/// Serves an NVR segment file over HTTP with Range support (for video seeking).
/// Query params: `path` (relative filename in data_dir/nvr/) + auth token already checked by middleware.
pub(crate) async fn nvr_stream(
    AxumState(s): AxumState<StreamState>,
    axum::extract::Query(params): axum::extract::Query<HashMap<String, String>>,
    req_headers: HeaderMap,
) -> Response {
    let filename = match params.get("file") {
        Some(f) => f.clone(),
        None => return StatusCode::BAD_REQUEST.into_response(),
    };
    // Security: only allow alphanumeric, dash, underscore, dot
    if !filename.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.')) {
        return StatusCode::FORBIDDEN.into_response();
    }
    let path = s.data_dir.join("nvr").join(&filename);
    let abs = match path.canonicalize() {
        Ok(p) => p,
        Err(_) => return StatusCode::NOT_FOUND.into_response(),
    };
    let abs_nvr = match s.data_dir.join("nvr").canonicalize() {
        Ok(p) => p,
        Err(_) => return StatusCode::NOT_FOUND.into_response(),
    };
    if !abs.starts_with(&abs_nvr) {
        return StatusCode::FORBIDDEN.into_response();
    }
    let mut file = match tokio::fs::File::open(&abs).await {
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

    let ct = if filename.ends_with(".webm") { "video/webm" } else { "video/mp4" };
    let mut headers = HeaderMap::new();
    headers.insert("Content-Type", HeaderValue::from_str(ct).unwrap());
    headers.insert("Accept-Ranges", HeaderValue::from_static("bytes"));
    headers.insert("Cache-Control", HeaderValue::from_static("no-cache"));

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

/// Server-side time-seek endpoint — mature NVRs' approach: instead of asking the
/// browser to seek within an MP4 (unreliable), we run `ffmpeg -ss OFFSET` on
/// the server and stream the result starting from second 0.
///
/// GET /nvr-seek?file=FILENAME&offset=SECONDS&token=TOKEN
///
/// The browser receives a fresh fMP4 stream starting at the requested time.
/// No client-side currentTime manipulation is ever needed.
pub(crate) async fn nvr_seek_stream(
    AxumState(s): AxumState<StreamState>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Response {
    let filename = match params.get("file") {
        Some(f) => f.clone(),
        None => return StatusCode::BAD_REQUEST.into_response(),
    };
    if !filename.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.')) {
        return StatusCode::FORBIDDEN.into_response();
    }
    let offset_secs: f64 = params.get("offset").and_then(|v| v.parse().ok()).unwrap_or(0.0);
    let path = s.data_dir.join("nvr").join(&filename);
    if !path.exists() {
        return StatusCode::NOT_FOUND.into_response();
    }

    // If offset is negligible, fall back to a simple file serve (faster start)
    if offset_secs < 1.0 {
        let file = match tokio::fs::File::open(&path).await {
            Ok(f) => f, Err(_) => return StatusCode::NOT_FOUND.into_response(),
        };
        let file_size = file.metadata().await.map(|m| m.len()).unwrap_or(0);
        let ct = if filename.ends_with(".webm") { "video/webm" } else { "video/mp4" };
        let mut headers = HeaderMap::new();
        headers.insert("Content-Type", HeaderValue::from_str(ct).unwrap());
        headers.insert("Accept-Ranges", HeaderValue::from_static("bytes"));
        headers.insert("Cache-Control", HeaderValue::from_static("no-cache"));
        headers.insert("Content-Length", HeaderValue::from_str(&file_size.to_string()).unwrap());
        return (StatusCode::OK, headers, Body::from_stream(tokio_util::io::ReaderStream::new(file))).into_response();
    }

    // Server-side seek via ffmpeg — streams the file starting from `offset_secs`.
    // `-ss` before `-i` = fast seek to nearest keyframe (accurate within one GOP).
    // `-c copy` = no re-encode, instant start.
    // `+frag_keyframe+default_base_moof` = proper fMP4 so browser plays from byte 0.
    let ffmpeg = match ensure_ffmpeg(&s.data_dir).await {
        Ok(f) => f, Err(_) => return StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    };
    let offset_str = format!("{:.3}", offset_secs);
    // Single-file seek: audio (when present) is safe here — no concat boundary —
    // but re-encode it with async resample so the seeked stream starts clean.
    let file_audio: bool = sqlx::query_scalar(
        "SELECT has_audio FROM nvr_segments WHERE path=? LIMIT 1")
        .bind(path.to_string_lossy().as_ref())
        .fetch_optional(&s.db).await.ok().flatten().map(|v: i64| v == 1).unwrap_or(false);
    let path_str = path.to_string_lossy().to_string();
    let mut sargs: Vec<String> = vec![
        "-hide_banner".into(), "-loglevel".into(), "error".into(),
        "-ss".into(), offset_str,
        "-i".into(), path_str,
        "-map".into(), "0:v:0".into(),
        "-c:v".into(), "copy".into(),
    ];
    if file_audio {
        sargs.extend(["-map".into(), "0:a:0".into(),
            "-c:a".into(), "copy".into(),
            "-avoid_negative_ts".into(), "make_zero".into()]);
    } else {
        sargs.push("-an".into());
    }
    sargs.extend(["-movflags".into(), "+frag_keyframe+default_base_moof".into(),
        "-f".into(), "mp4".into(), "pipe:1".into()]);
    let child = crate::proc::tokio_cmd(&ffmpeg)
        .args(&sargs)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .kill_on_drop(true)
        .spawn();

    match child {
        Ok(mut proc) => {
            let stdout = match proc.stdout.take() {
                Some(s) => s, None => return StatusCode::INTERNAL_SERVER_ERROR.into_response(),
            };
            // Keep process alive by moving it into the stream
            let stream = tokio_util::io::ReaderStream::new(stdout);
            tokio::spawn(async move { let _ = proc.wait().await; });
            let mut headers = HeaderMap::new();
            headers.insert("Content-Type", HeaderValue::from_static("video/mp4"));
            headers.insert("Cache-Control", HeaderValue::from_static("no-cache"));
            (StatusCode::OK, headers, Body::from_stream(stream)).into_response()
        }
        Err(e) => {
            tracing::warn!("nvr_seek_stream ffmpeg error: {e}");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

/// Seamless continuous playback — mature NVRs' concat approach.
///
/// GET /nvr-concat?cam=0&start=UNIX_SECS&token=TOKEN[&window=SECONDS]
///
/// Stitches the camera's segments starting at `start` into one fMP4 stream,
/// BOUNDED to a `window` of seconds (default 10 min). Mature NVRs never stitches a
/// whole day at once: with thousands of short segments an unbounded concat
/// builds a manifest so large that ffmpeg can't begin the stream and the
/// browser reports MediaError 4 ("src not supported"). A bounded window keeps
/// the manifest tiny and the stream instant; the frontend advances the window
/// (new `start`) as the playhead nears the edge for seamless continuous play.
///
/// DB-backed: queries nvr_segments for exact start/end times — no filename guessing.
// 30-minute playback chunk. Was 600 s: with in-playlist seeking (hlsAttach's
// `seekToDate`) the loaded window is exactly the region a click can reach with NO
// network at all, so widening it converts most scrubs into instant seeks - and
// continuous playback stops hitching at every window boundary. `/nvr-vod` clamps
// the query override to 3600.
pub(crate) const NVR_CONCAT_WINDOW_SECS: f64 = 1800.0;

pub(crate) async fn nvr_concat_stream(
    AxumState(s): AxumState<StreamState>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Response {
    let cam_id: u8 = params.get("cam").and_then(|v| v.parse().ok()).unwrap_or(0);
    let start_secs = params.get("start").and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0);
    // Bounded window. Clamp to a sane range so a malicious/huge value can't
    // recreate the unbounded-concat failure.
    let window = params.get("window")
        .and_then(|v| v.parse::<f64>().ok())
        .map(|w| w.clamp(30.0, 3600.0))
        .unwrap_or(NVR_CONCAT_WINDOW_SECS);
    // DEPRECATED for playback: the UI now plays recorded windows via HLS VOD
    // (/nvr-vod, see nvr_vod.rs). Kept for downloads/tools that want one mp4.
    stream_concat_window(&s, cam_id, start_secs, Some(start_secs + window), ConcatAudio::Copy).await
}

/// v13 export route: bounded NVR-segment concat that the browser is encouraged
/// to save as a file. Used by the Camera History modal's "Export time range"
/// presets ("Last 1 min", "Last hour", custom range, etc.).
///
/// Query params: `cam` (u8), `start` (unix epoch secs), `end` (unix epoch secs).
/// Optional: `download=1` adds `Content-Disposition: attachment` so a plain
/// browser context downloads instead of streaming inline. The Tauri save-dialog
/// flow uses `fetch + writeBinaryFile` so it doesn't strictly need the header,
/// but having it makes right-click → "Save link as" work too.
pub(crate) async fn nvr_export_stream(
    AxumState(s): AxumState<StreamState>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Response {
    let cam_id: u8 = params.get("cam").and_then(|v| v.parse().ok()).unwrap_or(0);
    let start_secs = params.get("start").and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0);
    let end_secs   = params.get("end").and_then(|v| v.parse::<f64>().ok());
    // BOUNDED window (its sibling /nvr-concat clamps to 3600s; export forgot):
    // an unclamped end let one request pin ffmpeg on a week-long concat. 6h is
    // generous for any real export; end must be after start.
    const EXPORT_MAX_SECS: f64 = 6.0 * 3600.0;
    let end_secs = match end_secs {
        Some(e) if e <= start_secs =>
            return (axum::http::StatusCode::BAD_REQUEST, "export 'end' must be after 'start'").into_response(),
        Some(e) if e - start_secs > EXPORT_MAX_SECS =>
            return (axum::http::StatusCode::BAD_REQUEST, "export window too large — maximum 6 hours per export").into_response(),
        Some(e) => Some(e),
        // Missing end used to mean "everything after start" — same unbounded
        // hazard through the back door. Default to a 1-hour window.
        None => Some(start_secs + 3600.0),
    };
    let download   = params.get("download")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(true); // /nvr-export defaults to download — that's the whole point of this route
    let suggested  = params.get("filename").cloned().unwrap_or_else(|| {
        format!("export_cam{}_{}.mp4", cam_id, start_secs as i64)
    });

    let mut resp = stream_concat_window(&s, cam_id, start_secs, end_secs, ConcatAudio::Copy).await;
    if download {
        // Sanitise the filename — only ASCII printable, no quotes / slashes.
        let safe: String = suggested.chars()
            .filter(|c| c.is_ascii() && !matches!(*c, '"' | '\\' | '/' | '\0' | '\n' | '\r'))
            .collect();
        let safe = if safe.is_empty() { "export.mp4".to_string() } else { safe };
        if let Ok(hv) = HeaderValue::from_str(&format!("attachment; filename=\"{}\"", safe)) {
            resp.headers_mut().insert("Content-Disposition", hv);
        }
    }
    resp
}

/// v12: shared ffmpeg-concat-stitch helper. Used by:
///   * `nvr_concat_stream` for the timeline scrubber (no end bound).
///   * `footage::footage_clip` for per-event "clips" that ARE a slice of the
///     continuous NVR recording (mature NVRs' `events/{id}/clip.mp4` model).
///
/// Finds NVR segments overlapping `[start_secs, end_secs]`, builds an ffmpeg
/// True when EVERY indexed segment overlapping [start, end] carries an audio
/// stream. Concat requires a uniform stream layout, so audio is mapped only for
/// uniformly-audible windows; mixed windows (spanning the silent-era archive)
/// gracefully fall back to video-only playback and resolve themselves as
/// retention prunes the old segments.
pub(crate) async fn window_all_audio(
    db: &sqlx::SqlitePool,
    cam_id: u8,
    start_secs: f64,
    end_secs: Option<f64>,
) -> bool {
    let from = chrono::DateTime::<Utc>::from_timestamp((start_secs as i64) - 20, 0)
        .unwrap_or_else(Utc::now).to_rfc3339();
    let to = chrono::DateTime::<Utc>::from_timestamp(
        end_secs.map(|e| e.ceil() as i64 + 60).unwrap_or_else(|| Utc::now().timestamp() + 60), 0)
        .unwrap_or_else(Utc::now).to_rfc3339();
    let min_audio: Option<i64> = sqlx::query_scalar(
        "SELECT MIN(has_audio) FROM nvr_segments WHERE cam_id=? AND started_at >= ? AND started_at <= ?")
        .bind(cam_id as i64).bind(&from).bind(&to)
        .fetch_optional(db).await
        .map_err(|e| tracing::warn!("window_all_audio query failed: {e}")).ok().flatten();
    tracing::info!("window_all_audio cam{} [{} .. {}] -> MIN={:?}", cam_id, from, to, min_audio);
    min_audio == Some(1)
}

/// Wall-clock ms where COPY-concat playback ACTUALLY starts for a requested
/// start. Input-side `-ss` with `-c copy` snaps DOWN to a keyframe; our recorder
/// forces keyframes on a 1-second cadence from each segment's start (PTS reset
/// per segment), so the snap is deterministic: `seg_start + floor(offset)`.
/// A request that predates coverage starts at the first available segment.
/// This is the single source of truth the player's needle aligns to — without
/// it the UI assumes playback starts at the REQUESTED time and every visual
/// runs up to 1 s ahead of the actual video (the seek-precision mismatch).
pub(crate) async fn snapped_playback_start_ms(
    db: &sqlx::SqlitePool,
    cam_id: u8,
    start_secs: f64,
) -> i64 {
    let start_dt = chrono::DateTime::<Utc>::from_timestamp(start_secs as i64, 0)
        .unwrap_or_else(Utc::now)
        .to_rfc3339();
    // Covering segment → keyframe at/below the offset.
    let covering: Option<(String,)> = sqlx::query_as(
        "SELECT started_at FROM nvr_segments
         WHERE cam_id=? AND started_at <= ? ORDER BY started_at DESC LIMIT 1")
        .bind(cam_id as i64).bind(&start_dt)
        .fetch_optional(db).await.unwrap_or(None);
    if let Some((started,)) = covering {
        if let Ok(seg) = chrono::DateTime::parse_from_rfc3339(&started) {
            let seg_secs = seg.timestamp() as f64;
            let off = (start_secs - seg_secs).max(0.0);
            // Only trust the covering segment while the offset is plausibly inside
            // it (segments are ~10 s; allow slack for the final segment growing).
            if off < 120.0 {
                return ((seg_secs + off.floor()) * 1000.0) as i64;
            }
        }
    }
    // No coverage at start → playback begins at the NEXT segment.
    let next: Option<(String,)> = sqlx::query_as(
        "SELECT started_at FROM nvr_segments
         WHERE cam_id=? AND started_at > ? ORDER BY started_at ASC LIMIT 1")
        .bind(cam_id as i64).bind(&start_dt)
        .fetch_optional(db).await.unwrap_or(None);
    if let Some((started,)) = next {
        if let Ok(seg) = chrono::DateTime::parse_from_rfc3339(&started) {
            return seg.timestamp_millis();
        }
    }
    (start_secs * 1000.0) as i64
}

/// concat manifest with an `inpoint` on the first segment so ffmpeg skips
/// into the right offset, and pipes the result through `ffmpeg -f concat -c copy`
/// to produce a fragmented MP4 stream the browser can decode immediately.
///
/// `end_secs = None` → unbounded (stream to EOF / client cancel).
/// Builds the ffmpeg concat manifest for the segments overlapping
/// `[start_secs, end_secs]` and writes it to a `_concat_*.txt` in `data_dir`.
/// Returns `(manifest_path, duration)` where `duration` is the `-t` cut value
/// (`None` for an unbounded window). Returns `None` when there's genuinely no
/// footage in range (callers map that to 404 / failure). Factored out so both
/// the streaming endpoint (`stream_concat_window`) and the file exporter
/// (`concat_window_to_file`) share the exact same fool-proof segment logic.
async fn build_concat_manifest(
    db: &sqlx::SqlitePool,
    data_dir: &std::path::Path,
    cam_id: u8,
    start_secs: f64,
    end_secs: Option<f64>,
) -> Option<(std::path::PathBuf, Option<f64>, f64)> {
    let start_dt = chrono::DateTime::<Utc>::from_timestamp(start_secs as i64, 0)
        .unwrap_or_else(Utc::now)
        .to_rfc3339();

    // Step 1: Find the segment that CONTAINS start_secs.
    // This is ALWAYS the correct first segment — its started_at <= start_secs.
    // We compute inpoint = start_secs - seg.started_at so ffmpeg skips to the right position.
    //
    // Fool-proof for FRESH events: the segment covering a just-happened event may
    // still be finalizing (its `.tmp.mp4` hasn't been indexed yet). Instead of a
    // hard 404 ("no recording in range"), retry briefly — the postprocessor now
    // finalizes within ~8-10s — so the clip plays on the first click, not "after
    // a while". Only retries when the window is recent (within the last ~30s of
    // now) to avoid delaying genuinely-empty historical requests.
    let query_first = || async {
        sqlx::query_as::<_, (String, String)>(
            "SELECT path, started_at FROM nvr_segments
             WHERE cam_id=? AND started_at <= ?
             ORDER BY started_at DESC LIMIT 1"
        )
        .bind(cam_id as i64).bind(&start_dt)
        .fetch_optional(db).await.unwrap_or(None)
    };
    let mut first_seg = query_first().await;
    if first_seg.is_none() {
        let now_secs = Utc::now().timestamp() as f64;
        let is_recent = (now_secs - start_secs) < 180.0; // within last 3 min
        if is_recent {
            for _ in 0..12 { // up to ~12s, polling every 1s
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                first_seg = query_first().await;
                if first_seg.is_some() { break; }
            }
        }
    }

    let (first_path, first_started, inpoint_secs) = match first_seg {
        Some((path, started)) => {
            let seg_unix = chrono::DateTime::parse_from_rfc3339(&started)
                .map(|t| t.timestamp() as f64).unwrap_or(start_secs);
            // Offset into the first segment. We DON'T trim here — callers pass this to
            // ffmpeg as `-ss` (output-side for re-encode, input-side for copy). The
            // concat-demuxer `inpoint` directive is broken for H.264 (it drops the
            // leading keyframe → black until the next one), so the manifest lists whole
            // segments and the seek is done by ffmpeg, the way mature NVRs do it.
            let inpoint = (start_secs - seg_unix).max(0.0);
            (path, started, inpoint)
        }
        None => {
            // No segment at/before start_secs. Accept the NEAREST segment that
            // starts AFTER it ONLY if it's close — a fresh event's covering
            // segment can be indexed a second or two later. Anything farther
            // means the requested time genuinely has no footage: return 404
            // rather than silently streaming a far-away (often another-DAY)
            // segment. The previous "else fall back to the most-recent segment"
            // branch was exactly what made an empty day play another day's
            // video. With this, DB+disk stay a truthful single source of truth
            // and the frontend shows its "camera was off" overlay instead.
            const FALLBACK_MAX_GAP: f64 = 120.0;
            let after: Option<(String, String)> = sqlx::query_as(
                "SELECT path, started_at FROM nvr_segments
                 WHERE cam_id=? AND started_at >= ? ORDER BY started_at ASC LIMIT 1"
            )
            .bind(cam_id as i64).bind(&start_dt).fetch_optional(db).await.unwrap_or(None);
            match after {
                Some((p, ts)) => {
                    let seg_unix = chrono::DateTime::parse_from_rfc3339(&ts)
                        .map(|t| t.timestamp() as f64).unwrap_or(start_secs);
                    if seg_unix - start_secs > FALLBACK_MAX_GAP {
                        return None;
                    }
                    (p, ts, 0.0)
                }
                None => return None,
            }
        }
    };

    // Step 2: Get all segments from first_started onwards.
    // If end_secs is bounded, stop at segments that start AFTER end_secs — they
    // contain nothing within the requested window. We still INCLUDE the segment
    // that overlaps end_secs because ffmpeg's `-t` flag handles the cutoff.
    let relevant: Vec<(String, String)> = if let Some(end_s) = end_secs {
        let end_dt = chrono::DateTime::<Utc>::from_timestamp(end_s.ceil() as i64 + 60, 0)
            .unwrap_or_else(Utc::now).to_rfc3339();
        sqlx::query_as(
            "SELECT path, started_at FROM nvr_segments
             WHERE cam_id=? AND started_at >= ? AND started_at <= ?
             ORDER BY started_at ASC"
        )
        .bind(cam_id as i64).bind(&first_started).bind(&end_dt)
        .fetch_all(db).await.unwrap_or_default()
    } else {
        sqlx::query_as(
            "SELECT path, started_at FROM nvr_segments
             WHERE cam_id=? AND started_at >= ?
             ORDER BY started_at ASC"
        )
        .bind(cam_id as i64).bind(&first_started)
        .fetch_all(db).await.unwrap_or_default()
    };

    // Keep a copy of the original first path for the inpoint comparison below
    // (it may be moved into the fallback vec).
    let inpoint_path = first_path.clone();
    // If the DB query returned nothing (e.g. requested time = exactly a segment boundary),
    // fall back to just playing the first segment from that point.
    let relevant = if relevant.is_empty() {
        vec![(first_path, first_started)]
    } else {
        relevant
    };

    // FOOL-PROOF: the DB is the index, but a row's file can be mid-rename
    // (.tmp.mp4 → .mp4), pruned, or never finalized. A single missing file makes
    // ffmpeg's concat demuxer FAIL THE WHOLE STREAM → the browser shows
    // "Clip not available" — then a later click works because the set changed.
    // So we keep only segments whose file ACTUALLY EXISTS on disk before building
    // the manifest. This is what makes DB+disk a consistent single source of truth.
    let mut relevant: Vec<(String, String)> = {
        let mut out = Vec::with_capacity(relevant.len());
        for (path, started) in relevant {
            if tokio::fs::try_exists(&path).await.unwrap_or(false) {
                out.push((path, started));
            }
        }
        out
    };
    // If every indexed file for the window is missing on disk, try the most recent
    // segment that DOES exist before giving up — never feed ffmpeg a bad manifest.
    if relevant.is_empty() {
        let candidates: Vec<(String, String)> = sqlx::query_as(
            "SELECT path, started_at FROM nvr_segments WHERE cam_id=? ORDER BY started_at DESC LIMIT 10"
        ).bind(cam_id as i64).fetch_all(db).await.unwrap_or_default();
        for (p, ts) in candidates {
            if tokio::fs::try_exists(&p).await.unwrap_or(false) { relevant.push((p, ts)); break; }
        }
        if relevant.is_empty() {
            return None;
        }
    }

    // Hardening: the first segment can be momentarily LOCKED on Windows — the
    // postprocess faststart pass writes nearby files, AV scanners hold handles,
    // or the file was just rotated. `try_exists` above only proves the path
    // exists, not that ffmpeg can open it; a locked first file makes the whole
    // concat fail and the browser shows MediaError 4 ("clip not available") even
    // though the recording is right there. A quick open-test with a couple of
    // short re-checks turns that hard failure into a sub-second wait. We never
    // hard-fail here — if it still won't open we proceed and let the client's
    // retry handle it.
    if let Some((first, _)) = relevant.first() {
        for attempt in 0..3 {
            if tokio::fs::File::open(first).await.is_ok() { break; }
            if attempt < 2 {
                tokio::time::sleep(std::time::Duration::from_millis(300)).await;
            }
        }
    }

    // Build ffmpeg concat manifest using exact paths from DB.
    // Include a unix-ns suffix so concurrent requests for the same cam don't race.
    let manifest_id = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0);
    let concat_path = data_dir.join(format!("_concat_{}_{}.txt", cam_id, manifest_id));
    let mut manifest = String::new();
    for (path, _started) in relevant.iter() {
        let escaped = path.replace('\\', "/");
        manifest.push_str(&format!("file '{}'\n", escaped));
    }
    if let Err(e) = tokio::fs::write(&concat_path, &manifest).await {
        tracing::warn!("nvr_concat: failed to write manifest: {e}");
        return None;
    }

    // Only honour the seek offset if the original first segment SURVIVED the existence
    // filter as index 0 — otherwise seeking would skip into the wrong file.
    let eff_inpoint = if relevant.first().map(|(p, _)| p == &inpoint_path).unwrap_or(false) {
        inpoint_secs
    } else { 0.0 };
    let out_dur = end_secs.map(|e| (e - start_secs).max(1.0));
    // DIAGNOSTIC: the exact concat inputs + seek so a 262-byte empty export is
    // explainable (inpoint past the only segment? duration ~0? wrong first file?).
    tracing::info!(
        "build_concat cam{}: segs={} inpoint={:.2} eff_inpoint={:.2} dur={:?} first={} last={}",
        cam_id, relevant.len(), inpoint_secs, eff_inpoint, out_dur,
        relevant.first().map(|(p,_)| p.rsplit(['/','\\']).next().unwrap_or("")).unwrap_or("-"),
        relevant.last().map(|(p,_)| p.rsplit(['/','\\']).next().unwrap_or("")).unwrap_or("-"),
    );
    Some((concat_path, out_dur, eff_inpoint))
}

/// How a glued (concat) stream should carry audio.
///
/// `Copy` keeps the native packets — fastest, but AAC priming leaves timestamp
/// holes at segment seams, so it's only fit for DOWNLOADS (players of saved
/// files tolerate seams; live <video> streaming of it stutters — that playback
/// moved to HLS VOD (`nvr_vod.rs`), which is seam-free by design).
/// `Reencode` produces ONE continuous AAC stream (`aresample=async=1` heals
/// every seam) with video still stream-copied — the recipe
/// `concat_window_to_file` proved clean (0 timestamp anomalies). Used by the
/// in-progress-event fallback where the <video> element streams this directly.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum ConcatAudio { Copy, Reencode }

/// Streams the bounded NVR window as a fragmented MP4 (browser-decodable
/// immediately). Used by exports/downloads (`Copy`) and the open-event
/// fallback in `footage::footage_clip` (`Reencode`). Seam-free PLAYBACK of
/// recorded windows lives in `nvr_vod.rs` (HLS VOD).
pub(crate) async fn stream_concat_window(
    s: &StreamState,
    cam_id: u8,
    start_secs: f64,
    end_secs: Option<f64>,
    audio_mode: ConcatAudio,
) -> Response {
    let (concat_path, duration, inpoint) =
        match build_concat_manifest(&s.db, &s.data_dir, cam_id, start_secs, end_secs).await {
            Some(p) => p,
            None => return StatusCode::NOT_FOUND.into_response(),
        };

    let ffmpeg = match ensure_ffmpeg(&s.data_dir).await {
        Ok(f) => f,
        Err(_) => {
            let _ = tokio::fs::remove_file(&concat_path).await;
            return StatusCode::INTERNAL_SERVER_ERROR.into_response();
        }
    };
    let concat_str = concat_path.to_string_lossy().replace('\\', "/");

    // INPUT-side `-ss` (before `-i`): with `-c copy` ffmpeg seeks to the keyframe at or
    // before the offset → a clean start (no black). Replaces the broken concat `inpoint`
    // directive. Segments are keyframe-aligned (~1s), so the start is tight.
    let mut args: Vec<String> = vec![
        "-hide_banner".into(), "-loglevel".into(), "error".into(),
    ];
    if inpoint > 0.05 {
        args.push("-ss".into());
        args.push(format!("{:.3}", inpoint));
    }
    // AUDIO: mapped only when the whole window is audible (uniform concat
    // layout); silent-era windows keep the proven video-only path. Mode:
    //  * Copy — native packets (downloads; seams tolerated in saved files).
    //  * Reencode — one continuous AAC stream via aresample=async=1 (heals the
    //    AAC-priming seam holes); `-max_interleave_delta` caps the muxer's
    //    audio-behind-encoder buffering so fMP4 interleave stays tight (the
    //    historical "bursty interleave → glitchy playback" regression guard).
    let audio = window_all_audio(&s.db, cam_id, start_secs, end_secs).await;
    args.extend([
        "-f".into(), "concat".into(), "-safe".into(), "0".into(),
        "-i".into(), concat_str,
        "-map".into(), "0:v:0".into(),
    ]);
    if audio {
        args.extend(["-map".into(), "0:a:0".into()]);
    }
    args.extend(["-c:v".into(), "copy".into()]);
    if audio {
        match audio_mode {
            ConcatAudio::Copy => args.extend(["-c:a".into(), "copy".into()]),
            ConcatAudio::Reencode => args.extend([
                "-c:a".into(), "aac".into(), "-b:a".into(), "96k".into(),
                "-af".into(), "aresample=async=1:first_pts=0".into(),
                "-max_interleave_delta".into(), "500000".into(),
            ]),
        }
    } else {
        args.push("-an".into());
    }
    args.extend(["-avoid_negative_ts".into(), "make_zero".into()]);
    args.extend([
        "-movflags".into(), "+frag_keyframe+default_base_moof".into(),
        "-f".into(), "mp4".into(),
    ]);
    // Bound the output duration when slicing an event window.
    if let Some(d) = duration {
        args.push("-t".into());
        args.push(format!("{:.3}", d));
    }
    args.push("pipe:1".into());
    tracing::info!("concat stream args: {}", args.join(" "));

    let child = crate::proc::tokio_cmd(&ffmpeg)
        .args(&args)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        .spawn();

    match child {
        Ok(mut proc) => {
            // Surface the concat ffmpeg's own errors — a silent audio-map failure
            // is indistinguishable from "no audio in source" without this.
            if let Some(stderr) = proc.stderr.take() {
                tokio::spawn(async move {
                    use tokio::io::{AsyncBufReadExt, BufReader};
                    let mut lines = BufReader::new(stderr).lines();
                    while let Ok(Some(line)) = lines.next_line().await {
                        let l = line.trim();
                        if !l.is_empty() { tracing::warn!("concat ffmpeg: {}", l); }
                    }
                });
            }
            let stdout = match proc.stdout.take() {
                Some(s) => s, None => return StatusCode::INTERNAL_SERVER_ERROR.into_response(),
            };
            let concat_cleanup = concat_path.clone();
            tokio::spawn(async move {
                let _ = proc.wait().await;
                let _ = tokio::fs::remove_file(&concat_cleanup).await;
            });
            let stream  = tokio_util::io::ReaderStream::new(stdout);
            let mut headers = HeaderMap::new();
            headers.insert("Content-Type", HeaderValue::from_static("video/mp4"));
            headers.insert("Cache-Control", HeaderValue::from_static("no-cache"));
            (StatusCode::OK, headers, Body::from_stream(stream)).into_response()
        }
        Err(e) => {
            tracing::warn!("nvr_concat ffmpeg error: {e}");
            let _ = tokio::fs::remove_file(&concat_path).await;
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

/// v14: like `stream_concat_window`, but writes the bounded window to a real
/// H.264 MP4 **file** (`-movflags +faststart`, moov atom at the front) instead
/// of a fragmented stream. Used by `agent::clip_export::ensure_event_clip` to
/// produce the clip the agent SENDS (Telegram / share link) — a native
/// inline-playable preview, replacing the old browser-recorded `.webm`.
/// Returns true when the output file was written. Caller owns `out_path`.
pub(crate) async fn concat_window_to_file(
    db: &sqlx::SqlitePool,
    data_dir: &std::path::Path,
    cam_id: u8,
    start_secs: f64,
    end_secs: f64,
    out_path: &std::path::Path,
) -> bool {
    let (concat_path, duration, inpoint) =
        match build_concat_manifest(db, data_dir, cam_id, start_secs, Some(end_secs)).await {
            Some(p) => p,
            None => return false,
        };

    let ffmpeg = match ensure_ffmpeg(data_dir).await {
        Ok(f) => f,
        Err(_) => {
            let _ = tokio::fs::remove_file(&concat_path).await;
            return false;
        }
    };
    let concat_str = concat_path.to_string_lossy().replace('\\', "/");
    // Encode to a UNIQUE temp file, then atomically rename to the final path. Several
    // callers can target the same deterministic `clip_{id}.mp4` at once; concurrent
    // ffmpeg writers to ONE file interleave into a corrupt, unplayable clip (this is
    // what corrupted event clips). A per-writer temp + rename guarantees the final
    // path only ever appears as a COMPLETE file. (single-flight in ensure_event_clip
    // removes the common race; this is belt-and-suspenders for every other caller.)
    let tmp_path = out_path.with_file_name(format!(
        "{}.tmp.{}.{}.mp4",
        out_path.file_stem().and_then(|s| s.to_str()).unwrap_or("clip"),
        std::process::id(),
        Utc::now().timestamp_nanos_opt().unwrap_or(0),
    ));
    let out_str = tmp_path.to_string_lossy().to_string();

    // RE-ENCODE + OUTPUT-side `-ss` (AFTER `-i`): ffmpeg decodes from the first
    // segment's keyframe and emits frames only from the seek point → FRAME-ACCURATE
    // start with ZERO black, and `-t` gives the exact length with no frozen tail.
    // (The concat `inpoint` directive is broken for H.264 — it drops the leading
    // keyframe; `-ss` is the mature NVRs-correct way to trim.) Cached once per event.
    let mut args: Vec<String> = vec![
        "-hide_banner".into(), "-loglevel".into(), "error".into(),
        "-y".into(),
        "-f".into(), "concat".into(), "-safe".into(), "0".into(),
        "-i".into(), concat_str,
    ];
    if inpoint > 0.05 {
        args.push("-ss".into());
        args.push(format!("{:.3}", inpoint));
    }
    if let Some(d) = duration {
        args.push("-t".into());
        args.push(format!("{:.3}", d));
    }
    // AUDIO: when the whole window is audible, decode+re-encode the audio with
    // async resampling — clean timestamps, no DTS stall (the old hang was
    // copy-concat only). Silent-era windows stay video-only and get audio from
    // the ring-buffer mux afterwards (which self-guards against double audio).
    let audio = window_all_audio(db, cam_id, start_secs, Some(end_secs)).await;
    args.extend([
        "-map".into(), "0:v:0".into(),
        // Force LIMITED ("tv") color range. USB cameras feed JPEG frames = FULL
        // range (yuvj420p), which x264 preserves even with `-pix_fmt yuv420p` —
        // and Telegram's mobile decoders reject full-range H.264 with "can't
        // play this format". `scale=out_range=tv` auto-reads the input range
        // (full→tv for USB, no-op for already-tv RTSP); `-color_range tv` tags
        // the output. Verified: turns yuvj420p(pc) → yuv420p.
        "-vf".into(), "scale=out_range=tv".into(),
        "-c:v".into(), "libx264".into(),
        "-preset".into(), "veryfast".into(),
        "-crf".into(), "23".into(),
        "-pix_fmt".into(), "yuv420p".into(),
        "-color_range".into(), "tv".into(),
        // Cap x264 threads (default ~1.5× cores each holding frame buffers).
        "-threads".into(), "2".into(),
    ]);
    if audio {
        args.extend([
            "-map".into(), "0:a:0".into(),
            "-c:a".into(), "aac".into(), "-b:a".into(), "96k".into(),
            "-af".into(), "aresample=async=1:first_pts=0".into(),
        ]);
    } else {
        args.push("-an".into());
    }
    args.extend([
        "-movflags".into(), "+faststart".into(),
        out_str,
    ]);

    // Capture stderr (was /dev/null) so a failed encode is visible, not silent.
    // HARD TIMEOUT: ffmpeg can hang indefinitely on a malformed/locked input; with no
    // cap, the awaiting caller wedges FOREVER — exactly what froze the Telegram poll
    // loop, which calls this inline. On timeout the future drops and `kill_on_drop`
    // kills the child, so no caller can be held hostage by a stuck encode.
    let run = crate::proc::tokio_cmd(&ffmpeg)
        .args(&args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        .output();
    let output = match tokio::time::timeout(std::time::Duration::from_secs(300), run).await {
        Ok(r) => r,
        Err(_) => {
            tracing::warn!("clip export timed out after 300s (ffmpeg killed): {}", tmp_path.display());
            let _ = tokio::fs::remove_file(&concat_path).await;
            let _ = tokio::fs::remove_file(&tmp_path).await;
            return false;
        }
    };
    let _ = tokio::fs::remove_file(&concat_path).await;

    let ok = matches!(&output, Ok(o) if o.status.success());
    // A 0-byte / tiny output means ffmpeg "succeeded" but encoded nothing (e.g. the
    // window's footage isn't recorded yet) — treat it as FAILURE so the caller never
    // caches an empty clip that would later serve as "no footage".
    let size = tokio::fs::metadata(&tmp_path).await.map(|m| m.len()).unwrap_or(0);
    if !ok || size < 4096 {
        if let Ok(o) = &output {
            let err = String::from_utf8_lossy(&o.stderr);
            let tail: String = err.lines().rev().take(4).collect::<Vec<_>>().into_iter().rev().collect::<Vec<_>>().join(" | ");
            tracing::warn!("clip export produced {} bytes (ffmpeg ok={}): {}", size, ok, tail);
        }
        let _ = tokio::fs::remove_file(&tmp_path).await;
        return false;
    }
    // Atomic publish: `out_path` flips from absent/old → COMPLETE in one rename, so no
    // reader ever observes a half-written or interleaved file. std::fs::rename replaces
    // an existing destination on Windows (MOVEFILE_REPLACE_EXISTING).
    if let Err(e) = tokio::fs::rename(&tmp_path, out_path).await {
        tracing::warn!("clip export rename {} -> {} failed: {e}", tmp_path.display(), out_path.display());
        let _ = tokio::fs::remove_file(&tmp_path).await;
        return false;
    }
    true
}

