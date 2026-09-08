//! HLS VOD playback for recorded NVR footage — the mature NVRs model.
//!
//! Why this exists: gluing 10s mp4 segments into ONE stream (`/nvr-concat`,
//! pure `-c copy`) leaves AAC-priming holes in the audio timeline at every
//! seam (measured: gaps up to 72 ms at nearly every boundary) plus a frozen
//! keyframe-aligned video head after seeks. Chromium stalls on audio holes →
//! pause/play stutter. The industry answer (mature NVRs' nginx-vod-module) is to
//! never glue: serve a VOD playlist over the segments and remux each one to
//! MPEG-TS **individually** (`-c copy`, zero transcode); the player (hls.js)
//! stitches at seams — which is exactly what HLS players are built to do.
//!
//! Two routes, both behind the global `require_token` middleware:
//!   GET /nvr-vod/:cam/playlist.m3u8?start=UNIX_SECS&window=SECS&token=…
//!   GET /nvr-vod/seg/:id.ts?token=…

use axum::{
    body::Body,
    extract::State as AxumState,
    http::{HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
};
use chrono::Utc;

use crate::{ensure_ffmpeg, StreamState};

/// One playlist row: everything the m3u8 needs about a segment.
struct VodSeg {
    id: String,
    start_unix: f64,
    /// RFC3339 `started_at` straight from the DB (PDT tag value).
    started_at: String,
    duration: f64,
    has_audio: bool,
}

/// GET /nvr-vod/:cam/playlist.m3u8?start&window&token
///
/// Emits a VOD playlist over the indexed segments covering
/// `[start, start+window]`. Mirrors `build_concat_manifest`'s fool-proof
/// window logic (covering-segment with 120s plausibility, nearest-after
/// fallback, on-disk existence filter) — but instead of a concat manifest the
/// output is standard HLS the player assembles itself:
///  * `EXT-X-START` puts the playhead exactly at the requested wall-clock
///    (PRECISE lets hls.js start mid-segment) — no frontend seek code.
///  * `EXT-X-PROGRAM-DATE-TIME` per segment anchors media time to wall-clock
///    (`hls.playingDate`), exact even across gaps.
///  * `EXT-X-DISCONTINUITY` at recording gaps AND `has_audio` flips, so a
///    window straddling the silent-era archive plays video throughout and
///    gains sound where it exists — no more all-or-nothing audio gate.
pub(crate) async fn nvr_vod_playlist(
    axum::extract::Path(cam): axum::extract::Path<String>,
    AxumState(s): AxumState<StreamState>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Response {
    let cam_id: u8 = match cam.parse() {
        Ok(c) => c,
        Err(_) => return StatusCode::BAD_REQUEST.into_response(),
    };
    let start_secs = params.get("start").and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0);
    let window = params.get("window")
        .and_then(|v| v.parse::<f64>().ok())
        .map(|w| w.clamp(30.0, 3600.0))
        .unwrap_or(crate::nvr_stream::NVR_CONCAT_WINDOW_SECS);
    let end_secs = start_secs + window;
    // Echo the caller's token into segment URIs — hls.js fetches them as plain
    // GETs and the global middleware wants `?token=` on every request.
    let token = params.get("token").cloned().unwrap_or_default();

    let segs = match vod_window_segments(&s.db, cam_id, start_secs, end_secs).await {
        Some(v) if !v.is_empty() => v,
        _ => return StatusCode::NOT_FOUND.into_response(),
    };

    // Requested start relative to the first segment: hls.js honors EXT-X-START
    // (default startPosition:-1), so playback begins mid-segment when asked.
    //
    // TIME-OFFSET is MEDIA time (the running sum of EXTINF), NOT wall clock. The
    // two diverge by exactly the size of every hole in the window — and this
    // playlist emits `#EXT-X-DISCONTINUITY` a few lines down precisely because
    // holes happen. `vod_window_segments` also accepts a covering segment up to
    // 120 s before the requested time, and EXTINF is only ever *shortened* to real
    // spacing, never lengthened across a gap. So a wall-clock offset always ran
    // LONG: the playhead landed past where the user clicked. Sum the durations
    // ahead of the covering segment, then add the offset into it.
    let k = segs.iter().rposition(|x| x.start_unix <= start_secs).unwrap_or(0);
    let start_offset = segs[..k].iter().map(|x| x.duration).sum::<f64>()
        + (start_secs - segs[k].start_unix).max(0.0);
    let target = segs.iter().map(|x| x.duration).fold(0.0f64, f64::max).ceil().max(1.0) as u64;

    let mut m3u8 = String::with_capacity(256 + segs.len() * 160);
    m3u8.push_str("#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-PLAYLIST-TYPE:VOD\n");
    m3u8.push_str(&format!("#EXT-X-TARGETDURATION:{}\n", target));
    if start_offset > 0.05 {
        m3u8.push_str(&format!("#EXT-X-START:TIME-OFFSET={:.3},PRECISE=YES\n", start_offset));
    }
    for (i, seg) in segs.iter().enumerate() {
        if i > 0 {
            let prev = &segs[i - 1];
            let gap = seg.start_unix - (prev.start_unix + prev.duration);
            if gap > 1.5 || seg.has_audio != prev.has_audio {
                // Recording gap or track-set change (audio appears/disappears):
                // tell the player to reinitialize rather than stall.
                m3u8.push_str("#EXT-X-DISCONTINUITY\n");
            }
        }
        m3u8.push_str(&format!("#EXT-X-PROGRAM-DATE-TIME:{}\n", seg.started_at));
        m3u8.push_str(&format!("#EXTINF:{:.3},\n", seg.duration));
        m3u8.push_str(&format!("/nvr-vod/seg/{}.ts?token={}\n", seg.id, token));
    }
    m3u8.push_str("#EXT-X-ENDLIST\n");

    let mut headers = HeaderMap::new();
    headers.insert("Content-Type", HeaderValue::from_static("application/vnd.apple.mpegurl"));
    headers.insert("Cache-Control", HeaderValue::from_static("no-cache"));
    headers.insert("Access-Control-Allow-Origin", HeaderValue::from_static("*"));
    (StatusCode::OK, headers, Body::from(m3u8)).into_response()
}

/// Per-segment "needs 4:2:0 transcode" verdict — cached because segments are
/// immutable, so the answer never changes. LEGACY footage recorded before the
/// USB recorder forced 4:2:0 is H.264 4:4:4 (`yuvj444p`), which Chromium/WebView2
/// CANNOT decode — the NVR timeline played black. Those segments get re-encoded to
/// 4:2:0 on serve; all new (4:2:0) footage keeps the zero-cost `-c copy` path. The
/// verdict is probed once via `ffmpeg -i` stderr (we ship ffmpeg, not ffprobe —
/// same approach as `file_has_audio`).
static SEG_NEEDS_420: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<String, bool>>> =
    std::sync::OnceLock::new();

async fn segment_needs_420_transcode(ffmpeg: &std::path::Path, seg_id: &str, path: &str) -> bool {
    if let Some(v) = SEG_NEEDS_420.get_or_init(Default::default).lock().unwrap().get(seg_id).copied() {
        return v;
    }
    // Single-flight: the probe is a whole awaited ffmpeg process and the cache is
    // only written after it returns, so concurrent requests for the same cold
    // segment each spawned their own.
    //
    // ponytail: one global gate, not per-id. The probe only fires for cold LEGACY
    // 4:4:4 segments, so contention is negligible; store the verdict as a column
    // on `nvr_segments` at index time if it ever shows up in a profile.
    static PROBE_GATE: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
    let _permit = PROBE_GATE.lock().await;
    if let Some(v) = SEG_NEEDS_420.get_or_init(Default::default).lock().unwrap().get(seg_id).copied() {
        return v; // filled while we waited on the gate
    }
    let out = crate::proc::tokio_cmd(ffmpeg)
        .args(["-hide_banner", "-i", path])
        .output().await;
    let needs = match out {
        Ok(o) => {
            let stderr = String::from_utf8_lossy(&o.stderr);
            let vline = stderr.lines().find(|l| l.contains("Video:")).unwrap_or("");
            // 4:2:0 tokens (yuv420p/yuvj420p/nv12) contain "420" → decodable, copy.
            // Only clear 4:4:4 / 4:2:2 needs a transcode; anything ambiguous stays copy.
            if vline.contains("420") { false } else { vline.contains("444") || vline.contains("422") }
        }
        Err(_) => false,
    };
    {
        let mut m = SEG_NEEDS_420.get_or_init(Default::default).lock().unwrap();
        // Bound the cache across a long session — heavy scrubbing over days adds
        // one entry per distinct segment served, and old segments age out of
        // retention anyway. Wholesale clear is fine: re-probing is one ffmpeg -i.
        if m.len() > 4096 { m.clear(); }
        m.insert(seg_id.to_string(), needs);
    }
    needs
}

/// GET /nvr-vod/seg/:file  (`file` = `<segment-uuid>.ts`)
///
/// Remuxes ONE indexed segment mp4 → MPEG-TS, stream-copy only (ffmpeg
/// auto-inserts h264_mp4toannexb + AAC→ADTS for mpegts). Segments are
/// immutable once indexed, so responses are cacheable — that also absorbs
/// ClipOverlay's blurred-clone double-fetch. Legacy 4:4:4 segments are the one
/// exception: they're transcoded to 4:2:0 so WebView2 can decode them.
pub(crate) async fn nvr_vod_segment(
    axum::extract::Path(file): axum::extract::Path<String>,
    AxumState(s): AxumState<StreamState>,
) -> Response {
    let id = match file.strip_suffix(".ts") {
        Some(v) => v,
        None => return StatusCode::BAD_REQUEST.into_response(),
    };
    if id.is_empty() || id.len() > 64
        || !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
        return StatusCode::FORBIDDEN.into_response();
    }

    let row: Option<(String, i64)> = sqlx::query_as(
        "SELECT path, has_audio FROM nvr_segments WHERE id=? LIMIT 1")
        .bind(id)
        .fetch_optional(&s.db).await.unwrap_or(None);
    let (path, has_audio) = match row {
        Some((p, a)) => (p, a == 1),
        None => return StatusCode::NOT_FOUND.into_response(),
    };
    if !tokio::fs::try_exists(&path).await.unwrap_or(false) {
        return StatusCode::NOT_FOUND.into_response();
    }
    let ffmpeg = match ensure_ffmpeg(&s.data_dir).await {
        Ok(f) => f,
        Err(_) => return StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    };

    // Legacy 4:4:4 segments can't be decoded by WebView2 — re-encode those to 4:2:0.
    // All new footage is 4:2:0 and takes the zero-transcode copy path.
    let needs_420 = segment_needs_420_transcode(&ffmpeg, id, &path).await;

    let mut args: Vec<String> = vec![
        "-hide_banner".into(), "-loglevel".into(), "error".into(),
        "-i".into(), path,
        "-map".into(), "0:v:0".into(),
    ];
    if has_audio {
        args.extend(["-map".into(), "0:a:0".into()]);
    }
    if needs_420 {
        args.extend([
            "-c:v".into(), "libx264".into(),
            "-preset".into(), "veryfast".into(),
            "-crf".into(), "23".into(),
            "-pix_fmt".into(), "yuv420p".into(),
            // Cap x264 threads (default ~1.5× cores each holding frame buffers).
            "-threads".into(), "2".into(),
        ]);
        if has_audio { args.extend(["-c:a".into(), "copy".into()]); }
    } else {
        args.extend(["-c".into(), "copy".into()]);
    }
    args.extend([
        "-muxdelay".into(), "0".into(), "-muxpreload".into(), "0".into(),
        "-f".into(), "mpegts".into(), "pipe:1".into(),
    ]);

    stream_ffmpeg_stdout(&ffmpeg, &args, "video/MP2T", "private, max-age=3600")
}

/// Spawn ffmpeg with `args` and stream its stdout as an HTTP response —
/// the shared spawn-to-HTTP pattern (`kill_on_drop` so a client disconnect
/// reaps the process; ACAO:* so hls.js can fetch cross-origin).
fn stream_ffmpeg_stdout(
    ffmpeg: &std::path::Path,
    args: &[String],
    content_type: &'static str,
    cache_control: &'static str,
) -> Response {
    let started = std::time::Instant::now();
    let child = crate::proc::tokio_cmd(ffmpeg)
        .args(args)
        .stdout(std::process::Stdio::piped())
        // stderr was Stdio::null(), so a failed remux was completely invisible:
        // the client got a truncated body, hls.js turned that into a parse error
        // and silently rebuilt its whole buffer, and nothing was logged anywhere.
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        .spawn();
    match child {
        Ok(mut proc) => {
            let stdout = match proc.stdout.take() {
                Some(o) => o,
                None => return StatusCode::INTERNAL_SERVER_ERROR.into_response(),
            };
            let stderr = proc.stderr.take();
            tokio::spawn(async move {
                let status = proc.wait().await;
                let failed = !matches!(&status, Ok(s) if s.success());
                if failed {
                    let mut msg = String::new();
                    if let Some(mut e) = stderr {
                        use tokio::io::AsyncReadExt;
                        let _ = e.read_to_string(&mut msg).await;
                    }
                    // A client disconnect (seek away, player torn down) also lands
                    // here via kill_on_drop, so this is debug, not warn.
                    tracing::debug!("nvr_vod remux ended {status:?} after {:?}: {}",
                                    started.elapsed(), msg.trim());
                }
            });
            let mut headers = HeaderMap::new();
            headers.insert("Content-Type", HeaderValue::from_static(content_type));
            headers.insert("Cache-Control", HeaderValue::from_static(cache_control));
            headers.insert("Access-Control-Allow-Origin", HeaderValue::from_static("*"));
            (StatusCode::OK, headers,
             Body::from_stream(tokio_util::io::ReaderStream::new(stdout))).into_response()
        }
        Err(e) => {
            tracing::warn!("nvr_vod ffmpeg spawn error: {e}");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

/// Segments overlapping `[start, end]`, existence-filtered, with real
/// durations. Covering segment (`started_at <= start`, plausible offset) else
/// the first segment inside the window, then rows forward to `end`+slack.
async fn vod_window_segments(
    db: &sqlx::SqlitePool,
    cam_id: u8,
    start_secs: f64,
    end_secs: f64,
) -> Option<Vec<VodSeg>> {
    let start_dt = chrono::DateTime::<Utc>::from_timestamp(start_secs as i64, 0)
        .unwrap_or_else(Utc::now).to_rfc3339();

    // Covering segment, else nearest-after (bounded — never another day's video).
    let covering: Option<(String,)> = sqlx::query_as(
        "SELECT started_at FROM nvr_segments
         WHERE cam_id=? AND started_at <= ? ORDER BY started_at DESC LIMIT 1")
        .bind(cam_id as i64).bind(&start_dt)
        .fetch_optional(db).await.unwrap_or(None);
    let first_started = match covering {
        Some((ts,)) => {
            let seg_unix = chrono::DateTime::parse_from_rfc3339(&ts)
                .map(|t| t.timestamp() as f64).unwrap_or(start_secs);
            if start_secs - seg_unix < 120.0 { Some(ts) } else { None }
        }
        None => None,
    };
    let first_started = match first_started {
        Some(ts) => ts,
        None => {
            // Nothing covers `start`. Take the first segment INSIDE THE REQUESTED
            // WINDOW.
            //
            // This used to be bounded by a fixed 120 s instead, which was right
            // when `start` was an arbitrary anchor snapped onto real footage: the
            // bound was what stopped a stray timestamp from dragging in another
            // day's video. The client now asks for a fixed, clock-aligned chunk,
            // so `start` routinely lands in dead air — and a camera that began
            // recording more than two minutes into the half hour 404'd the whole
            // chunk despite plainly having footage in it.
            //
            // The window is its own bound (clamped to 3600 s above), so "within
            // the window" is both the correct rule and a tighter guarantee than
            // the constant it replaces.
            let after: Option<(String,)> = sqlx::query_as(
                "SELECT started_at FROM nvr_segments
                 WHERE cam_id=? AND started_at >= ? ORDER BY started_at ASC LIMIT 1")
                .bind(cam_id as i64).bind(&start_dt)
                .fetch_optional(db).await.unwrap_or(None);
            match after {
                Some((ts,)) => {
                    let seg_unix = chrono::DateTime::parse_from_rfc3339(&ts)
                        .map(|t| t.timestamp() as f64).unwrap_or(f64::MAX);
                    if seg_unix > end_secs { return None; }
                    ts
                }
                None => return None,
            }
        }
    };

    let end_dt = chrono::DateTime::<Utc>::from_timestamp(end_secs.ceil() as i64 + 60, 0)
        .unwrap_or_else(Utc::now).to_rfc3339();
    let rows: Vec<(String, String, String, Option<f64>, i64)> = sqlx::query_as(
        "SELECT id, path, started_at, duration_secs, has_audio FROM nvr_segments
         WHERE cam_id=? AND started_at >= ? AND started_at <= ?
         ORDER BY started_at ASC")
        .bind(cam_id as i64).bind(&first_started).bind(&end_dt)
        .fetch_all(db).await.unwrap_or_default();

    let mut segs: Vec<VodSeg> = Vec::with_capacity(rows.len());
    for (id, path, started_at, duration_secs, has_audio) in rows {
        if !tokio::fs::try_exists(&path).await.unwrap_or(false) { continue; }
        let start_unix = chrono::DateTime::parse_from_rfc3339(&started_at)
            .map(|t| t.timestamp_millis() as f64 / 1000.0)
            .unwrap_or(0.0);
        segs.push(VodSeg {
            id, start_unix, started_at,
            duration: duration_secs.filter(|d| *d > 0.5).unwrap_or(10.0),
            has_audio: has_audio == 1,
        });
    }
    normalize_extinf(&mut segs);
    Some(segs)
}

/// EXTINF may only ever be SHORTENED, never lengthened.
///
/// EXTINF is a promise about how much media a fragment contains. Over-promising
/// is not a rounding error, it is a hole: hls.js appends the real media, finds the
/// buffer short of where the next fragment claims to start, and its GapController
/// escapes by HARD-SEEKING the element forward. That is "the video jumps forward
/// when I seek".
///
/// A previous pass set EXTINF to the spacing to the next segment in BOTH
/// directions, reasoning that spacing is the true elapsed time. It is — but the
/// elapsed time between two segment STARTS is not the media inside the first one.
/// When the recorder stalls, the difference is a genuine outage. Measured on the
/// live archive with `ffmpeg -i`:
///
/// ```text
///   segment 6b35ef98   real media 9.97 s   spacing 19 s  ->  9 s declared that does not exist
///   segment 9a92351d   real media 10.0 s   spacing 14 s  ->  4 s declared that does not exist
/// ```
///
/// Lengthening also silently disabled the discontinuity emitter: after it,
/// `seg.start_unix - (prev.start_unix + prev.duration)` is 0 by construction, so a
/// real recording outage was absorbed into one over-long fragment instead of
/// getting the `#EXT-X-DISCONTINUITY` it needs.
///
/// Under-promising is safe by comparison — the buffer simply runs longer than
/// declared. And cumulative EXTINF drift does NOT hurt wall-clock seeking here,
/// because `mediaTimeForDate`/`dateForMediaTime` map through the CONTAINING
/// fragment's own PROGRAM-DATE-TIME rather than summing EXTINF, so any error stays
/// bounded by one fragment instead of accumulating.
fn normalize_extinf(segs: &mut [VodSeg]) {
    for i in 0..segs.len().saturating_sub(1) {
        let spacing = segs[i + 1].start_unix - segs[i].start_unix;
        if spacing > 0.5 && spacing < segs[i].duration {
            segs[i].duration = spacing;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(start: f64, duration: f64) -> VodSeg {
        VodSeg {
            id: format!("s{start}"), start_unix: start,
            started_at: String::new(), duration, has_audio: false,
        }
    }

    /// THE invariant: EXTINF must never exceed the spacing to the next segment,
    /// and must never be lengthened past what the recorder claims to have written.
    /// Over-promising is a buffer hole, and hls.js escapes a hole by hard-seeking
    /// the element forward.
    #[test]
    fn extinf_is_only_ever_shortened() {
        let mut segs = vec![seg(0.0, 12.0), seg(10.0, 10.0), seg(20.0, 10.0)];
        normalize_extinf(&mut segs);
        assert_eq!(segs[0].duration, 10.0, "over-reported duration shortened to spacing");
        assert_eq!(segs[1].duration, 10.0, "an accurate duration is left alone");
        assert_eq!(segs[2].duration, 10.0, "last segment keeps its own duration");
    }

    /// The regression this file shipped and then had to undo. A 10 s segment
    /// followed 19 s later means a 9 s OUTAGE, not a 19 s segment — the file was
    /// probed with `ffmpeg -i` and holds 9.97 s. Declaring 19 s made hls.js
    /// gap-jump the playhead forward out of the hole.
    #[test]
    fn a_short_outage_is_never_absorbed_into_extinf() {
        let mut segs = vec![seg(0.0, 10.0), seg(19.0, 10.0)];
        normalize_extinf(&mut segs);
        assert_eq!(segs[0].duration, 10.0, "must not claim media the file does not contain");
        // What the playlist writer sees, and why it can now emit the discontinuity.
        let gap = segs[1].start_unix - (segs[0].start_unix + segs[0].duration);
        assert!(gap > 1.5, "a real outage must still register as a gap, got {gap}");
    }

    /// The same rule at a scale nobody disputes.
    #[test]
    fn a_long_recording_gap_is_not_swallowed_into_extinf() {
        let mut segs = vec![seg(0.0, 10.0), seg(300.0, 10.0)];
        normalize_extinf(&mut segs);
        assert_eq!(segs[0].duration, 10.0, "the gap must stay a gap");
    }
}
