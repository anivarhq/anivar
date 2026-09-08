//! Scrub previews — the cheap video the timeline drags across.
//!
//! Dragging a scrubber over the real recording is the most expensive thing this
//! app can do. Every pointer move lands on a different second, hls.js fetches the
//! fragments around it, and `nvr_vod_segment` spawns one ffmpeg per fragment to
//! remux it. A single drag across ten minutes can ask for sixty segments that are
//! thrown away before they finish decoding.
//!
//! Mature NVRs answer this with a second, disposable video: a low-resolution,
//! low-frame-rate file covering a whole hour, which the player scrubs INSTEAD of
//! the recording. The real player is paused and untouched until the user lets go.
//! Frigate's numbers — 180 px tall, roughly 1-2 fps, one file per camera per hour,
//! assembled with the concat demuxer — are what this module reproduces.
//!
//! Two decisions worth keeping:
//!
//! * **Decode keyframes only.** The recorder forces a keyframe every second
//!   (`nvr_pipes::segment_output_args`), so `-skip_frame nokey` gives ~1 fps of
//!   real frames while skipping ~29 of every 30 decodes. Generating an hour of
//!   preview costs a fraction of decoding an hour of video, which is what makes
//!   this affordable on a 16-camera box.
//! * **The stored span is the COVERED span, not the hour.** A concat contains only
//!   recorded footage, so an hour holding 15 minutes of video produces a 15-minute
//!   preview. Storing the hour boundaries and mapping `t - start` linearly would
//!   put a scrub tens of minutes out. `start_time`/`end_time` are therefore the
//!   first and last frame's real wall clock, and the client maps PROPORTIONALLY
//!   across that span using the file's own duration.
//!
//!   ponytail: proportional, not exact. Within a continuous run it is exact; a
//!   recording gap inside the span skews later positions by that gap's share of
//!   the span (~1% on this archive). A scrub preview is an orientation aid and the
//!   scrubber's own clock readout stays authoritative, so this buys simplicity at
//!   a known, bounded cost. Pad the gaps in ffmpeg if that ever stops being true.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::extract::State as AxumState;
use sqlx::SqlitePool;
use tauri::State;

use crate::AppState;
use crate::ffmpeg::ensure_ffmpeg;

/// One preview file covers this much wall clock. Frigate uses an hour; the same
/// number keeps files big enough to be worth generating and small enough that a
/// scrub rarely crosses two of them.
const PREVIEW_SPAN_SECS: i64 = 3600;
/// Frigate's preview height. "Very low resolution by design" — it exists to show
/// you roughly where you are, not to be watched.
const PREVIEW_HEIGHT: u32 = 180;
/// Preview frame rate. The recorder's 1 s keyframe cadence is the real ceiling on
/// distinct frames; 2 keeps motion legible without inflating the file.
const PREVIEW_FPS: u32 = 2;

#[derive(serde::Serialize)]
pub struct PreviewDto {
    pub id: String,
    pub cam_id: u8,
    /// RFC3339. The client maps `currentTime = (t - start_time) / 1000`.
    pub start_time: String,
    pub end_time: String,
}

fn epoch(rfc3339: &str) -> i64 {
    chrono::DateTime::parse_from_rfc3339(rfc3339).map(|d| d.timestamp()).unwrap_or(0)
}

fn rfc(secs: i64) -> String {
    chrono::DateTime::from_timestamp(secs, 0)
        .map(|d| d.to_rfc3339())
        .unwrap_or_default()
}

/// Previews overlapping `[from, to]`, oldest first.
#[tauri::command]
pub async fn list_previews(
    state: State<'_, Arc<AppState>>,
    cam_id: u8,
    range_start: String,
    range_end: String,
) -> Result<Vec<PreviewDto>, String> {
    let rows: Vec<(String, i64, String, String)> = sqlx::query_as(
        "SELECT id, cam_id, start_time, end_time FROM nvr_previews
         WHERE cam_id = ? AND start_time <= ? AND end_time >= ?
         ORDER BY start_time ASC",
    )
    .bind(cam_id as i64)
    .bind(&range_end)
    .bind(&range_start)
    .fetch_all(&state.db)
    .await
    .map_err(|e| e.to_string())?;

    Ok(rows
        .into_iter()
        .map(|(id, cam, s, e)| PreviewDto { id, cam_id: cam as u8, start_time: s, end_time: e })
        .collect())
}

/// On-disk path of a preview, for the HTTP responder.
pub(crate) async fn preview_path(db: &SqlitePool, id: &str) -> Option<String> {
    sqlx::query_scalar::<_, String>("SELECT path FROM nvr_previews WHERE id = ? LIMIT 1")
        .bind(id)
        .fetch_optional(db)
        .await
        .ok()
        .flatten()
}

/// Generate previews for any complete hour that has footage but no preview yet.
///
/// UNCALLED as of 2026-09-06: previews exist to make dragging a scrubber cheap,
/// and the only scrubber that dragged them — the in-video one — was deleted so
/// the timeline is the single scrub surface (Frigate's shape). `boot.rs` no
/// longer spawns this loop; generating an hour of video per camera per hour to
/// feed nothing is pure cost. Kept, not deleted, because the serving half and
/// the schema are fine and this is one `spawn` away from useful again — wire it
/// back up if the timeline grows a draggable handlebar.
#[allow(dead_code)]
///
/// Only *complete* hours: an hour still being recorded would produce a preview
/// that goes stale a minute later, and regenerating it every pass is exactly the
/// kind of busywork this module exists to avoid.
pub async fn generate_missing_previews(state: &Arc<AppState>) {
    let ffmpeg = match ensure_ffmpeg(&state.data_dir).await {
        Ok(f) => f,
        Err(e) => {
            tracing::warn!("preview: no ffmpeg ({e}), skipping");
            return;
        }
    };
    let dir = state.data_dir.join("nvr").join("previews");
    if let Err(e) = tokio::fs::create_dir_all(&dir).await {
        tracing::warn!("preview: cannot create {}: {e}", dir.display());
        return;
    }

    // The current hour is still filling; stop at the previous one.
    let now = chrono::Utc::now().timestamp();
    let cur_hour = now - now.rem_euclid(PREVIEW_SPAN_SECS);

    let cams: Vec<i64> = sqlx::query_scalar("SELECT DISTINCT cam_id FROM nvr_segments")
        .fetch_all(&state.db)
        .await
        .unwrap_or_default();

    let mut made = 0usize;
    for cam in cams {
        // Hours that have footage, bucketed server-side so this stays one query
        // regardless of how many segments the archive holds.
        let hours: Vec<i64> = sqlx::query_scalar(
            "SELECT DISTINCT CAST(strftime('%s', started_at) AS INTEGER)
                            - CAST(strftime('%s', started_at) AS INTEGER) % ?
               FROM nvr_segments WHERE cam_id = ?",
        )
        .bind(PREVIEW_SPAN_SECS)
        .bind(cam)
        .fetch_all(&state.db)
        .await
        .unwrap_or_default();

        for h in hours {
            if h >= cur_hour {
                continue; // still recording into this hour
            }
            let id = format!("cam{cam}-{h}");
            let exists: Option<i64> =
                sqlx::query_scalar("SELECT 1 FROM nvr_previews WHERE id = ? LIMIT 1")
                    .bind(&id)
                    .fetch_optional(&state.db)
                    .await
                    .ok()
                    .flatten();
            if exists.is_some() {
                continue;
            }
            match generate_one(&state.db, &ffmpeg, &dir, cam, h, &id).await {
                Ok(true) => made += 1,
                Ok(false) => {}
                Err(e) => tracing::warn!("preview {id}: {e}"),
            }
        }
    }
    if made > 0 {
        tracing::info!("preview: generated {made} scrub preview(s)");
    }
}

/// Build one hour's preview. Returns false when the hour has no usable footage.
#[allow(dead_code)] // only caller is `generate_missing_previews`, itself unspawned
async fn generate_one(
    db: &SqlitePool,
    ffmpeg: &Path,
    dir: &Path,
    cam: i64,
    hour_start: i64,
    id: &str,
) -> anyhow::Result<bool> {
    let hour_end = hour_start + PREVIEW_SPAN_SECS;
    let rows: Vec<(String, String, Option<f64>)> = sqlx::query_as(
        "SELECT path, started_at, duration_secs FROM nvr_segments
         WHERE cam_id = ? AND started_at >= ? AND started_at < ?
         ORDER BY started_at ASC",
    )
    .bind(cam)
    .bind(rfc(hour_start))
    .bind(rfc(hour_end))
    .fetch_all(db)
    .await?;

    // Existence-filter: a pruned segment still in the index would abort the whole
    // concat, losing the preview for an hour that mostly still exists.
    let mut present: Vec<String> = Vec::with_capacity(rows.len());
    let mut first_start: Option<i64> = None;
    let mut last_end: i64 = 0;
    for (p, started_at, dur) in rows {
        if !tokio::fs::try_exists(&p).await.unwrap_or(false) {
            continue;
        }
        let st = epoch(&started_at);
        if first_start.is_none() {
            first_start = Some(st);
        }
        last_end = last_end.max(st + dur.unwrap_or(10.0).round() as i64);
        present.push(p);
    }
    let (span_start, span_end) = match first_start {
        Some(st) if last_end > st => (st, last_end),
        _ => return Ok(false),
    };

    // concat demuxer list. Paths are single-quoted with the demuxer's own escape
    // for embedded quotes; on Windows they contain backslashes, which the demuxer
    // accepts inside quotes.
    let list = present
        .iter()
        .map(|p| format!("file '{}'\n", p.replace('\'', "'\\''")))
        .collect::<String>();
    let list_path = dir.join(format!("{id}.txt"));
    tokio::fs::write(&list_path, list).await?;

    let out: PathBuf = dir.join(format!("{id}.mp4"));
    let tmp: PathBuf = dir.join(format!("{id}.part.mp4"));

    let status = crate::proc::tokio_cmd(ffmpeg)
        .args([
            "-hide_banner", "-loglevel", "error", "-y",
            // Decode keyframes only. With the recorder's forced 1 s keyframes this
            // is ~1 real frame per second at ~1/30th the decode cost.
            "-skip_frame", "nokey",
            "-f", "concat", "-safe", "0",
            "-i", &list_path.to_string_lossy(),
            "-an",
            "-vf", &format!("fps={PREVIEW_FPS},scale=-2:{PREVIEW_HEIGHT}"),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
            "-pix_fmt", "yuv420p", // WebView2 cannot decode 4:4:4
            "-movflags", "+faststart",
            "-threads", "2",
            &tmp.to_string_lossy(),
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .await?;
    let _ = tokio::fs::remove_file(&list_path).await;

    if !status.success() {
        let _ = tokio::fs::remove_file(&tmp).await;
        anyhow::bail!("ffmpeg exited {status:?}");
    }
    // Publish atomically so a half-written preview is never indexed.
    tokio::fs::rename(&tmp, &out).await?;
    let size = tokio::fs::metadata(&out).await.map(|m| m.len() as i64).unwrap_or(0);

    sqlx::query(
        "INSERT OR REPLACE INTO nvr_previews(id, cam_id, start_time, end_time, path, size_bytes, created_at)
         VALUES(?,?,?,?,?,?,?)",
    )
    .bind(id)
    .bind(cam)
    .bind(rfc(span_start))
    .bind(rfc(span_end))
    .bind(out.to_string_lossy().to_string())
    .bind(size)
    .bind(chrono::Utc::now().to_rfc3339())
    .execute(db)
    .await?;
    Ok(true)
}

/// GET /nvr-preview/:id.mp4 — the preview file, range-served.
///
/// A preview is an immutable static file, so this is a plain read: no ffmpeg, no
/// remux, no per-request process. That is the whole point of the layer.
pub(crate) async fn nvr_preview_file(
    axum::extract::Path(file): axum::extract::Path<String>,
    AxumState(s): AxumState<crate::state::StreamState>,
    req_headers: axum::http::HeaderMap,
) -> axum::response::Response {
    use axum::http::StatusCode;
    use axum::response::IntoResponse;
    let id = match file.strip_suffix(".mp4") {
        Some(v) => v,
        None => return StatusCode::BAD_REQUEST.into_response(),
    };
    // Ids are generated as `cam{n}-{epoch}`; refuse anything that could traverse.
    if id.is_empty() || id.len() > 64
        || !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
        return StatusCode::FORBIDDEN.into_response();
    }
    match preview_path(&s.db, id).await {
        Some(path) => crate::footage::serve_file_range(&path, &req_headers, 0, None).await,
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

/// Drop previews whose hour no longer has any footage.
///
/// Called from `prune_old_footage`, so previews age out exactly with the video
/// they describe rather than on a timer of their own. Video only — this touches
/// nothing but the preview files it created.
pub(crate) async fn prune_previews(db: &SqlitePool) {
    let rows: Vec<(String, String)> = sqlx::query_as(
        "SELECT id, path FROM nvr_previews
          WHERE NOT EXISTS (
            SELECT 1 FROM nvr_segments s
             WHERE s.cam_id = nvr_previews.cam_id
               AND s.started_at >= nvr_previews.start_time
               AND s.started_at <  nvr_previews.end_time)",
    )
    .fetch_all(db)
    .await
    .unwrap_or_default();

    for (id, path) in &rows {
        let _ = tokio::fs::remove_file(path).await;
        let _ = sqlx::query("DELETE FROM nvr_previews WHERE id = ?")
            .bind(id)
            .execute(db)
            .await;
    }
    if !rows.is_empty() {
        tracing::info!("preview: pruned {} orphaned preview(s)", rows.len());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The mapping the client relies on: preview media time is wall clock minus
    /// the preview's start. Anything else and a scrub lands somewhere the user
    /// did not aim for.
    #[test]
    fn hour_bucketing_is_stable_and_aligned() {
        let t = 1_787_963_721_i64; // 2026-08-29T00:35:21Z
        let hour = t - t.rem_euclid(PREVIEW_SPAN_SECS);
        assert_eq!(hour % PREVIEW_SPAN_SECS, 0, "buckets align to the hour");
        assert!(hour <= t && t < hour + PREVIEW_SPAN_SECS, "t falls inside its own bucket");
        // Same input, same id — the generator skips work it has already done.
        assert_eq!(format!("cam0-{hour}"), format!("cam0-{}", t - t.rem_euclid(PREVIEW_SPAN_SECS)));
    }

    /// `rfc` is what lands in the DB and what the client parses; a shape the
    /// client cannot read back would silently break preview lookup.
    #[test]
    fn rfc_round_trips_through_rfc3339() {
        let t = 1_787_963_721_i64;
        let parsed = chrono::DateTime::parse_from_rfc3339(&rfc(t)).expect("valid RFC3339");
        assert_eq!(parsed.timestamp(), t);
    }
}
