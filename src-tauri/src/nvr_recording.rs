//! Tauri commands for browser-driven continuous NVR recording (start_nvr, stop_nvr, save_nvr_segment, query helpers).

use std::sync::Arc;

use base64::{engine::general_purpose::STANDARD as B64, Engine};
use chrono::Utc;
use tauri::{Emitter, State};
use uuid::Uuid;

use crate::{AppState, MotionEvent, spawn_nvr_pipe, spawn_hls_pipe};


/// Signal the frontend to start browser-based NVR recording.
/// The browser's MediaRecorder segments the stream and sends each segment via
/// Start NVR recording entirely in Rust — frames piped from process_frame directly
/// to an ffmpeg subprocess that writes H.264 MP4 segments.  Zero browser involvement.
/// Also starts HLS output so the frontend can play back live with low bandwidth.
#[tauri::command]
pub async fn start_nvr(
    state: State<'_, Arc<AppState>>,
    cam_id: u8,
    source_url: String,
) -> Result<serde_json::Value, String> {
    let settings = state.settings.read().await.clone();

    // For RTSP cameras, the relay already pushes frames into process_frame_inner,
    // so the same pipe path works. Nothing extra needed.
    let _ = source_url; // used by RTSP relay setup upstream

    // USB cams: recording is INTEGRATED into the capture ffmpeg (one clock =
    // A/V sync). RTSP cams: the relay's copy-recorder owns recording (and the
    // live HLS output). Spawning a pipe recorder/HLS here would double-record
    // and corrupt the playlist with a second writer.
    let capture_owned = state.capture_keys.lock().await.get(&cam_id)
        .map(|k| k.starts_with("usb:") || k.starts_with("rtsp:")).unwrap_or(false);
    if capture_owned {
        tracing::info!("start_nvr cam{cam_id}: capture-integrated recording (no pipe recorder)");
        return Ok(serde_json::json!({
            "mode": "capture_integrated",
            "segment_mins": settings.nvr_segment_mins,
        }));
    }

    // ── NVR: pipe frames → ffmpeg → segmented H.264 MP4 ───────────────────────
    let encoder = state.hw_encoder.read().unwrap().clone();
    match spawn_nvr_pipe(cam_id, &state.data_dir, settings.nvr_segment_mins, &encoder, state.app_handle.clone(), state.db.clone()).await {
        Ok((tx, child)) => {
            state.nvr_pipe_txs.lock().await.insert(cam_id, tx);
            state.nvr_processes.lock().await.insert(cam_id, child);
            tracing::info!("NVR (Rust/ffmpeg pipe) started for cam{}", cam_id);
        }
        Err(e) => {
            tracing::warn!("NVR ffmpeg pipe failed ({}); install ffmpeg for Rust-side NVR", e);
            // Fallback: browser MediaRecorder (legacy path)
            state.app_handle.emit("nvr:start", serde_json::json!({
                "cam_id": cam_id, "segment_mins": settings.nvr_segment_mins,
            })).ok();
        }
    }

    // ── HLS: pipe frames → ffmpeg → H.264 HLS for low-bandwidth playback ──────
    if !state.hls_pipe_txs.lock().await.contains_key(&cam_id) {
        match spawn_hls_pipe(cam_id, &state.data_dir, &encoder).await {
            Ok((tx, child)) => {
                state.hls_pipe_txs.lock().await.insert(cam_id, tx);
                // Keep the child alive — kill_on_drop(true) would otherwise kill the
                // HLS ffmpeg the moment this scope ends.
                state.hls_processes.lock().await.insert(cam_id, child);
                tracing::info!("HLS stream started for cam{}", cam_id);
            }
            Err(e) => tracing::warn!("HLS ffmpeg pipe failed: {}", e),
        }
    }

    Ok(serde_json::json!({ "mode": "rust_ffmpeg", "segment_mins": settings.nvr_segment_mins }))
}

#[tauri::command]
pub async fn stop_nvr(state: State<'_, Arc<AppState>>, cam_id: u8) -> Result<(), String> {
    // Drop the pipe sender — ffmpeg stdin closes, process exits cleanly and finalises the segment
    state.nvr_pipe_txs.lock().await.remove(&cam_id);
    // Kill any legacy ffmpeg process (RTSP relay path)
    if let Some(mut child) = state.nvr_processes.lock().await.remove(&cam_id) {
        child.kill().await.ok();
    }
    // Also signal legacy browser MediaRecorder (harmless if not running)
    state.app_handle.emit("nvr:stop", serde_json::json!({ "cam_id": cam_id })).ok();
    tracing::info!("NVR stopped for cam{}", cam_id);
    Ok(())
}

/// Receive a completed NVR segment blob from the browser MediaRecorder.
/// Saves it to data_dir/nvr/ and records metadata in nvr_segments.
#[tauri::command]
pub async fn save_nvr_segment(
    state: State<'_, Arc<AppState>>,
    cam_id: u8,
    filename: String,
    blob_b64: String,
    _mime_type: String,
) -> Result<(), String> {
    // Validate filename — only alphanumeric, dash, underscore, dot
    if !filename.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.')) {
        return Err("Invalid filename".into());
    }
    let bytes = B64.decode(blob_b64.trim()).map_err(|e| e.to_string())?;
    let seg_dir = state.data_dir.join("nvr");
    tokio::fs::create_dir_all(&seg_dir).await.map_err(|e| e.to_string())?;
    let path = seg_dir.join(&filename);
    tokio::fs::write(&path, &bytes).await.map_err(|e| e.to_string())?;
    let size = bytes.len() as i64;

    // Parse started_at from filename: cam{N}_{YYYYMMDD}_{HHMMSS}.webm
    // Browser JS creates filenames with LOCAL time — convert to UTC to match motion_events.
    let started_at = filename
        .trim_end_matches(".webm").trim_end_matches(".mp4")
        .splitn(3, '_')
        .collect::<Vec<_>>()
        .get(1..3)
        .and_then(|parts| crate::nvr_pipes::segment_local_name_to_utc(parts[0], parts[1]))
        .unwrap_or_else(|| Utc::now().to_rfc3339());

    let rec_id = Uuid::new_v4().to_string();
    let now    = Utc::now().to_rfc3339();
    sqlx::query(
        "INSERT INTO nvr_segments(id,cam_id,path,started_at,ended_at,size_bytes) VALUES(?,?,?,?,?,?)"
    ).bind(&rec_id).bind(cam_id as i64)
     .bind(path.to_string_lossy().as_ref())
     .bind(&started_at).bind(&now).bind(size)
     .execute(&state.db).await.ok();

    tracing::info!("NVR segment saved: {} ({} KB)", filename, size / 1024);

    // Convert to seekable MP4 using ffmpeg.
    // -c copy won't fix seeking — we must re-encode with +faststart moov atom at front.
    // Runs in background. If ffmpeg unavailable, keeps original WebM (non-seekable fallback).
    let path_clone   = path.clone();
    let app_handle   = state.app_handle.clone();
    let final_name   = filename.trim_end_matches(".webm").to_string() + ".mp4";
    let final_name_c = final_name.clone();
    tokio::spawn(async move {
        let mp4_path = path_clone.with_extension("mp4");

        let status = crate::proc::tokio_cmd("ffmpeg")
            .args([
                "-hide_banner", "-loglevel", "error", "-y",
                "-i", &path_clone.to_string_lossy(),
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                "-threads", "2", // cap x264's ~1.5×cores default frame-thread pool
                "-movflags", "+faststart",   // moov atom first = instant seek
                "-c:a", "aac",
                &mp4_path.to_string_lossy(),
            ])
            .status().await;

        if let Ok(s) = status {
            if s.success() && mp4_path.exists() {
                let _ = tokio::fs::remove_file(&path_clone).await; // delete original .webm
                tracing::info!("NVR segment converted to seekable MP4: {:?}", mp4_path.file_name());
                app_handle.emit("nvr:segment-saved",
                    serde_json::json!({ "filename": final_name_c, "size": size })).ok();
                return;
            }
        }

        // ffmpeg unavailable or failed — keep original .webm (scrubbing will be limited)
        tracing::warn!("ffmpeg not available — NVR segment saved as non-seekable WebM: {}. Install ffmpeg to enable scrubbing.", filename);
        app_handle.emit("nvr:segment-saved",
            serde_json::json!({ "filename": filename, "size": size })).ok();
    });

    Ok(())
}

#[tauri::command]
pub async fn get_nvr_segments(
    state: State<'_, Arc<AppState>>,
    cam_id: Option<u8>,
    limit: Option<i64>,
) -> Result<Vec<serde_json::Value>, String> {
    let rows: Vec<(String, i64, String, String, Option<String>, i64)> =
        if let Some(c) = cam_id {
            sqlx::query_as(
                "SELECT id,cam_id,path,started_at,ended_at,size_bytes FROM nvr_segments WHERE cam_id=? ORDER BY started_at DESC LIMIT ?"
            ).bind(c as i64).bind(limit.unwrap_or(100)).fetch_all(&state.db).await
        } else {
            sqlx::query_as(
                "SELECT id,cam_id,path,started_at,ended_at,size_bytes FROM nvr_segments ORDER BY started_at DESC LIMIT ?"
            ).bind(limit.unwrap_or(100)).fetch_all(&state.db).await
        }.map_err(|e| e.to_string())?;

    Ok(rows.into_iter().map(|(id, cam, path, started, ended, size)| serde_json::json!({
        "id": id, "cam_id": cam, "path": path,
        "started_at": started, "ended_at": ended, "size_bytes": size
    })).collect())
}

/// Does this segment still contain footage at or after `from_ts` (unix secs)?
///
/// The other half of "overlaps [from, to]". `list_nvr_recordings` widens its query
/// by an hour so a segment straddling the window's start is found; this narrows the
/// result back to segments that actually reach into the window. Without it the pad
/// leaks the tail of the PREVIOUS local day into this one — 121 of this archive's
/// 128 segments, at UTC-5 — and the timeline draws bands on a day whose playback
/// chunks begin an hour later, so clicking them does nothing at all.
///
/// A missing `duration_secs` errs long (60 s, matching the payload's own fallback):
/// keeping a doubtful straddler is cheaper than dropping real footage.
fn segment_ends_after(started_at: &str, duration_secs: Option<f64>, from_ts: Option<f64>) -> bool {
    let Some(from_ts) = from_ts else { return true };
    match chrono::DateTime::parse_from_rfc3339(started_at) {
        Ok(start) => start.timestamp() as f64 + duration_secs.unwrap_or(60.0) > from_ts,
        Err(_) => true, // unparseable timestamps are the indexer's problem, not ours
    }
}

/// Scan data_dir/nvr/ for MP4 files, parse timestamps from filenames,
/// and return a list of recordings sorted newest-first.
/// Query NVR segments from the DB — exact timestamps, no filesystem guessing.
/// Falls back to filesystem scan for segments not yet indexed (e.g. legacy files).
#[tauri::command]
pub async fn list_nvr_recordings(
    state: State<'_, Arc<AppState>>,
    cam_id: Option<u8>,
    // Optional UTC range [from, to] (RFC3339). When set, only segments overlapping
    // the window are returned — the frontend passes the selected day so it never
    // transfers tens of thousands of rows. None = all (back-compat).
    from: Option<String>,
    to: Option<String>,
) -> Result<Vec<serde_json::Value>, String> {
    // ── Query the indexed segments. A segment overlaps [from,to] if it starts
    //    before `to` AND ends after `from`. We bound the start with a 1h pad on
    //    `from` so a segment that began just before the window (and runs into it)
    //    is still included. ──────────────────────────────────────────────────
    let mut sql = String::from(
        "SELECT id, cam_id, path, started_at, ended_at, duration_secs, size_bytes
         FROM nvr_segments WHERE 1=1"
    );
    if cam_id.is_some() { sql.push_str(" AND cam_id = ?"); }
    if from.is_some()   { sql.push_str(" AND started_at >= ?"); }
    if to.is_some()     { sql.push_str(" AND started_at <= ?"); }
    sql.push_str(" ORDER BY started_at ASC");

    let mut q = sqlx::query_as::<_, (String, i64, String, String, Option<String>, Option<f64>, i64)>(&sql);
    if let Some(c) = cam_id { q = q.bind(c as i64); }
    // Pad `from` back 1h so a segment that started just before the window but
    // overlaps it is still returned.
    if let Some(f) = &from {
        let padded = chrono::DateTime::parse_from_rfc3339(f)
            .map(|t| (t - chrono::Duration::hours(1)).to_rfc3339())
            .unwrap_or_else(|_| f.clone());
        q = q.bind(padded);
    }
    if let Some(t) = &to { q = q.bind(t.clone()); }
    let rows = q.fetch_all(&state.db).await.map_err(|e| e.to_string())?;

    // Now apply the OTHER half of "overlaps [from,to]": ends after `from`.
    //
    // The 1 h pad above is a query widener, not the rule — and until now nothing
    // narrowed it back. At UTC-5 that handed the caller 121 of this archive's 128
    // segments for the day AFTER the one they belong to: 23:00–23:20 local Sept 3
    // is 04:00–04:20 UTC Sept 4, which the pad sweeps in when asked for Sept 4.
    //
    // That is not merely a cosmetic misfiling. The timeline drew those bands on a
    // day whose playback chunks start an hour later, so every click on them
    // resolved to `findChunk == -1` and `seekTo` returned without doing anything.
    // A dead timeline, no error, on the only day with footage in it.
    //
    // A genuine straddler (starts 23:59:55, runs into the next day) still passes,
    // which is the whole reason the pad exists.
    let from_ts = from.as_ref()
        .and_then(|f| chrono::DateTime::parse_from_rfc3339(f).ok())
        .map(|t| t.timestamp() as f64);

    // DB is the SINGLE SOURCE OF TRUTH. Every real segment is indexed at
    // record time (postprocess_nvr_segments) and on startup (reindex_existing_
    // nvr_segments), so the old per-call filesystem scan was pure redundancy —
    // and with 40k+ segments it re-read the whole nvr/ directory on every list
    // call, which was slow. Removed: trust the DB + index.
    let mut recordings: Vec<serde_json::Value> = rows.into_iter()
        .filter(|(_, _, _, started, _, dur, _)| segment_ends_after(started, *dur, from_ts))
        .map(|(_, cam, path, started, _ended, dur, size)| {
            let fname = std::path::Path::new(&path)
                .file_name().and_then(|n| n.to_str()).unwrap_or(&path).to_string();
            serde_json::json!({
                "filename": fname,
                "cam_id": cam,
                "started_at": started,
                "size_bytes": size,
                "duration_secs": dur.unwrap_or(60.0),
                "path": path,
            })
        })
        .collect();
    // Already ordered by started_at ASC from the query.
    recordings.sort_by(|a, b|
        a["started_at"].as_str().unwrap_or("").cmp(b["started_at"].as_str().unwrap_or(""))
    );
    Ok(recordings)
}

/// Local calendar days that actually hold recorded footage, as `YYYY-MM-DD`.
///
/// Without this the date picker is blind: every day looks identical, so hunting for
/// footage means clicking days at random and reading "camera was off" on the misses.
/// One grouped scan answers "where is there anything to watch" for a whole month.
///
/// Grouped in LOCAL time, deliberately. `started_at` is UTC and every other footage
/// query here compares UTC strings, but the client asks in local calendar days
/// (`dayBoundsUtc`), so a UTC grouping would light up the wrong cell for any camera
/// recording near midnight — the same off-by-a-day this module has been bitten by
/// before. This is a display query and nothing downstream computes from it; retention
/// and playback must keep using UTC.
#[tauri::command]
pub async fn list_recorded_days(
    state: State<'_, Arc<AppState>>,
    cam_id: Option<u8>,
    from: Option<String>,
    to: Option<String>,
) -> Result<Vec<String>, String> {
    let mut sql = String::from(
        "SELECT DISTINCT date(started_at, 'localtime') FROM nvr_segments WHERE 1=1");
    if cam_id.is_some() { sql.push_str(" AND cam_id = ?"); }
    if from.is_some()   { sql.push_str(" AND started_at >= ?"); }
    if to.is_some()     { sql.push_str(" AND started_at <= ?"); }
    sql.push_str(" ORDER BY 1");

    let mut q = sqlx::query_scalar::<_, String>(&sql);
    if let Some(c) = cam_id { q = q.bind(c as i64); }
    if let Some(f) = from   { q = q.bind(f); }
    if let Some(t) = to     { q = q.bind(t); }
    q.fetch_all(&state.db).await.map_err(|e| e.to_string())
}

/// Return all motion events that overlap with the given time window.
/// Used by the NVR viewer to show event markers on the timeline.
#[tauri::command]
pub async fn get_events_in_range(
    state: State<'_, Arc<AppState>>,
    range_start: String,
    range_end: String,
) -> Result<Vec<MotionEvent>, String> {
    // Fixed WHERE clause: started_at BETWEEN uses the idx_me_started index and only
    // loads events from the selected day — not all currently-active events from all time.
    // The original `started_at <= end AND (ended_at >= start OR ended_at IS NULL)`
    // was loading ALL events with ended_at=NULL (active events from any day), causing
    // 78 events × 50KB thumbnails = 4MB per poll with 3-5s query times.
    sqlx::query_as::<_, (String, String, Option<String>, Option<f64>, f32, Option<String>, Option<String>, Option<String>, Option<String>, Option<i64>, Option<String>, Option<String>, Option<String>, Option<String>, Option<String>, Option<f64>)>(
        "SELECT id, started_at, ended_at, duration_secs, peak_score, clip_path, CASE WHEN thumbnail IS NOT NULL AND thumbnail <> '' THEN '@thumb' END AS thumbnail, ai_summary, detections, cam_id, event_category, dominant_label, sub_label, first_object_at, recognized_plate, top_speed_kmh
         FROM motion_events
         WHERE started_at >= ? AND started_at <= ?
         ORDER BY started_at DESC
         LIMIT 800"
    )
    .bind(&range_start)
    .bind(&range_end)
    .fetch_all(&state.db)
    .await
    .map(|rows| rows.into_iter().map(|(id, sa, ea, ds, ps, cp, th, ai, det, cam, cat, dom, sub, foa, plate, spd)| MotionEvent {
        id, started_at: sa, ended_at: ea, duration_secs: ds,
        peak_score: ps, clip_path: cp, thumbnail: th, detections: det, ai_summary: ai,
        cam_id: cam.map(|c| c as u8),
        // v27: classification IS fetched here so the Review tiles show the real
        // object + sub-label in real time (not only after clip analysis).
        event_category: cat,
        recognized_plate: plate,
        dominant_label: dom,
        sub_label: sub,
        first_object_at: foa,
        top_speed_kmh: spd.map(|v| v as f32),
    }).collect())
    .map_err(|e| e.to_string())
}

/// All bookmarked events as full `MotionEvent`s, newest-first — powers the
/// Review "Bookmarks" tab. Same columns + mapping as `get_events_in_range`,
/// joined to `event_bookmarks`.
#[tauri::command]
pub async fn list_bookmarked_events(
    state: State<'_, Arc<AppState>>,
) -> Result<Vec<MotionEvent>, String> {
    sqlx::query_as::<_, (String, String, Option<String>, Option<f64>, f32, Option<String>, Option<String>, Option<String>, Option<String>, Option<i64>, Option<String>, Option<String>, Option<String>, Option<String>, Option<String>, Option<f64>)>(
        "SELECT m.id, m.started_at, m.ended_at, m.duration_secs, m.peak_score, m.clip_path, CASE WHEN m.thumbnail IS NOT NULL AND m.thumbnail <> '' THEN '@thumb' END AS thumbnail, m.ai_summary, m.detections, m.cam_id, m.event_category, m.dominant_label, m.sub_label, m.first_object_at, m.recognized_plate, m.top_speed_kmh
         FROM motion_events m JOIN event_bookmarks b ON b.event_id = m.id
         ORDER BY m.started_at DESC
         LIMIT 2000"
    )
    .fetch_all(&state.db)
    .await
    .map(|rows| rows.into_iter().map(|(id, sa, ea, ds, ps, cp, th, ai, det, cam, cat, dom, sub, foa, plate, spd)| MotionEvent {
        id, started_at: sa, ended_at: ea, duration_secs: ds,
        peak_score: ps, clip_path: cp, thumbnail: th, detections: det, ai_summary: ai,
        cam_id: cam.map(|c| c as u8),
        event_category: cat, recognized_plate: plate, dominant_label: dom, sub_label: sub,
        first_object_at: foa, top_speed_kmh: spd.map(|v| v as f32),
    }).collect())
    .map_err(|e| e.to_string())
}

/// Lightweight event marker for the NVR-player timeline — NO thumbnail/ai_summary,
/// so the WHOLE day loads cheaply (the timeline draws spikes, not images). This is
/// the single COMPLETE source the timeline reads; the heavy `get_events_in_range`
/// (with thumbnails) stays for the Review feed cards.
#[derive(serde::Serialize)]
pub struct EventMarker {
    pub id:              String,
    pub started_at:      String,
    pub ended_at:        Option<String>,
    pub duration_secs:   Option<f64>,
    pub first_object_at: Option<String>,
    pub cam_id:          Option<u8>,
    pub event_category:  Option<String>,
    pub peak_score:      f32,
    /// Whether a thumbnail exists (served via /footage/:id/thumbnail) — one
    /// bit instead of the blob, so marker payloads stay tiny but the player's
    /// Events panel can still render previews.
    pub has_thumb:       bool,
    /// Whether a standalone cached clip file exists (clip_path set). Such an
    /// event stays PLAYABLE even after its raw NVR footage is pruned — the
    /// "No video" badge must not dim it.
    pub has_clip:        bool,
}

/// Every event in the range as lightweight markers — no LIMIT drop (metadata-only
/// is tiny even for thousands/day), so the timeline never silently misses events.
/// Includes the fields the timeline needs to draw spikes AND anchor playback
/// (start/end/first_object_at) — just not the heavy thumbnail/ai_summary/detections.
#[tauri::command]
pub async fn get_event_markers(
    state: State<'_, Arc<AppState>>,
    range_start: String,
    range_end: String,
) -> Result<Vec<EventMarker>, String> {
    sqlx::query_as::<_, (String, String, Option<String>, Option<f64>, Option<String>, Option<i64>, Option<String>, f32, i64, i64)>(
        "SELECT id, started_at, ended_at, duration_secs, first_object_at, cam_id, event_category, peak_score,
                CASE WHEN thumbnail IS NOT NULL AND thumbnail != '' THEN 1 ELSE 0 END,
                CASE WHEN clip_path IS NOT NULL AND clip_path != '' THEN 1 ELSE 0 END
         FROM motion_events
         WHERE started_at >= ? AND started_at <= ?
         ORDER BY started_at ASC
         LIMIT 20000"
    )
    .bind(&range_start)
    .bind(&range_end)
    .fetch_all(&state.db)
    .await
    .map(|rows| rows.into_iter().map(|(id, sa, ea, ds, foa, cam, cat, ps, ht, hc)| EventMarker {
        id, started_at: sa, ended_at: ea, duration_secs: ds, first_object_at: foa,
        cam_id: cam.map(|c| c as u8), event_category: cat, peak_score: ps, has_thumb: ht != 0,
        has_clip: hc != 0,
    }).collect())
    .map_err(|e| e.to_string())
}

/// Whole-archive keyword search over event metadata — powers the Review search
/// box. Multi-keyword: every word must match SOMEWHERE (ai_summary / object
/// label / sub-label / plate / zone / category), so "person driveway" narrows to
/// person events that touched the driveway zone. Returns the same `MotionEvent`
/// shape as `get_events_in_range` so the Review grid renders results identically.
/// LIKE-based (no FTS5/embeddings) — instant at this scale; `idx_me_started`
/// serves the ORDER/LIMIT.
#[tauri::command]
pub async fn search_events(
    state: State<'_, Arc<AppState>>,
    query: String,
    limit: Option<i64>,
) -> Result<Vec<MotionEvent>, String> {
    search_events_core(state.inner(), &query, limit).await
}

/// The body of [`search_events`], callable from anywhere that holds the state.
///
/// Extracted so the AGENT can use it. Its `search_events` tool was a separate,
/// weaker function with the same name — a single case-sensitive `LIKE` over the
/// `ai_summary` JSON, which could not match a plate, a recognised name, a zone or
/// a sound because none of those live in that column, and had no semantic pass at
/// all. Two implementations of "search" is one too many.
pub(crate) async fn search_events_core(
    state: &Arc<AppState>,
    query: &str,
    limit: Option<i64>,
) -> Result<Vec<MotionEvent>, String> {
    let kws = query_keywords(query);
    if kws.is_empty() {
        return Ok(Vec::new());
    }
    let limit = limit.unwrap_or(300).clamp(1, 1000);

    // 1. Keyword results first — high precision (exact substring over the rich
    //    label columns), recency-ordered. This is the whole result set when the
    //    semantic skill isn't installed, so behaviour is unchanged without it.
    let mut events = keyword_search(&state.db, &kws, limit).await.map_err(|e| e.to_string())?;
    let seen: std::collections::HashSet<String> =
        events.iter().map(|e| e.id.clone()).collect();

    // 2. Semantic fill — when a search model is active + installed and keyword
    //    under-fills the limit, embed the query and cosine-match it against the
    //    event image+text embeddings to ADD conceptually-similar events the
    //    keyword pass missed. Keyword hits keep their top position.
    let model = state.settings.read().await.search_model.clone();
    if crate::embed::is_installed(&state.data_dir, &model) && (events.len() as i64) < limit {
        if let Some(qvec) = encode_query(&state.data_dir, &model, query).await {
            let ranked = semantic_rank(&state.db, &model, &qvec).await;
            let need = (limit as usize).saturating_sub(events.len());
            let extra_ids: Vec<String> = ranked.into_iter()
                .filter(|(id, score)| *score >= SEMANTIC_MIN_SIM && !seen.contains(id))
                .take(need)
                .map(|(id, _)| id)
                .collect();
            if !extra_ids.is_empty() {
                if let Ok(extra) = fetch_events_by_ids(&state.db, &extra_ids).await {
                    events.extend(extra);
                }
            }
        }
    }
    Ok(events)
}

/// Split a user query into the bounded `%like%` patterns the keyword pass uses.
fn query_keywords(query: &str) -> Vec<String> {
    query
        .split_whitespace()
        .filter(|w| w.chars().count() >= 2)
        .take(6)
        .map(|w| format!("%{}%", w.to_lowercase()))
        .collect()
}

/// "Find similar" — image→image semantic search (NVR parity). Ranks every
/// other event by cosine of its thumbnail embedding against this event's. Needs
/// the `jina_clip` skill; returns empty without it. Backfills the source event's
/// image embedding on demand so it works for events that closed before install.
#[tauri::command]
pub async fn find_similar_events(
    state: State<'_, Arc<AppState>>,
    event_id: String,
    limit: Option<i64>,
) -> Result<Vec<MotionEvent>, String> {
    let limit = limit.unwrap_or(40).clamp(1, 200);
    let model = state.settings.read().await.search_model.clone();
    if !crate::embed::is_installed(&state.data_dir, &model) {
        return Ok(Vec::new());
    }
    let mut src = load_image_embedding(&state.db, &event_id, &model).await;
    if src.is_none() {
        crate::agent::embed_event(state.inner().clone(), event_id.clone()).await;
        src = load_image_embedding(&state.db, &event_id, &model).await;
    }
    let Some(src) = src else { return Ok(Vec::new()); };

    let rows: Vec<(String, Vec<u8>)> = sqlx::query_as(
        "SELECT event_id, descriptor FROM event_embeddings WHERE kind='image' AND model=?"
    ).bind(&model).fetch_all(&state.db).await.map_err(|e| e.to_string())?;
    let mut ranked: Vec<(String, f32)> = rows.into_iter()
        .filter(|(id, _)| id != &event_id)
        .filter_map(|(id, blob)| {
            let v = crate::embed::blob_to_vec(&blob);
            if v.len() != src.len() { return None; }
            Some((id, crate::embed::cosine(&src, &v)))
        })
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let ids: Vec<String> = ranked.into_iter().take(limit as usize).map(|(id, _)| id).collect();
    fetch_events_by_ids(&state.db, &ids).await.map_err(|e| e.to_string())
}

async fn load_image_embedding(db: &sqlx::SqlitePool, event_id: &str, model: &str) -> Option<Vec<f32>> {
    let blob: Option<Vec<u8>> = sqlx::query_scalar(
        "SELECT descriptor FROM event_embeddings WHERE event_id=? AND kind='image' AND model=?"
    ).bind(event_id).bind(model).fetch_optional(db).await.ok().flatten();
    blob.map(|b| crate::embed::blob_to_vec(&b))
}

/// Backfill: embed every event that lacks an embedding for the ACTIVE search
/// model. Runs in the background (one event at a time, gently throttled) so
/// switching models makes history semantically searchable without blocking the
/// UI. Returns the count queued. No-ops when the active model isn't installed.
#[tauri::command]
pub async fn reindex_semantic_search(state: State<'_, Arc<AppState>>) -> Result<i64, String> {
    let model = state.settings.read().await.search_model.clone();
    if !crate::embed::is_installed(&state.data_dir, &model) {
        return Ok(0);
    }
    let ids: Vec<String> = sqlx::query_scalar(
        "SELECT id FROM motion_events m
         WHERE (m.thumbnail IS NOT NULL OR m.ai_summary IS NOT NULL)
           AND NOT EXISTS (SELECT 1 FROM event_embeddings e WHERE e.event_id = m.id AND e.model = ?)
         ORDER BY m.started_at DESC LIMIT 5000"
    ).bind(&model).fetch_all(&state.db).await.map_err(|e| e.to_string())?;

    let count = ids.len() as i64;
    if count == 0 { return Ok(0); }

    let st = state.inner().clone();
    tokio::spawn(async move {
        for id in ids {
            crate::agent::embed_event(st.clone(), id).await;
            tokio::time::sleep(std::time::Duration::from_millis(20)).await; // gentle throttle
        }
        tracing::info!("reindex_semantic_search: backfill complete");
    });
    Ok(count)
}

/// Cosine floor for a semantic match to be appended. CLIP cross-modal scores
/// run low in absolute terms; this is deliberately permissive because semantic
/// results only ever FILL behind exact keyword hits. May want per-model tuning
/// once a search model is validated against real embeddings.
const SEMANTIC_MIN_SIM: f32 = 0.15;

/// Columns selected for every `MotionEvent` result. Kept in one place so the
/// keyword and id-batch queries stay in lockstep.
// '@thumb' presence marker instead of inline base64 — see get_motion_events.
const EVENT_COLS: &str = "id, started_at, ended_at, duration_secs, peak_score, clip_path, CASE WHEN thumbnail IS NOT NULL AND thumbnail <> '' THEN '@thumb' END AS thumbnail, ai_summary, detections, cam_id, event_category, dominant_label, sub_label, first_object_at, recognized_plate, top_speed_kmh";

type EventRow = (String, String, Option<String>, Option<f64>, f32, Option<String>, Option<String>, Option<String>, Option<String>, Option<i64>, Option<String>, Option<String>, Option<String>, Option<String>, Option<String>, Option<f64>);

fn row_to_event(r: EventRow) -> MotionEvent {
    let (id, sa, ea, ds, ps, cp, th, ai, det, cam, cat, dom, sub, foa, plate, spd) = r;
    MotionEvent {
        id, started_at: sa, ended_at: ea, duration_secs: ds,
        peak_score: ps, clip_path: cp, thumbnail: th, detections: det, ai_summary: ai,
        cam_id: cam.map(|c| c as u8),
        event_category: cat,
        recognized_plate: plate,
        dominant_label: dom,
        sub_label: sub,
        first_object_at: foa,
        top_speed_kmh: spd.map(|v| v as f32),
    }
}

/// Keyword search: one AND group per keyword, OR across every searchable text
/// column within a group. Six binds per keyword, then the LIMIT.
async fn keyword_search(db: &sqlx::SqlitePool, kws: &[String], limit: i64) -> Result<Vec<MotionEvent>, sqlx::Error> {
    let per_kw = "(LOWER(ai_summary) LIKE ? OR LOWER(dominant_label) LIKE ? OR LOWER(sub_label) LIKE ? \
                  OR LOWER(recognized_plate) LIKE ? OR LOWER(zones_entered) LIKE ? OR LOWER(event_category) LIKE ?)";
    let where_sql = vec![per_kw; kws.len()].join(" AND ");
    let sql = format!(
        "SELECT {EVENT_COLS} FROM motion_events WHERE {where_sql} ORDER BY started_at DESC LIMIT ?"
    );
    let mut q = sqlx::query_as::<_, EventRow>(&sql);
    for kw in kws {
        for _ in 0..6 { q = q.bind(kw); }
    }
    q = q.bind(limit);
    Ok(q.fetch_all(db).await?.into_iter().map(row_to_event).collect())
}

/// Fetch specific events by id, preserving the order of `ids` (semantic rank).
async fn fetch_events_by_ids(db: &sqlx::SqlitePool, ids: &[String]) -> Result<Vec<MotionEvent>, sqlx::Error> {
    if ids.is_empty() { return Ok(Vec::new()); }
    let placeholders = vec!["?"; ids.len()].join(",");
    let sql = format!("SELECT {EVENT_COLS} FROM motion_events WHERE id IN ({placeholders})");
    let mut q = sqlx::query_as::<_, EventRow>(&sql);
    for id in ids { q = q.bind(id); }
    let rows = q.fetch_all(db).await?;
    // Re-order to match `ids` (SQL IN returns arbitrary order).
    let mut by_id: std::collections::HashMap<String, MotionEvent> =
        rows.into_iter().map(row_to_event).map(|e| (e.id.clone(), e)).collect();
    Ok(ids.iter().filter_map(|id| by_id.remove(id)).collect())
}

/// Embed the query text with the active model on a blocking thread (ORT is
/// sync/CPU-bound). Returns `None` when the model isn't loadable — caller falls
/// back to keyword-only.
async fn encode_query(data_dir: &std::path::Path, model: &str, query: &str) -> Option<Vec<f32>> {
    let dd = data_dir.to_path_buf();
    let m  = model.to_string();
    let q  = query.to_string();
    tokio::task::spawn_blocking(move || {
        crate::embed::with_model(&dd, &m, |enc| enc.encode_text(&q).ok())
    }).await.ok().flatten().flatten()
}

/// Enforce footage retention — mature NVRs' two-tier retain model, applied at PRUNE
/// time (recording stays continuous, so pre/post-event context is never lost):
///
///   * `nvr_record_mode = "always"`      — everything lives the continuous window;
///     event-overlapping footage lives the (usually longer) event window.
///   * `"motion_only"`  — footage NOT overlapping a motion event is dropped after a
///     1-hour grace; motion footage lives the continuous window; event footage the
///     event window. (mature NVRs `retain.mode: motion` semantics.)
///   * `"events_only"`  — only footage overlapping a review event survives the
///     grace; it lives the event window.
///
/// `retention_days` (the general Storage knob) stays the ABSOLUTE ceiling folded
/// into both windows, so existing setups behave exactly as before until the user
/// picks a mode. **Video only** — never touches face data, events, or embeddings
/// (hard rule). Overlap tests pad events ±15 s for pre/post context.
pub(crate) async fn prune_old_footage(state: &Arc<AppState>) {
    let (mode, cont_days, event_days, final_days) = {
        let s = state.settings.read().await;
        (s.nvr_record_mode.clone(), s.nvr_retain_days, s.nvr_retain_event_days, s.retention_days)
    };
    // Fold the absolute ceiling into each window (0 = forever on either side).
    let eff = |d: u32| -> u32 {
        if final_days == 0 { d } else if d == 0 { final_days } else { d.min(final_days) }
    };
    let cont = eff(cont_days);
    let evw  = eff(event_days);

    // Cutoffs come from SQLite's clock in the same 'T'-separated shape the rows
    // store — datetime('now') emits a SPACE separator that breaks string compares.
    let cutoff = |modifier: &str| -> String {
        format!("(SELECT strftime('%Y-%m-%dT%H:%M:%S','now','{modifier}'))")
    };

    // Segments overlapping (±15 s) a motion event / review event. The lower
    // bound on the event's start makes the correlated probe a NARROW index
    // range ([seg-1d, seg+15s] on idx_me_started) instead of scanning every
    // event per segment — measured on the real DB: 42 s → 3.5 s for a full
    // prune pass. Events longer than a day don't exist (5-min hard cap).
    const MOTION_OVERLAP: &str = "EXISTS (
        SELECT 1 FROM motion_events e
         WHERE e.started_at >= strftime('%Y-%m-%dT%H:%M:%S', s.started_at, '-1 days')
           AND e.started_at <= strftime('%Y-%m-%dT%H:%M:%S', COALESCE(s.ended_at, s.started_at), '+15 seconds')
           AND COALESCE(e.ended_at, e.started_at) >= strftime('%Y-%m-%dT%H:%M:%S', s.started_at, '-15 seconds'))";
    const EVENT_OVERLAP: &str = "EXISTS (
        SELECT 1 FROM review_segments r
         WHERE r.start_time >= strftime('%Y-%m-%dT%H:%M:%S', s.started_at, '-1 days')
           AND r.start_time <= strftime('%Y-%m-%dT%H:%M:%S', COALESCE(s.ended_at, s.started_at), '+15 seconds')
           AND COALESCE(r.end_time, r.start_time) >= strftime('%Y-%m-%dT%H:%M:%S', s.started_at, '-15 seconds'))";

    // Each rule contributes (label, WHERE clause over `nvr_segments s`).
    let mut rules: Vec<(&str, String)> = Vec::new();
    match mode.as_str() {
        "motion_only" => rules.push(("non-motion footage", format!(
            "s.started_at < {} AND NOT {MOTION_OVERLAP}", cutoff("-1 hours")))),
        "events_only" => rules.push(("non-event footage", format!(
            "s.started_at < {} AND NOT {EVENT_OVERLAP}", cutoff("-1 hours")))),
        _ => {}
    }
    if cont > 0 && mode != "events_only" {
        rules.push(("continuous window", format!(
            "s.started_at < {} AND NOT {EVENT_OVERLAP}", cutoff(&format!("-{cont} days")))));
    }
    // Final backstop: nothing outlives the longest enabled window.
    let backstop = match (cont, evw) {
        (0, _) | (_, 0) => 0,
        (c, e) => c.max(e),
    };
    if backstop > 0 {
        rules.push(("retention window", format!(
            "s.started_at < {}", cutoff(&format!("-{backstop} days")))));
    }

    // Keep-event-clips (opt-in): before the raw footage covering an event ages
    // out, export the event's bounded standalone clip — the clip file is a
    // self-contained re-encoded slice, so the moment survives while the bulk
    // continuous recording is deleted on schedule ("keep the curated clips,
    // drop the tape"). Sliding ±2-day window around the death line + per-pass
    // cap bound the work; ensure_event_clip is idempotent (skips events that
    // already have a clip) and returns fast when no footage remains.
    let keep_clips = state.settings.read().await.keep_event_clips;
    if keep_clips && backstop > 0 {
        let lo = format!("-{} days", backstop + 2);
        let hi = format!("-{} days", backstop.saturating_sub(2));
        let ids: Vec<String> = sqlx::query_scalar(
            "SELECT id FROM motion_events
              WHERE (clip_path IS NULL OR clip_path = '')
                AND ended_at IS NOT NULL
                AND started_at >= strftime('%Y-%m-%dT%H:%M:%S','now', ?)
                AND started_at <  strftime('%Y-%m-%dT%H:%M:%S','now', ?)
              ORDER BY started_at ASC LIMIT 50"
        ).bind(&lo).bind(&hi).fetch_all(&state.db).await.unwrap_or_default();
        let mut saved = 0u32;
        for id in &ids {
            if crate::agent::clip_export::ensure_event_clip(state, id).await.is_some() { saved += 1; }
        }
        if saved > 0 {
            tracing::info!("retention: pre-exported {saved} event clip(s) before footage prune (keep_event_clips)");
        }
    }

    let mut total_n = 0usize;
    let mut total_freed = 0u64;
    for (label, where_clause) in rules {
        let old: Vec<(String, String)> = sqlx::query_as(
            &format!("SELECT s.id, s.path FROM nvr_segments s WHERE {where_clause}")
        ).fetch_all(&state.db).await.unwrap_or_default();
        if old.is_empty() { continue; }
        let n = old.len();
        let (ids, paths): (Vec<String>, Vec<String>) = old.into_iter().unzip();
        // Delete the .mp4 files off the async executor (can be thousands).
        let freed = tokio::task::spawn_blocking(move || {
            let mut bytes = 0u64;
            for p in &paths {
                if let Ok(m) = std::fs::metadata(p) { bytes += m.len(); }
                let _ = std::fs::remove_file(p);
            }
            bytes
        }).await.unwrap_or(0);
        // Delete exactly the rows whose files we just removed (no clock drift).
        for chunk in ids.chunks(500) {
            let ph = vec!["?"; chunk.len()].join(",");
            let sql = format!("DELETE FROM nvr_segments WHERE id IN ({ph})");
            let mut q = sqlx::query(&sql);
            for id in chunk { q = q.bind(id); }
            let _ = q.execute(&state.db).await;
        }
        tracing::info!("retention[{mode}]: pruned {n} segment(s) via {label} ({} MB)", freed / 1_048_576);
        total_n += n;
        total_freed += freed;
    }
    // Event HISTORY retention runs every pass, independent of whether any
    // segments were deleted this time (metadata outlives footage, then ages out).
    prune_event_history(state).await;
    // "No footage → no card": events that just lost their footage (and have no
    // kept clip) go with it — Review only ever shows playable things.
    prune_footageless_events(state).await;
    // Scrub previews describe footage; they age out with it, not on a timer.
    crate::nvr_preview::prune_previews(&state.db).await;

    if total_n == 0 { return; }
    tracing::info!("retention: total {total_n} segment(s), {} MB reclaimed", total_freed / 1_048_576);

    // Footage just aged out → consolidate identities in the same window: keep the
    // recurring "regulars", drop one-off strangers whose clip is now gone. Runs here
    // (coupled to actual footage deletion) rather than on a blanket timer.
    consolidate_unknown_faces(state).await;
}

/// Event-HISTORY retention — the metadata half of the storage story. The event
/// tables (motion_events + its timeline/review/search satellites + agent alerts
/// + finished job-queue rows) previously grew FOREVER, so Events/Review/search
/// queries got slower every week and a returning user's tab fan-out hit ever-
/// bigger tables. Policy (user-approved): event history lives 3× the footage
/// retention, minimum 30 days — you can still see WHAT happened well after the
/// clip is gone, then it ages out. `retention_days == 0` (keep footage forever)
/// disables event pruning too.
///
/// NEVER touched (face-data hard rule): known_persons, face_embeddings,
/// face_sightings, body_embeddings, negatives. Bookmarked events are kept.
pub(crate) async fn prune_event_history(state: &Arc<AppState>) {
    let retention_days = state.settings.read().await.retention_days;
    if retention_days == 0 { return; }
    let days = (retention_days * 3).max(30);
    let modifier = format!("-{days} days");
    // Same 'T'-separated shape the rows store (datetime('now') emits a space).
    let cutoff: String = match sqlx::query_scalar("SELECT strftime('%Y-%m-%dT%H:%M:%S','now', ?)")
        .bind(&modifier).fetch_one(&state.db).await
    {
        Ok(c) => c,
        Err(_) => return,
    };

    // Unlink the doomed events' cached clip files BEFORE their rows go — every
    // OTHER delete path (manual delete, clear-all, range delete) removes the
    // file with the row, but this automatic pass didn't: clip_*.mp4 leaked on
    // disk forever once its motion_events row (the only pointer) was deleted.
    // Path-safety: only unlink files inside data_dir.
    {
        let doomed: Vec<String> = sqlx::query_scalar(
            "SELECT clip_path FROM motion_events
              WHERE started_at < ? AND clip_path IS NOT NULL AND clip_path != ''
                AND id NOT IN (SELECT event_id FROM event_bookmarks)"
        ).bind(&cutoff).fetch_all(&state.db).await.unwrap_or_default();
        let n = doomed.len();
        for p in doomed {
            if std::path::Path::new(&p).starts_with(&state.data_dir) {
                let _ = tokio::fs::remove_file(&p).await;
            }
        }
        if n > 0 { tracing::info!("event-history retention: unlinked {n} expired clip file(s)"); }
    }

    // Old, non-bookmarked events — satellites first (no FK cascade in SQLite here).
    const OLD_EVENTS: &str =
        "SELECT id FROM motion_events WHERE started_at < ? AND id NOT IN (SELECT event_id FROM event_bookmarks)";
    let mut pruned: Vec<(&str, u64)> = Vec::new();
    for (label, sql) in [
        ("event_embeddings", format!("DELETE FROM event_embeddings WHERE event_id IN ({OLD_EVENTS})")),
        ("event_timeline",   format!("DELETE FROM event_timeline WHERE event_id IN ({OLD_EVENTS})")),
        ("agent_alerts",     "DELETE FROM agent_alerts WHERE created_at < ?".to_string()),
        ("motion_events",    "DELETE FROM motion_events WHERE started_at < ? AND id NOT IN (SELECT event_id FROM event_bookmarks)".to_string()),
        ("review_segments",  "DELETE FROM review_segments WHERE start_time < ?".to_string()),
        // Finished job-queue rows are write-only history (apalis re-enqueues
        // orphans by status, never re-reads Done) — 8k+ rows observed live.
        ("jobs(done)",       "DELETE FROM Jobs WHERE status = 'Done'".to_string()),
    ] {
        let needs_cutoff = sql.contains('?');
        let q = if needs_cutoff { sqlx::query(&sql).bind(&cutoff) } else { sqlx::query(&sql) };
        match q.execute(&state.db).await {
            Ok(r) if r.rows_affected() > 0 => pruned.push((label, r.rows_affected())),
            _ => {}
        }
    }
    if !pruned.is_empty() {
        let parts: Vec<String> = pruned.iter().map(|(l, n)| format!("{l}={n}")).collect();
        tracing::info!("event-history retention ({days}d): {}", parts.join(" "));
    }
}

// NOTE: `prune_footageless_events` runs from `prune_old_footage` right after
// this, so an event whose footage was pruned THIS pass (and had no kept clip)
// disappears in the same cycle.

/// Delete a set of events AND every satellite row that references them
/// (embeddings, timeline, alerts) — the ONE deletion path, so no caller can
/// leave dangling references. `clear_all_events` / `delete_footage_in_range`
/// used to delete only `motion_events`, leaving `review_segments` to render
/// GHOST cards in Review for events that no longer existed.
/// Face/People tables are NEVER touched (hard rule).
pub(crate) async fn delete_events_and_satellites(db: &sqlx::SqlitePool, ids: &[String]) {
    for chunk in ids.chunks(500) {
        let ph = vec!["?"; chunk.len()].join(",");
        for sql in [
            format!("DELETE FROM event_embeddings WHERE event_id IN ({ph})"),
            format!("DELETE FROM event_timeline WHERE event_id IN ({ph})"),
            format!("DELETE FROM agent_alerts WHERE event_id IN ({ph})"),
            // A bookmark outliving its event kept the Bookmarks badge counting
            // items that could never render (list_bookmarked_events joins the
            // event row) — a saved item the user could not delete or open.
            format!("DELETE FROM event_bookmarks WHERE event_id IN ({ph})"),
            format!("DELETE FROM motion_events WHERE id IN ({ph})"),
        ] {
            let mut q = sqlx::query(&sql);
            for id in chunk { q = q.bind(id); }
            let _ = q.execute(db).await;
        }
    }
    repair_segment_members(db, ids).await;
}

/// Re-aggregate (or drop) review groups that still LIST a deleted event.
///
/// [`prune_empty_review_segments`] only removes a group whose whole WINDOW has
/// no events left; a group that keeps other members went on advertising the
/// dead id in `member_ids` / `clip_event_id`, so its Review card opened a player
/// on an event that no longer exists. Re-upserting from any surviving member
/// re-aggregates the group from scratch (the aggregator silently drops members
/// it can no longer load), which rewrites the stale list.
async fn repair_segment_members(db: &sqlx::SqlitePool, ids: &[String]) {
    // ponytail: one LIKE scan per deleted id — right for user deletes and
    // retention passes. Mass wipes (clear-all / range delete) exceed the cap and
    // rely on prune_empty_review_segments, which drops those groups wholesale.
    if ids.is_empty() || ids.len() > 200 { return; }
    for id in ids {
        let rows: Vec<(String, String)> = sqlx::query_as(
            "SELECT id, data FROM review_segments WHERE data LIKE ?"
        ).bind(format!("%{id}%")).fetch_all(db).await.unwrap_or_default();
        for (seg_id, data) in rows {
            let members: Vec<String> = serde_json::from_str::<serde_json::Value>(&data).ok()
                .and_then(|v| v.get("member_ids").and_then(|m| m.as_array()).cloned())
                .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
                .unwrap_or_default();
            match members.iter().find(|m| !ids.contains(m)) {
                Some(alive) => crate::review_segments::upsert_review_segment(db, alive).await,
                None => {
                    let _ = sqlx::query("DELETE FROM review_segments WHERE id=?")
                        .bind(&seg_id).execute(db).await;
                }
            }
        }
    }
}

/// Drop review groups whose window no longer overlaps ANY event — an emptied
/// group still renders as a card in Review (it's the feed's canonical row), so
/// every event-deletion path must sweep these afterwards.
pub(crate) async fn prune_empty_review_segments(db: &sqlx::SqlitePool) -> u64 {
    sqlx::query(
        "DELETE FROM review_segments WHERE NOT EXISTS (
            SELECT 1 FROM motion_events e
             WHERE e.started_at <= review_segments.end_time
               AND COALESCE(e.ended_at, e.started_at) >= review_segments.start_time)"
    ).execute(db).await.map(|r| r.rows_affected()).unwrap_or(0)
}

/// "No footage → no card" (user policy, 2026-07): a CLOSED, non-bookmarked
/// event whose window has no surviving footage AND no kept clip is deleted
/// outright — Review only ever shows playable things. Protections: bookmarks,
/// open events, a 1-day grace (clip export / indexing may still catch up), and
/// the whole rule only runs with NVR enabled (with NVR off, events are the only
/// record and must persist). Pairs with `keep_event_clips`: toggle ON = moments
/// survive as clips; OFF = events die with their tape.
pub(crate) async fn prune_footageless_events(state: &Arc<AppState>) {
    if !state.settings.read().await.nvr_enabled { return; }
    let ids: Vec<String> = sqlx::query_scalar(
        "SELECT e.id FROM motion_events e
          WHERE e.ended_at IS NOT NULL
            AND (e.clip_path IS NULL OR e.clip_path = '')
            AND e.started_at < strftime('%Y-%m-%dT%H:%M:%S','now','-1 days')
            AND e.id NOT IN (SELECT event_id FROM event_bookmarks)
            AND NOT EXISTS (
                SELECT 1 FROM nvr_segments s
                 WHERE s.cam_id = COALESCE(e.cam_id, 0)
                   AND s.started_at >= strftime('%Y-%m-%dT%H:%M:%S', e.started_at, '-1 days')
                   AND s.started_at <= strftime('%Y-%m-%dT%H:%M:%S', COALESCE(e.ended_at, e.started_at), '+15 seconds')
                   AND COALESCE(s.ended_at, s.started_at) >= strftime('%Y-%m-%dT%H:%M:%S', e.started_at, '-15 seconds'))"
    ).fetch_all(&state.db).await.unwrap_or_default();
    if ids.is_empty() { return; }
    let n = ids.len();
    delete_events_and_satellites(&state.db, &ids).await;
    let rs = prune_empty_review_segments(&state.db).await;
    tracing::info!("retention: deleted {n} footageless event(s) + {rs} emptied review group(s) (no footage → no card)");
}

/// Identity CONSOLIDATION — edge-AI NVRs' "learn who's a REGULAR, forget the noise",
/// coupled to footage deletion (called from `prune_old_footage`). Unknown FACES are
/// clustered (usearch + Chinese Whispers); a recurring stranger (a cluster seen ≥
/// `MIN_RECURRING` times) is KEPT — its vectors + best crops are what let us
/// recognise a regular next month — capped at `RECURRING_CAP` so a frequent regular
/// can't grow unbounded. A ONE-OFF stranger is pruned once its footage has aged out
/// (older than retention → nothing left to review + it never recurred = noise).
/// Anonymous BODY tracks + stranger sighting-log rows past retention are pruned
/// outright (soft signals). **Named/enrolled people are NEVER touched** (person_id
/// IS NULL only). Deletes the offloaded crop files too. The lean, bounded mirror of
/// Edge-AI NVRs' AutoGroup — keep the regulars, drop the noise.
pub(crate) async fn consolidate_unknown_faces(state: &Arc<AppState>) {
    const MIN_RECURRING: usize = 3;   // seen this many times → a "regular" worth keeping
    const RECURRING_CAP: usize = 50;  // cap a regular's crop gallery (bound growth)
    let (days, floor) = {
        let s = state.settings.read().await;
        (s.retention_days.max(1), s.face_recognition_threshold)
    };
    let cutoff: String = match sqlx::query_scalar("SELECT datetime('now', ?)")
        .bind(format!("-{days} days")).fetch_one(&state.db).await
    {
        Ok(c) => c,
        Err(_) => return,
    };

    // Cluster ALL unknown faces (recent + old together) to decide who's a regular.
    let rows: Vec<(String, Vec<u8>, i64, String, f64)> = sqlx::query_as(
        "SELECT id, descriptor, dim, seen_at, quality FROM face_embeddings
          WHERE person_id IS NULL AND dim = 512 ORDER BY seen_at DESC LIMIT 20000"
    ).fetch_all(&state.db).await.unwrap_or_default();
    let mut ids: Vec<String> = Vec::new();
    let mut vecs: Vec<Vec<f32>> = Vec::new();
    let mut seens: Vec<String> = Vec::new();
    let mut quals: Vec<f32> = Vec::new();
    for (id, blob, dim, seen, q) in rows {
        if blob.len() != dim as usize * 4 { continue; }
        vecs.push(blob.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect());
        ids.push(id); seens.push(seen); quals.push(q as f32);
    }

    let mut del: Vec<usize> = Vec::new();
    let (mut regulars, mut oneoff, mut capped) = (0u32, 0u32, 0u32);
    if vecs.len() >= 2 {
        let v = vecs.clone();
        let labels = tokio::task::spawn_blocking(move || {
            let edges = crate::vector_index::knn_graph(&v, 16, floor);
            crate::vector_index::chinese_whispers(v.len(), &edges, 30)
        }).await.unwrap_or_default();
        if labels.len() == vecs.len() {
            let mut groups: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
            for (i, &l) in labels.iter().enumerate() { groups.entry(l).or_default().push(i); }
            for idxs in groups.values() {
                if idxs.len() >= MIN_RECURRING {
                    regulars += 1;
                    if idxs.len() > RECURRING_CAP {
                        let mut ranked = idxs.clone();
                        ranked.sort_by(|&a, &b| quals[b].partial_cmp(&quals[a]).unwrap_or(std::cmp::Ordering::Equal));
                        for &j in &ranked[RECURRING_CAP..] { del.push(j); capped += 1; }
                    }
                } else {
                    for &j in idxs {
                        if seens[j].as_str() < cutoff.as_str() { del.push(j); oneoff += 1; }
                    }
                }
            }
        }
    }
    if !del.is_empty() {
        let del_ids: Vec<String> = del.iter().map(|&j| ids[j].clone()).collect();
        // Delete offloaded crop files off-thread (paths are deterministic from the id).
        let dd = state.data_dir.clone();
        let dids = del_ids.clone();
        tokio::task::spawn_blocking(move || {
            for id in &dids {
                crate::blobstore::delete(&dd, &format!("@file:faces/{id}_t.jpg"));
                crate::blobstore::delete(&dd, &format!("@file:faces/{id}_c.jpg"));
            }
        }).await.ok();
        for chunk in del_ids.chunks(400) {
            let ph = chunk.iter().map(|_| "?").collect::<Vec<_>>().join(",");
            let sql = format!("DELETE FROM face_embeddings WHERE person_id IS NULL AND id IN ({ph})");
            let mut q = sqlx::query(&sql);
            for id in chunk { q = q.bind(id); }
            let _ = q.execute(&state.db).await;
        }
    }

    // Anonymous BODY tracks + stranger sighting-log rows past retention (soft signals).
    let bodies = sqlx::query(
        "DELETE FROM body_embeddings WHERE known_person_id IS NULL AND person_id LIKE 'body\\_%' ESCAPE '\\' AND seen_at < ?"
    ).bind(&cutoff).execute(&state.db).await.map(|r| r.rows_affected()).unwrap_or(0);
    let sights = sqlx::query(
        "DELETE FROM face_sightings WHERE seen_at < ? AND (person_name IS NULL OR person_name='' OR person_name='unknown')"
    ).bind(&cutoff).execute(&state.db).await.map(|r| r.rows_affected()).unwrap_or(0);

    if oneoff + capped > 0 || bodies + sights > 0 {
        tracing::info!(
            "identity consolidation: kept {regulars} recurring stranger(s); pruned {oneoff} one-off + {capped} over-cap face crop(s) + {bodies} anon body + {sights} stranger-sighting row(s) (named identities kept)"
        );
    }
}

/// Cosine-rank every event by its best (image or text) embedding vs the query,
/// restricted to the active model's vectors (different models = different spaces).
///
/// Fast path: the usearch ANN index (`vector_index`) returns the nearest rows in
/// O(log N); we map their rowids back to event ids and keep the best score per
/// event. If the index is empty/unavailable (e.g. skill just installed, nothing
/// indexed yet) it returns nothing and we fall back to the exact O(N) brute-force
/// scan below — identical results, so behaviour is unchanged, just faster at scale.
async fn semantic_rank(db: &sqlx::SqlitePool, model: &str, qvec: &[f32]) -> Vec<(String, f32)> {
    // ── Fast path: ANN index ────────────────────────────────────────────────
    // Pull a generous candidate set (2 rows/event: image+text) so post-dedup we
    // still have plenty for the caller's threshold + limit.
    let hits = crate::vector_index::search(&crate::vector_index::events_index(model), qvec.len(), qvec, 400);
    if !hits.is_empty() {
        let placeholders = hits.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let sql = format!("SELECT rowid, event_id FROM event_embeddings WHERE rowid IN ({placeholders})");
        let mut q = sqlx::query_as::<_, (i64, String)>(&sql);
        for (key, _) in &hits { q = q.bind(*key as i64); }
        let id_of: std::collections::HashMap<i64, String> =
            q.fetch_all(db).await.unwrap_or_default().into_iter().collect();
        let mut best: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
        for (key, sim) in hits {
            if let Some(eid) = id_of.get(&(key as i64)) {
                best.entry(eid.clone()).and_modify(|s| if sim > *s { *s = sim }).or_insert(sim);
            }
        }
        if !best.is_empty() {
            let mut ranked: Vec<(String, f32)> = best.into_iter().collect();
            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            return ranked;
        }
    }

    // ── Fallback: exact brute-force scan (index cold / not built yet) ────────
    let rows: Vec<(String, Vec<u8>)> = sqlx::query_as(
        "SELECT event_id, descriptor FROM event_embeddings WHERE model=?"
    ).bind(model).fetch_all(db).await.unwrap_or_default();

    let mut best: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
    for (event_id, blob) in rows {
        let v = crate::embed::blob_to_vec(&blob);
        if v.len() != qvec.len() { continue; }
        let score = crate::embed::cosine(qvec, &v);
        best.entry(event_id)
            .and_modify(|s| if score > *s { *s = score })
            .or_insert(score);
    }
    let mut ranked: Vec<(String, f32)> = best.into_iter().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked
}


#[cfg(test)]
mod tests {
    use super::*;

    /// THE bug this guard exists for, with the archive's real numbers.
    ///
    /// 23:00-23:20 local Sept 3 is 04:00-04:20 UTC Sept 4. Asked for local Sept 4
    /// (which begins 05:00 UTC), the 1 h query pad sweeps all of it in. The
    /// timeline then drew twenty minutes of coverage on a day that has none, and
    /// because that instant precedes every playback chunk of that day, clicking it
    /// resolved to no chunk and the seek silently did nothing.
    #[test]
    fn the_previous_nights_footage_does_not_leak_into_this_day() {
        let day_start = chrono::DateTime::parse_from_rfc3339("2026-09-04T05:00:00+00:00")
            .unwrap().timestamp() as f64;
        // A segment from 23:20 local Sept 3 — squarely in the padded hour.
        assert!(!segment_ends_after("2026-09-04T04:20:06+00:00", Some(10.0), Some(day_start)),
            "footage that ends 40 min before the window starts is not in it");
        // ...and the first one the pad reaches.
        assert!(!segment_ends_after("2026-09-04T04:00:06+00:00", Some(10.0), Some(day_start)));
    }

    /// Why the pad exists at all: a segment that begins before the window and runs
    /// into it is genuinely part of it, and must survive.
    #[test]
    fn a_real_straddler_still_survives() {
        let day_start = chrono::DateTime::parse_from_rfc3339("2026-09-04T05:00:00+00:00")
            .unwrap().timestamp() as f64;
        assert!(segment_ends_after("2026-09-04T04:59:55+00:00", Some(10.0), Some(day_start)),
            "starts 5 s before the window, ends 5 s inside it");
        assert!(segment_ends_after("2026-09-04T06:00:00+00:00", Some(10.0), Some(day_start)),
            "plainly inside the window");
    }

    /// Doubt errs toward keeping footage, never toward dropping it.
    #[test]
    fn unknown_duration_and_unbounded_queries_keep_the_segment() {
        let day_start = 1_788_498_000.0;
        assert!(segment_ends_after("2026-09-04T04:59:30+00:00", None, Some(day_start)),
            "no duration_secs falls back to 60 s, which reaches the window");
        assert!(segment_ends_after("2026-09-04T00:00:00+00:00", Some(10.0), None),
            "no `from` bound means no filtering");
        assert!(segment_ends_after("not-a-timestamp", Some(10.0), Some(day_start)),
            "an unparseable row is the indexer's problem; do not silently drop it");
    }
}
