//! Agent utility / background-loop helpers.
//!
//! * `get_status`                 — assembles the `AgentStatus` payload.
//! * `analyze_snapshot`           — one-shot snapshot analysis (Test button).
//! * `run_disk_guard_loop`        — periodic NVR + clip cleanup so storage stays
//!                                   under the user-configured `nvr_max_gb`.
//! * `cross_camera_context`       — assembles per-camera narrative summaries.
//! * `run_heartbeat_loop`         — emits a `guardian:heartbeat` ping + analytical
//!                                   summary for the live UI.

use std::sync::Arc;
use std::time::Duration;

use base64::Engine as _;
use chrono::Utc;
use sqlx::SqlitePool;

use tauri::Emitter;

use crate::AppState;
use super::types::*;
use super::memory::{
    read_memory, write_memory, read_core_memory,
};
use super::llm::call_llm;

// ─── Status helper ────────────────────────────────────────────────────────────

pub async fn get_status(state: &Arc<AppState>) -> AgentStatus {
    let settings = state.settings.read().await.clone();
    let last_run = state.agent_last_run.read().await.clone();

    // "Reachable" must follow the ACTIVE provider. The on-device model has no
    // endpoint to ping — it is reachable exactly when its file is on disk. Cloud
    // providers are validated at call time, so they report reachable here.
    let provider_ready = if !settings.agent_enabled {
        false
    } else if matches!(settings.ai_provider.as_str(), "local" | "") {
        super::local_llm::model_ready()
    } else {
        true
    };

    let pending: i64 = sqlx::query_as::<_, (i64,)>(
        "SELECT COUNT(*) FROM motion_events WHERE ai_summary IS NULL AND ended_at IS NOT NULL AND started_at > datetime('now','-24 hours')"
    ).fetch_one(&state.db).await.map(|(n,)| n).unwrap_or(0);

    let total: i64 = sqlx::query_as::<_, (i64,)>(
        "SELECT COUNT(*) FROM agent_alerts"
    ).fetch_one(&state.db).await.map(|(n,)| n).unwrap_or(0);

    AgentStatus {
        enabled: settings.agent_enabled,
        provider_ready,
        last_run_at: last_run,
        model: settings.vision_model.clone(),
        vision_model: settings.vision_model.clone(),
        pending_events: pending,
        total_analyzed: total,
    }
}

// ─── Snapshot analysis (Test button) ─────────────────────────────────────────

pub async fn analyze_snapshot(
    state: &Arc<AppState>,
    image_data_url: String,
    live_detections: Vec<String>,
) -> anyhow::Result<String> {
    let settings = state.settings.read().await.clone();
    let camera_name = if settings.camera_name.is_empty() { "Security Camera" } else { &settings.camera_name };

    // Always send the image — a non-vision model just ignores it.
    // Model selection (vision_model) is handled inside call_llm via settings.
    // On-device needs no model NAME (the engine is compiled in), so only remote
    // providers are required to have one.
    if settings.vision_model.is_empty()
        && !matches!(settings.ai_provider.as_str(), "local" | "")
    {
        anyhow::bail!("No model configured — go to Guardian > Arsenal and select a model");
    }

    // Strip data-URL prefix; send raw base64 in the images array
    let b64 = image_data_url
        .trim_start_matches("data:image/jpeg;base64,")
        .trim_start_matches("data:image/png;base64,")
        .to_string();
    // Only attach image if we actually have frame data
    let images = if b64.len() > 100 { Some(vec![b64]) } else { None };

    // This request is "describe THIS image" — there is no useful text-only answer,
    // only an invented one. Say so instead of returning a confident fabrication.
    if images.is_some() && !super::llm::provider_supports_vision(&settings) {
        anyhow::bail!(
            "The on-device model is text-only — it can't look at snapshots. \
             Pick a cloud or self-hosted vision model in Guardian > Arsenal to analyse images."
        );
    }

    let camera_profile = read_memory(&state.db, "camera_profile").await
        .unwrap_or_else(|| "General-purpose security camera.".to_string());
    let threat_rules   = read_memory(&state.db, "threat_rules").await
        .unwrap_or_else(|| "Alert on unknown persons especially at night.".to_string());
    let known_fp       = read_memory(&state.db, "known_false_positives").await
        .unwrap_or_else(|| "No known false positives.".to_string());

    let recent: Vec<(String, f32, Option<String>)> = sqlx::query_as(
        "SELECT started_at, peak_score, ai_summary FROM motion_events ORDER BY started_at DESC LIMIT 5"
    ).fetch_all(&state.db).await.unwrap_or_default();

    let events_ctx = if recent.is_empty() {
        "No recent motion events on record.".to_string()
    } else {
        recent.iter().map(|(t, s, _)| format!("• {} — score {:.0}%", t, s * 100.0))
            .collect::<Vec<_>>().join("\n")
    };

    let system = format!(
        r#"You are Guardian, an AI security analyst for "{camera_name}".
You will be shown a live camera image. Describe exactly what you see with precision:
- How many people, where are they, what are they doing?
- Vehicles, animals, packages or other objects?
- Lighting conditions and environment?
- Is anything suspicious or threatening based on the threat rules?
Be specific and direct. Plain text only, 3-6 sentences.

## Camera Profile
{camera_profile}

## Threat Rules
{threat_rules}

## Known False Positives
{known_fp}

## Recent Motion History
{events_ctx}"#
    );

    // Load known persons for context
    let known_persons: Vec<(String, String, Option<String>)> = sqlx::query_as(
        "SELECT name, role, last_seen_at FROM known_persons ORDER BY name ASC"
    ).fetch_all(&state.db).await.unwrap_or_default();

    let persons_ctx = if known_persons.is_empty() {
        "No enrolled persons.".to_string()
    } else {
        known_persons.iter().map(|(name, role, last_seen)| {
            let seen = last_seen.as_deref().unwrap_or("never");
            format!("• {} ({}), last seen: {}", name, role, seen)
        }).collect::<Vec<_>>().join("\n")
    };

    let scene_ctx = if live_detections.is_empty() {
        "No AI detection data available.".to_string()
    } else {
        live_detections.join("\n")
    };

    let user_msg = format!(
        "Analyse this camera snapshot.\n\n\
         ## Edge AI Scene Analysis\n{scene_ctx}\n\n\
         ## Enrolled/Assigned Persons\n{persons_ctx}\n\n\
         Based on the image AND the scene analysis above:\n\
         - Name any assigned persons you see\n\
         - Count and describe any unidentified people (Person #1, #2, etc.)\n\
         - Note spatial relationships (who is near whom)\n\
         - Flag anything suspicious per the threat rules\n\
         Be specific. Plain text, 3-6 sentences."
    );

    call_llm(&settings, &system, &user_msg, images, false).await
}

// ─── Disk space guard ─────────────────────────────────────────────────────────
// (push_proactive_insights / push_situation_awareness stubs removed with their
// wake-forever loops — both features were disabled at the user's request in
// v28 as chat/Telegram noise. Real threat alerts come from clip analysis.)

/// Runs every 5 minutes. Deletes the oldest NVR segments when storage exceeds
/// the user-defined limit (nvr_max_gb). Ensures the disk never fills silently.
/// Camera-health watchdog. A security camera that dies SILENTLY is the worst
/// failure mode this product can have — worse than any false positive — so a
/// camera that stops producing frames for `OFFLINE_AFTER_SECS` raises ONE
/// critical alert (Telegram via the channel_alert_allowed chokepoint + a UI
/// toast), and one recovery note when frames resume. Liveness signal =
/// `scene_last_update`, stamped per processed frame in the inference loop, so
/// it covers every capture kind (native/RTSP/MJPEG) with zero added cost.
pub async fn run_camera_health_loop(state: Arc<AppState>) {
    const OFFLINE_AFTER_SECS: u64 = 90;
    // Boot grace — cameras auto-start and warm up; don't cry wolf during startup.
    tokio::time::sleep(Duration::from_secs(180)).await;
    let mut offline: std::collections::HashSet<u8> = std::collections::HashSet::new();
    loop {
        tokio::time::sleep(Duration::from_secs(30)).await;
        let enabled: Vec<i64> = sqlx::query_scalar(
            "SELECT cam_id FROM camera_configs WHERE enabled=1"
        ).fetch_all(&state.db).await.unwrap_or_default();
        if enabled.is_empty() { continue; }
        let now_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();
        let last = state.scene_last_update.read().await.clone();
        for cam in enabled {
            let cam = cam.clamp(0, 255) as u8;
            let fresh = last.get(&cam)
                .is_some_and(|t| now_secs.saturating_sub(*t) < OFFLINE_AFTER_SECS);
            if !fresh && !offline.contains(&cam) {
                offline.insert(cam);
                let summary = match last.get(&cam) {
                    Some(t) => format!("No frames for {} min — stream or capture is down",
                                       (now_secs.saturating_sub(*t) / 60).max(1)),
                    None => "No frames since startup — camera never came up".to_string(),
                };
                tracing::warn!("CAMERA OFFLINE cam{cam}: {summary}");
                state.app_handle.emit("camera:health",
                    serde_json::json!({ "cam_id": cam, "online": false })).ok();
                crate::agent::dispatch_intelligence_alert(&state, "camera_offline", &summary, cam, None).await;
            } else if fresh && offline.remove(&cam) {
                tracing::info!("CAMERA ONLINE cam{cam}: frames resumed");
                state.app_handle.emit("camera:health",
                    serde_json::json!({ "cam_id": cam, "online": true })).ok();
                crate::agent::dispatch_intelligence_alert(&state, "camera_online", "Camera is back online", cam, None).await;
            }
        }
    }
}

pub async fn run_disk_guard_loop(state: Arc<AppState>) {
    loop {
        tokio::time::sleep(Duration::from_secs(300)).await; // check every 5 min

        let settings = state.settings.read().await.clone();
        let max_bytes = settings.nvr_max_gb as u64 * 1024 * 1024 * 1024;
        if max_bytes == 0 { continue; } // unlimited

        let nvr_dir = state.data_dir.join("nvr");

        // The directory walk + per-file metadata + deletions are blocking
        // syscalls over potentially 100k+ files (16 cams × days of 1-min
        // segments) — running them on the async executor hitched the runtime
        // every 5 minutes. Do ALL filesystem work on a blocking thread and
        // come back with just the deleted filenames for the DB cleanup.
        let deleted_files: Vec<String> = tokio::task::spawn_blocking(move || {
            let Ok(entries) = std::fs::read_dir(&nvr_dir) else { return Vec::new() };
            let mut files: Vec<(std::time::SystemTime, std::path::PathBuf, u64)> = entries
                .flatten()
                .filter_map(|e| {
                    let path = e.path();
                    let meta = path.metadata().ok()?;
                    if !meta.is_file() { return None; }
                    Some((meta.modified().unwrap_or(std::time::UNIX_EPOCH), path, meta.len()))
                })
                .collect();
            files.sort_by_key(|(t, _, _)| *t); // oldest first

            let total: u64 = files.iter().map(|(_, _, s)| s).sum();
            if total <= max_bytes { return Vec::new(); }

            let mut to_free = total - max_bytes;
            let mut deleted = Vec::new();
            for (_, path, size) in &files {
                if to_free == 0 { break; }
                if std::fs::remove_file(path).is_ok() {
                    if let Some(f) = path.file_name().and_then(|n| n.to_str()) {
                        deleted.push(f.to_string());
                    }
                    to_free = to_free.saturating_sub(*size);
                }
            }
            deleted
        }).await.unwrap_or_default();

        if !deleted_files.is_empty() {
            for fname in &deleted_files {
                sqlx::query("DELETE FROM nvr_segments WHERE path LIKE ?")
                    .bind(format!("%{fname}"))
                    .execute(&state.db).await.ok();
            }
            tracing::info!("Disk guard: deleted {} NVR segments to stay under {}GB limit",
                deleted_files.len(), settings.nvr_max_gb);
        }
    }
}

// ─── Cross-camera context builder ─────────────────────────────────────────────

/// Build a short narrative of what happened across all cameras in the last N minutes.
/// Used to give the Guardian agent cross-camera awareness for causal reasoning.
pub(super) async fn build_cross_camera_context(db: &SqlitePool) -> String {
    // Query recent motion events with their camera info
    let rows: Vec<(String, i64, Option<String>)> = sqlx::query_as(
        "SELECT started_at, rowid, ai_summary
         FROM motion_events
         WHERE started_at > datetime('now', '-10 minutes')
         ORDER BY started_at ASC LIMIT 20"
    ).fetch_all(db).await.unwrap_or_default();

    if rows.is_empty() { return "No activity in the last 10 minutes.".into(); }

    // Build timeline narrative
    let mut lines: Vec<String> = Vec::new();
    for (started, rowid, summary) in &rows {
        let time = started.get(11..16).unwrap_or("?");
        let cam  = (rowid % 16) as u8; // rough cam hint from rowid
        let desc = summary.as_deref().unwrap_or("motion detected");
        lines.push(format!("[{time}] cam{cam}: {desc}"));
    }

    // Also include face sightings across cameras
    let sightings: Vec<(String, i64, String)> = sqlx::query_as(
        "SELECT seen_at, camera_id, person_name
         FROM face_sightings
         WHERE seen_at > datetime('now', '-10 minutes')
         ORDER BY seen_at ASC LIMIT 20"
    ).fetch_all(db).await.unwrap_or_default();

    for (seen, cam, name) in &sightings {
        let time = seen.get(11..16).unwrap_or("?");
        lines.push(format!("[{time}] cam{cam}: {name} sighted"));
    }

    if lines.is_empty() {
        "No cross-camera activity to report.".into()
    } else {
        lines.join("\n")
    }
}

// ─── Heartbeat loop ────────────────────────────────────────────────────────────

/// Scores how "interesting" the current scene is (0–10+).
/// Returns (score, description) so we can include the reason in the heartbeat thought.
pub(super) async fn scene_interest_score(state: &Arc<AppState>) -> (f32, String) {
    let now_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();
    let scene    = state.scene_objects.read().await.clone();
    let last_upd = state.scene_last_update.read().await.clone();

    let active_cams: Vec<u8> = last_upd.iter()
        .filter(|(_, t)| now_secs.saturating_sub(**t) < 30)
        .map(|(c, _)| *c).collect();
    if active_cams.is_empty() { return (0.0, "camera off".into()); }

    let all_objects: Vec<String> = active_cams.iter()
        .flat_map(|c| scene.get(c).map(|v| v.as_slice()).unwrap_or(&[]).iter()
            .filter(|o| o.score >= 0.4)
            .map(|o| o.label.clone()))
        .collect();

    let person_count = all_objects.iter().filter(|l| l.as_str() == "person").count();
    let object_count = all_objects.len();

    let mut score = 0f32;
    let mut reasons: Vec<String> = Vec::new();

    // Only people matter — objects are ignored for interestingness scoring
    if person_count >= 3 { score += 5.0; reasons.push(format!("{person_count} people in view")); }
    else if person_count == 2 { score += 3.5; reasons.push("2 people in view".into()); }
    else if person_count == 1 { score += 2.0; reasons.push("person detected".into()); }

    // Cell phone carried by a person is interesting (behaviour signal)
    let has_phone = all_objects.iter().any(|l| l.as_str() == "cell phone");
    if has_phone && person_count > 0 { score += 1.0; reasons.push("person with phone".into()); }

    let _ = object_count; // not used for scoring

    let desc = if reasons.is_empty() { "quiet scene".into() } else { reasons.join(", ") };
    (score, desc)
}

/// Autonomous heartbeat: observes the live scene every 15 s, fires a short Ollama
/// observation when the scene is interesting (score ≥ 4), and writes key findings
/// to agent memory. Emits `guardian:heartbeat` for the frontend to display live.
pub async fn run_heartbeat_loop(state: Arc<AppState>) {
    // Stagger startup so we don't hammer Ollama at launch
    tokio::time::sleep(Duration::from_secs(30)).await;

    let mut last_fired_at: u64 = 0;
    let min_interval_secs: u64 = 45; // don't fire more often than this

    loop {
        tokio::time::sleep(Duration::from_secs(15)).await;

        let settings = state.settings.read().await.clone();
        // Live scene analysis is a pure-VLM feature, so gate it on the configured
        // provider (not just an Ollama URL) — otherwise cloud users' live analysis
        // never fired despite a working agent.
        if !settings.agent_enabled || !super::llm::agent_configured(&settings) {
            continue;
        }

        let now_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();
        if now_secs.saturating_sub(last_fired_at) < min_interval_secs { continue; }

        let (score, scene_desc) = scene_interest_score(&state).await;
        if score < 4.0 { continue; }

        last_fired_at = now_secs;

        // Emit "thinking" status immediately so the UI shows activity
        state.app_handle.emit("guardian:heartbeat",
            serde_json::json!({
                "stage": "thinking",
                "scene": scene_desc,
                "score": score,
            })).ok();

        // Build a compact observation prompt — no JSON format, just natural language + REMEMBER tags
        let now_secs2 = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();
        let scene    = state.scene_objects.read().await.clone();
        let last_upd = state.scene_last_update.read().await.clone();
        // Only report people and personal devices — ignore all other objects
        const PERSON_LABELS: &[&str] = &["person", "cell phone"];
        let live_objects: Vec<String> = scene.iter()
            .filter(|(c, _)| last_upd.get(c).map(|t| now_secs2.saturating_sub(*t) < 30).unwrap_or(false))
            .flat_map(|(_, objs)| objs.iter()
                .filter(|o| o.score >= 0.35 && PERSON_LABELS.contains(&o.label.as_str()))
                .map(|o| format!("{} ({:.0}%)", o.label, o.score * 100.0)))
            .collect();

        // Load all structured memory for full context
        let all_memory  = read_core_memory(&state.db).await;
        let time_str    = Utc::now().format("%Y-%m-%d %H:%M UTC").to_string();

        // Build cross-camera sighting context — last 10 minutes across all cameras
        let cross_cam_context = build_cross_camera_context(&state.db).await;

        let cam_name_hb = if settings.camera_name.is_empty() { "Security Camera" } else { &settings.camera_name };
        let identity_hb = agent_identity(&settings, cam_name_hb);
        let system = format!(
r#"{identity_hb}
You have persistent memory and cross-camera awareness.

Current time: {time_str}

## Your Memory
{all_memory}

## Recent Cross-Camera Activity (last 10 min)
{cross_cam_context}

Your job right now:
1. Focus on people — behaviour, movements, and anything suspicious.
2. Cross-reference: if you see someone on this camera who was recently on another camera, note the timeline ("Same person seen on cam2 3 min ago").
3. Write 1-3 SHORT, factual observations. Flag causal patterns: "Person left cam1 → appeared cam3 (consistent with walking route)".
4. Embed memory tags when you learn something new:
   - [REMEMBER:person_NAME:observation with timestamp]
   - [REMEMBER:pattern_DESC:observation]
   - [REMEMBER:event_DESC:what happened and where]
5. Under 80 words. Plain sentences. REMEMBER tags stripped before display."#
        );

        let user_msg = format!(
            "Live scene (interest score {score:.1}/10): {}\nPeople detected: {}",
            scene_desc,
            if live_objects.is_empty() { "no people currently detected".to_string() } else { live_objects.join(", ") }
        );

        // VLM: attach latest frame image when the provider can actually look at one.
        // Text-only providers still get the heartbeat — `user_msg` above is built from
        // real YOLO detections, so there is nothing to invent — they just get no frame.
        let use_vision = !settings.vision_model.is_empty()
            && super::llm::provider_supports_vision(&settings);
        let (model_name, vision_images) = if use_vision {
            let frame_b64 = state.latest_frames.read().await
                .get(&0).map(|f| base64::engine::general_purpose::STANDARD.encode(f));
            (settings.vision_model.clone(), frame_b64.map(|f| vec![f]))
        } else {
            (settings.vision_model.clone(), None)
        };

        // Route the live heartbeat (with its optional vision frame) through the unified
        // provider dispatcher — was hardcoded to Ollama, so cloud providers got no
        // heartbeat analysis. `model_name` is informational; call_llm uses the selected
        // provider + settings.vision_model.
        let _ = model_name;
        let http_res = super::llm::call_llm(&settings, &system, &user_msg, vision_images, false).await;
        match http_res {
            Err(_) => continue,
            Ok(raw) => {
                if raw.trim().is_empty() { continue; }

                // Parse REMEMBER tags — route to structured narrative memory files
                let mut thought = raw.clone();
                let mut memories_written: Vec<(String, String)> = Vec::new();
                while let Some(s) = thought.find("[REMEMBER:") {
                    let Some(e) = thought[s..].find(']') else { break };
                    let inner = thought[s + 10..s + e].to_string();
                    let parts: Vec<&str> = inner.splitn(2, ':').collect();
                    if parts.len() == 2 {
                        let key   = parts[0].trim().to_lowercase().replace(' ', "_");
                        let entry = parts[1].trim().to_string();
                        if !key.is_empty() && !entry.is_empty() {
                            write_memory(&state.db, &key, &entry).await;
                            memories_written.push((key.clone(), entry.clone()));
                        }
                    }
                    thought = format!("{}{}", &thought[..s], &thought[s + e + 1..]);
                }
                let thought = thought.trim().to_string();

                // Emit completed heartbeat for the UI
                state.app_handle.emit("guardian:heartbeat",
                    serde_json::json!({
                        "stage": "done",
                        "thought": thought,
                        "scene": scene_desc,
                        "score": score,
                        "memories": memories_written,
                        "timestamp": Utc::now().to_rfc3339(),
                    })).ok();
            }
        }
    }
}

