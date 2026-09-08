//! Background-loop subsystems for the agent.
//!
//! * `run_cycle` / `run_now` — one pass through all unanalysed events.
//! * `run_agent_loop`        — the top-level agent worker (drives `run_cycle`).
//! * `run_escalation_loop`   — re-sends Telegram alerts that nobody acknowledged.
//! * `run_reflection_loop`   — periodic "what happened this hour" summary.

use std::sync::Arc;
use std::time::Duration;

use chrono::{Local, Utc};


use crate::{AppState, Settings};
use super::types::*;
use super::memory::{read_memory, write_memory, append_memory, decay_learned_memories};
use super::dispatch::send_telegram;
use super::dispatch::send_daily_digest;
use super::analyze_event_clip;
use super::run_live_alert_loop;
use super::run_backfill_analysis;
use super::util::{run_heartbeat_loop, run_disk_guard_loop, run_camera_health_loop};

// ─── Agent cycle ──────────────────────────────────────────────────────────────

pub(super) async fn run_cycle(state: &Arc<AppState>, _settings: &Settings) {
    // Analyse ALL completed events that haven't been described yet.
    // clip_path IS NOT NULL was removed — we analyse from thumbnail + metadata even without a clip.
    // Events are analysed as soon as ended_at is set (motion stopped for 15 s).
    let rows: Vec<(String, String, Option<String>, f32, Option<String>, Option<String>)> =
        sqlx::query_as(
            "SELECT id, started_at, ended_at, peak_score, thumbnail, detections
             FROM motion_events
             WHERE ai_summary IS NULL
               AND ended_at IS NOT NULL
               AND started_at > datetime('now', '-7 days')
             ORDER BY started_at DESC
             LIMIT 5",
        )
        .fetch_all(&state.db)
        .await
        .unwrap_or_default();

    if rows.is_empty() { return; }

    for (id, ..) in &rows {
        // Delegate to the full analysis pipeline (risk assessment + Telegram + DB)
        analyze_event_clip(Arc::clone(state), id.clone()).await;
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
}

// ─── Public trigger (for immediate on-demand cycle) ──────────────────────────

pub async fn run_now(state: &Arc<AppState>) {
    let settings = state.settings.read().await.clone();
    // Run whenever the agent is enabled — NOT gated on a specific provider's URL.
    // Event processing (ALPR + face recognition + the detection-summary floor) is
    // valuable even with no LLM; `analyze_event_clip` gates the VLM itself. (Was
    // `&& !<provider url>.is_empty()`, which froze the cycle for cloud/no-LLM setups.)
    if settings.agent_enabled {
        run_cycle(state, &settings).await;
        *state.agent_last_run.write().await = Some(Utc::now().to_rfc3339());
    }
}

// ─── Escalation timer loop ────────────────────────────────────────────────────

pub async fn run_escalation_loop(state: Arc<AppState>) {
    loop {
        tokio::time::sleep(Duration::from_secs(30)).await;
        let now = std::time::Instant::now();
        let (token, chat_id) = {
            let s = state.settings.read().await;
            (s.telegram_bot_token.clone(), s.telegram_chat_id.clone())
        };
        if token.is_empty() || chat_id.is_empty() { continue; }

        let mut to_escalate: Vec<EscalationState> = Vec::new();
        {
            let mut pending = state.pending_escalations.write().await;
            pending.retain(|_, esc| {
                if !esc.acknowledged && now.duration_since(esc.sent_at).as_secs() >= esc.timeout_secs {
                    to_escalate.push(esc.clone());
                    false // remove from pending
                } else {
                    true
                }
            });
        }
        for esc in to_escalate {
            let msg = format!(
                "⏰ *No response received* — auto-escalating {} alert\n\n{}\n\nPlease review immediately. If this is a genuine threat, contact emergency services.",
                esc.risk_level.to_uppercase(), esc.summary
            );
            send_telegram(&token, &chat_id, &msg).await;
            sqlx::query("UPDATE agent_alerts SET escalated=1 WHERE id=?")
                .bind(&esc.alert_id)
                .execute(&state.db)
                .await.ok();
        }
    }
}

// ─── Reflection loop ──────────────────────────────────────────────────────────

pub async fn run_reflection_loop(state: Arc<AppState>) {
    // Run once at startup after a long delay, then every 6 hours
    tokio::time::sleep(Duration::from_secs(3600)).await;
    loop {
        let settings = state.settings.read().await.clone();
        // Capability, not a model NAME: on-device has no tag, so the old
        // `vision_model.is_empty()` test slept this loop forever.
        if !settings.agent_enabled || !super::llm::agent_configured(&settings) {
            tokio::time::sleep(Duration::from_secs(3600)).await;
            continue;
        }

        // Fetch recent alerts with feedback
        let rows: Vec<(String, String, String, i32, Option<String>)> = sqlx::query_as(
            "SELECT risk_level, threat_type, summary, is_false_positive, feedback FROM agent_alerts ORDER BY created_at DESC LIMIT 50"
        ).fetch_all(&state.db).await.unwrap_or_default();

        if rows.len() < 5 {
            tokio::time::sleep(Duration::from_secs(3600)).await;
            continue;
        }

        let alerts_text = rows.iter().map(|(risk, ttype, summary, is_fp, feedback)| {
            format!("[{}] {} | fp={} | feedback={} | {}", risk.to_uppercase(), ttype, is_fp, feedback.as_deref().unwrap_or("none"), summary)
        }).collect::<Vec<_>>().join("\n");

        let current_rules = read_memory(&state.db, "threat_rules").await.unwrap_or_default();
        let current_profile = read_memory(&state.db, "camera_profile").await.unwrap_or_default();

        let system = "You are Guardian's self-improvement module. Review recent security alerts and user feedback to improve your threat detection guidelines.";
        let user = format!(
            r#"Review these recent alerts and improve the security guidelines.

Current threat rules:
{}

Current camera profile:
{}

Recent alerts (last 50):
{}

Your task:
1. Identify patterns: which alert types had feedback='bad' or is_false_positive=1? These are false positives to reduce.
2. Identify gaps: any concerning patterns that weren't flagged as high risk?
3. Suggest updated threat rules that would reduce false positives and catch real threats.
4. Output ONLY the updated threat rules text (same format as current rules). Keep it concise and actionable.
5. If no changes needed, output the current rules unchanged."#,
            current_rules, current_profile, alerts_text
        );

        // Route through the unified provider dispatcher (was hardcoded to Ollama).
        if let Ok(raw) = super::llm::call_llm(&settings, system, &user, None, false).await {
            let new_rules = raw.trim().to_string();
            // Validate before this becomes permanent.
            //
            // `threat_rules` is injected into the chat system prompt AND into clip
            // analysis, so whatever lands here shapes every answer from now on.
            // The on-device path caps generation at 512 tokens, which means a
            // ruleset could be — and was — stored truncated mid-sentence. This
            // loop runs every 6 hours on as few as 5 alerts, so one bad
            // generation used to degrade the agent permanently, with nothing to
            // undo it. `usable` is the same repeat-loop/garbage check the chat
            // path already applies to anything the user would see.
            let sane = super::llm::usable(&new_rules) && new_rules.len() < 4_000;
            if !sane {
                tracing::warn!(len = new_rules.len(),
                    "reflection: rejected unusable threat-rules draft — keeping current rules");
            }
            if sane && !new_rules.is_empty() && new_rules != current_rules {
                write_memory(&state.db, "threat_rules", &new_rules).await;
                // Also store a reflection note
                append_memory(&state.db, "pattern_reflection", &format!("Rules updated after reviewing {} alerts", rows.len())).await;
                let (tok, cid) = {
                    let s = state.settings.read().await;
                    (s.telegram_bot_token.clone(), s.telegram_chat_id.clone())
                };
                if !tok.is_empty() && !cid.is_empty() {
                    send_telegram(&tok, &cid, "🧠 Guardian has reviewed recent alerts and updated its threat detection guidelines.").await;
                }
            }
        }

        tokio::time::sleep(Duration::from_secs(6 * 3600)).await; // every 6 hours
    }
}

// ─── Ollama model pull ─────────────────────────────────────────────────────────

// ─── Main loop ────────────────────────────────────────────────────────────────

pub async fn run_agent_loop(state: Arc<AppState>) {
    tokio::time::sleep(Duration::from_secs(20)).await;

    // The startup "hallucination sanitizer" that used to live here is gone.
    //
    // It DELETEd camera_profile whenever the text contained two or more of
    // "chair, desk, mirror, curtain, ... window, door, wall, floor, monitor" —
    // which is the ordinary vocabulary of a security-camera description. "Front
    // door camera facing the driveway; the neighbour's window is on the left"
    // hit two markers and was wiped on every single app start, after which
    // chat.rs substituted "General-purpose security camera." and every analysis
    // lost its scene grounding.
    //
    // It was also guarding a write path that no longer exists: `memory.rs`
    // consumes `updates.camera_profile` from the VLM and deliberately does not
    // store it, so the only writer left is the user. A guard whose sole
    // remaining effect is deleting user-authored text is not a guard.

    // Spawn sibling tasks
    let tracker_state = Arc::clone(&state);
    tokio::spawn(async move { run_object_tracker_loop(tracker_state).await });
    let hb_state = Arc::clone(&state);
    tokio::spawn(async move { run_heartbeat_loop(hb_state).await });
    let esc_state = Arc::clone(&state);
    tokio::spawn(async move { run_escalation_loop(esc_state).await });
    let refl_state = Arc::clone(&state);
    tokio::spawn(async move { run_reflection_loop(refl_state).await });
    let live_state = Arc::clone(&state);
    tokio::spawn(async move { run_live_alert_loop(live_state).await });
    let backfill_state = Arc::clone(&state);
    tokio::spawn(async move { run_backfill_analysis(backfill_state).await });
    let dg_state = Arc::clone(&state);
    tokio::spawn(async move { run_disk_guard_loop(dg_state).await });
    let ch_state = Arc::clone(&state);
    tokio::spawn(async move { run_camera_health_loop(ch_state).await });
    // (The hourly "proactive insights" + 30-min "situation awareness" loops were
    // removed: both features were disabled at the user's request (v28 — chat
    // noise + an LLM call every 30 min), leaving loops that woke forever to
    // call empty stubs. Re-adding the feature = new loop + real impl here.)
    // Assistant: decay stale learned memories daily
    let decay_state = Arc::clone(&state);
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(24 * 3600));
        loop { interval.tick().await; decay_learned_memories(&decay_state.db).await; }
    });

    let mut last_digest_day: Option<u32> = None;

    loop {
        let settings = state.settings.read().await.clone();

        // Enabled-only gate (see run_now): the cycle's event processing — face/
        // plate recognition + the detection-summary floor — runs without an LLM;
        // the VLM is gated downstream in `analyze_event_clip`.
        if settings.agent_enabled {
            run_cycle(&state, &settings).await;
            *state.agent_last_run.write().await = Some(Utc::now().to_rfc3339());
        }

        // Daily digest — send at 08:00 local time once per day
        use chrono::{Datelike, Timelike};
        let now = Local::now();
        let today = now.day();
        if now.hour() == 8 && last_digest_day != Some(today) {
            last_digest_day = Some(today);
            let digest_state = Arc::clone(&state);
            tokio::spawn(async move { send_daily_digest(&digest_state).await });
        }

        let poll = state.settings.read().await.agent_poll_secs.max(10) as u64;
        tokio::time::sleep(Duration::from_secs(poll)).await;
    }
}

/// Runs every 60 s. Checks whether any user-tracked objects have gone missing
/// and sends a Telegram alert if they have been absent for longer than the threshold.
pub(super) async fn run_object_tracker_loop(state: Arc<AppState>) {
    tokio::time::sleep(Duration::from_secs(60)).await;

    loop {
        tokio::time::sleep(Duration::from_secs(60)).await;

        let json = read_memory(&state.db, "tracked_objects").await
            .unwrap_or_else(|| "[]".to_string());
        let mut tracked: Vec<TrackedObject> = serde_json::from_str(&json).unwrap_or_default();
        if tracked.is_empty() { continue; }

        // Build a flat set of all labels currently visible across all cameras
        let now_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let scene = state.scene_objects.read().await.clone();
        let last_update = state.scene_last_update.read().await.clone();

        // Only consider cameras that reported a scene update within the last 30 s
        let visible_labels: std::collections::HashSet<String> = scene
            .iter()
            .filter(|(cam, _)| {
                last_update.get(cam).map(|t| now_secs.saturating_sub(*t) < 30).unwrap_or(false)
            })
            .flat_map(|(_, objs)| objs.iter().map(|o| o.label.to_lowercase()))
            .collect();

        let camera_on = !visible_labels.is_empty()
            || last_update.values().any(|t| now_secs.saturating_sub(*t) < 30);

        if !camera_on { continue; } // can't declare something missing if camera is off

        let (token, chat_id, camera_name) = {
            let s = state.settings.read().await;
            (s.telegram_bot_token.clone(), s.telegram_chat_id.clone(), s.camera_name.clone())
        };
        let cam = if camera_name.is_empty() { "Security Camera" } else { &camera_name };
        let now = Utc::now();
        let mut dirty = false;

        for obj in tracked.iter_mut() {
            // Fuzzy match: "bottle" matches "wine bottle", "water bottle", etc.
            let is_visible = visible_labels.iter().any(|l| {
                l.contains(&obj.label.to_lowercase())
                    || obj.label.to_lowercase().contains(l.as_str())
            });

            if is_visible {
                obj.last_seen_at = Some(now.to_rfc3339());
                if obj.alert_sent {
                    // Object reappeared — notify user
                    let msg = format!(
                        "✅ {} is back in view on {}.",
                        obj.display_name, cam
                    );
                    send_telegram(&token, &chat_id, &msg).await;
                    obj.alert_sent = false;
                }
                dirty = true;
            } else if let Some(ref last_seen) = obj.last_seen_at.clone() {
                if let Ok(last) = chrono::DateTime::parse_from_rfc3339(last_seen) {
                    let mins_missing = (now - last.with_timezone(&Utc)).num_minutes();
                    if mins_missing >= obj.alert_after_mins as i64 && !obj.alert_sent {
                        let msg = format!(
                            "⚠️ Object Missing: {}\n\nLast seen {} minutes ago on {}.\nI've been watching and it hasn't appeared since.\n\nIf someone moved it, ask me to \"stop tracking {}\".",
                            obj.display_name, mins_missing, cam, obj.display_name
                        );
                        send_telegram(&token, &chat_id, &msg).await;
                        obj.alert_sent = true;
                        dirty = true;
                    }
                }
            }
            // If last_seen_at is None, we've never seen it — don't alert (user may have added it before the object was in frame)
        }

        if dirty {
            let updated = serde_json::to_string(&tracked).unwrap_or_default();
            write_memory(&state.db, "tracked_objects", &updated).await;
        }
    }
}
