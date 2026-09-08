//! Semantic alert conditions (assistant-parity).
//!
//! Users write plain-English conditions like:
//!   * "Alert if someone approaches the front door after 10pm"
//!   * "Notify me if a vehicle I don't recognise parks in the driveway"
//!   * "Alert if any motion detected between midnight and 6am"
//!
//! On every analysed event, the LLM evaluates all enabled conditions against
//! the event's description. Matching conditions dispatch their own alert.

use std::sync::Arc;

use chrono::Utc;
use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;
use uuid::Uuid;

use tauri::Emitter;

use crate::{AppState, Settings};
use super::memory::{read_memory, write_memory};
use super::llm::call_llm;
use super::analysis::risk_meets_threshold;
use super::dispatch_intelligence_alert;

// ─── assistant-parity: Semantic Alert Conditions ─────────────────────────────────
//
// Users write plain-English conditions like:
//   "Alert if someone approaches the front door after 10pm"
//   "Notify me if a vehicle I don't recognise parks in the driveway"
//   "Alert if any motion detected between midnight and 6am"
//
// On every analyzed event, the LLM evaluates ALL enabled conditions against
// the event's description. Matching conditions dispatch their own alert.

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertCondition {
    pub id:            String,
    pub name:          String,
    pub condition:     String,   // plain English
    pub channels:      String,   // "all" | "telegram" | "discord" | "silent"
    pub min_risk:      String,
    pub enabled:       bool,
    pub trigger_count: u32,
    pub created_at:    String,
}

pub async fn create_alert_condition(db: &SqlitePool, name: &str, condition: &str, channels: &str, min_risk: &str) -> AlertCondition {
    let id  = Uuid::new_v4().to_string();
    let now = Utc::now().to_rfc3339();
    sqlx::query(
        "INSERT INTO alert_conditions(id,name,condition,channels,min_risk,enabled,trigger_count,created_at)
         VALUES(?,?,?,?,?,1,0,?)"
    ).bind(&id).bind(name).bind(condition).bind(channels).bind(min_risk).bind(&now)
    .execute(db).await.ok();
    AlertCondition { id, name: name.into(), condition: condition.into(), channels: channels.into(),
        min_risk: min_risk.into(), enabled: true, trigger_count: 0, created_at: now }
}

pub async fn list_alert_conditions(db: &SqlitePool) -> Vec<AlertCondition> {
    sqlx::query_as::<_, (String,String,String,String,String,bool,u32,String)>(
        "SELECT id,name,condition,channels,min_risk,enabled,trigger_count,created_at
         FROM alert_conditions ORDER BY created_at DESC"
    ).fetch_all(db).await.unwrap_or_default()
    .into_iter().map(|(id,name,condition,channels,min_risk,enabled,trigger_count,created_at)| {
        AlertCondition { id, name, condition, channels, min_risk, enabled, trigger_count, created_at }
    }).collect()
}

pub async fn delete_alert_condition(db: &SqlitePool, id: &str) {
    sqlx::query("DELETE FROM alert_conditions WHERE id=?").bind(id).execute(db).await.ok();
}

pub async fn toggle_alert_condition(db: &SqlitePool, id: &str, enabled: bool) {
    sqlx::query("UPDATE alert_conditions SET enabled=? WHERE id=?")
        .bind(enabled as i32).bind(id).execute(db).await.ok();
}

/// Which of these rules match this event? Returns indices into `rules`.
///
/// ONE model call for every rule, not one call per rule. The original asked
/// "does this event match rule X? YES/NO" in a loop — on the on-device engine,
/// which is a single thread serving one generation at a time
/// (`local_llm.rs:20-24`), five rules meant five sequential generations for
/// every event the cameras recorded.
///
/// Matching a description against a list is classification, which is what a
/// small model is actually good at — the same doc comment that forbids it from
/// judging risk credits it with 91% JSON compliance and 81% tool use.
async fn rules_matching(
    settings: &Settings, when: &str, description: &str, rules: &[&AlertCondition],
) -> Vec<usize> {
    let list = rules.iter().enumerate()
        .map(|(i, c)| format!("{}. {}", i + 1, c.condition))
        .collect::<Vec<_>>().join("\n");
    let system = "You match a security event against a numbered list of user rules. \
                  Reply with ONLY the numbers that match, separated by commas. \
                  If none match, reply NONE. No words, no explanation.";
    let user = format!(
        "Event at {when}:\n\"{description}\"\n\nRules:\n{list}\n\nWhich numbers match?");

    let Ok(reply) = call_llm(settings, system, &user, None, false).await else {
        return Vec::new();
    };
    parse_rule_numbers(&reply, rules.len())
}

/// Rule indices out of a small model's reply.
///
/// A digit scan, not a parser: small models pad answers ("Rules 1 and 3 match")
/// and an unparsed reply must not silently read as "nothing matched" — that is
/// a missed alert, which is the failure this whole feature exists to prevent.
/// Out-of-range numbers are dropped rather than clamped: a model that answers
/// "7" against three rules is guessing, not pointing at rule 3.
fn parse_rule_numbers(reply: &str, count: usize) -> Vec<usize> {
    if reply.trim().to_uppercase().starts_with("NONE") { return Vec::new(); }
    let mut hits: Vec<usize> = Vec::new();
    // Clause by clause, so a number the model EXCLUDED is not read as a match.
    //
    // This was a flat digit scan over the whole reply, so "Rule 2 does not
    // match, rule 1 does" fired BOTH rules — the padding the doc comment above
    // anticipates is exactly what breaks a flat scan. Splitting on clause
    // boundaries and skipping denied clauses leaves the specified format
    // ("1,3") behaving identically: no clause there carries a negation.
    for clause in reply.split(|c: char| matches!(c, ',' | '.' | ';' | ':') || c.is_control()) {
        if super::retrieve::negated(clause) { continue; }
        for tok in clause.split(|c: char| !c.is_ascii_digit()) {
            if tok.is_empty() { continue; }
            if let Ok(n) = tok.parse::<usize>() {
                if (1..=count).contains(&n) && !hits.contains(&(n - 1)) { hits.push(n - 1); }
            }
        }
    }
    hits
}

/// Rules that could fire for an event at this risk level.
///
/// Pure and separate, so "no model call unless a rule could actually fire" is a
/// property with a test rather than a claim in a comment.
fn candidate_rules<'a>(all: &'a [AlertCondition], risk: &str) -> Vec<&'a AlertCondition> {
    all.iter()
        .filter(|c| c.enabled && risk_meets_threshold(risk, &c.min_risk))
        .collect()
}

/// Evaluate the user's plain-English alert rules against a completed event.
///
/// This is called from `clip::analyze_event_clip` — the LIVE pipeline. It used
/// to hang off `analysis::analyze_event`, which had no callers, so every rule a
/// user wrote in the UI silently never fired and `trigger_count` stayed at zero.
///
/// `cam_id` and `category` are here because the send must pass
/// `channel_alert_allowed` like every other alert. The previous version called
/// `send_telegram` directly, so a rule could message the user through a muted
/// category, a disabled camera, or below their alert threshold.
pub async fn evaluate_alert_conditions(
    state: &Arc<AppState>,
    settings: &Settings,
    event_description: &str,
    event_time: &str,
    risk_level: &str,
    event_id: &str,
    cam_id: u8,
    category: &str,
) {
    // ── Rust pre-filter: no model call unless a rule could actually fire ──────
    let conditions = list_alert_conditions(&state.db).await;
    let candidates = candidate_rules(&conditions, risk_level);
    if candidates.is_empty() { return; }
    // Nothing this rule could send would survive the user's alert settings, so
    // don't spend a generation deciding whether it matched.
    if !channel_alert_allowed(settings, risk_level, cam_id, category) { return; }
    if event_description.trim().is_empty() { return; }

    let dt = chrono::DateTime::parse_from_rfc3339(event_time)
        .map(|t| t.with_timezone(&chrono::Local).format("%A %H:%M").to_string())
        .unwrap_or_else(|_| event_time.to_string());

    for idx in rules_matching(settings, &dt, event_description, &candidates).await {
        let condition = candidates[idx];
        tracing::info!("Alert rule matched: '{}' for event {}",
            condition.name, event_id.chars().take(8).collect::<String>());

        sqlx::query("UPDATE alert_conditions SET trigger_count=trigger_count+1 WHERE id=?")
            .bind(&condition.id).execute(&state.db).await.ok();
        sqlx::query("UPDATE agent_alerts SET condition_id=? WHERE event_id=?")
            .bind(&condition.id).bind(event_id).execute(&state.db).await.ok();

        // Telegram gets the SAME evidence as any other answer — the picture, the
        // record, and a tap for the footage — instead of a line of text with
        // literal *asterisks* (`send_telegram` sets no parse_mode).
        let chan = condition.channels.as_str();
        if (chan == "all" || chan == "telegram")
            && !settings.telegram_bot_token.is_empty() && !settings.telegram_chat_id.is_empty()
        {
            super::dispatch::send_telegram(&settings.telegram_bot_token, &settings.telegram_chat_id,
                &format!("🔔 {} — your rule matched.\n{}", condition.name, condition.condition)).await;
            if let Some(card) = super::evidence::card_for(state, event_id).await {
                super::dispatch::render_evidence(
                    state, &settings.telegram_bot_token, &settings.telegram_chat_id,
                    &[super::evidence::Evidence::Events {
                        label: condition.name.clone(), cards: vec![card], play: false,
                    }]).await;
            }
        }

        state.app_handle.emit("intelligence:alert", serde_json::json!({
            "type":      "condition_match",
            "condition": condition.name,
            "summary":   format!("🔔 {} — {}", condition.name, event_description),
        })).ok();
    }
}

/// Map an event's (dominant_label, event_category) onto the user-facing alert
/// category the 👁 Alert-filter menu toggles. Deliberately coarse: users think
/// "people / vehicles / animals / sounds / everything else", not COCO classes.
pub fn event_category_of(dominant_label: &str, event_category: &str) -> &'static str {
    if event_category.eq_ignore_ascii_case("audio") { return "audio"; }
    match dominant_label.to_ascii_lowercase().as_str() {
        "person" => "person",
        "car" | "truck" | "bus" | "motorcycle" | "bicycle" | "vehicle" => "vehicle",
        "dog" | "cat" | "bird" | "horse" | "sheep" | "cow" | "bear" | "animal" => "animal",
        _ => "other",
    }
}

/// Whether a channel alert (Telegram/Discord/etc.) should be sent for an event.
/// One chokepoint for the user's Telegram `/menu` preferences:
///   • per-camera suppression (`alert_disabled_cameras`),
///   • minimum risk level (`alert_min_risk`, incl. "off" = none),
///   • quiet hours (only `critical` passes the do-not-disturb window),
///   • muted categories (`alert_muted_categories` — 👁 Alert filter; the event
///     still records + analyzes, the user just isn't messaged about it).
pub(super) fn channel_alert_allowed(settings: &Settings, risk: &str, cam_id: u8, category: &str) -> bool {
    if settings.alert_disabled_cameras.contains(&cam_id) { return false; }
    if settings.alert_muted_categories.iter().any(|c| c == category) { return false; }
    if !super::analysis::risk_meets_threshold(risk, &settings.alert_min_risk) { return false; }
    if is_quiet_hours(settings) && risk != "critical" { return false; }
    true
}

/// Returns true if the current local time is within the configured quiet hours.
pub fn is_quiet_hours(settings: &Settings) -> bool {
    if !settings.quiet_hours_enabled { return false; }
    if settings.quiet_hours_start.is_empty() || settings.quiet_hours_end.is_empty() { return false; }

    
    let now_hhmm = chrono::Local::now().format("%H:%M").to_string();

    let start = &settings.quiet_hours_start;
    let end   = &settings.quiet_hours_end;

    // Handles overnight ranges (e.g. 22:00 → 07:00)
    if start <= end {
        &now_hhmm >= start && &now_hhmm < end
    } else {
        &now_hhmm >= start || &now_hhmm < end
    }
}

/// Clip text search — search all historical footage by AI-generated descriptions.
/// Query is matched against ai_summary using LIKE (fast, no LLM needed).
/// Falls back to semantic matching via LLM for natural language queries.
pub async fn search_clips(state: &Arc<AppState>, query: &str) -> Vec<serde_json::Value> {
    let q = query.trim();
    if q.is_empty() { return vec![]; }

    // Extract keywords for SQL LIKE search
    let keywords: Vec<String> = q.split_whitespace()
        .filter(|w| w.len() > 2)
        .map(|w| w.to_lowercase())
        .collect();

    if keywords.is_empty() { return vec![]; }

    // BOUND, not interpolated.
    //
    // These keywords came straight off the user's query and were being formatted
    // into the SQL inside quotes: a term like `a'||1=1--` closed the string and
    // continued as code. Reachable from the frontend (`api.searchClips`) and, via
    // the agent, from Telegram — which is untrusted input by definition.
    //
    // The placeholder count is derived from the keyword COUNT, so the shape of the
    // statement can never be influenced by its content.
    let per_kw = "(LOWER(COALESCE(me.ai_summary,'')) LIKE ? \
                  OR LOWER(COALESCE(aa.threat_type,'')) LIKE ?)";
    let where_clause = vec![per_kw; keywords.len()].join(" OR ");

    // Same columns, same `card()`, as every other card query — this branch used
    // to hand-roll its own JSON and omit the thumbnail entirely, so a searched
    // event rendered as a blank tile while the identical event reached from the
    // events filter rendered with its picture.
    let sql = format!(
        "SELECT {CARD_COLS}
         FROM motion_events me
         LEFT JOIN agent_alerts aa ON aa.event_id = me.id
         WHERE me.ai_summary IS NOT NULL AND ({where_clause})
         ORDER BY me.started_at DESC LIMIT 20"
    );

    let mut qy = sqlx::query_as::<_, CardRow>(&sql);
    for k in &keywords {
        let like = format!("%{k}%");
        qy = qy.bind(like.clone()).bind(like);
    }
    let rows = qy.fetch_all(&state.db).await.unwrap_or_default();

    rows.into_iter().map(|r| card(&state.data_dir, r)).collect()
}

/// Structured event explorer — returns matching events as JSON for the frontend to render as cards.
/// Filter string is free-form: "today", "high", "person today", "last hour critical", etc.
// Two time filters deliberately resolve to the same 24-hour window: an explicit
// "last 24h" request and the default when nothing matches.
#[allow(clippy::if_same_then_else)]
pub async fn explore_events(state: &Arc<AppState>, filter: &str) -> Vec<serde_json::Value> {
    // `ids=<uuid>,<uuid>,…` — an EXACT set chosen by the resolver, short-circuiting
    // every clause below. Read from the RAW filter: ids are opaque and must never
    // be case-folded.
    // `str::get`, not `&raw[..4]`: the filter can come straight from a model, and
    // a byte slice through a multi-byte character panics. `[SHOW_EVENTS:日本語]`
    // was enough to take down the whole reply.
    let raw = filter.trim();
    if raw.get(..4).is_some_and(|p| p.eq_ignore_ascii_case("ids=")) {
        return events_by_ids(&state.db, &state.data_dir, &parse_ids(&raw[4..])).await;
    }
    let f = filter.to_lowercase();

    // Time window
    let time_clause = if f.contains("last hour") || f.contains("past hour") {
        "started_at > datetime('now', '-1 hour')"
    } else if f.contains("today") {
        "date(started_at, 'localtime') = date('now', 'localtime')"
    } else if f.contains("yesterday") {
        "date(started_at, 'localtime') = date('now', '-1 day', 'localtime')"
    } else if f.contains("last night") || f.contains("tonight") {
        "started_at > datetime('now', '-12 hours') AND cast(strftime('%H', started_at, 'localtime') as int) >= 20"
    } else if f.contains("this week") || f.contains("week") {
        "started_at > datetime('now', '-7 days')"
    } else if f.contains("last 24") || f.contains("24 hour") {
        "started_at > datetime('now', '-24 hours')"
    } else {
        "started_at > datetime('now', '-24 hours')" // default: last 24h
    };

    // Risk filter — supports both old scale and new Agies scale
    let risk_clause = if f.contains("critical") {
        "AND aa.risk_level = 'critical'"
    } else if f.contains("suspicious") || f.contains("high") {
        "AND aa.risk_level IN ('suspicious', 'high', 'critical')"
    } else if f.contains("monitor") || f.contains("medium") {
        "AND aa.risk_level IN ('monitor', 'medium')"
    } else if f.contains("normal") || f.contains("low") {
        "AND aa.risk_level IN ('normal', 'low')"
    } else {
        ""
    };

    // Type filter
    let type_clause = if f.contains("person") {
        "AND (aa.threat_type = 'person' OR me.ai_summary LIKE '%person%')"
    } else if f.contains("vehicle") || f.contains("car") {
        // `aa.threat_type` alone missed every event the edge model classified as a
        // vehicle that never got an `agent_alerts` row. Same literal set
        // `retrieve::push_common` uses, so one vocabulary serves both.
        "AND (aa.threat_type = 'vehicle' OR me.event_category = 'vehicle'           OR me.dominant_label IN ('car','truck','bus','motorcycle','bicycle','vehicle'))"
    } else if f.contains("false alarm") || f.contains("false positive") {
        "AND aa.is_false_positive = 1"
    } else {
        ""
    };

    let sql = format!(
        "SELECT {CARD_COLS}
         FROM motion_events me
         LEFT JOIN agent_alerts aa ON aa.event_id = me.id
         WHERE {time_clause} {risk_clause} {type_clause}
         ORDER BY me.started_at DESC LIMIT 20"
    );

    let rows: Vec<CardRow> =
        sqlx::query_as(&sql).fetch_all(&state.db).await.unwrap_or_default();

    rows.into_iter().map(|r| card(&state.data_dir, r)).collect()
}

/// One row of the card query, as every branch selects it.
#[allow(clippy::type_complexity)]
pub(super) type CardRow = (String, String, Option<f64>, Option<String>, Option<String>,
                Option<String>, Option<String>, Option<String>, i64);

/// The columns every card query must select, in `CardRow` order. Written once
/// so a new field reaches the filter branch, the `ids=` branch and search
/// together — the three used to select different columns, which is how Telegram
/// ended up rendering a card with no thumbnail.
pub(super) const CARD_COLS: &str =
    "me.id, me.started_at, me.duration_secs, me.ai_summary, me.clip_path,
     aa.risk_level, aa.threat_type, me.thumbnail, me.cam_id";

/// The card shape the chat and Telegram both render. Extracted so the filter
/// branch and the `ids=` branch cannot drift into two different contracts.
pub(super) fn card(data_dir: &std::path::Path, r: CardRow) -> serde_json::Value {
    let (id, started_at, dur, summary, clip, risk, ttype, thumb, cam) = r;
    let ts = chrono::DateTime::parse_from_rfc3339(&started_at)
        .map(|t| t.with_timezone(&chrono::Local).format("%b %d %H:%M").to_string())
        .unwrap_or_else(|_| started_at.clone());
    // Resolve @file: blob refs so the chat's evidence cards can render the
    // thumbnail directly (bare base64, same contract as the Events feed).
    let thumbnail = thumb.map(|t| crate::blobstore::resolve(data_dir, &t))
        .filter(|t| !t.is_empty());
    serde_json::json!({
        "id":          id,
        "started_at":  started_at,
        "ts":          ts,
        "cam":         cam,
        "duration":    dur.map(|d| format!("{:.0}s", d)),
        "summary":     summary,
        "risk_level":  risk.unwrap_or_else(|| "low".into()),
        "threat_type": ttype.unwrap_or_else(|| "motion".into()),
        "has_clip":    clip.is_some(),
        "thumbnail":   thumbnail,
    })
}

/// Split an `ids=` payload into safe, deduplicated event ids.
///
/// Event ids are opaque uuids: parsed from the RAW filter, never the lowercased
/// copy. Anything outside `[A-Za-z0-9-]` is dropped rather than escaped, and the
/// count is capped — so the placeholder count downstream is a function of this
/// vector's length and can never be influenced by the text.
pub(super) fn parse_ids(raw: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for part in raw.split(',') {
        let id = part.trim();
        if id.is_empty() || id.len() > 64 { continue; }
        if !id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-') { continue; }
        if !out.iter().any(|e| e == id) { out.push(id.to_string()); }
        if out.len() == 20 { break; }
    }
    out
}

/// Cards for an EXACT set of events, in the order asked for.
///
/// This is what lets the agent show the very events its answer describes. The
/// old path rebuilt a lossy filter string ("last 24") from the query, so the
/// cards regularly covered a different window than the prose.
///
/// No time clause at all: these rows were already chosen, and re-filtering them
/// by date would silently drop the ones being described.
pub(super) async fn events_by_ids(
    db: &sqlx::SqlitePool,
    data_dir: &std::path::Path,
    ids: &[String],
) -> Vec<serde_json::Value> {
    if ids.is_empty() { return Vec::new(); }
    let placeholders = vec!["?"; ids.len()].join(",");
    let sql = format!(
        "SELECT {CARD_COLS}
         FROM motion_events me
         LEFT JOIN agent_alerts aa ON aa.event_id = me.id
         WHERE me.id IN ({placeholders})");
    let mut q = sqlx::query_as::<_, CardRow>(&sql);
    for id in ids { q = q.bind(id); }
    let rows = q.fetch_all(db).await.unwrap_or_default();

    // Rebuild the requested order — `IN (…)` returns rows arbitrarily. Keying by
    // id also collapses the `LEFT JOIN agent_alerts` fan-out: an event carrying
    // two alert rows renders as ONE card, not two identical ones.
    let mut by_id: std::collections::HashMap<String, serde_json::Value> =
        rows.into_iter().map(|r| (r.0.clone(), card(data_dir, r))).collect();
    ids.iter().filter_map(|id| by_id.remove(id)).collect()
}

/// Deterministic mermaid chart of TODAY's activity (events per hour, by
/// category) — generated from SQL, never by the LLM, so it always renders.
/// The chat replaces a [DAY_CHART] tag with this fence (grok-build pattern:
/// diagrams as first-class agent output).
pub async fn day_chart_mermaid(db: &sqlx::SqlitePool) -> String {
    let rows: Vec<(i64, i64)> = sqlx::query_as(
        "SELECT CAST(strftime('%H', started_at, 'localtime') AS INTEGER), COUNT(*)
           FROM motion_events
          WHERE date(started_at,'localtime') = date('now','localtime')
          GROUP BY 1"
    ).fetch_all(db).await.unwrap_or_default();
    let mut counts = [0i64; 24];
    for (h, n) in rows { if (0..24).contains(&h) { counts[h as usize] = n; } }
    if counts.iter().all(|&n| n == 0) { return "No events yet today.".to_string(); }
    let x: Vec<String> = (0..24).map(|h| format!("\"{h:02}\"")).collect();
    let y: Vec<String> = counts.iter().map(|n| n.to_string()).collect();
    format!(
        "```mermaid\nxychart-beta\n  title \"Today's activity by hour\"\n  x-axis [{}]\n  y-axis \"events\"\n  bar [{}]\n```",
        x.join(", "), y.join(", "))
}

/// Compact text fallback of the same chart for surfaces that can't render
/// mermaid (Telegram).
pub async fn day_chart_text(db: &sqlx::SqlitePool) -> String {
    let rows: Vec<(i64, i64)> = sqlx::query_as(
        "SELECT CAST(strftime('%H', started_at, 'localtime') AS INTEGER)/6, COUNT(*)
           FROM motion_events
          WHERE date(started_at,'localtime') = date('now','localtime')
          GROUP BY 1"
    ).fetch_all(db).await.unwrap_or_default();
    let mut buckets = [0i64; 4];
    for (b, n) in rows { if (0..4).contains(&b) { buckets[b as usize] = n; } }
    format!("Activity today — night 00-06: {} · morning 06-12: {} · afternoon 12-18: {} · evening 18-24: {}",
        buckets[0], buckets[1], buckets[2], buckets[3])
}

/// Natural-language event query — passes full conversation history so "show me those"
/// and other contextual references work correctly.
pub async fn query_events_nl(state: &Arc<AppState>, question: &str) -> String {
    // Redirect to the full chat_with_agent which has all memory, person profiles, and history
    // This way the user gets the same smart agent rather than a stripped-down query engine
    let s = state.settings.read().await.clone();

    // Fetch events with risk levels for richer context
    let events: Vec<(String, String, Option<f64>, Option<String>, Option<String>)> =
        sqlx::query_as(
            "SELECT me.id, me.started_at, me.duration_secs, me.ai_summary, aa.risk_level
             FROM motion_events me
             LEFT JOIN agent_alerts aa ON aa.event_id = me.id
             ORDER BY me.started_at DESC LIMIT 50"
        )
        .fetch_all(&state.db).await.unwrap_or_default();

    if events.is_empty() {
        return "No events recorded yet. Start the camera to begin monitoring.".to_string();
    }

    // Load ALL person profiles from memory (not just recent)
    let person_profiles: Vec<(String, String)> = sqlx::query_as(
        "SELECT key, value FROM agent_memory
         WHERE key LIKE 'person_%' OR key LIKE 'appearance_cam%'
         ORDER BY updated_at DESC LIMIT 50"
    ).fetch_all(&state.db).await.unwrap_or_default();

    // Load camera profile and environment context
    let camera_profile = read_memory(&state.db, "camera_profile").await.unwrap_or_default();
    let camera_name    = if s.camera_name.is_empty() { "Security Camera" } else { &s.camera_name };

    // Build event list — include risk level and event ID for context
    let now = chrono::Local::now();
    let mut event_list = String::new();
    for (id, started_at, duration, summary, risk) in &events {
        let ts = chrono::DateTime::parse_from_rfc3339(started_at)
            .map(|t| t.with_timezone(&chrono::Local).format("%b %d %H:%M").to_string())
            .unwrap_or_else(|_| started_at[..16].to_string());
        let dur_str  = duration.map(|d| format!("{:.0}s", d)).unwrap_or_default();
        let sum_str  = summary.as_deref().unwrap_or("Motion detected");
        let risk_str = risk.as_deref().unwrap_or("?");
        event_list.push_str(&format!(
            "[{ts}] [{risk_str}] {sum_str}{} (id: {})\n",
            if dur_str.is_empty() { String::new() } else { format!(" · {dur_str}") },
            &id[..8.min(id.len())]
        ));
    }

    // Build person profile section — this is how the agent remembers who it has seen
    let person_ctx = if person_profiles.is_empty() {
        "No person descriptions recorded yet.".to_string()
    } else {
        person_profiles.iter()
            .map(|(k, v)| format!("• [{}] {}", k.replace('_', " "), v))
            .collect::<Vec<_>>().join("\n")
    };

    let system = format!(
        r#"You are Guardian, an expert AI security analyst watching "{camera_name}".
You have full memory of every person and event you have ever observed.
Current time: {}

## Camera Environment
{camera_profile}

## Person Profiles (everyone you have observed, with descriptions)
{person_ctx}

## All Events (newest first)
{event_list}

IMPORTANT RULES:
- When the user says "show me those", "those events", "that person" — use the CONVERSATION HISTORY to understand what they're referring to.
- When the user describes a person by appearance ("the man at the desk", "the person in blue"), search the person profiles above to find who matches.
- NEVER make up environment details (lounge, sofa, etc.) that aren't in the camera environment section.
- If you don't have information, say so clearly and ask for clarification.
- When showing events, use the event ID and timestamp so the user can locate them.
- Differentiate between multiple people when they appear in the same event."#,
        now.format("%Y-%m-%d %H:%M %Z")
    );

    match call_llm(&s, &system, question, None, false).await {
        Ok(answer) => answer,
        Err(e) => format!("Guardian AI error: {e}. Check AI provider settings."),
    }
}

/// Track repeat visitor appearances — alert when same re-ID appears > threshold times today.
pub async fn check_repeat_visitor(state: &Arc<AppState>, person_id: &str, cam_id: u8) {
    let s = state.settings.read().await.clone();
    if !s.repeat_visitor_detection { return; }
    let threshold = s.repeat_visitor_threshold as i64;
    drop(s);

    // Key on the DURABLE identity when this body is face-anchored — otherwise the
    // same person's many appearance-fragments each start their own counter and the
    // threshold never trips (the fragmentation that breaks recurring-visitor learning).
    let resolved: Option<String> = sqlx::query_scalar::<_, String>(
        "SELECT k.name FROM body_embeddings b JOIN known_persons k ON k.id = b.known_person_id \
         WHERE b.person_id = ? AND b.known_person_id IS NOT NULL LIMIT 1"
    ).bind(person_id).fetch_optional(&state.db).await.ok().flatten();
    let id_key = resolved.clone().unwrap_or_else(|| person_id.to_string());
    let key = format!("repeat_visitor_{}_{}", cam_id, id_key);
    let today = chrono::Local::now().format("%Y-%m-%d").to_string();
    let count_key = format!("{key}_{today}");

    let current: i64 = read_memory(&state.db, &count_key).await
        .and_then(|v| v.parse().ok()).unwrap_or(0);
    let new_count = current + 1;
    write_memory(&state.db, &count_key, &new_count.to_string()).await;

    if new_count == threshold {
        let who = resolved.unwrap_or_else(|| format!("ID {}", person_id.chars().take(6).collect::<String>()));
        let summary = format!(
            "Same person ({}) has appeared {} times today on cam{}. Possible loitering or surveillance.",
            who, new_count, cam_id + 1
        );
        tracing::warn!("[on-device assistants] Repeat visitor: {}", summary);
        dispatch_intelligence_alert(state, "repeat_visitor", &summary, cam_id, None).await;
    }
}

#[cfg(test)]
mod alert_rule_tests {
    use super::*;

    fn rule(name: &str, min_risk: &str, enabled: bool) -> AlertCondition {
        AlertCondition {
            id: name.into(), name: name.into(), condition: format!("{name} happened"),
            channels: "telegram".into(), min_risk: min_risk.into(), enabled,
            trigger_count: 0, created_at: "2026-08-07T00:00:00Z".into(),
        }
    }

    /// The pre-filter is what keeps this feature affordable. Every rule that
    /// survives it costs a share of ONE generation; every rule it drops costs
    /// nothing. On the on-device engine — one thread, one generation at a time —
    /// that is the difference between a usable feature and a stalled recorder.
    #[test]
    fn only_rules_that_could_fire_reach_the_model() {
        let all = vec![
            rule("shed",     "normal",     true),   // fires at any risk
            rule("intruder", "critical",   false),  // disabled
            rule("prowler",  "critical",   true),   // too high for a normal event
            rule("visitor",  "monitor",    true),   // too high for a normal event
        ];
        let names = |risk: &str| candidate_rules(&all, risk).iter()
            .map(|c| c.name.as_str()).collect::<Vec<&str>>();

        assert_eq!(names("normal"), vec!["shed"], "a quiet event wakes only the open rule");
        assert_eq!(names("critical"), vec!["shed", "prowler", "visitor"],
                   "a critical event wakes every enabled rule");
        assert!(!names("critical").contains(&"intruder"),
                "a disabled rule never reaches the model");
    }

    /// Small models pad their answers. An unparsed reply must not read as
    /// "nothing matched" — that is a silently missed alert.
    #[test]
    fn rule_numbers_survive_a_chatty_reply() {
        assert_eq!(parse_rule_numbers("1,3", 3), vec![0, 2]);
        assert_eq!(parse_rule_numbers("Rules 1 and 3 match.", 3), vec![0, 2]);
        assert_eq!(parse_rule_numbers(" 2 ", 3), vec![1]);
        assert!(parse_rule_numbers("NONE", 3).is_empty());
        assert!(parse_rule_numbers("none of them", 3).is_empty());
        // Repeats collapse; out-of-range is dropped, not clamped onto a real rule.
        assert_eq!(parse_rule_numbers("1, 1, 2", 3), vec![0, 1]);
        // A number the model EXCLUDED must not fire. The flat digit scan this
        // replaced returned [1, 0] here, so both rules alerted.
        assert_eq!(parse_rule_numbers("Rule 2 does not match, rule 1 does", 3), vec![0]);
        assert!(parse_rule_numbers("No rules match", 3).is_empty());
        assert!(parse_rule_numbers("7", 3).is_empty(),
                "a number past the list is a guess, not rule 3");
        assert!(parse_rule_numbers("", 3).is_empty());
    }

    /// The alert-gating chokepoint. A rule must not be able to message the user
    /// through a muted category, a disabled camera, or below their threshold —
    /// the previous implementation called `send_telegram` directly and could.
    #[test]
    fn a_matched_rule_still_obeys_the_alert_settings() {
        let mut s = crate::Settings {
            alert_min_risk: "normal".into(), ..Default::default()
        };
        assert!(channel_alert_allowed(&s, "normal", 0, "person"));

        s.alert_muted_categories = vec!["person".into()];
        assert!(!channel_alert_allowed(&s, "normal", 0, "person"),
                "a muted category silences rules too");

        s.alert_muted_categories.clear();
        s.alert_disabled_cameras = vec![0];
        assert!(!channel_alert_allowed(&s, "normal", 0, "person"),
                "a disabled camera silences rules too");
    }
}

#[cfg(test)]
mod category_tests {
    use super::event_category_of;

    #[test]
    fn labels_map_to_user_categories() {
        assert_eq!(event_category_of("person", "motion"), "person");
        assert_eq!(event_category_of("Person", "motion"), "person"); // case-insensitive
        assert_eq!(event_category_of("car", "motion"), "vehicle");
        assert_eq!(event_category_of("truck", "motion"), "vehicle");
        assert_eq!(event_category_of("bicycle", "motion"), "vehicle");
        assert_eq!(event_category_of("dog", "motion"), "animal");
        assert_eq!(event_category_of("cat", "motion"), "animal");
        assert_eq!(event_category_of("", "audio"), "audio");       // audio wins regardless of label
        assert_eq!(event_category_of("Speech", "AUDIO"), "audio");
        assert_eq!(event_category_of("", "motion"), "other");       // unknown/empty → other
        assert_eq!(event_category_of("chair", "motion"), "other");
    }
}

#[cfg(test)]
mod card_tests {
    use super::*;

    #[test]
    fn ids_are_sanitised_and_capped() {
        // Only uuid-shaped characters survive; order is kept; duplicates collapse.
        let got = parse_ids("a-1, b-2 ,a-1,, c-3");
        assert_eq!(got, vec!["a-1".to_string(), "b-2".into(), "c-3".into()]);

        // Anything that isn't [A-Za-z0-9-] is dropped, not escaped — so the
        // placeholder count downstream can never be influenced by the text.
        assert!(parse_ids("'); DROP TABLE motion_events;--").is_empty());
        assert!(parse_ids("a'b").is_empty());
        assert!(parse_ids(&"x".repeat(65)).is_empty(), "over-long id rejected");

        let many: String = (0..40).map(|i| format!("id{i},")).collect();
        assert_eq!(parse_ids(&many).len(), 20, "hard cap");
    }

    /// Cards must be the events asked for, in the order asked for — and an event
    /// carrying two alert rows must render ONCE, not twice.
    #[tokio::test]
    async fn events_by_ids_is_exact() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        crate::db::init_db(&pool).await.unwrap();
        for (id, at) in [("a", "2026-07-01T10:00:00Z"),   // 40+ days old on purpose
                         ("b", "2026-08-02T10:00:00Z"),
                         ("c", "2026-08-02T11:00:00Z")] {
            sqlx::query("INSERT INTO motion_events(id, started_at) VALUES(?,?)")
                .bind(id).bind(at).execute(&pool).await.unwrap();
        }
        // Two alerts on one event — the LEFT JOIN would otherwise fan out.
        for n in 0..2 {
            sqlx::query(
                "INSERT INTO agent_alerts(id, event_id, risk_level, threat_type, summary, created_at)
                 VALUES(?, 'b', 'normal', 'person', 's', datetime('now'))")
                .bind(format!("al{n}")).execute(&pool).await.unwrap();
        }

        let dir = std::path::PathBuf::from(".");
        let want = vec!["c".to_string(), "a".into(), "b".into(), "ghost".into()];
        let got = events_by_ids(&pool, &dir, &want).await;

        let ids: Vec<&str> = got.iter().map(|v| v["id"].as_str().unwrap()).collect();
        assert_eq!(ids, vec!["c", "a", "b"],
                   "requested order kept, unknown id skipped, no alert fan-out");
        // "a" is well outside the 24-hour default — asking by id must ignore it.
        assert!(ids.contains(&"a"), "an ids= lookup has no time window");
    }
}
