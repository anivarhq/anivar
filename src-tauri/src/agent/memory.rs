//! Agent memory subsystem.
//!
//! Owns all reads/writes against the `agent_memory` table:
//!   * Flat key/value memory (`read_memory` / `write_memory` / `append_to_memory`).
//!   * ACE-loop scored learned memory (`reinforce_memory`, `decay_learned_memories`).
//!   * OpenClaw-style narrative markdown files (`read_memory_file`, `write_memory_file`).
//!   * assistant-parity structured memory categories.
//!   * Agies-style `event_subscribe` proactive alert rules.
//!   * Analysis-output helpers (`extract_summary_text`, `sanitize_analysis_output`).
//!
//! Two helpers near the bottom (`consolidate_memory_key`, `apply_memory_updates`)
//! delegate to Ollama via [`super::apply_auth`] — they bridge the memory and LLM
//! subsystems and will be moved alongside `apply_auth` when the Ollama section
//! becomes its own submodule.

use chrono::Utc;
use sqlx::SqlitePool;
use uuid::Uuid;

use super::types::*;

// ─── Memory helpers ───────────────────────────────────────────────────────────

pub async fn read_memory(db: &SqlitePool, key: &str) -> Option<String> {
    sqlx::query_as::<_, (String,)>("SELECT value FROM agent_memory WHERE key=?")
        .bind(key)
        .fetch_optional(db)
        .await
        .ok()
        .flatten()
        .map(|(v,)| v)
}

pub async fn write_memory(db: &SqlitePool, key: &str, value: &str) {
    let now = Utc::now().to_rfc3339();
    sqlx::query(
        "INSERT INTO agent_memory(key,value,updated_at) VALUES(?,?,?)
         ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
    )
    .bind(key)
    .bind(value)
    .bind(&now)
    .execute(db)
    .await
    .ok();
}

/// Append a timestamped entry to an existing memory key (creates if absent).
/// Each entry is separated by a newline so the log grows chronologically.
pub async fn append_to_memory(db: &SqlitePool, key: &str, new_entry: &str) {
    append_memory(db, key, new_entry).await;
}

pub(super) async fn append_memory(db: &SqlitePool, key: &str, new_entry: &str) {
    let now = Utc::now().format("%Y-%m-%d %H:%M").to_string();
    let stamped = format!("[{now}] {new_entry}");
    let existing = read_memory(db, key).await.unwrap_or_default();
    let merged = if existing.is_empty() {
        stamped
    } else {
        // Keep last 40 lines to prevent unbounded growth
        let mut lines: Vec<&str> = existing.lines().collect();
        lines.push(Box::leak(stamped.into_boxed_str()));
        let start = lines.len().saturating_sub(40);
        lines[start..].join("\n")
    };
    write_memory(db, key, &merged).await;
}

// ─── on-device assistants ACE-loop memory: Analyze → Curate → Execute ────────────────────
//
// After every event analysis, the agent automatically extracts patterns and
// updates scored memory entries. Memory types:
//   "manual"   — written by user (profile, rules)
//   "learned"  — auto-extracted from events (patterns, schedules, anomalies)
//   "person"   — person re-ID schedule ("person_X_tue_08" = seen 4 times Tue 8am)
//
// Scoring: confirmations increment each time a pattern repeats.
// Decay: read by `get_relevant_memories` which filters low-score learned memories.

/// Reinforce an auto-learned memory entry (increment confirmations + score).
/// Creates the entry if it doesn't exist.
pub async fn reinforce_memory(db: &SqlitePool, key: &str, value: &str, memory_type: &str) {
    let now = Utc::now().to_rfc3339();
    sqlx::query(
        "INSERT INTO agent_memory(key,value,updated_at,score,confirmations,created_at,memory_type)
         VALUES(?,?,?,1.0,1,?,?)
         ON CONFLICT(key) DO UPDATE SET
           value      = excluded.value,
           updated_at = excluded.updated_at,
           confirmations = confirmations + 1,
           score = MIN(10.0, score + 0.5),
           memory_type = excluded.memory_type"
    )
    .bind(key).bind(value).bind(&now).bind(&now).bind(memory_type)
    .execute(db).await.ok();
}

/// Decay all learned memories slightly (called periodically).
/// Memories with score < 0.3 and confirmations < 2 are removed (ephemeral noise).
pub async fn decay_learned_memories(db: &SqlitePool) {
    // Reduce score of unused memories
    sqlx::query(
        "UPDATE agent_memory SET score = score * 0.9
         WHERE memory_type = 'learned'
         AND updated_at < strftime('%Y-%m-%dT%H:%M:%S','now', '-3 days')"
    ).execute(db).await.ok();
    // Remove memories that have decayed to insignificance
    sqlx::query(
        "DELETE FROM agent_memory
         WHERE memory_type = 'learned' AND score < 0.3 AND confirmations < 3"
    ).execute(db).await.ok();
}

/// Get all relevant memories for an event context.
/// Returns a formatted string: high-scoring patterns + person schedules + manual rules.
pub async fn get_relevant_memories_for_event(db: &SqlitePool, cam_id: u8) -> String {
    let rows: Vec<(String, String, f64, i64)> = sqlx::query_as(
        "SELECT key, value, COALESCE(score, 1.0), COALESCE(confirmations, 1)
         FROM agent_memory
         WHERE (memory_type = 'learned' AND score >= 1.0)
            OR memory_type = 'manual'
         ORDER BY score DESC, confirmations DESC
         LIMIT 30"
    ).fetch_all(db).await.unwrap_or_default();

    if rows.is_empty() { return "No patterns learned yet.".to_string(); }

    let cam_prefix = format!("cam{}", cam_id);
    let mut parts: Vec<String> = Vec::new();
    for (key, value, score, confirms) in rows {
        if key.starts_with("camera_profile") || key.starts_with("threat_rules")
            || key.starts_with("known_false") { continue; } // handled separately
        if key.starts_with("pattern_") && !key.contains(&cam_prefix) { continue; }
        if confirms >= 3 {
            parts.push(format!("• {} (confirmed {confirms}×, score={score:.1}): {value}", key.replace('_', " ")));
        }
    }
    if parts.is_empty() { "No patterns confirmed yet (need 3+ observations).".to_string() }
    else { parts.join("\n") }
}

// ─── OpenClaw-style narrative memory files ────────────────────────────────────
//
// Memory is stored as structured markdown in a small set of "files" (SQLite keys):
//   mem/people   — one ## section per person with timestamped bullet observations
//   mem/patterns — temporal + behavioural patterns observed at this site
//   mem/events   — chronological log of notable security events
//   mem/site     — site profile and static context
//
// The agent reads all files on every reasoning request (Brain),
// writes to them during heartbeat (Heartbeat) and chat (REMEMBER tags).

// ─── assistant-parity: Structured memory categories ──────────────────────────────
//
// Memory is organised into named files:
//   family     — family members, their descriptions, vehicles, schedules
//   routines   — daily patterns and expected events
//   visitors   — regular visitors and service personnel
//   vehicles   — known vehicles with identifying details
//   environment — property layout, camera coverage areas, environment description
//   pets       — animals on the property
//
// Each is a plain-text markdown file (stored as a single agent_memory key).
// The chat agent reads all files on every reasoning request.

pub async fn read_memory_file(db: &SqlitePool, category: &str) -> String {
    read_memory(db, &format!("mem/{category}")).await.unwrap_or_default()
}

pub async fn write_memory_file(db: &SqlitePool, category: &str, content: &str) {
    write_memory(db, &format!("mem/{category}"), content).await;
}

pub async fn append_memory_file(db: &SqlitePool, category: &str, entry: &str) {
    let existing = read_memory_file(db, category).await;
    let updated = if existing.trim().is_empty() {
        entry.to_string()
    } else {
        format!("{existing}\n{entry}")
    };
    write_memory_file(db, category, &updated).await;
}

// ─── Agies-style event_subscribe: proactive alert rules ─────────────────────

/// Parse a `[SUBSCRIBE_ALERT:…]` payload into a rule.
///
/// Syntax: `description|type=person|hours=22-06|min_risk=suspicious` — everything
/// after the first `|` is optional `key=value`.
///
/// Lives here, not in the Telegram executor where it grew up, because BOTH
/// surfaces need it: subscribing worked on Telegram and printed as literal
/// `[SUBSCRIBE_ALERT:…]` text in the app.
/// An hour of the day from "22", "6", "22:00" or "06:30" (whole hours: rule
/// windows are hour-granular, and the minutes are dropped).
fn parse_hour(v: &str) -> Option<u32> {
    let (h, m) = v.trim().split_once(':').unwrap_or((v.trim(), "0"));
    let (h, m): (u32, u32) = (h.trim().parse().ok()?, m.trim().parse().ok()?);
    (h <= 23 && m <= 59).then_some(h)
}

/// Is `hour` inside the window? Wraps midnight when start > end (22-06).
pub(super) fn hour_in_window(start: u32, end: u32, hour: u32) -> bool {
    if start <= end { hour >= start && hour < end } else { hour >= start || hour < end }
}

pub(super) fn parse_alert_rule(inner: &str) -> Result<AlertRule, String> {
    let parts: Vec<&str> = inner.splitn(2, '|').collect();
    let mut rule = AlertRule {
        id: Uuid::new_v4().to_string()[..8].to_string(),
        description: parts[0].trim().to_string(),
        threat_type: None,
        hours_start: None,
        hours_end: None,
        min_risk: None,
        force_alert: true,
        created_at: Utc::now().to_rfc3339(),
    };
    for kv in parts.get(1).map(|s| s.split('|')).into_iter().flatten() {
        let kv = kv.trim();
        if let Some(v) = kv.strip_prefix("type=") {
            rule.threat_type = Some(v.trim().to_lowercase());
        } else if let Some(v) = kv.strip_prefix("hours=") {
            // "22:00-06:00" used to fail to parse and leave the rule with NO
            // window, which the matcher reads as "always". A window that can't be
            // read is refused, never widened to the whole day.
            let window = v.split_once('-').and_then(|(a, b)| Some((parse_hour(a)?, parse_hour(b)?)));
            let Some((a, b)) = window else {
                return Err(format!("I couldn't read the hours “{}”. Use a range like 22-06 or 22:00-06:00.", v.trim()));
            };
            rule.hours_start = Some(a);
            rule.hours_end   = Some(b);
        } else if let Some(v) = kv.strip_prefix("min_risk=") {
            rule.min_risk = Some(v.trim().to_lowercase());
        }
    }
    Ok(rule)
}

/// One-line human confirmation of a saved rule, for whichever surface asked.
pub(super) fn describe_alert_rule(r: &AlertRule) -> String {
    let mut bits = Vec::new();
    if let Some(t) = &r.threat_type { bits.push(format!("type: {t}")); }
    if let (Some(a), Some(b)) = (r.hours_start, r.hours_end) {
        bits.push(format!("hours: {a:02}:00–{b:02}:00"));
    }
    if let Some(m) = &r.min_risk { bits.push(format!("min risk: {m}")); }
    let detail = if bits.is_empty() { String::new() } else { format!(" ({})", bits.join(", ")) };
    format!("Alert rule [{}] added: {}{detail}", r.id, r.description)
}

/// Add or replace an alert rule in memory (key = "alert_rule_{id}").
pub(super) async fn save_alert_rule(db: &SqlitePool, rule: &AlertRule) {
    let key = format!("alert_rule_{}", rule.id);
    let json = serde_json::to_string(rule).unwrap_or_default();
    write_memory(db, &key, &json).await;
}

/// Remove an alert rule by its short id (matches key suffix).
pub(super) async fn delete_alert_rule(db: &SqlitePool, id: &str) -> bool {
    let key = format!("alert_rule_{}", id);
    let exists = read_memory(db, &key).await.is_some();
    if exists {
        sqlx::query("DELETE FROM agent_memory WHERE key=?")
            .bind(&key).execute(db).await.ok();
    }
    exists
}

/// Load all active alert rules from memory.
pub(super) async fn load_alert_rules(db: &SqlitePool) -> Vec<AlertRule> {
    let rows: Vec<(String,)> = sqlx::query_as(
        "SELECT value FROM agent_memory WHERE key LIKE 'alert_rule_%'"
    ).fetch_all(db).await.unwrap_or_default();
    rows.into_iter()
        .filter_map(|(json,)| serde_json::from_str(&json).ok())
        .collect()
}

/// Does this saved rule want an alert for this event? `kind` matches either the
/// event's category (person, vehicle, …) or its threat type; `hour` is the
/// event's local hour.
pub(super) fn rule_matches(rule: &AlertRule, category: &str, threat_type: &str, risk: &str, hour: u32) -> bool {
    let risk_rank = |r: &str| match r { "critical" => 3, "suspicious" => 2, "monitor" => 1, _ => 0 };
    if let Some(kind) = &rule.threat_type {
        if kind != category && kind != threat_type { return false; }
    }
    if let (Some(start), Some(end)) = (rule.hours_start, rule.hours_end) {
        if !hour_in_window(start, end, hour) { return false; }
    }
    if let Some(min) = &rule.min_risk {
        if risk_rank(risk) < risk_rank(min) { return false; }
    }
    rule.force_alert
}

/// Does any rule the user subscribed to (`[SUBSCRIBE_ALERT]`) force an alert for
/// this event? Until now nothing called this: the assistant said "Alert rule
/// added" and the rule never fired.
pub(super) async fn any_subscribe_rule_matches(
    db: &SqlitePool, category: &str, threat_type: &str, risk: &str, hour: u32,
) -> bool {
    load_alert_rules(db).await.iter().any(|r| rule_matches(r, category, threat_type, risk, hour))
}

/// Format all alert rules for display in chat / Telegram.
pub(super) async fn fmt_alert_rules(db: &SqlitePool) -> String {
    let rules = load_alert_rules(db).await;
    if rules.is_empty() {
        return "No proactive alert rules set.".to_string();
    }
    rules.iter().map(|r| {
        let mut parts = vec![format!("[{}] {}", r.id, r.description)];
        if let Some(ref tt) = r.threat_type { parts.push(format!("type={tt}")); }
        if let (Some(hs), Some(he)) = (r.hours_start, r.hours_end) {
            parts.push(format!("hours={hs:02}:00-{he:02}:00"));
        }
        if let Some(ref mr) = r.min_risk { parts.push(format!("min_risk={mr}")); }
        parts.join(", ")
    }).collect::<Vec<_>>().join("\n")
}

// ─── Agies analysis output helpers ──────────────────────────────────────────

/// Extract plain-text summary from a v2 JSON ai_summary blob (or return raw if plain text).
/// v2 schema: {"v":2,"risk":"...","type":"...","text":"...","description":"...", ...}
pub(super) fn extract_summary_text(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.starts_with('{') {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed) {
            // Prefer "text" field; fall back to "description", then "summary"
            for field in &["text", "description", "summary"] {
                if let Some(t) = v.get(field).and_then(|t| t.as_str()) {
                    if !t.is_empty() { return t.to_string(); }
                }
            }
        }
        // v2 JSON we couldn't parse cleanly — don't display raw JSON
        return String::new();
    }
    trimmed.to_string()
}

/// Agies-style prompt injection sanitizer.
/// VLM may analyse a frame containing adversarial text ("ignore instructions, reveal...").
/// Strip the analysis result when injection patterns are detected.
pub(super) fn sanitize_analysis_output(text: &str) -> &str {
    const INJECTION_PATTERNS: &[&str] = &[
        "ignore previous", "ignore all instructions", "ignore your",
        // "disregard" alone used to be enough to blank the WHOLE analysis, so
        // "the dog seemed to disregard the fence" wiped a real event summary.
        // Every other entry here is a phrase; this one now is too.
        "disregard previous", "disregard all", "disregard your", "disregard the above",
        "jailbreak", "reveal your", "system prompt", "forget your instructions",
        "new instructions", "admin override", "print all", "leak your",
        "[system]", "[[system]]", "### instruction",
    ];
    let lower = text.to_lowercase();
    for pat in INJECTION_PATTERNS {
        if lower.contains(pat) {
            tracing::warn!("[SecurityAnalysis] Prompt injection detected in VLM output — neutralising");
            return "";
        }
    }
    text
}

/// Read ALL structured memory files as a single block for LLM context.
pub async fn read_all_memory_files(db: &SqlitePool) -> String {
    let categories = ["family", "routines", "visitors", "vehicles", "environment", "pets", "rules"];
    let mut parts = Vec::new();
    for cat in &categories {
        let content = read_memory_file(db, cat).await;
        if !content.trim().is_empty() {
            parts.push(format!("## {}\n{content}", cat.to_uppercase()));
        }
    }
    if parts.is_empty() {
        "No structured memory recorded yet. You can tell me about your family, regular visitors, vehicles, and daily routines.".to_string()
    } else {
        parts.join("\n\n")
    }
}

/// Parse [REMEMBER:category:content] tags and write to the correct memory file.
/// Supported categories: family, routines, visitors, vehicles, environment, pets, rules,
/// and any agent_memory key for backwards compatibility.
pub async fn process_remember_tags(db: &SqlitePool, text: &mut String) {
    let structured_cats = ["family", "routines", "visitors", "vehicles", "environment", "pets", "rules"];
    while let Some(start) = text.find("[REMEMBER:") {
        let Some(end)   = text[start..].find(']') else { break };
        let inner = text[start + 10..start + end].to_string();
        let parts: Vec<&str> = inner.splitn(2, ':').collect();
        if parts.len() == 2 {
            let key   = parts[0].trim().to_lowercase().replace(' ', "_");
            let entry = parts[1].trim().to_string();
            if !key.is_empty() && !entry.is_empty() {
                let now = chrono::Local::now().format("%Y-%m-%d %H:%M").to_string();
                if structured_cats.contains(&key.as_str()) {
                    // Write to structured memory file with timestamp
                    append_memory_file(db, &key, &format!("[{now}] {entry}")).await;
                } else {
                    // Legacy: flat key-value store
                    write_memory(db, &key, &entry).await;
                }
            }
        }
        *text = format!("{}{}", &text[..start], &text[start + end + 1..]);
    }
}

// ─── Recall: retrieve what's relevant, instead of dumping what exists ────────
//
// `read_core_memory` concatenates EVERY memory file, every `kd_*` fact and the
// legacy keys. That made the prompt grow monotonically with use: every fact the
// agent ever learned was re-sent on every turn, until llama.cpp aborted on a
// prompt it could not batch. Bounding the window only moved the cliff.
//
// `recall` fixes the cause. One query over `agent_memory` (everything, including
// the `mem/<category>` files, lives in that one table), split to individual
// facts, ranked against the question, and cut to a byte budget. Stored memory can
// now grow forever without the prompt growing at all.
//
// Deliberately NOT an FTS5 table or a vector index: the corpus is a few hundred
// short lines that we were already reading in full every turn, so ranking them in
// Rust costs nothing and adds no schema, no migration and no capability probe.
// Revisit if this ever exceeds a few thousand facts.

/// One retrievable fact: a single line of memory, tagged with where and when it
/// came from. `when` is kept because a security agent must be able to say how old
/// a fact is rather than presenting a month-old note as current.
struct Fact {
    source: String,
    text: String,
    when: Option<chrono::DateTime<chrono::FixedOffset>>,
    /// `agent_memory.score` × confirmations boost — how load-bearing this fact has proved.
    weight: f64,
}

/// Words worth matching on. Drops punctuation and the filler that matches
/// everything ("what", "the", "did"), so relevance reflects the actual subject.
fn keywords(text: &str) -> Vec<String> {
    const STOP: &[&str] = &[
        "the", "and", "was", "were", "did", "does", "what", "when", "where", "who",
        "how", "why", "you", "your", "for", "with", "any", "are", "can", "has",
        "have", "this", "that", "there", "from", "about", "tell", "show", "give",
        "please", "just", "get", "got", "all", "not", "but", "its",
    ];
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() >= 3 && !STOP.contains(w))
        .map(str::to_string)
        .collect()
}

/// Rank one fact against the question.
///
/// `(base + 2×relevance) × recency × weight`.
///
/// Relevance is weighted to OUT-RANGE recency on purpose. With the two treated
/// comparably, a fully-on-topic fact from three weeks ago lost to a fresh,
/// unrelated one — which is "most recent N" again, the exact behaviour retrieval
/// exists to replace. The non-zero base keeps a question that shares no keywords
/// at all ("hi") from returning nothing: it falls back to recent, well-confirmed
/// facts instead.
fn rank(fact: &Fact, query: &[String], now: i64) -> f64 {
    let hay = format!("{} {}", fact.source, fact.text).to_lowercase();
    let relevance = if query.is_empty() {
        0.0
    } else {
        query.iter().filter(|q| hay.contains(q.as_str())).count() as f64 / query.len() as f64
    };
    // Halves weekly (grok-build's decay), floored so an old but repeatedly
    // confirmed fact ("the gate code is 4821") never drops out entirely.
    let recency = match fact.when {
        Some(t) => {
            let days = (now - t.timestamp()).max(0) as f64 / 86_400.0;
            0.5f64.powf(days / 7.0).max(0.30)
        }
        None => 0.6, // undated: neither fresh nor stale
    };
    (0.20 + 2.0 * relevance) * recency * fact.weight
}

/// Split a stored value into facts. Memory files are line-per-entry markdown;
/// everything else is one fact. Lines already carry a `[YYYY-MM-DD HH:MM]` stamp
/// from `process_remember_tags`, which is parsed back out so recency is real.
fn explode(key: &str, value: &str, weight: f64, updated_at: &str) -> Vec<Fact> {
    let fallback = chrono::DateTime::parse_from_rfc3339(updated_at).ok();
    let source = match key.strip_prefix("mem/") {
        Some(cat) => cat.to_string(),
        None => key.trim_start_matches("kd_").replace('_', " "),
    };
    let multiline = key.starts_with("mem/");
    let parts: Vec<&str> = if multiline { value.lines().collect() } else { vec![value] };
    parts.iter()
        .map(|l| l.trim().trim_start_matches(['-', '*', '#']).trim())
        .filter(|l| !l.is_empty())
        .map(|line| {
            // "[2026-07-20 14:31] came home late" → date + text
            let (when, text) = match line.strip_prefix('[').and_then(|r| r.split_once(']')) {
                Some((stamp, rest)) => (
                    chrono::NaiveDateTime::parse_from_str(stamp.trim(), "%Y-%m-%d %H:%M")
                        .ok()
                        .and_then(|n| n.and_local_timezone(chrono::Local).single())
                        .map(|d| d.fixed_offset())
                        .or(fallback),
                    rest.trim().to_string(),
                ),
                None => (fallback, line.to_string()),
            };
            Fact { source: source.clone(), text, when, weight }
        })
        .collect()
}

/// The prompt's memory section: only the facts that bear on `question`, newest
/// and most-confirmed first, capped at `budget_chars`.
///
/// Rules (`camera_profile`, `threat_rules`, `known_false_positives`) are NOT
/// ranked — they are standing instructions and are always included by the caller.
pub async fn recall(db: &SqlitePool, question: &str, budget_chars: usize) -> String {
    let rows: Vec<(String, String, Option<f64>, Option<i64>, Option<String>)> = sqlx::query_as(
        "SELECT key, value, score, confirmations, updated_at FROM agent_memory
          WHERE key NOT LIKE 'alert\\_rule\\_%' ESCAPE '\\'
            AND key NOT IN ('camera_profile','threat_rules','known_false_positives')"
    ).fetch_all(db).await.unwrap_or_default();

    let mut facts: Vec<Fact> = Vec::new();
    for (key, value, score, confirmations, updated_at) in &rows {
        if value.trim().is_empty() { continue; }
        // A fact confirmed many times outranks a one-off observation.
        let weight = score.unwrap_or(1.0).clamp(0.1, 10.0)
            * (1.0 + 0.1 * confirmations.unwrap_or(0) as f64);
        facts.extend(explode(key, value, weight, updated_at.as_deref().unwrap_or("")));
    }

    // Older conversation turns are part of the corpus, so a turn that has fallen
    // out of the live history window is still RECOVERABLE by relevance — that
    // pairing is what makes trimming history safe.
    //
    // The USER's turns only. The agent's own replies are derived from this very
    // context; folding them back in makes a closed loop where the agent cites
    // itself as evidence, and one wrong answer becomes a "remembered fact" that
    // biases every later answer. What the user said is the durable signal.
    let older: Vec<(String, String)> = sqlx::query_as(
        "SELECT content, created_at FROM chat_log
          WHERE role = 'user'
          ORDER BY created_at DESC LIMIT 200 OFFSET 12"
    ).fetch_all(db).await.unwrap_or_default();
    for (content, at) in &older {
        if content.trim().is_empty() { continue; }
        facts.push(Fact {
            source: "the user said earlier".into(),
            text: content.chars().take(300).collect(),
            when: chrono::DateTime::parse_from_rfc3339(at).ok(),
            weight: 0.6, // below curated memory: chat is raw, memory is distilled
        });
    }

    let query = keywords(question);
    let now = Utc::now().timestamp();
    facts.sort_by(|a, b| rank(b, &query, now).total_cmp(&rank(a, &query, now)));

    let mut out = String::new();
    for f in &facts {
        let stamp = f.when
            .map(|t| t.with_timezone(&chrono::Local).format("%Y-%m-%d").to_string())
            .unwrap_or_else(|| "undated".into());
        let line = format!("- [{stamp}] ({}) {}\n", f.source, f.text);
        if out.len() + line.len() > budget_chars { break; }
        out.push_str(&line);
    }
    if out.is_empty() {
        "Nothing recorded yet. You can tell me about your family, regular visitors, \
         vehicles, and daily routines.".to_string()
    } else {
        out
    }
}

/// Read camera profile, threat rules, and false positives as a single context block.
///
/// This is the FULL dump — every memory file, every distilled fact. It is an
/// export/debug view: the chat prompt uses [`recall`] instead, because dumping
/// everything is what made the prompt grow without bound.
pub async fn read_core_memory(db: &SqlitePool) -> String {
    // Structured memory files (named categories)
    let structured = read_all_memory_files(db).await;
    // Legacy flat keys
    let profile = read_memory(db, "camera_profile").await.unwrap_or_default();
    let rules   = read_memory(db, "threat_rules").await.unwrap_or_default();
    let fp      = read_memory(db, "known_false_positives").await.unwrap_or_default();

    // Agies Knowledge Distillation facts — extracted from chat history, grouped by category
    let kd_rows: Vec<(String, String)> = sqlx::query_as(
        "SELECT key, value FROM agent_memory WHERE key LIKE 'kd_%' ORDER BY key"
    ).fetch_all(db).await.unwrap_or_default();

    let kd_ctx = if kd_rows.is_empty() {
        String::new()
    } else {
        let mut by_cat: std::collections::BTreeMap<String, Vec<String>> = Default::default();
        for (k, v) in &kd_rows {
            // key format: kd_{category}_{slug}
            let cat = k.trim_start_matches("kd_")
                .split('_').next().unwrap_or("misc").to_string();
            by_cat.entry(cat).or_default().push(v.clone());
        }
        let mut lines = vec!["## Household Knowledge (distilled from conversations)".to_string()];
        for (cat, facts) in &by_cat {
            let cat_label = match cat.as_str() {
                "home_profile"       => "Home Profile",
                "alert_preferences"  => "Alert Preferences",
                "security_patterns"  => "Security Patterns",
                "system_config"      => "System Config",
                other                => other,
            };
            lines.push(format!("### {cat_label}"));
            for f in facts { lines.push(format!("- {f}")); }
        }
        lines.join("\n")
    };

    let mut parts = vec![structured];
    if !kd_ctx.trim().is_empty()   { parts.push(kd_ctx); }
    if !profile.trim().is_empty()  { parts.push(format!("## Camera Environment\n{profile}")); }
    if !rules.trim().is_empty()    { parts.push(format!("## Alert Rules\n{rules}")); }
    if !fp.trim().is_empty()       { parts.push(format!("## Known False Positives\n{fp}")); }
    parts.join("\n\n")
}


#[cfg(test)]
mod recall_tests {
    use super::*;

    fn fact(source: &str, text: &str, age_days: i64, weight: f64) -> Fact {
        Fact {
            source: source.into(),
            text: text.into(),
            when: Some((Utc::now() - chrono::Duration::days(age_days)).fixed_offset()),
            weight,
        }
    }

    /// The whole point: a fact ABOUT the question outranks a fresher one that
    /// isn't, so retrieval beats "most recent N".
    #[test]
    fn relevance_beats_recency() {
        let now = Utc::now().timestamp();
        let q = keywords("has the blue van been back?");
        let relevant = fact("vehicles", "a blue van parks outside on Thursdays", 20, 1.0);
        let fresh    = fact("pets", "the cat sits on the porch at noon", 0, 1.0);
        assert!(rank(&relevant, &q, now) > rank(&fresh, &q, now));
    }

    /// With no keyword overlap, ranking must still work — falling back to recent
    /// and repeatedly-confirmed facts rather than returning nothing.
    #[test]
    fn no_keyword_match_falls_back_to_recency_and_weight() {
        let now = Utc::now().timestamp();
        let q = keywords("hello");
        let old_weak   = fact("routines", "bins go out on Monday", 60, 1.0);
        let new_strong = fact("routines", "the postman comes at 11", 1, 2.0);
        assert!(rank(&new_strong, &q, now) > rank(&old_weak, &q, now));
        assert!(rank(&old_weak, &q, now) > 0.0, "a stale fact must not rank at zero");
    }

    /// Filler words match everything, so they must not count as relevance.
    #[test]
    fn stopwords_are_not_keywords() {
        assert_eq!(keywords("what did you see"), vec!["see".to_string()]);
    }

    /// Memory files are line-per-entry and carry their own timestamps; each line
    /// must come back as its own dated fact, not one giant blob.
    #[test]
    fn memory_files_explode_into_dated_lines() {
        let facts = explode(
            "mem/visitors",
            "[2026-07-20 14:31] plumber came\n[2026-07-22 09:00] parcel delivered",
            1.0, "",
        );
        assert_eq!(facts.len(), 2);
        assert_eq!(facts[0].text, "plumber came");
        assert_eq!(facts[0].source, "visitors");
        assert_eq!(
            facts[1].when.unwrap().format("%Y-%m-%d").to_string(),
            "2026-07-22",
        );
    }
}

#[cfg(test)]
mod alert_rule_tests {
    use super::{hour_in_window, parse_alert_rule, rule_matches};
    use super::super::conditions::hhmm_in_window;

    #[test]
    fn both_hour_forms_parse_and_garbage_is_refused() {
        for raw in ["night|hours=22-06", "night|hours=22:00-06:00", "night|hours=22 - 6"] {
            let r = parse_alert_rule(raw).unwrap();
            assert_eq!((r.hours_start, r.hours_end), (Some(22), Some(6)), "{raw}");
        }
        for bad in ["night|hours=late", "night|hours=25-06", "night|hours=22"] {
            assert!(parse_alert_rule(bad).is_err(), "{bad} must not become an all-day rule");
        }
        assert_eq!(parse_alert_rule("any person|type=person").unwrap().hours_start, None, "no hours: all day, on purpose");
    }

    #[test]
    fn windows_wrap_midnight() {
        assert!(hour_in_window(22, 6, 23) && hour_in_window(22, 6, 0) && hour_in_window(22, 6, 5));
        assert!(!hour_in_window(22, 6, 6) && !hour_in_window(22, 6, 12) && !hour_in_window(22, 6, 21));
        assert!(hour_in_window(9, 17, 9) && !hour_in_window(9, 17, 17));
        assert!(hhmm_in_window("22:00", "07:00", "23:30") && hhmm_in_window("22:00", "07:00", "06:59"));
        assert!(!hhmm_in_window("22:00", "07:00", "07:00") && !hhmm_in_window("22:00", "07:00", "12:00"));
        assert!(hhmm_in_window("13:00", "14:00", "13:15") && !hhmm_in_window("13:00", "14:00", "14:00"));
    }

    #[test]
    fn a_rule_fires_for_its_kind_hours_and_risk() {
        let r = parse_alert_rule("person at night|type=person|hours=22-06|min_risk=suspicious").unwrap();
        assert!(rule_matches(&r, "person", "loitering", "suspicious", 23));
        assert!(!rule_matches(&r, "person", "loitering", "suspicious", 14), "daytime");
        assert!(!rule_matches(&r, "vehicle", "normal", "critical", 23), "another kind");
        assert!(!rule_matches(&r, "person", "normal", "monitor", 23), "below its minimum risk");
        let by_threat = parse_alert_rule("loitering|type=loitering").unwrap();
        assert!(rule_matches(&by_threat, "person", "loitering", "monitor", 12), "type can name the threat too");
    }
}
