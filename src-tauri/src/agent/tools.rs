//! SINGLE SOURCE OF TRUTH for the Guardian agent's tools / capabilities.
//!
//! Every capability is declared **once** in `TOOLS`. From this one list we generate:
//!   * the chat system-prompt tool catalog (`prompt_catalog`),
//!   * the Telegram `/help` text (`help_text`),
//!   * the native tool-calling schemas (`tool_schemas`) — for the tool loop,
//!   * the tag↔tool bridge (`tag_for_call`) so a native `tool_call` maps to the
//!     existing `[TAG]` executor with zero duplicated execution logic.
//!
//! Mirrors on-device assistants Agies' tool set (video_search / video_analyze / video_send /
//! system_status / event_subscribe) and is the menu the agent reasons over.

use std::sync::Arc;

use serde_json::{json, Value};

use crate::AppState;

/// Short human timestamp for tool output: "2026-07-02 13:45" (UTC → local).
fn fmt_when(iso: &str) -> String {
    chrono::DateTime::parse_from_rfc3339(iso)
        .map(|d| d.with_timezone(&chrono::Local))
        .or_else(|_| chrono::NaiveDateTime::parse_from_str(iso, "%Y-%m-%d %H:%M:%S")
            .map(|n| n.and_utc().with_timezone(&chrono::Local)))
        .map(|d| d.format("%Y-%m-%d %H:%M").to_string())
        .unwrap_or_else(|_| iso.chars().take(16).collect())
}

pub struct ToolParam {
    pub name: &'static str,
    pub ty: &'static str,           // "string" | "integer"
    pub desc: &'static str,
    pub required: bool,
}

pub struct ToolSpec {
    /// Canonical snake_case name (the native tool-call name).
    pub name: &'static str,
    /// The `[TAG…]` template the existing executor understands. Placeholders are the
    /// param names in order, joined by ':'. Empty for query-only/slash-only tools.
    pub tag: &'static str,
    /// Telegram slash command (without '/'), if this tool is also a command.
    pub slash: Option<&'static str>,
    /// One-line description (used in the prompt catalog + /help + tool schema).
    pub desc: &'static str,
    pub params: &'static [ToolParam],
}

const P_CAM:  ToolParam = ToolParam { name: "camera", ty: "string", desc: "the camera's name, or its number as shown (Camera 1 is 1)", required: false };
const P_EVID: ToolParam = ToolParam { name: "event_id", ty: "string", desc: "the event id", required: true };
const P_MINS: ToolParam = ToolParam { name: "minutes", ty: "integer", desc: "link validity in minutes", required: false };

pub const TOOLS: &[ToolSpec] = &[
    // ── System / status ──────────────────────────────────────────────────────
    ToolSpec { name: "get_status", tag: "", slash: Some("status"),
        desc: "Current camera + system status (which cameras are live, recent activity).", params: &[] },

    // ── Events / search / analysis (Agies video_search / video_analyze) ───────
    ToolSpec { name: "recent_events", tag: "", slash: Some("events"),
        desc: "List the most RECENT motion events (newest first).",
        params: &[ToolParam { name: "count", ty: "integer", desc: "how many (default 5, max 20)", required: false }] },
    ToolSpec { name: "search_events", tag: "[SEARCH_EVENTS:{query}]", slash: Some("search"),
        desc: "Search past events by keyword (people, vehicles, plates, zones, summaries).",
        params: &[ToolParam { name: "query", ty: "string", desc: "what to look for", required: true }] },
    ToolSpec { name: "daily_summary", tag: "", slash: Some("summary"),
        desc: "A narrative digest of the last 24 hours, newest first.", params: &[] },
    ToolSpec { name: "get_event_analysis", tag: "", slash: None,
        desc: "Get the full AI analysis of ONE specific event by its id (use after recent_events/search_events to read the details of a particular event).",
        params: &[P_EVID] },
    // These two were documented in the system prompt but never registered, so
    // they were absent from `tool_schemas()` — a model with native function
    // calling could not request them AT ALL, and only ever got them because
    // `retrieve.rs` appends the tag from Rust.
    ToolSpec { name: "show_events", tag: "[SHOW_EVENTS:{filter}]", slash: None,
        desc: "Show matching events as tappable picture cards. Filter is free-form: 'today', 'last night', 'person today', 'critical this week'. Prefer this over describing events in words.",
        params: &[ToolParam { name: "filter", ty: "string", desc: "free-form filter, e.g. 'person last night'", required: false }] },
    ToolSpec { name: "day_chart", tag: "[DAY_CHART]", slash: None,
        desc: "Show a chart of today's activity by hour.", params: &[] },

    // ── Media share/send (Agies video_send) ──────────────────────────────────
    ToolSpec { name: "snapshot", tag: "[SNAPSHOT:{camera}]", slash: Some("snap"),
        desc: "Send a live photo from a camera.", params: &[P_CAM] },
    ToolSpec { name: "live_video", tag: "[LIVE_VIDEO:{camera}]", slash: None,
        desc: "Send a short live multi-frame album from a camera.", params: &[P_CAM] },
    ToolSpec { name: "send_clip", tag: "[SEND_CLIP:{event_id}]", slash: None,
        desc: "Send the recorded clip of an event (plays inline).", params: &[P_EVID] },
    ToolSpec { name: "share_clip", tag: "[SHARE_CLIP:{event_id}:{minutes}]", slash: None,
        desc: "Create a private shareable link to an event's clip.", params: &[P_EVID, P_MINS] },
    ToolSpec { name: "share_live", tag: "[SHARE_LIVE:{camera}:{minutes}]", slash: None,
        desc: "Create a private shareable link to a camera's live view.", params: &[P_CAM, P_MINS] },
    ToolSpec { name: "send_person", tag: "[SEND_PERSON:{name}]", slash: None,
        desc: "Send the enrolled photo of a known person by name.",
        params: &[ToolParam { name: "name", ty: "string", desc: "the enrolled person's name", required: true }] },

    // ── Proactive alert rules (Agies event_subscribe) ─────────────────────────
    ToolSpec { name: "subscribe_alert", tag: "[SUBSCRIBE_ALERT:{rule}]", slash: None,
        desc: "Add a proactive alert rule, e.g. 'person at night|type=person|hours=22-06'.",
        params: &[ToolParam { name: "rule", ty: "string", desc: "description|type=X|hours=22-06 (or 22:00-06:00)|min_risk=Y", required: true }] },
    ToolSpec { name: "unsubscribe_alert", tag: "[UNSUBSCRIBE_ALERT:{id}]", slash: None,
        desc: "Remove an alert rule by its short id.",
        params: &[ToolParam { name: "id", ty: "string", desc: "the rule's short id", required: true }] },
    ToolSpec { name: "list_rules", tag: "[LIST_RULES]", slash: Some("rules"),
        desc: "List all active proactive alert rules.", params: &[] },

    // ── Entities — the edge models' knowledge (faces / plates / sounds) exposed
    //    as agent skills, so the Guardian can ANSWER questions about who/what it
    //    knows instead of only describing raw events (the agentic-NVR pattern:
    //    edge AI produces structured entities, the LLM orchestrates them). ──────
    ToolSpec { name: "people_overview", tag: "", slash: Some("people"),
        desc: "Who the system knows: every enrolled person with role, last-seen time and recent sighting count, plus how many unidentified faces were seen recently.",
        params: &[] },
    ToolSpec { name: "person_activity", tag: "", slash: None,
        desc: "One enrolled person's recent activity: sighting count, days active, which cameras, usual hours, last seen. Use for questions like 'when was X here?'.",
        params: &[ToolParam { name: "name", ty: "string", desc: "the enrolled person's name", required: true }] },
    ToolSpec { name: "vehicle_activity", tag: "", slash: Some("vehicles"),
        desc: "Vehicles seen recently (licence plates): per-plate sighting count, friendly name if known, cameras, last seen. Optional plate filter.",
        params: &[ToolParam { name: "plate", ty: "string", desc: "filter to one plate (optional)", required: false }] },
    ToolSpec { name: "audio_activity", tag: "", slash: Some("sounds"),
        desc: "Sounds detected recently (barking, alarms, glass, speech…): per-sound counts, last heard, and the busiest hours.",
        params: &[] },

    // ── Memory (read/write the agent's durable knowledge) ─────────────────────
    // Searchable, not a dump: the prompt already carries the memory relevant to
    // the current question, so this is how the model asks for MORE on a specific
    // subject ("what do I know about the blue van?") without every fact ever
    // learned being resident in every prompt.
    ToolSpec { name: "recall_memory", tag: "", slash: None,
        desc: "Search what the agent has learned (family, routines, regular visitors, vehicles) for a subject.",
        params: &[ToolParam { name: "query", ty: "string", desc: "subject to recall (blank = most relevant recent)", required: false }] },
    ToolSpec { name: "remember", tag: "[REMEMBER:{category}:{content}]", slash: None,
        desc: "Save a durable fact under a category (family|routines|visitors|vehicles|environment|pets|rules).",
        params: &[
            ToolParam { name: "category", ty: "string", desc: "one of the memory categories", required: true },
            ToolParam { name: "content", ty: "string", desc: "the fact to remember", required: true },
        ] },
];

pub fn find(name: &str) -> Option<&'static ToolSpec> { TOOLS.iter().find(|t| t.name == name) }

/// The chat system-prompt tool catalog — the agent embeds `[TAG]`s that execute.
/// Generated so the prompt can never drift from the registry.
pub fn prompt_catalog() -> String {
    let mut out = String::from("Embed these tags ANYWHERE in your reply — they execute automatically:\n");
    for t in TOOLS {
        if t.tag.is_empty() { continue; } // query tools are listed below instead
        out.push_str(&format!("  {:<46} — {}\n", t.tag, t.desc));
    }
    // Query tools get a bracket form too, so models WITHOUT native tool calling
    // can still use every capability: the tag is REPLACED inline with live data
    // before the user sees the reply (see `execute_query_tags`).
    out.push_str("\nThese DATA tags are replaced inline with live information — embed one instead of guessing:\n");
    for t in TOOLS {
        if !t.tag.is_empty() { continue; }
        let arg = t.params.first().map(|p| format!(":{{{}}}", p.name)).unwrap_or_default();
        out.push_str(&format!("  [{}{}]{} — {}\n",
            t.name.to_uppercase(), arg,
            " ".repeat(30_usize.saturating_sub(t.name.len() + arg.len())), t.desc));
    }
    out
}

/// Human-readable "what I'm doing" line for the live activity feed (the
/// frontier-model pattern: show the work while it happens, then the answer).
pub fn activity_label(name: &str, args: &Value) -> String {
    let arg = |k: &str| args.get(k).and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
    match name {
        "get_status"         => "Checking camera & system status".into(),
        "recent_events"      => "Looking through recent events".into(),
        "search_events"      => {
            let q = arg("query");
            if q.is_empty() { "Searching footage events".into() } else { format!("Searching footage for “{q}”") }
        }
        "daily_summary"      => "Reviewing the last 24 hours".into(),
        "get_event_analysis" => "Reading the event's analysis".into(),
        "snapshot"           => "Taking a live snapshot".into(),
        "live_video"         => "Grabbing live frames".into(),
        "send_clip"          => "Fetching the event clip".into(),
        "share_clip"         => "Creating a private clip link".into(),
        "share_live"         => "Creating a live-view link".into(),
        "send_person"        => "Fetching the enrolled photo".into(),
        "subscribe_alert"    => "Adding the alert rule".into(),
        "unsubscribe_alert"  => "Removing the alert rule".into(),
        "list_rules"         => "Checking alert rules".into(),
        "people_overview"    => "Checking known people".into(),
        "person_activity"    => {
            let n = arg("name");
            if n.is_empty() { "Checking person activity".into() } else { format!("Checking {n}'s recent activity") }
        }
        "vehicle_activity"   => "Reviewing vehicle sightings".into(),
        "audio_activity"     => "Reviewing recent sounds".into(),
        "recall_memory"      => "Recalling learned memory".into(),
        "remember"           => "Saving that to memory".into(),
        _ => format!("Running {}", name.replace('_', " ")),
    }
}

/// Emit one live activity line to the app chat (best-effort; other surfaces
/// simply have no listener).
pub fn emit_activity(state: &Arc<AppState>, name: &str, args: &Value) {
    use tauri::Emitter;
    let _ = state.app_handle.emit("guardian:activity",
        serde_json::json!({ "text": activity_label(name, args) }));
}

/// One bracket tag the model emitted, resolved against the registry.
pub struct TagCall {
    pub tool: &'static ToolSpec,
    pub args: serde_json::Value,
    /// Byte range of the tag in the original text, so a caller can splice a
    /// result in place of it.
    pub span: std::ops::Range<usize>,
}

/// THE bracket-tag scanner. Every surface goes through this one.
///
/// It replaced three incompatible parsers — `execute_query_tags` here,
/// `chat_app`'s six-entry `specs` array, and `execute_telegram_commands`' hand-rolled
/// loops — which had drifted into two user-visible bugs:
///
/// * `[SEARCH_EVENTS]`, `[SUBSCRIBE_ALERT]`, `[UNSUBSCRIBE_ALERT]` and `[LIST_RULES]`
///   executed on Telegram but printed as literal bracket text in the app, even
///   though the system prompt instructs the model to emit them.
/// * `chat_app` took only the FIRST `:`-separated field as the payload, so a
///   36-character event UUID arrived truncated and every clip resolved to nothing.
///   Here the fields map positionally onto `ToolSpec.params`, so the id survives.
///
/// Unknown heads are left alone: prose containing `[note: low confidence]` must
/// come through untouched, and an unclosed `[` is not a tag.
pub fn parse_tags(text: &str) -> Vec<TagCall> {
    let bytes = text.as_bytes();
    let mut calls = Vec::new();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] != b'[' { i += 1; continue; }
        let Some(end_rel) = text[i..].find(']') else { break }; // unclosed — not a tag
        let end = i + end_rel;
        let inner = &text[i + 1..end];
        // `head` is the tool name; the rest are positional parameter values.
        let mut fields = inner.split(':');
        let head = fields.next().unwrap_or("").trim();

        let found = TOOLS.iter().find(|t| {
            t.name.eq_ignore_ascii_case(head)
                // …or the literal prefix of the tag template, so `[SEND_CLIP:…]`
                // resolves as readily as `[send_clip:…]`.
                || t.tag.strip_prefix('[')
                    .map(|s| s.split([':', ']']).next().unwrap_or(""))
                    .is_some_and(|n| n.eq_ignore_ascii_case(head))
        });

        match found {
            None => { i = end + 1; }
            Some(t) => {
                let mut args = serde_json::Map::new();
                for (p, raw) in t.params.iter().zip(fields) {
                    let v = raw.trim();
                    if v.is_empty() { continue; }
                    args.insert(p.name.to_string(), if p.ty == "integer" {
                        json!(v.parse::<i64>().unwrap_or(0))
                    } else {
                        json!(v)
                    });
                }
                calls.push(TagCall {
                    tool: t,
                    args: serde_json::Value::Object(args),
                    span: i..end + 1,
                });
                i = end + 1;
            }
        }
    }
    calls
}

/// Execute QUERY-tool bracket tags found in a reply and replace each with its
/// live result — `[AUDIO_ACTIVITY]`, `[daily_summary]`, `[PERSON_ACTIVITY:John]`…
/// This is what makes the text-tag fallback capable on providers without native
/// tool calling, and what stops raw placeholders leaking into the chat when a
/// model invents the bracket form on its own.
///
/// Action/media tags are left in place for the surface to render (the app turns
/// them into playable cards, Telegram uploads the media).
pub async fn execute_query_tags(state: &Arc<AppState>, text: &mut String) {
    // Right to left, so each splice leaves the earlier spans valid.
    let calls = parse_tags(text);
    for call in calls.into_iter().rev() {
        if !call.tool.tag.is_empty() { continue; } // action tags run downstream
        emit_activity(state, call.tool.name, &call.args); // live "doing X…" line
        // An unmatched tool used to be replaced with the EMPTY string: the tag
        // vanished and the prose around it read as though it had data.
        let result = execute(state, call.tool.name, &call.args).await
            .unwrap_or_else(|| "(no data available)".into());
        text.replace_range(call.span, result.trim());
    }
}

/// Telegram `/help` text, generated from the slash-bound tools.
pub fn help_text() -> String {
    let mut out = String::from("<b>Commands</b>\n");
    for t in TOOLS {
        if let Some(s) = t.slash {
            out.push_str(&format!("/{} — {}\n", s, t.desc));
        }
    }
    out.push_str("/menu — buttons for footage &amp; alert settings\n");
    out.push_str("/pause · /resume — mute &amp; unmute alerts\n\n");
    out.push_str("💬 Or just <b>ask in plain language</b> — e.g. “show the front door at 3pm”, \
                  “anyone last night?”, or “share a clip of the last event”.");
    out
}

/// Native tool-calling schemas (the `tools` array). Ready for the tool loop.
pub fn tool_schemas() -> Vec<Value> {
    TOOLS.iter().map(|t| {
        let mut props = serde_json::Map::new();
        let mut required: Vec<Value> = Vec::new();
        for p in t.params {
            props.insert(p.name.to_string(), json!({ "type": p.ty, "description": p.desc }));
            if p.required { required.push(json!(p.name)); }
        }
        json!({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.desc,
                "parameters": { "type": "object", "properties": Value::Object(props), "required": required }
            }
        })
    }).collect()
}

/// Execute a QUERY/INFO tool and return the text result for the model — these have
/// no side effects, always hit the DB LIVE (so the agent uses current data, never a
/// stale snapshot), and behave identically across every provider. Returns `None`
/// for action/media tools (snapshot/send_clip/share/…) which the caller bridges to
/// the proven `[TAG]` executor via `tag_for_call`.
/// The archive could not be read — say so, instead of reporting a confident "none".
///
/// Every query in `execute` used `.unwrap_or_default()` and then reported the
/// empty result as a finding, so a locked or corrupt database reached the model
/// as "No motion events in the last 7 days." That is the same class of falsehood
/// as an invented event, and it is exactly the bug `retrieve`'s `Stats.failed`
/// was added to fix — a fix that never reached this file, which is the path
/// every cloud provider's tool calls take.
fn db_unreadable(what: &str, e: sqlx::Error) -> Option<String> {
    tracing::error!(error = %e, what, "guardian tool: query failed");
    Some(format!(
        "ERROR: the {what} query failed and the archive could not be read. \
         Tell the user you could not check — do NOT report this as nothing found."))
}

/// Plates seen in the last 30 days, most frequent first, optionally narrowed to
/// those containing `filter` (already upper-cased). Filtering happens in SQL,
/// before the LIMIT, so a rarely seen plate is still found.
async fn vehicle_rows(db: &sqlx::SqlitePool, filter: &str)
    -> Result<Vec<(String, i64, String)>, sqlx::Error>
{
    sqlx::query_as(
        "SELECT recognized_plate, COUNT(*), MAX(started_at) FROM motion_events
         WHERE recognized_plate IS NOT NULL AND recognized_plate <> ''
           AND started_at > strftime('%Y-%m-%dT%H:%M:%S','now','-30 days')
           AND (?1 = '' OR instr(upper(recognized_plate), ?1) > 0)
         GROUP BY recognized_plate ORDER BY COUNT(*) DESC LIMIT 20")
        .bind(filter)
        .fetch_all(db).await
}

pub async fn execute(state: &Arc<AppState>, name: &str, args: &Value) -> Option<String> {
    let one_line = |raw: &str, max: usize| {
        let t = super::memory::extract_summary_text(raw);
        if t.is_empty() { "(not analysed yet)".to_string() } else { t.chars().take(max).collect() }
    };
    match name {
        "recent_events" => {
            let n = args.get("count").and_then(|v| v.as_i64()).unwrap_or(5).clamp(1, 20);
            let rows = sqlx::query_as(
                "SELECT id, started_at, peak_score, ai_summary FROM motion_events
                 WHERE started_at > strftime('%Y-%m-%dT%H:%M:%S','now','-7 days') ORDER BY started_at DESC LIMIT ?"
            ).bind(n).fetch_all(&state.db).await;
            let rows: Vec<(String, String, f32, Option<String>)> =
                match rows { Ok(r) => r, Err(e) => return db_unreadable("recent events", e) };
            if rows.is_empty() { return Some("No motion events in the last 7 days.".into()); }
            Some(rows.iter().map(|(id, started, score, sum)| {
                // FULL id: `[SEND_CLIP:{event_id}]` needs the whole uuid.
                format!("- {id} ({}, {:.0}%) — {}",
                    super::chat::rel_time(started), score * 100.0, one_line(sum.as_deref().unwrap_or(""), 120))
            }).collect::<Vec<_>>().join("\n"))
        }
        "search_events" => {
            let q = args.get("query").and_then(|v| v.as_str()).unwrap_or("").trim();
            if q.is_empty() { return Some("No search query given.".into()); }
            // THE searcher — six label columns, multi-keyword, plus the CLIP
            // semantic pass when a search model is installed. This used to be a
            // second, weaker implementation living here under the same name.
            let rows = match crate::nvr_recording::search_events_core(state, q, Some(8)).await {
                Ok(r) => r,
                Err(e) => {
                    tracing::error!(error = %e, "guardian tool: search failed");
                    return Some(format!(
                        "ERROR: the search for \"{q}\" failed and the archive could not be \
                         read. Tell the user you could not check — do NOT report this as \
                         nothing found."));
                }
            };
            if rows.is_empty() { return Some(format!("No events match \"{q}\".")); }
            Some(rows.iter().map(|e| {
                // FULL id — [SEND_CLIP:…] needs the whole uuid, and an 8-char
                // prefix here is why every clip the agent offered resolved to nothing.
                format!("- {} ({}) — {}", e.id, super::chat::rel_time(&e.started_at),
                    one_line(e.ai_summary.as_deref().unwrap_or(""), 120))
            }).collect::<Vec<_>>().join("\n"))
        }
        "get_event_analysis" => {
            let id = args.get("event_id").and_then(|v| v.as_str()).unwrap_or("");
            // `.ok()` used to fold a failed read into "No such event."
            let sum: Result<Option<Option<String>>, _> =
                sqlx::query_scalar("SELECT ai_summary FROM motion_events WHERE id=?")
                    .bind(id).fetch_optional(&state.db).await;
            match sum {
                Ok(Some(s)) => Some(one_line(s.as_deref().unwrap_or(""), 400)),
                Ok(None) => Some("No such event.".into()),
                Err(e) => db_unreadable("event", e),
            }
        }
        "daily_summary" => Some(super::dispatch::build_daily_digest(state).await),
        "recall_memory" => {
            let q = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
            // Larger budget than the prompt's slice — this is a deliberate ask.
            Some(super::memory::recall(&state.db, q, 6_000).await)
        }
        "list_rules" => Some(super::memory::fmt_alert_rules(&state.db).await),
        // Executed here rather than only in the Telegram tag handler, which is why
        // the app used to print the literal text "[SUBSCRIBE_ALERT:…]" back at the
        // user while the same tag worked fine on Telegram.
        "subscribe_alert" => {
            let raw = args.get("rule").and_then(|v| v.as_str()).unwrap_or("").trim();
            if raw.is_empty() { return Some("No rule given.".into()); }
            Some(match super::memory::parse_alert_rule(raw) {
                Ok(rule) => {
                    super::memory::save_alert_rule(&state.db, &rule).await;
                    super::memory::describe_alert_rule(&rule)
                }
                Err(why) => why,
            })
        }
        "unsubscribe_alert" => {
            let id = args.get("id").and_then(|v| v.as_str()).unwrap_or("").trim();
            Some(if super::memory::delete_alert_rule(&state.db, id).await {
                format!("Alert rule [{id}] removed.")
            } else {
                format!("No rule with id '{id}'.")
            })
        }
        // The description promises "which cameras are live" — deliver that instead
        // of a constant "System OK" and one number. `build_situation_ctx` is the
        // code that genuinely knows, and it is already local-day correct.
        "get_status" => Some(super::chat::build_situation_ctx(&state.db, None).await),
        // ── Entity skills — live DB reads over what the edge models have learned ──
        "people_overview" => {
            // A failed read used to become an empty list, reported as
            // "No people enrolled yet." — a confident falsehood.
            let people: Vec<(String, String, Option<String>)> = match sqlx::query_as(
                "SELECT name, role, last_seen_at FROM known_persons
                 ORDER BY last_seen_at IS NULL, last_seen_at DESC")
                .fetch_all(&state.db).await
            {
                Ok(p) => p,
                Err(e) => return db_unreadable("enrolled people", e),
            };
            let mut out = String::new();
            for (name, role, last) in &people {
                let n7: i64 = sqlx::query_scalar(
                    "SELECT COUNT(*) FROM face_sightings
                     WHERE person_name=? AND seen_at > strftime('%Y-%m-%dT%H:%M:%S','now','-7 days')")
                    .bind(name).fetch_one(&state.db).await.unwrap_or(0);
                out.push_str(&format!("• {name} ({role}) — {} sightings this week, last seen {}\n",
                    n7, last.as_deref().map(fmt_when).unwrap_or_else(|| "never".into())));
            }
            if people.is_empty() { out.push_str("No people enrolled yet.\n"); }
            let unknown: i64 = sqlx::query_scalar(
                "SELECT COUNT(*) FROM face_embeddings
                 WHERE person_id IS NULL AND seen_at > datetime('now','-14 days')")
                .fetch_one(&state.db).await.unwrap_or(0);
            if unknown > 0 {
                out.push_str(&format!("Plus {unknown} unidentified face sighting(s) in the last 14 days — the People → Train tab groups recurring strangers for naming."));
            }
            Some(out)
        }
        "person_activity" => {
            let name = args.get("name").and_then(|v| v.as_str()).unwrap_or("").trim();
            if name.is_empty() { return Some("Which person? Give me their enrolled name.".into()); }
            let rows = sqlx::query_as(
                "SELECT seen_at, camera_id FROM face_sightings
                 WHERE person_name=? COLLATE NOCASE AND seen_at > strftime('%Y-%m-%dT%H:%M:%S','now','-30 days')
                 ORDER BY seen_at DESC LIMIT 500")
                .bind(name).fetch_all(&state.db).await;
            let rows: Vec<(String, i64)> =
                match rows { Ok(r) => r, Err(e) => return db_unreadable("person sightings", e) };
            if rows.is_empty() {
                return Some(format!("No sightings of {name} in the last 30 days (check the exact enrolled name with people_overview)."));
            }
            let total = rows.len();
            let last = fmt_when(&rows[0].0);
            let days: std::collections::HashSet<&str> =
                rows.iter().map(|(t, _)| &t[..10.min(t.len())]).collect();
            let cams: std::collections::HashSet<i64> = rows.iter().map(|(_, c)| *c).collect();
            // Usual hours: histogram of LOCAL hour-of-day over the sightings.
            let mut hours = [0u32; 24];
            for (t, _) in &rows {
                if let Ok(utc) = chrono::DateTime::parse_from_rfc3339(t)
                    .map(|d| d.with_timezone(&chrono::Local))
                    .or_else(|_| chrono::NaiveDateTime::parse_from_str(t, "%Y-%m-%d %H:%M:%S")
                        .map(|n| n.and_utc().with_timezone(&chrono::Local)))
                { hours[chrono::Timelike::hour(&utc) as usize] += 1; }
            }
            let peak = hours.iter().enumerate().max_by_key(|(_, c)| **c).map(|(h, _)| h).unwrap_or(0);
            let names = super::retrieve::camera_names(&state.db).await;
            let mut cam_list: Vec<String> = cams.iter().map(|c| super::retrieve::cam_label(&names, *c)).collect();
            cam_list.sort();
            Some(format!(
                "{name}: {total} sighting(s) across {} day(s) in the last 30 days. Last seen {last}. Cameras: {}. Most often around {:02}:00–{:02}:00 local.",
                days.len(), cam_list.join(", "), peak, (peak + 1) % 24))
        }
        "vehicle_activity" => {
            let filter = args.get("plate").and_then(|v| v.as_str()).unwrap_or("").trim().to_uppercase();
            // The plate filter is part of the query. It used to run AFTER
            // `LIMIT 20`, so asking about any plate outside the 20 most frequent
            // came back as "no vehicles".
            let rows = vehicle_rows(&state.db, &filter).await;
            let rows: Vec<(String, i64, String)> =
                match rows { Ok(r) => r, Err(e) => return db_unreadable("licence plate", e) };
            if rows.is_empty() {
                return Some(if filter.is_empty() {
                    "No licence plates recognised in the last 30 days (the ALPR skill reads plates when vehicles are close enough).".into()
                } else {
                    format!("No plate matching {filter} in the last 30 days.")
                });
            }
            let known = state.settings.read().await.known_plates.clone();
            let name_of = |plate: &str| crate::alpr::match_known_plate(plate, &known);
            let mut out = String::from("Vehicles in the last 30 days:\n");
            for (plate, n, last) in rows {
                let label = name_of(&plate).map(|n| format!(" ({n})")).unwrap_or_default();
                out.push_str(&format!("• {plate}{label} — {n}×, last seen {}\n", fmt_when(&last)));
            }
            Some(out)
        }
        "audio_activity" => {
            let rows = sqlx::query_as(
                "SELECT dominant_label, COUNT(*), MAX(started_at) FROM motion_events
                 WHERE event_category='audio' AND started_at > strftime('%Y-%m-%dT%H:%M:%S','now','-7 days')
                 GROUP BY dominant_label ORDER BY COUNT(*) DESC LIMIT 15")
                .fetch_all(&state.db).await;
            let rows: Vec<(Option<String>, i64, String)> =
                match rows { Ok(r) => r, Err(e) => return db_unreadable("audio", e) };
            if rows.is_empty() {
                return Some("No sounds detected in the last 7 days (audio detection listens for alarms, glass, barking, speech…).".into());
            }
            let mut out = String::from("Sounds in the last 7 days:\n");
            for (sound, n, last) in rows {
                out.push_str(&format!("• {} — {n}×, last heard {}\n",
                    sound.unwrap_or_else(|| "sound".into()), fmt_when(&last)));
            }
            Some(out)
        }
        _ => None, // action/media tools → handled via the [TAG] bridge
    }
}

/// Bridge a native tool call `{name, args}` to the existing `[TAG]` the executor
/// understands — so native tool-calling reuses the proven execution path with no
/// duplicated logic. Returns None for query-only tools (handled separately) or
/// unknown names.
pub fn tag_for_call(name: &str, args: &Value) -> Option<String> {
    let t = find(name)?;
    if t.tag.is_empty() { return None; }
    let mut tag = t.tag.to_string();
    for p in t.params {
        let v = args.get(p.name)
            .map(|x| match x { Value::String(s) => s.clone(), other => other.to_string().trim_matches('"').to_string() })
            .unwrap_or_default();
        tag = tag.replace(&format!("{{{}}}", p.name), &v);
    }
    // Clean up empty optional placeholders → drop a trailing ":" segment.
    while tag.contains(":]") { tag = tag.replace(":]", "]"); }
    while tag.contains("::") { tag = tag.replace("::", ":"); }
    Some(tag)
}

#[cfg(test)]
mod tag_tests {
    use super::*;

    /// A plate outside the 20 most frequent used to come back as "no vehicles":
    /// the filter ran after `LIMIT 20`.
    #[tokio::test]
    async fn a_rare_plate_is_found_past_the_top_twenty() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        crate::db::init_db(&pool).await.unwrap();
        for i in 0..25 {
            // Plate i is seen (25 - i) times, so "RARE24" is seen once and ranks last.
            for n in 0..(25 - i) {
                sqlx::query("INSERT INTO motion_events(id, started_at, recognized_plate)
                             VALUES(?, strftime('%Y-%m-%dT%H:%M:%f+00:00','now','-1 hours'), ?)")
                    .bind(format!("e{i}-{n}")).bind(format!("RARE{i}"))
                    .execute(&pool).await.unwrap();
            }
        }
        let all = vehicle_rows(&pool, "").await.unwrap();
        assert_eq!(all.len(), 20, "unfiltered is still capped at 20");
        assert!(!all.iter().any(|r| r.0 == "RARE24"));
        let one = vehicle_rows(&pool, "RARE24").await.unwrap();
        assert_eq!(one.len(), 1, "the rare plate is found: {one:?}");
        assert_eq!(one[0].1, 1);
    }

    /// A database that can't be read must say so, never "nothing found".
    #[tokio::test]
    async fn an_unreadable_archive_is_reported_not_read_as_empty() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        // Deliberately not initialised: motion_events doesn't exist.
        let e = vehicle_rows(&pool, "").await.unwrap_err();
        let msg = db_unreadable("licence plate", e).unwrap();
        assert!(msg.starts_with("ERROR"), "{msg}");
    }

    fn names(text: &str) -> Vec<&'static str> {
        parse_tags(text).into_iter().map(|c| c.tool.name).collect()
    }

    /// THE regression. `chat_app` used to take only the first ':'-separated field
    /// as the payload, so a 36-char event uuid arrived as "a1b2c3d4" and every
    /// clip the agent offered to send resolved to nothing.
    #[test]
    fn a_full_uuid_survives_parsing() {
        const ID: &str = "3f2504e0-4f89-11d3-9a0c-0305e82c3301";
        let calls = parse_tags(&format!("Here it is. [SEND_CLIP:{ID}]"));
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].args["event_id"], ID);
    }

    /// Both spellings must resolve: the model is shown `[SEND_CLIP:…]` in the
    /// prompt but the registry name is `send_clip`, and it uses both.
    #[test]
    fn tag_and_tool_name_both_resolve() {
        assert_eq!(names("[SEND_CLIP:abc]"), vec!["send_clip"]);
        assert_eq!(names("[send_clip:abc]"), vec!["send_clip"]);
        assert_eq!(names("[Audio_Activity]"), vec!["audio_activity"]);
    }

    /// Trailing fields map onto the remaining params — they used to be discarded.
    #[test]
    fn extra_fields_map_positionally() {
        let calls = parse_tags("[SHARE_CLIP:ev-1:15]");
        assert_eq!(calls[0].args["event_id"], "ev-1");
        assert_eq!(calls[0].args["minutes"], 15, "integer params are coerced");
    }

    /// Prose is not a tag. A model writing brackets in a sentence, or opening one
    /// it never closes, must not have its text mangled.
    #[test]
    fn prose_and_unclosed_brackets_are_left_alone() {
        assert!(names("I saw a person [note: low confidence] at 14:02.").is_empty());
        assert!(names("[SEND_CLIP:no-closing-bracket").is_empty());
        assert!(names("nothing bracketed here").is_empty());
    }

    /// Spans must be in document order and non-overlapping, because callers splice
    /// results back in by walking them in reverse.
    #[test]
    fn spans_are_ordered_and_usable_for_splicing() {
        let text = "a [GET_STATUS] b [PEOPLE_OVERVIEW] c";
        let calls = parse_tags(text);
        assert_eq!(calls.len(), 2);
        assert!(calls[0].span.end <= calls[1].span.start);
        assert_eq!(&text[calls[0].span.clone()], "[GET_STATUS]");

        let mut out = text.to_string();
        for c in calls.into_iter().rev() { out.replace_range(c.span, "X"); }
        assert_eq!(out, "a X b X c");
    }
}
