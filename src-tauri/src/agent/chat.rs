//! Agent chat — interactive Q&A + streamed token-by-token responses.
//!
//! * `ChatMessage` — the wire shape of one chat turn (`role`, `content`).
//! * `chat_with_agent` — THE brain, shared by the app and Telegram. Native tool
//!   loop where the endpoint supports one, a real two-hop tag loop where it
//!   doesn't, and `retrieve::answer` in front of both for data questions.
//!
//! Both call out to the LLM via [`super::llm::call_llm`]
//! and consume the standard memory + identity helpers.

use std::sync::Arc;

use chrono::Utc;
use serde::{Deserialize, Serialize};

use crate::AppState;
use super::memory::{
    read_memory, process_remember_tags, extract_summary_text, fmt_alert_rules,
};

/// How much of the prompt the memory section may occupy (~600 tokens).
///
/// The point of a budget is that stored memory can grow forever while the prompt
/// does not. Every other section is already windowed (10 events, 5 alerts, a
/// fixed tool catalog); memory and history were the two that were not.
const MEMORY_BUDGET_CHARS: usize = 2_500;

/// How much of the prompt verbatim conversation history may occupy (~1000 tokens).
/// Turns that don't fit are dropped, not truncated — and stay reachable through
/// [`super::memory::recall`], which searches the durable `chat_log`.
const HISTORY_BUDGET_CHARS: usize = 4_000;

/// Recent conversation, newest-first into [`HISTORY_BUDGET_CHARS`], as prompt text.
///
/// Dropping WHOLE turns beats truncating bytes: a missing turn is a known absence
/// the model can work around, a severed sentence is corruption it can't detect.
/// Anything dropped is still in `chat_log` and still reachable through
/// [`super::memory::recall`], which searches it.
///
/// This is shared by BOTH answer paths on purpose. The retrieve-then-answer path
/// used to run before history existed, so every question it handled — which is
/// most of them — was answered cold. "and yesterday?", "what about camera 2" and
/// "who was that" resolved against nothing, which is what made the agent feel
/// stupid regardless of which model was behind it.
pub(super) fn recent_turns(history: &[ChatMessage]) -> String {
    let mut kept: Vec<String> = Vec::new();
    let mut used = 0usize;
    let mut dropped = 0usize;
    for m in history.iter().rev() {
        let line = format!("[{}]: {}", m.role.to_uppercase(), m.content);
        if used + line.len() > HISTORY_BUDGET_CHARS && !kept.is_empty() {
            dropped = history.len() - kept.len();
            break;
        }
        used += line.len();
        kept.push(line);
    }
    kept.reverse();
    if dropped > 0 {
        kept.insert(0, format!(
            "({dropped} earlier turn(s) omitted for length — they are in my memory, \
             ask and I can recall them.)"));
    }
    kept.join("\n")
}

/// The bracket-tag half of the tool catalogue, for endpoints with no native tool
/// channel. Native-tool providers get function schemas instead and never see
/// this — see `tool_guidance` in the prompt builder.
const TAG_GUIDANCE: &str = r#"
## Showing & sharing the actual footage (do this, don't just describe it)
When the user wants to SEE or be SENT an event, attach the real media — never just text:
- [SEND_CLIP:event_uuid]  — sends/plays the recorded clip of that event.
- [SHARE_CLIP:event_uuid]  — gives a private shareable LINK to that event's clip (use this when they
  ask for a "link" or to "share" it).
- [SNAPSHOT] / [SNAPSHOT:N] — a live photo right now.
These are delivered on whatever channel the user is on (the app plays it inline; Telegram gets the
video/photo or a private share link). NEVER paste a raw http://localhost… URL — embed the tag and the
system delivers the media for you. Prefer sending the clip/snapshot over a long text description.

## Exploring Events
When the user asks to SEE events ("show me", "what events", "any alerts", "what happened", "show tonight's events", etc.) embed a SHOW_EVENTS tag in your reply — the app will fetch and display matching event cards directly in the chat:
  [SHOW_EVENTS:today]                  — all events today
  [SHOW_EVENTS:today high]             — high/critical events today
  [SHOW_EVENTS:last hour]              — events in the last hour
  [SHOW_EVENTS:yesterday person]       — person events from yesterday
  [SHOW_EVENTS:this week critical]     — critical events this week
  [SHOW_EVENTS:last night]             — overnight events
The tag is replaced by real event cards — user can click to view clips. Include it whenever the user wants to browse or review footage.
Also: [DAY_CHART] — replaced by a visual chart of today's activity by hour. Include it when the user asks for an overview, a summary of the day, or "what happened today".

## Writing to Memory
Whenever you learn something important — a family member, visitor, vehicle, daily routine, or rule — embed a REMEMBER tag with the appropriate category:
  [REMEMBER:family:Ranjith — arrives daily at 9am, blue backpack, drives a white Honda]
  [REMEMBER:visitors:Amazon delivery driver comes Tuesday afternoons]
  [REMEMBER:routines:Kids leave for school at 8:15am weekdays]
  [REMEMBER:vehicles:White Honda CRV — Ranjith's car, reg plate ABC123]
  [REMEMBER:environment:Camera faces the front door and driveway. Street visible on left.]
  [REMEMBER:rules:Alert me if any unknown person approaches after 10pm]
  [REMEMBER:pets:Black Labrador named Max, often in the garden]
Categories: family, visitors, routines, vehicles, environment, pets, rules.
Multiple REMEMBER tags per reply are fine. They are invisible to the user but written permanently to your memory files.
"#;

/// Tag syntax for the alert-rule tools, for the same non-native endpoints.
const ALERT_TAG_HINT: &str = "Use [SUBSCRIBE_ALERT:...] to add a new rule when the user asks to be notified of specific conditions.
Use [UNSUBSCRIBE_ALERT:id] to remove a rule. Use [LIST_RULES] to show all active rules.
Use [SEARCH_EVENTS:query] to find past events matching a keyword or description.
";

/// How a message should be answered.
///
/// Guardian is monitor-only by hard constraint — it watches, summarises and
/// shows, and the one thing it may write is the user's own alert rules. None of
/// these modes changes that: they differ in how HARD it looks, not in what it
/// may do. (This used to say "every variant is READ-ONLY", which was never true
/// of `subscribe_alert`/`unsubscribe_alert` and put three separate places in the
/// codebase at odds with each other.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    /// One retrieval pass, answer in seconds. The default and the common case.
    #[default]
    Ask,
    /// Sweep the archive: widen the window when a pass comes up short, and keep
    /// going while new evidence appears. Slower, and says what it looked at.
    Investigate,
    /// Summarise a period rather than answer a question.
    Brief,
}

impl Mode {
    /// Tool-loop hops this mode may spend. Ask keeps the old bound; Investigate
    /// gets room to follow a lead, still hard-capped so it cannot run forever.
    pub(super) fn max_hops(self) -> usize {
        match self { Mode::Investigate => 10, _ => 4 }
    }
}

/// Roughly what the fixed sections of a prompt cost before any conversation:
/// the persona, the tool catalog, the situation context and the memory section.
/// Measured against a typical built prompt rather than guessed — it moves only
/// when those sections are rewritten.
const FIXED_PROMPT_TOKENS: usize = 1_400;

/// How full the model's context is, and how much it holds.
///
/// Returned with every in-app reply so the composer can show a ring. Both
/// numbers are ESTIMATES (~4 chars/token) and the UI says so: exact counts would
/// need each provider's tokenizer, which is a lot of machinery to draw a circle.
/// What matters is that it moves in the right direction and warns before the
/// oldest turns start dropping.
///
/// History is already bounded by [`HISTORY_BUDGET_CHARS`], so this plateaus
/// rather than climbing forever — which is itself the honest picture: past that
/// point older turns ARE being dropped, and the ring sitting near its cap is
/// what tells the user so.
pub fn context_usage(
    settings: &crate::Settings,
    history: &[ChatMessage],
    message: &str,
    reply: &str,
) -> (usize, usize) {
    let convo_chars: usize = history.iter().map(|m| m.content.len()).sum::<usize>()
        + message.len() + reply.len();
    // Only what actually reaches the prompt counts: history above the budget is
    // dropped before the model ever sees it.
    let counted = convo_chars.min(HISTORY_BUDGET_CHARS);
    let used = FIXED_PROMPT_TOKENS + counted.div_ceil(4);
    let limit = super::llm::context_limit(settings);
    (used.min(limit), limit)
}

/// Human "time ago" for an RFC3339 timestamp, so the agent frames recent activity
/// as current ("just now", "5m ago") instead of echoing a raw UTC string.
///
/// Anything an hour or older also carries its **local** calendar date, because a
/// security agent must be able to say *which day* it is reporting on — "3 events"
/// is useless if the reader can't tell whether that was today or last Tuesday.
pub(super) fn rel_time(rfc3339: &str) -> String {
    let Ok(dt) = chrono::DateTime::parse_from_rfc3339(rfc3339) else { return rfc3339.to_string() };
    let secs = (Utc::now().timestamp() - dt.timestamp()).max(0);
    let stamp = || dt.with_timezone(&chrono::Local).format("%a %b %d %H:%M").to_string();
    if secs < 60 { "just now".into() }
    else if secs < 3600 { format!("{}m ago", secs / 60) }
    else if secs < 86_400 { format!("{}h ago, {}", secs / 3600, stamp()) }
    else { format!("{}d ago, {}", secs / 86_400, stamp()) }
}

/// The agent's clock: current **local** date/time plus the UTC offset, and an
/// explicit statement of what "today" means. Injected into every system prompt.
///
/// Local, not UTC. Event timestamps are stored as UTC but every "today"/"last
/// night" question is asked in the user's timezone — reporting a UTC day here is
/// the same class of bug as the timeline day-bounds one (after ~19:00 local, the
/// UTC date is already tomorrow).
pub(super) fn now_local_str() -> String {
    let now = chrono::Local::now();
    format!("{} (local time, UTC{}). \"Today\" means {}.",
        now.format("%A, %d %B %Y, %H:%M"),
        now.format("%:z"),
        now.format("%Y-%m-%d"))
}

/// Provider-agnostic native tool-calling loop (the in-house agent brain). Runs the
/// SAME `agent::tools` registry on any tool-capable backend: the model requests a
/// tool, we run it (query tools return live text; action/media tools bridge to the
/// proven `[TAG]` executor and are appended to the reply), feed results back, loop
/// (bounded). Returns `Ok(None)` for providers without a native tool path so the
/// caller uses the text-tag fallback; `Err` on a hard failure (also → fallback).
async fn agent_tool_loop(
    state: &Arc<AppState>,
    settings: &crate::Settings,
    system: &str,
    user: &str,
    mode: Mode,
) -> anyhow::Result<Option<String>> {
    use serde_json::json;
    // What this ENDPOINT can do, not what its provider string is called. The
    // hardcoded five-provider allowlist that used to live here denied Anthropic
    // and Gemini the tools they support, and handed a 400 on every turn to any
    // self-hosted OpenAI-compatible server that doesn't implement them.
    if super::llm::tool_mode(settings) != super::llm::ToolMode::Native {
        return Ok(None);
    }
    let tools = super::tools::tool_schemas();
    // Qwen3 soft switch — reasoning off even where API-level think:false is
    // ignored (qwen3-vl). Harmless text for non-Qwen thinking models.
    let system_txt = if super::llm::thinking_model(&settings.vision_model) {
        format!("{system}\n/no_think")
    } else { system.to_string() };
    let mut messages: Vec<serde_json::Value> = vec![
        json!({ "role": "system", "content": system_txt }),
        json!({ "role": "user",   "content": user }),
    ];
    let mut action_tags: Vec<String> = Vec::new();
    let mut final_content = String::new();
    let mut ran_a_tool = false;

    for _hop in 0..mode.max_hops() {
        let turn = match super::llm::call_llm_tools(settings, &messages, &tools).await {
            Ok(t) => t,
            // A hop failed AFTER actions were queued: deliver what we have
            // (the media/action tags) instead of losing the user's request.
            Err(e) if !action_tags.is_empty() => {
                tracing::warn!("tool loop hop failed after actions queued — delivering queued actions: {e}");
                break;
            }
            Err(e) => return Err(e),
        };
        final_content = turn.content.clone();
        if turn.tool_calls.is_empty() { break; }
        ran_a_tool = true;
        // Record the assistant turn (with its tool calls) so the model keeps context.
        messages.push(json!({
            "role": "assistant", "content": turn.content,
            "tool_calls": turn.tool_calls.iter().map(|c| json!({
                "id": c.id, "type": "function",
                "function": { "name": c.name,
                    "arguments": json!(c.args.to_string()) }
            })).collect::<Vec<_>>(),
        }));
        for c in &turn.tool_calls {
            // Live activity line — the user sees "Searching footage…" etc.
            // in the chat while the agent works (frontier-model pattern).
            super::tools::emit_activity(state, &c.name, &c.args);
            // Arguments that didn't parse as JSON used to become `{}` silently,
            // so the tool ran with defaults or reported a missing parameter the
            // model had actually supplied. Hand the raw text back instead.
            if let Some(raw) = c.args.get("__malformed").and_then(|v| v.as_str()) {
                messages.push(json!({ "role": "tool", "tool_call_id": c.id, "name": c.name,
                    "content": format!("Error: arguments were not valid JSON: {raw}") }));
                continue;
            }
            let result = match super::tools::execute(state, &c.name, &c.args).await {
                Some(text) => text, // query/info tool → live result fed back to the model
                None => match super::tools::find(&c.name) {
                    // Unknown tool — report it honestly so the model picks a real one
                    // instead of believing a phantom call succeeded.
                    None => format!("Error: no tool named '{}'. Use one of the provided tools.", c.name),
                    Some(spec) => {
                        // Action/media tool. Validate required params BEFORE bridging so
                        // a missing arg is fed back (the model retries with it) rather
                        // than silently executing an empty [TAG].
                        let missing: Vec<&str> = spec.params.iter()
                            .filter(|p| p.required && c.args.get(p.name)
                                .map(|v| v.as_str().map(|s| s.trim().is_empty()).unwrap_or(false))
                                .unwrap_or(true))
                            .map(|p| p.name).collect();
                        if !missing.is_empty() {
                            format!("Error: '{}' needs parameter(s): {}.", c.name, missing.join(", "))
                        } else if let Some(tag) = super::tools::tag_for_call(&c.name, &c.args) {
                            action_tags.push(tag); // bridge to the proven channel executor
                            "Done — queued for delivery on the user's channel.".to_string()
                        } else {
                            "Done.".to_string()
                        }
                    }
                },
            };
            messages.push(json!({ "role": "tool", "tool_call_id": c.id, "name": c.name, "content": result }));
        }
    }

    // Out of hops, still mid-investigation: ask for the answer instead of
    // throwing the evidence away.
    //
    // The loop exits here with `final_content` set to whatever text accompanied
    // the LAST tool request, which for the OpenAI family is the empty string. An
    // empty result then sent the caller into `tag_tool_loop` — a completely fresh
    // generation that never sees any of the tool results just gathered. The
    // harder the question, the more certain the answer got worse.
    if ran_a_tool && final_content.trim().is_empty() {
        tracing::debug!(hops = mode.max_hops(), "tool loop exhausted — asking for a final answer");
        messages.push(json!({ "role": "user", "content":
            "You have used your entire tool budget. Answer now, using only what the              tool results above already told you. Do not request any more tools. If              something could not be established, say so plainly." }));
        match super::llm::call_llm_tools(settings, &messages, &tools).await {
            Ok(t) => final_content = t.content,
            Err(e) => tracing::warn!("tool loop: final answer call failed: {e}"),
        }
    }

    let mut out = final_content;
    if !action_tags.is_empty() {
        if !out.trim().is_empty() { out.push('\n'); }
        out.push_str(&action_tags.join("\n")); // executor/strip rules handle these downstream
    }
    Ok(Some(out))
}

/// The tool loop for models with no native function-calling channel.
///
/// This used to be a single generation whose `[TAG]`s were executed AFTERWARDS
/// and spliced into prose the model had already written — so the model wrote
/// "there were several events last night" around a placeholder it never read.
/// That is templating, not agency, and it is where a lot of the confident wrong
/// answers came from.
///
/// Now: generate, run the data tags, feed the results back, generate again with
/// the data actually in front of the model. Two hops, because a model without a
/// tool channel is usually a small one and each hop is a full re-read of the
/// prompt — the value is in seeing the data at all, not in seeing it four times.
///
/// The on-device 350M model deliberately never gets here with tags to run: its
/// prompt forbids them and `retrieve.rs` resolves its queries in Rust.
async fn tag_tool_loop(
    state: &Arc<AppState>,
    settings: &crate::Settings,
    system: &str,
    camera_name: &str,
    situation_ctx: &str,
    user: &str,
) -> anyhow::Result<String> {
    // Providers WITHOUT a native tool loop (the on-device engine, by default)
    // get a short prompt instead of the full catalogue.
    //
    // The big prompt is ~13,000 characters of instructions, bracket-tag syntax
    // and a COCO class list. Given that and an open question, a small model
    // reads its own prompt back: asked "what are your capabilities" it replied
    // with the class list and the literal line "Guardian agent (you):". A model
    // can only recite what it is shown, so it is shown less.
    let sys = if super::llm::provider_can_classify_risk(settings) {
        system.to_string()
    } else {
        small_model_prompt(settings, camera_name, situation_ctx)
    };

    let first = call_llm(settings, &sys, user, None, false).await
        .map_err(|e| anyhow::anyhow!("Guardian AI error: {e}"))?;

    // Which DATA tags did it ask for? (Media/action tags are evidence for the
    // surface to render, not information the model needs read back to it.)
    let asked: Vec<(String, serde_json::Value)> = super::tools::parse_tags(&first)
        .into_iter()
        .filter(|c| c.tool.tag.is_empty())
        .map(|c| (c.tool.name.to_string(), c.args))
        .collect();
    if asked.is_empty() { return Ok(first); }

    let mut results = String::new();
    for (name, args) in &asked {
        super::tools::emit_activity(state, name, args);
        let out = super::tools::execute(state, name, args).await
            .unwrap_or_else(|| "(no data)".into());
        results.push_str(&format!("\n[{}] {}\n", name.to_uppercase(), out.trim()));
    }

    // Second pass: same question, with the data. Answer ONLY from it.
    let grounded_user = format!(
        "{user}\n\n\
         DATA YOU REQUESTED — answer only from this, and do not repeat it verbatim:\n{results}");
    match call_llm(settings, &sys, &grounded_user, None, false).await {
        Ok(second) if super::llm::usable(&second) => Ok(second),
        // The second pass adds nothing if it fails; the first reply plus its
        // spliced results is still better than an error.
        other => {
            if let Err(e) = &other { tracing::warn!("tag loop second pass failed: {e}"); }
            let mut spliced = first;
            super::tools::execute_query_tags(state, &mut spliced).await;
            Ok(spliced)
        }
    }
}
use super::llm::call_llm;
use super::types::agent_identity;

// ─── Streaming chat ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

// `stream_chat` was DELETED. It was a THIRD tag parser: its own five-line
// system prompt, its own hand-rolled [REMEMBER:] loop, and it never called
// `parse_tags` — so a reply from it could contain tag syntax no surface knew
// how to render. It had no callers. Token streaming comes from
// `retrieve::answer`, which emits the same `guardian:chat-token` event.

// ─── Interactive chat ─────────────────────────────────────────────────────────

/// Live situation snapshot (pure SQL, no LLM): what's happening right now and
/// today, per camera, plus what changed since the last conversation. This is
/// the persistent-awareness backbone — every answer starts from CURRENT
/// reality instead of whatever the model can guess from 10 raw event rows.
pub(super) async fn build_situation_ctx(db: &sqlx::SqlitePool, last_chat: Option<&str>) -> String {
    let mut out = String::new();

    // Since-last-conversation delta — continuity across sessions/surfaces.
    match last_chat {
        Some(ts) => {
            let events: i64 = sqlx::query_scalar(
                "SELECT COUNT(*) FROM motion_events WHERE started_at > ?")
                .bind(ts).fetch_one(db).await.unwrap_or(0);
            let persons: i64 = sqlx::query_scalar(
                "SELECT COUNT(*) FROM motion_events WHERE started_at > ? AND event_category='person'")
                .bind(ts).fetch_one(db).await.unwrap_or(0);
            let sounds: i64 = sqlx::query_scalar(
                "SELECT COUNT(*) FROM motion_events WHERE started_at > ? AND event_category='audio'")
                .bind(ts).fetch_one(db).await.unwrap_or(0);
            out.push_str(&format!(
                "Since the previous conversation ({}): {events} new events ({persons} person, {sounds} sound).\n",
                rel_time(ts)));
        }
        None => out.push_str("This is the first conversation on record.\n"),
    }

    // Per-camera activity, last 6 hours.
    let cams: Vec<(Option<i64>, i64, String)> = sqlx::query_as(
        "SELECT cam_id, COUNT(*), MAX(started_at) FROM motion_events
          WHERE started_at > datetime('now','-6 hours') GROUP BY cam_id"
    ).fetch_all(db).await.unwrap_or_default();
    if cams.is_empty() {
        out.push_str("Last 6h: no activity on any camera.\n");
    } else {
        for (cam, n, last) in cams {
            out.push_str(&format!("Camera {}: {n} events in the last 6h, latest {}.\n",
                cam.unwrap_or(0), rel_time(&last)));
        }
    }

    // Who/what TODAY — the user's local calendar day, the same definition
    // `conditions.rs` resolves the "today" filter to. This used to be a rolling
    // `-24 hours` window labelled "today", so at 09:00 the agent reported half of
    // yesterday as today. One definition, everywhere.
    let today = chrono::Local::now().format("%Y-%m-%d").to_string();
    const TODAY_SQL: &str = "date(started_at,'localtime') = date('now','localtime')";

    let people: Vec<(String,)> = sqlx::query_as(&format!(
        "SELECT DISTINCT sub_label FROM motion_events
          WHERE {TODAY_SQL} AND sub_label IS NOT NULL AND sub_label != '' LIMIT 8")
    ).fetch_all(db).await.unwrap_or_default();
    if !people.is_empty() {
        out.push_str(&format!("Recognized today ({today}): {}.\n",
            people.iter().map(|(p,)| p.as_str()).collect::<Vec<_>>().join(", ")));
    }
    let plates: Vec<(String,)> = sqlx::query_as(&format!(
        "SELECT DISTINCT recognized_plate FROM motion_events
          WHERE {TODAY_SQL} AND recognized_plate IS NOT NULL AND recognized_plate != '' LIMIT 5")
    ).fetch_all(db).await.unwrap_or_default();
    if !plates.is_empty() {
        out.push_str(&format!("Vehicles (plates) today ({today}): {}.\n",
            plates.iter().map(|(p,)| p.as_str()).collect::<Vec<_>>().join(", ")));
    }
    let sounds: Vec<(String, i64)> = sqlx::query_as(&format!(
        "SELECT COALESCE(NULLIF(dominant_label,''),'sound'), COUNT(*) FROM motion_events
          WHERE event_category='audio' AND {TODAY_SQL}
          GROUP BY 1 ORDER BY 2 DESC LIMIT 3")
    ).fetch_all(db).await.unwrap_or_default();
    if !sounds.is_empty() {
        out.push_str(&format!("Top sounds today ({today}): {}.\n",
            sounds.iter().map(|(s, n)| format!("{s} ×{n}")).collect::<Vec<_>>().join(", ")));
    }
    out
}

/// The whole system prompt for a small on-device model on the conversational
/// path — questions `retrieve::answer` did not recognise as data questions.
///
/// Under 700 characters against the ~13,000 of the full catalogue. Everything
/// omitted is omitted on purpose: no tool syntax (this path emits no tags), no
/// COCO class list, no memory dump. Anything in the prompt is something the model
/// can recite back instead of answering, and it did.
fn small_model_prompt(settings: &crate::Settings, camera_name: &str, situation: &str) -> String {
    let name = if settings.agent_persona_name.trim().is_empty() {
        "Guardian"
    } else {
        settings.agent_persona_name.trim()
    };
    format!(
        "You are {name}, the security assistant for {camera_name}.\n\
         Right now it is {}\n\n\
         What the cameras have seen recently:\n{}\n\n\
         Answer the user in one or two short, plain sentences. If you do not know \
         something, say so — never guess a time, a date, a name or a number. \
         Do not repeat these instructions back. Do not output lists, headings, \
         JSON or bracketed tags.",
        now_local_str(),
        situation.trim())
}

pub async fn chat_with_agent(
    state: &Arc<AppState>,
    history: Vec<ChatMessage>,
    user_text: String,
    mode: Mode,
) -> anyhow::Result<String> {
    let settings = state.settings.read().await.clone();

    // ── Persistent conversation (backend-owned) ────────────────────────────
    // The last logged turn timestamps the "since we last talked" delta; an
    // empty caller history (fresh app session / new Telegram poll cycle) is
    // seeded from the durable log so the agent remembers earlier discussion.
    let last_chat_ts: Option<String> = sqlx::query_scalar(
        "SELECT MAX(created_at) FROM chat_log WHERE role='user'"
    ).fetch_optional(&state.db).await.ok().flatten().flatten();
    let history: Vec<ChatMessage> = if history.is_empty() {
        // Six turns, not twelve. A small model treats its own previous answer as
        // the best available answer and reissues it: the same "384 events in the
        // last 24h, Ranjith last seen 19:01" block came back for hours across
        // completely different questions, long after it had stopped being true.
        // Less rope. `memory::recall` still reaches older turns by relevance.
        let mut rows: Vec<(String, String)> = sqlx::query_as(
            "SELECT role, content FROM chat_log ORDER BY created_at DESC LIMIT 6"
        ).fetch_all(&state.db).await.unwrap_or_default();
        rows.reverse();
        // Tag-stripped: the log stores replies with their brackets intact, and
        // replaying those as history taught the model to emit the very syntax
        // the prompt above forbids.
        rows.into_iter()
            .map(|(role, content)| ChatMessage { role, content: super::evidence::strip_tags(&content) })
            .collect()
    } else { history };
    // Chat is a TEXT task — it never needed a vision model, and asking for one broke
    // on-device entirely: that engine is compiled in and has no model NAME, so
    // `vision_model` is legitimately empty and this bailed before any LLM call.
    // `agent_configured` asks the real question (has this provider a usable engine?)
    // and already answers it correctly for every provider including on-device.
    if !super::llm::agent_configured(&settings) {
        anyhow::bail!("No AI engine configured — open Guardian → Arsenal and pick a provider.");
    }

    // ── Retrieve-then-answer ────────────────────────────────────────────────
    // A data question ("what happened today", "how many cars this week") is
    // answered from the DATABASE, with the model used only to phrase the result.
    //
    // This runs for every provider, not just the on-device one. A frontier model
    // given the real rows also answers better than one given a tool catalogue and
    // asked to guess which to call — and here the answer survives the model
    // failing entirely, because `retrieve` already computed it.
    //
    // `None` means the question wasn't a data question at all ("who are you?"),
    // and the full conversational path below handles it.
    // Built HERE, above the retrieve path, so both paths get the same context.
    let history_ctx = recent_turns(&history);

    if let Some(answer) = super::retrieve::answer(state, &settings, &user_text, &history_ctx, mode).await {
        // This path used to return here without running `process_remember_tags`,
        // so "remember that the blue van is my brother's" was silently dropped
        // whenever the question also looked like a data question — which is most
        // of the time.
        let mut answer = answer;
        process_remember_tags(&state.db, &mut answer).await;
        persist_turn(&state.db, &user_text, &answer).await;
        return Ok(answer);
    }

    let camera_name = if settings.camera_name.is_empty() { "Security Camera" } else { &settings.camera_name };
    let camera_profile = read_memory(&state.db, "camera_profile").await
        .unwrap_or_else(|| "General-purpose security camera.".to_string());
    let threat_rules   = read_memory(&state.db, "threat_rules").await
        .unwrap_or_else(|| "Alert on unknown persons especially at night.".to_string());
    let known_fp       = read_memory(&state.db, "known_false_positives").await
        .unwrap_or_else(|| "No known false positives.".to_string());

    // Pull recent events (last 3 days, newest first) for context — windowed so the
    // agent talks about CURRENT activity, never a stale snapshot of old events.
    let recent_events: Vec<(String, String, f32, Option<String>)> = sqlx::query_as(
        "SELECT id, started_at, peak_score, ai_summary
         FROM motion_events
         WHERE started_at > datetime('now','-3 days')
         ORDER BY started_at DESC LIMIT 10"
    ).fetch_all(&state.db).await.unwrap_or_default();

    let recent_alerts: Vec<(String, String, String, String)> = sqlx::query_as(
        "SELECT risk_level, threat_type, summary, created_at
         FROM agent_alerts
         ORDER BY created_at DESC LIMIT 5"
    ).fetch_all(&state.db).await.unwrap_or_default();

    // Build system prompt for conversational mode
    let events_ctx = if recent_events.is_empty() {
        "No recent motion events.".to_string()
    } else {
        recent_events.iter().map(|(id, started, score, summary)| {
            let raw = summary.as_deref().unwrap_or("");
            let text = extract_summary_text(raw);
            let display = if text.is_empty() { "not yet analysed".to_string() } else { text.chars().take(100).collect::<String>() };
            // Relative time ("3m ago") so the model frames things as current.
            format!("• {} ({}, score {:.0}%) — {}", &id[..8], rel_time(started), score * 100.0, display)
        }).collect::<Vec<_>>().join("\n")
    };

    let alerts_ctx = if recent_alerts.is_empty() {
        "No recent alerts.".to_string()
    } else {
        recent_alerts.iter().map(|(risk, ttype, summary, at)| {
            // rel_time, not the raw UTC string: an alert the model reads as
            // "2026-08-01T02:11Z" is one it cannot place on the user's calendar.
            format!("• [{}] {} — {} ({})", risk.to_uppercase(), ttype, summary, rel_time(at))
        }).collect::<Vec<_>>().join("\n")
    };

    // Agies event_subscribe: load active alert rules for context
    let alert_rules_ctx = fmt_alert_rules(&state.db).await;

    // Load enrolled persons
    let known_persons: Vec<(String, String, Option<String>)> = sqlx::query_as(
        "SELECT name, role, last_seen_at FROM known_persons ORDER BY name ASC"
    ).fetch_all(&state.db).await.unwrap_or_default();

    let persons_ctx = if known_persons.is_empty() {
        "No enrolled persons.".to_string()
    } else {
        known_persons.iter().map(|(name, role, last_seen)| {
            let seen = last_seen.as_deref().unwrap_or("never");
            format!("• {} ({}) — last seen: {}", name, role, seen)
        }).collect::<Vec<_>>().join("\n")
    };

    // Gather camera + AI engine awareness
    let settings_snap = state.settings.read().await.clone();
    let active_cams: Vec<u8> = state.latest_frames.read().await.keys().cloned().collect();
    let ai_engine  = if settings_snap.ai_model.is_empty()     { "none".to_string() } else { settings_snap.ai_model.clone() };
    // These two go into the system prompt as "Vision model:" and "Guardian agent (you):".
    // On-device has no model NAME, so keying off `vision_model` told the model it was
    // literally "none" — and a 350M model takes that at face value (the DETECTED_001
    // lesson). State what it actually is, and be honest that it cannot see.
    let on_device  = !super::llm::provider_supports_vision(&settings_snap);
    let vis_model  = if on_device { "none (this engine is text-only)".to_string() }
                     else if settings_snap.vision_model.is_empty() { "none".to_string() }
                     else { settings_snap.vision_model.clone() };
    let agt_model  = if on_device {
                         format!("{}, running on-device inside the app",
                             super::local_llm::tier_label(&settings_snap.local_llm_tier))
                     }
                     else if settings_snap.vision_model.is_empty() { "none".to_string() }
                     else { settings_snap.vision_model.clone() };
    let active_cam_ids = if active_cams.is_empty() {
        "none (camera is off)".to_string()
    } else {
        active_cams.iter().map(|c| format!("Slot {c}")).collect::<Vec<_>>().join(", ")
    };

    // Behavior events from frontend tracker
    let behavior_ctx = {
        let be = state.behavior_events.read().await;
        let all: Vec<String> = be.values().flat_map(|evs| evs.iter().map(|e| {
            let name = e.name.as_deref().unwrap_or("Unknown");
            format!("{name}: {}", e.flags.join(", "))
        })).collect();
        if all.is_empty() { "No behavior flags.".to_string() } else { all.join("; ") }
    };

    // Current scene objects (what the AI model sees right now)
    let now_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();
    let scene = state.scene_objects.read().await.clone();
    let last_upd = state.scene_last_update.read().await.clone();
    let scene_ctx = {
        let mut labels: Vec<String> = scene.iter()
            .filter(|(cam, _)| last_upd.get(cam).map(|t| now_secs.saturating_sub(*t) < 30).unwrap_or(false))
            .flat_map(|(_, objs)| objs.iter().filter(|o| o.score >= 0.4).map(|o| format!("{} ({:.0}%)", o.label, o.score * 100.0)))
            .collect();
        labels.dedup();
        if labels.is_empty() { "Nothing currently detected (camera may be off or no objects in frame).".to_string() }
        else { labels.join(", ") }
    };

    // Retrieve the memory that bears on THIS question — not the whole store.
    // (This used to be `read_core_memory`, i.e. every fact ever learned, on every
    // turn. See `memory::recall`.)
    let learned_memory_ctx = super::memory::recall(&state.db, &user_text, MEMORY_BUDGET_CHARS).await;

    // Live situation snapshot + since-last-conversation continuity.
    let situation_ctx = build_situation_ctx(&state.db, last_chat_ts.as_deref()).await;

    // v27: monitor + communicate + share ONLY. The agent CANNOT control the app
    // (no camera/model/settings/tracking tags) — these media/info tags are the
    // entire surface. The catalog is GENERATED from the single tool registry
    // (`super::tools`), so the prompt, /help, and tool schemas never drift.
    //
    // Described ONCE, in whichever syntax this endpoint actually uses.
    //
    // A tool-capable model used to be handed both: `prompt_catalog()` telling it
    // to "embed these tags ANYWHERE in your reply — they execute automatically",
    // AND the same 23 tools as native function schemas. Given two ways to do one
    // thing, models mix them — which is where a literal "[SEND_CLIP:uuid]"
    // printed in the chat comes from. It also cost ~4,000 characters of prompt
    // for the half that was never going to be used.
    let native_tools = super::llm::tool_mode(&settings_snap) == super::llm::ToolMode::Native;
    let tool_guidance = if native_tools {
        // The schemas carry every name, parameter and description already.
        "## Action Tools
         You have tools for showing and sending footage (clips, snapshots, live 
         frames, share links), for browsing and searching events, for people, 
         vehicles and sounds, for alert rules, and for reading and writing your 
         own memory. CALL them — do not write their names, and never paste a raw 
         http://localhost… URL. Prefer sending the actual clip or snapshot over 
         describing it. When you learn something durable about a person, vehicle, 
         routine or rule, record it with the memory tool.".to_string()
    } else {
        format!("## Action Tools
{}{}", super::tools::prompt_catalog(), TAG_GUIDANCE)
    };
    let alert_tag_hint = if native_tools { "" } else { ALERT_TAG_HINT };

    let identity = agent_identity(&settings_snap, camera_name);
    let now_str = now_local_str();
    let today_date = chrono::Local::now().format("%Y-%m-%d").to_string();
    let system = format!(
        r#"{identity}
You are a MONITORING assistant. You can SEE the camera, query history, search and
review past events, raise proactive alert rules, and SHARE visual information
(snapshots, live frames, recorded clips) through the user's channels.
Respond in plain text — helpful, concise, security-focused.
Do NOT output JSON unless asked.

## Current date and time
{now_str}
Every timestamp below is in that same local timezone. ALWAYS state the date or
date range you are reporting on ("today, {today_date}" / "on Fri 25 Jul") — a
security report without a date is worthless. If a question has no date in it
("what happened?"), it means today. Never present an older event as current: if
the newest thing you have is from a previous day, say so explicitly.

## What you can and cannot change
You can read everything, and you can show or send any of it. The ONE thing you
can change is the user's own alert rules — add one when they ask to be notified
about something, remove one when they ask you to stop.
Nothing else. You cannot start/stop cameras, switch the AI model, change
sensitivity/thresholds/retention, track objects, or trigger the alarm. If asked
to do any of those, say plainly that you watch and report, and that the change
lives in the app's settings — then do not attempt it.

## Your Learned Memory (observations from watching the camera + past conversations)
{learned_memory_ctx}

## Current Situation (live, auto-tracked — trust this over guesses)
{situation_ctx}
## Live Camera Status
- Active slots: {active_cam_ids}
- AI detection engine: {ai_engine}
- Vision model: {vis_model}
- Guardian agent (you): {agt_model}

## What the AI Currently Sees (live detections)
{scene_ctx}
The AI detects all 80 COCO categories including: person, bicycle, car, bottle, cup, handbag, backpack, suitcase, laptop, cell phone, book, chair, couch, bed, tv, clock, vase, scissors, and more.

{tool_guidance}
## Camera Profile
{camera_profile}

## Threat Rules
{threat_rules}

## Known False Positives
{known_fp}

## Enrolled / Assigned Persons
{persons_ctx}

## Active Behavior Flags
{behavior_ctx}

## Active Alert Rules (event_subscribe)
{alert_rules_ctx}
{alert_tag_hint}
## Recent Motion Events (newest 10 of the last 3 days — NOT necessarily today)
{events_ctx}

## Recent Alerts (last 5)
{alerts_ctx}"#
    );

    // `history_ctx` was built above, before the retrieve path, so both paths see
    // the same conversation.
    let user_with_history = if history_ctx.is_empty() {
        user_text.clone()
    } else {
        format!("CONVERSATION HISTORY:\n{history_ctx}\n\n[USER]: {user_text}")
    };

    // The measurement that makes the budget falsifiable: this number must stay
    // flat as stored memory grows. If it climbs turn over turn, retrieval is
    // leaking and the llama.cpp batch abort is coming back.
    tracing::debug!(
        system_chars = system.len(), user_chars = user_with_history.len(),
        memory_chars = learned_memory_ctx.len(), history_chars = history_ctx.len(),
        "guardian prompt assembled");

    // Provider-agnostic NATIVE tool-calling first (the in-house agent brain drives
    // the same `agent::tools` registry on any backend). Falls back to the proven
    // single-shot text path on any error / unsupported provider / empty result —
    // zero regression.
    let raw = match agent_tool_loop(state, &settings, &system, &user_with_history, mode).await {
        Ok(Some(text)) if !text.trim().is_empty() => text,
        Ok(_) => {
            // No native tool channel for this endpoint — the tag loop is the
            // fallback, and it is a real loop, not a single shot.
            tag_tool_loop(state, &settings, &system, camera_name, &situation_ctx,
                          &user_with_history).await?
        }
        Err(e) => {
            // This arm used to be folded in with "provider unsupported" and
            // logged nothing at all, so a wedged endpoint and an unconfigured
            // one were indistinguishable in the logs.
            tracing::warn!("native tool loop failed, falling back to tags: {e}");
            tag_tool_loop(state, &settings, &system, camera_name, &situation_ctx,
                          &user_with_history).await?
        }
    };

    // Parse [REMEMBER:category:content] tags — routes to structured memory files
    // (family, routines, visitors, vehicles, environment, pets, rules) or flat keys
    let mut cleaned = raw.clone();
    process_remember_tags(&state.db, &mut cleaned).await;

    // Execute QUERY-tool bracket tags ([AUDIO_ACTIVITY], [DAILY_SUMMARY],
    // [PERSON_ACTIVITY:name]…) — replaced INLINE with live data. This is what
    // makes the text-tag path fully capable on models without native tool
    // calling, and stops raw placeholders leaking into the chat.
    super::tools::execute_query_tags(state, &mut cleaned).await;

    // [SHOW_EVENTS:filter] is left INTACT here — it's SURFACE-specific:
    // the app (`chat_app`) turns it into rendered event cards via the
    // guardian:show-events event; Telegram (`execute_telegram_commands`)
    // sends a real event list with clip buttons. Processing it here with a
    // "*(loading events…)*" placeholder left Telegram users staring at a
    // loading line that never resolved.

    // Media tags ([SEND_CLIP], [SNAPSHOT], [SEND_PERSON], …) are left INTACT —
    // each surface post-processes them: Telegram via `execute_telegram_commands`
    // (which this strip used to run BEFORE, silently killing chat-requested
    // clips/snapshots there), the app via `chat_app`'s tag→parts converter
    // (playable evidence cards). Stripping here broke both.

    let reply = strip_scaffold(cleaned.trim());
    persist_turn(&state.db, &user_text, &reply).await;
    Ok(reply)
}

/// Remove prompt scaffolding a small model has echoed into its reply.
///
/// `process_remember_tags` only strips well-formed `[REMEMBER:cat:fact]`. What
/// actually shipped to the user was a bare `REMEMBER:` line followed by
/// `Categories: family, routines, vehicles…` — the *instructions* for the tag,
/// not the tag — plus `## `-headed sections lifted straight out of the system
/// prompt. None of that is an answer, and none of it is recoverable by asking
/// the model more nicely.
fn strip_scaffold(text: &str) -> String {
    const LEAKS: &[&str] = &[
        "REMEMBER:", "Categories: family", "## ", "Guardian agent (you)",
        "AI detection engine:", "Vision model:", "Active slots:",
        "Detects all COCO", "The AI detects all",
    ];
    let kept: Vec<&str> = text.lines()
        .filter(|l| {
            let t = l.trim();
            !LEAKS.iter().any(|p| t.starts_with(p))
        })
        .collect();
    // Collapse the blank lines the removals leave behind.
    let mut out = String::with_capacity(text.len());
    let mut blank = false;
    for line in kept {
        if line.trim().is_empty() {
            if !blank && !out.is_empty() { out.push('\n'); }
            blank = true;
        } else {
            out.push_str(line);
            out.push('\n');
            blank = false;
        }
    }
    out.trim().to_string()
}

/// Record one exchange in the durable, cross-surface conversation, capped at 500
/// rows. Both surfaces write the SAME log, so the agent remembers a Telegram
/// exchange when you continue in the app and vice-versa.
///
/// The user's turn is always kept. The assistant's is kept only if it passed
/// [`super::llm::usable`] — a degenerate reply that got logged came back as
/// history on the next turn AND was folded into retrieved memory, so one bad
/// answer taught the agent to give more of them.
pub(super) async fn persist_turn(db: &sqlx::SqlitePool, user_text: &str, reply: &str) {
    let now = Utc::now().to_rfc3339();
    let _ = sqlx::query("INSERT INTO chat_log(id, role, content, created_at) VALUES(?,?,?,?)")
        .bind(uuid::Uuid::new_v4().to_string()).bind("user").bind(user_text).bind(&now)
        .execute(db).await;
    if super::llm::usable(reply) {
        let _ = sqlx::query("INSERT INTO chat_log(id, role, content, created_at) VALUES(?,?,?,?)")
            .bind(uuid::Uuid::new_v4().to_string()).bind("assistant").bind(reply)
            .bind(Utc::now().to_rfc3339()).execute(db).await;
    } else {
        tracing::warn!("guardian: reply failed the usability check — not logged as history");
    }
    let _ = sqlx::query(
        "DELETE FROM chat_log WHERE id NOT IN (SELECT id FROM chat_log ORDER BY created_at DESC LIMIT 500)"
    ).execute(db).await;
}


#[cfg(test)]
mod tests {
    use super::*;

    /// A security report has to be datable. Anything an hour or older must carry
    /// its local calendar date, and "just now" must stay short.
    #[test]
    fn rel_time_dates_anything_older_than_an_hour() {
        let fresh = (Utc::now() - chrono::Duration::seconds(30)).to_rfc3339();
        assert_eq!(rel_time(&fresh), "just now");

        let old = Utc::now() - chrono::Duration::days(2);
        let s = rel_time(&old.to_rfc3339());
        assert!(s.starts_with("2d ago, "), "{s}");
        // …and the date shown is the LOCAL one, not the UTC one.
        assert!(s.contains(&old.with_timezone(&chrono::Local).format("%b %d").to_string()), "{s}");
    }

    /// The clock line must name the day and the offset, so the model can never
    /// silently answer in UTC.
    #[test]
    fn now_local_str_states_the_local_day_and_offset() {
        let s = now_local_str();
        assert!(s.contains("local time, UTC"), "{s}");
        assert!(s.contains(&chrono::Local::now().format("%Y-%m-%d").to_string()), "{s}");
    }
}

#[cfg(test)]
mod scaffold_tests {
    use super::strip_scaffold;

    /// Verbatim from a real session: asked for old footage, the model replied
    /// with the INSTRUCTIONS for the remember tag and two sections lifted out of
    /// its own system prompt. None of that is an answer.
    #[test]
    fn removes_prompt_scaffolding_the_model_echoed() {
        let leaked = "REMEMBER:\n\n\
                      Categories: family, routines, vehicles, environment, pets\n\n\
                      ---\n\n\
                      ## Capabilities\n\
                      Detects all COCO categories including person, bicycle, car\n\
                      Guardian agent (you): LFM2.5-1.2B, running on-device\n\
                      I found nothing in the last 30 days.";
        let out = strip_scaffold(leaked);
        assert_eq!(out, "---\n\nI found nothing in the last 30 days.");
    }

    /// It must not eat real answers. A reply that merely mentions a person, or
    /// uses a dash, is not scaffolding.
    #[test]
    fn leaves_a_real_answer_alone() {
        let good = "Three events today, 2026-08-02.\n\
                    The last was at 14:32 on Front Door — a person, 12 seconds.";
        assert_eq!(strip_scaffold(good), good);
    }
}
