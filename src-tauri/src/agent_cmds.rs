//! Tauri command wrappers around `crate::agent` — alerts inbox, memory KV, model picker, agent chat.

use std::sync::Arc;

use serde::Deserialize;
use tauri::State;

use crate::AppState;


#[tauri::command]
pub async fn get_agent_alerts(
    state: State<'_, Arc<AppState>>,
    limit: Option<u32>,
) -> Result<Vec<crate::AgentAlert>, String> {
    let n = limit.unwrap_or(50) as i64;
    let rows: Vec<(String, String, String, String, String, bool, Option<String>, String)> =
        sqlx::query_as(
            "SELECT id,event_id,risk_level,threat_type,summary,is_false_positive,actions_taken,created_at
             FROM agent_alerts ORDER BY created_at DESC LIMIT ?",
        )
        .bind(n)
        .fetch_all(&state.db)
        .await
        .map_err(|e| e.to_string())?;
    Ok(rows.into_iter().map(|(id, event_id, risk_level, threat_type, summary, is_false_positive, actions_taken, created_at)| {
        crate::AgentAlert { id, event_id, risk_level, threat_type, summary, is_false_positive, actions_taken, created_at }
    }).collect())
}

#[tauri::command]
pub async fn delete_agent_alert(
    state: State<'_, Arc<AppState>>,
    id: String,
) -> Result<(), String> {
    // Before deleting, write the dismissed alert's pattern to false-positive memory
    // so the agent learns not to repeat-alert on the same pattern (assistant behaviour).
    let row: Option<(String, String)> = sqlx::query_as(
        "SELECT threat_type, summary FROM agent_alerts WHERE id = ?"
    ).bind(&id).fetch_optional(&state.db).await.ok().flatten();

    if let Some((ttype, summary)) = row {
        let entry = format!("{ttype}: {}", summary.chars().take(100).collect::<String>());
        crate::agent::reinforce_memory(
            &state.db, &format!("fp_dismiss_{}", &id[..8]), &entry, "manual"
        ).await;
        // Also append to the known_false_positives memory that the analysis prompt reads
        crate::agent::append_to_memory(
            &state.db, "known_false_positives", &entry
        ).await;
    }

    sqlx::query("DELETE FROM agent_alerts WHERE id = ?")
        .bind(&id)
        .execute(&state.db)
        .await
        .map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
pub async fn clear_all_agent_alerts(
    state: State<'_, Arc<AppState>>,
) -> Result<(), String> {
    sqlx::query("DELETE FROM agent_alerts")
        .execute(&state.db)
        .await
        .map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
pub async fn set_alert_feedback(
    state: State<'_, Arc<AppState>>,
    id: String,
    feedback: String,
) -> Result<(), String> {
    sqlx::query("UPDATE agent_alerts SET feedback=? WHERE id=?")
        .bind(&feedback)
        .bind(&id)
        .execute(&state.db)
        .await
        .map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
pub async fn get_reflection_prompt(
    state: State<'_, Arc<AppState>>,
) -> Result<String, String> {
    let rows: Vec<(String, String, String, i32, Option<String>)> = sqlx::query_as(
        "SELECT risk_level, threat_type, summary, is_false_positive, feedback FROM agent_alerts ORDER BY created_at DESC LIMIT 50"
    )
    .fetch_all(&state.db)
    .await
    .map_err(|e| e.to_string())?;
    serde_json::to_string(&rows.iter().map(|(risk, ttype, summary, is_fp, feedback)| {
        serde_json::json!({
            "risk_level": risk,
            "threat_type": ttype,
            "summary": summary,
            "is_false_positive": is_fp,
            "feedback": feedback,
        })
    }).collect::<Vec<_>>()).map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn report_behavior_events(
    state: State<'_, Arc<AppState>>,
    cam_id: u8,
    events: Vec<crate::BehaviorEvent>,
) -> Result<(), String> {
    state.behavior_events.write().await.insert(cam_id, events);
    Ok(())
}

#[tauri::command]
pub async fn get_agent_memory(
    state: State<'_, Arc<AppState>>,
    key: String,
) -> Result<String, String> {
    Ok(crate::agent::read_memory(&state.db, &key).await.unwrap_or_default())
}


#[tauri::command]
pub async fn set_agent_memory(
    state: State<'_, Arc<AppState>>,
    key: String,
    value: String,
) -> Result<(), String> {
    crate::agent::write_memory(&state.db, &key, &value).await;
    Ok(())
}

#[tauri::command]
pub async fn list_agent_memory(
    state: State<'_, Arc<AppState>>,
) -> Result<Vec<(String, String, String)>, String> {
    let rows: Vec<(String, String, String)> = sqlx::query_as(
        "SELECT key, value, updated_at FROM agent_memory ORDER BY updated_at DESC"
    ).fetch_all(&state.db).await.map_err(|e| e.to_string())?;
    Ok(rows)
}

#[tauri::command]
pub async fn delete_agent_memory(
    state: State<'_, Arc<AppState>>,
    key: String,
) -> Result<(), String> {
    sqlx::query("DELETE FROM agent_memory WHERE key=?")
        .bind(&key).execute(&state.db).await.map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
pub async fn get_agent_status(
    state: State<'_, Arc<AppState>>,
) -> Result<crate::AgentStatus, String> {
    Ok(crate::agent::get_status(&state).await)
}

#[derive(Deserialize)]
pub struct DetectionInput { label: String, score: f32 }

#[tauri::command]
pub async fn analyze_snapshot(
    state: State<'_, Arc<AppState>>,
    image_b64: String,
    detections: Vec<DetectionInput>,
    scene_context: Option<String>,
) -> Result<String, String> {
    let mut det_strs: Vec<String> = detections.iter()
        .map(|d| format!("{} ({:.0}%)", d.label, d.score * 100.0))
        .collect();
    // Prepend rich scene context if provided (built by the frontend's buildSceneContext)
    if let Some(ctx) = scene_context {
        det_strs.insert(0, ctx);
    }
    crate::agent::analyze_snapshot(&state, image_b64, det_strs)
        .await
        .map_err(|e| e.to_string())
}

// The raw-string `chat_with_agent` command was DELETED: it returned the reply
// with its bracket tags intact and had no callers, so its only possible effect
// was to leak tag syntax into a UI. `chat_app` is the in-app surface.

/// One piece of PLAYABLE evidence for the in-app chat: a live snapshot, a person
/// card, or a share link. Events are NOT parts — they ride on `events` below,
/// as the same cards Telegram renders into an album.
#[derive(Debug, serde::Serialize)]
pub struct ChatPart {
    #[serde(rename = "type")]
    pub kind:      String,          // "snapshot" | "person" | "link" | "chart"
    pub cam:       Option<u8>,      // snapshot
    pub name:      Option<String>,  // person
    pub thumbnail: Option<String>,  // person (bare base64)
    pub url:       Option<String>,  // link
    pub label:     Option<String>,  // link
    pub body:      Option<String>,  // chart (mermaid fence)
}

impl ChatPart {
    fn of(kind: &str) -> Self {
        ChatPart { kind: kind.into(), cam: None, name: None,
                   thumbnail: None, url: None, label: None, body: None }
    }
}

#[derive(Debug, serde::Serialize)]
pub struct ChatAppReply {
    pub text:  String,
    pub parts: Vec<ChatPart>,
    /// The events this answer is about, as cards — the SAME shape Telegram
    /// renders. They ride on the reply rather than arriving through a global
    /// `guardian:show-events` event, so cards belong to the message that
    /// produced them instead of to whichever panel happened to be listening.
    pub events: Vec<serde_json::Value>,
    /// Estimated prompt tokens in play, and the model's window. Drives the
    /// context ring in the composer. Both approximate — see `agent::context_usage`.
    pub context_used:  usize,
    pub context_limit: usize,
}

/// The in-app Guardian chat: the SAME brain AND the same evidence as Telegram.
///
/// Both surfaces call `agent::chat_with_agent`, then hand the reply to the one
/// resolver in `agent::evidence`. This function used to re-parse the tags itself
/// while Telegram re-parsed them a second, different way — which is how the
/// phone ended up with a line of text where the desktop had a picture, and how
/// `[SHARE_CLIP]` minted a real URL on one surface and nothing on the other.
///
/// Bounded so the chat can never hang on a wedged provider — the user always
/// gets an honest error bubble.
#[tauri::command]
pub async fn chat_app(
    state: State<'_, Arc<AppState>>,
    history: Vec<crate::agent::ChatMessage>,
    message: String,
    // Optional so an older caller still compiles; absent means Ask.
    mode: Option<crate::agent::Mode>,
) -> Result<ChatAppReply, String> {
    let mode = mode.unwrap_or_default();
    // Cloned before the call consumes them — the context ring needs to know how
    // much conversation was actually in play for this turn.
    let usage_history = history.clone();
    let usage_message = message.clone();
    let raw = tokio::time::timeout(
        std::time::Duration::from_secs(120),
        crate::agent::chat_with_agent(&state, history, message, mode),
    ).await
        .map_err(|_| "Guardian took too long to answer — try again.".to_string())?
        .map_err(|e| e.to_string())?;

    // ONE resolver, shared with Telegram. Everything below is presentation.
    let (text, evidence) = crate::agent::resolve_evidence(&state, &raw).await;

    let mut parts: Vec<ChatPart> = Vec::new();
    let mut events: Vec<serde_json::Value> = Vec::new();
    for item in evidence {
        use crate::agent::Evidence;
        match item {
            Evidence::Events { cards, .. } => {
                for c in cards {
                    if events.len() >= 40 { break; }
                    if let Ok(v) = serde_json::to_value(&c) { events.push(v); }
                }
            }
            Evidence::Snapshot { cam, .. } => {
                let mut p = ChatPart::of("snapshot");
                p.cam = Some(cam);
                parts.push(p);
            }
            Evidence::Person { name, thumbnail } => {
                let mut p = ChatPart::of("person");
                p.name = Some(name);
                p.thumbnail = thumbnail;
                parts.push(p);
            }
            // The app used to fold share tags into a plain inline card and mint
            // NOTHING, while Telegram minted a real Tailscale URL from the same
            // tag — so "send me a link" worked on the phone and silently didn't
            // on the desktop.
            Evidence::Link { url, label, .. } => {
                let mut p = ChatPart::of("link");
                p.url = Some(url);
                p.label = Some(label);
                parts.push(p);
            }
            Evidence::Chart => {
                let mut p = ChatPart::of("chart");
                p.body = Some(crate::agent::day_chart_mermaid(&state.db).await);
                parts.push(p);
            }
        }
    }
    // Snapshots/people/links repeat when a model restates itself; events are
    // already deduplicated by the resolver.
    parts.dedup_by(|a, b| a.kind == b.kind && a.cam == b.cam
        && a.name == b.name && a.url == b.url);
    parts.truncate(6);

    let (context_used, context_limit) = crate::agent::context_usage(
        &*state.settings.read().await, &usage_history, &usage_message, &text);

    // Never hand the UI a silent empty bubble. But evidence with no prose is the
    // model doing the right thing, not a failure — a reply of exactly
    // `[SHOW_EVENTS:today]` used to be reported to the user as a model that had
    // failed and should be replaced with a bigger one.
    let text = if !text.is_empty() {
        text
    } else if !events.is_empty() || !parts.is_empty() {
        "Here's what I found —".into()
    } else {
        "I couldn't compose a reply that time — ask again, or pick a slightly larger model in Arsenal if this keeps happening.".into()
    };

    Ok(ChatAppReply { text, parts, events, context_used, context_limit })
}

/// One persisted chat turn for the app's conversation restore.
#[derive(Debug, serde::Serialize)]
pub struct ChatLogRow {
    pub role:       String,
    /// Prose only — the stored brackets are resolved away by `evidence::replay`.
    pub content:    String,
    pub created_at: String,
    /// Event cards this turn stood for, re-read from the archive. Same shape as
    /// `ChatAppReply::events`, so the panel renders a restored turn with the
    /// same strip as a live one.
    pub events:     Vec<serde_json::Value>,
}

/// The durable conversation (oldest→newest) so the chat panel reopens with
/// the discussion intact instead of a blank slate.
#[tauri::command]
pub async fn get_chat_log(
    state: State<'_, Arc<AppState>>,
    limit: Option<i64>,
) -> Result<Vec<ChatLogRow>, String> {
    let limit = limit.unwrap_or(30).clamp(1, 200);
    let rows: Vec<(String, String, String)> = sqlx::query_as(
        "SELECT role, content, created_at FROM chat_log ORDER BY created_at DESC LIMIT ?"
    ).bind(limit).fetch_all(&state.db).await.map_err(|e| e.to_string())?;
    // Newest first, so the card budget is spent on the turns nearest the bottom
    // of the transcript — the ones the user actually lands on. The same 40-card
    // cap `chat_app` uses: cards carry base64 thumbnails, and thirty uncapped
    // turns is a fat payload on every mount.
    let mut budget = 40usize;
    let mut out: Vec<ChatLogRow> = Vec::with_capacity(rows.len());
    for (role, content, created_at) in rows {
        // A user turn is literal — only the agent emits tags.
        if role == "user" {
            out.push(ChatLogRow { role, content, created_at, events: Vec::new() });
            continue;
        }
        let (content, cards) = crate::agent::replay_evidence(&state, &content, &mut budget).await;
        let events: Vec<serde_json::Value> = cards.iter().filter_map(|c| serde_json::to_value(c).ok()).collect();
        // A turn that was ONLY a tag strips to nothing. Same wording `chat_app`
        // gives the live bubble; with no cards left to caption either, the turn
        // has nothing to say and is dropped rather than restored as a blank.
        let content = if !content.is_empty() { content }
            else if !events.is_empty() { "Here's what I found —".to_string() }
            else { continue };
        out.push(ChatLogRow { role, content, created_at, events });
    }
    out.reverse();
    Ok(out)
}

/// Erase the durable conversation.
///
/// Deletes ONLY `chat_log`. Learned memory, events, footage and enrolled people
/// are untouched — clearing a conversation is not a request to forget the house.
///
/// It is also the repair for a conversation that has gone circular: a stale reply
/// in the log comes back as context, and a small model will happily reissue it
/// for hours. Starting the thread again is the fix, and now the user can do it.
#[tauri::command]
pub async fn clear_chat_log(
    state: State<'_, Arc<AppState>>,
) -> Result<(), String> {
    sqlx::query("DELETE FROM chat_log").execute(&state.db).await
        .map_err(|e| e.to_string())?;
    tracing::info!("chat history cleared by the user");
    Ok(())
}

#[tauri::command]
pub async fn trigger_agent_now(
    state: State<'_, Arc<AppState>>,
) -> Result<(), String> {
    let state_clone = Arc::clone(&*state);
    tauri::async_runtime::spawn(async move {
        crate::agent::run_now(&state_clone).await;
    });
    Ok(())
}

/// Assistant: natural-language event query — "What happened at the door today?"
#[tauri::command]
pub async fn query_events(
    state: State<'_, Arc<AppState>>,
    question: String,
    history: Option<Vec<crate::agent::ChatMessage>>,
) -> Result<String, String> {
    // Build question with conversation history prepended so contextual phrases
    // like "show me those" and "that person" resolve against prior messages
    let full_question = match history {
        Some(h) if !h.is_empty() => {
            let hist = h.iter().take(6).map(|m| format!("[{}]: {}", m.role.to_uppercase(), m.content)).collect::<Vec<_>>().join("\n");
            format!("CONVERSATION CONTEXT:\n{hist}\n\nCURRENT QUESTION: {question}")
        }
        _ => question,
    };
    Ok(crate::agent::query_events_nl(&state, &full_question).await)
}

/// Structured event explorer — returns event cards the frontend renders inline in chat.
#[tauri::command]
pub async fn explore_events(
    state: State<'_, Arc<AppState>>,
    filter: String,
) -> Result<Vec<serde_json::Value>, String> {
    Ok(crate::agent::explore_events(&state, &filter).await)
}
