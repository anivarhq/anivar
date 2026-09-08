//! Agent tool commands — person history, similar-event search, alarm trigger, notification channel tests.

use std::sync::Arc;

use serde::Serialize;
use tauri::{Emitter, State};

use crate::AppState;


#[derive(Debug, Serialize)]
pub struct PersonHistoryEntry {
    pub event_id: String,
    pub started_at: String,
    pub thumbnail: Option<String>,
    pub ai_summary: Option<String>,
    pub peak_score: f32,
}

#[tauri::command]
pub async fn get_person_history(
    state: State<'_, Arc<AppState>>,
    name: String,
    limit: Option<i64>,
) -> Result<Vec<PersonHistoryEntry>, String> {
    let lim = limit.unwrap_or(20).min(100);
    // ai_summary contains name references; search detections JSON for person label
    let like = format!("%{}%", name.to_lowercase());
    let rows: Vec<(String, String, Option<String>, Option<String>, f32)> =
        sqlx::query_as(
            "SELECT id, started_at, thumbnail, ai_summary, peak_score
             FROM motion_events
             WHERE LOWER(ai_summary) LIKE ? OR LOWER(detections) LIKE ?
             ORDER BY started_at DESC LIMIT ?"
        )
        .bind(&like).bind(&like).bind(lim)
        .fetch_all(&state.db).await.map_err(|e| e.to_string())?;
    Ok(rows.into_iter().map(|(id, sa, th, ais, ps)| PersonHistoryEntry {
        event_id: id, started_at: sa, thumbnail: th, ai_summary: ais, peak_score: ps,
    }).collect())
}

#[derive(Debug, Serialize)]
pub struct SimilarEvent {
    pub event_id: String,
    pub started_at: String,
    pub ai_summary: Option<String>,
    pub peak_score: f32,
}

#[tauri::command]
pub async fn search_similar_events(
    state: State<'_, Arc<AppState>>,
    threat_type: String,
    time_of_day: Option<String>, // "morning" | "afternoon" | "evening" | "night"
    limit: Option<i64>,
) -> Result<Vec<SimilarEvent>, String> {
    let lim = limit.unwrap_or(10).min(50);
    let like = format!("%{}%", threat_type.to_lowercase());
    // started_at is stored in UTC; "morning/afternoon/evening/night" are the user's LOCAL
    // hours, so convert with SQLite's 'localtime' modifier (verified against real data —
    // without it, "morning" matched the UTC hour and returned the wrong events).
    let hour_filter = match time_of_day.as_deref() {
        Some("morning")   => "AND CAST(strftime('%H', started_at, 'localtime') AS INT) BETWEEN 6 AND 11",
        Some("afternoon") => "AND CAST(strftime('%H', started_at, 'localtime') AS INT) BETWEEN 12 AND 17",
        Some("evening")   => "AND CAST(strftime('%H', started_at, 'localtime') AS INT) BETWEEN 18 AND 21",
        Some("night")     => "AND (CAST(strftime('%H', started_at, 'localtime') AS INT) >= 22 OR CAST(strftime('%H', started_at, 'localtime') AS INT) < 6)",
        _ => "",
    };
    let sql = format!(
        "SELECT id, started_at, ai_summary, peak_score
         FROM motion_events
         WHERE LOWER(ai_summary) LIKE ? {hour_filter}
         ORDER BY started_at DESC LIMIT ?"
    );
    let rows: Vec<(String, String, Option<String>, f32)> =
        sqlx::query_as(&sql).bind(&like).bind(lim)
        .fetch_all(&state.db).await.map_err(|e| e.to_string())?;
    Ok(rows.into_iter().map(|(id, sa, ais, ps)| SimilarEvent {
        event_id: id, started_at: sa, ai_summary: ais, peak_score: ps,
    }).collect())
}

#[tauri::command]
pub async fn trigger_alarm(state: State<'_, Arc<AppState>>) -> Result<(), String> {
    state.app_handle.emit("alarm:trigger", ()).ok();
    Ok(())
}

/// Result of the one-tap Telegram "Connect" — what the UI shows the user.
#[derive(serde::Serialize)]
pub struct TelegramConnect {
    pub bot_username: String,      // e.g. "MyAnivarBot" — proves the token works
    pub chat_id:      String,      // auto-detected; empty if the user hasn't messaged the bot yet
    pub chat_name:    String,      // who the chat is with (first name / title)
    pub needs_message: bool,       // true → prompt "message your bot, then Connect again"
}

/// One-tap connect: validate the bot token (getMe) and auto-detect the chat ID
/// (getUpdates) so the user never has to hunt for a numeric ID via a 3rd-party
/// bot. Flow: paste token → Connect. If no chat is found, we return the bot's
/// @username and `needs_message=true` so the UI can say "send /start to @X, then
/// Connect again". Trims accidental whitespace and a pasted "bot" prefix.
#[tauri::command]
pub async fn telegram_connect(bot_token: String) -> Result<TelegramConnect, String> {
    let token = bot_token.trim().trim_start_matches("bot").trim().to_string();
    if token.is_empty() { return Err("Paste your bot token first (from @BotFather).".into()); }
    let client = reqwest::Client::new();

    // 1) Validate the token — getMe. A wrong/revoked token 401s here.
    let me: serde_json::Value = client
        .get(format!("https://api.telegram.org/bot{token}/getMe"))
        .timeout(std::time::Duration::from_secs(10))
        .send().await.map_err(|e| format!("Network error: {e}"))?
        .json().await.unwrap_or_default();
    if !me.get("ok").and_then(|v| v.as_bool()).unwrap_or(false) {
        let desc = me.get("description").and_then(|v| v.as_str()).unwrap_or("Unauthorized");
        return Err(format!("Telegram rejected this token ({desc}). Copy it fresh from @BotFather (/mybots → your bot → API Token)."));
    }
    let bot_username = me["result"]["username"].as_str().unwrap_or("").to_string();

    // 2) Auto-detect the chat ID — getUpdates returns recent messages sent TO the
    //    bot. We take the most recent private/group chat. Empty ⇒ the user hasn't
    //    messaged the bot yet (Telegram only surfaces updates after first contact).
    let updates: serde_json::Value = client
        .get(format!("https://api.telegram.org/bot{token}/getUpdates"))
        .timeout(std::time::Duration::from_secs(10))
        .send().await.map_err(|e| format!("Network error: {e}"))?
        .json().await.unwrap_or_default();
    let mut chat_id = String::new();
    let mut chat_name = String::new();
    if let Some(arr) = updates["result"].as_array() {
        for upd in arr.iter().rev() {
            // message / edited_message / channel_post all carry a `chat`.
            let chat = upd.get("message").or_else(|| upd.get("edited_message"))
                .or_else(|| upd.get("channel_post")).and_then(|m| m.get("chat"));
            if let Some(chat) = chat {
                if let Some(id) = chat.get("id").and_then(|v| v.as_i64()) {
                    chat_id = id.to_string();
                    chat_name = chat.get("title").and_then(|v| v.as_str())
                        .or_else(|| chat.get("first_name").and_then(|v| v.as_str()))
                        .or_else(|| chat.get("username").and_then(|v| v.as_str()))
                        .unwrap_or("your chat").to_string();
                    break;
                }
            }
        }
    }
    let needs_message = chat_id.is_empty();
    Ok(TelegramConnect { bot_username, chat_id, chat_name, needs_message })
}

#[tauri::command]
pub async fn send_telegram_test(bot_token: String, chat_id: String) -> Result<String, String> {
    if bot_token.is_empty() { return Err("Bot token is empty".to_string()); }
    if chat_id.is_empty()   { return Err("Chat ID is empty".to_string()); }
    let url = format!("https://api.telegram.org/bot{}/sendMessage", bot_token);
    let resp = reqwest::Client::new()
        .post(&url)
        .timeout(std::time::Duration::from_secs(10))
        .json(&serde_json::json!({ "chat_id": &chat_id, "text": "Anivar Guardian connected! This is a test message." }))
        .send()
        .await
        .map_err(|e| format!("Network error: {e}"))?;
    let status = resp.status();
    let body: serde_json::Value = resp.json().await.unwrap_or_default();
    if body.get("ok").and_then(|v| v.as_bool()).unwrap_or(false) {
        Ok("Message sent successfully".to_string())
    } else {
        let desc = body.get("description").and_then(|v| v.as_str()).unwrap_or("unknown error");
        Err(format!("Telegram error (HTTP {status}): {desc}"))
    }
}

/// Set a new password — hashes it with SHA-256 before storing.
/// Never stores the plaintext password.
#[tauri::command]
pub async fn set_auth_password(password: String, state: State<'_, Arc<AppState>>) -> Result<(), String> {
    let hash = if password.is_empty() {
        String::new()
    } else {
        use sha2::{Sha256, Digest};
        let mut h = Sha256::new();
        h.update(password.as_bytes());
        format!("{:x}", h.finalize())
    };
    let mut settings = state.settings.write().await;
    settings.auth_password_hash = hash;
    let s = settings.clone();
    drop(settings);
    crate::db::save_settings_to_db(&state.db, &s).await.map_err(|e| e.to_string())
}

