//! Agent tool commands — person history, similar-event search, alarm trigger, notification channel tests.





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
        .send().await.map_err(|e| format!("Network error: {}", e.without_url()))?
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
        .send().await.map_err(|e| format!("Network error: {}", e.without_url()))?
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
        .map_err(|e| format!("Network error: {}", e.without_url()))?;
    let status = resp.status();
    let body: serde_json::Value = resp.json().await.unwrap_or_default();
    if body.get("ok").and_then(|v| v.as_bool()).unwrap_or(false) {
        Ok("Message sent successfully".to_string())
    } else {
        let desc = body.get("description").and_then(|v| v.as_str()).unwrap_or("unknown error");
        Err(format!("Telegram error (HTTP {status}): {desc}"))
    }
}


