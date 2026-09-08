//! Alert dispatch + multi-channel notification.
//!
//! Owns the fan-out to Telegram (inline keyboard + photo + album), Discord,
//! Slack, Pushover, Signal, WhatsApp, and ntfy. Also hosts the per-risk
//! emoji / priority / colour helpers.
//!
//! The `process_telegram_updates` long-poll loop lives here too — it owns
//! state for inline-keyboard ack callbacks and slash-commands sent from
//! the Telegram side.

use std::sync::Arc;
use std::time::Duration;

use base64::Engine as _;
use chrono::{Local, Utc, Datelike, TimeZone, NaiveDate, Duration as ChronoDuration};

use serde::Deserialize;

use crate::{AppState, Settings};
use super::types::*;
use super::memory::{
    read_memory, write_memory, extract_summary_text, fmt_alert_rules, append_memory,
};
use super::llm::call_llm;
use super::{chat_with_agent, ChatMessage};
use super::analysis::risk_meets_threshold;

// ─── Alert dispatch ───────────────────────────────────────────────────────────

pub(super) fn risk_to_emoji(risk: &str) -> &'static str {
    match risk {
        "critical"   => "🚨",
        "suspicious" => "⚠️",
        "monitor"    => "👁",
        _            => "✅",      // normal
    }
}

/// Rich alert dispatch — includes photo, detected objects, duration, time context.
pub(super) async fn dispatch(
    settings: &Settings,
    alert: &AgentAlert,
    thumbnail: Option<&str>,
    duration_secs: Option<f64>,
    detections_json: Option<&str>,
    data_dir: &std::path::Path,
) {
    dispatch_with_clip(settings, alert, thumbnail, duration_secs, detections_json, None, data_dir).await;
}

pub(super) async fn dispatch_with_clip(
    settings: &Settings,
    alert: &AgentAlert,
    thumbnail: Option<&str>,
    duration_secs: Option<f64>,
    detections_json: Option<&str>,
    clip_path: Option<&str>,
    // v11: data_dir lets the Telegram clip branch find ffmpeg for the faststart
    // remux. Callers always have the AppState in scope.
    data_dir: &std::path::Path,
) {
    // Gate at dispatch level — every caller is protected regardless of call site
    if !risk_meets_threshold(&alert.risk_level, &settings.alert_min_risk) {
        return;
    }

    let emoji = risk_to_emoji(&alert.risk_level);
    let now_str = Local::now().format("%H:%M %Z").to_string();

    // Build detection summary: "Person × 2, Vehicle × 1"
    let det_summary: String = detections_json
        .and_then(|j| serde_json::from_str::<Vec<serde_json::Value>>(j).ok())
        .map(|v| {
            let mut counts: std::collections::HashMap<String, usize> = Default::default();
            for d in &v {
                let label = d["label"].as_str().unwrap_or("Object").to_string();
                *counts.entry(label).or_insert(0) += 1;
            }
            let mut parts: Vec<String> = counts.into_iter()
                .map(|(k, n)| if n > 1 { format!("{k} ×{n}") } else { k })
                .collect();
            parts.sort();
            parts.join(", ")
        })
        .unwrap_or_default();

    let dur_str = duration_secs
        .map(|d| if d >= 60.0 { format!("{:.0}m {:.0}s", d / 60.0, d % 60.0) }
                 else { format!("{:.0}s", d) })
        .unwrap_or_default();

    let title = format!("{emoji} {} — {}", alert.risk_level.to_uppercase(), alert.threat_type);

    // Decode thumbnail once for channels that support photo
    let thumb_bytes: Option<Vec<u8>> = thumbnail
        .and_then(|t| base64::engine::general_purpose::STANDARD
            .decode(t.trim_start_matches("data:image/jpeg;base64,")).ok());

    // ── Telegram — clean, informative, with photo or clip ────────────────
    if !settings.telegram_bot_token.is_empty() && !settings.telegram_chat_id.is_empty() {
        let cam = if settings.camera_name.is_empty() { "Camera" } else { &settings.camera_name };
        // Clean caption — no JSON, no technical noise
        // Add context line
        let mut ctx_parts = vec![format!("📷 {}", cam), format!("🕐 {}", now_str)];
        if !det_summary.is_empty() { ctx_parts.push(format!("👁 {}", det_summary)); }
        if !dur_str.is_empty()     { ctx_parts.push(format!("⏱ {}", dur_str)); }
        // Header line = risk + threat type (at-a-glance), then the summary, then context.
        let mut caption = format!("{title}\n{}\n\n{}", alert.summary, ctx_parts.join("  ·  "));
        if !alert.recommended_action_display().is_empty() {
            caption.push_str(&format!("\n\n💡 {}", alert.recommended_action_display()));
        }

        // The buttons ride on the alert ITSELF. They used to arrive as a second,
        // separate message ("What do you want to do about this?"), so the
        // picture and the actions about that picture were two bubbles apart.
        let keyboard = alert_action_keyboard(alert);

        // Try the clip first (most informative), fall back to the thumbnail,
        // then to text — `send_telegram_photo_kb` does the last two steps, so a
        // failed photo upload still delivers the words and the buttons.
        let mut sent = false;
        if let Some(path) = clip_path {
            let src = std::path::Path::new(path);
            // Prefer a faststart remux; fall back to the raw file if ffmpeg fails.
            let fast = ensure_faststart(data_dir, src).await;
            let send_path = fast.as_deref().unwrap_or(src);
            if let Ok(clip_data) = tokio::fs::read(send_path).await {
                if clip_data.len() <= TELEGRAM_MAX_UPLOAD {
                    let fname = src.file_name()
                        .and_then(|n| n.to_str()).unwrap_or("clip.mp4").to_string();
                    sent = send_telegram_video(
                        &settings.telegram_bot_token, &settings.telegram_chat_id,
                        clip_data, &fname, &caption, Some(keyboard.clone()),
                    ).await;
                }
            }
            if let Some(p) = fast { let _ = tokio::fs::remove_file(p).await; }
        }
        if !sent {
            send_telegram_photo_kb(
                &settings.telegram_bot_token, &settings.telegram_chat_id,
                thumb_bytes.clone(), &caption, keyboard,
            ).await;
        }
    }

}

/// Buttons for an alert message.
///
/// Every alert with an event gets the media row — a fresh snapshot, the
/// recorded clip, or a link that opens off the LAN — so "show me more" is one
/// tap from the notification. Suspicious and critical also get triage.
pub(super) fn alert_action_keyboard(alert: &AgentAlert) -> serde_json::Value {
    let mut rows: Vec<serde_json::Value> = Vec::new();
    if !alert.event_id.is_empty() {
        rows.push(serde_json::json!([
            { "text": "📸 Snapshot", "callback_data": format!("snap:{}",  alert.event_id) },
            { "text": "🎬 Clip",     "callback_data": format!("clip:{}",  alert.event_id) },
            { "text": "🔗 Link",     "callback_data": format!("share:{}", alert.event_id) },
        ]));
    }
    if matches!(alert.risk_level.as_str(), "suspicious" | "critical") && !alert.id.is_empty() {
        rows.push(serde_json::json!([
            { "text": "✅ Acknowledge", "callback_data": format!("ack:{}", alert.id) },
            { "text": "🚫 False alarm", "callback_data": format!("fp:{}",  alert.id) },
        ]));
        rows.push(serde_json::json!([
            { "text": "🚨 Escalate", "callback_data": format!("esc:{}", alert.id) },
        ]));
    }
    serde_json::Value::Array(rows)
}

// ─── Telegram helpers ─────────────────────────────────────────────────────────

pub(super) async fn send_telegram_action(bot_token: &str, chat_id: &str, action: &str) {
    if bot_token.is_empty() || chat_id.is_empty() { return; }
    let url = format!("https://api.telegram.org/bot{}/sendChatAction", bot_token);
    reqwest::Client::new().post(&url)
        .timeout(Duration::from_secs(5))
        .json(&serde_json::json!({ "chat_id": chat_id, "action": action }))
        .send().await.ok();
}

/// Escape text for Telegram **HTML** parse_mode. HTML needs only these three
/// entities — far safer than Markdown (where a stray `_`/`*` silently drops the
/// whole message). Apply to EVERY dynamic interpolation that goes into an HTML
/// message (summaries, labels, camera names, user queries).
pub(super) fn tg_esc(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

/// Send a message rendered with HTML parse_mode (bold headers, etc.). ONLY pass
/// HTML you built yourself with every dynamic part run through `tg_esc` —
/// otherwise Telegram 400s and drops the message. Use this for STATIC-structure
/// command output (/help, /status, /summary, …); keep dynamic-only text (raw LLM
/// answers, alert bodies) on plain `send_telegram`.
pub(super) async fn send_telegram_html(bot_token: &str, chat_id: &str, html: &str) {
    if bot_token.is_empty() || chat_id.is_empty() || html.is_empty() { return; }
    let url = format!("https://api.telegram.org/bot{}/sendMessage", bot_token);
    let result = reqwest::Client::new()
        .post(&url)
        .timeout(Duration::from_secs(10))
        .json(&serde_json::json!({
            "chat_id": chat_id,
            "text": html,
            "parse_mode": "HTML",
            "disable_web_page_preview": true,
        }))
        .send()
        .await;
    if let Err(e) = result {
        eprintln!("[Telegram] sendMessage(HTML) error: {e}");
    }
}

/// HTML message carrying an inline keyboard — [`send_telegram_html`] plus
/// buttons, for lists whose rows should be tappable rather than describing an
/// id the user then has to type.
pub(super) async fn send_telegram_html_kb(
    bot_token: &str, chat_id: &str, html: &str, keyboard: serde_json::Value,
) {
    if bot_token.is_empty() || chat_id.is_empty() || html.is_empty() { return; }
    let url = format!("https://api.telegram.org/bot{}/sendMessage", bot_token);
    let result = reqwest::Client::new()
        .post(&url)
        .timeout(Duration::from_secs(10))
        .json(&serde_json::json!({
            "chat_id": chat_id,
            "text": html,
            "parse_mode": "HTML",
            "disable_web_page_preview": true,
            "reply_markup": keyboard,
        }))
        .send()
        .await;
    if let Err(e) = result {
        eprintln!("[Telegram] sendMessage(HTML+kb) error: {e}");
    }
}

// The Telegram search flow is GONE — button, force-reply prompt, reply
// intercept, `/search` handler, and the empty-result hint. It was a menu
// reimplementation of the thing the agent already does better: typing
// "the red van last Tuesday" gets the same rows through `retrieve.rs`,
// which understands the date and the description instead of matching them
// as keywords. `[SEARCH_EVENTS:]` still works — it resolves through
// `evidence.rs` like every other tag.

pub async fn send_telegram(bot_token: &str, chat_id: &str, text: &str) {
    if bot_token.is_empty() || chat_id.is_empty() || text.is_empty() { return; }
    let url = format!("https://api.telegram.org/bot{}/sendMessage", bot_token);
    // No parse_mode — model output may contain unescaped chars that Telegram rejects
    let result = reqwest::Client::new()
        .post(&url)
        .timeout(Duration::from_secs(10))
        .json(&serde_json::json!({ "chat_id": chat_id, "text": text }))
        .send()
        .await;
    if let Err(e) = result {
        eprintln!("[Telegram] sendMessage error: {e}");
    }
}

/// Build the standard alert inline keyboard. Row 1 = triage (Acknowledge /
/// False alarm), keyed by `alert_id`. Row 2 = on-demand media (Snapshot / Clip /
/// Share), keyed by `event_id` so the bot can fetch a fresh look, re-send the
/// recorded clip, or mint a private share link without re-pushing media every
/// time. Row 3 = Escalate. All actions are monitor/share-only — none of them
/// changes any app setting.
pub(super) fn alert_keyboard(alert_id: &str, event_id: &str) -> serde_json::Value {
    let mut rows: Vec<serde_json::Value> = vec![serde_json::json!([
        { "text": "✅ Acknowledge", "callback_data": format!("ack:{}", alert_id) },
        { "text": "🚫 False alarm", "callback_data": format!("fp:{}", alert_id) },
    ])];
    if !event_id.is_empty() {
        rows.push(serde_json::json!([
            { "text": "📸 Snapshot", "callback_data": format!("snap:{}", event_id) },
            { "text": "🎬 Clip",     "callback_data": format!("clip:{}", event_id) },
            { "text": "🔗 Share",    "callback_data": format!("share:{}", event_id) },
        ]));
    }
    rows.push(serde_json::json!([
        { "text": "🚨 Escalate", "callback_data": format!("esc:{}", alert_id) },
    ]));
    serde_json::Value::Array(rows)
}

pub(super) async fn send_telegram_with_keyboard(token: &str, chat_id: &str, text: &str, keyboard: serde_json::Value) -> bool {
    if token.is_empty() || chat_id.is_empty() { return false; }
    let url = format!("https://api.telegram.org/bot{}/sendMessage", token);
    // No parse_mode: alert text is dynamic (LLM summaries / object labels) and
    // unescaped Markdown special chars make Telegram silently drop the whole
    // message. Plain text is always delivered.
    let body = serde_json::json!({
        "chat_id": chat_id,
        "text": text,
        "reply_markup": { "inline_keyboard": keyboard }
    });
    let Ok(resp) = reqwest::Client::new()
        .post(&url)
        .timeout(Duration::from_secs(10))
        .json(&body)
        .send()
        .await
    else { return false };
    resp.status().is_success()
}

/// Send photo with a markdown caption (used for rich alerts).
pub(super) async fn send_telegram_photo_caption(bot_token: &str, chat_id: &str, jpeg: Vec<u8>, caption: &str) {
    if bot_token.is_empty() || chat_id.is_empty() || jpeg.is_empty() { return; }
    let url  = format!("https://api.telegram.org/bot{}/sendPhoto", bot_token);
    let Ok(part) = reqwest::multipart::Part::bytes(jpeg)
        .file_name("alert.jpg")
        .mime_str("image/jpeg")
    else { return; };
    // No parse_mode — the caption embeds dynamic LLM/label text; unescaped
    // Markdown would make Telegram silently reject the whole photo+caption.
    let form = reqwest::multipart::Form::new()
        .text("chat_id", chat_id.to_string())
        .text("caption", caption.to_string())
        .part("photo", part);
    if let Err(e) = reqwest::Client::new()
        .post(&url).timeout(Duration::from_secs(30))
        .multipart(form).send().await
    { eprintln!("[Telegram] sendPhoto(caption) error: {e}"); }
}

pub(super) async fn send_telegram_photo(bot_token: &str, chat_id: &str, jpeg: Vec<u8>, caption: &str) {
    if bot_token.is_empty() || chat_id.is_empty() || jpeg.is_empty() { return; }
    let url = format!("https://api.telegram.org/bot{}/sendPhoto", bot_token);
    let part = match reqwest::multipart::Part::bytes(jpeg)
        .file_name("snapshot.jpg")
        .mime_str("image/jpeg")
    {
        Ok(p) => p,
        Err(e) => { eprintln!("[Telegram] mime error: {e}"); return; }
    };
    let form = reqwest::multipart::Form::new()
        .text("chat_id", chat_id.to_string())
        .text("caption", caption.to_string())
        .part("photo", part);
    if let Err(e) = reqwest::Client::new().post(&url)
        .timeout(Duration::from_secs(30))
        .multipart(form)
        .send().await
    {
        eprintln!("[Telegram] sendPhoto error: {e}");
    }
}

// ─── Telegram helpers ─────────────────────────────────────────────────────────

// ─── Telegram helpers ─────────────────────────────────────────────────────────

/// Remux a clip so its `moov` atom is at the front (`+faststart`). This is what
/// lets Telegram render the clip as an INLINE, progressively-playable video
/// instead of forcing a raw download. `-c copy` means no re-encode — it's a
/// fast container rewrite (our NVR already records H.264, no audio). Returns the
/// path to a temp mp4 the caller must delete, or `None` if ffmpeg is missing/
/// failed (caller should then fall back to sending the original).
pub(super) async fn ensure_faststart(
    data_dir: &std::path::Path,
    clip_path: &std::path::Path,
) -> Option<std::path::PathBuf> {
    let ffmpeg_bin = crate::ensure_ffmpeg(data_dir).await.ok()?;
    if !clip_path.exists() { return None; }
    let temp_path = data_dir.join(format!(
        "faststart_{}_{}.mp4",
        std::process::id(),
        chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0),
    ));
    let ffmpeg_bin_c = ffmpeg_bin.clone();
    let temp_c = temp_path.clone();
    let clip_c = clip_path.to_string_lossy().to_string();
    let status = tokio::task::spawn_blocking(move || {
        crate::proc::std_cmd(&ffmpeg_bin_c)
            .args([
                "-y", "-i", &clip_c,
                "-c", "copy",
                "-movflags", "+faststart",
                &temp_c.to_string_lossy(),
            ])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
    }).await.ok()?.ok()?;
    if status.success() && temp_path.exists() { Some(temp_path) } else {
        let _ = std::fs::remove_file(&temp_path);
        None
    }
}

/// Send a clip as an INLINE-playable Telegram video (`sendVideo` +
/// `supports_streaming`). Telegram renders a real video player (tap to play)
/// rather than a file-download card. The clip should already be faststart —
/// pass it through [`ensure_faststart`] first. Optionally attaches an inline
/// keyboard. Returns whether the API accepted the upload.
pub(super) async fn send_telegram_video(
    bot_token: &str, chat_id: &str, data: Vec<u8>, filename: &str,
    caption: &str, keyboard: Option<serde_json::Value>,
) -> bool {
    if bot_token.is_empty() || chat_id.is_empty() || data.is_empty() { return false; }
    let url = format!("https://api.telegram.org/bot{}/sendVideo", bot_token);
    let Ok(part) = reqwest::multipart::Part::bytes(data)
        .file_name(filename.to_string())
        .mime_str("video/mp4")
    else { return false; };
    let mut form = reqwest::multipart::Form::new()
        .text("chat_id", chat_id.to_string())
        .text("caption", caption.to_string())
        .text("supports_streaming", "true")
        .part("video", part);
    if let Some(kb) = keyboard {
        form = form.text(
            "reply_markup",
            serde_json::json!({ "inline_keyboard": kb }).to_string(),
        );
    }
    match reqwest::Client::new().post(&url)
        .timeout(Duration::from_secs(120))
        .multipart(form)
        .send().await
    {
        Ok(r) => r.status().is_success(),
        Err(e) => { eprintln!("[Telegram] sendVideo error: {e}"); false }
    }
}

/// Send 2–10 photos as ONE album (`sendMediaGroup`), each with its own HTML
/// caption you see when you swipe.
///
/// This is the real API for the thing the live view used to fake: six separate
/// `sendPhoto` calls 100 ms apart, hoping Telegram would group them. It didn't
/// always, and a slow upload split the "album" across the chat. Returns whether
/// the group was accepted — callers fall back to individual photos.
pub(super) async fn send_telegram_media_group(
    bot_token: &str, chat_id: &str, items: Vec<(Vec<u8>, String)>,
) -> bool {
    // Telegram's own bounds: an album is 2..=10 items.
    if bot_token.is_empty() || chat_id.is_empty() || items.len() < 2 { return false; }
    let url = format!("https://api.telegram.org/bot{}/sendMediaGroup", bot_token);
    let mut media: Vec<serde_json::Value> = Vec::new();
    let mut form = reqwest::multipart::Form::new().text("chat_id", chat_id.to_string());
    for (i, (jpeg, caption)) in items.into_iter().take(10).enumerate() {
        let name = format!("file{i}");
        let Ok(part) = reqwest::multipart::Part::bytes(jpeg)
            .file_name(format!("{name}.jpg")).mime_str("image/jpeg")
        else { continue };
        media.push(serde_json::json!({
            "type": "photo",
            "media": format!("attach://{name}"),
            // Captions come from `caption_html`, which escapes every dynamic
            // field — one unescaped '<' in an AI summary sinks the whole album.
            "caption": caption.chars().take(1000).collect::<String>(),
            "parse_mode": "HTML",
        }));
        form = form.part(name, part);
    }
    if media.len() < 2 { return false; }
    form = form.text("media", serde_json::Value::Array(media).to_string());
    match reqwest::Client::new().post(&url)
        .timeout(Duration::from_secs(90))
        .multipart(form).send().await
    {
        Ok(r) => r.status().is_success(),
        Err(e) => { eprintln!("[Telegram] sendMediaGroup error: {e}"); false }
    }
}

/// Collect 6 frames from the given camera (1.5 s apart) and send them as one album.
pub(super) async fn send_telegram_live_album(bot_token: &str, chat_id: &str, state: &Arc<AppState>, cam_id: u8) {
    send_telegram_action(bot_token, chat_id, "upload_photo").await;
    let mut frames: Vec<Vec<u8>> = Vec::new();
    for i in 0u8..6 {
        if let Some(j) = state.latest_frames.read().await.get(&cam_id).cloned() {
            frames.push(j);
        }
        if i < 5 { tokio::time::sleep(Duration::from_millis(1500)).await; }
    }
    if frames.is_empty() {
        send_telegram(bot_token, chat_id, &format!("No frames from camera {cam_id} yet.")).await;
        return;
    }
    let cam = camera_label(state, cam_id as i64).await;
    let total = frames.len();
    let items: Vec<(Vec<u8>, String)> = frames.iter().enumerate()
        .map(|(i, j)| (j.clone(),
            format!("<b>{}</b> — live, frame {} of {}", tg_esc(&cam), i + 1, total)))
        .collect();
    if send_telegram_media_group(bot_token, chat_id, items).await { return; }
    // Album refused (a single frame, or an upload failure) — never silence.
    for jpeg in frames {
        send_telegram_photo(bot_token, chat_id, jpeg, &format!("{cam} — live")).await;
    }
}

/// Check whether a Guardian tool is enabled (defaults to true).
pub(super) async fn tool_enabled(db: &sqlx::SqlitePool, tool_id: &str) -> bool {
    read_memory(db, &format!("tool_{tool_id}")).await
        .map(|v| v.trim() != "false")
        .unwrap_or(true)
}

/// How many events one tap grid carries. Two buttons per row, so this is six
/// rows — past that the keyboard is taller than the screen and stops being a
/// grid you can scan.
const GRID_MAX: usize = 12;

/// Telegram's bot upload ceiling. Past it a link is the answer, not an apology.
const TELEGRAM_MAX_UPLOAD: usize = 50 * 1024 * 1024;

/// Render resolved evidence into Telegram messages.
///
/// This replaced twelve hand-rolled `next_tag` blocks, each of which re-parsed
/// the agent's reply and decided for itself how much of an event to show. The
/// result was a phone receiving one line of text where the desktop received a
/// picture, an AI summary, a duration and a risk level — from the same query,
/// against the same rows. Both surfaces now render the same `Vec<Evidence>`;
/// only the medium differs.
pub(super) async fn render_evidence(
    state: &Arc<AppState>, token: &str, chat_id: &str, items: &[super::evidence::Evidence],
) {
    use super::evidence::Evidence;
    for item in items {
        match item {
            Evidence::Events { cards, play, label } =>
                render_events(state, token, chat_id, cards, *play, label).await,

            Evidence::Snapshot { cam, burst } if *burst =>
                send_telegram_live_album(token, chat_id, state, *cam).await,

            Evidence::Snapshot { cam, .. } => {
                let name = camera_label(state, *cam as i64).await;
                match state.latest_frames.read().await.get(cam).cloned() {
                    Some(jpeg) => send_telegram_photo_kb(token, chat_id, Some(jpeg),
                        &format!("📸 {name} — live"),
                        serde_json::json!([[
                            { "text": "🔗 Live link", "callback_data": format!("getlive:{cam}") }
                        ]])).await,
                    None => send_telegram(token, chat_id,
                        &format!("No frame from {name} yet — is it active?")).await,
                }
            }

            Evidence::Person { name, thumbnail } => {
                use base64::Engine;
                let bytes = thumbnail.as_deref().and_then(|t|
                    base64::engine::general_purpose::STANDARD.decode(t.trim()).ok());
                match bytes {
                    Some(b) => send_telegram_photo(token, chat_id, b, &format!("👤 {name}")).await,
                    None => send_telegram(token, chat_id, &format!("No photo on file for {name}.")).await,
                }
            }

            Evidence::Link { url, expires_at, label, .. } => {
                send_telegram_with_keyboard(token, chat_id,
                    &format!("🔗 {label} — works for {}.\n{url}\n\nExpired? Just ask again for a fresh link.",
                        link_life(*expires_at)),
                    serde_json::json!([[ { "text": "▶️ Open", "url": url } ]])).await;
            }

            Evidence::Chart => {
                let txt = super::day_chart_text(&state.db).await;
                send_telegram(token, chat_id, &txt).await;
            }
        }
    }
}

/// A camera's configured name, or a plain "Camera N".
async fn camera_label(state: &Arc<AppState>, cam: i64) -> String {
    super::retrieve::camera_names(&state.db).await
        .get(&cam).cloned().unwrap_or_else(|| format!("Camera {cam}"))
}

/// Events → a numbered list and a tap grid.
///
/// **The detail lives in the TEXT; a button is a target, not a record.** Telegram
/// gives every button in a row the SAME width, so a long label beside a narrow
/// one renders as a truncated string next to a mostly-empty box — which is what
/// `[🎬 Aug 03 21:14 · Front Door · 12s · person] [🔗]` actually looked like.
/// Numbered buttons are short and uniform, and the number ties each one to the
/// line above it.
async fn render_events(
    state: &Arc<AppState>, token: &str, chat_id: &str,
    cards: &[super::evidence::EventCard], play: bool, label: &str,
) {
    if cards.is_empty() { return; }
    let names = super::retrieve::camera_names(&state.db).await;
    let cam_of = |c: &super::evidence::EventCard| names.get(&c.cam).cloned()
        .unwrap_or_else(|| format!("Camera {}", c.cam));

    // "Send me that clip" — upload the footage itself, not a card about it.
    if play {
        for c in cards.iter().take(3) {
            send_event_clip(state, token, chat_id, c, &cam_of(c)).await;
        }
        return;
    }

    let shown: Vec<&super::evidence::EventCard> = cards.iter().take(GRID_MAX).collect();

    // ONE event: its picture, its full record, and both actions on one message.
    // A grid of one is just a list with a single row.
    if shown.len() == 1 {
        let c = shown[0];
        send_telegram_photo_kb(token, chat_id, c.thumb_bytes(),
            &caption_plain(c, &cam_of(c)),
            serde_json::json!([[
                { "text": "🎬 Send clip",  "callback_data": format!("getclip:{}", c.id) },
                { "text": "🔗 Share link", "callback_data": format!("share:{}",   c.id) },
            ]])).await;
        return;
    }

    // `describe()` returns "events" for an exact id set — the resolver already
    // chose those rows, so naming a filter would be inventing one.
    let subject = if label == "events" { String::new() } else { format!(" · {label}") };
    let mut html = format!("📋 <b>{} events</b>{}\n\n", cards.len(), tg_esc(&subject));

    let mut rows: Vec<serde_json::Value> = Vec::new();
    let mut pair: Vec<serde_json::Value> = Vec::new();
    for (i, c) in shown.iter().enumerate() {
        let n = i + 1;
        html.push_str(&list_line(n, c, &cam_of(c)));
        pair.push(serde_json::json!({
            "text": format!("{n} · {}", short_time(&c.ts)),
            "callback_data": format!("getclip:{}", c.id),
        }));
        // Two per row: wide enough that "12 · 21:14" never truncates, narrow
        // enough that twelve events are six rows rather than twelve.
        if pair.len() == 2 { rows.push(serde_json::Value::Array(std::mem::take(&mut pair))); }
    }
    if !pair.is_empty() { rows.push(serde_json::Value::Array(pair)); }

    if cards.len() > shown.len() {
        html.push_str(&format!("\n<i>…and {} more — narrow it with a camera, a day or a word.</i>\n",
            cards.len() - shown.len()));
    }
    html.push_str("\n<i>Tap a number for its clip; the clip carries a share link.</i>");
    send_telegram_html_kb(token, chat_id, &html,
        serde_json::json!({ "inline_keyboard": rows })).await;
}

/// One line of the event record: number, when, where, how long, how risky, what.
/// This is where the detail belongs — everything a button label cannot hold.
fn list_line(n: usize, c: &super::evidence::EventCard, cam: &str) -> String {
    let dur = c.duration.as_deref()
        .map(|d| format!(" · {}", tg_esc(d))).unwrap_or_default();
    let badge = c.risk_badge();
    let badge = if badge.is_empty() { String::new() } else { format!("{badge} · ") };
    let what: String = c.label().chars().take(60).collect();
    format!("<b>{n}</b> · {} · {}{}\n     {badge}{}\n",
        tg_esc(&c.ts), tg_esc(cam), dur, tg_esc(&what))
}

/// "Aug 03 21:14" → "21:14". A button shows only what distinguishes it; the
/// full timestamp is on the line the number points at.
fn short_time(ts: &str) -> &str {
    ts.rsplit(' ').next().filter(|s| s.contains(':')).unwrap_or(ts)
}

/// The full identity of an event: when, where, how long, how risky, what was
/// seen. Telegram used to render the first two and drop the rest.
fn caption_plain(c: &super::evidence::EventCard, cam: &str) -> String {
    let mut s = format!("🕐 {} · {}", c.ts, cam);
    if let Some(d) = &c.duration { s.push_str(&format!(" · {d}")); }
    let badge = c.risk_badge();
    if !badge.is_empty() { s.push_str(&format!("\n{badge}")); }
    s.push('\n');
    s.push_str(&c.label());
    s
}

/// THE clip upload path.
///
/// `clip:`, `getclip:` and the agent's own send-clip tag used to be three
/// near-identical copies that disagreed about the oversize case: one told the
/// user to "use 🔗 Share instead", another quietly minted the link itself, the
/// third just said the event had no footage. They all end up here now, and the
/// answer to "too big" is always a link, never an instruction.
pub(super) async fn send_event_clip(
    state: &Arc<AppState>, token: &str, chat_id: &str,
    c: &super::evidence::EventCard, cam: &str,
) {
    send_telegram_action(token, chat_id, "upload_video").await;
    let caption = caption_plain(c, cam);
    let share_row = serde_json::json!([[
        { "text": "🔗 Share link", "callback_data": format!("share:{}", c.id) }
    ]]);

    if let Some(path) = super::clip_export::ensure_event_clip(state, &c.id).await {
        match tokio::fs::read(&path).await {
            Ok(data) if data.len() <= TELEGRAM_MAX_UPLOAD => {
                if send_telegram_video(token, chat_id, data, "clip.mp4",
                        &caption, Some(share_row.clone())).await {
                    return;
                }
            }
            Ok(_) => {
                // Past Telegram's ceiling: mint the link rather than describing it.
                let mins = state.settings.read().await.live_share_default_minutes;
                let msg = match mint_link_bounded(state, "clip", &c.id, mins).await {
                    Ok((url, exp)) => format!("{caption}\n\n🔗 Too large to upload — open it here (works for {}):\n{url}",
                        link_life(exp)),
                    Err(e) => format!("{caption}\n\n{e}"),
                };
                send_telegram(token, chat_id, &msg).await;
                return;
            }
            Err(e) => eprintln!("[Telegram] clip read error: {e}"),
        }
    }

    // No finalized footage yet — the thumbnail is still real evidence, and the
    // caption still says when and where. Never a bare "no footage".
    send_telegram_photo_kb(token, chat_id, c.thumb_bytes(),
        &format!("{caption}\n\n(no recorded clip for this event yet)"), share_row).await;
}

/// Fetch one event as a card and send its footage — the entry point for the
/// `clip:` / `getclip:` buttons.
pub(super) async fn send_event_clip_by_id(
    state: &Arc<AppState>, token: &str, chat_id: &str, event_id: &str,
) {
    match super::evidence::card_for(state, event_id).await {
        Some(c) => {
            let cam = camera_label(state, c.cam).await;
            send_event_clip(state, token, chat_id, &c, &cam).await;
        }
        // `chars`, not a byte slice: a callback payload is untrusted input, and
        // `&s[..8]` panics on a multi-byte boundary — which would kill the poll
        // loop task, not just this reply.
        None => send_telegram(token, chat_id,
            &format!("I no longer have event {}.",
                event_id.chars().take(8).collect::<String>())).await,
    }
}

/// Mint a share link with a HARD deadline, translating failures into the
/// actionable message the user should actually see. The old raw-error path
/// could (a) HANG the whole poll loop when the tunnel provider blocked — the
/// per-update timeout then abandoned the reply, so the user got pure silence —
/// and (b) dump provider internals instead of telling them what to do.
pub(super) async fn mint_link_bounded(
    state: &Arc<AppState>, kind: &str, resource: &str, mins: u32,
) -> Result<(String, i64), String> {
    let fut = crate::share_cmds::mint_share_link(
        state, state.app_handle.clone(), kind.to_string(), resource.to_string(), mins);
    match tokio::time::timeout(Duration::from_secs(45), fut).await {
        Ok(Ok(res)) => Ok((res.url, res.expires_at)),
        Ok(Err(e)) => Err(link_error_guidance(&e)),
        Err(_) => Err("Couldn't create the link right now — try again in a minute.".into()),
    }
}

/// Human lifetime for a link message, from the ACTUAL expiry — a reused link
/// honestly reports its remaining minutes, not the requested window.
fn link_life(expires_at: i64) -> String {
    if expires_at == 0 { return "until the app restarts".into(); }
    let mins = ((expires_at - chrono::Utc::now().timestamp()).max(0) + 59) / 60;
    format!("{mins} min")
}

/// Turn a share-link failure into guidance a non-technical user can act on.
/// The Tailscale one-time consent URL is deliberately surfaced as a tappable
/// link: approving it from the phone right in Telegram completes the setup.
fn link_error_guidance(e: &str) -> String {
    if let Some(url) = e.split_whitespace()
        .find(|w| w.starts_with("https://login.tailscale.com/f/funnel"))
    {
        return format!(
            "🔗 Remote links need a ONE-TIME approval (free, ~30 seconds):\n\n\
             1. Tap this link and sign in to Tailscale:\n{url}\n\
             2. Click “Enable” on that page\n\
             3. Come back and tap 📡 Live again\n\n\
             You only ever do this once."
        );
    }
    if e.contains("isn't installed") || e.contains("not signed in") || e.contains("Tailscale") {
        return format!(
            "Remote links aren't set up yet.\n\n\
             On the Anivar computer: open Settings → 📡 Remote access and follow \
             the steps (about 2 minutes, free). Then tap 📡 Live again.\n\n\
             Details: {e}"
        );
    }
    format!("Couldn't create the link: {e}")
}

/// One full agent turn delivered to Telegram: ask, resolve the reply into
/// evidence, send the prose, then render the evidence. The same three steps the
/// poll loop runs — factored out so `/investigate` and `/brief` cannot drift
/// into a different, poorer rendering than a plain question gets.
async fn run_agent_turn(
    state: &Arc<AppState>, token: &str, chat_id: &str, question: &str, mode: super::Mode,
) {
    send_telegram_action(token, chat_id, "typing").await;
    let reply = match chat_with_agent(state, Vec::new(), question.to_string(), mode).await {
        Ok(r) => r,
        Err(e) => { send_telegram(token, chat_id, &format!("Guardian is unavailable right now.\nError: {e}")).await; return; }
    };
    let (clean, evidence) = super::evidence::resolve(state, &reply).await;
    if !clean.is_empty() { send_telegram(token, chat_id, &clean).await; }
    render_evidence(state, token, chat_id, &evidence).await;
}

/// Handle Telegram slash commands instantly without invoking the model.
///
/// Case is folded on the COMMAND WORD only. Everything after it is the user's
/// own text — a name, a plate, a question — and lowercasing that before it
/// reaches the archive is a quiet loss of information.
pub(super) async fn handle_slash_command(state: &Arc<AppState>, cmd: &str, token: &str, chat_id: &str) -> String {
    let parts: Vec<&str> = cmd.trim().splitn(2, ' ').collect();
    // "/search@GuardianBot foo" — Telegram appends the bot name in groups.
    let head = parts[0].split('@').next().unwrap_or("").to_lowercase();
    match head.as_str() {
        "/start" => {
            // Warm welcome + the control menu, so a new user can act immediately.
            let welcome = "🛡️ <b>Anivar Guardian</b>\n\n\
                I watch your cameras and message you the moment something matters — with a \
                snapshot or a tap-to-play clip.\n\n\
                Use the buttons below to pull footage or tune your alerts. \
                Type <b>/help</b> for every command, or just ask me in plain language.";
            send_telegram_html(token, chat_id, welcome).await;
            let (text, kb) = build_alert_menu(state).await;
            send_telegram_with_keyboard(token, chat_id, &text, kb).await;
            String::new()
        }
        "/help" => {
            // Generated from the single tool registry so /help can never drift from
            // what the agent can actually do (descriptions are static → HTML-safe).
            let body = format!("🛡️ <b>Anivar Guardian</b>\n\n{}", super::tools::help_text());
            send_telegram_html(token, chat_id, &body).await;
            String::new()
        }
        "/status" => {
            let cameras = state.latest_frames.read().await;
            let active: Vec<String> = cameras.keys().map(|k| format!("Camera {}", k)).collect();
            let cam_str = if active.is_empty() { "No cameras active".into() } else { active.join(", ") };
            let settings = state.settings.read().await;
            let agent_ok = settings.agent_enabled;
            let event_count: i64 = sqlx::query_scalar(
                "SELECT COUNT(*) FROM motion_events WHERE started_at > datetime('now','-24 hours')"
            ).fetch_one(&state.db).await.unwrap_or(0);
            let guardian = if agent_ok { "On" } else { "Off" };
            let alerts = alert_level_label(&settings.alert_min_risk);
            let now = Local::now().format("%H:%M %Z").to_string();
            let cam = tg_esc(&cam_str);
            send_telegram_html(token, chat_id, &format!(
                "📊 <b>Status</b>\n\n\
                🎥 Cameras: {cam}\n\
                🤖 Guardian: {guardian}\n\
                🔔 Alerts: {alerts}\n\
                📋 Events (24h): <b>{event_count}</b>\n\
                🕐 {now}"
            )).await;
            String::new()
        }
        "/events" => {
            let n: i64 = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(5).min(20);
            let rows: Vec<(String, f32, Option<f64>, Option<String>)> = sqlx::query_as(
                "SELECT started_at, peak_score, duration_secs, ai_summary FROM motion_events ORDER BY started_at DESC LIMIT ?"
            ).bind(n).fetch_all(&state.db).await.unwrap_or_default();
            if rows.is_empty() { return "No events recorded yet.".into(); }
            let mut html = format!("📋 <b>Last {} events</b>\n", rows.len());
            for (ts, _score, dur, summary) in rows {
                let t = chrono::DateTime::parse_from_rfc3339(&ts)
                    .map(|d| d.with_timezone(&Local).format("%b %d · %H:%M").to_string())
                    .unwrap_or(ts[..16.min(ts.len())].to_string());
                let dur_s = dur.map(|d| format!("  ·  {:.0}s", d)).unwrap_or_default();
                // Parse v2 JSON ai_summary to show clean text (not raw JSON)
                let raw = summary.as_deref().unwrap_or("");
                let text = extract_summary_text(raw);
                let display = if text.is_empty() { "No analysis yet".to_string() }
                    else { text.chars().take(100).collect() };
                html.push_str(&format!("\n<b>{}</b>{}\n{}\n", tg_esc(&t), tg_esc(&dur_s), tg_esc(&display)));
            }
            send_telegram_html(token, chat_id, &html).await;
            String::new()
        }
        "/summary" => {
            // Plain text — the digest is multi-channel (Telegram/Discord/Slack).
            build_daily_digest(state).await
        }
        "/pause" => {
            // Folded here rather than over the whole command, so "2H" still
            // parses without lowercasing every other command's arguments.
            let dur_str = &parts.get(1).unwrap_or(&"1h").trim().to_lowercase();
            let mins: u64 = if dur_str.ends_with('h') {
                dur_str.trim_end_matches('h').parse::<u64>().unwrap_or(1) * 60
            } else {
                dur_str.trim_end_matches('m').parse::<u64>().unwrap_or(60)
            };
            write_memory(&state.db, "alerts_paused_until",
                &(chrono::Utc::now() + chrono::Duration::minutes(mins as i64))
                    .to_rfc3339()).await;
            format!("🔕 Alerts paused for {}. Use /resume to re-enable.", dur_str)
        }
        "/resume" => {
            write_memory(&state.db, "alerts_paused_until", "").await;
            "🔔 Alerts resumed.".into()
        }
        "/menu" | "/settings" => {
            let (text, kb) = build_alert_menu(state).await;
            send_telegram_with_keyboard(token, chat_id, &text, kb).await;
            String::new() // menu sent with keyboard
        }
        "/video" | "/get" | "/footage" => {
            let (text, kb) = build_footage_menu();
            send_telegram_with_keyboard(token, chat_id, &text, kb).await;
            String::new() // footage menu sent with keyboard
        }
        "/people" => {
            let (text, kb) = build_people_menu(state).await;
            send_telegram_with_keyboard(token, chat_id, &text, kb).await;
            String::new()
        }
        "/vehicles" => {
            let (text, kb) = build_vehicles_menu(state, 0).await;
            send_telegram_with_keyboard(token, chat_id, &text, kb).await;
            String::new()
        }
        "/sounds" => {
            let (text, kb) = build_sounds_menu(state, 0).await;
            send_telegram_with_keyboard(token, chat_id, &text, kb).await;
            String::new()
        }
        "/rules" => {
            let body = fmt_alert_rules(&state.db).await;
            send_telegram_html(token, chat_id,
                &format!("📋 <b>Alert rules</b>\n{}", tg_esc(&body))).await;
            String::new()
        }
        // /search is gone: ask in plain language instead. Kept as a redirect
        // rather than an "unknown command", because Telegram clients remember
        // commands and the muscle memory outlives the menu entry.
        "/search" | "/find" => {
            let q = parts.get(1).unwrap_or(&"").trim();
            if q.is_empty() {
                "Just ask me — e.g. “the red van last Tuesday” or “anyone at the door last night”.".into()
            } else {
                run_agent_turn(state, token, chat_id, q, super::Mode::Ask).await;
                String::new()
            }
        }
        // The two deeper modes, reachable from the phone. Telegram used to pass
        // `Mode::Ask` unconditionally, so Investigate and Brief existed only in
        // the desktop app — the same brain, deliberately handicapped on the
        // surface people actually carry.
        "/investigate" => {
            let q = parts.get(1).unwrap_or(&"").trim();
            if q.is_empty() {
                return "Give me something to investigate — e.g. /investigate who was at the door last night".into();
            }
            run_agent_turn(state, token, chat_id, q, super::Mode::Investigate).await;
            String::new()
        }
        "/brief" => {
            let q = parts.get(1).map(|s| s.trim()).filter(|s| !s.is_empty())
                .unwrap_or("brief me on what happened");
            run_agent_turn(state, token, chat_id, q, super::Mode::Brief).await;
            String::new()
        }
        "/snap" | "/snapshot" => {
            let cam_id: u8 = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
            match state.latest_frames.read().await.get(&cam_id).cloned() {
                Some(jpeg) => {
                    send_telegram_photo(token, chat_id, jpeg, &format!("Camera {} — snapshot {}", cam_id, Local::now().format("%H:%M"))).await;
                    String::new() // photo sent, no text needed
                }
                None => format!("No frame from Camera {} — is it active?", cam_id),
            }
        }
        _ => String::new(), // unknown command — let the model handle it
    }
}

/// Build a human-readable 24-hour event digest using Agies-style narrative synthesis.
///
/// Uses the LLM when available to write a natural homeowner narrative:
/// — Suspicious/critical events emphasised first
/// — Routine events summarised briefly
/// — Chronological within each group
/// — No raw data leakage (no clip IDs, timestamps, scores, camera names)
pub(super) async fn build_daily_digest(state: &Arc<AppState>) -> String {
    // Fetch events with their risk levels from the last 24 hours
    let rows: Vec<(String, f32, Option<f64>, Option<String>, Option<String>, Option<String>)> = sqlx::query_as(
        "SELECT me.started_at, me.peak_score, me.duration_secs, me.ai_summary, me.detections, aa.risk_level \
         FROM motion_events me \
         LEFT JOIN agent_alerts aa ON aa.event_id = me.id \
         WHERE me.started_at > datetime('now','-24 hours') \
         ORDER BY me.started_at DESC"
    ).fetch_all(&state.db).await.unwrap_or_default();

    let total = rows.len();
    let date_str = Local::now().format("%b %d").to_string();

    if total == 0 {
        return format!("📊 Daily Summary — {}\n\nAll clear — no events in the past 24 hours. ✅", date_str);
    }

    // Separate suspicious/critical from routine for emphasis ordering
    let high_risk_count = rows.iter()
        .filter(|r| r.5.as_deref().map(|s| s == "suspicious" || s == "critical").unwrap_or(false))
        .count();

    // Build a timeline of events for the LLM — suspicious events first, then routine
    let mut event_lines: Vec<String> = Vec::new();

    // Suspicious/critical first
    for r in rows.iter().filter(|r| r.5.as_deref().map(|s| s == "suspicious" || s == "critical").unwrap_or(false)) {
        let t = chrono::DateTime::parse_from_rfc3339(&r.0)
            .map(|d| d.with_timezone(&Local).format("%H:%M").to_string())
            .unwrap_or_else(|_| "??:??".to_string());
        let risk = r.5.as_deref().unwrap_or("suspicious");
        let summary = extract_summary_text(r.3.as_deref().unwrap_or("Motion detected"));
        event_lines.push(format!("[{t}] [{risk}] {summary}"));
    }

    // Then routine events (normal/monitor)
    for r in rows.iter().filter(|r| !r.5.as_deref().map(|s| s == "suspicious" || s == "critical").unwrap_or(false)) {
        let t = chrono::DateTime::parse_from_rfc3339(&r.0)
            .map(|d| d.with_timezone(&Local).format("%H:%M").to_string())
            .unwrap_or_else(|_| "??:??".to_string());
        let risk = r.5.as_deref().unwrap_or("normal");
        let summary = extract_summary_text(r.3.as_deref().unwrap_or("Motion detected"));
        event_lines.push(format!("[{t}] [{risk}] {summary}"));
    }

    // Try LLM narrative synthesis (Agies narrative synthesis prompt)
    let settings = state.settings.read().await.clone();
    let camera_name = if settings.camera_name.is_empty() { "home camera".to_string() } else { settings.camera_name.clone() };
    let event_feed = event_lines.join("\n");

    // Narrative synthesis is pure text, so a text-only engine does it fine — gate on
    // "is there a usable engine", not on a model name on-device never has.
    if settings.agent_enabled && super::llm::agent_configured(&settings) {
        let system = "You are a home security AI assistant. Summarize camera events naturally for the homeowner. \
            Do NOT dump raw data or timestamps — write a clear, human narrative. \
            Emphasise suspicious or concerning events first. Group similar routine events together. \
            Keep the summary under 200 words. Use simple, reassuring language for routine days.";
        let user = format!(
            "Here are today's security camera events for {camera_name} ({date_str}):\n\n{event_feed}\n\n\
            Write a brief natural narrative summary for the homeowner. \
            Start with any concerning events, then summarise the routine activity."
        );

        if let Ok(narrative) = call_llm(&settings, system, &user, None, false).await {
            let narrative = narrative.trim().to_string();
            if !narrative.is_empty() && narrative.len() > 20 {
                return format!(
                    "📊 Daily Summary — {date_str}\n\n{narrative}\n\n\
                    — {total} events · {high_risk_count} requiring attention"
                );
            }
        }
    }

    // Fallback: structured digest (no LLM)
    let most_active = {
        let mut hour_counts = [0usize; 24];
        for r in &rows {
            if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(&r.0) {
                use chrono::Timelike;
                hour_counts[dt.hour() as usize] += 1;
            }
        }
        hour_counts.iter().enumerate().max_by_key(|(_, &c)| c).map(|(h, _)| h).unwrap_or(0)
    };

    let mut msg = format!(
        "📊 Daily Summary — {date_str}\n\n\
        📋 Total events: {total}\n\
        ⚠️ Requiring attention: {high_risk_count}\n\
        🕐 Peak activity: {most_active:02}:00–{:02}:00\n",
        (most_active + 1) % 24
    );

    // Notable events (suspicious/critical)
    let notable: Vec<_> = rows.iter()
        .filter(|r| r.5.as_deref().map(|s| s == "suspicious" || s == "critical").unwrap_or(false))
        .take(5).collect();
    if !notable.is_empty() {
        msg.push_str("\nFlagged events:\n");
        for r in notable {
            let t = chrono::DateTime::parse_from_rfc3339(&r.0)
                .map(|d| d.with_timezone(&Local).format("%H:%M").to_string()).unwrap_or_default();
            let emoji = risk_to_emoji(r.5.as_deref().unwrap_or("suspicious"));
            let sum = extract_summary_text(r.3.as_deref().unwrap_or("Motion detected"));
            msg.push_str(&format!("{emoji} {t} — {}\n", sum.chars().take(80).collect::<String>()));
        }
    }
    msg
}

/// Send the daily digest to all configured channels.
pub async fn send_daily_digest(state: &Arc<AppState>) {
    let settings = state.settings.read().await.clone();
    let digest   = build_daily_digest(state).await;

    if !settings.telegram_bot_token.is_empty() && !settings.telegram_chat_id.is_empty() {
        send_telegram(&settings.telegram_bot_token, &settings.telegram_chat_id, &digest).await;
    }
}

// ─── Telegram alert-settings menu (user-initiated config; NOT an LLM tool) ──────

/// Register the bot's command menu (the "/" autocomplete + the Menu button) so
/// the commands are discoverable. Called once when the loop has a token.
async fn register_bot_commands(client: &reqwest::Client, token: &str) {
    let url = format!("https://api.telegram.org/bot{}/setMyCommands", token);
    let body = serde_json::json!({ "commands": [
        { "command": "menu",    "description": "Open the control menu" },
        { "command": "video",   "description": "Get a clip or the live view" },
        { "command": "people",  "description": "Enrolled people & sightings" },
        { "command": "vehicles","description": "Recognized vehicles & plates" },
        { "command": "sounds",  "description": "Recent sounds heard" },
        { "command": "snap",    "description": "Live snapshot from a camera" },
        { "command": "status",  "description": "Camera & system status" },
        { "command": "events",  "description": "Recent motion events" },
        { "command": "summary", "description": "Last 24 hours, summarised" },
        { "command": "investigate", "description": "Sweep the archive and follow leads" },
        { "command": "brief",   "description": "What happened, summarised" },
        { "command": "rules",   "description": "Active alert rules" },
        { "command": "pause",   "description": "Pause alerts (e.g. /pause 1h)" },
        { "command": "resume",  "description": "Resume alerts" },
        { "command": "help",    "description": "All commands" },
    ]});
    let _ = client.post(&url).json(&body).send().await;
}

fn alert_level_label(min: &str) -> &'static str {
    match min.to_lowercase().as_str() {
        "off" => "Off", "critical" => "Critical", "suspicious" => "Suspicious", _ => "All",
    }
}
/// Cycle Off → Critical → Suspicious → All → Off.
fn next_alert_level(min: &str) -> &'static str {
    match min.to_lowercase().as_str() {
        "off" => "critical", "critical" => "suspicious", "suspicious" => "normal", _ => "off",
    }
}

/// Build the main alert-settings menu text + inline keyboard reflecting current settings.
pub(super) async fn build_alert_menu(state: &Arc<AppState>) -> (String, serde_json::Value) {
    let s = state.settings.read().await.clone();
    let on = |b: bool| if b { "On" } else { "Off" };
    let text = "🛡 Anivar — browse footage, people, vehicles & sounds, or tune what alerts you.".to_string();
    // No search button, and no /search command.
    //
    // Searching was a button that asked a question that ran a query — a menu
    // reimplementation of the thing you can already do by typing "the red van
    // last Tuesday" at the bot. The agent answers that with the same rows, the
    // same pictures and the same tap-for-the-clip grid, and it understands
    // "last Tuesday" and "the red van" rather than matching them as keywords.
    // The menu is for BROWSING, which is the part plain language is bad at.
    let kb = serde_json::json!([
        // ── Browse ──
        [{ "text": "📹 Footage & live view", "callback_data": "cfg:footage" }],
        [
            { "text": "👤 People",   "callback_data": "cfg:people" },
            { "text": "🚗 Vehicles", "callback_data": "cfg:vehicles" },
        ],
        [
            { "text": "🔊 Sounds",       "callback_data": "cfg:sounds" },
            { "text": "👁 Alert filter", "callback_data": "cfg:see" },
        ],
        // ── Alert settings ──
        [{ "text": format!("🔔 Alert level · {}", alert_level_label(&s.alert_min_risk)), "callback_data": "cfg:level" }],
        [
            { "text": format!("📸 Snapshots · {}", on(s.attach_snapshot_to_alerts)), "callback_data": "cfg:snap" },
            { "text": format!("🎬 Clips · {}", on(s.attach_clip_to_alerts)), "callback_data": "cfg:clip" },
        ],
        [
            { "text": format!("🌙 Quiet hours · {}", on(s.quiet_hours_enabled)), "callback_data": "cfg:quiet" },
            { "text": "🎥 Cameras", "callback_data": "cfg:cams" },
        ],
    ]);
    (text, kb)
}

/// The 👁 Alert-filter menu: per-category mute toggles. ⬜ = muted (events still
/// record + analyze; the user just isn't messaged). Synced with the app's
/// Settings alerts section — both edit `settings.alert_muted_categories`.
const ALERT_CATEGORIES: [(&str, &str); 5] = [
    ("person",  "👤 People"),
    ("vehicle", "🚗 Vehicles"),
    ("animal",  "🐾 Animals"),
    ("audio",   "🔊 Sounds"),
    ("other",   "📦 Other motion"),
];

async fn build_categories_menu(state: &Arc<AppState>) -> (String, serde_json::Value) {
    let s = state.settings.read().await.clone();
    let mut rows: Vec<serde_json::Value> = Vec::new();
    for (key, label) in ALERT_CATEGORIES {
        let muted = s.alert_muted_categories.iter().any(|c| c == key);
        rows.push(serde_json::json!([
            { "text": format!("{} {}", if muted { "⬜" } else { "✅" }, label),
              "callback_data": format!("cfgcat:{key}") }
        ]));
    }
    rows.push(serde_json::json!([{ "text": "‹ Back", "callback_data": "cfg:back" }]));
    ("👁 What alerts you — ✅ sends messages · ⬜ muted (still recorded, just not messaged). Tap to toggle."
        .to_string(), serde_json::Value::Array(rows))
}

/// Build the per-camera sub-menu (✅ = alerts on, ⬜ = muted).
async fn build_camera_menu(state: &Arc<AppState>) -> (String, serde_json::Value) {
    let s = state.settings.read().await.clone();
    let cams: Vec<(i64, String)> = sqlx::query_as(
        "SELECT cam_id, name FROM camera_configs WHERE enabled=1 ORDER BY cam_id ASC"
    ).fetch_all(&state.db).await.unwrap_or_default();
    let mut rows: Vec<serde_json::Value> = Vec::new();
    if cams.is_empty() {
        rows.push(serde_json::json!([{ "text": "No cameras configured", "callback_data": "cfg:back" }]));
    } else {
        for (id, name) in cams {
            let muted = s.alert_disabled_cameras.contains(&(id as u8));
            let nm = if name.trim().is_empty() { format!("CAM {}", id + 1) } else { name };
            rows.push(serde_json::json!([
                { "text": format!("{} {}", if muted { "⬜" } else { "✅" }, nm),
                  "callback_data": format!("cfgcam:{}", id) }
            ]));
        }
    }
    rows.push(serde_json::json!([{ "text": "‹ Back", "callback_data": "cfg:back" }]));
    ("🎥 Cameras — ✅ alerts on · ⬜ muted. Tap one to toggle.".to_string(), serde_json::Value::Array(rows))
}

/// Footage picker — choose a recorded event clip or the live view.
fn build_footage_menu() -> (String, serde_json::Value) {
    let kb = serde_json::json!([
        [{ "text": "📡 Live view", "callback_data": "cfg:live" }],
        [
            { "text": "🎬 Recent clips",  "callback_data": "cfg:events" },
            { "text": "📅 Browse by day", "callback_data": "cal:" },
        ],
        [{ "text": "‹ Back", "callback_data": "cfg:back" }],
    ]);
    ("📹 Footage — a live view, recent clips, or browse the archive by day.".to_string(), kb)
}

/// Inclusive UTC RFC3339 bounds `[start, end)` for one LOCAL calendar day, so the
/// day shown in Telegram matches the user's wall clock (events are stored UTC).
fn local_day_utc_bounds(date: NaiveDate) -> (String, String) {
    // Local midnight → next local midnight, each mapped to UTC. `unwrap_or` guards
    // the (rare) DST-gap where local midnight doesn't exist; single() picks the
    // unambiguous instant, else we fall back to the date at 00:00 UTC.
    let start_local = Local.from_local_datetime(&date.and_hms_opt(0, 0, 0).unwrap()).single();
    let end_local   = Local.from_local_datetime(&(date + ChronoDuration::days(1)).and_hms_opt(0, 0, 0).unwrap()).single();
    let start = start_local.map(|d| d.with_timezone(&Utc)).unwrap_or_else(|| Utc.from_utc_datetime(&date.and_hms_opt(0,0,0).unwrap()));
    let end   = end_local.map(|d| d.with_timezone(&Utc)).unwrap_or_else(|| start + ChronoDuration::days(1));
    (start.to_rfc3339(), end.to_rfc3339())
}

/// The calendar flow serves three browsers: all footage, 🚗 vehicles and 🔊
/// sounds. The category rides in the callback payload as a `v:`/`s:` prefix
/// (`cal:v:2026-07`, `calday:s:2026-07-14`); no prefix = all footage, so the
/// original grammar keeps working. Returns (category, rest-of-payload).
fn cal_cat(payload: &str) -> (Option<&'static str>, &str) {
    if let Some(rest) = payload.strip_prefix("v:") { (Some("vehicle"), rest) }
    else if let Some(rest) = payload.strip_prefix("s:") { (Some("audio"), rest) }
    else { (None, payload) }
}
fn cal_pfx(cat: Option<&str>) -> &'static str {
    match cat { Some("vehicle") => "v:", Some("audio") => "s:", _ => "" }
}
/// SQL clause narrowing motion_events to the calendar's category ("" = all).
fn cal_cat_where(cat: Option<&str>) -> &'static str {
    match cat {
        Some("vehicle") => " AND event_category='vehicle'",
        Some("audio")   => " AND event_category='audio'",
        _ => "",
    }
}

/// Calendar month grid (standard "browse by day"). Days that HAVE events are
/// tappable (`calday:[v:|s:]YYYY-MM-DD`); empty/padding cells are inert (`cal:noop`).
/// `‹`/`›` switch month (`cal:[v:|s:]YYYY-MM`). One DB query over the month's UTC
/// span, bucketed to LOCAL dates so day boundaries are correct.
async fn build_calendar_menu(state: &Arc<AppState>, year: i32, month: u32, cat: Option<&str>) -> (String, serde_json::Value) {
    let first = NaiveDate::from_ymd_opt(year, month, 1).unwrap_or_else(|| Local::now().date_naive().with_day(1).unwrap());
    let (year, month) = (first.year(), first.month());
    let next_month_first = if month == 12 { NaiveDate::from_ymd_opt(year + 1, 1, 1) } else { NaiveDate::from_ymd_opt(year, month + 1, 1) }
        .unwrap_or(first);
    let days_in_month = (next_month_first - first).num_days() as u32;

    // Which local days in this month have events — query the whole month's UTC span once.
    let (mstart, _) = local_day_utc_bounds(first);
    let (_, mend)   = local_day_utc_bounds(next_month_first - ChronoDuration::days(1));
    let sql = format!(
        "SELECT started_at FROM motion_events WHERE started_at >= ? AND started_at < ?{}",
        cal_cat_where(cat));
    let rows: Vec<(String,)> = sqlx::query_as(&sql)
        .bind(&mstart).bind(&mend).fetch_all(&state.db).await.unwrap_or_default();
    let mut have: std::collections::HashSet<u32> = std::collections::HashSet::new();
    for (s,) in rows {
        if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(&s) {
            let local = dt.with_timezone(&Local).date_naive();
            if local.year() == year && local.month() == month { have.insert(local.day()); }
        }
    }

    let pfx = cal_pfx(cat);
    let mut kb: Vec<serde_json::Value> = Vec::new();
    // Header: ‹ prev · Month Year · next ›
    let prev = if month == 1 { format!("{}-12", year - 1) } else { format!("{}-{:02}", year, month - 1) };
    let next = if month == 12 { format!("{}-01", year + 1) } else { format!("{}-{:02}", year, month + 1) };
    let title = format!("{} {}", month_name(month), year);
    kb.push(serde_json::json!([
        { "text": "‹", "callback_data": format!("cal:{pfx}{prev}") },
        { "text": title.clone(), "callback_data": "cal:noop" },
        { "text": "›", "callback_data": format!("cal:{pfx}{next}") },
    ]));
    // Weekday labels (Mon-first to match most of the world; cosmetic).
    kb.push(serde_json::json!([
        {"text":"M","callback_data":"cal:noop"},{"text":"T","callback_data":"cal:noop"},
        {"text":"W","callback_data":"cal:noop"},{"text":"T","callback_data":"cal:noop"},
        {"text":"F","callback_data":"cal:noop"},{"text":"S","callback_data":"cal:noop"},
        {"text":"S","callback_data":"cal:noop"},
    ]));
    // Leading blanks: Monday=0 offset.
    let lead = first.weekday().num_days_from_monday();
    let mut week: Vec<serde_json::Value> = Vec::new();
    for _ in 0..lead { week.push(serde_json::json!({"text":" ","callback_data":"cal:noop"})); }
    for d in 1..=days_in_month {
        if have.contains(&d) {
            week.push(serde_json::json!({ "text": format!("·{d}·"), "callback_data": format!("calday:{pfx}{}-{:02}-{:02}", year, month, d) }));
        } else {
            week.push(serde_json::json!({ "text": d.to_string(), "callback_data": "cal:noop" }));
        }
        if week.len() == 7 { kb.push(serde_json::Value::Array(std::mem::take(&mut week))); }
    }
    if !week.is_empty() {
        while week.len() < 7 { week.push(serde_json::json!({"text":" ","callback_data":"cal:noop"})); }
        kb.push(serde_json::Value::Array(week));
    }
    let back = match cat { Some("vehicle") => "cfg:vehicles", Some("audio") => "cfg:sounds", _ => "cfg:footage" };
    kb.push(serde_json::json!([{ "text": "‹ Back", "callback_data": back }]));
    let what = match cat { Some("vehicle") => "🚗 vehicle sightings", Some("audio") => "🔊 sounds", _ => "footage" };
    (format!("📅 {title} — tap a ·highlighted· day with {what}."), serde_json::Value::Array(kb))
}

/// Parse a "YYYY-MM" calendar payload → (year, month). None for empty/invalid.
fn parse_year_month(s: &str) -> Option<(i32, u32)> {
    let (y, m) = s.split_once('-')?;
    let year: i32 = y.parse().ok()?;
    let month: u32 = m.parse().ok()?;
    if (1..=12).contains(&month) { Some((year, month)) } else { None }
}

fn month_name(m: u32) -> &'static str {
    ["", "January", "February", "March", "April", "May", "June",
     "July", "August", "September", "October", "November", "December"]
        .get(m as usize).copied().unwrap_or("")
}

/// Day-list filters (whitelists — these interpolate into SQL): vehicle types
/// match dominant_label; sound groups reuse the audio category patterns.
const CAL_VEH_FILTERS: [(&str, &str); 5] =
    [("car", "Car"), ("truck", "Truck"), ("bus", "Bus"), ("motorcycle", "Moto"), ("bicycle", "Bike")];
const CAL_SND_FILTERS: [(&str, &str); 5] =
    [("high_pitch", "High-pitch"), ("human", "Human"), ("alarm", "Alarms"), ("animal", "Animals"), ("impact", "Impact")];

/// SQL clause for a day-list filter (whitelisted; "" when absent/unknown).
fn cal_filter_where(cat: Option<&str>, filter: Option<&str>) -> String {
    match (cat, filter) {
        (Some("vehicle"), Some(f)) if CAL_VEH_FILTERS.iter().any(|(k, _)| *k == f) =>
            format!(" AND dominant_label='{f}'"),
        (Some("audio"), Some("high_pitch")) =>
            " AND json_extract(audio_meta, '$.high_pitch') = 1".to_string(),
        (Some("audio"), Some(f)) => match crate::audio::category_label_patterns(f) {
            Some(pats) => {
                let ors: Vec<String> = pats.iter()
                    .map(|p| format!("LOWER(dominant_label) LIKE '%{p}%'")).collect();
                format!(" AND ({})", ors.join(" OR "))
            }
            None => String::new(),
        },
        _ => String::new(),
    }
}

/// One local day's events as tappable clip buttons (newest first, paginated). Shows
/// EVERY event — not only those whose clip is already on disk — because `getclip`
/// generates the clip on tap; the old `clip_path IS NOT NULL` filter hid fresh events.
/// Vehicle/sound calendars add a filter row; the active filter rides in the payload
/// after `~` (`calday:v:2026-07-14~car`, `calp:v:2026-07-14~car:2`).
async fn build_day_events_menu(
    state: &Arc<AppState>, date_str: &str, page: u32,
    cat: Option<&str>, filter: Option<&str>,
) -> (String, serde_json::Value) {
    const PER_PAGE: i64 = 8;
    let date = NaiveDate::parse_from_str(date_str, "%Y-%m-%d").unwrap_or_else(|_| Local::now().date_naive());
    let (start, end) = local_day_utc_bounds(date);
    let pfx = cal_pfx(cat);
    let where_extra = format!("{}{}", cal_cat_where(cat), cal_filter_where(cat, filter));
    let count_sql = format!(
        "SELECT COUNT(*) FROM motion_events WHERE started_at >= ? AND started_at < ?{where_extra}");
    let total: i64 = sqlx::query_scalar(&count_sql)
        .bind(&start).bind(&end).fetch_one(&state.db).await.unwrap_or(0);
    let pages = ((total + PER_PAGE - 1) / PER_PAGE).max(1);
    let page = (page as i64).min(pages - 1).max(0);
    let list_sql = format!(
        "SELECT id, started_at, dominant_label, event_category, recognized_plate, loudness_db
           FROM motion_events
          WHERE started_at >= ? AND started_at < ?{where_extra}
          ORDER BY started_at DESC LIMIT ? OFFSET ?");
    let rows_db: Vec<(String, String, Option<String>, Option<String>, Option<String>, Option<f64>)> =
        sqlx::query_as(&list_sql)
            .bind(&start).bind(&end).bind(PER_PAGE).bind(page * PER_PAGE)
            .fetch_all(&state.db).await.unwrap_or_default();

    let mut kb: Vec<serde_json::Value> = Vec::new();
    // Filter row (vehicle/sound calendars): All + the category's groups; the
    // active one is checked. Each button re-renders this day filtered.
    if let Some(filters) = match cat {
        Some("vehicle") => Some(&CAL_VEH_FILTERS), Some("audio") => Some(&CAL_SND_FILTERS), _ => None,
    } {
        let mut row: Vec<serde_json::Value> = vec![serde_json::json!(
            { "text": if filter.is_none() { "✓ All" } else { "All" }.to_string(),
              "callback_data": format!("calday:{pfx}{date_str}") })];
        for (k, label) in filters.iter() {
            let text = if filter == Some(*k) { format!("✓ {label}") } else { (*label).to_string() };
            row.push(serde_json::json!({ "text": text, "callback_data": format!("calday:{pfx}{date_str}~{k}") }));
            if row.len() == 3 { kb.push(serde_json::Value::Array(std::mem::take(&mut row))); }
        }
        if !row.is_empty() { kb.push(serde_json::Value::Array(row)); }
    }
    if rows_db.is_empty() {
        kb.push(serde_json::json!([{ "text": if filter.is_some() { "No matching events that day" } else { "No events that day" }, "callback_data": "cal:noop" }]));
    } else {
        for (id, started_at, dom, ecat, plate, loud) in rows_db {
            let hhmm = chrono::DateTime::parse_from_rfc3339(&started_at)
                .map(|d| d.with_timezone(&Local).format("%H:%M").to_string())
                .unwrap_or_else(|_| started_at.get(11..16).unwrap_or("--:--").to_string());
            let label = dom.filter(|s| !s.is_empty())
                .or(ecat.filter(|s| !s.is_empty() && s != "other"))
                .unwrap_or_else(|| "motion".to_string());
            // Category-flavored rows: plates on vehicle days, loudness on sound days.
            let (icon, extra) = match cat {
                Some("vehicle") => ("🚗", plate.filter(|p| !p.is_empty()).map(|p| format!(" · {p}")).unwrap_or_default()),
                Some("audio")   => ("🔊", loud.map(|d| format!(" · {d:.0} dB")).unwrap_or_default()),
                _ => ("🎬", String::new()),
            };
            kb.push(serde_json::json!([
                { "text": format!("{icon} {hhmm} · {label}{extra}"), "callback_data": format!("getclip:{}", id) }
            ]));
        }
    }
    // Pager row — carries the filter so pages stay filtered.
    let fsuf = filter.map(|f| format!("~{f}")).unwrap_or_default();
    if pages > 1 {
        let mut nav: Vec<serde_json::Value> = Vec::new();
        if page > 0 { nav.push(serde_json::json!({ "text": "‹ Prev", "callback_data": format!("calp:{pfx}{date_str}{fsuf}:{}", page - 1) })); }
        nav.push(serde_json::json!({ "text": format!("{}/{}", page + 1, pages), "callback_data": "cal:noop" }));
        if page < pages - 1 { nav.push(serde_json::json!({ "text": "Next ›", "callback_data": format!("calp:{pfx}{date_str}{fsuf}:{}", page + 1) })); }
        kb.push(serde_json::Value::Array(nav));
    }
    kb.push(serde_json::json!([{ "text": "‹ Back to calendar", "callback_data": format!("cal:{pfx}{}-{:02}", date.year(), date.month()) }]));
    let title = date.format("%a %b %-d, %Y").to_string();
    let what = match cat { Some("vehicle") => "vehicle sighting", Some("audio") => "sound", _ => "event" };
    (format!("📋 {title} — {total} {what}{} · tap one for its clip.", if total == 1 { "" } else { "s" }), serde_json::Value::Array(kb))
}

/// Recent events that HAVE a recorded clip, as tappable buttons. Tapping sends
/// that event's video (getclip:<id>).
async fn build_events_list_menu(state: &Arc<AppState>) -> (String, serde_json::Value) {
    // Show ALL recent events — not only those whose clip is already on disk. Clips
    // are generated lazily, so the old `clip_path IS NOT NULL` filter hid brand-new
    // events ("events don't update"); `getclip` builds the clip on tap regardless.
    let rows_db: Vec<(String, String, Option<String>, Option<String>)> = sqlx::query_as(
        "SELECT id, started_at, dominant_label, event_category FROM motion_events
         ORDER BY started_at DESC LIMIT 8"
    ).fetch_all(&state.db).await.unwrap_or_default();
    let mut rows: Vec<serde_json::Value> = Vec::new();
    if rows_db.is_empty() {
        rows.push(serde_json::json!([{ "text": "No recorded clips yet", "callback_data": "cfg:footage" }]));
    } else {
        for (id, started_at, dom, cat) in rows_db {
            let hhmm = chrono::DateTime::parse_from_rfc3339(&started_at)
                .map(|d| d.with_timezone(&Local).format("%H:%M").to_string())
                .unwrap_or_else(|_| started_at.get(11..16).unwrap_or("--:--").to_string());
            let label = dom.filter(|s| !s.is_empty())
                .or(cat.filter(|s| !s.is_empty() && s != "other"))
                .unwrap_or_else(|| "motion".to_string());
            rows.push(serde_json::json!([
                { "text": format!("🎬 {hhmm} · {label}"), "callback_data": format!("getclip:{}", id) }
            ]));
        }
    }
    rows.push(serde_json::json!([{ "text": "‹ Back", "callback_data": "cfg:footage" }]));
    ("🎬 Recent clips — tap one to play it here.".to_string(), serde_json::Value::Array(rows))
}

/// Live-view picker — one button per enabled camera. Tapping sends a snapshot +
/// a live share link (getlive:<cam_id>).
async fn build_live_menu(state: &Arc<AppState>) -> (String, serde_json::Value) {
    let cams: Vec<(i64, String)> = sqlx::query_as(
        "SELECT cam_id, name FROM camera_configs WHERE enabled=1 ORDER BY cam_id ASC"
    ).fetch_all(&state.db).await.unwrap_or_default();
    let mut rows: Vec<serde_json::Value> = Vec::new();
    if cams.is_empty() {
        rows.push(serde_json::json!([{ "text": "No cameras configured", "callback_data": "cfg:footage" }]));
    } else {
        for (id, name) in cams {
            let nm = if name.trim().is_empty() { format!("CAM {}", id + 1) } else { name };
            rows.push(serde_json::json!([
                { "text": format!("📡 {nm}"), "callback_data": format!("getlive:{}", id) }
            ]));
        }
    }
    rows.push(serde_json::json!([{ "text": "‹ Back", "callback_data": "cfg:footage" }]));
    ("📡 Live view — tap a camera for a snapshot + live link.".to_string(), serde_json::Value::Array(rows))
}

// ─── People / Vehicles / Sounds menus ────────────────────────────────────────

/// Human "2h ago"-style relative time. Handles BOTH timestamp shapes in the DB:
/// RFC3339 ("2026-07-09T19:31:07+00:00") and SQLite datetime('now') ("2026-07-09
/// 19:31:07", UTC with a space) — known_persons.last_seen_at mixes them.
fn rel_time(ts: &str) -> String {
    let parsed = chrono::DateTime::parse_from_rfc3339(ts)
        .map(|d| d.with_timezone(&Utc))
        .or_else(|_| chrono::NaiveDateTime::parse_from_str(ts, "%Y-%m-%d %H:%M:%S")
            .map(|n| n.and_utc()));
    let Ok(t) = parsed else { return "unknown".into() };
    let secs = (Utc::now() - t).num_seconds().max(0);
    match secs {
        0..=59        => "just now".into(),
        60..=3599     => format!("{}m ago", secs / 60),
        3600..=86399  => format!("{}h ago", secs / 3600),
        _             => format!("{}d ago", secs / 86400),
    }
}

/// 👤 People — the enrolled roster, freshest first. Tap → photo + stats card.
async fn build_people_menu(state: &Arc<AppState>) -> (String, serde_json::Value) {
    let rows_db: Vec<(String, String, Option<String>)> = sqlx::query_as(
        "SELECT id, name, last_seen_at FROM known_persons
          ORDER BY COALESCE(last_seen_at,'') DESC, name ASC LIMIT 20"
    ).fetch_all(&state.db).await.unwrap_or_default();
    let empty = rows_db.is_empty();
    let mut rows: Vec<serde_json::Value> = Vec::new();
    if empty {
        rows.push(serde_json::json!([{ "text": "No people enrolled yet", "callback_data": "cal:noop" }]));
    } else {
        for (id, name, last) in rows_db {
            let seen = last.as_deref().map(rel_time).unwrap_or_else(|| "never seen".into());
            rows.push(serde_json::json!([
                { "text": format!("👤 {name} · {seen}"), "callback_data": format!("pers:{id}") }
            ]));
        }
    }
    rows.push(serde_json::json!([{ "text": "‹ Back", "callback_data": "cfg:back" }]));
    let text = if empty {
        "👤 People — nobody enrolled yet. Enroll faces in the app: People → Enroll.".to_string()
    } else {
        "👤 People — tap someone for their photo, pattern and recent sightings.".to_string()
    };
    (text, serde_json::Value::Array(rows))
}

/// Photo + caption + inline keyboard (sendPhoto with reply_markup). Falls back
/// to a plain text message when the photo bytes are missing/undecodable.
pub(super) async fn send_telegram_photo_kb(
    bot_token: &str, chat_id: &str, jpeg: Option<Vec<u8>>, caption: &str,
    keyboard: serde_json::Value,
) {
    if bot_token.is_empty() || chat_id.is_empty() { return; }
    if let Some(jpeg) = jpeg.filter(|j| !j.is_empty()) {
        let url = format!("https://api.telegram.org/bot{}/sendPhoto", bot_token);
        if let Ok(part) = reqwest::multipart::Part::bytes(jpeg)
            .file_name("person.jpg").mime_str("image/jpeg")
        {
            let form = reqwest::multipart::Form::new()
                .text("chat_id", chat_id.to_string())
                .text("caption", caption.to_string())
                .text("reply_markup", serde_json::json!({ "inline_keyboard": keyboard }).to_string())
                .part("photo", part);
            let ok = reqwest::Client::new().post(&url)
                .timeout(Duration::from_secs(30))
                .multipart(form).send().await
                .map(|r| r.status().is_success()).unwrap_or(false);
            if ok { return; }
        }
    }
    // Text-only fallback (no/bad photo or upload failure) — never silence.
    send_telegram_with_keyboard(bot_token, chat_id, caption, keyboard).await;
}

/// 👤 person detail: photo + role/last-seen/30-day pattern + sightings button.
async fn send_person_detail(state: &Arc<AppState>, token: &str, chat_id: &str, person_id: &str) {
    let row: Option<(String, String, Option<String>, Option<String>)> = sqlx::query_as(
        "SELECT name, role, last_seen_at, thumbnail FROM known_persons WHERE id=?"
    ).bind(person_id).fetch_optional(&state.db).await.ok().flatten();
    let Some((name, role, last_seen, thumb)) = row else {
        // Stale button — the person was deleted in the app since the menu rendered.
        send_telegram(token, chat_id, "That person no longer exists — send /people for the current roster.").await;
        return;
    };

    // 30-day pattern from face_sightings (same aggregation the app's roster uses).
    let sights: Vec<(String, i64)> = sqlx::query_as(
        "SELECT seen_at, camera_id FROM face_sightings
          WHERE person_name=? AND seen_at > datetime('now','-30 days')
          ORDER BY seen_at DESC LIMIT 5000"
    ).bind(&name).fetch_all(&state.db).await.unwrap_or_default();
    let n = sights.len();
    let mut days: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut cams: std::collections::HashSet<i64> = std::collections::HashSet::new();
    let mut hours = [0u32; 24];
    for (seen, cam) in &sights {
        days.insert(seen.chars().take(10).collect());
        cams.insert(*cam);
        let local = chrono::DateTime::parse_from_rfc3339(seen)
            .map(|d| d.with_timezone(&Local))
            .or_else(|_| chrono::NaiveDateTime::parse_from_str(seen, "%Y-%m-%d %H:%M:%S")
                .map(|nv| nv.and_utc().with_timezone(&Local)));
        if let Ok(t) = local { hours[chrono::Timelike::hour(&t) as usize] += 1; }
    }
    let peak = if n >= 3 {
        hours.iter().enumerate().max_by_key(|(_, c)| **c)
            .map(|(h, _)| format!(" · usually ~{h:02}:00"))
            .unwrap_or_default()
    } else { String::new() };
    let mut cam_list: Vec<i64> = cams.into_iter().collect();
    cam_list.sort_unstable();
    let cam_str = if cam_list.is_empty() { String::new() }
        else { format!(" · cam {}", cam_list.iter().map(|c| (c + 1).to_string()).collect::<Vec<_>>().join(", ")) };
    let seen_line = last_seen.as_deref().map(rel_time).unwrap_or_else(|| "never".into());
    let pattern = if n == 0 { "No sightings in the last 30 days.".to_string() }
        else { format!("{n} sighting{} over {} day{}{peak}{cam_str} (30d)",
            if n == 1 { "" } else { "s" }, days.len(), if days.len() == 1 { "" } else { "s" }) };
    // Plain text caption (no parse_mode) — names are user-supplied.
    let caption = format!("👤 {name} · {role}\nLast seen: {seen_line}\n{pattern}");

    let kb = serde_json::json!([
        [{ "text": "🎬 Recent sightings", "callback_data": format!("persev:{person_id}:0") }],
        [{ "text": "‹ Back to people",    "callback_data": "cfg:people" }],
    ]);
    // known_persons.thumbnail is inline base64 (data-URI or bare); resolve handles refs.
    let jpeg = thumb
        .map(|t| crate::blobstore::resolve(&state.data_dir, &t))
        .map(|t| t.trim_start_matches("data:image/jpeg;base64,").to_string())
        .and_then(|t| base64::Engine::decode(&base64::engine::general_purpose::STANDARD, t.trim()).ok());
    send_telegram_photo_kb(token, chat_id, jpeg, &caption, kb).await;
}

/// 🎬 one person's recent events (via face_sightings→event_id), paginated.
async fn build_person_events_menu(state: &Arc<AppState>, person_id: &str, page: u32) -> (String, serde_json::Value) {
    const PER_PAGE: i64 = 6;
    let name: Option<String> = sqlx::query_scalar("SELECT name FROM known_persons WHERE id=?")
        .bind(person_id).fetch_optional(&state.db).await.ok().flatten();
    let Some(name) = name else {
        return ("That person no longer exists — send /people for the current roster.".to_string(),
            serde_json::json!([[{ "text": "‹ Back", "callback_data": "cfg:people" }]]));
    };
    let total: i64 = sqlx::query_scalar(
        "SELECT COUNT(DISTINCT event_id) FROM face_sightings WHERE person_name=? AND event_id IS NOT NULL"
    ).bind(&name).fetch_one(&state.db).await.unwrap_or(0);
    let pages = ((total + PER_PAGE - 1) / PER_PAGE).max(1);
    let page = (page as i64).min(pages - 1).max(0);
    let rows_db: Vec<(String, String)> = sqlx::query_as(
        "SELECT e.id, e.started_at FROM motion_events e
          WHERE e.id IN (SELECT DISTINCT event_id FROM face_sightings WHERE person_name=? AND event_id IS NOT NULL)
          ORDER BY e.started_at DESC LIMIT ? OFFSET ?"
    ).bind(&name).bind(PER_PAGE).bind(page * PER_PAGE)
     .fetch_all(&state.db).await.unwrap_or_default();

    let mut kb: Vec<serde_json::Value> = Vec::new();
    if rows_db.is_empty() {
        kb.push(serde_json::json!([{ "text": "No recorded sightings yet", "callback_data": "cal:noop" }]));
    } else {
        for (id, started) in rows_db {
            let when = chrono::DateTime::parse_from_rfc3339(&started)
                .map(|d| d.with_timezone(&Local).format("%b %-d · %H:%M").to_string())
                .unwrap_or_else(|_| started.get(..16).unwrap_or("?").to_string());
            kb.push(serde_json::json!([
                { "text": format!("🎬 {when}"), "callback_data": format!("getclip:{id}") }
            ]));
        }
    }
    if pages > 1 {
        let mut nav: Vec<serde_json::Value> = Vec::new();
        if page > 0 { nav.push(serde_json::json!({ "text": "‹ Prev", "callback_data": format!("persev:{person_id}:{}", page - 1) })); }
        nav.push(serde_json::json!({ "text": format!("{}/{}", page + 1, pages), "callback_data": "cal:noop" }));
        if page < pages - 1 { nav.push(serde_json::json!({ "text": "Next ›", "callback_data": format!("persev:{person_id}:{}", page + 1) })); }
        kb.push(serde_json::Value::Array(nav));
    }
    kb.push(serde_json::json!([{ "text": "‹ Back to people", "callback_data": "cfg:people" }]));
    (format!("🎬 {name} — {total} recorded sighting{} · tap one for its clip.", if total == 1 { "" } else { "s" }),
     serde_json::Value::Array(kb))
}

/// 🚗 Vehicles — recognized plates (30 days), most-seen first, paginated.
async fn build_vehicles_menu(state: &Arc<AppState>, page: u32) -> (String, serde_json::Value) {
    const PER_PAGE: usize = 8;
    let rows_db: Vec<(String, i64, String)> = sqlx::query_as(
        "SELECT recognized_plate, COUNT(*), MAX(started_at) FROM motion_events
          WHERE recognized_plate IS NOT NULL AND recognized_plate <> ''
            AND started_at > datetime('now','-30 days')
          GROUP BY recognized_plate ORDER BY 2 DESC, 3 DESC"
    ).fetch_all(&state.db).await.unwrap_or_default();
    let known = state.settings.read().await.known_plates.clone();
    let name_of = |plate: &str| -> Option<String> {
        known.lines().find_map(|l| l.split_once('=')
            .filter(|(p, _)| p.trim().eq_ignore_ascii_case(plate.trim()))
            .map(|(_, n)| n.trim().to_string()).filter(|n| !n.is_empty()))
    };
    let pages = rows_db.len().div_ceil(PER_PAGE).max(1);
    let page = (page as usize).min(pages - 1);

    let mut kb: Vec<serde_json::Value> = Vec::new();
    if rows_db.is_empty() {
        kb.push(serde_json::json!([{ "text": "No vehicles recognized yet", "callback_data": "cal:noop" }]));
    } else {
        for (plate, count, _) in rows_db.iter().skip(page * PER_PAGE).take(PER_PAGE) {
            let label = match name_of(plate) {
                Some(n) => format!("🚗 {plate} · {n} · {count}×"),
                None    => format!("🚗 {plate} · {count}×"),
            };
            // Cap the plate for the 64-byte callback budget (plates are short;
            // the detail lookup prefix-matches so a capped value still resolves).
            let cap: String = plate.chars().take(16).collect();
            kb.push(serde_json::json!([{ "text": label, "callback_data": format!("veh:{cap}") }]));
        }
        if pages > 1 {
            let mut nav: Vec<serde_json::Value> = Vec::new();
            if page > 0 { nav.push(serde_json::json!({ "text": "‹ Prev", "callback_data": format!("vehp:{}", page - 1) })); }
            nav.push(serde_json::json!({ "text": format!("{}/{}", page + 1, pages), "callback_data": "cal:noop" }));
            if page + 1 < pages { nav.push(serde_json::json!({ "text": "Next ›", "callback_data": format!("vehp:{}", page + 1) })); }
            kb.push(serde_json::Value::Array(nav));
        }
    }
    kb.push(serde_json::json!([{ "text": "📅 Browse by day", "callback_data": "cal:v:" }]));
    kb.push(serde_json::json!([{ "text": "‹ Back", "callback_data": "cfg:back" }]));
    let text = if rows_db.is_empty() {
        "🚗 Vehicles — none recognized yet. Plates appear here when a camera reads one. 📅 browses all vehicle sightings by day.".to_string()
    } else {
        "🚗 Vehicles — recognized plates (30 days). Tap one for details & clips, or 📅 browse sightings by day (with type filters).".to_string()
    };
    (text, serde_json::Value::Array(kb))
}

/// 🚗 plate detail: latest thumbnail + summary + recent event clips.
async fn send_vehicle_detail(state: &Arc<AppState>, token: &str, chat_id: &str, plate_key: &str) {
    // The button may carry a CAPPED plate — prefix-match. Escape LIKE wildcards.
    let like = format!("{}%", plate_key.replace(['%', '_'], ""));
    let events: Vec<(String, String, Option<String>, String)> = sqlx::query_as(
        "SELECT id, started_at, thumbnail, recognized_plate FROM motion_events
          WHERE recognized_plate LIKE ? AND recognized_plate <> ''
          ORDER BY started_at DESC LIMIT 6"
    ).bind(&like).fetch_all(&state.db).await.unwrap_or_default();
    if events.is_empty() {
        send_telegram(token, chat_id, "That vehicle has no recent events anymore — send /vehicles for the current list.").await;
        return;
    }
    let plate = events[0].3.clone();
    let count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM motion_events WHERE recognized_plate=?"
    ).bind(&plate).fetch_one(&state.db).await.unwrap_or(events.len() as i64);
    let first: Option<String> = sqlx::query_scalar(
        "SELECT MIN(started_at) FROM motion_events WHERE recognized_plate=?"
    ).bind(&plate).fetch_optional(&state.db).await.ok().flatten();
    let known = state.settings.read().await.known_plates.clone();
    let name = known.lines().find_map(|l| l.split_once('=')
        .filter(|(p, _)| p.trim().eq_ignore_ascii_case(plate.trim()))
        .map(|(_, n)| n.trim().to_string()).filter(|n| !n.is_empty()));

    let title = match &name { Some(n) => format!("🚗 {plate} — {n}"), None => format!("🚗 {plate}") };
    let first_seen = first.as_deref().map(rel_time).unwrap_or_else(|| "?".into());
    let last_seen  = rel_time(&events[0].1);
    let caption = format!("{title}\nSeen {count}× · first {first_seen} · last {last_seen}\nTap a sighting below for its clip.");

    let mut kb: Vec<serde_json::Value> = Vec::new();
    for (id, started, _, _) in &events {
        let when = chrono::DateTime::parse_from_rfc3339(started)
            .map(|d| d.with_timezone(&Local).format("%b %-d · %H:%M").to_string())
            .unwrap_or_else(|_| started.get(..16).unwrap_or("?").to_string());
        kb.push(serde_json::json!([{ "text": format!("🎬 {when}"), "callback_data": format!("getclip:{id}") }]));
    }
    kb.push(serde_json::json!([{ "text": "‹ Back to vehicles", "callback_data": "cfg:vehicles" }]));

    let jpeg = events[0].2.clone()
        .map(|t| crate::blobstore::resolve(&state.data_dir, &t))
        .and_then(|t| base64::Engine::decode(&base64::engine::general_purpose::STANDARD, t.trim()).ok());
    send_telegram_photo_kb(token, chat_id, jpeg, &caption, serde_json::Value::Array(kb)).await;
}

/// 🔊 Sounds — recent audio events with class + loudness, paginated.
async fn build_sounds_menu(state: &Arc<AppState>, page: u32) -> (String, serde_json::Value) {
    const PER_PAGE: i64 = 8;
    let total: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM motion_events WHERE event_category='audio'"
    ).fetch_one(&state.db).await.unwrap_or(0);
    let pages = ((total + PER_PAGE - 1) / PER_PAGE).max(1);
    let page = (page as i64).min(pages - 1).max(0);
    let rows_db: Vec<(String, String, Option<String>, Option<f64>)> = sqlx::query_as(
        "SELECT id, started_at, dominant_label, loudness_db FROM motion_events
          WHERE event_category='audio' ORDER BY started_at DESC LIMIT ? OFFSET ?"
    ).bind(PER_PAGE).bind(page * PER_PAGE).fetch_all(&state.db).await.unwrap_or_default();

    // 7-day headline: which sounds, how often.
    let tops: Vec<(String, i64)> = sqlx::query_as(
        "SELECT COALESCE(NULLIF(dominant_label,''),'sound'), COUNT(*) FROM motion_events
          WHERE event_category='audio' AND started_at > datetime('now','-7 days')
          GROUP BY 1 ORDER BY 2 DESC LIMIT 3"
    ).fetch_all(&state.db).await.unwrap_or_default();
    let headline = if tops.is_empty() { String::new() } else {
        format!("\nThis week: {}", tops.iter()
            .map(|(s, c)| format!("{s} {c}×")).collect::<Vec<_>>().join(" · "))
    };

    let mut kb: Vec<serde_json::Value> = Vec::new();
    if rows_db.is_empty() {
        kb.push(serde_json::json!([{ "text": "No sounds detected yet", "callback_data": "cal:noop" }]));
    } else {
        for (id, started, label, db) in rows_db {
            let hhmm = chrono::DateTime::parse_from_rfc3339(&started)
                .map(|d| d.with_timezone(&Local).format("%b %-d · %H:%M").to_string())
                .unwrap_or_else(|_| started.get(..16).unwrap_or("?").to_string());
            let label = label.filter(|s| !s.is_empty()).unwrap_or_else(|| "sound".into());
            let loud = db.map(|d| format!(" · {:.0} dB", d)).unwrap_or_default();
            kb.push(serde_json::json!([
                { "text": format!("🔊 {hhmm} · {label}{loud}"), "callback_data": format!("getclip:{id}") }
            ]));
        }
        if pages > 1 {
            let mut nav: Vec<serde_json::Value> = Vec::new();
            if page > 0 { nav.push(serde_json::json!({ "text": "‹ Prev", "callback_data": format!("sndp:{}", page - 1) })); }
            nav.push(serde_json::json!({ "text": format!("{}/{}", page + 1, pages), "callback_data": "cal:noop" }));
            if page < pages - 1 { nav.push(serde_json::json!({ "text": "Next ›", "callback_data": format!("sndp:{}", page + 1) })); }
            kb.push(serde_json::Value::Array(nav));
        }
    }
    kb.push(serde_json::json!([{ "text": "📅 Browse by day", "callback_data": "cal:s:" }]));
    kb.push(serde_json::json!([{ "text": "‹ Back", "callback_data": "cfg:back" }]));
    let text = if total == 0 {
        "🔊 Sounds — nothing detected yet. Sounds appear when a camera with a mic hears something.".to_string()
    } else {
        format!("🔊 Sounds — recent audio events · tap one for its clip, or 📅 browse by day (with sound filters).{headline}")
    };
    (text, serde_json::Value::Array(kb))
}

/// Persist a settings change made from the Telegram menu (memory + DB).
async fn persist_settings(state: &Arc<AppState>, new_s: Settings) {
    // Shared path: encrypts secrets for the DB (the old inline write stored them
    // in PLAINTEXT, corrupting them on next load) AND emits `settings:updated` so
    // the app's Settings panel reflects /menu changes live.
    crate::events_cmds::apply_settings_update(state, new_s).await;
}

/// Edit an existing menu message in place (so toggles update the same message).
async fn edit_telegram_menu(
    client: &reqwest::Client, token: &str, chat_id: i64, message_id: i64,
    text: &str, keyboard: serde_json::Value,
) {
    let url = format!("https://api.telegram.org/bot{}/editMessageText", token);
    let _ = client.post(&url).json(&serde_json::json!({
        "chat_id": chat_id, "message_id": message_id, "text": text,
        "reply_markup": { "inline_keyboard": keyboard },
    })).send().await;
}

/// Poll Telegram for new messages, route each through the agent chat, reply back.
pub async fn run_telegram_loop(state: Arc<AppState>) {
    tokio::time::sleep(Duration::from_secs(15)).await;

    #[derive(Deserialize)]
    struct TgUser { id: i64 }
    #[derive(Deserialize)]
    struct TgCallbackQuery { id: String, #[allow(dead_code)] from: TgUser, data: Option<String>, #[allow(dead_code)] message: Option<TgMessage> }
    #[derive(Deserialize)]
    struct TgUpdate { update_id: i64, message: Option<TgMessage>, callback_query: Option<TgCallbackQuery> }
    #[derive(Deserialize)]
    struct TgMessage { chat: TgChat, text: Option<String>, #[serde(default)] message_id: Option<i64>,
                      /// Boxed: TgMessage would otherwise be infinitely sized.
                      #[serde(default)] reply_to_message: Option<Box<TgMessage>> }
    #[derive(Deserialize)]
    struct TgChat { id: i64 }
    #[derive(Deserialize)]
    struct TgResp { ok: bool, #[serde(default)] result: Vec<TgUpdate> }

    let mut offset: i64 = read_memory(&state.db, "telegram_update_offset").await
        .and_then(|s| s.parse().ok()).unwrap_or(0);

    let mut histories: std::collections::HashMap<String, Vec<ChatMessage>> =
        std::collections::HashMap::new();

    let client = reqwest::Client::new();
    let mut commands_registered = false;
    loop {
        let (token, allowed_chat) = {
            let s = state.settings.read().await;
            (s.telegram_bot_token.clone(), s.telegram_chat_id.clone())
        };

        if token.is_empty() {
            tokio::time::sleep(Duration::from_secs(30)).await;
            continue;
        }

        // Register the bot's command menu once (discoverable "/" list + Menu button).
        if !commands_registered {
            register_bot_commands(&client, &token).await;
            commands_registered = true;
        }

        // Use POST with JSON body to avoid URL-encoding issues with allowed_updates array
        let url = format!("https://api.telegram.org/bot{}/getUpdates", token);
        let req_body = serde_json::json!({
            "offset": offset,
            "timeout": 25,
            "allowed_updates": ["message", "callback_query"]
        });

        let resp = client.post(&url)
            .timeout(Duration::from_secs(35))
            .json(&req_body)
            .send()
            .await;

        let resp = match resp {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[Telegram] poll error: {e}");
                tokio::time::sleep(Duration::from_secs(10)).await;
                continue;
            }
        };

        let tg: TgResp = match resp.json().await {
            Ok(v) => v,
            Err(e) => {
                eprintln!("[Telegram] parse error: {e}");
                tokio::time::sleep(Duration::from_secs(10)).await;
                continue;
            }
        };

        if !tg.ok { tokio::time::sleep(Duration::from_secs(10)).await; continue; }

        for update in tg.result {
            let update_id = update.update_id;
            offset = update_id + 1;

            // Cap per-update processing in a timeout: a single slow/hung handler (a
            // stuck ffmpeg clip export, a wedged share-link CLI, a slow LLM turn) must
            // never freeze the whole poll loop — that is exactly what silently killed
            // the /menu. Every heavy call inside is separately bounded; this is the
            // outer safety net. Fields are destructured out first so the block can own
            // them while borrowing `client`/`token`/`state`/`histories` by reference.
            let cb_opt = update.callback_query;
            let msg_opt = update.message;
            let fut = async {

            // Handle callback_query (inline keyboard button press)
            if let Some(cb) = cb_opt {
                // Answer the callback to remove the "loading" spinner
                let answer_url = format!("https://api.telegram.org/bot{}/answerCallbackQuery", token);
                let _ = client.post(&answer_url)
                    .json(&serde_json::json!({"callback_query_id": cb.id}))
                    .send().await;

                // Capture the source message (for editing the /menu in place).
                let menu_chat = cb.message.as_ref().map(|m| m.chat.id);
                let menu_mid  = cb.message.as_ref().and_then(|m| m.message_id);

                if let Some(data) = cb.data {
                    let parts: Vec<&str> = data.splitn(2, ':').collect();
                    if parts.len() == 2 {
                        let action = parts[0];
                        let alert_id = parts[1];
                        match action {
                            "ack" => {
                                state.pending_escalations.write().await.remove(alert_id);
                                send_telegram(&token, &allowed_chat, "✅ Alert acknowledged.").await;
                                sqlx::query("UPDATE agent_alerts SET feedback='good' WHERE id=?")
                                    .bind(alert_id).execute(&state.db).await.ok();
                            }
                            "fp" => {
                                state.pending_escalations.write().await.remove(alert_id);
                                send_telegram(&token, &allowed_chat, "Marked as false alarm. Guardian will learn from this.").await;
                                sqlx::query("UPDATE agent_alerts SET is_false_positive=1, feedback='bad' WHERE id=?")
                                    .bind(alert_id).execute(&state.db).await.ok();
                                // Append the summary to known false positives
                                if let Some((summary,)) = sqlx::query_as::<_, (String,)>(
                                    "SELECT summary FROM agent_alerts WHERE id=?"
                                ).bind(alert_id).fetch_optional(&state.db).await.unwrap_or(None) {
                                    append_memory(&state.db, "known_false_positives", &summary).await;
                                }
                            }
                            "esc" => {
                                send_telegram(&token, &allowed_chat, "🚨 Escalating. Please contact emergency services if needed. Full incident report follows.").await;
                                sqlx::query("UPDATE agent_alerts SET escalated=1 WHERE id=?")
                                    .bind(alert_id).execute(&state.db).await.ok();
                            }
                            // ── On-demand media (monitor/share only; here `alert_id` carries the event_id) ──
                            "snap" => {
                                let event_id = alert_id;
                                // Snapshot from the camera that fired this event.
                                let cam_id: u8 = sqlx::query_scalar::<_, i64>(
                                    "SELECT cam_id FROM motion_events WHERE id=?")
                                    .bind(event_id).fetch_optional(&state.db).await
                                    .ok().flatten().unwrap_or(0) as u8;
                                send_telegram_action(&token, &allowed_chat, "upload_photo").await;
                                match state.latest_frames.read().await.get(&cam_id).cloned() {
                                    Some(jpeg) => send_telegram_photo(&token, &allowed_chat, jpeg,
                                        "📸 Live snapshot").await,
                                    None => send_telegram(&token, &allowed_chat,
                                        "Camera isn't live right now — no snapshot available.").await,
                                }
                            }
                            // Both clip buttons go through THE clip path, which
                            // captions the upload with when/where/what and mints
                            // a link when the file is past Telegram's ceiling.
                            // These used to be two copies that disagreed: this
                            // one told the user to "use 🔗 Share instead".
                            "clip" =>
                                send_event_clip_by_id(&state, &token, &allowed_chat, alert_id).await,
                            "share" => {
                                let event_id = alert_id;
                                send_telegram_action(&token, &allowed_chat, "typing").await;
                                let mins = state.settings.read().await.live_share_default_minutes;
                                match mint_link_bounded(&state, "clip", event_id, mins).await {
                                    Ok((url, expires_at)) => {
                                        let life = link_life(expires_at);
                                        send_telegram(&token, &allowed_chat,
                                            &format!("🔗 Private clip link — works for {life}.\n{url}\n\nExpired? Tap 🔗 Share again for a fresh one.")).await;
                                    }
                                    Err(msg) => send_telegram(&token, &allowed_chat, &msg).await,
                                }
                            }
                            // ── /menu preference toggles + footage navigation (user-initiated) ──
                            "cfg" => {
                                let key = alert_id;
                                // Navigation keys switch the menu VIEW; the rest toggle a setting.
                                // New views MUST be listed here or opening them would fire a
                                // (no-op) settings write.
                                let nav = matches!(key,
                                    "footage" | "events" | "live" | "cams" | "back"
                                    | "people" | "vehicles" | "sounds" | "see");
                                if !nav {
                                    let mut s = state.settings.read().await.clone();
                                    match key {
                                        "level" => s.alert_min_risk = next_alert_level(&s.alert_min_risk).to_string(),
                                        "quiet" => s.quiet_hours_enabled = !s.quiet_hours_enabled,
                                        "snap"  => s.attach_snapshot_to_alerts = !s.attach_snapshot_to_alerts,
                                        "clip"  => s.attach_clip_to_alerts = !s.attach_clip_to_alerts,
                                        _ => {}
                                    }
                                    persist_settings(&state, s).await;
                                }
                                if let (Some(chat), Some(mid)) = (menu_chat, menu_mid) {
                                    let (text, kb) = match key {
                                        "footage"  => build_footage_menu(),
                                        "events"   => build_events_list_menu(&state).await,
                                        "live"     => build_live_menu(&state).await,
                                        "cams"     => build_camera_menu(&state).await,
                                        "people"   => build_people_menu(&state).await,
                                        "vehicles" => build_vehicles_menu(&state, 0).await,
                                        "sounds"   => build_sounds_menu(&state, 0).await,
                                        "see"      => build_categories_menu(&state).await,
                                        _          => build_alert_menu(&state).await, // back / after a toggle
                                    };
                                    edit_telegram_menu(&client, &token, chat, mid, &text, kb).await;
                                }
                            }
                            // ── 👁 Alert filter: toggle a category mute ──
                            "cfgcat" => {
                                let cat = alert_id;
                                if ALERT_CATEGORIES.iter().any(|(k, _)| *k == cat) {
                                    let mut s = state.settings.read().await.clone();
                                    if let Some(pos) = s.alert_muted_categories.iter().position(|c| c == cat) {
                                        s.alert_muted_categories.remove(pos);
                                    } else {
                                        s.alert_muted_categories.push(cat.to_string());
                                    }
                                    persist_settings(&state, s).await;
                                }
                                if let (Some(chat), Some(mid)) = (menu_chat, menu_mid) {
                                    let (text, kb) = build_categories_menu(&state).await;
                                    edit_telegram_menu(&client, &token, chat, mid, &text, kb).await;
                                }
                            }
                            // ── 👤 person detail card (photo + stats + sightings) ──
                            "pers" => {
                                send_person_detail(&state, &token, &allowed_chat, alert_id).await;
                            }
                            // ── 👤 person recent sightings (payload = "<uuid>:<page>") ──
                            "persev" => {
                                let (pid, page) = alert_id.rsplit_once(':')
                                    .map(|(p, pg)| (p.to_string(), pg.parse::<u32>().unwrap_or(0)))
                                    .unwrap_or_else(|| (alert_id.to_string(), 0));
                                let (text, kb) = build_person_events_menu(&state, &pid, page).await;
                                // Detail cards are photo messages (can't edit-in-place) —
                                // send the sightings list as its own message.
                                send_telegram_with_keyboard(&token, &allowed_chat, &text, kb).await;
                            }
                            // ── 🚗 vehicle detail card ──
                            "veh" => {
                                send_vehicle_detail(&state, &token, &allowed_chat, alert_id).await;
                            }
                            // ── 🚗 vehicles pager ──
                            "vehp" => {
                                if let (Some(chat), Some(mid)) = (menu_chat, menu_mid) {
                                    let page = alert_id.parse::<u32>().unwrap_or(0);
                                    let (text, kb) = build_vehicles_menu(&state, page).await;
                                    edit_telegram_menu(&client, &token, chat, mid, &text, kb).await;
                                }
                            }
                            // ── 🔊 sounds pager ──
                            "sndp" => {
                                if let (Some(chat), Some(mid)) = (menu_chat, menu_mid) {
                                    let page = alert_id.parse::<u32>().unwrap_or(0);
                                    let (text, kb) = build_sounds_menu(&state, page).await;
                                    edit_telegram_menu(&client, &token, chat, mid, &text, kb).await;
                                }
                            }
                            "cfgcam" => {
                                if let Ok(id) = alert_id.parse::<u8>() {
                                    let mut s = state.settings.read().await.clone();
                                    if let Some(pos) = s.alert_disabled_cameras.iter().position(|&c| c == id) {
                                        s.alert_disabled_cameras.remove(pos);
                                    } else {
                                        s.alert_disabled_cameras.push(id);
                                    }
                                    persist_settings(&state, s).await;
                                    if let (Some(chat), Some(mid)) = (menu_chat, menu_mid) {
                                        let (text, kb) = build_camera_menu(&state).await;
                                        edit_telegram_menu(&client, &token, chat, mid, &text, kb).await;
                                    }
                                }
                            }
                            // ── Calendar: month grid navigation (all footage / v: vehicles / s: sounds) ──
                            "cal" => {
                                if let (Some(chat), Some(mid)) = (menu_chat, menu_mid) {
                                    if alert_id != "noop" {
                                        let (cat, rest) = cal_cat(alert_id);
                                        // Empty payload = current month; "YYYY-MM" = that month.
                                        let (y, m) = parse_year_month(rest)
                                            .unwrap_or_else(|| { let n = Local::now(); (n.year(), n.month()) });
                                        let (text, kb) = build_calendar_menu(&state, y, m, cat).await;
                                        edit_telegram_menu(&client, &token, chat, mid, &text, kb).await;
                                    }
                                }
                            }
                            // ── Calendar: open one day's events (optional "~<filter>" suffix) ──
                            "calday" => {
                                if let (Some(chat), Some(mid)) = (menu_chat, menu_mid) {
                                    let (cat, rest) = cal_cat(alert_id);
                                    let (date_str, filter) = match rest.split_once('~') {
                                        Some((d, f)) => (d, Some(f)),
                                        None => (rest, None),
                                    };
                                    let (text, kb) = build_day_events_menu(&state, date_str, 0, cat, filter).await;
                                    edit_telegram_menu(&client, &token, chat, mid, &text, kb).await;
                                }
                            }
                            // ── Calendar: paginate a day (payload = "[v:|s:]YYYY-MM-DD[~filter]:<page>") ──
                            "calp" => {
                                if let (Some(chat), Some(mid)) = (menu_chat, menu_mid) {
                                    let (cat, rest) = cal_cat(alert_id);
                                    let (datefilt, page) = rest.rsplit_once(':')
                                        .map(|(d, p)| (d.to_string(), p.parse::<u32>().unwrap_or(0)))
                                        .unwrap_or_else(|| (rest.to_string(), 0));
                                    let (date_str, filter) = match datefilt.split_once('~') {
                                        Some((d, f)) => (d.to_string(), Some(f.to_string())),
                                        None => (datefilt, None),
                                    };
                                    let (text, kb) = build_day_events_menu(&state, &date_str, page, cat, filter.as_deref()).await;
                                    edit_telegram_menu(&client, &token, chat, mid, &text, kb).await;
                                }
                            }
                            // ── Footage menu: send a picked event's clip ──
                            "getclip" =>
                                send_event_clip_by_id(&state, &token, &allowed_chat, alert_id).await,
                            // ── Footage menu: send a camera's live snapshot + live link ──
                            "getlive" => {
                                if let Ok(cam_id) = alert_id.parse::<u8>() {
                                    send_telegram_action(&token, &allowed_chat, "upload_photo").await;
                                    match state.latest_frames.read().await.get(&cam_id).cloned() {
                                        Some(jpeg) => send_telegram_photo(&token, &allowed_chat, jpeg,
                                            &format!("📸 CAM {} — live", cam_id + 1)).await,
                                        None => send_telegram(&token, &allowed_chat,
                                            &format!("CAM {} isn't live right now.", cam_id + 1)).await,
                                    }
                                    let mins = state.settings.read().await.live_share_default_minutes;
                                    match mint_link_bounded(&state, "live", &cam_id.to_string(), mins).await {
                                        Ok((url, expires_at)) => {
                                            let life = link_life(expires_at);
                                            send_telegram(&token, &allowed_chat,
                                                &format!("📡 Live link — works for {life}.\n{url}\n\nExpired? Tap 📡 Live again for a fresh one.")).await;
                                        }
                                        Err(msg) => send_telegram(&token, &allowed_chat, &msg).await,
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
                return;
            }

            let Some(msg) = msg_opt else { return; };
            let Some(ref user_text) = msg.text else { return; };
            let from_chat = msg.chat.id.to_string();

            eprintln!("[Telegram] message from chat_id={from_chat}: {user_text}");

            // If configured chat_id doesn't match: send the user their numeric ID so they can configure it
            // Also handle the case where user entered a @username instead of numeric ID
            let is_allowed = allowed_chat.is_empty()
                || from_chat == allowed_chat
                || allowed_chat.starts_with('@'); // username format — can't compare numerically

            if !is_allowed {
                let info = format!(
                    "This bot is configured for a different chat. Your numeric chat ID is: {from_chat}\nEnter this in Anivar Settings > Telegram > Chat ID."
                );
                send_telegram(&token, &from_chat, &info).await;
                return;
            }

            // If user entered @username, auto-notify them of the real numeric ID
            if allowed_chat.starts_with('@') {
                let hint = format!(
                    "Your numeric Telegram chat ID is: {from_chat}\nPlease update Settings > Telegram Bot > Chat ID with this number instead of your username."
                );
                send_telegram(&token, &from_chat, &hint).await;
            }

            // (There is no search-prompt intercept here any more. Searching was
            // a button that asked a question that ran a keyword query; the agent
            // answers the same question typed plainly, with the same evidence.)

            // ── Fast slash commands (no model needed) ─────────────────────
            // The ORIGINAL text, not a lowercased copy: `handle_slash_command`
            // folds case on the command word itself, and everything after it is
            // the user's own words — "/investigate who was with Alex" must not
            // reach the archive as "alex".
            let cmd = user_text.trim();
            if cmd.starts_with('/') {
                let reply = handle_slash_command(&state, cmd, &token, &from_chat).await;
                if !reply.is_empty() {
                    send_telegram(&token, &from_chat, &reply).await;
                }
                return;
            }

            let history = histories.entry(from_chat.clone()).or_default();

            // Show "typing…" indicator while the model works (repeat every 4 s)
            let typing_active = Arc::new(std::sync::atomic::AtomicBool::new(true));
            let typing_active2 = Arc::clone(&typing_active);
            let typ_tok = token.clone();
            let typ_chat = from_chat.clone();
            tokio::spawn(async move {
                loop {
                    send_telegram_action(&typ_tok, &typ_chat, "typing").await;
                    for _ in 0..40u8 { // 40 × 100 ms = 4 s
                        tokio::time::sleep(Duration::from_millis(100)).await;
                        if !typing_active2.load(std::sync::atomic::Ordering::Relaxed) { return; }
                    }
                }
            });

            let reply = match chat_with_agent(&state, history.clone(), user_text.clone(), Default::default()).await {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("[Telegram] agent error: {e}");
                    format!("Guardian is unavailable right now.\nError: {e}")
                }
            };
            typing_active.store(false, std::sync::atomic::Ordering::Relaxed);

            // Resolve tags ONCE, exactly as the app does, then render. History
            // keeps the CLEAN text: the tagged form used to be replayed back to
            // the model as conversation, teaching it to imitate bracket syntax
            // the on-device prompt explicitly forbids.
            let (clean_reply, evidence) = super::evidence::resolve(&state, &reply).await;

            history.push(ChatMessage { role: "user".into(),     content: user_text.clone() });
            history.push(ChatMessage { role: "assistant".into(), content: clean_reply.clone() });
            if history.len() > 20 { history.drain(..history.len() - 20); }

            // Prose first, then the evidence it describes — the alternative put
            // an album above the sentence explaining it.
            if !clean_reply.is_empty() {
                send_telegram(&token, &from_chat, &clean_reply).await;
            }
            render_evidence(&state, &token, &from_chat, &evidence).await;
            }; // end per-update `fut`

            // 200s > every inner deadline (LLM chat 180s, link mint 45s, ffmpeg 300s
            // runs detached) so bounded inner paths resolve FIRST and their honest
            // error replies reach the user; this outer cap only catches the unknown.
            if tokio::time::timeout(Duration::from_secs(200), fut).await.is_err() {
                eprintln!("[Telegram] update {update_id} handling timed out; skipping to keep the loop alive");
            }
            // Persist the offset PER UPDATE (not once per batch): a slow/failed handler
            // must never leave the offset stuck or cause the batch to replay — a stuck
            // offset is precisely how the menu went silent.
            write_memory(&state.db, "telegram_update_offset", &offset.to_string()).await;
        }
    }
}

impl AgentAlert {
    fn recommended_action_display(&self) -> String {
        self.actions_taken
            .as_deref()
            .and_then(|s| serde_json::from_str::<Vec<String>>(s).ok())
            .map(|v| v.join(", "))
            .unwrap_or_else(|| "Monitor".to_string())
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    /// The command word is matched case-insensitively (and with Telegram's
    /// group-mode `@BotName` suffix stripped); everything AFTER it is the
    /// user's own words and must survive untouched. The poll loop used to
    /// lowercase the whole line, so "/investigate who was with Alex" reached
    /// the archive asking about "alex".
    #[test]
    fn a_slash_command_folds_its_verb_but_never_its_argument() {
        let split = |cmd: &str| {
            let parts: Vec<&str> = cmd.trim().splitn(2, ' ').collect();
            let head = parts[0].split('@').next().unwrap_or("").to_lowercase();
            (head, parts.get(1).unwrap_or(&"").to_string())
        };

        assert_eq!(split("/Investigate Who was with Alex"),
                   ("/investigate".into(), "Who was with Alex".into()));
        assert_eq!(split("/search@GuardianBot Red Van"),
                   ("/search".into(), "Red Van".into()));
        assert_eq!(split("/BRIEF"), ("/brief".into(), String::new()));
    }

    /// Every registered slash command must be reachable — a command advertised
    /// by `setMyCommands` with no match arm is a dead menu entry.
    #[test]
    fn the_two_new_modes_are_advertised_and_handled() {
        let src = include_str!("dispatch.rs");
        for cmd in ["investigate", "brief"] {
            assert!(src.contains(&format!("\"command\": \"{cmd}\"")),
                    "/{cmd} must be registered with Telegram");
            assert!(src.contains(&format!("\"/{cmd}\" =>")),
                    "/{cmd} must have a handler");
        }
    }

    /// Search is deliberately NOT a menu button or a registered command — the
    /// agent answers "the red van last Tuesday" with the same evidence, and
    /// understands the date and the description rather than keyword-matching
    /// them. This pins the decision so it isn't re-added as an oversight fix.
    #[test]
    fn search_is_not_a_button_or_a_command() {
        let src = include_str!("dispatch.rs");
        // Needles are BUILT, never written as literals — this test scans its own
        // source file, so a literal here would match itself and always pass.
        for (needle, what) in [
            (format!("cfg:{}", "srch"),               "a search button in the menu"),
            (format!("\"command\": \"{}\"", "search"), "/search advertised to Telegram"),
            (format!("force{}reply", "_"),             "the keyword-prompt flow"),
        ] {
            assert!(!src.contains(&needle), "{what} came back — search belongs to the agent");
        }
    }

    /// The grid is buttons only — one album of thumbnails per answer was the
    /// wrong shape for a list you scan.
    #[test]
    fn an_event_list_renders_as_a_grid_not_an_album() {
        // Two per row, so the cap is six rows of uniform-width buttons.
        assert_eq!(GRID_MAX, 12);
        assert_eq!(GRID_MAX % 2, 0, "an odd cap leaves a lone half-width button");
        // A button shows only what distinguishes it; the record is in the text.
        assert_eq!(short_time("Aug 03 21:14"), "21:14");
        assert_eq!(short_time("21:14"), "21:14");
        assert_eq!(short_time("whenever"), "whenever", "no colon, no truncation");
    }

    #[test]
    fn alert_level_cycles_off_to_all_and_wraps() {
        assert_eq!(next_alert_level("off"), "critical");
        assert_eq!(next_alert_level("critical"), "suspicious");
        assert_eq!(next_alert_level("suspicious"), "normal");
        assert_eq!(next_alert_level("normal"), "off"); // wraps
        assert_eq!(next_alert_level("unknown"), "off"); // catch-all
    }

    #[test]
    fn alert_level_labels_are_stable() {
        assert_eq!(alert_level_label("off"), "Off");
        assert_eq!(alert_level_label("critical"), "Critical");
        assert_eq!(alert_level_label("suspicious"), "Suspicious");
        assert_eq!(alert_level_label("normal"), "All");
        assert_eq!(alert_level_label("garbage"), "All"); // default arm
    }

    #[test]
    fn callback_payloads_fit_telegram_64_byte_limit() {
        // Telegram rejects callback_data > 64 bytes — every payload shape we
        // generate must fit at its worst case (36-char UUIDs, 16-char plates,
        // 4-digit pages).
        let uuid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeffff";
        let worst = [
            format!("pers:{uuid}"),
            format!("persev:{uuid}:9999"),
            format!("getclip:{uuid}"),
            format!("veh:{}", "A".repeat(16)),
            format!("vehp:9999"),
            format!("sndp:9999"),
            format!("cfgcat:vehicle"),
            format!("calp:2026-12-31:9999"),
            format!("calday:2026-12-31"),
        ];
        for p in &worst {
            assert!(p.len() <= 64, "callback payload too long ({}): {p}", p.len());
        }
    }

    #[test]
    fn link_life_reports_remaining_not_requested() {
        assert_eq!(link_life(0), "until the app restarts");
        let now = chrono::Utc::now().timestamp();
        assert_eq!(link_life(now + 20 * 60), "20 min"); // reused link, 20 of 30 left
        assert_eq!(link_life(now + 61), "2 min");       // ceil, never "0 min" while alive
        assert_eq!(link_life(now - 10), "0 min");       // already dead clamps at 0
    }

    #[test]
    fn parse_year_month_accepts_valid_rejects_invalid() {
        assert_eq!(parse_year_month("2026-06"), Some((2026, 6)));
        assert_eq!(parse_year_month("2026-01"), Some((2026, 1)));
        assert_eq!(parse_year_month("2026-12"), Some((2026, 12)));
        assert_eq!(parse_year_month("2026-13"), None); // month > 12
        assert_eq!(parse_year_month("2026-00"), None); // month 0
        assert_eq!(parse_year_month("2026"), None);    // no month
        assert_eq!(parse_year_month(""), None);
        assert_eq!(parse_year_month("noop"), None);
    }
}
