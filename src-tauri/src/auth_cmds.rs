//! Desktop login gate commands — set/change password, login (+ optional Telegram
//! 2FA), Telegram-delivered recovery, and "remember this device". See `auth.rs`
//! for the security core + threat model.

use std::sync::Arc;

use tauri::State;

use crate::auth::{self, OtpKind};
use crate::{AppState, Settings};

// ── Helpers ───────────────────────────────────────────────────────────────────

fn tg_ready(s: &Settings) -> bool {
    !s.telegram_bot_token.is_empty() && !s.telegram_chat_id.is_empty()
}

/// Send a message to the configured Telegram (best-effort, never blocks the flow).
async fn tg_send(state: &Arc<AppState>, text: &str) {
    let (bot, chat) = {
        let s = state.settings.read().await;
        (s.telegram_bot_token.clone(), s.telegram_chat_id.clone())
    };
    crate::agent::send_telegram(&bot, &chat, text).await;
}

/// Mutate settings through the shared apply path (encrypts secrets, persists,
/// updates in-memory, emits `settings:updated`).
async fn mutate_settings<F: FnOnce(&mut Settings)>(state: &Arc<AppState>, f: F) {
    let mut s = state.settings.read().await.clone();
    f(&mut s);
    crate::events_cmds::apply_settings_update(state, s).await;
}

/// Issue a "remember this device" token (hashed + expiry in `auth_remember`) when
/// the user opted in and the feature is enabled. Returns the raw token for the
/// frontend to keep in localStorage.
async fn issue_remember(state: &Arc<AppState>, want: bool) -> Option<String> {
    let (enabled, days) = {
        let s = state.settings.read().await;
        (s.auth_remember_enabled, s.auth_remember_days.max(1))
    };
    if !want || !enabled { return None; }
    let token = auth::gen_remember_token();
    let hash = crate::http_handlers::sha256_hex(token.as_bytes());
    let _ = sqlx::query("INSERT OR REPLACE INTO auth_remember(token_hash, expires_at) VALUES(?, datetime('now', ?))")
        .bind(&hash).bind(format!("+{days} days")).execute(&state.db).await;
    Some(token)
}

// ── Status ─────────────────────────────────────────────────────────────────────

#[derive(serde::Serialize)]
pub struct AuthStatus {
    pub login_required:   bool,
    pub unlocked:         bool,
    pub has_password:     bool,
    pub twofa_enabled:    bool,
    pub telegram_ready:   bool,
    pub remember_enabled: bool,
    pub remember_days:    u32,
}

#[tauri::command]
pub async fn auth_status(state: State<'_, Arc<AppState>>) -> Result<AuthStatus, String> {
    let s = state.settings.read().await;
    Ok(AuthStatus {
        login_required:   s.login_required,
        // When login isn't required the app is always "unlocked".
        unlocked:         !s.login_required || state.auth.is_unlocked(),
        has_password:     !s.login_password_hash.is_empty(),
        twofa_enabled:    s.auth_2fa_enabled,
        telegram_ready:   tg_ready(&s),
        remember_enabled: s.auth_remember_enabled,
        remember_days:    s.auth_remember_days,
    })
}

// ── Result type for the login flow ──────────────────────────────────────────────

#[derive(serde::Serialize)]
#[serde(tag = "status")]
pub enum LoginResult {
    /// Authenticated — app unlocked. `remember_token` set when the user opted in.
    Unlocked { remember_token: Option<String> },
    /// Password OK but a Telegram 2FA code is required; verify with `challenge`.
    Needs2fa { challenge: String },
}

// ── Configuration commands ──────────────────────────────────────────────────────

/// Set or change the desktop login password. Requires Telegram to be configured
/// first (mandatory recovery channel — so the user can never be locked out). When
/// a password already exists, the current one must be supplied.
#[tauri::command]
pub async fn set_login_password(
    state: State<'_, Arc<AppState>>,
    new_password: String,
    current_password: Option<String>,
) -> Result<(), String> {
    let st = state.inner().clone();
    let (existing, tg) = {
        let s = st.settings.read().await;
        (s.login_password_hash.clone(), tg_ready(&s))
    };
    if !tg {
        return Err("Set up Telegram first (Channels) — it's the mandatory recovery method.".into());
    }
    if !existing.is_empty() {
        let cur = current_password.unwrap_or_default();
        if !auth::verify_password(&cur, &existing) {
            return Err("Current password is incorrect.".into());
        }
    }
    auth::validate_strength(&new_password)?;
    let hash = auth::hash_password(&new_password)?;
    mutate_settings(&st, |s| s.login_password_hash = hash).await;
    st.auth.set_unlocked(true);
    st.auth.reset_fails();
    tg_send(&st, "🔐 Anivar: your login password was set/changed. If this wasn't you, change it immediately.").await;
    Ok(())
}

/// Enable/disable the desktop lock screen. Enabling requires a password + Telegram.
#[tauri::command]
pub async fn set_login_required(state: State<'_, Arc<AppState>>, enabled: bool) -> Result<(), String> {
    let st = state.inner().clone();
    if enabled {
        let s = st.settings.read().await;
        if s.login_password_hash.is_empty() { return Err("Set a password first.".into()); }
        if !tg_ready(&s) { return Err("Set up Telegram first (mandatory recovery channel).".into()); }
    }
    mutate_settings(&st, |s| s.login_required = enabled).await;
    if !enabled {
        // Turning the gate off clears any remembered devices.
        let _ = sqlx::query("DELETE FROM auth_remember").execute(&st.db).await;
    }
    Ok(())
}

#[tauri::command]
pub async fn set_2fa_enabled(state: State<'_, Arc<AppState>>, enabled: bool) -> Result<(), String> {
    let st = state.inner().clone();
    if enabled && !tg_ready(&*st.settings.read().await) {
        return Err("Set up Telegram first — it delivers the 2FA code.".into());
    }
    mutate_settings(&st, |s| s.auth_2fa_enabled = enabled).await;
    Ok(())
}

#[tauri::command]
pub async fn set_remember_device(state: State<'_, Arc<AppState>>, enabled: bool, days: u32) -> Result<(), String> {
    let st = state.inner().clone();
    mutate_settings(&st, |s| { s.auth_remember_enabled = enabled; s.auth_remember_days = days.clamp(1, 365); }).await;
    if !enabled { let _ = sqlx::query("DELETE FROM auth_remember").execute(&st.db).await; }
    Ok(())
}

// ── Login flow ───────────────────────────────────────────────────────────────────

#[tauri::command]
pub async fn login(state: State<'_, Arc<AppState>>, password: String, remember: bool) -> Result<LoginResult, String> {
    let st = state.inner().clone();
    if let Some(rem) = st.auth.lockout_remaining() {
        return Err(format!("Too many attempts. Try again in {}s.", rem.as_secs() + 1));
    }
    let (hash, twofa) = {
        let s = st.settings.read().await;
        (s.login_password_hash.clone(), s.auth_2fa_enabled)
    };
    if hash.is_empty() { return Err("No password is set.".into()); }
    if !auth::verify_password(&password, &hash) {
        if let Some(rem) = st.auth.record_fail() {
            return Err(format!("Too many attempts. Locked for {}s.", rem.as_secs()));
        }
        return Err("Incorrect password.".into());
    }
    st.auth.reset_fails();
    if twofa {
        let (challenge, code) = st.auth.new_challenge(OtpKind::TwoFactor)?;
        tg_send(&st, &format!("🔑 Anivar login code: {code}\nExpires in 5 minutes. If you didn't try to log in, change your password.")).await;
        return Ok(LoginResult::Needs2fa { challenge });
    }
    st.auth.set_unlocked(true);
    let remember_token = issue_remember(&st, remember).await;
    Ok(LoginResult::Unlocked { remember_token })
}

#[tauri::command]
pub async fn login_verify_otp(
    state: State<'_, Arc<AppState>>,
    challenge: String,
    code: String,
    remember: bool,
) -> Result<LoginResult, String> {
    let st = state.inner().clone();
    match st.auth.verify_challenge(&challenge, &code)? {
        OtpKind::TwoFactor => {
            st.auth.set_unlocked(true);
            let remember_token = issue_remember(&st, remember).await;
            Ok(LoginResult::Unlocked { remember_token })
        }
        OtpKind::Recovery => Err("Wrong code type.".into()),
    }
}

/// Resume on launch from a remembered device token. Returns true if it unlocked.
#[tauri::command]
pub async fn auth_resume(state: State<'_, Arc<AppState>>, remember_token: String) -> Result<bool, String> {
    let st = state.inner().clone();
    if remember_token.is_empty() { return Ok(false); }
    let hash = crate::http_handlers::sha256_hex(remember_token.as_bytes());
    let row: Option<(String,)> = sqlx::query_as(
        "SELECT token_hash FROM auth_remember WHERE token_hash=? AND expires_at > datetime('now')"
    ).bind(&hash).fetch_optional(&st.db).await.map_err(|e| e.to_string())?;
    if row.is_some() { st.auth.set_unlocked(true); Ok(true) } else { Ok(false) }
}

/// Soft re-lock (keeps any remembered-device token).
#[tauri::command]
pub async fn lock(state: State<'_, Arc<AppState>>) -> Result<(), String> {
    state.auth.set_unlocked(false);
    Ok(())
}

/// Full logout — re-lock and forget this device's remember token.
#[tauri::command]
pub async fn logout(state: State<'_, Arc<AppState>>, remember_token: Option<String>) -> Result<(), String> {
    let st = state.inner().clone();
    st.auth.set_unlocked(false);
    if let Some(t) = remember_token {
        let hash = crate::http_handlers::sha256_hex(t.as_bytes());
        let _ = sqlx::query("DELETE FROM auth_remember WHERE token_hash=?").bind(&hash).execute(&st.db).await;
    }
    Ok(())
}

// ── Recovery (forgot password → code on Telegram → reset) ────────────────────────

/// "Forgot password" — send a recovery code to the user's Telegram. Returns the
/// (non-secret) challenge id; the code itself only goes to Telegram. Throttled.
#[tauri::command]
pub async fn request_recovery(state: State<'_, Arc<AppState>>) -> Result<String, String> {
    let st = state.inner().clone();
    {
        let s = st.settings.read().await;
        if s.login_password_hash.is_empty() { return Err("No password is set.".into()); }
        if !tg_ready(&s) { return Err("Telegram isn't configured — recovery is unavailable.".into()); }
    }
    let (challenge, code) = st.auth.new_challenge(OtpKind::Recovery)?;
    tg_send(&st, &format!("🆘 Anivar password recovery code: {code}\nEnter it in the app to set a new password. Expires in 10 minutes. If you didn't request this, ignore it.")).await;
    Ok(challenge)
}

/// Complete recovery: verify the Telegram code, set a new password, unlock.
#[tauri::command]
pub async fn recovery_reset(
    state: State<'_, Arc<AppState>>,
    challenge: String,
    code: String,
    new_password: String,
    remember: bool,
) -> Result<LoginResult, String> {
    let st = state.inner().clone();
    match st.auth.verify_challenge(&challenge, &code)? {
        OtpKind::Recovery => {}
        OtpKind::TwoFactor => return Err("Wrong code type.".into()),
    }
    auth::validate_strength(&new_password)?;
    let hash = auth::hash_password(&new_password)?;
    mutate_settings(&st, |s| s.login_password_hash = hash).await;
    st.auth.set_unlocked(true);
    st.auth.reset_fails();
    tg_send(&st, "🔐 Anivar: your password was reset via Telegram recovery.").await;
    let remember_token = issue_remember(&st, remember).await;
    Ok(LoginResult::Unlocked { remember_token })
}
