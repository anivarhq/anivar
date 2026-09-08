//! v11 Tauri commands for the on-demand share-link flow.
//!
//! Exposed:
//!   * `generate_share_link(kind, resource_id, expiry_mins) -> Url`
//!   * `revoke_all_shares() -> ()`
//!   * `list_active_shares() -> Vec<ShareEntry>`
//!
//! Adding a share:
//!   1. Bring the public URL up (`tailscale::ensure_public_url`) — idempotent.
//!   2. Mint an HMAC-signed token bound to the current `share_generation`.
//!   3. Register the entry in `state.active_shares` so the auto-stop loop
//!      can see active shares + the UI can list them.
//!
//! Revoking everything:
//!   1. Bump `share_generation` — every outstanding token's HMAC stops
//!      verifying immediately (the cookie derivation depends on it too).
//!   2. Clear `active_shares`. The auto-stop loop will tear the tunnel down
//!      on its next tick.

use std::sync::Arc;

use serde::Serialize;
use tauri::State;

use crate::AppState;
use crate::share_security::{sign_share_token, SharePayload};
use crate::state::ShareEntry;

#[derive(Debug, Serialize)]
pub struct ShareLinkResult {
    /// Full https:// URL the user copies and shares.
    pub url:        String,
    pub kind:       String,
    pub resource_id: String,
    pub expires_at: i64,
}

#[tauri::command]
pub async fn generate_share_link(
    state: State<'_, Arc<AppState>>,
    app: tauri::AppHandle,
    kind: String,
    resource_id: String,
    expiry_mins: u32,
) -> Result<ShareLinkResult, String> {
    mint_share_link(&state, app, kind, resource_id, expiry_mins).await
}

/// Serializes mints. Two triggers can request a link at the same moment (a
/// Telegram tap + the alert pipeline's auto-share run on different tasks) —
/// unserialized, both raced `ensure_tunnel_up` (double `tailscale funnel`
/// spawn) and each minted its own token. With
/// the gate, the second request lands on the reuse path and returns the SAME
/// link the first one just made.
static MINT_GATE: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// Minimum useful life left on an outstanding link for it to be REUSED instead
/// of minting a fresh one. Below this, tapping again mints fresh (matching the
/// "Expired? Tap again for a fresh one" contract).
const REUSE_MIN_REMAINING_SECS: i64 = 120;

/// Core share-link minting — usable outside the Tauri command boundary (e.g. the
/// Telegram bot's "🔗 Share" inline button). Brings the tunnel up, signs an
/// HMAC token bound to the current generation, registers the active share, and
/// returns the `/redeem` URL.
///
/// IDEMPOTENT per (kind, resource): a repeat request while the previous link is
/// still healthy returns the SAME url — tapping 📡 Live twice must not hand the
/// user two different both-valid links and duplicate tracker rows. When a fresh
/// link IS minted, the old entry for that resource is REPLACED (one truthful
/// row per resource; the old token stays cryptographically valid until its own
/// expiry — stateless HMAC by design, `revoke_all_shares` is the kill switch).
// `app` is consumed only by the platform-gated branches below, so it reads as
// unused on whichever target those are compiled out of. Renaming it `_app` is
// NOT the fix — that breaks the targets that do use it.
#[allow(unused_variables)]
pub(crate) async fn mint_share_link(
    state: &Arc<AppState>,
    app: tauri::AppHandle,
    kind: String,
    resource_id: String,
    expiry_mins: u32,
) -> Result<ShareLinkResult, String> {
    if kind != "live" && kind != "clip" {
        return Err(format!("unknown kind '{}'", kind));
    }
    let _mint = MINT_GATE.lock().await;

    // 1. Bring the tunnel up (idempotent, bounded).
    let tunnel_url = crate::tailscale::ensure_public_url(Arc::clone(state)).await?;
    let base = tunnel_url.trim_end_matches('/').to_string();

    let now = chrono::Utc::now().timestamp();

    // 2. Prune expired entries, then REUSE an outstanding healthy link for this
    //    (kind, resource): same base (guards against a re-minted public
    //    hostname — the old URL would be dead), non-empty url (pre-field rows),
    //    and either never-expiring or ≥2 min of life left.
    {
        let mut shares = state.active_shares.write().await;
        shares.retain(|s| s.expires_at == 0 || s.expires_at > now);
        if let Some(existing) = shares.iter().find(|s|
            s.kind == kind && s.resource_id == resource_id
            && !s.url.is_empty()
            && s.url.starts_with(&base)
            && (s.expires_at == 0 || s.expires_at - now >= REUSE_MIN_REMAINING_SECS))
        {
            return Ok(ShareLinkResult {
                url:         existing.url.clone(),
                kind,
                resource_id,
                expires_at:  existing.expires_at,
            });
        }
    }

    // 3. Mint fresh: compute expiry, sign the token.
    let expires_at = if expiry_mins == 0 {
        0 // never (until app restart)
    } else {
        now + (expiry_mins as i64) * 60
    };
    let payload = SharePayload {
        kind:        kind.clone(),
        resource_id: resource_id.clone(),
        expires_at,
    };
    let generation = *state.share_generation.read().await;
    let token = sign_share_token(&state.master_key, generation, &payload);
    let url = format!("{}/redeem?t={}", base, token);

    // 4. REPLACE any older entry for this resource, then register + persist.
    {
        let mut shares = state.active_shares.write().await;
        shares.retain(|s| !(s.kind == kind && s.resource_id == resource_id));
        shares.push(ShareEntry {
            kind:        kind.clone(),
            resource_id: resource_id.clone(),
            expires_at,
            created_at:  chrono::Utc::now().to_rfc3339(),
            url:         url.clone(),
        });
        persist_shares(&state.db, &shares, generation).await;
    }

    Ok(ShareLinkResult {
        url,
        kind,
        resource_id,
        expires_at,
    })
}

// ─── Restart persistence ─────────────────────────────────────────────────────
// `active_shares` used to be memory-only: any app restart emptied the list, so
// the cookie gate (`expected_cookie_for`, which reconstructs candidate tokens
// FROM this list) rejected still-valid links with "share not found or expired"
// — the user-reported "invalid or missing token" on a link minted minutes ago.
// Timed entries + the generation now round-trip through the settings KV table.
// `expires_at == 0` entries are deliberately NOT persisted: "until app restart"
// is their documented contract.

/// Save timed shares + the generation counter (best-effort; failures only cost
/// restart-survival, never the mint itself).
pub(crate) async fn persist_shares(db: &sqlx::SqlitePool, shares: &[ShareEntry], generation: u64) {
    let timed: Vec<&ShareEntry> = shares.iter().filter(|s| s.expires_at != 0).collect();
    if let Ok(json) = serde_json::to_string(&timed) {
        let _ = sqlx::query("INSERT OR REPLACE INTO settings(key,value) VALUES('active_shares',?)")
            .bind(&json).execute(db).await;
    }
    let _ = sqlx::query("INSERT OR REPLACE INTO settings(key,value) VALUES('share_generation',?)")
        .bind(generation.to_string()).execute(db).await;
}

/// Boot-time restore: unexpired timed shares + the generation counter, so links
/// survive an app restart within their lifetime.
pub(crate) async fn load_persisted_shares(db: &sqlx::SqlitePool) -> (Vec<ShareEntry>, u64) {
    let generation: u64 = sqlx::query_scalar::<_, String>(
        "SELECT value FROM settings WHERE key='share_generation'"
    ).fetch_optional(db).await.ok().flatten()
        .and_then(|s| s.parse().ok()).unwrap_or(0);
    let now = chrono::Utc::now().timestamp();
    let shares: Vec<ShareEntry> = sqlx::query_scalar::<_, String>(
        "SELECT value FROM settings WHERE key='active_shares'"
    ).fetch_optional(db).await.ok().flatten()
        .and_then(|json| serde_json::from_str::<Vec<ShareEntry>>(&json).ok())
        .unwrap_or_default()
        .into_iter()
        .filter(|s| s.expires_at != 0 && s.expires_at > now)
        .collect();
    (shares, generation)
}

#[tauri::command]
pub async fn revoke_all_shares(state: State<'_, Arc<AppState>>) -> Result<(), String> {
    // Bump the generation — every outstanding token's HMAC stops matching.
    let generation = {
        let mut gen = state.share_generation.write().await;
        *gen = gen.wrapping_add(1);
        *gen
    };
    // Drop the active list. The auto-stop loop will close the tunnel on
    // its next tick (within ~60 s).
    state.active_shares.write().await.clear();
    // Persist BOTH: without this, a restart reloaded the old generation and
    // the old share list — resurrecting every link the user just revoked.
    persist_shares(&state.db, &[], generation).await;
    Ok(())
}

#[tauri::command]
pub async fn list_active_shares(state: State<'_, Arc<AppState>>) -> Result<Vec<ShareEntry>, String> {
    // Prune expired entries before returning so the UI doesn't show zombies.
    let now = chrono::Utc::now().timestamp();
    {
        let mut shares = state.active_shares.write().await;
        shares.retain(|s| s.expires_at == 0 || s.expires_at > now);
    }
    Ok(state.active_shares.read().await.clone())
}
