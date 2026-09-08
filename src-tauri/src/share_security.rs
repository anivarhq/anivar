//! v11 share-link security primitives.
//!
//! Two concerns live here:
//!   1. **Signed share tokens** — HMAC-SHA256 over the share payload, mixed
//!      with the per-app `master_key` and a monotonically-increasing
//!      `share_generation` counter. Revoking all shares = bump the counter;
//!      every outstanding token becomes invalid in O(1).
//!   2. **Per-IP rate limiter** — token-bucket-ish (sliding window) that
//!      caps how fast a single remote IP can hammer the share routes. Used
//!      by the `/redeem`, `/clips/*`, `/live/*` handlers.
//!
//! Tokens are URL-safe base64 of:
//!     base16(generation_u64_be) "." base64(payload_json) "." base16(hmac16)
//! The 16-byte truncated HMAC is plenty for shares that expire within hours
//! and 64 bits of unforgeability — more is overkill for personal use.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;

use hmac::{Hmac, Mac};
use sha2::Sha256;
use base64::Engine as _;
use base64::engine::general_purpose::URL_SAFE_NO_PAD as B64URL;
use serde::{Deserialize, Serialize};

type HmacSha256 = Hmac<Sha256>;

/// The body of a share token. Kind decides which HTTP route the token grants
/// access to; resource_id picks WHICH camera or event within that kind.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharePayload {
    pub kind:        String,   // "live" | "clip"
    pub resource_id: String,   // cam_id (as str) for live, event_id for clip
    pub expires_at:  i64,      // unix epoch seconds; 0 = never (until restart)
}

/// Sign a share payload. The `generation` is mixed into the HMAC so a global
/// revoke (incrementing the counter) immediately invalidates the token without
/// needing a per-token table to delete from.
pub fn sign_share_token(master_key: &[u8; 32], generation: u64, payload: &SharePayload) -> String {
    let payload_json = serde_json::to_vec(payload).unwrap_or_default();
    let payload_b64  = B64URL.encode(&payload_json);
    let gen_b64      = B64URL.encode(generation.to_be_bytes());

    let mut mac = HmacSha256::new_from_slice(master_key).expect("HMAC key");
    mac.update(gen_b64.as_bytes());
    mac.update(b".");
    mac.update(payload_b64.as_bytes());
    let tag = mac.finalize().into_bytes();
    let tag_b64 = B64URL.encode(&tag[..16]); // 128-bit truncation

    format!("{}.{}.{}", gen_b64, payload_b64, tag_b64)
}

/// Verify a share token. Returns the parsed payload if:
///   * the format parses,
///   * the HMAC matches under the current generation, AND
///   * the embedded expiry hasn't passed (0 = never expires).
pub fn verify_share_token(master_key: &[u8; 32], generation: u64, token: &str) -> Option<SharePayload> {
    let mut parts = token.splitn(3, '.');
    let gen_b64     = parts.next()?;
    let payload_b64 = parts.next()?;
    let tag_b64     = parts.next()?;

    // Generation must match the current counter exactly. Re-encode the
    // current generation under the same scheme so we compare bytes-of-bytes,
    // not number-of-bytes — avoids subtle Base64 padding mismatches.
    let expected_gen_b64 = B64URL.encode(generation.to_be_bytes());
    if !ct_eq(gen_b64.as_bytes(), expected_gen_b64.as_bytes()) { return None; }

    let mut mac = HmacSha256::new_from_slice(master_key).ok()?;
    mac.update(gen_b64.as_bytes());
    mac.update(b".");
    mac.update(payload_b64.as_bytes());
    let expected_tag = mac.finalize().into_bytes();
    let provided_tag = B64URL.decode(tag_b64).ok()?;
    if !ct_eq(&provided_tag, &expected_tag[..16.min(expected_tag.len())]) { return None; }

    let payload_json = B64URL.decode(payload_b64).ok()?;
    let payload: SharePayload = serde_json::from_slice(&payload_json).ok()?;
    if payload.expires_at != 0 {
        let now = chrono::Utc::now().timestamp();
        if now > payload.expires_at { return None; }
    }
    Some(payload)
}

fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() { return false; }
    a.iter().zip(b.iter()).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// Sliding-window rate limiter keyed by client IP. One global mutex per
/// instance is fine — entries are cheap and contention is bounded by the
/// number of distinct remote IPs hitting the tunnel.
pub struct RateLimiter {
    limit_per_window:   u32,
    window:             std::time::Duration,
    state:              Mutex<HashMap<String, (Instant, u32)>>,
}

impl RateLimiter {
    pub fn new(limit_per_window: u32, window: std::time::Duration) -> Self {
        Self {
            limit_per_window,
            window,
            state: Mutex::new(HashMap::new()),
        }
    }

    /// Check the bucket for `ip`. Returns `(allowed, retry_after_secs)`.
    /// On allow, the bucket is incremented as a side effect.
    pub fn check(&self, ip: &str) -> (bool, u64) {
        let now = Instant::now();
        let mut state = match self.state.lock() {
            Ok(g) => g,
            Err(_) => return (true, 0), // fail-open on lock poisoning
        };
        // Opportunistic GC so the per-IP map can't grow unboundedly. Runs only when the
        // map gets large (cheap amortised), since there's no background sweeper. Drops
        // entries whose window expired comfortably ago.
        if state.len() > 256 {
            state.retain(|_, (started, _)| now.duration_since(*started) < self.window * 4);
        }
        let entry = state.entry(ip.to_string()).or_insert((now, 0));
        if now.duration_since(entry.0) >= self.window {
            *entry = (now, 1);
            return (true, 0);
        }
        if entry.1 < self.limit_per_window {
            entry.1 += 1;
            (true, 0)
        } else {
            let elapsed = now.duration_since(entry.0);
            let retry_after = self.window.saturating_sub(elapsed).as_secs().max(1);
            (false, retry_after)
        }
    }
}
