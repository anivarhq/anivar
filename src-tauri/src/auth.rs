//! Secure desktop login core — Argon2id password hashing + Telegram OTP delivery
//! for the optional 2nd factor and password recovery.
//!
//! Threat model: this gate stops *casual physical access* to the desktop app (no
//! one can open it to view cameras/footage without the password). It is NOT a
//! sandbox against an attacker who fully controls the machine — the real hard
//! boundary is the remote/HTTP surface (token-gated + rate-limited elsewhere).
//!
//! Security properties:
//! - Passwords: **Argon2id** (salted PHC string), min 12 chars, hash encrypted at
//!   rest with the existing AES-GCM master key (see `crypto.rs`).
//! - OTPs (2FA + recovery): 6-digit, cryptographically random, stored **hashed**
//!   in memory with a short expiry, **single-use**, constant-time compare, and the
//!   sends are **throttled** so an attacker can't spam the owner's Telegram.
//! - Brute force: failed unlocks trigger an **exponential-backoff lockout** (never
//!   permanent — Telegram recovery always works).
//! - The `unlocked` state lives in the backend (not a frontend-only flag).

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use argon2::password_hash::{rand_core::OsRng, PasswordHash, SaltString};
use argon2::{Argon2, PasswordHasher, PasswordVerifier};
use rand::Rng;

pub const MIN_PASSWORD_LEN: usize = 12;

const OTP_2FA_TTL:       Duration = Duration::from_secs(300);  // 5 min
const OTP_RECOVERY_TTL:  Duration = Duration::from_secs(600);  // 10 min
const OTP_MAX_TRIES:     u32 = 5;
const SEND_COOLDOWN:     Duration = Duration::from_secs(60);
const SEND_MAX_PER_WIN:  usize = 3;
const SEND_WINDOW:       Duration = Duration::from_secs(600);
const LOCKOUT_AFTER:     u32 = 5;
const LOCKOUT_BASE:      Duration = Duration::from_secs(30);
const LOCKOUT_MAX:       Duration = Duration::from_secs(900);  // 15 min cap

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum OtpKind { TwoFactor, Recovery }

impl OtpKind {
    fn ttl(self) -> Duration {
        match self { OtpKind::TwoFactor => OTP_2FA_TTL, OtpKind::Recovery => OTP_RECOVERY_TTL }
    }
}

struct Challenge { code_hash: String, kind: OtpKind, expires: Instant, tries: u32 }

/// Runtime auth state — held in `AppState`. Nothing here is persisted (OTPs and
/// the unlocked flag deliberately reset on restart).
#[derive(Default)]
pub struct AuthState {
    unlocked: AtomicBool,
    inner: Mutex<AuthInner>,
}

#[derive(Default)]
struct AuthInner {
    challenges:   HashMap<String, Challenge>,
    login_fails:  u32,
    locked_until: Option<Instant>,
    send_times:   Vec<Instant>,
}

impl AuthState {
    pub fn is_unlocked(&self) -> bool { self.unlocked.load(Ordering::Acquire) }
    pub fn set_unlocked(&self, v: bool) { self.unlocked.store(v, Ordering::Release); }

    /// Remaining lockout, if the gate is currently locked out from brute-force.
    pub fn lockout_remaining(&self) -> Option<Duration> {
        let g = self.inner.lock().ok()?;
        g.locked_until.and_then(|t| t.checked_duration_since(Instant::now())).filter(|d| !d.is_zero())
    }

    /// Record a failed unlock. Returns `Some(remaining)` once the lockout trips.
    pub fn record_fail(&self) -> Option<Duration> {
        let mut g = self.inner.lock().ok()?;
        g.login_fails = g.login_fails.saturating_add(1);
        if g.login_fails >= LOCKOUT_AFTER {
            let extra = (g.login_fails - LOCKOUT_AFTER).min(20);
            let dur = LOCKOUT_BASE.saturating_mul(1u32 << extra.min(5)).min(LOCKOUT_MAX);
            let until = Instant::now() + dur;
            g.locked_until = Some(until);
            return Some(dur);
        }
        None
    }

    pub fn reset_fails(&self) {
        if let Ok(mut g) = self.inner.lock() { g.login_fails = 0; g.locked_until = None; }
    }

    /// Enforce the OTP send throttle (cooldown + max-per-window). Reuses the
    /// sliding-window idea so a single Telegram chat can't be flooded.
    fn check_send_throttle(&self) -> Result<(), String> {
        let mut g = self.inner.lock().map_err(|_| "auth busy")?;
        let now = Instant::now();
        g.send_times.retain(|t| now.duration_since(*t) < SEND_WINDOW);
        if let Some(last) = g.send_times.last() {
            if now.duration_since(*last) < SEND_COOLDOWN {
                let wait = SEND_COOLDOWN - now.duration_since(*last);
                return Err(format!("Please wait {}s before requesting another code.", wait.as_secs() + 1));
            }
        }
        if g.send_times.len() >= SEND_MAX_PER_WIN {
            return Err("Too many codes requested — try again later.".into());
        }
        g.send_times.push(now);
        Ok(())
    }

    /// Create a single-use OTP challenge; returns `(challenge_id, plaintext_code)`.
    /// Caller delivers the plaintext via Telegram; only the hash is retained.
    pub fn new_challenge(&self, kind: OtpKind) -> Result<(String, String), String> {
        self.check_send_throttle()?;
        let code = gen_otp();
        let id = uuid::Uuid::new_v4().to_string();
        let mut g = self.inner.lock().map_err(|_| "auth busy")?;
        // Drop any expired challenges opportunistically.
        let now = Instant::now();
        g.challenges.retain(|_, c| c.expires > now);
        g.challenges.insert(id.clone(), Challenge {
            code_hash: crate::http_handlers::sha256_hex(code.as_bytes()),
            kind, expires: now + kind.ttl(), tries: 0,
        });
        Ok((id, code))
    }

    /// Verify a code for a challenge. Single-use: consumed on success; removed
    /// after `OTP_MAX_TRIES` wrong attempts. Returns the challenge kind on success.
    pub fn verify_challenge(&self, id: &str, code: &str) -> Result<OtpKind, String> {
        let mut g = self.inner.lock().map_err(|_| "auth busy")?;
        let now = Instant::now();
        let Some(ch) = g.challenges.get_mut(id) else { return Err("This code has expired — request a new one.".into()) };
        if ch.expires <= now { g.challenges.remove(id); return Err("This code has expired — request a new one.".into()); }
        ch.tries += 1;
        if ch.tries > OTP_MAX_TRIES { g.challenges.remove(id); return Err("Too many attempts — request a new code.".into()); }
        let want = ch.code_hash.clone();
        let kind = ch.kind;
        let got = crate::http_handlers::sha256_hex(code.trim().as_bytes());
        if crate::constant_time_eq(want.as_bytes(), got.as_bytes()) {
            g.challenges.remove(id);
            Ok(kind)
        } else {
            Err("Incorrect code.".into())
        }
    }
}

/// 6-digit cryptographically-random one-time code.
fn gen_otp() -> String {
    let n: u32 = rand::thread_rng().gen_range(0..1_000_000);
    format!("{n:06}")
}

/// Hash a password with Argon2id (salted PHC string). Caller has already enforced
/// the strength policy.
pub fn hash_password(pw: &str) -> Result<String, String> {
    let salt = SaltString::generate(&mut OsRng);
    Argon2::default()
        .hash_password(pw.as_bytes(), &salt)
        .map(|h| h.to_string())
        .map_err(|e| format!("hash error: {e}"))
}

/// Constant-time-ish verify (Argon2 verification is inherently constant-time).
pub fn verify_password(pw: &str, phc: &str) -> bool {
    let Ok(parsed) = PasswordHash::new(phc) else { return false };
    Argon2::default().verify_password(pw.as_bytes(), &parsed).is_ok()
}

/// Reject weak passwords (length policy, NIST-aligned). Returns the reason.
pub fn validate_strength(pw: &str) -> Result<(), String> {
    if pw.chars().count() < MIN_PASSWORD_LEN {
        return Err(format!("Password must be at least {MIN_PASSWORD_LEN} characters."));
    }
    Ok(())
}

/// A random opaque token for "remember this device" (stored hashed in the DB).
pub fn gen_remember_token() -> String {
    let bytes: [u8; 32] = rand::random();
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
