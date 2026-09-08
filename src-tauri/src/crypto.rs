//! AES-256-GCM helpers for protecting secret settings fields at rest.
//!
//! * [`load_or_create_master_key`] loads (or generates) the local symmetric key
//!   from `.master_key` in the app-data dir. Created with `0o600` perms on
//!   Unix.
//! * [`encrypt_secret`] / [`decrypt_secret`] handle one value at a time. The
//!   wire format is `"enc:" || base64(nonce) || ":" || base64(ciphertext)`.
//! * [`encrypt_settings_secrets`] / [`decrypt_settings_secrets`] walk every
//!   secret field on a [`Settings`] struct — applied symmetrically on save and
//!   load so the on-disk DB never contains plaintext credentials.

use base64::{engine::general_purpose::STANDARD as B64, Engine};

use crate::Settings;

// ─── Encryption helpers (AES-256-GCM for sensitive settings) ─────────────────
use aes_gcm::{Aes256Gcm, KeyInit, aead::{Aead, AeadCore, OsRng as AeadOsRng}};

pub(crate) fn load_or_create_master_key(data_dir: &std::path::Path) -> [u8; 32] {
    let key_path = data_dir.join(".master_key");
    if let Ok(bytes) = std::fs::read(&key_path) {
        if bytes.len() == 32 {
            let mut key = [0u8; 32];
            key.copy_from_slice(&bytes);
            return key;
        }
    }
    let key: [u8; 32] = rand::random();
    let _ = std::fs::write(&key_path, key);
    #[cfg(unix)] {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&key_path, std::fs::Permissions::from_mode(0o600));
    }
    key
}

pub(crate) fn encrypt_secret(key: &[u8; 32], plaintext: &str) -> String {
    if plaintext.is_empty() { return String::new(); }
    let cipher = Aes256Gcm::new_from_slice(key).expect("valid key");
    let nonce = Aes256Gcm::generate_nonce(&mut AeadOsRng);
    let ciphertext = cipher.encrypt(&nonce, plaintext.as_bytes()).unwrap_or_default();
    format!("enc:{}:{}", B64.encode(nonce), B64.encode(ciphertext))
}

pub(crate) fn decrypt_secret(key: &[u8; 32], stored: &str) -> String {
    if stored.is_empty() || !stored.starts_with("enc:") { return stored.to_string(); }
    let rest = &stored[4..];
    let colon = match rest.find(':') { Some(i) => i, None => return stored.to_string() };
    let nonce_b64 = &rest[..colon];
    let ct_b64 = &rest[colon + 1..];
    let Ok(nonce_bytes) = B64.decode(nonce_b64) else { return stored.to_string(); };
    let Ok(ct_bytes)    = B64.decode(ct_b64)    else { return stored.to_string(); };
    if nonce_bytes.len() != 12 { return stored.to_string(); }
    let cipher = Aes256Gcm::new_from_slice(key).expect("valid key");
    let nonce  = aes_gcm::Nonce::from_slice(&nonce_bytes);
    cipher.decrypt(nonce, ct_bytes.as_ref()).ok()
        .and_then(|v| String::from_utf8(v).ok())
        .unwrap_or_else(|| stored.to_string())
}

/// The settings fields that contain secrets and must be encrypted at rest.
/// Applied symmetrically in save_settings_to_db (encrypt) and load (decrypt).
pub(crate) fn encrypt_settings_secrets(key: &[u8; 32], s: &mut Settings) {
    s.telegram_bot_token   = encrypt_secret(key, &s.telegram_bot_token);
    s.openai_api_key       = encrypt_secret(key, &s.openai_api_key);
    s.anthropic_api_key    = encrypt_secret(key, &s.anthropic_api_key);
    s.groq_api_key         = encrypt_secret(key, &s.groq_api_key);
    s.xai_api_key          = encrypt_secret(key, &s.xai_api_key);
    s.gemini_api_key       = encrypt_secret(key, &s.gemini_api_key);
    s.openai_compatible_key = encrypt_secret(key, &s.openai_compatible_key);
    s.login_password_hash  = encrypt_secret(key, &s.login_password_hash);
}

pub(crate) fn decrypt_settings_secrets(key: &[u8; 32], s: &mut Settings) {
    s.telegram_bot_token   = decrypt_secret(key, &s.telegram_bot_token);
    s.openai_api_key       = decrypt_secret(key, &s.openai_api_key);
    s.anthropic_api_key    = decrypt_secret(key, &s.anthropic_api_key);
    s.groq_api_key         = decrypt_secret(key, &s.groq_api_key);
    s.xai_api_key          = decrypt_secret(key, &s.xai_api_key);
    s.gemini_api_key       = decrypt_secret(key, &s.gemini_api_key);
    s.openai_compatible_key = decrypt_secret(key, &s.openai_compatible_key);
    s.login_password_hash  = decrypt_secret(key, &s.login_password_hash);
}
