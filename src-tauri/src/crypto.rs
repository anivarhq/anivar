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

/// Load the key, or create one on a genuine first run.
///
/// An existing key file is NEVER overwritten. It used to be: any read error or
/// wrong length wrote a fresh key over the old file, which made every stored
/// secret undecryptable for good — including the login password hash, which
/// locked the user out with no working recovery (that needs the Telegram token,
/// also lost). Now a malformed file is moved aside (`.master_key.unreadable-<t>`)
/// so it can still be recovered, and a file that merely couldn't be READ (locked,
/// permissions) is left alone: this run uses a temporary key and the next start
/// tries the real one again.
pub(crate) fn load_or_create_master_key(data_dir: &std::path::Path) -> [u8; 32] {
    let key_path = data_dir.join(".master_key");
    match std::fs::read(&key_path) {
        Ok(bytes) if bytes.len() == 32 => {
            let mut key = [0u8; 32];
            key.copy_from_slice(&bytes);
            return key;
        }
        Ok(bytes) => {
            let aside = data_dir.join(format!(".master_key.unreadable-{}", chrono::Utc::now().timestamp()));
            tracing::error!("crypto: .master_key is {} bytes, not 32 — moving it to {} and creating a new key; \
                             secrets saved under the old key will need re-entering", bytes.len(), aside.display());
            if let Err(e) = std::fs::rename(&key_path, &aside) {
                tracing::error!("crypto: couldn't move the old key aside ({e}); using a temporary key and leaving it in place");
                return rand::random();
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {} // first run
        Err(e) => {
            tracing::error!("crypto: couldn't read .master_key ({e}); using a temporary key for this run and \
                             NOT replacing the file");
            return rand::random();
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
    // Already encrypted: wrapping it again made it undecryptable in one step.
    if plaintext.starts_with("enc:") { return plaintext.to_string(); }
    let cipher = Aes256Gcm::new_from_slice(key).expect("valid key");
    let nonce = Aes256Gcm::generate_nonce(&mut AeadOsRng);
    let ciphertext = cipher.encrypt(&nonce, plaintext.as_bytes()).unwrap_or_default();
    format!("enc:{}:{}", B64.encode(nonce), B64.encode(ciphertext))
}

/// `Ok` with the plaintext (an empty or never-encrypted value passes through),
/// `Err` when the value IS encrypted but can't be decrypted with this key. A
/// failure used to return the ciphertext itself, which was then sent to OpenAI
/// or Telegram as if it were the key.
fn try_decrypt(key: &[u8; 32], stored: &str) -> Result<String, ()> {
    if stored.is_empty() || !stored.starts_with("enc:") { return Ok(stored.to_string()); }
    let rest = &stored[4..];
    let colon = rest.find(':').ok_or(())?;
    let nonce_bytes = B64.decode(&rest[..colon]).map_err(|_| ())?;
    let ct_bytes    = B64.decode(&rest[colon + 1..]).map_err(|_| ())?;
    if nonce_bytes.len() != 12 { return Err(()); }
    let cipher = Aes256Gcm::new_from_slice(key).expect("valid key");
    let nonce  = aes_gcm::Nonce::from_slice(&nonce_bytes);
    cipher.decrypt(nonce, ct_bytes.as_ref()).ok()
        .and_then(|v| String::from_utf8(v).ok())
        .ok_or(())
}

/// Secret fields that were stored encrypted but couldn't be decrypted this run
/// (the key changed), with their stored ciphertext. They read as empty in
/// memory, and the ciphertext is written back on save, so saving Settings for
/// some other reason never erases a value the original key could still open.
static UNREADABLE: std::sync::Mutex<Vec<(&'static str, String)>> = std::sync::Mutex::new(Vec::new());

/// Human names of the secrets that couldn't be decrypted, for Settings to show.
pub(crate) fn unreadable_secrets() -> Vec<&'static str> {
    UNREADABLE.lock().map(|u| u.iter().map(|(f, _)| label(f)).collect()).unwrap_or_default()
}

pub(crate) fn login_hash_unreadable() -> bool {
    UNREADABLE.lock().map(|u| u.iter().any(|(f, _)| *f == "login_password_hash")).unwrap_or(false)
}

fn label(field: &str) -> &'static str {
    match field {
        "telegram_bot_token"    => "Telegram bot token",
        "openai_api_key"        => "OpenAI API key",
        "anthropic_api_key"     => "Anthropic API key",
        "groq_api_key"          => "Groq API key",
        "xai_api_key"           => "xAI API key",
        "gemini_api_key"        => "Gemini API key",
        "openai_compatible_key" => "OpenAI-compatible API key",
        "mqtt_password"         => "MQTT broker password",
        _                       => "login password",
    }
}

/// Every secret field on `Settings`, by name, so encrypt and decrypt can't drift.
fn secret_fields(s: &mut Settings) -> [(&'static str, &mut String); 9] {
    [
        ("telegram_bot_token",    &mut s.telegram_bot_token),
        ("openai_api_key",        &mut s.openai_api_key),
        ("anthropic_api_key",     &mut s.anthropic_api_key),
        ("groq_api_key",          &mut s.groq_api_key),
        ("xai_api_key",           &mut s.xai_api_key),
        ("gemini_api_key",        &mut s.gemini_api_key),
        ("openai_compatible_key", &mut s.openai_compatible_key),
        ("login_password_hash",   &mut s.login_password_hash),
        ("mqtt_password",         &mut s.mqtt_password),
    ]
}

/// The settings fields that contain secrets and must be encrypted at rest.
/// Applied symmetrically in save (apply_settings_update) and load (boot).
pub(crate) fn encrypt_settings_secrets(key: &[u8; 32], s: &mut Settings) {
    let mut unreadable = UNREADABLE.lock().unwrap_or_else(|e| e.into_inner());
    for (name, value) in secret_fields(s) {
        if value.is_empty() {
            // Still empty because it couldn't be decrypted: keep what's stored.
            if let Some((_, ct)) = unreadable.iter().find(|(f, _)| *f == name) {
                *value = ct.clone();
            }
        } else {
            // A new value replaces the unreadable one.
            unreadable.retain(|(f, _)| *f != name);
            *value = encrypt_secret(key, value);
        }
    }
}

pub(crate) fn decrypt_settings_secrets(key: &[u8; 32], s: &mut Settings) {
    let mut unreadable = UNREADABLE.lock().unwrap_or_else(|e| e.into_inner());
    unreadable.clear();
    for (name, value) in secret_fields(s) {
        match try_decrypt(key, value) {
            Ok(plain) => *value = plain,
            Err(()) => {
                tracing::error!("crypto: the stored {} can't be decrypted with this key — \
                                 it reads as empty until re-entered", label(name));
                unreadable.push((name, std::mem::take(value)));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_and_no_double_encryption() {
        let key = [7u8; 32];
        let ct = encrypt_secret(&key, "sk-secret");
        assert!(ct.starts_with("enc:"));
        assert_eq!(try_decrypt(&key, &ct).unwrap(), "sk-secret");
        assert_eq!(encrypt_secret(&key, &ct), ct, "encrypting a ciphertext is a no-op");
    }

    #[test]
    fn the_mqtt_broker_password_is_stored_encrypted() {
        let key = [3u8; 32];
        let mut s = Settings { mqtt_password: "broker-pass".into(), ..Settings::default() };
        encrypt_settings_secrets(&key, &mut s);
        assert!(s.mqtt_password.starts_with("enc:"), "{}", s.mqtt_password);
        decrypt_settings_secrets(&key, &mut s);
        assert_eq!(s.mqtt_password, "broker-pass");
    }

    #[test]
    fn a_wrong_key_never_yields_the_ciphertext() {
        let ct = encrypt_secret(&[1u8; 32], "sk-secret");
        assert!(try_decrypt(&[2u8; 32], &ct).is_err(), "used to return the enc: string itself");
    }

    #[test]
    fn an_unreadable_secret_survives_an_unrelated_save() {
        let (old, new) = ([1u8; 32], [2u8; 32]);
        let mut s = Settings::default();
        s.openai_api_key = encrypt_secret(&old, "sk-openai");
        s.telegram_bot_token = encrypt_secret(&old, "123:tg");
        let stored_openai = s.openai_api_key.clone();

        decrypt_settings_secrets(&new, &mut s); // the key changed
        assert_eq!(s.openai_api_key, "");
        assert!(unreadable_secrets().contains(&"OpenAI API key"));

        // The user re-enters only the Telegram token and saves.
        s.telegram_bot_token = "456:tg".into();
        let mut saved = s.clone();
        encrypt_settings_secrets(&new, &mut saved);
        assert_eq!(saved.openai_api_key, stored_openai, "the unreadable value is kept, not wiped");
        assert_eq!(try_decrypt(&new, &saved.telegram_bot_token).unwrap(), "456:tg");
        assert!(!unreadable_secrets().contains(&"Telegram bot token"));
    }

    #[test]
    fn a_malformed_key_file_is_moved_aside_not_overwritten() {
        let dir = std::env::temp_dir().join(format!("anivar-key-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(".master_key"), b"short").unwrap();
        let key = load_or_create_master_key(&dir);
        assert_eq!(std::fs::read(dir.join(".master_key")).unwrap(), key.to_vec());
        let kept = std::fs::read_dir(&dir).unwrap().filter_map(|e| e.ok())
            .any(|e| e.file_name().to_string_lossy().starts_with(".master_key.unreadable-")
                  && std::fs::read(e.path()).unwrap() == b"short");
        assert!(kept, "the old key file must survive");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
