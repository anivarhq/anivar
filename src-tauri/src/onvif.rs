//! ONVIF SOAP helpers.
//!
//! Minimal SOAP envelope construction + WS-UsernameToken (digest) auth for
//! ONVIF cameras. The two public commands (`discover_onvif`, `get_onvif_streams`)
//! that consume these helpers live with the rest of the Tauri commands in
//! [`lib.rs`].

use base64::{engine::general_purpose::STANDARD as B64, Engine};


/// Simple XML text extractor — finds first <tag>text</tag> or <ns:tag>text</ns:tag>.
pub(crate) fn extract_xml_text(xml: &str, local_tag: &str) -> Option<String> {
    // Try both bare and namespaced forms
    for prefix in &["", "tt:", "trt:", "tds:", "dn:"] {
        let open  = format!("<{}{}>", prefix, local_tag);
        let close = format!("</{}{}>", prefix, local_tag);
        if let Some(s) = xml.find(&open) {
            let start = s + open.len();
            if let Some(e) = xml[start..].find(&close) {
                return Some(xml[start..start + e].trim().to_string());
            }
        }
    }
    // Also try attribute form: token="VALUE"
    None
}

/// Send an ONVIF SOAP request with optional WS-UsernameToken auth.
pub(crate) async fn onvif_request(
    device_url: &str,
    body: &str,
    username: &Option<String>,
    password: &Option<String>,
) -> anyhow::Result<String> {
    let security_header = if let (Some(u), Some(p)) = (username, password) {
        if !u.is_empty() {
            let created = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();
            let nonce_bytes: [u8; 16] = rand::random();
            let nonce_b64 = B64.encode(nonce_bytes);
            use sha2::Digest;
            let mut hasher = sha2::Sha256::new();
            hasher.update(nonce_bytes);
            hasher.update(created.as_bytes());
            hasher.update(p.as_bytes());
            let digest = B64.encode(hasher.finalize());
            format!(r#"<wsse:Security xmlns:wsse="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd" xmlns:wsu="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd"><wsse:UsernameToken><wsse:Username>{u}</wsse:Username><wsse:Password Type="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0#PasswordDigest">{digest}</wsse:Password><wsse:Nonce EncodingType="http://docs.oasis-open.org/wss/2004/08/security#Base64Binary">{nonce_b64}</wsse:Nonce><wsu:Created>{created}</wsu:Created></wsse:UsernameToken></wsse:Security>"#)
        } else { String::new() }
    } else { String::new() };

    let envelope = format!(r#"<?xml version="1.0" encoding="UTF-8"?><s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"><s:Header>{security_header}</s:Header><s:Body>{body}</s:Body></s:Envelope>"#);

    let client = reqwest::Client::builder().timeout(std::time::Duration::from_secs(8)).build()?;
    let resp = client.post(device_url)
        .header("Content-Type", "application/soap+xml; charset=utf-8")
        .body(envelope).send().await?;
    Ok(resp.text().await?)
}

