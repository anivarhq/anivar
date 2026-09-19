//! ONVIF SOAP helpers.
//!
//! Minimal SOAP envelope construction + WS-UsernameToken (digest) auth for
//! ONVIF cameras. The two public commands (`discover_onvif`, `get_onvif_streams`)
//! that consume these helpers live with the rest of the Tauri commands in
//! [`lib.rs`].

use base64::{engine::general_purpose::STANDARD as B64, Engine};


/// Text of the first `local_tag` element, whatever its namespace prefix:
/// `<XAddrs>`, `<d:XAddrs>` and `<wsdd:XAddrs xmlns:wsdd="…">` all match.
/// Cameras choose their own prefixes, so a fixed list silently drops devices.
pub(crate) fn extract_xml_text(xml: &str, local_tag: &str) -> Option<String> {
    let mut rest = xml;
    while let Some(lt) = rest.find('<') {
        rest = &rest[lt + 1..];
        let name_len = rest.find(|c: char| c == '>' || c == '/' || c.is_whitespace())?;
        let name = &rest[..name_len];
        if name.rsplit(':').next() != Some(local_tag) { continue; }
        let gt = rest.find('>')?;
        if rest[..gt].ends_with('/') { return Some(String::new()); }
        let body = &rest[gt + 1..];
        let end = body.find(&format!("</{name}>"))?;
        return Some(body[..end].trim().to_string());
    }
    None
}

/// WS-UsernameToken PasswordDigest: Base64(SHA-1(nonce + created + password)),
/// nonce as raw bytes. The spec and every camera use SHA-1.
pub(crate) fn wsse_password_digest(nonce: &[u8], created: &str, password: &str) -> String {
    use sha1::{Digest, Sha1};
    let mut hasher = Sha1::new();
    hasher.update(nonce);
    hasher.update(created.as_bytes());
    hasher.update(password.as_bytes());
    B64.encode(hasher.finalize())
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
            let digest = wsse_password_digest(&nonce_bytes, &created, p);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn password_digest_matches_onvif_spec_example() {
        // ONVIF Application Programmer's Guide, WS-UsernameToken example.
        let nonce = B64.decode("LKqI6G/AikKCQrN0zqZFlg==").unwrap();
        assert_eq!(
            wsse_password_digest(&nonce, "2010-09-16T07:50:45Z", "userpassword"),
            "tuOSpGlFlIXsozq4HFNeeGeFLEI="
        );
    }

    #[test]
    fn extract_ignores_namespace_prefix() {
        let url = "http://192.168.1.64/onvif/device_service";
        for xml in [
            format!("<d:ProbeMatch><d:XAddrs>{url}</d:XAddrs></d:ProbeMatch>"),
            format!("<wsdd:XAddrs>{url}</wsdd:XAddrs>"),
            format!("<XAddrs>{url}</XAddrs>"),
            format!(r#"<d:XAddrs xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery"> {url} </d:XAddrs>"#),
        ] {
            assert_eq!(extract_xml_text(&xml, "XAddrs").as_deref(), Some(url), "{xml}");
        }
    }

    #[test]
    fn extract_matches_whole_local_name_only() {
        let xml = "<trt:MediaUri><tt:Uri>rtsp://cam/main</tt:Uri></trt:MediaUri>";
        assert_eq!(extract_xml_text(xml, "Uri").as_deref(), Some("rtsp://cam/main"));
        assert_eq!(extract_xml_text("</tt:Uri><tt:Uri>x</tt:Uri>", "Uri").as_deref(), Some("x"));
        assert_eq!(extract_xml_text("<tt:Uri/>", "Uri").as_deref(), Some(""));
        assert_eq!(extract_xml_text("<tds:Model>x</tds:Model>", "Uri"), None);
    }
}

