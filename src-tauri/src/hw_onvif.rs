//! Hardware encoder query + Windows Firewall fix + ONVIF discovery/profiles + Skill download/install.

use std::sync::Arc;

use tauri::State;
use uuid::Uuid;

use crate::AppState;
use crate::onvif::{extract_xml_text, onvif_request};


#[tauri::command]
pub async fn get_hw_encoder(state: State<'_, Arc<AppState>>) -> Result<String, String> {
    Ok(state.hw_encoder.read().unwrap().clone())
}

/// Open Windows Firewall for the stream port using UAC-elevated PowerShell.
/// Writes commands to a temp .ps1 file — avoids all nested-quote escaping issues.
#[tauri::command]
pub async fn fix_firewall(state: State<'_, Arc<AppState>>) -> Result<String, String> {
    // Consumed by the Windows firewall rule below; unused on other targets.
    #[allow(unused_variables)]
    let port = state.settings.read().await.stream_port;

    #[cfg(target_os = "windows")]
    {
        // Write the netsh commands to a temp script — no escaping needed
        let script_path = std::env::temp_dir().join("anivar_firewall.ps1");
        // The pre-rename rules are deleted first: a renamed app leaves its old
        // inbound-allow rules behind forever, and a stale hole in the firewall
        // is worse than an untidy list. `delete` is a no-op when absent.
        let script = format!(
            "netsh advfirewall firewall delete rule name=\"SecureCam\" >$null 2>&1\r\n\
             netsh advfirewall firewall delete rule name=\"SecureCam-P2P\" >$null 2>&1\r\n\
             netsh advfirewall firewall delete rule name=\"Anvil NVR\" >$null 2>&1\r\n\
             netsh advfirewall firewall delete rule name=\"Anvil NVR-P2P\" >$null 2>&1\r\n\
             netsh advfirewall firewall delete rule name=\"Nivar\" >$null 2>&1\r\n\
             netsh advfirewall firewall delete rule name=\"Nivar-P2P\" >$null 2>&1\r\n\
             netsh advfirewall firewall add rule name=\"Anivar\" protocol=TCP dir=in action=allow localport={port} enable=yes profile=any\r\n\
             netsh advfirewall firewall add rule name=\"Anivar-P2P\" protocol=TCP dir=in action=allow localport=4002-4003 enable=yes profile=any\r\n\
             Write-Host \"Anivar firewall rules added! Port {port} and 4002-4003 are now open.\" -ForegroundColor Green\r\n\
             Start-Sleep 2\r\n"
        );
        tokio::fs::write(&script_path, script).await.map_err(|e| e.to_string())?;

        // Run the script with -Verb RunAs — triggers UAC prompt
        crate::proc::tokio_cmd("powershell")
            .args([
                "-NoProfile", "-Command",
                &format!(
                    "Start-Process powershell -Verb RunAs -Wait -ArgumentList '-NoProfile -ExecutionPolicy Bypass -File \"{}\"'",
                    script_path.display()
                ),
            ])
            .spawn()
            .map_err(|e| e.to_string())?;

        Ok(format!("UAC prompt opened — allow it to add firewall rule for port {}", port))
    }

    #[cfg(not(target_os = "windows"))]
    Ok("Not needed on this platform".to_string())
}

/// ONVIF WS-Discovery — sends UDP multicast probe, returns discovered devices.
#[tauri::command]
pub async fn discover_onvif(timeout_ms: Option<u64>) -> Result<Vec<serde_json::Value>, String> {
    use std::net::{UdpSocket, SocketAddr};
    use std::time::Duration;

    let timeout  = Duration::from_millis(timeout_ms.unwrap_or(3000));
    let probe_id = Uuid::new_v4();
    let probe = format!(r#"<?xml version="1.0" encoding="UTF-8"?>
<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope" xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing" xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery" xmlns:dn="http://www.onvif.org/ver10/network/wsdl">
  <e:Header><w:MessageID>uuid:{probe_id}</w:MessageID><w:To>urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To><w:Action>http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action></e:Header>
  <e:Body><d:Probe><d:Types>dn:NetworkVideoTransmitter</d:Types></d:Probe></e:Body>
</e:Envelope>"#);

    let results = tokio::task::spawn_blocking(move || -> Result<Vec<serde_json::Value>, String> {
        let s = UdpSocket::bind("0.0.0.0:0").map_err(|e| e.to_string())?;
        s.set_broadcast(true).ok();
        s.set_read_timeout(Some(timeout)).ok();
        let dest: SocketAddr = "239.255.255.250:3702".parse().unwrap();
        s.send_to(probe.as_bytes(), dest).map_err(|e| e.to_string())?;

        let mut results: Vec<serde_json::Value> = Vec::new();
        let mut buf = vec![0u8; 65536];
        let deadline = std::time::Instant::now() + timeout;
        loop {
            if std::time::Instant::now() >= deadline { break; }
            match s.recv_from(&mut buf) {
                Ok((n, addr)) => {
                    let xml = String::from_utf8_lossy(&buf[..n]).to_string();
                    let xaddrs = extract_xml_text(&xml, "XAddrs").unwrap_or_default();
                    if xaddrs.is_empty() { continue; }
                    let device_url = xaddrs.split_whitespace().next().unwrap_or("").to_string();
                    results.push(serde_json::json!({
                        "device_url": device_url,
                        "xaddrs": xaddrs,
                        "source_ip": addr.ip().to_string(),
                    }));
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock
                       || e.kind() == std::io::ErrorKind::TimedOut => break,
                Err(_) => break,
            }
        }
        Ok(results)
    }).await.map_err(|e| e.to_string())??;

    Ok(results)
}

/// Fetch ONVIF media profiles and RTSP stream URIs from a specific device.
#[tauri::command]
pub async fn get_onvif_streams(
    device_url: String,
    username: Option<String>,
    password: Option<String>,
) -> Result<Vec<serde_json::Value>, String> {
    let profiles_body = r#"<trt:GetProfiles xmlns:trt="http://www.onvif.org/ver10/media/wsdl"/>"#;
    let xml = onvif_request(&device_url, profiles_body, &username, &password).await
        .map_err(|e| format!("GetProfiles: {e}"))?;

    let mut results = Vec::new();
    // Walk all token="" attributes to collect profile tokens
    let mut search = xml.as_str();
    while let Some(tok_start) = search.find("token=\"") {
        let rest = &search[tok_start + 7..];
        let tok_end = rest.find('"').unwrap_or(rest.len());
        let token = rest[..tok_end].to_string();
        search = if tok_start + 7 + tok_end + 1 < search.len() {
            &search[tok_start + 7 + tok_end + 1..]
        } else { break; };

        let uri_body = format!(r#"<trt:GetStreamUri xmlns:trt="http://www.onvif.org/ver10/media/wsdl"><trt:StreamSetup><tt:Stream xmlns:tt="http://www.onvif.org/ver10/schema">RTP-Unicast</tt:Stream><tt:Transport xmlns:tt="http://www.onvif.org/ver10/schema"><tt:Protocol>RTSP</tt:Protocol></tt:Transport></trt:StreamSetup><trt:ProfileToken>{token}</trt:ProfileToken></trt:GetStreamUri>"#);
        let uri_xml = onvif_request(&device_url, &uri_body, &username, &password).await.unwrap_or_default();
        let mut rtsp = extract_xml_text(&uri_xml, "Uri").unwrap_or_default();
        // Inject credentials into RTSP URL if provided
        if let (Some(u), Some(p)) = (&username, &password) {
            if !u.is_empty() && rtsp.starts_with("rtsp://") {
                rtsp = rtsp.replacen("rtsp://", &format!("rtsp://{}:{}@", u, p), 1);
            }
        }
        if !rtsp.is_empty() {
            results.push(serde_json::json!({
                "profile_token": token,
                "rtsp_url": rtsp,
                "device_url": device_url,
            }));
        }
    }
    Ok(results)
}

/// Get device information (model, firmware, manufacturer) from ONVIF device service.
#[tauri::command]
pub async fn get_onvif_device_info(
    device_url: String,
    username: Option<String>,
    password: Option<String>,
) -> Result<serde_json::Value, String> {
    let body = r#"<tds:GetDeviceInformation xmlns:tds="http://www.onvif.org/ver10/device/wsdl"/>"#;
    let xml = onvif_request(&device_url, body, &username, &password).await
        .map_err(|e| format!("GetDeviceInformation: {e}"))?;
    Ok(serde_json::json!({
        "manufacturer":      extract_xml_text(&xml, "Manufacturer").unwrap_or_default(),
        "model":             extract_xml_text(&xml, "Model").unwrap_or_default(),
        "firmware_version":  extract_xml_text(&xml, "FirmwareVersion").unwrap_or_default(),
        "serial_number":     extract_xml_text(&xml, "SerialNumber").unwrap_or_default(),
        "hardware_id":       extract_xml_text(&xml, "HardwareId").unwrap_or_default(),
    }))
}

/// One-shot: discover all ONVIF cameras on LAN and return their stream URIs + device info.
#[tauri::command]
pub async fn discover_and_configure_onvif(
    username: Option<String>,
    password: Option<String>,
    timeout_ms: Option<u64>,
) -> Result<Vec<serde_json::Value>, String> {
    let devices = discover_onvif(timeout_ms).await?;
    let mut all = Vec::new();
    for dev in &devices {
        let url = dev["device_url"].as_str().unwrap_or("").to_string();
        if url.is_empty() { continue; }
        let info    = get_onvif_device_info(url.clone(), username.clone(), password.clone()).await.unwrap_or(serde_json::json!({}));
        let streams = get_onvif_streams(url.clone(), username.clone(), password.clone()).await.unwrap_or_default();
        for mut s in streams {
            if let Some(obj) = s.as_object_mut() {
                obj.insert("manufacturer".into(), info["manufacturer"].clone());
                obj.insert("model".into(),        info["model"].clone());
                obj.insert("source_ip".into(),    dev["source_ip"].clone());
            }
            all.push(s);
        }
    }
    Ok(all)
}
