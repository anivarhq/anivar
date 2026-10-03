//! MQTT bridge: events, per-camera motion and online state, and Home Assistant
//! discovery, published to a broker you run (Mosquitto, the Home Assistant
//! Mosquitto add-on, …). Off until a broker host is set in Settings.
//!
//! Fire-and-forget by design. Recording never waits on the broker: messages go
//! through a bounded queue to ONE long-lived connection (rumqttc reconnects by
//! itself), and are dropped rather than queued without limit when the broker is
//! slow or down. An earlier version opened a fresh connection per event.
//!
//! Topics (`p` = the topic prefix, "anivar" by default):
//!   p/available            "online" / "offline"  retained; "offline" is the last will
//!   p/camN/motion          "ON" / "OFF"          retained
//!   p/camN/online          "ON" / "OFF"          retained
//!   p/camN/last_detection  label of the last finished event, retained
//!   p/camN/event           JSON, one per event start and end, not retained
//!   homeassistant/…/config discovery, retained, so entities appear by themselves
//!
//! N is the 1-based camera number the app shows. State only: quiet hours and
//! alert mutes don't apply here, because a home-automation system needs the
//! truth about motion, not the alert policy.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use rumqttc::{AsyncClient, Event, LastWill, MqttOptions, Packet, QoS};
use serde_json::json;
use tokio::sync::mpsc;

use crate::state::AppState;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Msg { pub topic: String, pub payload: String, pub retain: bool }

/// The live connection's queue and the prefix it publishes under. `None` while
/// the bridge is off.
struct Link { tx: mpsc::Sender<Msg>, prefix: String, task: tokio::task::JoinHandle<()> }
static LINK: Mutex<Option<Link>> = Mutex::new(None);

fn send(suffix: &str, payload: String, retain: bool) {
    let link = LINK.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(l) = link.as_ref() {
        // try_send: a full queue drops the message instead of stalling capture.
        let _ = l.tx.try_send(Msg { topic: format!("{}/{suffix}", l.prefix), payload, retain });
    }
}

/// An event opened on camera slot `cam` (0-based).
pub(crate) fn event_started(cam: u8, id: &str, score: f32) {
    let n = cam as u32 + 1;
    send(&format!("cam{n}/motion"), "ON".into(), true);
    send(&format!("cam{n}/event"),
         json!({ "type": "start", "id": id, "camera": n, "score": score }).to_string(), false);
}

/// An event on `cam` closed, after `duration` seconds, with these object labels.
pub(crate) fn event_ended(cam: u8, id: &str, duration: f64, labels: &[String]) {
    let n = cam as u32 + 1;
    send(&format!("cam{n}/motion"), "OFF".into(), true);
    send(&format!("cam{n}/event"),
         json!({ "type": "end", "id": id, "camera": n,
                 "duration_secs": (duration * 10.0).round() / 10.0, "labels": labels }).to_string(), false);
    if let Some(top) = labels.first() {
        send(&format!("cam{n}/last_detection"), top.clone(), true);
    }
}

/// Camera `cam` stopped (or resumed) delivering frames.
pub(crate) fn camera_online(cam: u8, online: bool) {
    send(&format!("cam{}/online", cam as u32 + 1), if online { "ON" } else { "OFF" }.into(), true);
}

/// Home Assistant discovery configs for the given cameras (slot, name).
pub(crate) fn discovery(prefix: &str, cams: &[(u8, String)]) -> Vec<Msg> {
    let device = json!({
        "identifiers": ["anivar_nvr"], "name": "Anivar NVR", "manufacturer": "Anivar",
        "model": "NVR", "sw_version": env!("CARGO_PKG_VERSION"),
    });
    let avail = format!("{prefix}/available");
    let mut out = Vec::new();
    for (slot, name) in cams {
        let n = *slot as u32 + 1;
        let entity = |component: &str, key: &str, label: &str, extra: serde_json::Value| {
            let mut cfg = json!({
                "name": format!("{name} {label}"),
                "unique_id": format!("anivar_cam{n}_{key}"),
                "state_topic": format!("{prefix}/cam{n}/{key}"),
                "availability_topic": avail,
                "device": device,
            });
            if let (Some(c), Some(e)) = (cfg.as_object_mut(), extra.as_object()) {
                for (k, v) in e { c.insert(k.clone(), v.clone()); }
            }
            Msg { topic: format!("homeassistant/{component}/anivar_cam{n}_{key}/config"),
                  payload: cfg.to_string(), retain: true }
        };
        out.push(entity("binary_sensor", "motion", "motion", json!({ "device_class": "motion" })));
        out.push(entity("binary_sensor", "online", "online", json!({ "device_class": "connectivity" })));
        out.push(entity("sensor", "last_detection", "last detection", json!({ "icon": "mdi:cctv" })));
    }
    out
}

fn options(host: &str, port: u16, user: &str, pass: &str, prefix: &str) -> MqttOptions {
    let host_tag: String = host.chars().filter(|c| c.is_ascii_alphanumeric()).take(12).collect();
    let mut o = MqttOptions::new(format!("anivar-{host_tag}-{}", std::process::id()), host, port);
    o.set_keep_alive(Duration::from_secs(30));
    if !user.is_empty() { o.set_credentials(user, pass); }
    o.set_last_will(LastWill::new(format!("{prefix}/available"), "offline", QoS::AtLeastOnce, true));
    o
}

/// Start, restart or stop the bridge to match Settings. Called at boot and
/// whenever settings are saved.
pub(crate) async fn restart(state: &Arc<AppState>) {
    if let Some(old) = LINK.lock().unwrap_or_else(|e| e.into_inner()).take() { old.task.abort(); }
    let (host, port, user, pass, prefix) = {
        let s = state.settings.read().await;
        (s.mqtt_host.trim().to_string(), s.mqtt_port, s.mqtt_username.clone(),
         s.mqtt_password.clone(), s.mqtt_topic_prefix.trim().trim_matches('/').to_string())
    };
    if host.is_empty() { return; }
    let prefix = if prefix.is_empty() { "anivar".to_string() } else { prefix };
    let cams: Vec<(u8, String)> = sqlx::query_as::<_, (i64, String)>(
        "SELECT cam_id, name FROM camera_configs WHERE enabled=1 ORDER BY cam_id")
        .fetch_all(&state.db).await.unwrap_or_default()
        .into_iter().map(|(id, name)| (id as u8, name)).collect();
    start(host, port, &user, &pass, prefix, cams);
}

/// Connect to the broker and serve the queue until stopped.
fn start(host: String, port: u16, user: &str, pass: &str, prefix: String, cams: Vec<(u8, String)>) {
    if let Some(old) = LINK.lock().unwrap_or_else(|e| e.into_inner()).take() { old.task.abort(); }
    let (client, mut events) = AsyncClient::new(options(&host, port, user, pass, &prefix), 64);
    let (tx, mut rx) = mpsc::channel::<Msg>(256);
    let p = prefix.clone();
    let announcer = client.clone();
    // Two loops joined, never raced: one drives the connection (rumqttc
    // reconnects on its own as long as it is polled), the other forwards the
    // queue. While the broker is down `publish` waits, the queue fills, and
    // `send` starts dropping — capture is never held up.
    let task = tokio::spawn(async move {
        let connection = async {
            let mut warned = false;
            loop {
                match events.poll().await {
                    Ok(Event::Incoming(Packet::ConnAck(_))) => {
                        tracing::info!("mqtt: connected to {host}:{port}");
                        warned = false;
                        let _ = announcer.try_publish(format!("{p}/available"), QoS::AtLeastOnce, true, "online");
                        for m in discovery(&p, &cams) {
                            let _ = announcer.try_publish(m.topic, QoS::AtLeastOnce, m.retain, m.payload);
                        }
                    }
                    Ok(_) => {}
                    Err(e) => {
                        if !warned { tracing::warn!("mqtt: {host}:{port} unreachable ({e}); retrying"); warned = true; }
                        tokio::time::sleep(Duration::from_secs(5)).await;
                    }
                }
            }
        };
        let forward = async {
            while let Some(m) = rx.recv().await {
                let _ = client.publish(m.topic, QoS::AtLeastOnce, m.retain, m.payload).await;
            }
        };
        tokio::join!(connection, forward);
    });
    *LINK.lock().unwrap_or_else(|e| e.into_inner()) = Some(Link { tx, prefix, task });
}

/// Settings → MQTT → "Test connection": connect with these values, wait for
/// the broker's answer, disconnect.
#[tauri::command]
pub async fn mqtt_test(host: String, port: u16, username: String, password: String) -> Result<String, String> {
    let host = host.trim().to_string();
    if host.is_empty() { return Err("Enter the broker's host name or IP address.".into()); }
    let (client, mut events) = AsyncClient::new(options(&host, port, &username, &password, "anivar-test"), 4);
    let outcome = tokio::time::timeout(Duration::from_secs(6), async {
        loop {
            match events.poll().await {
                Ok(Event::Incoming(Packet::ConnAck(ack))) =>
                    return if ack.code == rumqttc::ConnectReturnCode::Success { Ok(()) }
                           else { Err(format!("the broker refused the connection ({:?})", ack.code)) },
                Ok(_) => {}
                Err(e) => return Err(e.to_string()),
            }
        }
    }).await;
    let _ = client.disconnect().await;
    match outcome {
        Ok(Ok(())) => Ok(format!("Connected to {host}:{port}.")),
        Ok(Err(e)) => Err(format!("Couldn't connect to {host}:{port}: {e}")),
        Err(_) => Err(format!("No answer from {host}:{port} within 6 seconds.")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovery_gives_each_camera_three_entities_on_its_own_topics() {
        let msgs = discovery("anivar", &[(0, "Front door".into()), (3, "Garage".into())]);
        assert_eq!(msgs.len(), 6);
        let motion = msgs.iter().find(|m| m.topic == "homeassistant/binary_sensor/anivar_cam1_motion/config").unwrap();
        assert!(motion.retain);
        let cfg: serde_json::Value = serde_json::from_str(&motion.payload).unwrap();
        assert_eq!(cfg["name"], "Front door motion");
        assert_eq!(cfg["state_topic"], "anivar/cam1/motion");
        assert_eq!(cfg["device_class"], "motion");
        assert_eq!(cfg["availability_topic"], "anivar/available");
        assert_eq!(cfg["device"]["identifiers"][0], "anivar_nvr");
        // Slot 3 is "camera 4" everywhere a person sees it.
        assert!(msgs.iter().any(|m| m.topic == "homeassistant/sensor/anivar_cam4_last_detection/config"));
    }

    /// Against a real broker on localhost:1883 (e.g. `mosquitto`). Ignored in CI:
    /// `cargo test --lib mqtt::tests::live -- --ignored`
    #[tokio::test]
    #[ignore]
    async fn live_broker_receives_discovery_state_and_events() {
        // Subscriber first, so it sees retained and live messages alike.
        let mut o = MqttOptions::new("anivar-test-sub", "127.0.0.1", 1883);
        o.set_keep_alive(Duration::from_secs(10));
        let (sub, mut ev) = AsyncClient::new(o, 64);
        sub.subscribe("anivar-live/#", QoS::AtLeastOnce).await.unwrap();
        sub.subscribe("homeassistant/#", QoS::AtLeastOnce).await.unwrap();
        let got = std::sync::Arc::new(Mutex::new(Vec::<(String, String)>::new()));
        let g = got.clone();
        tokio::spawn(async move {
            while let Ok(e) = ev.poll().await {
                if let Event::Incoming(Packet::Publish(p)) = e {
                    g.lock().unwrap().push((p.topic.clone(), String::from_utf8_lossy(&p.payload).into()));
                }
            }
        });
        tokio::time::sleep(Duration::from_millis(500)).await;

        start("127.0.0.1".into(), 1883, "", "", "anivar-live".into(), vec![(0, "Front door".into())]);
        tokio::time::sleep(Duration::from_millis(800)).await;
        event_started(0, "evt-1", 0.42);
        event_ended(0, "evt-1", 3.24, &["person".into(), "dog".into()]);
        camera_online(0, false);
        tokio::time::sleep(Duration::from_millis(800)).await;

        let got = got.lock().unwrap().clone();
        let has = |t: &str, p: &str| got.iter().any(|(gt, gp)| gt == t && gp.contains(p));
        assert!(has("anivar-live/available", "online"), "{got:#?}");
        assert!(has("homeassistant/binary_sensor/anivar_cam1_motion/config", "Front door motion"));
        assert!(has("anivar-live/cam1/motion", "ON"));
        assert!(has("anivar-live/cam1/motion", "OFF"));
        assert!(has("anivar-live/cam1/event", "\"type\":\"end\""));
        assert!(has("anivar-live/cam1/last_detection", "person"));
        assert!(has("anivar-live/cam1/online", "OFF"));
        assert!(mqtt_test("127.0.0.1".into(), 1883, String::new(), String::new()).await.is_ok());
    }

    #[test]
    fn nothing_is_sent_while_the_bridge_is_off() {
        // With no link these are no-ops; they must not panic or block.
        event_started(0, "e1", 0.5);
        event_ended(0, "e1", 4.0, &["person".into()]);
        camera_online(0, false);
    }
}
