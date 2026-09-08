//! Shared application state — types and structs referenced by every module.
//!
//! Contains:
//!   * [`Settings`]              — the full user-configurable settings record (encrypted at rest).
//!   * [`AppState`]               — the central `Arc<...>` runtime state shared between every
//!     Tauri command, HTTP handler, and background worker.
//!   * [`StreamState`]            — the slimmer state struct passed to axum routes
//!     (a subset of `AppState` plus a few HTTP-specific fields).
//!   * Frame / clip / event records (`FrameResult`, `MotionEvent`, `StreamInfo`).
//!   * Per-camera mutable state (`PerCamState`, `SceneObject`, `BehaviorEvent`,
//!     `CameraInventory`, …).
//!   * SignalRoom (WebRTC SDP/ICE relay).
//!   * Client session bookkeeping (`ClientSession`).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;
use tokio::sync::{broadcast, mpsc, watch, Mutex, RwLock};

use crate::DiscoveredCamera;


pub(crate) fn default_ai_model() -> String { "onnx-community/yolo11s-uint8".to_string() }
/// On-device by default: the model runs inside this process (see
/// `agent::local_llm`). No daemon, no install, nothing resident when idle.
fn default_ai_provider_name() -> String { "local".to_string() }
fn default_vision_model() -> String { String::new() }
fn default_agent_poll_secs() -> u32 { 10 } // check for new events every 10s
fn default_alert_min_risk() -> String { "suspicious".to_string() }
fn default_agent_enabled() -> bool { false }
fn default_true() -> bool { true }
fn default_reid_threshold() -> f32 { 0.50 }
fn default_nvr_segment_mins() -> u32 { 1 } // kept for settings UI (minutes display)
fn default_loitering_secs()      -> u32    { 30 }
fn default_crowd_threshold()     -> u32    { 3  }
fn default_repeat_threshold()    -> u32    { 3  }
fn default_agent_persona_name()  -> String { "Guardian".to_string() }
fn default_strobe_frames()       -> u32    { 4  }
fn default_face_model()          -> String { "off".to_string() }   // "off" | "small" | "large"
fn default_face_det_threshold()  -> f32    { 0.5 }                 // liberal detection (recognition stays strict)
// NOTE: our matcher uses RAW ArcFace cosine (dot of L2-normalised 512-d vectors),
// NOT mature NVRs' probability/SVC scale. Same-person cosine ≈ 0.5 (the value the
// clustering code uses for "same person"), so these live on a 0.4–0.6 band — the
// old 0.9/0.8 (mature NVRs' probability defaults) meant a real match never fired.
fn default_face_rec_threshold()  -> f32    { 0.5 }                 // confident same-person (raw cosine)
fn default_face_unknown_score()  -> f32    { 0.4 }                 // near-miss / candidate floor (raw cosine)
// Hybrid head: minimum CALIBRATED PROBABILITY (softmax, 0..1 — NOT cosine) from the
// trained classifier before it names a person. Below this we fall back to cosine-NN.
fn default_face_class_confidence() -> f32  { 0.6 }
// Liveness / anti-spoofing gate (needs the `face_liveness` skill). Off by default —
// a home NVR rarely faces photo-spoofing, and it adds per-face inference cost.
fn default_face_liveness()         -> bool { false }
fn default_yolo_variant()        -> String { "xlarge".to_string() } // "nano" | "small" | "medium" | "large" | "xlarge"
/// "balanced" — the 1.2B. The 350M is faster but its own model card warns it off
/// conversation, and vision is a 1.3 GB download nobody should get by default.
fn default_local_llm_tier()      -> String { "balanced".to_string() } // "fast" | "balanced" | "vision"
fn default_search_model()        -> String { "off".to_string() }   // "off" | "mobileclip_s0" | "clip_b32" | "jina_clip"
fn default_inference_device()    -> String { "auto".to_string() }  // "auto" | "gpu" | "cpu" — ONNX execution provider
fn default_audio_listen()        -> String { "scream,glass,alarm,gunshot,bark,yell,speech".to_string() }
fn default_audio_threshold()     -> f32    { 0.30 }
// ── v8 motion hysteresis knobs ───────────────────────────────────────────────
fn default_motion_min_frames()       -> u32  { 3 }      // v27: ~150ms debounce — catch quick/distant motion (sensitive).
fn default_motion_open_score_mult()  -> f32  { 1.0 }    // v27: open at sensitivity (no extra bar) so real motion isn't missed.
fn default_motion_lightning_threshold() -> f32 { 0.85 } // Mature NVRs lightning guard: ignore frames where >85% changed at once.
fn default_require_object_to_open()  -> bool { false }  // opt-in edge-AI-style gate
// ── v9 mature NVRs-model knobs (object-driven close, mid-event re-alerting) ──────
fn default_max_disappeared_frames()  -> u32  { 15 }     // ≈ 5× inference fps (3fps × 5s)
fn default_max_event_secs()          -> u32  { 300 }    // 5-min hard cap on a single event (0 = unlimited)
fn default_re_analysis_interval()    -> u32  { 60 }     // re-run VLM clip analysis every N seconds during long events
// ── v11 channel-first defaults ──────────────────────────────────────────────
fn default_attach_snapshot_to_alerts()    -> bool { true }
fn default_attach_clip_to_alerts()        -> bool { true }
fn default_live_share_default_minutes()   -> u32  { 30 }
fn default_tunnel_auto_stop()             -> bool { true }
// ── v7 richness knobs ────────────────────────────────────────────────────────
fn default_strobe_profile()         -> String { "balanced".to_string() } // "conservative" | "balanced" | "aggressive"
fn default_face_quality_floor()     -> f32    { 0.10 }                   // Laplacian blur floor on the ÷300 quality scale (was 0.20 on ÷1500 — too high, dropped real faces)
fn default_yolo_conf_threshold()    -> f32    { 0.40 }                   // v27: mature NVRs min_score — discard weak detections (confirm tier is YOLO_CONFIRM_THRESHOLD)
fn default_yolo_class_filter()      -> String { String::new() }          // empty = no filter (all 80 COCO classes pass)
fn default_alpr_region()            -> String { "global".to_string() }   // "global" | "european" | "argentinian"
fn default_nvr_max_gb() -> u32 { 50 }
fn default_nvr_record_mode() -> String { "always".into() }
fn default_depth_model() -> String { "balanced".into() }
fn default_remote_provider() -> String { "tailscale".into() }
fn default_nvr_retain_days() -> u32 { 7 }
fn default_nvr_retain_event_days() -> u32 { 30 }
fn default_remember_days() -> u32 { 30 }   // desktop login "remember this device" window

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    pub sensitivity: f32,
    pub motion_threshold: u8,
    pub record_on_motion: bool,
    pub record_pre_buffer_secs: u32,
    pub record_post_buffer_secs: u32,
    /// Hard cap (seconds) on a single motion event's length. `0` = unlimited.
    /// Force-closes runaway events so nothing records for minutes with nobody acting.
    #[serde(default = "default_max_event_secs")]
    pub record_max_event_secs: u32,
    pub stream_port: u16,
    pub stream_quality: u8,
    pub retention_days: u32,
    #[serde(default = "default_ai_model")]
    pub ai_model: String,
    // ── Guardian agent ────────────────────────────────────────────────────
    #[serde(default = "default_agent_enabled")]
    pub agent_enabled: bool,
    #[serde(default = "default_vision_model")]
    pub vision_model: String,
    #[serde(default = "default_agent_poll_secs")]
    pub agent_poll_secs: u32,
    #[serde(default = "default_alert_min_risk")]
    pub alert_min_risk: String,
    #[serde(default)]
    pub camera_name: String,
    // ── AI Provider ───────────────────────────────────────────────────────
    // "local" (default, in-process llama.cpp) | "lmstudio" | "openai" | "anthropic"
    // | "groq" | "xai" | "gemini" | "openai_compatible"
    #[serde(default = "default_ai_provider_name")]
    pub ai_provider: String,
    #[serde(default)]
    pub openai_api_key: String,
    #[serde(default)]
    pub anthropic_api_key: String,
    #[serde(default)]
    pub groq_api_key: String,
    #[serde(default)]
    pub xai_api_key: String,             // xAI (Grok) — OpenAI-compatible
    #[serde(default)]
    pub gemini_api_key: String,
    #[serde(default)]
    pub openai_compatible_url: String,   // custom base URL for OpenAI-compatible APIs
    #[serde(default)]
    pub openai_compatible_key: String,
    #[serde(default)]
    pub telegram_bot_token: String,
    #[serde(default)]
    pub telegram_chat_id: String,
    #[serde(default)]
    pub github_repo: String,         // owner/repo for in-app updates, e.g. "owner/repo"
    // ── Local network discovery & auth ────────────────────────────────────
    #[serde(default)]
    pub device_name: String,         // friendly name shown during discovery, e.g. "Living Room PC"
    #[serde(default)]
    pub auth_username: String,       // username for mobile login (empty = token-only mode)
    #[serde(default)]
    pub auth_password_hash: String,  // remote login: legacy SHA-256 hex (HTTP /login)
    // ── Desktop login gate (optional) — Argon2id + Telegram recovery/2FA ──────
    #[serde(default)]
    pub login_required: bool,             // master toggle for the desktop lock screen
    #[serde(default)]
    pub login_password_hash: String,      // Argon2id PHC string (encrypted at rest)
    #[serde(default)]
    pub auth_2fa_enabled: bool,           // optional Telegram OTP second factor
    #[serde(default)]
    pub auth_remember_enabled: bool,      // allow "remember this device"
    #[serde(default = "default_remember_days")]
    pub auth_remember_days: u32,          // how long a remembered device stays unlocked
    // ── Auto Re-ID ────────────────────────────────────────────────────────
    #[serde(default = "default_true")]
    pub auto_reid: bool,           // recognize faces automatically without manual enrollment
    #[serde(default = "default_reid_threshold")]
    pub reid_threshold: f32,       // cosine distance threshold for face matching (0.0–1.0)
    // ── NVR ───────────────────────────────────────────────────────────────
    #[serde(default)]
    pub nvr_enabled: bool,
    #[serde(default = "default_nvr_segment_mins")]
    pub nvr_segment_mins: u32,     // minutes per NVR segment file
    #[serde(default = "default_nvr_max_gb")]
    pub nvr_max_gb: u32,
    /// standard retain model, enforced at PRUNE time (recording itself is
    /// always continuous so pre/post-event context is never lost):
    ///   "always"      — keep everything for the continuous window
    ///   "motion_only" — non-motion footage dropped after a 1h grace
    ///   "events_only" — only review-event footage kept past the grace
    #[serde(default = "default_nvr_record_mode")]
    pub nvr_record_mode: String,
    /// Continuous-footage window (days, 0 = forever). Mature NVRs: record.retain.days.
    #[serde(default = "default_nvr_retain_days")]
    pub nvr_retain_days: u32,
    /// Event-footage window (days, 0 = forever). Mature NVRs: alerts/detections retain.
    #[serde(default = "default_nvr_retain_event_days")]
    pub nvr_retain_event_days: u32,
    /// Keep event clips past footage retention: before raw NVR footage covering
    /// an event is pruned, export the event's bounded standalone clip so the
    /// moment survives (for the event's own history window) while the bulk
    /// continuous recording is deleted on schedule. The enterprise "keep the
    /// curated clips, drop the tape" model. Off = today's behavior (clips die
    /// with the footage unless someone happened to view/share them first).
    #[serde(default)]
    pub keep_event_clips: bool,
    /// Appliance mode: a Windows Scheduled Task relaunches the app (headless)
    /// within 5 min if the process is gone — covers native aborts the panic
    /// hook can't catch (e.g. GPU-driver 0xc0000409). Opt-in.
    #[serde(default)]
    pub relaunch_after_crash: bool,
    /// Remote-access provider for live/clip share links. Tailscale Funnel is now
    /// the only implementation (Cloudflare was removed 2026-07-28 — trycloudflare
    /// quick tunnels are testing-only under its ToS). The field is retained so
    /// stored settings JSON keeps round-tripping; nothing reads it.
    #[serde(default = "default_remote_provider")]
    pub remote_provider: String,
    #[serde(default)]
    pub camera_masks: String,
    /// Depth Map Anonymization per cam: JSON {"<camId>": true}. When ON, ONLY
    /// colorized depth frames are persisted/served (recordings, streams,
    /// snapshots, sends); raw pixels live in RAM for local detection only.
    #[serde(default)]
    pub depth_anonymize: String,
    /// Depth anonymization SPEED preset: "fast" (252px) | "balanced" (378px,
    /// default) | "quality" (518px). The efficiency lever is inference
    /// resolution — the FP16 model is shared. Picked in Arsenal's Depth card.
    #[serde(default = "default_depth_model")]
    pub depth_model: String,
    /// Client-side depth VIEW preferences per cam (cosmetic Overlay Mode):
    /// JSON {"<camId>": {"mode":"off"|"overlay"|"replace","opacity":0..1}}.
    #[serde(default)]
    pub depth_privacy: String,
    // on-device assistants-inspired intelligence
    #[serde(default = "default_true")]
    pub loitering_detection: bool,
    #[serde(default = "default_loitering_secs")]
    pub loitering_threshold_secs: u32,
    #[serde(default)]
    pub crowd_detection: bool,
    #[serde(default = "default_crowd_threshold")]
    pub crowd_threshold: u32,
    #[serde(default = "default_true")]
    pub repeat_visitor_detection: bool,
    #[serde(default = "default_repeat_threshold")]
    pub repeat_visitor_threshold: u32,
    // ── assistant-parity: Quiet hours ─────────────────────────────────────────
    #[serde(default)]
    pub quiet_hours_enabled: bool,
    #[serde(default)]
    pub quiet_hours_start: String,   // "22:00"
    #[serde(default)]
    pub quiet_hours_end: String,     // "07:00"
    // ── assistant-parity: Soul / Persona ─────────────────────────────────────
    #[serde(default = "default_agent_persona_name")]
    pub agent_persona_name: String,  // "Guardian" by default
    #[serde(default)]
    pub agent_persona_text: String,  // free-text personality description
    // ── assistant-parity: Stroboscopic analysis ──────────────────────────────
    #[serde(default = "default_strobe_frames")]
    pub strobe_frames: u32,          // how many frames to sample per clip (default 4)
    // ── Semantic event search (standard CLIP) ───────────────────────
    /// Active embedding model for Review search + "Find similar".
    /// "off" | "mobileclip_s0" (Apple, ~207 MB) | "clip_b32" (OpenAI, ~579 MB) | "jina_clip" (~850 MB).
    #[serde(default = "default_search_model")]
    pub search_model: String,

    /// ONNX hardware acceleration. "auto"/"gpu" → try the GPU execution provider
    /// (DirectML on Windows) with automatic CPU fallback; "cpu" → force CPU.
    #[serde(default = "default_inference_device")]
    pub inference_device: String,

    // ── Face recognition (standard) ─────────────────────────────────
    /// "off" | "small" (ArcFace 512-d NCHW, CPU) | "large" (ArcFace 512-d NHWC, GPU/NPU).
    /// Both tiers embed to the SAME 512-d ArcFace space — they differ only in model
    /// size + tensor layout, so enrolled/recognised faces are comparable across tiers.
    #[serde(default = "default_face_model")]
    pub face_model: String,
    /// Face-detection confidence required before recognition runs (detector score).
    #[serde(default = "default_face_det_threshold")]
    pub face_detection_threshold: f32,
    /// RAW ArcFace cosine required to assign a name/sub-label (same-person ≈ 0.5).
    #[serde(default = "default_face_rec_threshold")]
    pub face_recognition_threshold: f32,
    /// Cosine floor below which there's no candidate at all; the [unknown,rec) band
    /// is a near-miss "looks like X" surfaced in the Train tab.
    #[serde(default = "default_face_unknown_score")]
    pub face_unknown_score: f32,
    /// Hybrid head: minimum softmax probability (0..1, calibrated — NOT a cosine)
    /// the trained classifier must reach to name a person; below it, recognition
    /// falls back to the cosine-NN path. See `face_classifier.rs`.
    #[serde(default = "default_face_class_confidence")]
    pub face_class_confidence: f32,
    /// Liveness / anti-spoofing: when on (and the `face_liveness` skill installed),
    /// a face that scores as a photo/screen presentation attack is demoted to
    /// "unknown" instead of being named. Fail-open when the model is missing.
    #[serde(default = "default_face_liveness")]
    pub face_liveness: bool,
    /// One-time guard: have the face thresholds been migrated off the legacy
    /// Mature NVRs PROBABILITY scale (0.9/0.8) onto our raw-cosine scale (0.5/0.4)?
    /// Missing in old saved settings → serde gives `false` → the migration in
    /// `db.rs::load_settings_from_db` runs once, then this is set so deliberate
    /// user values are never re-clobbered.
    #[serde(default)]
    pub face_thresholds_migrated: bool,
    // ── YOLO 2026 variant ────────────────────────────────────────────────
    /// `"nano"` | `"small"` | `"medium"` | `"large"` | `"xlarge"` — picks
    /// which installed `skills/yolo26{n,s,m,l,x}/model.onnx` the inference
    /// loop loads. The five variants trade size for accuracy:
    /// nano ~10 MB, small ~37 MB, medium ~78 MB, large ~95 MB, xlarge ~175 MB.
    #[serde(default = "default_yolo_variant")]
    pub yolo_variant: String,
    /// `"fast"` | `"balanced"` | `"vision"` — which installed on-device language
    /// model runs, mirroring how `yolo_variant` picks a detector tier.
    ///
    /// A separate field because `vision_model` is empty for the on-device
    /// provider (the engine is compiled in and has no API model name), so there
    /// was nowhere to record WHICH local model the user chose.
    #[serde(default = "default_local_llm_tier")]
    pub local_llm_tier: String,
    // ── v7 richness knobs ───────────────────────────────────────────────
    /// Strobe-extraction profile picker. Three named bands of duration→frame-count tables.
    /// "conservative" (cheap, small local VLMs), "balanced" (default), "aggressive" (cloud VLMs).
    #[serde(default = "default_strobe_profile")]
    pub strobe_profile: String,
    /// Laplacian blur floor below which a face crop is dropped before embedding.
    /// 0.0 = keep everything; 0.5+ = only sharp faces. Default 0.20 matches the
    /// old hardcoded value in face.rs:321 so behaviour is unchanged out of the box.
    #[serde(default = "default_face_quality_floor")]
    pub face_quality_floor: f32,
    /// YOLO 2026 confidence threshold (lower = catches more, more false positives).
    /// Default 0.20 matches the previous hardcoded value in inference.rs.
    #[serde(default = "default_yolo_conf_threshold")]
    pub yolo_confidence_threshold: f32,
    /// CSV of COCO class names to keep, e.g. `"person,car,truck"`. Empty = no filter.
    #[serde(default = "default_yolo_class_filter")]
    pub yolo_class_filter: String,
    /// ALPR regional model. Picks which `skills/alpr_{region}/model.onnx` is used.
    /// "global" | "european" | "argentinian". Falls back to global if the chosen
    /// region isn't installed; falls back to any installed alpr_* otherwise.
    #[serde(default = "default_alpr_region")]
    pub alpr_region: String,
    /// Known license plates, one "PLATE=Name" per line (e.g. "ABC1234=Mom's car").
    /// A recognised plate that matches (exact or within 1 char) is labelled with
    /// the name as the event sub_label, standard. Empty = no naming.
    #[serde(default)]
    pub known_plates: String,

    /// Serve the stream/API on the LAN (0.0.0.0) so other devices on the same
    /// network can view. OFF (the default) binds 127.0.0.1 only.
    ///
    /// Off by default because this is the app's whole network attack surface and
    /// it is reachable before anyone has decided they want that. Remote viewing
    /// does not need it: `tailscale funnel --bg 8882` proxies to loopback, so
    /// share links keep working with this off. Turn it on to open a phone on the
    /// same Wi-Fi at `http://<this-machine>:8882`. Applied on next app start.
    #[serde(default)]
    pub lan_access: bool,

    /// Versioned one-time migrations marker — see `db.rs::load_settings_from_db`.
    /// Defaults only apply to FRESH installs (saved settings re-write every field on
    /// Save), so intentional behavior changes bump this and apply exactly once.
    #[serde(default)]
    pub settings_version: u32,

    // ── Audio event detection (standard YAMNet) ─────────────────────
    /// Master toggle. Requires the `audio_yamnet` skill + an RTSP camera with an
    /// audio track (browser/USB cameras don't deliver audio server-side yet).
    #[serde(default)]
    pub audio_detection: bool,
    /// Comma-separated AudioSet class-name substrings to alert on (matched
    /// case-insensitively against YAMNet's class_map). Empty → a safe default set.
    #[serde(default = "default_audio_listen")]
    pub audio_listen: String,
    /// Per-class score (0..1) required to fire an audio event. Default 0.30.
    #[serde(default = "default_audio_threshold")]
    pub audio_threshold: f32,
    /// Over-speed alert threshold in km/h for speed zones (0 = no alert, just record).
    #[serde(default)]
    pub speed_alert_kmh: f32,
    // ── v8 motion hysteresis ────────────────────────────────────────────
    /// Consecutive motion=true frames required to open (or sustain) an event.
    /// Defaults to 3 — one or two spurious noisy frames no longer fire an event.
    /// The lifecycle bug ("event stuck on") came from a single spike refreshing
    /// the close timer; this counter is the primary fix.
    #[serde(default = "default_motion_min_frames")]
    pub motion_min_frames: u32,
    /// Multiplier on `sensitivity` for the OPEN threshold. Sustaining uses raw
    /// `sensitivity`. Higher = harder to open an event but easy to track an
    /// already-open one (so a person whose motion drops below 0.1% briefly
    /// doesn't lose the event). Default 1.5×.
    #[serde(default = "default_motion_open_score_mult")]
    pub motion_open_score_mult: f32,
    /// Mature NVRs `lightning_threshold`: if more than this FRACTION of the unmasked
    /// frame changes in one tick, treat it as a lighting / IR / exposure shift and
    /// ignore the frame (no motion). Stops lights-on/off and day↔night IR switches
    /// from opening a whole-frame phantom event. 0 disables. Default 0.85.
    #[serde(default = "default_motion_lightning_threshold")]
    pub motion_lightning_threshold: f32,
    /// Opt-in edge-AI-style gate: when true, events that aren't confirmed
    /// by a YOLO detection of a tracked class (person/vehicle/animal/package)
    /// within `record_post_buffer_secs` are dropped silently. Off by default
    /// so users without YOLO installed (or on CPU-only) still get raw-motion
    /// events. Power-user "zero phantom events" mode.
    ///
    /// v9: implicitly forced TRUE whenever YOLO is active. The setting is
    /// effectively only consulted when YOLO is off / not installed.
    #[serde(default = "default_require_object_to_open")]
    pub require_object_to_open_event: bool,
    // ── v9 mature NVRs-model close + mid-event re-alerting ──────────────────
    /// Frames of bbox absence before the tracked-object "presence" signal goes
    /// false. Combined with stillness_streak, this is what closes events when
    /// YOLO is active. Default 15 ≈ mature NVRs' `detect.max_disappeared` (5×fps).
    #[serde(default = "default_max_disappeared_frames")]
    pub detect_max_disappeared_frames: u32,
    /// How often (seconds) the agent re-analyses a long-running event so the
    /// user gets fresh alerts mid-event instead of one summary at close.
    /// 0 disables. Default 60.
    #[serde(default = "default_re_analysis_interval")]
    pub re_analysis_interval_secs: u32,
    // ── v11 channels-first knobs ────────────────────────────────────────
    /// Whether motion-event alerts should carry the event thumbnail to
    /// channels that can render an inline image (Telegram, Discord, Pushover,
    /// Text-only channels (
    /// WhatsApp CallMeBot) get a "▶ View snapshot" share link instead.
    #[serde(default = "default_attach_snapshot_to_alerts")]
    pub attach_snapshot_to_alerts: bool,
    /// Whether motion-event alerts should carry the recorded clip inline
    /// when the channel supports video upload (Telegram ≤50 MB, Discord
    /// ≤8 MB after transcode). Larger clips fall back to a tunnel share
    /// link. Disable to skip clip uploads entirely.
    #[serde(default = "default_attach_clip_to_alerts")]
    pub attach_clip_to_alerts: bool,
    /// Cameras (by slot id) for which alerts are SUPPRESSED. Empty = alert for
    /// all cameras. Toggled from the Telegram `/menu` per-camera sub-menu. The
    /// dispatch path skips any channel send when the event's cam is in this set.
    #[serde(default)]
    pub alert_disabled_cameras: Vec<u8>,
    /// Event CATEGORIES for which channel alerts are muted ("person"|"vehicle"|
    /// "animal"|"audio"|"other"). Empty = alert about everything. Events in a
    /// muted category still record + analyze — the user just isn't messaged.
    /// Toggled from Telegram (👁 Alert filter) AND the app's Settings alerts
    /// section; single field, so both stay in sync by construction.
    #[serde(default)]
    pub alert_muted_categories: Vec<String>,
    /// Default expiry for "Share Live View" links the user generates from
    /// the camera tile. Per-share dropdown overrides this.
    #[serde(default = "default_live_share_default_minutes")]
    pub live_share_default_minutes: u32,
    /// When true (default), the public tunnel stops automatically once
    /// all active share links expire — minimises remote-exposure window.
    #[serde(default = "default_tunnel_auto_stop")]
    pub tunnel_auto_stop: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            sensitivity: 0.04,   // v26: mature NVRs default — kills false positives from shadows / flicker / sensor noise.
            motion_threshold: 8,   // low per-pixel threshold so small changes count
            record_on_motion: true,
            record_pre_buffer_secs: 3,
            record_post_buffer_secs: 10, // 10s no-motion before stopping
            record_max_event_secs: default_max_event_secs(),
            stream_port: 8882,
            stream_quality: 60,
            retention_days: 7,
            ai_model: default_ai_model(),
            agent_enabled: false,
            vision_model: default_vision_model(),
            agent_poll_secs: default_agent_poll_secs(),
            alert_min_risk: default_alert_min_risk(),
            camera_name: String::new(),
            telegram_bot_token: String::new(),
            telegram_chat_id: String::new(),
            github_repo: String::new(),
            device_name: String::new(),
            auth_username: String::new(),
            auth_password_hash: String::new(),
            login_required: false,
            login_password_hash: String::new(),
            auth_2fa_enabled: false,
            auth_remember_enabled: false,
            auth_remember_days: default_remember_days(),
            auto_reid: true,
            reid_threshold: 0.50,
            nvr_enabled: true,
            nvr_segment_mins: 1,
            nvr_max_gb: 50,
            nvr_record_mode: default_nvr_record_mode(),
            nvr_retain_days: default_nvr_retain_days(),
            nvr_retain_event_days: default_nvr_retain_event_days(),
            keep_event_clips: false,
            relaunch_after_crash: false,
            depth_model: default_depth_model(),
            remote_provider: default_remote_provider(),
            camera_masks: String::new(),
            depth_anonymize: String::new(),
            depth_privacy: String::new(),
            loitering_detection: true,
            loitering_threshold_secs: 30,
            crowd_detection: false,
            crowd_threshold: 3,
            repeat_visitor_detection: true,
            repeat_visitor_threshold: 3,
            ai_provider: "local".into(),
            openai_api_key: String::new(),
            anthropic_api_key: String::new(),
            groq_api_key: String::new(),
            xai_api_key: String::new(),
            gemini_api_key: String::new(),
            openai_compatible_url: String::new(),
            openai_compatible_key: String::new(),
            // assistant-parity defaults
            quiet_hours_enabled:  false,
            quiet_hours_start:    "22:00".into(),
            quiet_hours_end:      "07:00".into(),
            agent_persona_name:   default_agent_persona_name(),
            agent_persona_text:   String::new(),
            strobe_frames:        default_strobe_frames(),
            search_model:               default_search_model(),
            inference_device:           default_inference_device(),
            face_model:                 default_face_model(),
            face_detection_threshold:   default_face_det_threshold(),
            face_recognition_threshold: default_face_rec_threshold(),
            face_unknown_score:         default_face_unknown_score(),
            face_class_confidence:      default_face_class_confidence(),
            face_liveness:              default_face_liveness(),
            // Fresh installs ship the cosine-scale defaults above — nothing to migrate.
            face_thresholds_migrated:   true,
            yolo_variant:               default_yolo_variant(),
            local_llm_tier:             default_local_llm_tier(),
            strobe_profile:             default_strobe_profile(),
            face_quality_floor:         default_face_quality_floor(),
            yolo_confidence_threshold:  default_yolo_conf_threshold(),
            yolo_class_filter:          default_yolo_class_filter(),
            alpr_region:                default_alpr_region(),
            known_plates:               String::new(),
            lan_access:                 false,   // opt in; see the field's doc comment
            settings_version:           3, // fresh installs start at the current version
            // On by default for a security appliance: when the audio (YAMNet) skill is
            // installed AND a mic is present, listen for THREAT sounds (glass, alarm,
            // gunshot, scream…). No-ops with no skill/mic, so it's safe as a default.
            audio_detection:            true,
            audio_listen:               default_audio_listen(),
            audio_threshold:            default_audio_threshold(),
            speed_alert_kmh:            0.0,
            motion_min_frames:          default_motion_min_frames(),
            motion_open_score_mult:     default_motion_open_score_mult(),
            motion_lightning_threshold: default_motion_lightning_threshold(),
            require_object_to_open_event: default_require_object_to_open(),
            detect_max_disappeared_frames: default_max_disappeared_frames(),
            re_analysis_interval_secs:    default_re_analysis_interval(),
            // v11 channels-first
            attach_snapshot_to_alerts:    default_attach_snapshot_to_alerts(),
            attach_clip_to_alerts:        default_attach_clip_to_alerts(),
            alert_disabled_cameras:       Vec::new(),
            alert_muted_categories:       Vec::new(),
            live_share_default_minutes:   default_live_share_default_minutes(),
            tunnel_auto_stop:             default_tunnel_auto_stop(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotionEvent {
    pub id: String,
    pub started_at: String,
    pub ended_at: Option<String>,
    pub duration_secs: Option<f64>,
    pub peak_score: f32,
    pub clip_path: Option<String>,
    pub thumbnail: Option<String>,
    pub detections: Option<String>,
    pub ai_summary: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cam_id: Option<u8>,
    /// standard category: "person" / "vehicle" / "animal" / "package" / "other".
    /// Null on rows pre-dating the v6 categorisation migration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event_category: Option<String>,
    /// License-plate text from the `alpr` skill (when installed + a vehicle was visible).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recognized_plate: Option<String>,
    /// v27: specific dominant COCO label (e.g. "person", "dog", "car") — the real
    /// object, shown as the Review chip distinct from the broad `event_category`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dominant_label: Option<String>,
    /// v27: sub-label refinement (known face name / recognised plate / delivery brand).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sub_label: Option<String>,
    /// v28: RFC3339 wall-clock when YOLO first confirmed an object. The frontend
    /// anchors the event clip's playhead here (minus the pre-buffer) so the
    /// timeline needle matches the trimmed clip start.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_object_at: Option<String>,
    /// Top ground speed (km/h) seen for any tracked object inside a speed zone
    /// during the event. Null unless a calibrated speed zone is configured.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_speed_kmh: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameResult {
    pub motion_detected: bool,
    pub motion_score: f32,
    pub recording: bool,
    pub event_id: Option<String>,
    /// Normalised [x1,y1,x2,y2] motion bounding boxes (0.0-1.0).
    /// The frontend passes these to the YOLO worker so it only runs
    /// inference on changed regions — same technique mature NVRs use.
    pub motion_regions: Vec<[f32; 4]>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StreamInfo {
    pub local_ip:      String,
    pub public_ip:     Option<String>,  // for NAT hairpinning
    pub ipv6:          Option<String>,  // for IPv6 bypass of AP isolation
    pub port:          u16,
    pub url:           String,
    pub url_with_token: String,
    pub auth_token:    String,
    pub qr_data_url:   String,
}

// ─── WebRTC Signaling Room ────────────────────────────────────────────────────
// Scaffolding for the planned WebRTC SDP/ICE relay: wired into StreamState but the
// signaling path isn't active yet (host/viewer are never populated). Kept intact rather
// than removed so the relay can be completed without re-threading StreamState.
#[allow(dead_code)]
pub(crate) struct SignalRoom {
    pub(crate) host:   Option<mpsc::UnboundedSender<String>>,
    pub(crate) viewer: Option<mpsc::UnboundedSender<String>>,
}

// ─── Per-camera motion state ──────────────────────────────────────────────────

#[derive(Default)]
pub struct PerCamState {
    pub(crate) prev_frame:     Option<Vec<u8>>,
    pub(crate) prev_dims:      (u32, u32),
    pub(crate) motion_active:  Option<String>,
    pub(crate) motion_peak:    f32,
    /// Wall-clock time of the last frame that triggered motion.
    pub(crate) last_motion_at: Option<Instant>,
    /// Track max crowd count seen this event (for repeat suppression)
    pub(crate) crowd_alerted_count: u32,
    /// Crowd streak: (streak start, last over-threshold report). A crowd must be
    /// SUSTAINED before it alerts — one frame of YOLO double-boxes is not a crowd.
    pub(crate) crowd_over: Option<(Instant, Instant)>,
    /// YOLO26 detections accumulated during the current motion event.
    /// Keeps the best-confidence detection per class — flushed to DB atomically
    /// when the event closes (mature NVRs ReviewSegment pattern).
    /// Eliminates the frontend race-condition where detections arrived after event end.
    pub(crate) detection_buffer: Vec<serde_json::Value>,
    // ── v8 temporal hysteresis for the motion lifecycle ──────────────────
    /// Consecutive frames with motion >= sensitivity. Used to require N-frame
    /// confirmation before opening an event or refreshing the close timer.
    pub(crate) motion_streak:    u32,
    /// Consecutive frames with motion < sensitivity. An event closes when this
    /// reaches a target derived from `record_post_buffer_secs * fps`.
    /// Replaces the buggy "last_motion_at.elapsed()" close path that any single
    /// noisy frame could reset.
    pub(crate) stillness_streak: u32,
    /// Whether the in-flight event has been confirmed by YOLO (a tracked object
    /// — person/vehicle/animal/package — was detected). Powers the opt-in
    /// `require_object_to_open_event` mode: events open provisionally on raw
    /// motion but persist only after this flips true within the grace window.
    pub(crate) event_object_confirmed: bool,
    /// Wall-clock when the in-flight event opened (used together with the
    /// settings grace-window to decide whether to drop an un-confirmed event).
    pub(crate) event_opened_at: Option<Instant>,
    // ── v9 object-presence tracking (mature NVRs-model close) ──────────────
    /// Wall-clock when YOLO last saw ANY tracked class in this camera.
    /// `None` means we've never seen one (or the event reset).
    pub(crate) last_object_seen_at: Option<Instant>,
    /// Number of consecutive inference ticks where YOLO ran but found no
    /// tracked class. Closes the event when it crosses
    /// `settings.detect_max_disappeared_frames`.
    pub(crate) object_absence_frames: u32,
    // ── v9 mid-event re-alerting state ─────────────────────────────────
    /// Wall-clock of the last clip-analysis VLM call for this in-flight event.
    /// Used to space out periodic re-analysis at `re_analysis_interval_secs`.
    pub(crate) last_analysis_at: Option<Instant>,
    /// Wall-clock of the last activity-burst alert sent for this event.
    /// Rate-limits bursts to one per 30s so a moving person doesn't spam.
    pub(crate) last_burst_at: Option<Instant>,
    /// Highest-confidence tracked class label seen so far in this event.
    /// A new tracked class appearing fires an "activity burst" alert.
    pub(crate) classes_seen: std::collections::HashSet<String>,
    /// Last dominant specific label written to the event row (e.g. "person",
    /// "dog"). Used to avoid redundant DB writes during real-time classification.
    pub(crate) last_dominant: Option<String>,
}

/// One entry per camera while a clip is being recorded.
/// Dropping the sender signals the write_clip task to finalise the file.
type ClipTxMap = Arc<tokio::sync::Mutex<HashMap<u8, tokio::sync::mpsc::UnboundedSender<Arc<Vec<u8>>>>>>;

/// Handle to a running native-capture task.  Set `cancel` to true to stop the OS thread.
pub struct CaptureTask {
    pub(crate) cancel: Arc<AtomicBool>,
    /// The nokhwa device index this task is capturing — lets `start_native_camera`
    /// DEDUPE: a second start for the same slot+device is a no-op instead of opening
    /// the camera twice (the two opens preempt each other → "device preempted").
    pub(crate) device_index: u32,
}
type CaptureHandles = Arc<tokio::sync::Mutex<HashMap<u8, CaptureTask>>>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NativeCameraDevice {
    pub index: u32,
    pub name: String,
    pub description: String,
}

/// A single detected object from the browser-side AI model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneObject {
    pub label: String,
    pub score: f32,
}

/// Behavior analysis event from frontend person tracker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorEvent {
    pub track_id: i64,
    pub name: Option<String>,
    pub flags: Vec<String>,
    pub duration_secs: u32,
}

/// A browser camera device reported by the frontend (from navigator.mediaDevices.enumerateDevices).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BrowserCameraInfo {
    pub device_id: String,
    pub label: String,
}

/// Full camera inventory: everything the app knows about available cameras.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CameraInventory {
    pub native: Vec<NativeCameraDevice>,
    pub network: Vec<DiscoveredCamera>,
    pub browser: Vec<BrowserCameraInfo>,
}

// ─── Storage info ─────────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct StorageInfo {
    pub(crate) total_bytes:    u64,
    pub(crate) clip_count:     u64,
    pub(crate) event_count:    u64,
    pub(crate) orphaned_clips: u64,
    pub(crate) oldest_event:   Option<String>,
    pub(crate) newest_event:   Option<String>,
    pub(crate) nvr_bytes:      u64,
    pub(crate) nvr_count:      u64,
}

// ─── App State ───────────────────────────────────────────────────────────────

/// Info about one active remote WebSocket viewer (for the "Connected Devices" panel).
#[derive(Debug, Clone, Serialize)]
pub struct ClientSession {
    pub id: String,
    pub ip: String,
    pub connected_at: u64,   // unix secs
    pub cam_id: usize,
}

pub struct AppState {
    /// One broadcast channel per camera slot (index 0-3).
    pub frame_txs: Arc<Vec<broadcast::Sender<Arc<Vec<u8>>>>>,
    /// Per-camera motion-detection state (lazily created per camera).
    pub cam_states: Arc<tokio::sync::Mutex<HashMap<u8, PerCamState>>>,
    pub settings: RwLock<Settings>,
    pub recording_active: Mutex<bool>,
    pub clip_txs: ClipTxMap,
    pub capture_handles: CaptureHandles,
    pub db: SqlitePool,
    pub data_dir: PathBuf,
    /// Shared with StreamState so `revoke_token` takes effect immediately in the HTTP middleware.
    pub auth_token: Arc<RwLock<String>>,
    /// Incrementing generation counter — WebSocket handlers watch this and
    /// close themselves as soon as the token is revoked.
    pub revoke_tx: watch::Sender<u64>,
    // Shared with StreamState for camera ↔ phone sync
    pub camera_active: Arc<RwLock<bool>>,
    pub camera_state_tx: broadcast::Sender<bool>,
    pub app_handle: tauri::AppHandle,
    /// Timestamp of the last agent analysis cycle, used by the status endpoint.
    pub agent_last_run: RwLock<Option<String>>,
    /// Latest JPEG frame per camera — used by Telegram snapshot command.
    pub latest_frames: Arc<RwLock<HashMap<u8, Vec<u8>>>>,
    /// Camera inventory: native, network, and browser cameras last seen.
    pub camera_inventory: Arc<RwLock<CameraInventory>>,
    /// Latest detected objects per camera slot (updated every ~5 s from frontend).
    pub scene_objects: Arc<RwLock<HashMap<u8, Vec<SceneObject>>>>,
    /// Unix timestamp (secs) of last scene update per camera — used to detect camera-off.
    pub scene_last_update: Arc<RwLock<HashMap<u8, u64>>>,
    /// Latest behavior events per camera slot (updated every 10 s from frontend tracker).
    pub behavior_events: Arc<RwLock<HashMap<u8, Vec<BehaviorEvent>>>>,
    /// Pending escalations awaiting user acknowledgement via Telegram inline keyboard.
    pub pending_escalations: Arc<RwLock<HashMap<String, crate::agent::EscalationState>>>,
    /// LibP2P PeerID for this node (base58 encoded, stable across restarts).
    /// Per-camera NVR recording process handles (ffmpeg subprocesses).
    pub nvr_processes: Arc<tokio::sync::Mutex<HashMap<u8, tokio::process::Child>>>,
    /// Per-camera live-HLS process handles. MUST be kept alive — `spawn_hls_pipe` sets
    /// `kill_on_drop(true)`, so dropping the child kills the HLS ffmpeg instantly (the
    /// bug that made `/hls/` produce nothing and forced the MJPEG fallback).
    pub hls_processes: Arc<tokio::sync::Mutex<HashMap<u8, tokio::process::Child>>>,
    /// RTSP relay subprocess handles (ffmpeg pulling remote streams).
    pub rtsp_processes: Arc<tokio::sync::Mutex<HashMap<u8, tokio::process::Child>>>,
    /// Per-camera "capture key" identifying the currently-running external capture
    /// (e.g. "dshow:Integrated Camera" or "rtsp:<url>"). Makes start_* idempotent:
    /// a duplicate start for the same key (grid tile + focused view both mounting,
    /// or a React re-render) is a NO-OP instead of a kill+respawn that gaps recording.
    pub capture_keys: Arc<tokio::sync::Mutex<HashMap<u8, String>>>,
    /// USB captures killed by a footage-delete flow, queued for respawn
    /// (cam_id, device). The capture ffmpeg records directly, so it must be
    /// stopped to release .tmp.mp4 locks and restarted afterwards.
    pub stopped_usb_captures: Arc<tokio::sync::Mutex<Vec<(u8, String)>>>,
    /// Serializes external-capture STARTS so two near-simultaneous calls (grid +
    /// focused mounting at once) can't both pass the idempotency check before either
    /// has inserted its capture_key — the TOCTOU race that spawned duplicate ffmpegs.
    pub capture_start_lock: Arc<tokio::sync::Mutex<()>>,
    /// Per-camera audio-detection ffmpeg (RTSP audio → PCM → YAMNet). Killed on stop_rtsp_relay.
    pub audio_processes: Arc<tokio::sync::Mutex<HashMap<u8, tokio::process::Child>>>,
    /// Per-camera NVR stdin pipe senders — frames piped directly to ffmpeg (no browser roundtrip).
    /// BOUNDED (drop-on-full): if the encoder stalls, frames are dropped instead of
    /// backing up in RAM without limit (~2.4 MB/s/pipe when unbounded — the classic
    /// encoder-slower-than-realtime balloon). Video degrades; memory doesn't.
    pub nvr_pipe_txs: Arc<tokio::sync::Mutex<HashMap<u8, tokio::sync::mpsc::Sender<Arc<Vec<u8>>>>>>,
    /// Per-camera HLS ffmpeg stdin pipe senders — H.264 HLS stream for bandwidth-efficient viewing.
    /// Bounded, same drop-on-full contract as `nvr_pipe_txs`.
    pub hls_pipe_txs: Arc<tokio::sync::Mutex<HashMap<u8, tokio::sync::mpsc::Sender<Arc<Vec<u8>>>>>>,
    /// ONNX inference intake — per-camera latest-frame slots drained round-robin
    /// by the inference loop (mature NVRs' shared-detection-queue model).
    pub infer_queue: Arc<InferQueue>,
    /// list_unknown_clusters result cache keyed by a cheap DB fingerprint
    /// (COUNT+MAX(seen_at) over the same window) — re-clustering 20k face
    /// descriptors on every People-tab visit froze the app; any tag/assign/
    /// delete/new-face changes the fingerprint, so no invalidation hooks.
    pub unknown_clusters_cache: tokio::sync::Mutex<Option<(String, Vec<crate::persons::UnknownCluster>)>>,
    /// The event ids the agent last put in front of the user, when, and the span
    /// it described. Lets "show me those" resolve to the SAME events instead of a
    /// fresh guess — before this there was no cross-turn state at all, so the
    /// word "those" had no possible referent.
    ///
    /// ponytail: one slot shared by the app and Telegram. Split per-chat if a
    /// second household member ever gets their own thread.
    pub last_shown: RwLock<Option<(Instant, Vec<String>, String)>>,
    /// Latest YOLO26 detections per camera (full JSON with normalised bboxes).
    /// Updated every inference tick — used by live analysis so it doesn't need to wait for event end.
    pub latest_detections: Arc<RwLock<HashMap<u8, Vec<serde_json::Value>>>>,
    /// Best available H.264 hardware encoder ("h264_nvenc", "h264_qsv", "h264_amf", or "libx264").
    pub hw_encoder: Arc<std::sync::RwLock<String>>,
    /// Best hardware video DECODER (`-hwaccel` value, "" = software). Detected at
    /// boot; offloads live detection-stream decode from the CPU to the GPU.
    pub hw_decoder: Arc<std::sync::RwLock<String>>,
    /// Durable background-job queue (Apalis/SQLite) for deferred work like event
    /// embedding. `None` if the queue couldn't be created — callers fall back to
    /// an in-process `tokio::spawn`, so behaviour degrades gracefully.
    pub embed_jobs: Option<crate::jobs::EmbedStorage>,
    /// AES-256 master key for encrypting sensitive settings at rest.
    pub master_key: [u8; 32],
    /// Desktop login gate runtime state (unlocked flag, OTP challenges, lockout).
    pub auth: crate::auth::AuthState,
    /// Active remote viewer sessions — keyed by session UUID.
    pub client_sessions: Arc<tokio::sync::RwLock<HashMap<String, ClientSession>>>,
    /// Per-session kick channels — send () to forcibly disconnect that session.
    pub kick_txs: Arc<tokio::sync::Mutex<HashMap<String, tokio::sync::oneshot::Sender<()>>>>,
    // ── v9 pull-state for the YOLO badge ───────────────────────────────
    /// Current inference-loop status, mirrored from the `inference:status`
    /// event stream so the frontend can ask for it via `get_inference_status`
    /// on mount (instead of relying on the one-shot event firing while a
    /// listener was attached). Fixes the "stuck orange ◌ YOLO26 loading…"
    /// badge when CameraView mounts after the loop already settled.
    pub inference_status: Arc<RwLock<InferenceStatusSnapshot>>,
    // ── share tracking ───────────────────────────────────────────────────
    /// Active share links: kind + resource_id + expiry. Used by the auto-stop
    /// task to decide whether the tunnel should still be alive, and by the
    /// "Revoke all shares" command to invalidate everything at once.
    pub active_shares: Arc<RwLock<Vec<ShareEntry>>>,
    /// Monotonic counter for share-token generations. Incremented by
    /// `revoke_all_shares`; tokens that don't match the current generation
    /// are rejected immediately. Mixed into the HMAC payload, so revoking
    /// invalidates every outstanding link in O(1) without a per-token table.
    pub share_generation: Arc<RwLock<u64>>,
}

/// Tracker entry for an outstanding share link.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShareEntry {
    /// "live" (cam_id) | "clip" (event_id).
    pub kind:        String,
    pub resource_id: String,
    /// Unix epoch seconds. 0 = never expires (until app restart).
    pub expires_at:  i64,
    /// rfc3339 timestamp the share was minted at (for UI display).
    pub created_at:  String,
    /// The full minted /redeem URL. Lets a repeat request for the same
    /// (kind, resource) REUSE the outstanding link instead of minting another
    /// (idempotent re-mint), and lets the UI offer copy. Old persisted entries
    /// deserialize with an empty url (serde default) — they just never reuse.
    #[serde(default)]
    pub url:         String,
}

/// Shared inference-loop status, populated by `run_inference_loop` and
/// queried by the frontend via the `get_inference_status` Tauri command.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceStatusSnapshot {
    /// "not_installed" | "loading" | "ready" | "error".
    pub state:   String,
    pub variant: Option<String>,
    pub fps:     u32,
    /// rfc3339 of the last state transition.
    pub since:   Option<String>,
}

impl Default for InferenceStatusSnapshot {
    fn default() -> Self {
        Self { state: "not_installed".to_string(), variant: None, fps: 0, since: None }
    }
}

// ─── HTTP / WebSocket server ──────────────────────────────────────────────────

// Streaming-server request state. Several fields are wired scaffolding for dormant
// features (WebRTC `signal_room`) or mirror AppState handles used only on some paths;
// allow(dead_code) keeps the server wiring intact without per-field warnings.
#[derive(Clone)]
#[allow(dead_code)]
pub(crate) struct StreamState {
    pub(crate) frame_txs: Arc<Vec<broadcast::Sender<Arc<Vec<u8>>>>>,
    pub(crate) camera_state_tx: broadcast::Sender<bool>,
    pub(crate) camera_active: Arc<RwLock<bool>>,
    /// Shared Arc with AppState — revocation takes effect instantly for all requests.
    pub(crate) auth_token: Arc<RwLock<String>>,
    pub(crate) app_handle: tauri::AppHandle,
    /// Number of currently connected WebSocket viewers (atomic for lock-free increment/decrement).
    pub(crate) connected_clients: Arc<std::sync::atomic::AtomicUsize>,
    // Per-IP failed-auth tracking for rate limiting: IP → list of attempt timestamps
    pub(crate) failed_auth: Arc<tokio::sync::Mutex<HashMap<String, Vec<Instant>>>>,
    // WebRTC signaling: relays SDP offer/answer and ICE candidates
    pub(crate) signal_room: Arc<Mutex<SignalRoom>>,
    /// SQLite pool — used by footage API endpoints.
    pub(crate) db: sqlx::SqlitePool,
    /// App data directory — used for path-traversal guard on clip files.
    pub(crate) data_dir: PathBuf,
    /// Watch receiver: open WS connections break their loop when generation changes.
    pub(crate) revoke_rx: watch::Receiver<u64>,
    /// Shared with AppState — active remote viewer sessions.
    pub(crate) client_sessions: Arc<tokio::sync::RwLock<HashMap<String, ClientSession>>>,
    /// Shared with AppState — per-session kick oneshot senders.
    pub(crate) kick_txs: Arc<tokio::sync::Mutex<HashMap<String, tokio::sync::oneshot::Sender<()>>>>,
}

// ─── Fair inference intake ────────────────────────────────────────────────────

/// Per-camera latest-frame slots drained round-robin — the in-process version of
/// Mature NVRs' common detection queue ("detectors pull from a common queue of
/// detection requests from across all cameras").
///
/// The previous design was ONE `watch` channel shared by every camera:
/// last-writer-wins, so with many cameras whichever wrote most recently erased
/// everyone else's frame — non-deterministic starvation. Here each camera owns a
/// slot (latest-wins per camera = drop-oldest backpressure stays real-time) and
/// the single consumer visits slots round-robin, so every active camera gets
/// serviced no matter how busy its neighbors are.
pub struct InferQueue {
    slots: Vec<std::sync::Mutex<InferSlot>>,
    /// Wakes the consumer; permits collapse, and the consumer drains until empty.
    pub notify: tokio::sync::Notify,
}

#[derive(Default)]
struct InferSlot {
    frame: Option<Arc<Vec<u8>>>,
    last_accept: Option<std::time::Instant>,
}

/// Per-camera intake cap (~6 fps): a 30 fps USB capture must not demand 5× the
/// detector time of a 5 fps RTSP relay. Mature NVRs budgets ~5 fps detect per cam.
const INFER_MIN_INTERVAL_MS: u128 = 150;

impl InferQueue {
    pub fn new(n_slots: usize) -> Self {
        Self {
            slots: (0..n_slots).map(|_| std::sync::Mutex::new(InferSlot::default())).collect(),
            notify: tokio::sync::Notify::new(),
        }
    }

    /// Offer a frame. Latest-wins per camera; rate-gated per camera.
    pub fn push(&self, cam: u8, frame: Arc<Vec<u8>>) {
        let idx = (cam as usize).min(self.slots.len().saturating_sub(1));
        let mut s = self.slots[idx].lock().unwrap_or_else(|e| e.into_inner());
        if s.frame.is_some() {
            // A pending frame just gets refreshed in place — no added work for
            // the consumer, fresher pixels when it arrives at this slot.
            s.frame = Some(frame);
            return;
        }
        let now = std::time::Instant::now();
        if s.last_accept.map(|t| now.duration_since(t).as_millis() < INFER_MIN_INTERVAL_MS).unwrap_or(false) {
            return; // rate gate: this camera was serviced very recently
        }
        s.frame = Some(frame);
        s.last_accept = Some(now);
        drop(s);
        self.notify.notify_one();
    }

    /// Take the next pending frame, round-robin from `cursor`. Returns None when
    /// every slot is empty (consumer should await `notify`).
    pub fn pop_rr(&self, cursor: &mut usize) -> Option<(u8, Arc<Vec<u8>>)> {
        let n = self.slots.len();
        for i in 0..n {
            let idx = (*cursor + i) % n;
            let mut s = self.slots[idx].lock().unwrap_or_else(|e| e.into_inner());
            if let Some(f) = s.frame.take() {
                *cursor = (idx + 1) % n;
                return Some((idx as u8, f));
            }
        }
        None
    }
}
