//! Agent types: public output types, Ollama wire types, and the analysis
//! schema (LLM-produced JSON shape).
//!
//! Everything internal to the agent is `pub(super)` so sibling submodules can
//! import without re-exporting through `mod.rs`. The truly public surface
//! (consumed by `crate::lib.rs`) is `pub`.

use serde::{Deserialize, Serialize};

use crate::Settings;

// ─── Public output types ──────────────────────────────────────────────────────

/// State of a pending alert awaiting user acknowledgement via Telegram inline keyboard.
#[derive(Debug, Clone)]
pub struct EscalationState {
    pub alert_id: String,
    pub risk_level: String,
    pub summary: String,
    pub sent_at: std::time::Instant,
    pub timeout_secs: u64,
    pub acknowledged: bool,
}

/// A user-defined object Guardian will watch for and alert if it goes missing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct TrackedObject {
    pub(super) label: String,          // COCO label to match, e.g. "bottle", "handbag"
    pub(super) display_name: String,   // Human name the user used, e.g. "water bottle"
    pub(super) alert_after_mins: f32,
    pub(super) last_seen_at: Option<String>, // RFC3339
    pub(super) alert_sent: bool,
}

/// Agies-inspired event_subscribe: user-defined proactive alert rules.
/// Stored as individual `alert_rule_{id}` keys in agent_memory.
/// When a rule matches, the alert is sent even if below the normal risk threshold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct AlertRule {
    pub(super) id: String,
    /// Human description of the rule, e.g. "person detected after 10pm"
    pub(super) description: String,
    /// Optional: only match this threat_type ("person"|"vehicle"|"animal"|...)
    #[serde(default)]
    pub(super) threat_type: Option<String>,
    /// Optional: only match in this hour range (wraps midnight when start > end)
    #[serde(default)]
    pub(super) hours_start: Option<u32>,
    #[serde(default)]
    pub(super) hours_end: Option<u32>,
    /// Optional: minimum risk level to trigger ("normal"|"monitor"|"suspicious"|"critical")
    #[serde(default)]
    pub(super) min_risk: Option<String>,
    /// If true, override risk threshold and send alert when this rule matches
    #[serde(default = "yes")]
    pub(super) force_alert: bool,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentAlert {
    pub id: String,
    pub event_id: String,
    pub risk_level: String,   // "normal" | "monitor" | "suspicious" | "critical"
    pub threat_type: String,  // "person" | "vehicle" | "animal" | "package_delivery" | "false_alarm" | "unknown"
                              // SmartHome: "wildlife" | "elderly_concern" | "baby_unsupervised" | "pet_anomaly" | "package_theft" | "appliance_hazard"
    pub summary: String,
    pub is_false_positive: bool,
    pub actions_taken: Option<String>, // JSON array of strings
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct AgentStatus {
    pub enabled: bool,
    pub provider_ready: bool,
    pub last_run_at: Option<String>,
    pub model: String,
    pub vision_model: String,
    pub pending_events: i64,
    pub total_analyzed: i64,
}

// ─── Ollama wire types ────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, Clone)]
pub(super) struct OllamaMsg {
    pub(super) role: String,
    pub(super) content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) images: Option<Vec<String>>, // base64 strings for vision models
}

#[derive(Debug, Serialize)]
pub(super) struct OllamaChatReq {
    pub(super) model: String,
    pub(super) messages: Vec<OllamaMsg>,
    pub(super) stream: bool,
    pub(super) format: String, // "json" — requests JSON-mode output
    pub(super) options: OllamaOpts,
    /// `Some(false)` for thinking-family models (qwen3, deepseek-r1…): without
    /// it their reasoning burns the whole num_predict budget and `content`
    /// comes back EMPTY — the agent chat showed no reply at all. Omitted for
    /// ordinary models (Ollama rejects the param on non-thinking models).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) think: Option<bool>,
}

#[derive(Debug, Serialize)]
pub(super) struct OllamaOpts {
    pub(super) temperature: f32,
    pub(super) num_predict: u32,
    /// Cap the loaded context. Without this Ollama sizes the KV cache to the
    /// MODEL'S max (gemma4:e4b = 131k → 9.8 GB — didn't fit the 8 GB GPU,
    /// split to CPU, and its llama-server segfaulted mid-run, destabilizing
    /// the shared GPU and taking the NVR down with it). Our prompts are a few
    /// KB; 8k is generous and keeps the model fully on-GPU.
    pub(super) num_ctx: u32,
}

#[derive(Debug, Deserialize)]
pub(super) struct OllamaChatResp {
    pub(super) message: Option<OllamaMsg>,
    #[serde(default)]
    pub(super) done: bool,
    // Ollama returns this on errors (e.g. model not found)
    pub(super) error: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct OllamaTagsResp {
    pub models: Vec<OllamaModelInfo>,
}

#[derive(Debug, Deserialize)]
pub struct OllamaModelInfo {
    pub name: String,
}

// ─── Agent analysis schema ────────────────────────────────────────────────────

#[derive(Debug, Deserialize, Default)]
pub(super) struct MemoryUpdates {
    /// Append a new pattern observation (timestamped automatically)
    #[serde(default)]
    pub(super) patterns: Option<String>,
    /// Append a new known false positive entry
    #[serde(default)]
    pub(super) known_false_positives: Option<String>,
    /// Rewrite threat rules if they need updating
    #[serde(default)]
    pub(super) threat_rules: Option<String>,
    /// Rewrite camera profile if it needs updating
    #[serde(default)]
    pub(super) camera_profile: Option<String>,
    /// Per-person observations (key = person name, value = observation to append)
    #[serde(default)]
    pub(super) person_notes: Option<std::collections::HashMap<String, String>>,
}

/// Structured output from the clip/live VLM analysis path.
/// Separate from the full ACE-loop `Analysis` so the clip path can use
/// JSON mode without dragging in memory-update fields.
#[derive(Debug, Deserialize, Default)]
pub(super) struct ClipAnalysis {
    /// 3-6 word headline (standard review title). Optional — older
    /// models may omit it; the UI falls back to the summary's first words.
    #[serde(default)]
    pub(super) title: String,
    #[serde(default = "level_normal")]
    pub(super) risk_level: String,
    /// "person" | "vehicle" | "animal" | "package_delivery" | "motion" | "false_alarm"
    /// SmartHome: "wildlife" | "elderly_concern" | "baby_unsupervised" | "pet_anomaly" | "package_theft" | "appliance_hazard"
    #[serde(default = "type_unknown")]
    pub(super) threat_type: String,
    /// 1-2 sentence description — only confirmed visuals, no guessing
    #[serde(default)]
    pub(super) summary: String,
    /// "male" | "female" | "group" | "unknown" — ONLY when a human is clearly visible
    #[serde(default = "unknown")]
    pub(super) gender: String,
    /// Number of distinct humans visible (0 if no person)
    #[serde(default)]
    pub(super) person_count: u32,
    /// Appearance: gender, height, build, hair, clothing. Empty if no person.
    #[serde(default)]
    pub(super) person_description: String,
    /// 0.0–1.0 — how confident the model is
    #[serde(default = "default_confidence")]
    pub(super) confidence: f32,
    #[serde(default)]
    pub(super) is_false_positive: bool,
    /// Notable non-person objects (bags, vehicles, animals, packages)
    #[serde(default)]
    pub(super) objects_seen: Vec<String>,
    /// Person Re-ID: matches a previously seen person description (set by reid_match, not VLM)
    #[serde(default)]
    pub(super) matches_previous: Option<String>,
    /// True if the same person has been seen before (set by reid_match or VLM)
    #[serde(default)]
    pub(super) is_recurring: bool,
}

#[derive(Debug, Deserialize, Default)]
pub(super) struct Analysis {
    /// Short headline (optional — models without the field just omit it).
    #[serde(default)]
    pub(super) title: String,
    #[serde(default = "level_monitor")]
    pub(super) risk_level: String,
    #[serde(default = "type_unknown")]
    pub(super) threat_type: String,
    #[serde(default)]
    pub(super) summary: String,
    #[serde(default)]
    pub(super) is_false_positive: bool,
    #[serde(default)]
    pub(super) recommended_action: String,
    #[serde(default = "yes")]
    pub(super) alert_worthy: bool,
    #[serde(default)]
    pub(super) trigger_alarm: bool,
    #[serde(default)]
    pub(super) telegram_message: Option<String>,
    #[serde(default)]
    pub(super) memory_updates: Option<MemoryUpdates>,
    /// How confident the agent is (0-1). Low confidence = don't notify, just log.
    #[serde(default = "default_confidence")]
    pub(super) confidence: f32,
    /// "male" | "female" | "group" | "unknown" — from visual analysis
    #[serde(default = "unknown")]
    pub(super) gender: String,
    /// Number of distinct humans visible
    #[serde(default)]
    pub(super) person_count: u32,
    /// Physical appearance — gender first, then height, build, hair, clothing
    #[serde(default)]
    pub(super) person_description: String,
    #[serde(default)]
    pub(super) matches_previous: Option<String>,
    #[serde(default)]
    pub(super) is_recurring: bool,
}

// ─── Default helpers (serde-default callbacks) ───────────────────────────────

pub(super) fn level_normal()      -> String { "normal".to_string() }
pub(super) fn level_monitor()     -> String { "monitor".to_string() }
pub(super) fn type_unknown()      -> String { "unknown".to_string() }
pub(super) fn unknown()           -> String { "unknown".to_string() }
pub(super) fn default_confidence() -> f32   { 0.7 }
pub(super) fn yes()               -> bool   { true }

// ─── Rule-based fallback + JSON encoder ──────────────────────────────────────

/// Rule-based analysis fallback — used when LLM is unavailable or returns garbage.
/// Ensures the agent is ALWAYS useful even without AI connectivity.
// The risk ladder keeps one branch per REASON (crowd, evening, unexplained
// motion), several of which land on the same level. Collapsing them into a
// single condition would satisfy clippy and delete the reasoning.
#[allow(clippy::if_same_then_else)]
pub(super) fn rule_based_analysis(
    peak_score: f32,
    duration_val: f64,
    detections_json: Option<&str>,
    hour_num: i32,
    monitoring_days: u32,
) -> Analysis {
    let has_person = detections_json.map(|j| j.contains("\"person\"")).unwrap_or(false);
    let has_vehicle = detections_json.map(|j| j.contains("\"car\"") || j.contains("\"truck\"")).unwrap_or(false);
    let has_animal = detections_json.map(|j| j.contains("\"dog\"") || j.contains("\"cat\"") || j.contains("\"bird\"")).unwrap_or(false);
    let is_night = !(6..22).contains(&hour_num);
    let is_extended = duration_val > 45.0;

    let risk_level = if peak_score > 0.6 && is_night { "suspicious".to_string() }
        else if peak_score > 0.6 && has_person { "monitor".to_string() }
        else if peak_score > 0.3 && is_night  { "monitor".to_string() }
        else { "normal".to_string() };

    let threat_type = if has_person { "person" }
        else if has_vehicle { "vehicle" }
        else if has_animal { "animal" }
        else { "motion" }.to_string();

    // Contextual summary — describe behaviour, never echo raw numbers
    let summary = if has_person {
        if is_night && is_extended  { "Person present on camera at night with prolonged activity.".into() }
        else if is_night            { "Person detected on camera during night hours.".into() }
        else if is_extended         { "Person visible in the camera view for an extended period.".into() }
        else if peak_score > 0.5   { "Person moving through the camera's field of view.".into() }
        else                        { "Person detected in the camera view.".into() }
    } else if has_vehicle {
        if is_night { "Vehicle activity detected on camera at night.".into() }
        else        { "Vehicle movement detected in the camera view.".into() }
    } else if has_animal {
        if is_night { "Animal movement detected on camera at night.".into() }
        else        { "Animal activity detected in the camera view.".into() }
    } else if is_night && peak_score > 0.4 {
        "Unexplained motion detected at night — origin unclear.".into()
    } else if is_night {
        "Motion detected during night hours.".into()
    } else if peak_score > 0.5 {
        "Significant motion in the camera view.".into()
    } else {
        "Low-level background motion detected.".into()
    };

    // Low confidence on rule-based: don't send notifications, just log
    let confidence = if monitoring_days < 3 { 0.35 } else { 0.5 };

    Analysis {
        risk_level,
        threat_type,
        summary,
        alert_worthy: peak_score > 0.3 && (is_night || has_person),
        confidence,
        ..Default::default()
    }
}

/// Encode a completed analysis as the v2 structured JSON stored in motion_events.ai_summary.
/// The UI parses this to render risk badges, type chips, person details, and object lists.
/// Falls back gracefully: if the UI can't parse it, it just shows `text` as plain string.
pub(super) fn make_summary_json(
    title: &str,
    risk: &str,
    ttype: &str,
    text: &str,
    description: &str,
    objects: &[String],
    confidence: f32,
    fp: bool,
    persons: u32,
) -> String {
    serde_json::json!({
        "v": 2,
        "title": title,
        "risk": risk,
        "type": ttype,
        "text": text,
        "description": description,
        "objects": objects,
        "confidence": confidence,
        "fp": fp,
        "persons": persons,
    }).to_string()
}

/// Returns the agent identity line for system prompts.
/// Uses the user-configured persona name + personality if set, otherwise "Guardian".
pub(super) fn agent_identity(settings: &Settings, camera_name: &str) -> String {
    let name = if settings.agent_persona_name.trim().is_empty() {
        "Guardian".to_string()
    } else {
        settings.agent_persona_name.trim().to_string()
    };
    let persona = if settings.agent_persona_text.trim().is_empty() {
        String::new()
    } else {
        format!("\nPersonality: {}", settings.agent_persona_text.trim())
    };
    format!(
        "You are {name}, an AI security analyst watching \"{camera_name}\".{persona}"
    )
}
