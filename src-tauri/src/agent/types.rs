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

// ─── Agent analysis schema ────────────────────────────────────────────────────

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
}

// ─── Default helpers (serde-default callbacks) ───────────────────────────────

pub(super) fn level_normal()      -> String { "normal".to_string() }
pub(super) fn type_unknown()      -> String { "unknown".to_string() }
pub(super) fn unknown()           -> String { "unknown".to_string() }
pub(super) fn default_confidence() -> f32   { 0.7 }
pub(super) fn yes()               -> bool   { true }

// ─── JSON encoder ────────────────────────────────────────────────────────────

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

#[cfg(test)]
mod schema_doc_tests {
    use super::make_summary_json;

    /// docs/SCHEMAS.md documents the stored `ai_summary` format. Its example must
    /// have exactly the fields the writer produces (v2 + `attributes` for v3), so
    /// a format change can't leave the doc behind.
    #[test]
    fn the_documented_ai_summary_matches_what_is_written() {
        let doc = include_str!("../../../docs/SCHEMAS.md");
        let start = doc.find("```json\n").or_else(|| doc.find("```json\r\n")).expect("an example");
        let body = &doc[start..];
        let body = &body[body.find('\n').unwrap() + 1..];
        let example: serde_json::Value = serde_json::from_str(&body[..body.find("```").unwrap()]).expect("valid JSON");

        let written: serde_json::Value = serde_json::from_str(
            &make_summary_json("t", "monitor", "person", "x", "d", &["parcel".into()], 0.5, false, 1)).unwrap();
        let mut want: Vec<&str> = written.as_object().unwrap().keys().map(String::as_str).collect();
        want.push("attributes");
        want.sort_unstable();
        let mut have: Vec<&str> = example.as_object().unwrap().keys().map(String::as_str).collect();
        have.sort_unstable();
        assert_eq!(have, want, "docs/SCHEMAS.md's example has drifted from make_summary_json");
        assert_eq!(example["v"], 3);
        for a in example["attributes"].as_array().unwrap() {
            assert!(["face", "plate", "color", "outfit", "object"].contains(&a["type"].as_str().unwrap()));
        }
        assert!(!super::super::memory::extract_summary_text(&example.to_string()).is_empty());
    }
}
