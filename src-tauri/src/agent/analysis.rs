//! Shared analysis helpers.
//!
//! THE event pipeline is [`super::clip::analyze_event_clip`]. What remains here
//! is what that pipeline and others call into: detection formatting, the
//! rule-based risk ladder used when the engine may not judge, JSON extraction
//! from chatty VLM replies, and the risk threshold comparison.
//!
//! The legacy `analyze_event` that used to live here is gone — it had no
//! callers, and it exclusively owned the alert-rule evaluator, which is why
//! user-written rules never fired.

use serde::Deserialize;

// ─── Core analysis ────────────────────────────────────────────────────────────

pub(super) fn format_detections(json: Option<&str>) -> String {
    let Some(j) = json else { return "none".to_string() };
    #[derive(Deserialize)]
    struct Det { label: String, score: f32 }
    let Ok(dets) = serde_json::from_str::<Vec<Det>>(j) else { return "none".to_string() };
    if dets.is_empty() { return "none".to_string() }
    dets.iter()
        .map(|d| format!("{} ({:.0}%)", d.label, d.score * 100.0))
        .collect::<Vec<_>>()
        .join(", ")
}

// `analyze_event` DELETED. It was the legacy analysis path with no callers —
// and it exclusively owned `evaluate_alert_conditions`, which is why the
// user's plain-English alert rules never fired. The evaluator now hangs off
// `clip::analyze_event_clip`, the pipeline that actually runs.

/// Strip common VLM JSON wrappers and extract the first balanced top-level {...}.
/// Handles:
///   ```json\n{...}\n```      ← Qwen2.5-VL, LLaVA habit
///   ```\n{...}\n```
///   "Here is the analysis: {...}"   ← chatty preambles
///   Nested objects — only matches the OUTERMOST braces correctly.
pub(super) fn extract_json_block(raw: &str) -> String {
    let s = raw.trim();

    // Step 1: strip markdown code fences
    let stripped = if let Some(rest) = s.strip_prefix("```json").or_else(|| s.strip_prefix("```")) {
        // Find closing fence and slice out the middle
        rest.rsplit_once("```").map(|(inner, _)| inner).unwrap_or(rest).trim()
    } else {
        s
    };

    // Step 2: find first '{' and walk to its matching '}' (depth-tracked)
    let bytes = stripped.as_bytes();
    let Some(start) = bytes.iter().position(|&b| b == b'{') else {
        return stripped.to_string();
    };
    let mut depth = 0i32;
    let mut in_str = false;
    let mut esc = false;
    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if esc { esc = false; continue; }
        if in_str {
            match b { b'\\' => esc = true, b'"' => in_str = false, _ => {} }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return stripped[start..=i].to_string();
                }
            }
            _ => {}
        }
    }
    // No balanced close — return what we have stripped
    stripped.to_string()
}

/// The model's word for how bad this is, mapped onto our four levels.
///
/// The synonyms matter. Every unrecognised string used to fall through to
/// "normal", which is the level that raises NO alert — so a model that answered
/// "elevated", "severe", "warning" or "danger" had its judgement silently
/// discarded and the event went unreported. Downgrading an alert on a spelling
/// difference is the worst direction for this function to fail in, so anything
/// still unrecognised now lands on "monitor": logged, not alerted, and visible.
pub(super) fn normalise_risk(r: &str) -> &'static str {
    match r.trim().to_lowercase().as_str() {
        "low" | "none" | "normal" | "routine" | "ok" | "safe"      => "normal",
        "medium" | "moderate" | "monitor" | "watch" | "elevated" |
        "caution" | "warning" | "unusual"                          => "monitor",
        "high" | "suspicious" | "severe" | "concerning" | "threat"  => "suspicious",
        "critical" | "emergency" | "danger" | "urgent" | "alarm"    => "critical",
        other => {
            if !other.is_empty() {
                tracing::warn!(risk = other, "unrecognised risk level - treating as monitor");
            }
            "monitor"
        }
    }
}

pub(super) fn risk_meets_threshold(risk: &str, min: &str) -> bool {
    // "off" (set from the Telegram /menu) suppresses ALL alerts.
    if min.eq_ignore_ascii_case("off") { return false; }
    // Supports both old scale (low/medium/high) and new Agies scale (normal/monitor/suspicious/critical)
    let order = ["normal", "monitor", "suspicious", "critical"];
    let risk = normalise_risk(risk);
    // An UNSET threshold is not an unrecognised one. `normalise_risk` now sends
    // anything it doesn't know to "monitor" so a model's odd wording can never
    // silently downgrade an alert — but applying that to the user's own setting
    // would make a blank threshold stricter than the shipped default, not looser.
    let min = if min.trim().is_empty() { "suspicious" } else { normalise_risk(min) };
    let ri = order.iter().position(|&r| r == risk).unwrap_or(1);
    let mi = order.iter().position(|&r| r == min).unwrap_or(1);
    ri >= mi
}


// ─── Risk-gate evaluation ─────────────────────────────────────────────────────

/// Measure the risk classifier against the user's OWN labels.
///
/// `provider_can_classify_risk` (`llm::123`) denies the on-device engine any risk
/// verdict, citing 17% security-classification accuracy. That number belongs to
/// the 350M model; the default is now LFM2.5-1.2B (`local_llm.rs:11`) and the
/// gate never moved with it. Before opening it — or leaving it shut on purpose —
/// there should be a number.
///
/// Ground truth already exists and cost nothing to collect: every "🚫 False
/// alarm" tap sets `agent_alerts.is_false_positive`. This reports how the
/// CURRENT path does against those taps, which is the baseline any replacement
/// has to beat.
///
/// Run it against a real archive:
/// ```text
/// SC_EVAL_DB="C:\Users\you\AppData\Roaming\com.anivar.app\anivar.db" \
///   cargo +1.97 test --lib risk_gate -- --ignored --nocapture
/// ```
#[cfg(test)]
#[tokio::test]
#[ignore = "needs a real archive — set SC_EVAL_DB to your anivar.db"]
async fn risk_gate_baseline_against_user_labels() {
    let Ok(path) = std::env::var("SC_EVAL_DB") else {
        eprintln!("SC_EVAL_DB not set — nothing to measure against.");
        return;
    };
    let pool = sqlx::SqlitePool::connect(&format!("sqlite://{path}?mode=ro"))
        .await.expect("open archive read-only");

    let rows: Vec<(String, i64, Option<String>)> = sqlx::query_as(
        "SELECT risk_level, is_false_positive, feedback FROM agent_alerts"
    ).fetch_all(&pool).await.expect("read agent_alerts");

    if rows.is_empty() {
        eprintln!("No alerts in this archive yet — nothing to measure.");
        return;
    }
    // A label is a tap, not an absence of one: an alert nobody marked is not
    // evidence that it was correct.
    let labelled: Vec<&(String, i64, Option<String>)> = rows.iter()
        .filter(|(_, fp, fb)| *fp == 1 || fb.as_deref().is_some_and(|f| !f.trim().is_empty()))
        .collect();

    println!("\n── Risk gate baseline ──────────────────────────────────────");
    println!("alerts in archive : {}", rows.len());
    println!("user-labelled     : {}", labelled.len());
    if labelled.is_empty() {
        println!("\nNo labels yet. Tap 🚫 False alarm on a few alerts, then re-run.");
        return;
    }

    for level in ["normal", "monitor", "suspicious", "critical"] {
        let at: Vec<&&(String, i64, Option<String>)> = labelled.iter()
            .filter(|(r, _, _)| normalise_risk(r) == level).collect();
        if at.is_empty() { continue; }
        let wrong = at.iter().filter(|(_, fp, _)| *fp == 1).count();
        println!("{level:<11} {:>4} labelled, {:>4} false alarms  ({:.0}% wrong)",
            at.len(), wrong, wrong as f64 * 100.0 / at.len() as f64);
    }

    // The number that decides the gate: of everything the current path thought
    // was worth waking someone for, how much was noise.
    let alerted: Vec<&&(String, i64, Option<String>)> = labelled.iter()
        .filter(|(r, _, _)| risk_meets_threshold(r, "suspicious")).collect();
    if !alerted.is_empty() {
        let wrong = alerted.iter().filter(|(_, fp, _)| *fp == 1).count();
        println!("\nsuspicious+ precision: {:.0}%  ({} of {} were real)",
            (alerted.len() - wrong) as f64 * 100.0 / alerted.len() as f64,
            alerted.len() - wrong, alerted.len());
        println!("\nThis is the bar a model-judged risk has to beat before");
        println!("`provider_can_classify_risk` should open for on-device.");
    }
    println!("────────────────────────────────────────────────────────────\n");
}
