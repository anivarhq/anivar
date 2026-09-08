//! Guardian — local AI security agent.
//!
//! Module map (each submodule has its own crate-level docs):
//!
//! | Submodule    | Responsibility                                                                          |
//! |--------------|-----------------------------------------------------------------------------------------|
//! | [`types`]    | Public output types, Ollama wire schema, analysis JSON shape, small helpers             |
//! | [`memory`]   | `agent_memory` table: flat KV, scored learned memory, narrative markdown files          |
//! | [`llm`]      | Multi-provider LLM routing (Ollama, OpenAI, Anthropic, Groq, Gemini, OpenAI-compatible) |
//! | [`dispatch`] | Alert dispatch + multi-channel fan-out (Telegram, Discord, Slack, Pushover, Signal)     |
//! | [`analysis`] | Shared analysis helpers (JSON extraction, risk thresholds). NOTE: the live event pipeline is `clip::analyze_event_clip`; the `analyze_event` fn here is LEGACY and unwired. |
//! | [`chat`]     | `ChatMessage`, `Mode`, `chat_with_agent` (THE brain, both surfaces)                     |
//! | [`evidence`] | THE tag resolver: one reply → typed evidence both surfaces render                       |
//! | [`cycle`]    | Background loops: agent cycle, escalation, reflection, main loop, model pull            |
//! | [`conditions`] | Semantic alert conditions (plain-English rules evaluated by the LLM)                  |
//! | [`clip`]    | THE event analysis pipeline (`analyze_event_clip`): face/plate recognition + multi-frame recall + frame annotation + VLM + live-event + backfill |
//! | [`util`]    | Status helper, snapshot test, disk-guard, heartbeat, cross-camera context              |

#![allow(dead_code)]

mod analysis;
mod chat;
mod clip;
pub(crate) mod clip_export;
pub(crate) mod conditions;
mod cycle;
mod dispatch;
/// THE evidence resolver: the agent's tagged reply, parsed once into typed
/// evidence both surfaces render identically.
pub(crate) mod evidence;
/// On-device LLM (llama.cpp in-process) — replaces the Ollama daemon.
pub(crate) mod local_llm;
mod llm;
pub mod memory;
mod retrieve;
mod slots;
mod tools;
mod types;
mod util;

// ── Re-exports ─────────────────────────────────────────────────────────────
// These items are part of the agent's public surface. They are referenced
// from `lib.rs` (Tauri command bodies, the AppState type, the main run loop).

// `analysis` exposes only `pub(super)` items today — no re-export needed.
pub use chat::{chat_with_agent, context_usage, ChatMessage, Mode};
pub use clip::{
    analyze_event_clip, dispatch_intelligence_alert, run_backfill_analysis,
    run_live_alert_loop, embed_event,
};
// Crate-internal classification helpers (used by the inference loop).
pub(crate) use clip::{categorise_labels, dominant_label};
// `evaluate_alert_conditions` is NOT re-exported: it has exactly one caller,
// `clip::analyze_event_clip`, and a second entry point is how it ended up
// wired to a dead function in the first place.
pub use conditions::{
    create_alert_condition, day_chart_mermaid, day_chart_text, delete_alert_condition,
    explore_events, list_alert_conditions,
    query_events_nl, search_clips, toggle_alert_condition, AlertCondition,
};
pub use cycle::{run_agent_loop, run_now};
pub use dispatch::run_telegram_loop;
pub(crate) use dispatch::send_telegram;
pub use llm::list_provider_models;
pub use memory::{
    append_to_memory, read_all_memory_files, read_memory, read_memory_file,
    reinforce_memory, write_memory, write_memory_file,
};
pub use types::{AgentAlert, AgentStatus, EscalationState};
pub use util::{analyze_snapshot, get_status};

/// THE evidence resolver — the only way out of the agent for a tagged reply.
///
/// `parse_tags` is deliberately NOT re-exported any more: every surface used to
/// scan the reply for itself, and the two scanners drifted. Surfaces get typed
/// evidence now, never the tag string.
pub use evidence::{replay as replay_evidence, resolve as resolve_evidence, Evidence};

/// Run one registry tool and get its text, for surfaces that render a tag's
/// result inline. Returns an empty string for action/media tools, which the
/// surface renders itself.
pub async fn run_tool(
    state: &std::sync::Arc<crate::AppState>,
    name: &str,
    args: &serde_json::Value,
) -> String {
    tools::execute(state, name, args).await.unwrap_or_default()
}
