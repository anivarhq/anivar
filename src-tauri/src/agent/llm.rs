//! LLM provider integration.
//!
//! * Ollama HTTP helpers (chat completion, vision, model listing, library
//!   discovery, model pull).
//! * Multi-provider routing — Ollama / OpenAI / Anthropic / Groq / Gemini /
//!   any OpenAI-compatible endpoint.
//!
//! [`apply_auth`] is the shared bearer-token helper. It is `pub(super)` so the
//! memory module can also use it for its embedded Ollama call.

use std::time::Duration;


use crate::Settings;

// ─── Ollama HTTP helpers ──────────────────────────────────────────────────────

pub(super) fn apply_auth(req: reqwest::RequestBuilder, api_key: &str) -> reqwest::RequestBuilder {
    if api_key.is_empty() {
        req
    } else {
        req.header("Authorization", format!("Bearer {api_key}"))
    }
}

/// Shared HTTP client (connection pooling across all provider calls) — cheaper than a
/// fresh `reqwest::Client::new()` per request and the recommended pattern.
fn http() -> reqwest::Client {
    use std::sync::OnceLock;
    static C: OnceLock<reqwest::Client> = OnceLock::new();
    C.get_or_init(|| reqwest::Client::builder().build().unwrap_or_default()).clone()
}

/// POST `body` as JSON with retry on 429 / 5xx / transient network errors, using
/// exponential backoff (0.4s → 0.8s → 1.6s). `build` adds provider-specific auth/headers
/// to each fresh attempt (a RequestBuilder is consumed by `send`). This is the
/// rate-limit/transient-failure resilience every LLM-gateway best-practice calls for.
pub(super) async fn post_json_retry<B: serde::Serialize>(
    url: &str,
    body: &B,
    build: impl Fn(reqwest::RequestBuilder) -> reqwest::RequestBuilder,
) -> reqwest::Result<reqwest::Response> {
    const MAX: u32 = 3;
    let mut attempt = 0u32;
    loop {
        let req = build(http().post(url).timeout(Duration::from_secs(120)).json(body));
        match req.send().await {
            Ok(r) if (r.status().as_u16() == 429 || r.status().is_server_error()) && attempt + 1 < MAX => {
                tokio::time::sleep(Duration::from_millis(400 * 2u64.pow(attempt))).await;
                attempt += 1;
            }
            Ok(r) => return Ok(r),
            Err(e) if (e.is_timeout() || e.is_connect()) && attempt + 1 < MAX => {
                tokio::time::sleep(Duration::from_millis(400 * 2u64.pow(attempt))).await;
                attempt += 1;
            }
            Err(e) => return Err(e),
        }
    }
}

/// Guard a vision payload before sending: drop empty/oversized images and cap the count,
/// so a giant or malformed base64 frame can't trigger a silent 400/413 from the provider.
/// Frames are small JPEGs, so this is a safety net (providers cap at ~20MB; we cap lower).
pub(super) fn sanitize_images(images: Option<Vec<String>>) -> Option<Vec<String>> {
    const MAX_B64: usize = 7_000_000; // ≈5MB binary — comfortably under provider limits
    let kept: Vec<String> = images?
        .into_iter()
        .filter(|i| !i.is_empty() && i.len() <= MAX_B64)
        .take(4)
        .collect();
    if kept.is_empty() { None } else { Some(kept) }
}

/// Can the ACTIVE provider actually LOOK at an image?
///
/// The on-device engine is text-only (llama.cpp + a 350M text model), and handing
/// it a frame does not fail — it quietly answers from the prompt alone. That is how
/// a clip summary ends up confidently describing a scene nobody ever saw: observed
/// live as `"summary": "Unknown person at door"` with `person_count: 0` and
/// `person_description` echoing the schema's own placeholder text.
///
/// Check the PROVIDER, never `vision_model`: that field keeps whatever name was last
/// selected, so a leftover string reads as "vision configured" long after the engine
/// that could use it is gone. Callers that need to SEE must degrade to
/// detection-only output rather than let the model invent one.
pub(super) fn provider_supports_vision(settings: &crate::Settings) -> bool {
    !matches!(settings.ai_provider.as_str(), "local" | "")
}

/// As [`provider_supports_vision`], but knows about the on-device **Vision tier**.
///
/// Needs the data dir because "can this engine see" is, for on-device, a question
/// about which files are on disk: the VL weights AND their projector. A VL model
/// without its projector loads happily and then describes nothing, so both are
/// checked.
///
/// Deliberately NOT merged into `provider_supports_vision`: most callers only
/// hold `Settings`, and a version that silently answered "no" when it couldn't
/// see the disk would be the same trap as keying off `vision_model`.
pub(super) fn can_see(settings: &crate::Settings, data_dir: &std::path::Path) -> bool {
    if provider_supports_vision(settings) { return true; }
    super::local_llm::vision_ready(data_dir, &settings.local_llm_tier)
}

/// Is the active engine strong enough to JUDGE — risk level, threat type, "is this
/// the same event as that one"?
///
/// Deliberately separate from [`provider_supports_vision`], because they are different
/// questions that happen to have the same answer today. Vision is about perception;
/// this is about reasoning, and a text-only engine could still be a 70B model.
///
/// The on-device engine is NOT — as a standing decision, not a measured one.
///
/// **This gate is currently un-evidenced for the model it gates.** It was written
/// for the 350M default, where published benchmarks gave **17% on security
/// classification** and 25% on event deduplication (both annotated "needs larger
/// model") against 81% tool use and 91% JSON compliance — a router and an
/// extractor, not an analyst. The default is now LFM2.5-1.2B
/// (`local_llm.rs:11`), and this gate did not move with it.
///
/// It stays shut until there are numbers, because the failure is asymmetric: a
/// wrong risk level is a missed alert or a false one, not a cosmetic error. The
/// numbers are cheap to get — every "🚫 False alarm" tap is a label already in
/// `agent_alerts`. See `analysis::risk_gate_baseline_against_user_labels`, which
/// reports the current path's precision against those labels; whatever replaces
/// the rule ladder has to beat it.
///
/// Until then, on-device events take the rule-based path: a summary derived from
/// real YOLO labels beats a confident guess of unknown accuracy.
pub(super) fn provider_can_classify_risk(settings: &crate::Settings) -> bool {
    !matches!(settings.ai_provider.as_str(), "local" | "")
}

/// Thinking-family models (reason-before-answer). They need `think:false` or
/// their reasoning silently consumes the output budget; ordinary models must
/// NOT receive the param (Ollama rejects it).
pub(super) fn thinking_model(model: &str) -> bool {
    let m = model.to_lowercase();
    ["qwen3", "deepseek-r1", "magistral", "gpt-oss"].iter().any(|p| m.starts_with(p))
}

/// Belt-and-suspenders for thinking models that IGNORE `think:false` (qwen3-vl
/// does, and Ollama doesn't extract the tags on tool-enabled requests): strip
/// literal `<think>…</think>` blocks from content. An UNCLOSED `<think>` (the
/// budget ran out mid-reasoning) keeps its tail — silently returning an empty
/// string there gave the user a blank bubble and no explanation.
pub(super) fn strip_think(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    loop {
        match rest.find("<think>") {
            None => { out.push_str(rest); break; }
            Some(i) => {
                out.push_str(&rest[..i]);
                match rest[i..].find("</think>") {
                    Some(j) => rest = &rest[i + j + "</think>".len()..],
                    // Truncated mid-think. Keeping the reasoning is ugly; the
                    // alternative is worse — dropping the tail returned an
                    // EMPTY string and the user got a blank bubble with no clue
                    // why. `usable()` still rejects it if it is genuinely junk.
                    None => {
                        out.push_str(rest[i + "<think>".len()..].trim());
                        break;
                    }
                }
            }
        }
    }
    // qwen3-vl with /no_think wraps the reply in <answer>…</answer> — unwrap
    // the markers, keep the reply.
    out.replace("<answer>", "").replace("</answer>", "").trim().to_string()
}

/// Is this reply fit to show a human?
///
/// The signature failures of a small model are a degenerate repeat loop (it
/// latches onto a phrase and emits it until the token budget runs out) and a
/// two-word non-answer. Both read to the user as "the AI is broken", and both are
/// detectable without a model — so the caller can substitute a real answer instead
/// of delivering nonsense.
///
/// Deliberately permissive: this rejects text that is *structurally* broken, never
/// text that is merely wrong or terse-but-valid. A false reject costs a good answer.
pub(super) fn usable(text: &str) -> bool {
    let t = text.trim();
    if t.is_empty() || t.contains('\u{FFFD}') { return false; }
    // Measured in characters, not words. A word floor rejected "Nothing today." —
    // which is the single most common CORRECT answer a security camera can give.
    //
    // The floor was 6, which also rejected the shortest CORRECT answers a count
    // question can have: "2 cars." is five alphanumerics and "None." is four.
    // Losing those costs more than it saves — a rejected reply is replaced by a
    // raw row dump and is never written to `chat_log`, so the user sees an
    // answer the agent has no record of giving.
    //
    // 4 keeps the original intent intact: "ok" and "Yes." are still non-answers
    // (the phrasing prompt asks for two or three sentences, so a bare "Yes." is
    // a degenerate generation, not a terse one).
    if t.chars().filter(|c| c.is_alphanumeric()).count() < 4 { return false; }
    let words: Vec<&str> = t.split_whitespace().collect();
    // Repeat-loop detection by 5-word shingles. Under ~120 chars there aren't
    // enough shingles for the ratio to mean anything, so short replies pass.
    if t.len() > 120 && words.len() >= 10 {
        let shingles: Vec<String> = words.windows(5)
            .map(|w| w.join(" ").to_lowercase())
            .collect();
        let distinct: std::collections::HashSet<&String> = shingles.iter().collect();
        if (distinct.len() as f64) / (shingles.len() as f64) < 0.35 { return false; }
    }
    true
}

// ─── Multi-provider LLM routing ──────────────────────────────────────────────
// Supports: on-device (llama.cpp), openai, anthropic, groq, gemini, openai_compatible

/// Is the LLM actually usable for the SELECTED provider? Replaces the old
/// provider-BLIND URL checks, which silently disabled the entire VLM pipeline for
/// cloud users (Anthropic / OpenAI / Groq / Gemini) — events then fell back to
/// detection-only summaries with no explanation.
///
/// "Usable" = a model is selected AND the selected provider has the credential it
/// needs. The on-device provider needs neither: it has no endpoint and no model
/// tag, only a GGUF on disk.
pub fn agent_configured(settings: &Settings) -> bool {
    if matches!(settings.ai_provider.as_str(), "local" | "") {
        return super::local_llm::model_ready();
    }
    if settings.vision_model.trim().is_empty() { return false; }
    match settings.ai_provider.as_str() {
        "openai"            => !settings.openai_api_key.trim().is_empty(),
        "groq"              => !settings.groq_api_key.trim().is_empty(),
        "xai"               => !settings.xai_api_key.trim().is_empty(),
        "anthropic"         => !settings.anthropic_api_key.trim().is_empty(),
        "gemini"            => !settings.gemini_api_key.trim().is_empty(),
        "openai_compatible" => !settings.openai_compatible_url.trim().is_empty(),
        "lmstudio"          => true, // local server; defaults to http://localhost:1234/v1
        _                   => false, // unknown provider — treat as unconfigured
    }
}

/// The active model's context window, in tokens.
///
/// Two consumers: the row budget that decides how many events a prompt can carry
/// (`retrieve`), and the context ring in the composer. Both were previously a
/// hard-coded 12 and nothing respectively, which is why a 200k-context cloud
/// model was shown exactly as many events as an 8k on-device one.
///
/// Cloud numbers are a lookup on the model tag, matched loosest-last. Exact
/// accounting would need each provider's own tokenizer for a progress ring;
/// [`estimate_tokens`] is honest about being an estimate instead.
pub(super) fn context_limit(settings: &Settings) -> usize {
    if matches!(settings.ai_provider.as_str(), "local" | "") {
        return super::local_llm::context_tokens();
    }
    let m = settings.vision_model.to_lowercase();
    // Longest / most specific patterns first — "gpt-4o" must not match "gpt-4".
    for (needle, window) in [
        ("gpt-4.1", 1_000_000usize), ("gpt-5", 400_000), ("o3", 200_000), ("o1", 200_000),
        ("gpt-4o", 128_000), ("gpt-4-turbo", 128_000), ("gpt-4", 8_192), ("gpt-3.5", 16_385),
        ("claude-sonnet-4", 200_000), ("claude-opus-4", 200_000), ("claude-3", 200_000),
        ("claude", 200_000),
        ("gemini-2", 1_000_000), ("gemini-1.5", 1_000_000), ("gemini", 32_768),
        ("llama-3.3", 128_000), ("llama-3.1", 128_000), ("llama", 8_192),
        ("qwen3", 32_768), ("qwen2.5", 32_768), ("qwen", 32_768),
        ("mistral", 32_768), ("mixtral", 32_768), ("deepseek", 64_000), ("grok", 131_072),
    ] {
        if m.contains(needle) { return window; }
    }
    // Unknown model: assume the modern floor rather than the 2023 one. Being
    // slightly generous costs a truncated prompt; being wrong the other way
    // wastes most of a large window.
    32_768
}

/// Rough token count for a prompt.
///
/// ~4 characters per token is the standard English approximation, and it is what
/// BOTH paths use — including on-device. Counting exactly there would mean
/// loading the GGUF just to tokenise, which is a model load to draw a progress
/// ring. The estimate is good to roughly ±20 %, which is fine for "how full is
/// this" and for choosing a row budget; the place where exactness actually
/// matters — not overrunning the KV cache — is handled properly at generation
/// time by `local_llm::split_context`, which tokenises for real.
pub(super) fn estimate_tokens(text: &str) -> usize {
    text.len().div_ceil(4)
}

/// Unified LLM call — routes to the correct provider based on settings.
/// All providers produce a JSON string response (for structured analysis).
pub async fn call_llm(
    settings: &Settings,
    system: &str,
    user: &str,
    images: Option<Vec<String>>,   // base64 images (vision)
    json_mode: bool,               // request JSON output
) -> anyhow::Result<String> {
    let images = sanitize_images(images); // drop empty/oversized frames, cap count
    match settings.ai_provider.as_str() {
        "openai" => call_openai_compat(
            "https://api.openai.com/v1",
            &settings.openai_api_key,
            &settings.vision_model,
            system, user, images, json_mode,
        ).await,

        "groq" => call_openai_compat(
            "https://api.groq.com/openai/v1",
            &settings.groq_api_key,
            &settings.vision_model,
            system, user, images, json_mode,
        ).await,

        // xAI (Grok) — OpenAI-compatible endpoint.
        "xai" => call_openai_compat(
            "https://api.x.ai/v1",
            &settings.xai_api_key,
            &settings.vision_model,
            system, user, images, json_mode,
        ).await,

        "anthropic" => call_anthropic(
            &settings.anthropic_api_key,
            &settings.vision_model,
            system, user, images,
        ).await,

        "gemini" => call_gemini(
            &settings.gemini_api_key,
            &settings.vision_model,
            system, user, images,
        ).await,

        "openai_compatible" => call_openai_compat(
            settings.openai_compatible_url.trim_end_matches('/'),
            &settings.openai_compatible_key,
            &settings.vision_model,
            system, user, images, json_mode,
        ).await,

        // LM Studio shares the OpenAI-compatible protocol; only the default
        // URL differs. User can still override `openai_compatible_url` to
        // point at a non-default port.
        "lmstudio" => {
            let raw = settings.openai_compatible_url.trim_end_matches('/');
            let url = if raw.is_empty() { "http://localhost:1234/v1" } else { raw };
            call_openai_compat(
                url,
                &settings.openai_compatible_key,
                &settings.vision_model,
                system, user, images, json_mode,
            ).await
        }

        // On-device model — llama.cpp compiled into this binary (see
        // `agent::local_llm`). No server, no daemon, no port. Text-only by
        // design: a 350M text model has no vision tower, and `agent::clip`
        // already falls back to its text path, so images are dropped with a note
        // rather than failing the call.
        "local" => {
            let system = if json_mode {
                format!("{system}\n\nRespond with JSON only — no prose, no code fences.")
            } else {
                system.to_string()
            };
            // The Vision tier can genuinely look at the frames. Every other tier
            // is text-only, and handing IT images doesn't fail — it answers from
            // the prompt and invents a scene — so they are dropped with a note.
            let frames: Vec<Vec<u8>> = if super::local_llm::vision_available() {
                images.iter().flatten()
                    .filter_map(|b64| base64::Engine::decode(
                        &base64::engine::general_purpose::STANDARD,
                        b64.trim_start_matches("data:image/jpeg;base64,")).ok())
                    .take(2)
                    .collect()
            } else {
                if let Some(imgs) = &images {
                    if !imgs.is_empty() {
                        tracing::debug!("on-device tier is text-only — ignoring {} image(s)", imgs.len());
                    }
                }
                Vec::new()
            };
            // Deterministic by default. `json_mode` is NOT a safe proxy for
            // "wants prose": the alert-rule matcher and the question router both
            // pass false and both parse a fixed shape out of the reply
            // ("1,3" / "events today"), so sampling them creatively would make
            // rule matching non-deterministic. Prose opts IN, via
            // `call_llm_streaming` below. See `local_llm::sampler_chain`.
            if frames.is_empty() {
                super::local_llm::chat(&system, user, 512, false).await
            } else {
                super::local_llm::chat_vision(&system, user, 512, false, frames).await
            }
        }

        // Anything unrecognised (including a stale saved provider) falls back to
        // the on-device model rather than an endpoint that may not exist.
        _ => {
            let system = if json_mode {
                format!("{system}

Respond with JSON only — no prose, no code fences.")
            } else {
                system.to_string()
            };
            super::local_llm::chat(&system, user, 512, false).await
        }
    }
}

/// [`call_llm`], but streaming the reply piece by piece where the provider can.
///
/// Only the on-device engine streams: it generates token by token in-process, so
/// the pieces are already there. The HTTP providers here are called
/// non-streaming and return one complete body, so for those this is exactly
/// `call_llm` — the caller gets the whole answer at once and the UI simply has
/// nothing to animate. That asymmetry is fine: the local engine is the slow one,
/// and it is the default.
pub(super) async fn call_llm_streaming(
    settings: &Settings,
    system: &str,
    user: &str,
    on_token: Option<super::local_llm::OnToken>,
) -> anyhow::Result<String> {
    match settings.ai_provider.as_str() {
        // Streaming exists to animate a reply as it is written, and its ONE
        // caller is the phrasing pass in `retrieve` — whose entire job is to word
        // rows that Rust already computed. That is the call that was reading
        // robotic, so that is the call that gets prose sampling.
        "local" | "" => super::local_llm::chat_streaming(system, user, 512, true, on_token).await,
        _ => call_llm(settings, system, user, None, false).await,
    }
}

// ─── Provider-agnostic native tool/function calling ──────────────────────────

/// One normalised tool call the model requested.
pub(super) struct ToolCall { pub name: String, pub id: String, pub args: serde_json::Value }
/// One turn of the tool loop: assistant text + any tool calls it wants run.
pub(super) struct LlmTurn { pub content: String, pub tool_calls: Vec<ToolCall> }

/// How this model is asked to use tools.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ToolMode {
    /// The provider has a real function-calling channel: schemas go over the
    /// wire and results come back as structured turns.
    Native,
    /// No tool channel. The model writes `[TAG]`s in prose, Rust runs them and
    /// feeds the results back for a second pass (`chat::tag_tool_loop`).
    Tags,
}

/// Endpoints proven at runtime NOT to accept a `tools` array, keyed by
/// `provider/model`.
///
/// The alternative was a hardcoded allowlist of five provider strings, which
/// got the answer wrong in both directions: Anthropic and Gemini were denied
/// tools they support, and any self-hosted OpenAI-compatible server that does
/// NOT implement them got a 400 on every single turn. A capability is a
/// property of the endpoint, not of the word the user picked in a dropdown.
static NO_NATIVE_TOOLS: std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<String>>> =
    std::sync::OnceLock::new();

fn probe_key(settings: &Settings) -> String {
    format!("{}/{}", settings.ai_provider, settings.vision_model.trim())
}

/// Remember that this endpoint rejected a tool-carrying request, so the next
/// turn goes straight to the tag path instead of paying for another 400.
fn remember_no_native_tools(settings: &Settings) {
    if let Ok(mut set) = NO_NATIVE_TOOLS.get_or_init(Default::default).lock() {
        set.insert(probe_key(settings));
    }
}

/// Which tool channel to use for the configured model.
pub(super) fn tool_mode(settings: &Settings) -> ToolMode {
    // The on-device 350M model has no tool channel, and does not need one:
    // `retrieve.rs` resolves its queries deterministically in Rust and the
    // model only phrases the answer.
    if matches!(settings.ai_provider.as_str(), "local" | "") { return ToolMode::Tags; }
    if settings.vision_model.trim().is_empty() { return ToolMode::Tags; }
    let known_bad = NO_NATIVE_TOOLS.get()
        .and_then(|m| m.lock().ok().map(|s| s.contains(&probe_key(settings))))
        .unwrap_or(false);
    if known_bad { ToolMode::Tags } else { ToolMode::Native }
}

/// A single provider-agnostic tool-calling turn.
///
/// Three wire formats, one tool schema (`agent::tools::tool_schemas`):
/// OpenAI-family `/chat/completions`, Anthropic `tool_use`/`tool_result`
/// blocks, and Gemini `functionCall`/`functionResponse` parts. Callers gate on
/// [`tool_mode`] first; a `4xx` here marks the endpoint as tool-less so the
/// next turn takes the tag path without another failed round trip.
pub(super) async fn call_llm_tools(
    settings: &Settings,
    messages: &[serde_json::Value],
    tools: &[serde_json::Value],
) -> anyhow::Result<LlmTurn> {
    use serde_json::json;
    let model = settings.vision_model.trim();
    if model.is_empty() { anyhow::bail!("no model selected"); }

    match settings.ai_provider.as_str() {
        "anthropic" => return call_anthropic_tools(settings, model, messages, tools).await,
        "gemini"    => return call_gemini_tools(settings, model, messages, tools).await,
        _ => {}
    }

    // ── OpenAI-family: /chat/completions, tool `arguments` is a JSON string ────
    let (base, key) = match settings.ai_provider.as_str() {
        "openai" => ("https://api.openai.com/v1".to_string(), settings.openai_api_key.clone()),
        "groq"   => ("https://api.groq.com/openai/v1".to_string(), settings.groq_api_key.clone()),
        "xai"    => ("https://api.x.ai/v1".to_string(), settings.xai_api_key.clone()),
        "openai_compatible" => (settings.openai_compatible_url.trim_end_matches('/').to_string(), settings.openai_compatible_key.clone()),
        "lmstudio" => {
            let raw = settings.openai_compatible_url.trim_end_matches('/');
            ((if raw.is_empty() { "http://localhost:1234/v1" } else { raw }).to_string(), settings.openai_compatible_key.clone())
        }
        // An unrecognised provider string is a configuration problem, not a
        // reason to keep retrying a URL we cannot build.
        other => {
            remember_no_native_tools(settings);
            anyhow::bail!("provider '{other}' has no known endpoint");
        }
    };
    let url = format!("{}/chat/completions", base.trim_end_matches('/'));
    let body = json!({ "model": model, "messages": messages, "tools": tools, "tool_choice": "auto", "temperature": 0.2 });
    let resp = apply_auth(reqwest::Client::new().post(&url).timeout(Duration::from_secs(120)).json(&body), &key)
        .send().await?;
    let status = resp.status();
    if !status.is_success() {
        // A 4xx on a tool-carrying request is the endpoint telling us it does
        // not do tools (or does not do THIS schema). Remember it and let the
        // caller drop to tags rather than failing every turn identically.
        if status.is_client_error() { remember_no_native_tools(settings); }
        anyhow::bail!("tools HTTP {status}");
    }
    let v: serde_json::Value = resp.json().await?;
    let msg = &v["choices"][0]["message"];
    let mut calls = Vec::new();
    if let Some(arr) = msg["tool_calls"].as_array() {
        for c in arr {
            let name = c["function"]["name"].as_str().unwrap_or("").to_string();
            if name.is_empty() { continue; }
            calls.push(ToolCall {
                args: parse_tool_args(c["function"]["arguments"].as_str().unwrap_or("")),
                id: c["id"].as_str().unwrap_or("call").to_string(),
                name,
            });
        }
    }
    Ok(LlmTurn { content: msg["content"].as_str().unwrap_or("").to_string(), tool_calls: calls })
}

/// Tool arguments as JSON.
///
/// A model that emits malformed JSON used to have it silently replaced with
/// `{}` — the tool then ran with defaults or reported a bogus missing-parameter
/// error, and nobody ever learned the arguments were the problem. The raw text
/// is kept under `__malformed` so the loop can feed it back verbatim.
fn parse_tool_args(raw: &str) -> serde_json::Value {
    if raw.trim().is_empty() { return serde_json::json!({}); }
    serde_json::from_str::<serde_json::Value>(raw)
        .unwrap_or_else(|_| serde_json::json!({ "__malformed": raw }))
}

/// Anthropic Messages API with `tools` — `tool_use` blocks out,
/// `tool_result` blocks back in.
async fn call_anthropic_tools(
    settings: &Settings, model: &str,
    messages: &[serde_json::Value], tools: &[serde_json::Value],
) -> anyhow::Result<LlmTurn> {
    let key = settings.anthropic_api_key.clone();
    if key.is_empty() { anyhow::bail!("Anthropic API key not configured."); }
    // Same schemas, different wrapper: OpenAI nests under `function`, Anthropic
    // is flat with `input_schema`.
    let tools: Vec<serde_json::Value> = tools.iter().map(|t| {
        let f = &t["function"];
        serde_json::json!({
            "name": f["name"], "description": f["description"],
            "input_schema": f["parameters"],
        })
    }).collect();

    let system = messages.iter().find(|m| m["role"] == "system")
        .and_then(|m| m["content"].as_str()).unwrap_or("").to_string();
    let convo: Vec<serde_json::Value> = messages.iter()
        .filter(|m| m["role"] != "system")
        .map(anthropic_turn).collect();

    let body = serde_json::json!({
        // temperature to match every other provider. Anthropic was the only one
        // left at the API default of 1.0 on both its paths, so the same question
        // wandered further here than on OpenAI or Gemini for no chosen reason.
        "model": model, "max_tokens": 2048, "temperature": 0.2, "system": system,
        "messages": convo, "tools": tools,
    });
    let resp = reqwest::Client::new()
        .post("https://api.anthropic.com/v1/messages")
        .timeout(Duration::from_secs(120))
        .header("x-api-key", key).header("anthropic-version", "2023-06-01")
        .json(&body).send().await?;
    let status = resp.status();
    if !status.is_success() {
        if status.is_client_error() { remember_no_native_tools(settings); }
        anyhow::bail!("anthropic tools HTTP {status}");
    }
    let v: serde_json::Value = resp.json().await?;
    let mut content = String::new();
    let mut calls = Vec::new();
    for block in v["content"].as_array().unwrap_or(&Vec::new()) {
        match block["type"].as_str() {
            Some("text") => content.push_str(block["text"].as_str().unwrap_or("")),
            Some("tool_use") => calls.push(ToolCall {
                name: block["name"].as_str().unwrap_or("").to_string(),
                id:   block["id"].as_str().unwrap_or("call").to_string(),
                args: block["input"].clone(),
            }),
            _ => {}
        }
    }
    Ok(LlmTurn { content, tool_calls: calls })
}

/// One OpenAI-shaped message as an Anthropic turn. Tool results are user turns
/// carrying a `tool_result` block; assistant tool calls are `tool_use` blocks.
fn anthropic_turn(m: &serde_json::Value) -> serde_json::Value {
    if m["role"] == "tool" {
        return serde_json::json!({ "role": "user", "content": [{
            "type": "tool_result",
            "tool_use_id": m["tool_call_id"],
            "content": m["content"].as_str().unwrap_or(""),
        }]});
    }
    if let Some(tcs) = m["tool_calls"].as_array() {
        let mut blocks: Vec<serde_json::Value> = Vec::new();
        let text = m["content"].as_str().unwrap_or("");
        if !text.trim().is_empty() {
            blocks.push(serde_json::json!({ "type": "text", "text": text }));
        }
        for c in tcs {
            blocks.push(serde_json::json!({
                "type": "tool_use", "id": c["id"], "name": c["function"]["name"],
                // We stored the arguments as a JSON *string* for the OpenAI
                // wire format; Anthropic wants the object back.
                "input": c["function"]["arguments"].as_str()
                    .and_then(|s| serde_json::from_str::<serde_json::Value>(s).ok())
                    .unwrap_or_else(|| serde_json::json!({})),
            }));
        }
        return serde_json::json!({ "role": "assistant", "content": blocks });
    }
    serde_json::json!({ "role": m["role"], "content": m["content"].as_str().unwrap_or("") })
}

/// Gemini `generateContent` with `functionDeclarations`.
async fn call_gemini_tools(
    settings: &Settings, model: &str,
    messages: &[serde_json::Value], tools: &[serde_json::Value],
) -> anyhow::Result<LlmTurn> {
    let key = settings.gemini_api_key.clone();
    if key.is_empty() { anyhow::bail!("Gemini API key not configured."); }
    let decls: Vec<serde_json::Value> = tools.iter().map(|t| {
        let f = &t["function"];
        serde_json::json!({
            "name": f["name"], "description": f["description"], "parameters": f["parameters"],
        })
    }).collect();

    let system = messages.iter().find(|m| m["role"] == "system")
        .and_then(|m| m["content"].as_str()).unwrap_or("").to_string();
    let contents: Vec<serde_json::Value> = messages.iter()
        .filter(|m| m["role"] != "system")
        .map(gemini_turn).collect();

    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}");
    let body = serde_json::json!({
        "system_instruction": { "parts": [{ "text": system }] },
        "contents": contents,
        "tools": [{ "functionDeclarations": decls }],
        "generationConfig": { "temperature": 0.2, "maxOutputTokens": 2048 },
    });
    let resp = reqwest::Client::new().post(&url)
        .timeout(Duration::from_secs(120)).json(&body).send().await?;
    let status = resp.status();
    if !status.is_success() {
        if status.is_client_error() { remember_no_native_tools(settings); }
        anyhow::bail!("gemini tools HTTP {status}");
    }
    let v: serde_json::Value = resp.json().await?;
    let mut content = String::new();
    let mut calls = Vec::new();
    for (i, part) in v["candidates"][0]["content"]["parts"]
        .as_array().unwrap_or(&Vec::new()).iter().enumerate()
    {
        if let Some(t) = part["text"].as_str() { content.push_str(t); }
        if let Some(fc) = part.get("functionCall").filter(|f| !f.is_null()) {
            calls.push(ToolCall {
                name: fc["name"].as_str().unwrap_or("").to_string(),
                // Gemini has no call ids; the tool NAME is the correlation key
                // in `functionResponse`, so index keeps ours unique.
                id: format!("gem{i}"),
                args: fc["args"].clone(),
            });
        }
    }
    Ok(LlmTurn { content, tool_calls: calls })
}

/// One OpenAI-shaped message as a Gemini `contents` entry.
fn gemini_turn(m: &serde_json::Value) -> serde_json::Value {
    if m["role"] == "tool" {
        return serde_json::json!({ "role": "user", "parts": [{ "functionResponse": {
            "name": m["name"], "response": { "result": m["content"].as_str().unwrap_or("") },
        }}]});
    }
    if let Some(tcs) = m["tool_calls"].as_array() {
        let mut parts: Vec<serde_json::Value> = Vec::new();
        let text = m["content"].as_str().unwrap_or("");
        if !text.trim().is_empty() { parts.push(serde_json::json!({ "text": text })); }
        for c in tcs {
            parts.push(serde_json::json!({ "functionCall": {
                "name": c["function"]["name"],
                "args": c["function"]["arguments"].as_str()
                    .and_then(|s| serde_json::from_str::<serde_json::Value>(s).ok())
                    .unwrap_or_else(|| serde_json::json!({})),
            }}));
        }
        return serde_json::json!({ "role": "model", "parts": parts });
    }
    // Gemini's assistant role is "model"; everything else is "user".
    let role = if m["role"] == "assistant" { "model" } else { "user" };
    serde_json::json!({ "role": role, "parts": [{ "text": m["content"].as_str().unwrap_or("") }] })
}

/// OpenAI-compatible API (works for OpenAI, Groq, Ollama /v1, LM Studio, etc.)
pub(super) async fn call_openai_compat(
    base_url: &str,
    api_key: &str,
    model: &str,
    system: &str,
    user: &str,
    images: Option<Vec<String>>,
    json_mode: bool,
) -> anyhow::Result<String> {
    let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));

    // Build user content — text or multimodal with images
    let user_content: serde_json::Value = if let Some(imgs) = images {
        let mut parts = vec![serde_json::json!({ "type": "text", "text": user })];
        for img in imgs {
            parts.push(serde_json::json!({
                "type": "image_url",
                "image_url": { "url": format!("data:image/jpeg;base64,{}", img) }
            }));
        }
        serde_json::Value::Array(parts)
    } else {
        serde_json::Value::String(user.to_string())
    };

    let mut body = serde_json::json!({
        "model": model,
        "messages": [
            { "role": "system",    "content": system },
            { "role": "user",      "content": user_content },
        ],
        "temperature": 0.2,
        "max_tokens": 1024,
    });
    if json_mode {
        body["response_format"] = serde_json::json!({ "type": "json_object" });
    }

    let resp = post_json_retry(&url, &body, |r| {
        if api_key.is_empty() { r } else { r.bearer_auth(api_key) }
    }).await?.json::<serde_json::Value>().await?;
    resp["choices"][0]["message"]["content"]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow::anyhow!("OpenAI response missing content: {:?}", resp))
}

/// Anthropic Messages API (Claude models)
pub(super) async fn call_anthropic(
    api_key: &str,
    model: &str,
    system: &str,
    user: &str,
    images: Option<Vec<String>>,
) -> anyhow::Result<String> {
    if api_key.is_empty() { anyhow::bail!("Anthropic API key not configured."); }

    let mut content_parts = vec![serde_json::json!({ "type": "text", "text": user })];
    if let Some(imgs) = images {
        for img in imgs {
            content_parts.push(serde_json::json!({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": img,
                }
            }));
        }
    }

    let body = serde_json::json!({
        "model": if model.is_empty() { "claude-3-haiku-20240307" } else { model },
        "max_tokens": 1024,
        "temperature": 0.2,
        "system": system,
        "messages": [{ "role": "user", "content": content_parts }],
    });

    let resp = post_json_retry("https://api.anthropic.com/v1/messages", &body, |r| {
        r.header("x-api-key", api_key).header("anthropic-version", "2023-06-01")
    }).await?.json::<serde_json::Value>().await?;

    resp["content"][0]["text"]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow::anyhow!("Anthropic response missing text: {:?}", resp))
}

/// Google Gemini API (now VISION-capable — base64 frames go in as `inline_data`, so
/// image events get real visual analysis on Gemini instead of text-only summaries).
pub(super) async fn call_gemini(
    api_key: &str,
    model: &str,
    system: &str,
    user: &str,
    images: Option<Vec<String>>,
) -> anyhow::Result<String> {
    if api_key.is_empty() { anyhow::bail!("Gemini API key not configured."); }
    let model_id = if model.is_empty() { "gemini-1.5-flash" } else { model };
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
        model_id, api_key
    );
    // Gemini parts: the text first, then one inline_data block per image.
    let mut parts = vec![serde_json::json!({ "text": user })];
    if let Some(imgs) = images {
        for img in imgs {
            parts.push(serde_json::json!({
                "inline_data": { "mime_type": "image/jpeg", "data": img }
            }));
        }
    }
    let body = serde_json::json!({
        "system_instruction": { "parts": [{ "text": system }] },
        "contents": [{ "role": "user", "parts": parts }],
        "generationConfig": { "temperature": 0.2, "maxOutputTokens": 1024 },
    });
    let resp = post_json_retry(&url, &body, |r| r).await?
        .json::<serde_json::Value>().await?;
    resp["candidates"][0]["content"]["parts"][0]["text"]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow::anyhow!("Gemini response missing text: {:?}", resp))
}

/// Curated default models for the UI picker, per provider. These are sensible,
/// vision-first picks for a security camera (the live `/models` list from
/// `test_ai_provider` is the source of truth when reachable — these just give a
/// clean starting set + offline fallback). `category: "vision"` drives the UI's
/// "Vision" badge; only vision-capable models can analyse camera frames.
pub async fn list_provider_models(settings: &Settings) -> Vec<serde_json::Value> {
    match settings.ai_provider.as_str() {
        "openai" => vec![
            serde_json::json!({ "id": "gpt-4o", "name": "GPT-4o", "category": "vision" }),
            serde_json::json!({ "id": "gpt-4o-mini", "name": "GPT-4o mini (cheap)", "category": "vision" }),
            serde_json::json!({ "id": "gpt-4.1", "name": "GPT-4.1", "category": "vision" }),
            serde_json::json!({ "id": "gpt-4.1-mini", "name": "GPT-4.1 mini", "category": "vision" }),
        ],
        "anthropic" => vec![
            serde_json::json!({ "id": "claude-opus-4-7", "name": "Claude Opus 4.7", "category": "vision" }),
            serde_json::json!({ "id": "claude-sonnet-4-6", "name": "Claude Sonnet 4.6", "category": "vision" }),
            serde_json::json!({ "id": "claude-haiku-4-5-20251001", "name": "Claude Haiku 4.5 (cheap)", "category": "vision" }),
        ],
        "groq" => vec![
            serde_json::json!({ "id": "meta-llama/llama-4-scout-17b-16e-instruct", "name": "Llama 4 Scout (vision)", "category": "vision" }),
            serde_json::json!({ "id": "meta-llama/llama-4-maverick-17b-128e-instruct", "name": "Llama 4 Maverick (vision)", "category": "vision" }),
            serde_json::json!({ "id": "llama-3.3-70b-versatile", "name": "Llama 3.3 70B (text)", "category": "text" }),
        ],
        "xai" => vec![
            serde_json::json!({ "id": "grok-2-vision-1212", "name": "Grok 2 Vision", "category": "vision" }),
            serde_json::json!({ "id": "grok-4", "name": "Grok 4", "category": "vision" }),
        ],
        "gemini" => vec![
            serde_json::json!({ "id": "gemini-2.5-flash", "name": "Gemini 2.5 Flash", "category": "vision" }),
            serde_json::json!({ "id": "gemini-2.5-pro", "name": "Gemini 2.5 Pro", "category": "vision" }),
            serde_json::json!({ "id": "gemini-2.0-flash", "name": "Gemini 2.0 Flash (cheap)", "category": "vision" }),
        ],
        // Ollama + local OpenAI-compatible: fetched live from the server.
        _ => vec![],
    }
}


#[cfg(test)]
mod tool_wire_tests {
    use super::*;

    /// The tool loop speaks OpenAI internally and translates on the way out.
    /// A tool RESULT that doesn't reach the model is the failure mode that made
    /// the fallback path useless for a year, so both translations are pinned.
    #[test]
    fn a_tool_result_reaches_anthropic_as_a_tool_result_block() {
        let msg = serde_json::json!({
            "role": "tool", "tool_call_id": "abc", "name": "recent_events",
            "content": "3 events since 21:00",
        });
        let out = anthropic_turn(&msg);
        assert_eq!(out["role"], "user", "Anthropic carries results on a user turn");
        assert_eq!(out["content"][0]["type"], "tool_result");
        assert_eq!(out["content"][0]["tool_use_id"], "abc");
        assert_eq!(out["content"][0]["content"], "3 events since 21:00");
    }

    #[test]
    fn a_tool_result_reaches_gemini_as_a_function_response() {
        let msg = serde_json::json!({
            "role": "tool", "tool_call_id": "gem0", "name": "recent_events",
            "content": "3 events since 21:00",
        });
        let out = gemini_turn(&msg);
        // Gemini correlates by NAME, not by id — getting this wrong drops the
        // result silently and the model answers from nothing.
        assert_eq!(out["parts"][0]["functionResponse"]["name"], "recent_events");
        assert_eq!(out["parts"][0]["functionResponse"]["response"]["result"],
                   "3 events since 21:00");
    }

    /// Arguments are stored as a JSON *string* for the OpenAI wire format; both
    /// other providers want the object back.
    #[test]
    fn tool_call_arguments_are_re_inflated_for_both_providers() {
        let msg = serde_json::json!({
            "role": "assistant", "content": "",
            "tool_calls": [{ "id": "c1", "type": "function",
                "function": { "name": "search_events", "arguments": "{\"query\":\"red van\"}" } }],
        });
        assert_eq!(anthropic_turn(&msg)["content"][0]["input"]["query"], "red van");
        assert_eq!(gemini_turn(&msg)["parts"][0]["functionCall"]["args"]["query"], "red van");
        assert_eq!(gemini_turn(&msg)["role"], "model", "Gemini's assistant role is 'model'");
    }

    /// Malformed arguments used to become `{}` silently — the tool then ran with
    /// defaults, or blamed the model for a parameter it had actually supplied.
    #[test]
    fn malformed_arguments_are_kept_so_they_can_be_fed_back() {
        let v = parse_tool_args("{query: red van");
        assert_eq!(v["__malformed"], "{query: red van");
        assert!(parse_tool_args("").get("__malformed").is_none(), "empty args are just empty");
        assert_eq!(parse_tool_args("{\"query\":\"x\"}")["query"], "x");
    }

    /// Capability is a property of the endpoint, not of the provider string.
    #[test]
    fn a_rejected_endpoint_drops_to_tags_and_stays_there() {
        let mut s = crate::Settings::default();
        s.ai_provider = "openai_compatible".into();
        s.vision_model = "some-local-server-model".into();
        assert_eq!(tool_mode(&s), ToolMode::Native, "tools are tried first");

        remember_no_native_tools(&s);
        assert_eq!(tool_mode(&s), ToolMode::Tags, "a 4xx is remembered, not retried forever");

        // A different model on the same provider is a different endpoint.
        s.vision_model = "another-model".into();
        assert_eq!(tool_mode(&s), ToolMode::Native);

        // The on-device engine never claims a tool channel.
        s.ai_provider = "local".into();
        assert_eq!(tool_mode(&s), ToolMode::Tags);
    }

    /// Anthropic and Gemini were denied tools they support by a hardcoded
    /// five-provider allowlist. This is the regression test for that.
    #[test]
    fn the_flagship_cloud_providers_get_a_tool_channel() {
        let mut s = crate::Settings::default();
        for (provider, model) in [("anthropic", "claude-sonnet-4-6"),
                                  ("gemini", "gemini-2.0-flash")] {
            s.ai_provider = provider.into();
            s.vision_model = model.into();
            assert_eq!(tool_mode(&s), ToolMode::Native, "{provider} must get native tools");
        }
    }
}

#[cfg(test)]
mod usable_tests {
    use super::usable;

    #[test]
    fn rejects_structurally_broken_replies() {
        assert!(!usable(""));
        assert!(!usable("   \n "));
        assert!(!usable("ok"));
        assert!(!usable("Yes."));
        assert!(!usable("here is the \u{FFFD} answer"));  // broken decode
        // The signature 350M failure: one phrase to the token limit.
        let loop_text = "the camera saw a person at the door ".repeat(20);
        assert!(!usable(&loop_text), "repeat loop must be rejected");
    }

    #[test]
    fn accepts_real_answers() {
        assert!(usable("Nothing today."));  // terse but valid
        // The shortest correct answers a count question has. The old 6-character
        // floor rejected both, and the row dump that replaced them was worse.
        assert!(usable("2 cars."));
        assert!(usable("None."));
        assert!(usable(
            "Today, 2026-08-02: 7 events on Front Door. Three were people, one was a \
             vehicle at 14:32, and the rest were motion without a confident label. \
             The most recent was 20 minutes ago."
        ));
        // Long and varied — near the shingle threshold but legitimately diverse.
        assert!(usable(
            "At 08:14 a person approached the front door and left after ten seconds. \
             At 11:02 a white van parked on the driveway for four minutes. \
             At 16:47 the dog set off motion in the garden. Nothing else was flagged."
        ));
    }
}
