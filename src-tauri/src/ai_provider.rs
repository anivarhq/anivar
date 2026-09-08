//! Multi-provider LLM management Tauri commands (on-device / OpenAI / Anthropic / Gemini / Groq / OpenAI-compatible).

use std::sync::Arc;

use tauri::State;

use crate::AppState;


/// List models for the currently configured AI provider.
/// For the on-device provider: the single installed GGUF.
/// For cloud providers: a hardcoded curated list.
/// `provider` (optional) lets the UI list models for a provider the user is just
/// BROWSING, without committing it as the active engine. When omitted, falls back
/// to the saved `ai_provider`.
#[tauri::command]
pub async fn list_provider_models(
    provider: Option<String>,
    state: State<'_, Arc<AppState>>,
) -> Result<Vec<serde_json::Value>, String> {
    let mut settings = state.settings.read().await.clone();
    if let Some(p) = provider {
        if !p.is_empty() { settings.ai_provider = p; }
    }
    Ok(crate::agent::list_provider_models(&settings).await)
}

/// Validate the active provider's connection / API key by hitting its cheapest
/// "list models" endpoint. Returns `{ ok, models, count, error? }` consistently
/// across providers. Short 5 s timeout — wrong keys should fail fast, not hang.
///
/// Endpoints used (each is a vanilla GET, no streaming, no billable inference):
///   • On-device — no endpoint; checks the model file is on disk
///   • OpenAI    — `https://api.openai.com/v1/models`            (Bearer)
///   • Anthropic — `https://api.anthropic.com/v1/models`         (x-api-key)
///   • Groq      — `https://api.groq.com/openai/v1/models`       (Bearer)
///   • Gemini    — `…/v1beta/models?key=…`                       (query param)
#[tauri::command]
pub async fn test_ai_provider(
    provider: Option<String>,
    state: State<'_, Arc<AppState>>,
) -> Result<serde_json::Value, String> {
    let mut settings = state.settings.read().await.clone();
    // Browse-without-commit: test the provider the UI is VIEWING, not necessarily
    // the saved active one. Credentials are stored per-field, so overriding just
    // the provider selector is enough to validate any provider's key/URL.
    if let Some(p) = provider {
        if !p.is_empty() { settings.ai_provider = p; }
    }
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .map_err(|e| e.to_string())?;

    let send = |req: reqwest::RequestBuilder| async move {
        match req.send().await {
            Ok(r) if r.status().is_success() => Ok(r.json::<serde_json::Value>().await.unwrap_or_default()),
            Ok(r)  => Err(format!("HTTP {}", r.status())),
            Err(e) => Err(e.to_string()),
        }
    };

    match settings.ai_provider.as_str() {
        // On-device: there is no endpoint to reach. "Connected" means the model
        // file is on disk — the engine is compiled into this binary, so nothing
        // else can be misconfigured.
        "local" | "" => {
            let tier = state.settings.read().await.local_llm_tier.clone();
            let installed = crate::agent::local_llm::is_installed_tier(&state.data_dir, &tier);
            Ok(serde_json::json!({
                "ok": installed,
                "count": if installed { 1 } else { 0 },
                "models": if installed { vec![format!("{} (on-device)", crate::agent::local_llm::tier_label(&tier))] } else { vec![] },
                "error": if installed { serde_json::Value::Null }
                         else { serde_json::json!("on-device model not installed yet — use Install on-device AI") },
            }))
        }

        "openai" => {
            if settings.openai_api_key.is_empty() {
                return Ok(serde_json::json!({ "ok": false, "error": "Missing OpenAI API key" }));
            }
            let req = client.get("https://api.openai.com/v1/models")
                .bearer_auth(&settings.openai_api_key);
            match send(req).await {
                Ok(data) => {
                    let models: Vec<String> = data["data"].as_array()
                        .map(|a| a.iter().filter_map(|m| m["id"].as_str().map(|s| s.to_string())).collect())
                        .unwrap_or_default();
                    Ok(serde_json::json!({ "ok": true, "models": models, "count": models.len() }))
                }
                Err(e) => Ok(serde_json::json!({ "ok": false, "error": e })),
            }
        }
        "anthropic" => {
            if settings.anthropic_api_key.is_empty() {
                return Ok(serde_json::json!({ "ok": false, "error": "Missing Anthropic API key" }));
            }
            let req = client.get("https://api.anthropic.com/v1/models")
                .header("x-api-key", &settings.anthropic_api_key)
                .header("anthropic-version", "2023-06-01");
            match send(req).await {
                Ok(data) => {
                    let models: Vec<String> = data["data"].as_array()
                        .map(|a| a.iter().filter_map(|m| m["id"].as_str().map(|s| s.to_string())).collect())
                        .unwrap_or_default();
                    Ok(serde_json::json!({ "ok": true, "models": models, "count": models.len() }))
                }
                Err(e) => Ok(serde_json::json!({ "ok": false, "error": e })),
            }
        }
        "groq" => {
            if settings.groq_api_key.is_empty() {
                return Ok(serde_json::json!({ "ok": false, "error": "Missing Groq API key" }));
            }
            let req = client.get("https://api.groq.com/openai/v1/models")
                .bearer_auth(&settings.groq_api_key);
            match send(req).await {
                Ok(data) => {
                    let models: Vec<String> = data["data"].as_array()
                        .map(|a| a.iter().filter_map(|m| m["id"].as_str().map(|s| s.to_string())).collect())
                        .unwrap_or_default();
                    Ok(serde_json::json!({ "ok": true, "models": models, "count": models.len() }))
                }
                Err(e) => Ok(serde_json::json!({ "ok": false, "error": e })),
            }
        }
        "gemini" => {
            if settings.gemini_api_key.is_empty() {
                return Ok(serde_json::json!({ "ok": false, "error": "Missing Gemini API key" }));
            }
            let url = format!(
                "https://generativelanguage.googleapis.com/v1beta/models?key={}",
                urlencoding::encode(&settings.gemini_api_key),
            );
            let req = client.get(&url);
            match send(req).await {
                Ok(data) => {
                    // Gemini returns `models[].name` like "models/gemini-1.5-flash"; strip prefix.
                    let models: Vec<String> = data["models"].as_array()
                        .map(|a| a.iter().filter_map(|m| {
                            m["name"].as_str().map(|s| s.trim_start_matches("models/").to_string())
                        }).collect())
                        .unwrap_or_default();
                    Ok(serde_json::json!({ "ok": true, "models": models, "count": models.len() }))
                }
                Err(e) => Ok(serde_json::json!({ "ok": false, "error": e })),
            }
        }
        // Local OpenAI-compatible servers (LM Studio, vLLM, llama.cpp server,
        // Jan, TabbyAPI, KoboldCPP, …). All share `/v1/models` + Bearer auth.
        // LM Studio gets a friendly default URL fallback; everyone else needs
        // the user to fill in `openai_compatible_url` explicitly.
        "openai_compatible" | "lmstudio" => {
            let raw_url = settings.openai_compatible_url.trim_end_matches('/');
            let url = if raw_url.is_empty() && settings.ai_provider == "lmstudio" {
                "http://localhost:1234/v1".to_string()
            } else if raw_url.is_empty() {
                return Ok(serde_json::json!({
                    "ok": false,
                    "error": "Set a Base URL (e.g. http://localhost:8080/v1)",
                }));
            } else {
                raw_url.to_string()
            };
            let mut req = client.get(format!("{url}/models"));
            if !settings.openai_compatible_key.is_empty() {
                req = req.bearer_auth(&settings.openai_compatible_key);
            }
            match send(req).await {
                Ok(data) => {
                    let models: Vec<String> = data["data"].as_array()
                        .map(|a| a.iter().filter_map(|m| m["id"].as_str().map(|s| s.to_string())).collect())
                        .unwrap_or_default();
                    Ok(serde_json::json!({ "ok": true, "models": models, "count": models.len() }))
                }
                Err(e) => Ok(serde_json::json!({ "ok": false, "error": e })),
            }
        }
        other => Ok(serde_json::json!({ "ok": false, "error": format!("unknown provider: {other}") })),
    }
}
