//! On-device language model — llama.cpp compiled into this binary.
//!
//! Replaces the managed Ollama daemon, which was the single worst resource
//! citizen in the app: it downloaded a server, spawned `ollama serve`, and ran a
//! watchdog that restarted it. On a 16 GB machine that produced a **6 GB
//! `llama-server` that `ollama ps` did not even list** — an orphan nothing would
//! ever unload, with the pagefile at 11 GB and 1.9 GB of RAM free.
//!
//! The architecture here is on-device assistant designs's: a small GGUF loaded IN-PROCESS. The
//! model's lifetime is ours, so it cannot orphan, and it is released the moment
//! it goes idle. Default model is LFM2.5-1.2B-Instruct (Q4_K_M, ~731 MB).
//!
//! It replaced the 350M, which is a genuinely good extractor — Liquid's own card
//! recommends it for "data extraction, structured outputs, and tool use" and warns
//! against "knowledge-intensive tasks" — but reads as slow-witted the moment it is
//! asked to hold a conversation. `agent::retrieve` now does the retrieval and the
//! arithmetic in Rust and leaves the model only the phrasing, so the extra
//! capacity goes exactly where it shows.
//!
//! **Threading.** `LlamaModel`/`LlamaContext` are not `Send`, so they cannot live
//! in a tokio `Mutex` across an await. Instead one dedicated OS thread owns the
//! backend + model and serves requests over a channel. That also gives, for free:
//! serialization (one generation at a time, which is what a single model wants)
//! and a natural place to drop the model after an idle timeout.
//!
//! Text-only by design. Vision stays with a vision provider — `agent::clip`
//! already falls back to its text path when images aren't usable.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};

use crate::provision::Requirement;

/// Unload the model after this long with no requests. The whole point of running
/// in-process is that idle costs nothing.
const IDLE_UNLOAD: Duration = Duration::from_secs(180);

/// Context window. KV-cache is the memory that grows with this, so it is sized to
/// the real workload rather than the model's maximum: Guardian's chat system prompt
/// is ~1.5k tokens of rules plus a live situation/memory/scene block, and 4096 was
/// measurably too tight for it. Overflow is still handled (see `generate_blocking`),
/// this just keeps the common case from being truncated at all.
const N_CTX: u32 = 8192;

/// The on-device context window, for callers deciding how much to put in a
/// prompt (the row budget) or how full it is (the composer's context ring).
pub(super) fn context_tokens() -> usize { N_CTX as usize }

/// Tokens of the context window kept free rather than used.
///
/// Filling the KV cache to its last slot left zero headroom, and llama.cpp answers a
/// full cache with an assert — which is an `abort()`, not an error (see the n_batch
/// crash). The last 64 tokens are never worth a crash in a 24/7 recorder.
const CTX_SLACK: usize = 64;

/// A GGUF this small isn't a model. Guards a truncated download.
const MIN_MODEL_BYTES: u64 = 50 * 1024 * 1024;

/// The installable on-device tiers, mirroring how `yolo_variant` picks a detector.
///
/// `(settings value, skill id, human name)`. The skill id is also the directory
/// under `skills/`, so each tier installs and uninstalls independently and none
/// can silently overwrite another.
pub(crate) const TIERS: &[(&str, &str, &str)] = &[
    ("fast",     "local_llm_fast",   "LFM2.5-350M"),
    ("balanced", "local_llm",        "LFM2.5-1.2B"),
    ("vision",   "local_llm_vision", "LFM2.5-VL-1.6B"),
];

/// Skill id for a tier, falling back to the balanced default for an unknown value.
///
/// `"balanced"` deliberately keeps the original `local_llm` id: it is where every
/// existing install's weights already are, so nobody re-downloads on upgrade.
pub(crate) fn tier_skill_id(tier: &str) -> &'static str {
    TIERS.iter().find(|(t, _, _)| *t == tier).map(|(_, id, _)| *id).unwrap_or("local_llm")
}

/// Human name of a tier, for prompts and status strings.
pub(crate) fn tier_label(tier: &str) -> &'static str {
    TIERS.iter().find(|(t, _, _)| *t == tier).map(|(_, _, n)| *n).unwrap_or("LFM2.5-1.2B")
}

/// Where a tier's weights live — the normal skill layout, so it installs and
/// uninstalls through the same UI as every other model.
pub(crate) fn model_path_for(data_dir: &Path, tier: &str) -> PathBuf {
    data_dir.join("skills").join(tier_skill_id(tier)).join("model.gguf")
}

/// The vision projector that sits beside a VL model's weights. Without it the
/// GGUF is a perfectly good text model that simply cannot see.
pub(crate) fn mmproj_path_for(data_dir: &Path, tier: &str) -> PathBuf {
    data_dir.join("skills").join(tier_skill_id(tier)).join("mmproj.gguf")
}

/// Is this tier installed and non-truncated?
pub(crate) fn is_installed_tier(data_dir: &Path, tier: &str) -> bool {
    Requirement::MinSize(MIN_MODEL_BYTES).met(&model_path_for(data_dir, tier))
}

/// Can the ACTIVE tier see an image? Both files must be present — a VL model
/// without its projector loads fine and then quietly describes nothing.
pub(crate) fn vision_ready(data_dir: &Path, tier: &str) -> bool {
    tier == "vision"
        && is_installed_tier(data_dir, tier)
        && Requirement::MinSize(MIN_MODEL_BYTES).met(&mmproj_path_for(data_dir, tier))
}

/// Is the selected on-device model ready to serve? Uses the path registered at
/// boot (and re-registered on a tier switch), so callers that only hold
/// `Settings` can ask without threading the data dir through.
pub(crate) fn model_ready() -> bool {
    current_path()
        .map(|p| Requirement::MinSize(MIN_MODEL_BYTES).met(&p))
        .unwrap_or(false)
}

// ─── Worker protocol ─────────────────────────────────────────────────────────

/// Called with each decoded piece as it is generated, on the worker thread.
///
/// The loop already produced these and threw them away until the reply was
/// complete, so streaming costs one channel send per token. Boxed rather than
/// generic because `Job` travels down an `mpsc` channel.
pub(crate) type OnToken = Box<dyn FnMut(&str) + Send>;

struct Job {
    system: String,
    user: String,
    max_tokens: usize,
    /// Raise the temperature for prose. See [`sampler_chain`].
    creative: bool,
    /// `None` = collect silently, the original behaviour.
    on_token: Option<OnToken>,
    /// Decoded JPEG frames for the Vision tier. Empty on the text path.
    images: Vec<Vec<u8>>,
    reply: std::sync::mpsc::Sender<Result<String>>,
}

/// Work items for the single llama.cpp thread. `Unload` exists because the worker
/// OWNS the model — nobody else can drop it — and because llama.cpp memory-maps the
/// GGUF: on Windows a mapped file cannot be deleted, so uninstalling the model while
/// it is loaded fails with a sharing violation until it is released.
enum Msg {
    Generate(Job),
    Unload,
}

fn sender() -> &'static std::sync::mpsc::Sender<Msg> {
    static TX: OnceLock<std::sync::mpsc::Sender<Msg>> = OnceLock::new();
    TX.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::channel::<Msg>();
        std::thread::Builder::new()
            .name("local-llm".into())
            .spawn(move || worker(rx))
            .expect("spawn local-llm thread");
        tx
    })
}

/// Drop the model now, releasing its memory and its mmap of the GGUF.
///
/// Must be called before deleting the model file (see `skills::remove_skill`).
/// Fire-and-forget: if the worker was never started there is nothing loaded anyway,
/// and if it is mid-generation the unload is handled as soon as that finishes.
pub(crate) fn unload() {
    let _ = sender().send(Msg::Unload);
}

/// Model-load parameters, including how much to put on the GPU.
///
/// `n_gpu_layers` is inert unless the binary was built with a GPU backend
/// (`--features vulkan`; Apple Silicon gets Metal automatically). When there is
/// no usable device llama.cpp keeps the layers on the CPU, so ONE build runs
/// everywhere — a GPU is an optimisation here, never a requirement.
///
/// `SC_LLM_GPU_LAYERS` dials it back. The discrete GPU is shared with the ONNX
/// detectors, and on a small card a fully-offloaded language model can crowd out
/// the thing actually watching the cameras — which is the wrong trade.
fn model_params() -> llama_cpp_2::model::params::LlamaModelParams {
    use llama_cpp_2::model::params::LlamaModelParams;
    let layers = std::env::var("SC_LLM_GPU_LAYERS").ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(u32::MAX); // "all of them", clamped by llama.cpp to what exists
    LlamaModelParams::default().with_n_gpu_layers(layers)
}

/// The one thread that ever touches llama.cpp.
fn worker(rx: std::sync::mpsc::Receiver<Msg>) {
    use llama_cpp_2::llama_backend::LlamaBackend;
    use llama_cpp_2::model::LlamaModel;

    let backend = match LlamaBackend::init() {
        Ok(b) => b,
        Err(e) => {
            tracing::error!("local LLM: backend init failed: {e}");
            // Drain forever so callers get a clean error instead of hanging.
            while let Ok(msg) = rx.recv() {
                if let Msg::Generate(job) = msg {
                    let _ = job.reply.send(Err(anyhow!("local LLM backend unavailable")));
                }
            }
            return;
        }
    };

    let mut loaded: Option<(LlamaModel, Instant)> = None;

    loop {
        // Wait for work, but wake up to drop an idle model.
        let job = match rx.recv_timeout(IDLE_UNLOAD) {
            Ok(Msg::Generate(j)) => j,
            Ok(Msg::Unload) => {
                if loaded.take().is_some() {
                    // Dropped here: frees the weights AND releases the mmap, which is
                    // what lets the file be deleted on Windows.
                    tracing::info!("local LLM: unloaded on request");
                }
                continue;
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                if let Some((_, last)) = &loaded {
                    if last.elapsed() >= IDLE_UNLOAD {
                        loaded = None; // model dropped here — memory returned to the OS
                        tracing::info!("local LLM: unloaded after {}s idle", IDLE_UNLOAD.as_secs());
                    }
                }
                continue;
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => return,
        };

        // Load on demand.
        if loaded.is_none() {
            let path = match current_path() {
                Some(p) => p,
                None => {
                    let _ = job.reply.send(Err(anyhow!("local LLM: model path not set")));
                    continue;
                }
            };
            let t0 = Instant::now();
            match LlamaModel::load_from_file(&backend, &path, &model_params()) {
                Ok(m) => {
                    tracing::info!("local LLM: loaded {} in {:?}", path.display(), t0.elapsed());
                    loaded = Some((m, Instant::now()));
                }
                Err(e) => {
                    let _ = job.reply.send(Err(anyhow!("local LLM: failed to load model: {e}")));
                    continue;
                }
            }
        }

        let (model, last_used) = loaded.as_mut().expect("model loaded above");
        // Build the prompt with the model's OWN chat template. Hardcoding ChatML
        // made LFM2.5 emit log-like fragments ("DETECTED_001") instead of
        // answering — a wrong template turns instruction-following into raw text
        // completion. Reading the template from the GGUF also means swapping in a
        // different model doesn't silently break the prompt format.
        let mut job = job;
        let on_token = job.on_token.take();
        let images = std::mem::take(&mut job.images);
        let out = if images.is_empty() {
            match build_prompt(model, &job.system, &job.user) {
                Ok((prompt, add_bos)) => {
                    generate_blocking(&backend, model, &prompt, add_bos, job.max_tokens,
                                      job.creative, on_token)
                }
                Err(e) => Err(e),
            }
        } else {
            generate_vision(&backend, model, &job.system, &job.user, &images,
                            job.max_tokens, job.creative, on_token)
        };
        *last_used = Instant::now();
        let _ = job.reply.send(out);
    }
}

/// Render `system` + `user` through the template baked into the GGUF, falling
/// back to ChatML only if the model carries none.
///
/// Returns the prompt AND whether the tokenizer should prepend BOS. Most chat
/// templates emit the model's start token themselves, so tokenizing the result
/// with `AddBos::Always` gave the model TWO — which is off-distribution for every
/// instruct model and degrades the first tokens of the reply. The answer is
/// probed rather than assumed, because a swapped-in GGUF may template differently.
fn build_prompt(
    model: &llama_cpp_2::model::LlamaModel,
    system: &str,
    user: &str,
) -> Result<(String, llama_cpp_2::model::AddBos)> {
    use llama_cpp_2::model::{AddBos, LlamaChatMessage};

    let msgs = vec![
        LlamaChatMessage::new("system".to_string(), system.trim().to_string())
            .map_err(|e| anyhow!("local LLM: system message: {e}"))?,
        LlamaChatMessage::new("user".to_string(), user.trim().to_string())
            .map_err(|e| anyhow!("local LLM: user message: {e}"))?,
    ];

    match model.chat_template(None) {
        Ok(tmpl) => {
            let rendered = model
                .apply_chat_template(&tmpl, &msgs, true)
                .map_err(|e| anyhow!("local LLM: apply chat template: {e}"))?;
            #[cfg(test)]
            eprintln!("[local_llm] using the model's own template; prompt =\n{rendered}\n---");
            // `true` renders the token as its literal text ("<|startoftext|>")
            // rather than as nothing, which is what makes this comparison work.
            let mut dec = encoding_rs::UTF_8.new_decoder();
            let bos = model
                .token_to_piece(model.token_bos(), &mut dec, true, None)
                .unwrap_or_default();
            let add = if !bos.is_empty() && rendered.starts_with(&bos) {
                AddBos::Never
            } else {
                AddBos::Always
            };
            Ok((rendered, add))
        }
        Err(e) => {
            #[cfg(test)]
            eprintln!("[local_llm] NO embedded template ({e}) — falling back to ChatML");
            tracing::warn!("local LLM: model has no chat template ({e}) — falling back to ChatML");
            // This literal carries no BOS of its own, so the tokenizer adds it.
            Ok((format!(
                "<|im_start|>system\n{}<|im_end|>\n\
                 <|im_start|>user\n{}<|im_end|>\n\
                 <|im_start|>assistant\n",
                system.trim(), user.trim()
            ), AddBos::Always))
        }
    }
}

/// Text that means "the turn is over" even though no end-of-generation token
/// arrived. A small model that has run out of things to say will happily invent
/// the next turn of the conversation and answer itself — and because the caller
/// frames history as `[USER]: …`, that is exactly the shape it imitates.
const STOPS: &[&str] = &[
    "\n[USER]", "[USER]:", "\nUser:", "\n[ASSISTANT]", "[ASSISTANT]:",
    "<|im_start|>", "<|im_end|>", "</s>",
];

/// Byte offset where generation should be cut, if any stop marker has appeared.
///
/// Scans the whole buffer each token deliberately: it is a couple of KB at most,
/// and this runs between forward passes that each cost milliseconds — the bounded
/// tail-window version would be more code for time that cannot be measured.
fn stop_at(out: &str) -> Option<usize> {
    STOPS.iter().filter_map(|s| out.find(s)).min()
}

/// One completion, start to finish, on the worker thread.
/// Divide the fixed context window between prompt and generation.
///
/// Returns `(prompt_tokens_to_keep, tokens_to_generate)`, and the two ALWAYS sum
/// to at most `n_ctx` — overrunning it makes llama.cpp fail the decode mid-reply,
/// which loses the whole answer. A large `max_tokens` can never take more than
/// half the window, so a long prompt still gets room to say something.
fn split_context(n_ctx: usize, prompt_len: usize, max_tokens: usize) -> (usize, usize) {
    let usable = n_ctx.saturating_sub(CTX_SLACK);
    let prompt_budget = usable.saturating_sub(max_tokens).max(usable / 2);
    let keep = prompt_len.min(prompt_budget);
    (keep, (usable - keep).min(max_tokens))
}

/// How many CPU threads to give llama.cpp.
///
/// It defaults to **4** regardless of the machine — on a 20-core box that left
/// most of the CPU idle while the user waited. Decoding a small quantised model
/// is memory-bandwidth bound, so scaling flattens out well before core count and
/// piling on threads starts to cost; physical cores capped at 8 is the usual
/// sweet spot, and `available_parallelism` counts logical ones, hence the halving.
///
/// `SC_LLM_THREADS` overrides it. Real machines differ (hybrid P/E cores, VMs,
/// shared hosts) in ways this heuristic cannot see, so the knob stays.
fn n_threads() -> i32 {
    if let Ok(n) = std::env::var("SC_LLM_THREADS").unwrap_or_default().parse::<i32>() {
        if n > 0 { return n; }
    }
    let logical = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    ((logical / 2).clamp(4, 8)) as i32
}

/// The sampler chain, in one place for both the text and vision paths.
///
/// Two jobs, two settings. Liquid AI's published values for LFM2.5 — temperature
/// 0.1, top_k 50, repetition_penalty 1.05 — are tuned for EXTRACTION, and that is
/// what `creative == false` keeps: classification, risk scoring, routing, JSON.
/// A fixed seed there is a feature, because a diagnosis you cannot reproduce is
/// not a diagnosis.
///
/// But the same chain also drove the phrasing pass, whose entire job is to write
/// a natural sentence about rows Rust has already computed. Temperature 0.1 with
/// a CONSTANT seed makes that byte-identical for the same rows every time, which
/// is a large part of why the agent reads like a machine — the same day's events
/// always came back in the same words. `creative == true` raises the temperature,
/// adds top_p, and seeds from the clock.
///
/// The repetition penalty stays on in BOTH modes: the signature failure of a
/// small model is a degenerate repeat loop, and 1.05 is a light touch that costs
/// nothing.
///
/// Order matters: llama.cpp applies the chain in sequence, so penalties and
/// truncation run before temperature, and `dist` makes the final pick.
fn sampler_chain(creative: bool) -> llama_cpp_2::sampling::LlamaSampler {
    use llama_cpp_2::sampling::LlamaSampler;
    if creative {
        LlamaSampler::chain_simple([
            LlamaSampler::penalties(64, 1.05, 0.0, 0.0), // last_n, repeat, freq, presence
            LlamaSampler::top_k(50),
            LlamaSampler::top_p(0.9, 1),
            LlamaSampler::temp(0.6),
            LlamaSampler::dist(rand_seed()),
        ])
    } else {
        LlamaSampler::chain_simple([
            LlamaSampler::penalties(64, 1.05, 0.0, 0.0),
            LlamaSampler::top_k(50),
            LlamaSampler::temp(0.1),
            LlamaSampler::dist(0x5EC0_0CAF),
        ])
    }
}

/// A seed that changes per generation, so the phrasing pass does not repeat
/// itself verbatim. Wall-clock is enough — nothing here is security-relevant.
fn rand_seed() -> u32 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos() ^ (d.as_secs() as u32))
        .unwrap_or(0x5EC0_0CAF)
}

fn generate_blocking(
    backend: &llama_cpp_2::llama_backend::LlamaBackend,
    model: &llama_cpp_2::model::LlamaModel,
    prompt: &str,
    add_bos: llama_cpp_2::model::AddBos,
    max_tokens: usize,
    creative: bool,
    mut on_token: Option<OnToken>,
) -> Result<String> {
    use llama_cpp_2::context::params::LlamaContextParams;
    use llama_cpp_2::llama_batch::LlamaBatch;
    use std::num::NonZeroU32;

    let mut ctx = model
        .new_context(backend, LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(N_CTX))
            .with_n_threads(n_threads())
            .with_n_threads_batch(n_threads()))
        .map_err(|e| anyhow!("local LLM: context: {e}"))?;

    let mut tokens = model
        .str_to_token(prompt, add_bos)
        .map_err(|e| anyhow!("local LLM: tokenize: {e}"))?;

    // LAST-RESORT GUARD, not routine behaviour.
    //
    // The caller now assembles prompts against a budget (`chat::MEMORY_BUDGET_CHARS`
    // / `HISTORY_BUDGET_CHARS`, with memory RETRIEVED rather than dumped), so a
    // Guardian prompt should never reach this. Reaching it means the budget maths
    // is wrong — hence `error!`, not `warn!`: it is a bug signal to act on, not a
    // routine notice to scroll past.
    //
    // When it does fire, drop from the MIDDLE: that keeps the identity and rules
    // at the head AND the user's actual question at the tail, which is the least
    // destructive cut available at this point.
    let (keep, gen) = split_context(N_CTX as usize, tokens.len(), max_tokens);
    if tokens.len() > keep {
        let head = keep / 2;
        let cut  = tokens.len() - (keep - head);
        tracing::error!(
            "local LLM: prompt {} tokens > {keep} budget — context budgeting FAILED, \
             dropping {} tokens from the middle (answer quality will suffer)",
            tokens.len(), cut - head,
        );
        tokens.drain(head..cut);
    }

    // Prefill in n_batch-sized chunks.
    //
    // `n_ctx` is how many tokens the KV cache HOLDS; `n_batch` is how many may be
    // decoded in ONE call, and it defaults to 512 regardless of n_ctx. Submitting the
    // whole prompt at once trips
    //     GGML_ASSERT(n_tokens_all <= cparams.n_batch)
    // which is an abort() — the process dies instantly, taking recording with it, and
    // no Rust error handling can intercept it. It crashed the app on the first real
    // chat while a two-sentence smoke test passed, because the bug only bites above
    // 512 tokens and Guardian's system prompt is ~1.5k.
    //
    // Chunking here rather than raising n_batch on purpose: the compute buffer scales
    // with n_batch (132 MB at 512), so an 8192 batch would cost GBs for no gain.
    let n_batch = ctx.n_batch() as usize;
    let mut batch = LlamaBatch::new(n_batch, 1);
    let last = tokens.len() - 1;
    for (chunk_idx, chunk) in tokens.chunks(n_batch).enumerate() {
        batch.clear();
        let base = chunk_idx * n_batch;
        for (j, tok) in chunk.iter().enumerate() {
            let pos = base + j;
            // Only the final token of the final chunk needs logits — that is the
            // one we sample the first reply token from.
            batch.add(*tok, pos as i32, &[0], pos == last)
                .map_err(|e| anyhow!("local LLM: batch: {e}"))?;
        }
        ctx.decode(&mut batch).map_err(|e| anyhow!("local LLM: prefill: {e}"))?;
    }

    let mut sampler = sampler_chain(creative);
    // Generation starts at the position after the WHOLE prompt — not
    // `batch.n_tokens()`, which after chunked prefill holds only the final chunk
    // and would rewind over the prompt's own cache entries. The loop below counts
    // from there.
    let mut out = String::new();
    // Streaming UTF-8 decoder: a token can end mid-codepoint, so bytes must be
    // decoded incrementally rather than per-token.
    let mut decoder = encoding_rs::UTF_8.new_decoder();

    // `n` is the KV-cache position for each generated token, continuing straight on
    // from the prompt — so the loop counts positions rather than iterations.
    for n in (tokens.len() as i32..).take(gen) {
        let tok = sampler.sample(&ctx, -1);
        if model.is_eog_token(tok) { break; }
        let piece = model.token_to_piece(tok, &mut decoder, false, None).unwrap_or_default();
        out.push_str(&piece);
        // A stop marker means the model has started a turn that isn't its own.
        // Everything from there on is imitation, so it is cut, not delivered.
        //
        // Checked BEFORE emitting, so a streaming caller never sees a fragment
        // that the final answer won't contain.
        if let Some(cut) = stop_at(&out) {
            out.truncate(cut);
            break;
        }
        if let Some(cb) = on_token.as_mut() {
            if !piece.is_empty() { cb(&piece); }
        }
        batch.clear();
        batch.add(tok, n, &[0], true).map_err(|e| anyhow!("local LLM: batch: {e}"))?;
        ctx.decode(&mut batch).map_err(|e| anyhow!("local LLM: decode: {e}"))?;
    }
    Ok(out)
}

/// One completion that can SEE — the Vision tier's path.
///
/// llama.cpp's multimodal helper does the hard part: `tokenize` splices the image
/// into the prompt at a marker, and `eval_chunks` runs the vision encoder and the
/// text prefill together, returning the position to generate from. Everything
/// after that is the ordinary sampling loop.
///
/// The projector (`mmproj.gguf`) is a separate file from the weights, and without
/// it the model loads perfectly and describes nothing — so `vision_ready` checks
/// for both before anything gets here.
fn generate_vision(
    backend: &llama_cpp_2::llama_backend::LlamaBackend,
    model: &llama_cpp_2::model::LlamaModel,
    system: &str,
    user: &str,
    images: &[Vec<u8>],
    max_tokens: usize,
    creative: bool,
    mut on_token: Option<OnToken>,
) -> Result<String> {
    use llama_cpp_2::context::params::LlamaContextParams;
    use llama_cpp_2::llama_batch::LlamaBatch;
    use llama_cpp_2::mtmd::{MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputText};
    use std::num::NonZeroU32;

    let mmproj = current_mmproj()
        .ok_or_else(|| anyhow!("local LLM: vision requested but no projector is installed"))?;

    let params = MtmdContextParams {
        use_gpu: true, // inert on a CPU-only build; free where a backend exists
        print_timings: false,
        n_threads: n_threads(),
        ..Default::default()
    };
    let mtmd = MtmdContext::init_from_file(&mmproj.to_string_lossy(), model, &params)
        .map_err(|e| anyhow!("local LLM: projector: {e}"))?;
    if !mtmd.support_vision() {
        return Err(anyhow!("local LLM: this projector does not do vision"));
    }

    let mut ctx = model
        .new_context(backend, LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(N_CTX))
            .with_n_threads(n_threads())
            .with_n_threads_batch(n_threads()))
        .map_err(|e| anyhow!("local LLM: context: {e}"))?;

    // The marker is where the image is spliced in. Bounded to two frames: each
    // one costs hundreds of visual tokens out of the same window the question and
    // the answer have to share.
    let marker = llama_cpp_2::mtmd::mtmd_default_marker();
    let mut bitmaps = Vec::new();
    let mut markers = String::new();
    for jpeg in images.iter().take(2) {
        match MtmdBitmap::from_buffer(&mtmd, jpeg, false) {
            Ok(b) => { bitmaps.push(b); markers.push_str(marker); markers.push('\n'); }
            Err(e) => tracing::warn!("local LLM: skipping an unreadable frame: {e}"),
        }
    }
    if bitmaps.is_empty() {
        return Err(anyhow!("local LLM: no usable frames"));
    }

    let text = MtmdInputText {
        text: format!("{}\n\n{markers}{}", system.trim(), user.trim()),
        add_special: true,
        parse_special: true,
    };
    let refs: Vec<&MtmdBitmap> = bitmaps.iter().collect();
    let chunks = mtmd.tokenize(text, &refs)
        .map_err(|e| anyhow!("local LLM: multimodal tokenize: {e}"))?;

    // Runs the vision encoder AND the text prefill, and hands back the position
    // to start generating from.
    let n_batch = ctx.n_batch() as i32;
    let start_pos = chunks.eval_chunks(&mtmd, &ctx, 0, 0, n_batch, true)
        .map_err(|e| anyhow!("local LLM: multimodal prefill: {e}"))?;

    // BOUND THE WINDOW. The text path divides it with `split_context`; this path
    // never did. A large context block plus two images can fill the KV cache to its
    // last slot, and llama.cpp answers a full cache with
    // `GGML_ASSERT(n_tokens_all <= cparams.n_batch)` — an abort(), not an error, so
    // no `?` and no catch_unwind can save it. With `panic = "abort"` in the release
    // profile that takes the recorder down with the reply. Generate only into the
    // room the prefill actually left.
    let usable = (N_CTX as usize).saturating_sub(CTX_SLACK);
    let gen = usable.saturating_sub(start_pos.max(0) as usize).min(max_tokens);
    if gen == 0 {
        return Err(anyhow!(
            "local LLM: the prompt and images fill the context window — no room left to \
             answer. Use fewer frames or a shorter prompt."));
    }
    if gen < max_tokens {
        tracing::warn!("local LLM: vision prefill used {start_pos} of {usable} tokens — \
                        capping the reply at {gen} instead of {max_tokens}");
    }

    // Same sampler and stop rules as the text path — a vision model invents the
    // user's next turn just as readily.
    let mut sampler = sampler_chain(creative);
    let mut out = String::new();
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut batch = LlamaBatch::new(n_batch as usize, 1);

    // `n_past` continues straight on from where the prefill left off — the loop
    // counts KV-cache positions, not iterations.
    for n_past in (start_pos..).take(gen) {
        let tok = sampler.sample(&ctx, -1);
        if model.is_eog_token(tok) { break; }
        let piece = model.token_to_piece(tok, &mut decoder, false, None).unwrap_or_default();
        out.push_str(&piece);
        if let Some(cut) = stop_at(&out) { out.truncate(cut); break; }
        if let Some(cb) = on_token.as_mut() {
            if !piece.is_empty() { cb(&piece); }
        }
        batch.clear();
        batch.add(tok, n_past, &[0], true).map_err(|e| anyhow!("local LLM: batch: {e}"))?;
        ctx.decode(&mut batch).map_err(|e| anyhow!("local LLM: decode: {e}"))?;
    }
    Ok(out)
}

/// Which weights the worker should load, set at boot and again on a tier switch.
///
/// An `RwLock`, not a `OnceLock`: switching tiers has to change it, and
/// `OnceLock::set` silently fails the second time — so with the old type the user
/// would pick a different model, be told it worked, and keep talking to the old one.
static MODEL_PATH: std::sync::RwLock<Option<PathBuf>> = std::sync::RwLock::new(None);

fn current_path() -> Option<PathBuf> {
    MODEL_PATH.read().ok().and_then(|p| p.clone())
}

/// The projector beside the current weights, if one is installed.
fn current_mmproj() -> Option<PathBuf> {
    let p = current_path()?.parent()?.join("mmproj.gguf");
    p.exists().then_some(p)
}

/// Point the worker at a tier's weights.
///
/// **Unloads first.** llama.cpp memory-maps the GGUF, and on Windows a mapped
/// file cannot be replaced or deleted — so switching without releasing the old
/// model leaves it resident, serving answers from the model the user just
/// switched away from.
pub(crate) fn set_tier(data_dir: &Path, tier: &str) {
    let next = model_path_for(data_dir, tier);
    let changed = current_path().as_ref() != Some(&next);
    if let Ok(mut p) = MODEL_PATH.write() { *p = Some(next); }
    // Cached because `call_llm` decides whether to send images and only holds
    // `Settings` — it has no data dir to check the projector with.
    VISION_AVAILABLE.store(vision_ready(data_dir, tier), std::sync::atomic::Ordering::Relaxed);
    if changed {
        unload();
        tracing::info!("local LLM: tier → {tier} ({})", tier_label(tier));
    }
}

/// Is the on-device engine able to look at a frame right now?
static VISION_AVAILABLE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

pub(crate) fn vision_available() -> bool {
    VISION_AVAILABLE.load(std::sync::atomic::Ordering::Relaxed)
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// Run a chat completion on-device. Returns the assistant's text.
///
/// Blocking work happens on the dedicated worker thread; this only waits for the
/// reply, so the async runtime is never blocked.
pub(crate) async fn chat(
    system: &str, user: &str, max_tokens: usize, creative: bool,
) -> Result<String> {
    chat_streaming(system, user, max_tokens, creative, None).await
}

/// As [`chat`], but calls `on_token` with each piece as it is produced.
///
/// The callback runs on the llama.cpp worker thread, so it must be cheap and must
/// never block — emitting a Tauri event is fine, awaiting anything is not.
pub(crate) async fn chat_streaming(
    system: &str,
    user: &str,
    max_tokens: usize,
    creative: bool,
    on_token: Option<OnToken>,
) -> Result<String> {
    chat_full(system, user, max_tokens, creative, on_token, Vec::new()).await
}

/// A completion that can look at frames. Requires the Vision tier — check
/// [`vision_ready`] first; a text tier given images will error rather than
/// quietly describe a scene it never saw.
pub(crate) async fn chat_vision(
    system: &str,
    user: &str,
    max_tokens: usize,
    creative: bool,
    images: Vec<Vec<u8>>,
) -> Result<String> {
    chat_full(system, user, max_tokens, creative, None, images).await
}

async fn chat_full(
    system: &str,
    user: &str,
    max_tokens: usize,
    creative: bool,
    on_token: Option<OnToken>,
    images: Vec<Vec<u8>>,
) -> Result<String> {
    // The prompt is rendered on the worker thread, which owns the model and so
    // can use the chat template baked into the GGUF (see `build_prompt`).
    let (reply_tx, reply_rx) = std::sync::mpsc::channel();
    sender()
        .send(Msg::Generate(Job {
            system: system.to_string(),
            user: user.to_string(),
            max_tokens,
            creative,
            on_token,
            images,
            reply: reply_tx,
        }))
        .map_err(|_| anyhow!("local LLM worker is gone"))?;

    // The worker is single-threaded and a generation can take seconds; wait for it
    // on the blocking pool rather than stalling an async worker.
    tokio::task::spawn_blocking(move || {
        reply_rx.recv().unwrap_or_else(|_| Err(anyhow!("local LLM produced no reply")))
    })
    .await
    .map_err(|e| anyhow!("local LLM join: {e}"))?
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Point the worker at a specific GGUF, bypassing the tier lookup. Only the
    /// `#[ignore]`d live-model tests need this — everything else runs modelless.
    fn set_model_path_for_test(p: PathBuf) {
        if let Ok(mut g) = MODEL_PATH.write() { *g = Some(p); }
    }

    /// The model answering itself is the failure this catches: it writes a reply,
    /// then invents the user's next question and answers that too. Everything from
    /// the marker on must be cut, and ordinary prose must be left alone.
    #[test]
    fn stop_at_cuts_an_invented_turn() {
        assert_eq!(stop_at("A calm answer about the driveway."), None);
        // Brackets that aren't role markers must survive.
        assert_eq!(stop_at("I saw a person [note: low confidence] at 14:02."), None);

        let s = "Nothing unusual today.\n[USER]: and yesterday?";
        let cut = stop_at(s).expect("marker found");
        assert_eq!(&s[..cut], "Nothing unusual today.");

        // Template leakage counts too.
        let t = "Two events.<|im_end|>\n<|im_start|>user";
        assert_eq!(&t[..stop_at(t).unwrap()], "Two events.");
    }

    /// The one invariant that matters: prompt + generation never exceed the
    /// context window (llama.cpp fails the decode mid-reply if they do), and
    /// generation is never zeroed out by a prompt that wants the whole window.
    #[test]
    fn context_split_never_overruns_the_window() {
        let n_ctx = 8192;
        for &prompt in &[1usize, 100, 4095, 8191, 8192, 40_000] {
            for &want in &[1usize, 256, 512, 4096, 100_000] {
                let (keep, gen) = split_context(n_ctx, prompt, want);
                assert!(keep + gen <= n_ctx, "overrun: {prompt}/{want} -> {keep}+{gen}");
                assert!(gen > 0, "no room to answer: {prompt}/{want}");
                assert!(keep <= prompt, "invented prompt tokens: {prompt} -> {keep}");
                // A prompt that fits in the USABLE window (n_ctx minus the reserved
                // slack) must not be truncated at all.
                if prompt + want <= n_ctx - CTX_SLACK { assert_eq!(keep, prompt); }
                // And the slack must actually be left free.
                assert!(keep + gen <= n_ctx - CTX_SLACK, "slack consumed: {keep}+{gen}");
            }
        }
    }

    /// Load the real GGUF and generate — the proof that the in-process engine
    /// works end to end (backend init, model load, tokenize, prefill, decode,
    /// UTF-8 assembly). Prints throughput so the CPU cost is a measured number
    /// rather than an assumption.
    ///
    /// `#[ignore]`d because it needs the ~230 MB model on disk. Run with:
    ///   SC_LLM_MODEL=<...>/model.gguf cargo test --lib local_llm -- --ignored --nocapture
    /// An over-long prompt must not take the process down.
    ///
    /// Regression for a live crash (WER: `ucrtbase.dll`, `0xc0000409` = native
    /// `abort()`) that killed the app seconds after the model loaded. `split_context`
    /// handed back `keep + gen == n_ctx` EXACTLY when the prompt overflowed, leaving
    /// the KV cache zero slack — llama.cpp asserts rather than returning an error, and
    /// a GGML_ASSERT is an abort, which no amount of Rust error handling can catch.
    /// A crash in a 24/7 recorder is the worst possible failure: recording stops.
    #[tokio::test]
    #[ignore = "needs SC_LLM_MODEL=<path to model.gguf>"]
    async fn oversized_prompt_does_not_abort() {
        let Ok(path) = std::env::var("SC_LLM_MODEL") else {
            eprintln!("SC_LLM_MODEL unset — skipping");
            return;
        };
        set_model_path_for_test(PathBuf::from(&path));

        // Comfortably past the prompt budget so the truncation path runs and the
        // prompt/generation split lands on its boundary. 512 is what `call_llm` uses.
        let huge = "The camera observed a person near the door. ".repeat(3000);
        let out = chat("You summarise security events in one sentence.", &huge, 512, false).await;
        // Either answer or return an Err — but the process must still be alive.
        eprintln!("oversized prompt -> {:?}", out.as_deref().map(|s| &s[..s.len().min(120)]));
        assert!(out.is_ok() || out.is_err(), "unreachable; the point is surviving to here");
    }

    #[tokio::test]
    #[ignore = "needs SC_LLM_MODEL=<path to model.gguf>"]
    async fn generates_text_on_device() {
        let Ok(path) = std::env::var("SC_LLM_MODEL") else {
            eprintln!("SC_LLM_MODEL unset — skipping");
            return;
        };
        let p = PathBuf::from(&path);
        assert!(Requirement::MinSize(MIN_MODEL_BYTES).met(&p), "{path} is not a usable model");
        set_model_path_for_test(p);

        // Two prompts on purpose. A leading, jargon-heavy instruction can make a
        // small model echo the pattern instead of answering, so a neutral
        // question is the control that tells "engine broken" apart from
        // "prompt too leading" / "model too weak".
        // NOTE ON PROMPTING SMALL MODELS: an earlier version of case 2 said
        // "You write one-line security-camera log entries. Be terse and factual."
        // and the model answered `DETECTED_001` — literally a one-line log entry.
        // A 350M model obeys the letter of an instruction, so ask for a SENTENCE,
        // not for a genre. Same lesson applies to `agent/prompts.rs`.
        let cases: [(&str, &str); 2] = [
            ("You are a helpful assistant. Answer in one short sentence.",
             "What is a security camera used for?"),
            ("You summarise home-security events for a homeowner. \
              Reply with one plain-English sentence describing what happened.",
             "Event: person detected at the front door at 14:32; they stayed 45 seconds."),
        ];

        let rss = || -> u64 {
            use sysinfo::{ProcessRefreshKind, RefreshKind, System};
            // `ProcessRefreshKind::new()` sets every field to false — memory()
            // then reads 0. `everything()` is what actually populates RSS.
            let s = System::new_with_specifics(
                RefreshKind::new().with_processes(ProcessRefreshKind::everything()));
            s.process(sysinfo::get_current_pid().unwrap())
                .map(|p| p.memory() / 1_048_576).unwrap_or(0)
        };
        let before = rss();

        let mut first = String::new();
        for (i, (sys, usr)) in cases.iter().enumerate() {
            let t0 = Instant::now();
            let out = chat(sys, usr, 64, false).await.expect("generation should succeed");
            eprintln!("--- case {} ({:.1}s) ---\nQ: {}\nA: {}\n",
                i + 1, t0.elapsed().as_secs_f64(), usr, out.trim());
            if i == 0 { first = out.clone(); }
        }
        eprintln!("--- process RSS: {} MB before, {} MB with the model loaded ---", before, rss());

        let out = first;
        assert!(!out.trim().is_empty(), "model returned nothing");
        // A coherent answer to a plain question is several words, not a token or
        // two — this is what caught the bad prompt in the first place.
        assert!(out.split_whitespace().count() >= 4, "answer looks degenerate: {out:?}");
        // Greedy decoding of a coherent model shouldn't emit replacement chars.
        assert!(!out.contains('\u{FFFD}'), "UTF-8 assembly is broken: {out:?}");
    }
}
