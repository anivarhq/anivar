//! CLIP-family semantic event search — image + text encoder (NVR-parity).
//!
//! Mature NVRs' "search" indexes each tracked event by embedding the event
//! thumbnail (image) and its description (text) with a CLIP model into one
//! shared vector space, then does cosine search. We do exactly the same: on
//! event close [`crate::agent::clip::embed_event`] embeds the thumbnail + the
//! `ai_summary`, and `nvr_recording::search_events` embeds the query text and
//! cosine-matches it against both — a text query can hit either the image or
//! the text vector because they live in the same space.
//!
//! **Resource-tiered models.** The active model is `settings.search_model`; each
//! is a separate two-tower skill installed under `<data_dir>/skills/<id>/` as
//! `vision_model.onnx` + `text_model.onnx` + `tokenizer.json`:
//!   * `mobileclip_s0` — Apple MobileCLIP-S0, 256², /255 (no mean-std), 512-d. Small/fast default.
//!   * `clip_b32`      — OpenAI CLIP ViT-B/32, 224², CLIP mean/std, 512-d.
//!   * `jina_clip`     — Jina-CLIP-v1, 224², CLIP mean/std, 768-d.
//! All from the transformers.js ONNX exports (`Xenova/*`, `jinaai/jina-clip-v1`).
//! Different models live in different vector spaces, so stored embeddings are
//! tagged with their model id and only ever cosine-matched within the same model.
//!
//! Lazy-loaded into a process-wide cache keyed by the active model id (switching
//! reloads). When the active skill isn't installed every call returns `None`/`Err`
//! and the caller falls back to keyword search, so nothing breaks.
//!
//! Defensive on purpose: input names (`input_ids`/`attention_mask`/
//! `token_type_ids`), the projected-output name, and 2-D vs 3-D (needs mean
//! pooling) output are all resolved from the loaded session rather than
//! hard-coded, because exports vary.

use std::path::Path;
use std::sync::{Mutex, OnceLock};

use ort::session::Session as OrtSession;
use ort::value::Tensor;
use tokenizers::Tokenizer;

use crate::inference::build_ort_session_cpu;

// OpenAI-CLIP normalisation constants (CLIP ViT-B/32 + Jina-CLIP v1 inherit these).
const CLIP_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const CLIP_STD:  [f32; 3] = [0.26862954, 0.261_302_6, 0.275_777_1];

/// Per-model preprocessing spec. The two-tower ONNX run is identical across
/// models — only image size / normalisation / token cap differ.
#[derive(Clone, Copy)]
struct ModelSpec {
    image_size: usize,
    normalize:  bool,        // false = rescale /255 only (MobileCLIP); true = subtract CLIP mean/std
    mean:       [f32; 3],
    std:        [f32; 3],
    max_tokens: usize,
    /// Fixed-context text tower (original CLIP / MobileCLIP): the positional
    /// embedding is a hard `max_tokens` length, so input_ids MUST be padded to it
    /// or the positional `Add` fails to broadcast. Dynamic towers (Jina/RoBERTa)
    /// are variable-length and use the attention mask instead → false.
    fixed_context: bool,
}

/// Returns the spec for a known model id, or `None` for "off"/unknown.
fn model_spec(id: &str) -> Option<ModelSpec> {
    match id {
        "mobileclip_s0" => Some(ModelSpec { image_size: 256, normalize: false, mean: [0.0; 3], std: [1.0; 3], max_tokens: 77, fixed_context: true }),
        "clip_b32"      => Some(ModelSpec { image_size: 224, normalize: true,  mean: CLIP_MEAN, std: CLIP_STD, max_tokens: 77, fixed_context: true }),
        "jina_clip"     => Some(ModelSpec { image_size: 224, normalize: true,  mean: CLIP_MEAN, std: CLIP_STD, max_tokens: 512, fixed_context: false }),
        _ => None,
    }
}

/// Process-wide model cache, keyed by the active model id so a switch reloads.
static MODEL: OnceLock<Mutex<Option<(String, ClipEncoder)>>> = OnceLock::new();

pub struct ClipEncoder {
    spec:   ModelSpec,
    vision: OrtSession,
    text:   OrtSession,
    tokenizer: Tokenizer,
    vision_input: String,
    vision_out:   usize,
    // Text input wiring resolved from the export.
    text_ids_name:  String,
    text_attn_name: Option<String>,
    text_type_name: Option<String>,
    text_out:       usize,
}

impl ClipEncoder {
    pub fn try_load(data_dir: &Path, model_id: &str) -> anyhow::Result<Self> {
        let spec = model_spec(model_id)
            .ok_or_else(|| anyhow::anyhow!("unknown search model '{model_id}'"))?;
        let dir = data_dir.join("skills").join(model_id);
        let vision_path = dir.join("vision_model.onnx");
        let text_path   = dir.join("text_model.onnx");
        let tok_path    = dir.join("tokenizer.json");
        for p in [&vision_path, &text_path, &tok_path] {
            if !p.exists() { anyhow::bail!("{model_id}: missing {}", p.display()); }
        }
        // CPU on purpose: this background embedder must not run concurrently with
        // the real-time detector on one GPU (DirectML raises "Add node parameter is
        // incorrect" under concurrent inference). CPU CLIP embedding is plenty fast
        // for background jobs and frees the GPU for detection.
        let vision = build_ort_session_cpu(&vision_path)?;
        let text   = build_ort_session_cpu(&text_path)?;
        let tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow::anyhow!("{model_id} tokenizer load: {e}"))?;

        let vision_input = vision.inputs().first()
            .map(|i| i.name().to_string()).unwrap_or_else(|| "pixel_values".into());
        let vision_out = pick_embed_output(&vision);

        let text_in: Vec<String> = text.inputs().iter().map(|i| i.name().to_string()).collect();
        let find = |needle: &str| text_in.iter().find(|n| n.contains(needle)).cloned();
        let text_ids_name = find("input_ids")
            .or_else(|| text_in.first().cloned())
            .unwrap_or_else(|| "input_ids".into());
        let text_attn_name = find("attention");
        let text_type_name = find("token_type");
        let text_out = pick_embed_output(&text);

        tracing::info!(
            "search model '{model_id}' loaded — img={} norm={} vision_in={vision_input} vision_out#{vision_out} \
             text_ids={text_ids_name} text_attn={text_attn_name:?} text_type={text_type_name:?} text_out#{text_out}",
            spec.image_size, spec.normalize
        );
        Ok(Self {
            spec,
            vision, text, tokenizer,
            vision_input, vision_out,
            text_ids_name, text_attn_name, text_type_name, text_out,
        })
    }

    /// Embed a JPEG into the shared CLIP space (L2-normalised). Preprocessing
    /// matches each model's image processor: resize shortest edge → centre-crop
    /// → /255 → (conditionally) subtract CLIP mean/std.
    pub fn encode_image(&mut self, jpeg: &[u8]) -> anyhow::Result<Vec<f32>> {
        let s = self.spec.image_size;
        let src = image::load_from_memory(jpeg)?.to_rgb8();
        let (w, h) = (src.width(), src.height());
        // Resize so the shortest edge == s (preserve aspect), then centre-crop s×s.
        let scale = s as f32 / (w.min(h).max(1) as f32);
        let nw = ((w as f32 * scale).round() as u32).max(s as u32);
        let nh = ((h as f32 * scale).round() as u32).max(s as u32);
        let resized = image::imageops::resize(&src, nw, nh, image::imageops::FilterType::CatmullRom);
        let x0 = (nw - s as u32) / 2;
        let y0 = (nh - s as u32) / 2;

        let mut data = vec![0.0f32; 3 * s * s];
        let plane = s * s;
        for y in 0..s {
            for x in 0..s {
                let p = resized.get_pixel(x0 + x as u32, y0 + y as u32);
                let d = y * s + x;
                for c in 0..3 {
                    let mut v = p[c] as f32 / 255.0;
                    if self.spec.normalize { v = (v - self.spec.mean[c]) / self.spec.std[c]; }
                    data[c * plane + d] = v;
                }
            }
        }
        let tensor = Tensor::<f32>::from_array(([1usize, 3, s, s], data))
            .map_err(|e| anyhow::anyhow!("image tensor: {e}"))?;
        let outputs = { let _t = crate::inference::infer_timer("clip_vision"); self.vision.run(ort::inputs![self.vision_input.as_str() => tensor]) }
            .map_err(|e| anyhow::anyhow!("vision run: {e}"))?;
        let v = extract_pooled(&outputs, self.vision_out, None)?;
        Ok(l2_normalise(v))
    }

    /// Embed a text string into the shared CLIP space (L2-normalised).
    pub fn encode_text(&mut self, text: &str) -> anyhow::Result<Vec<f32>> {
        let enc = self.tokenizer.encode(text, true)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
        let mut ids:  Vec<i64> = enc.get_ids().iter().map(|&i| i as i64).collect();
        let mut mask: Vec<i64> = enc.get_attention_mask().iter().map(|&i| i as i64).collect();
        if ids.is_empty() { anyhow::bail!("empty token sequence"); }
        let cap = self.spec.max_tokens;
        if ids.len() > cap { ids.truncate(cap); mask.truncate(cap); }
        // Fixed-context CLIP text towers require the input padded to the full
        // context length (their positional embedding is that fixed size) — else the
        // positional Add can't broadcast (e.g. "8 by 77"). Pad with the 0 token +
        // a 0 mask; dynamic towers are left variable-length.
        if self.spec.fixed_context {
            while ids.len() < cap { ids.push(0); mask.push(0); }
        }
        let seq = ids.len();

        let id_t = Tensor::<i64>::from_array(([1usize, seq], ids))
            .map_err(|e| anyhow::anyhow!("ids tensor: {e}"))?;

        // Build the run with exactly the inputs this export declares.
        let mask_for_pool = mask.clone();
        let outputs = match (&self.text_attn_name, &self.text_type_name) {
            (Some(attn), Some(tt)) => {
                let m  = Tensor::<i64>::from_array(([1usize, seq], mask)).map_err(te)?;
                let zt = Tensor::<i64>::from_array(([1usize, seq], vec![0i64; seq])).map_err(te)?;
                self.text.run(ort::inputs![
                    self.text_ids_name.as_str() => id_t,
                    attn.as_str() => m,
                    tt.as_str()   => zt
                ]).map_err(tr)?
            }
            (Some(attn), None) => {
                let m = Tensor::<i64>::from_array(([1usize, seq], mask)).map_err(te)?;
                self.text.run(ort::inputs![
                    self.text_ids_name.as_str() => id_t,
                    attn.as_str() => m
                ]).map_err(tr)?
            }
            _ => self.text.run(ort::inputs![self.text_ids_name.as_str() => id_t]).map_err(tr)?,
        };
        let v = extract_pooled(&outputs, self.text_out, Some(&mask_for_pool))?;
        Ok(l2_normalise(v))
    }
}

fn te(e: ort::Error) -> anyhow::Error { anyhow::anyhow!("text tensor: {e}") }
fn tr(e: ort::Error) -> anyhow::Error { anyhow::anyhow!("text run: {e}") }

/// Index of the output that looks like a projected embedding (name contains
/// "embed"), else the first output.
fn pick_embed_output(s: &OrtSession) -> usize {
    s.outputs().iter().position(|o| o.name().to_lowercase().contains("embed")).unwrap_or(0)
}

/// Pull output `idx` as a pooled `[D]` vector. Handles the projected `[1, D]`
/// case directly and the raw `[1, S, D]` hidden-state case by masked mean
/// pooling (so we still work if a token-tower export skips the pooling head).
fn extract_pooled(
    outputs: &ort::session::SessionOutputs,
    idx: usize,
    mask: Option<&[i64]>,
) -> anyhow::Result<Vec<f32>> {
    let (shape, data) = outputs[idx].try_extract_tensor::<f32>()
        .map_err(|e| anyhow::anyhow!("extract: {e}"))?;
    let dims: Vec<usize> = shape.iter().map(|&v| v.max(0) as usize).collect();
    match dims.as_slice() {
        // [1, D] or [D] — already pooled / projected.
        [_, d] if dims.len() == 2 => Ok(data[..*d.min(&data.len())].to_vec()),
        [d] => Ok(data[..*d.min(&data.len())].to_vec()),
        // [1, S, D] — masked mean pool over the sequence.
        [_, s, d] => {
            let (s, d) = (*s, *d);
            let mut out = vec![0.0f32; d];
            let mut denom = 0.0f32;
            for t in 0..s {
                let w = mask.and_then(|m| m.get(t)).map(|&v| v as f32).unwrap_or(1.0);
                if w == 0.0 { continue; }
                denom += w;
                let base = t * d;
                for k in 0..d { out[k] += w * data[base + k]; }
            }
            if denom > 0.0 { out.iter_mut().for_each(|x| *x /= denom); }
            Ok(out)
        }
        _ => anyhow::bail!("unexpected output rank {:?}", dims),
    }
}

fn l2_normalise(mut v: Vec<f32>) -> Vec<f32> {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 { v.iter_mut().for_each(|x| *x /= n); }
    v
}

// ─── Process-wide accessors (mirror face.rs's cache discipline) ──────────────

/// Run `f` with the cached encoder for `model_id`, lazily loading (or reloading
/// on a model switch) from `data_dir`. Returns `None` if `model_id` is "off"/
/// unknown or the skill isn't installed / failed to load — callers treat that as
/// "fall back to keyword search".
pub fn with_model<R>(data_dir: &Path, model_id: &str, f: impl FnOnce(&mut ClipEncoder) -> R) -> Option<R> {
    model_spec(model_id)?;
    let cell = MODEL.get_or_init(|| Mutex::new(None));
    let mut guard = cell.lock().ok()?;
    let needs_load = !matches!(guard.as_ref(), Some((id, _)) if id == model_id);
    if needs_load {
        match ClipEncoder::try_load(data_dir, model_id) {
            Ok(m) => *guard = Some((model_id.to_string(), m)),
            Err(e) => { tracing::debug!("search model '{model_id}' unavailable: {e}"); return None; }
        }
    }
    guard.as_mut().map(|(_, enc)| f(enc))
}

/// True if the given model's skill is installed (all three files present).
pub fn is_installed(data_dir: &Path, model_id: &str) -> bool {
    if model_spec(model_id).is_none() { return false; }
    let dir = data_dir.join("skills").join(model_id);
    dir.join("vision_model.onnx").exists()
        && dir.join("text_model.onnx").exists()
        && dir.join("tokenizer.json").exists()
}

// ─── f32 ⇆ BLOB (little-endian, same encoding as face/body embeddings) ───────

pub fn vec_to_blob(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub fn blob_to_vec(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Cosine similarity of two equal-length vectors. Both are L2-normalised at
/// store time, so this is just a dot product; we still divide by norms for
/// safety against any un-normalised input.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() { return -1.0; }
    let mut dot = 0.0f32;
    let (mut na, mut nb) = (0.0f32, 0.0f32);
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 { return -1.0; }
    dot / (na.sqrt() * nb.sqrt())
}
