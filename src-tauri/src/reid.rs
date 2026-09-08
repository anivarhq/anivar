//! Body Re-ID via HSV histogram appearance descriptor.
//!
//! Computes a 144-dimensional colour histogram from a person crop:
//!   3 vertical strips x (16 hue + 16 sat + 16 val bins) = 144 dims total.
//!
//! Gives soft-biometric matching that survives masks, distance and poor
//! lighting. Threshold >= 0.85 cosine similarity = same person.

use std::path::Path;
use std::sync::{Mutex, OnceLock};

use ort::session::Session as OrtSession;
use ort::value::Tensor;
use sqlx::SqlitePool;


//
// Computes a 288-dimensional color histogram from a person crop:
//   3 vertical strips × (16 hue bins + 16 sat bins + 16 val bins) = 144 dims per strip
// Wait — 3 strips × 48 bins = 144 dims total.
// This gives soft-biometric matching that survives masks, distance and poor lighting.
// Threshold ≥ 0.85 cosine similarity → same person.

/// Crop a person bounding box from a JPEG, resize to 128×256 (Re-ID standard),
/// and compute a 144-dimensional HSV histogram descriptor.
pub(crate) fn compute_body_descriptor(jpeg: &[u8], box_: &[f32; 4]) -> Option<Vec<f32>> {
    let img = image::load_from_memory(jpeg).ok()?;
    let (iw, ih) = (img.width() as f32, img.height() as f32);

    // Clamp bounding box to image bounds
    let x1 = (box_[0]).max(0.0).min(iw - 1.0) as u32;
    let y1 = (box_[1]).max(0.0).min(ih - 1.0) as u32;
    let x2 = (box_[2]).max(0.0).min(iw) as u32;
    let y2 = (box_[3]).max(0.0).min(ih) as u32;
    if x2 <= x1 || y2 <= y1 { return None; }

    let crop = img.crop_imm(x1, y1, x2 - x1, y2 - y1);
    // Standard Re-ID resolution: 64 wide × 128 tall
    let resized = crop.resize_exact(64, 128, image::imageops::FilterType::Triangle).to_rgb8();
    let w = resized.width();
    let h = resized.height();
    let strip_h = h / 3;

    const H_BINS: usize = 16;
    const S_BINS: usize = 16;
    const V_BINS: usize = 16;
    const STRIP_DIMS: usize = H_BINS + S_BINS + V_BINS; // 48 per strip
    let mut desc = vec![0.0f32; 3 * STRIP_DIMS];

    for py in 0..h {
        let strip = ((py / strip_h) as usize).min(2);
        for px in 0..w {
            let pix = resized.get_pixel(px, py);
            let (r, g, b) = (pix[0] as f32 / 255.0, pix[1] as f32 / 255.0, pix[2] as f32 / 255.0);
            // RGB → HSV
            let cmax = r.max(g).max(b);
            let cmin = r.min(g).min(b);
            let delta = cmax - cmin;
            let h_val = if delta < 1e-6 { 0.0 } else if cmax == r {
                60.0 * (((g - b) / delta) % 6.0)
            } else if cmax == g {
                60.0 * ((b - r) / delta + 2.0)
            } else {
                60.0 * ((r - g) / delta + 4.0)
            };
            let h_norm = ((h_val / 360.0 + 1.0) % 1.0).clamp(0.0, 0.9999);
            let s_norm = if cmax > 0.0 { (delta / cmax).min(0.9999) } else { 0.0 };
            let v_norm = (cmax).min(0.9999);

            let hi = (h_norm * H_BINS as f32) as usize;
            let si = (s_norm * S_BINS as f32) as usize;
            let vi = (v_norm * V_BINS as f32) as usize;
            let base = strip * STRIP_DIMS;
            desc[base + hi]             += 1.0;
            desc[base + H_BINS + si]     += 1.0;
            desc[base + H_BINS + S_BINS + vi] += 1.0;
        }
    }

    // L2 normalise so cosine similarity = dot product
    let norm: f32 = desc.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 { desc.iter_mut().for_each(|v| *v /= norm); }
    Some(desc)
}

/// Cosine similarity between two descriptors (both assumed L2-normalised).
pub(crate) fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ─── Body hard-negatives (durable corrections) ───────────────────────────────────
// The body-Re-ID analog of `face_negatives`. A negative says "this appearance is NOT
// known_person_id"; matching/clustering then SUPPRESS that person whenever a candidate
// resembles one of their negatives — so a tracking correction sticks and the same
// appearance mistake can't silently re-merge on the next sighting.

/// Recent body negatives keyed by `known_person_id` (descriptor vectors).
async fn load_body_negatives(db: &SqlitePool) -> std::collections::HashMap<String, Vec<Vec<f32>>> {
    let rows: Vec<(String, Vec<u8>)> = sqlx::query_as(
        "SELECT known_person_id, descriptor FROM body_negatives ORDER BY created_at DESC LIMIT 2000"
    ).fetch_all(db).await.unwrap_or_default();
    let mut m: std::collections::HashMap<String, Vec<Vec<f32>>> = std::collections::HashMap::new();
    for (kid, blob) in rows {
        if blob.len() % 4 != 0 { continue; }
        let v: Vec<f32> = blob.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        m.entry(kid).or_default().push(v);
    }
    m
}

/// True if `candidate` resembles a hard-negative for `known_id` (cosine ≥ floor) — i.e.
/// the user has said "this appearance is NOT that person", so the match must be dropped.
fn is_body_negative(
    candidate: &[f32], known_id: &str,
    negs: &std::collections::HashMap<String, Vec<Vec<f32>>>, floor: f32,
) -> bool {
    negs.get(known_id).map(|list| {
        list.iter().filter(|n| n.len() == candidate.len()).any(|n| cosine_sim(candidate, n) >= floor)
    }).unwrap_or(false)
}

/// The enrolled people's BODY galleries + their negatives, for "looks like X" suggestion.
/// Keyed by known_person_id (so negatives apply); carries id→name for display.
struct KnownBodyGalleries {
    by_id: std::collections::HashMap<String, Vec<Vec<f32>>>,
    name:  std::collections::HashMap<String, String>,
    negs:  std::collections::HashMap<String, Vec<Vec<f32>>>,
}

impl KnownBodyGalleries {
    async fn load(db: &SqlitePool) -> Self {
        let rows: Vec<(String, Option<String>, Vec<u8>)> = sqlx::query_as(
            "SELECT k.id, k.name, b.descriptor FROM body_embeddings b
             JOIN known_persons k ON k.id = b.known_person_id
             WHERE b.known_person_id IS NOT NULL ORDER BY b.seen_at DESC LIMIT 600"
        ).fetch_all(db).await.unwrap_or_default();
        let mut by_id: std::collections::HashMap<String, Vec<Vec<f32>>> = std::collections::HashMap::new();
        let mut name:  std::collections::HashMap<String, String> = std::collections::HashMap::new();
        for (id, nm, blob) in rows {
            if blob.len() % 4 != 0 || blob.len() / 4 < 256 { continue; } // deep only
            by_id.entry(id.clone()).or_default()
                .push(blob.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect());
            if let Some(n) = nm { name.entry(id).or_insert(n); }
        }
        let negs = load_body_negatives(db).await;
        KnownBodyGalleries { by_id, name, negs }
    }

    fn is_empty(&self) -> bool { self.by_id.is_empty() }

    /// Best known person for a centroid by KNOWN_ID (top-3 mean cosine ≥ floor), honoring
    /// negatives. The id-returning core used by both the display suggestion and automation.
    fn suggest_id(&self, centroid: &[f32], floor: f32) -> Option<(String, f32)> {
        let mut best = (0.0f32, String::new());
        for (id, embs) in &self.by_id {
            if is_body_negative(centroid, id, &self.negs, floor) { continue; } // user said NOT this person
            let mut sims: Vec<f32> = embs.iter().filter(|e| e.len() == centroid.len())
                .map(|e| cosine_sim(centroid, e)).collect();
            if sims.is_empty() { continue; }
            sims.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let k = sims.len().min(3);
            let s = sims[..k].iter().sum::<f32>() / k as f32;
            if s > best.0 { best = (s, id.clone()); }
        }
        if best.0 >= floor && !best.1.is_empty() { Some((best.1, best.0)) } else { None }
    }

    /// "Looks like X" for display suggestions — negative-aware.
    /// Returns (person_id, name, score); the ID is what confirm actions bind to.
    fn suggest(&self, centroid: &[f32], floor: f32) -> (Option<String>, Option<String>, Option<f32>) {
        match self.suggest_id(centroid, floor) {
            Some((id, s)) => (Some(id.clone()), self.name.get(&id).cloned(), Some(s)),
            None => (None, None, None),
        }
    }
}

// ─── Deep person Re-ID — edge-AI-style durable embedding ───────────────────
// When a deep Re-ID skill is installed we replace the soft 144-d HSV histogram
// with a deep CNN/ViT embedding (robust to clothing/lighting → durable cross-camera
// identity). The descriptor LENGTH identifies its space; `query_body_reid` only
// compares same-length descriptors, so different backbones never cross-contaminate
// across a model switch. Falls back to HSV when no deep skill is installed.
//
// Backbones are tried in PRIORITY order (best first). The default OSNet x0.25
// (`reid_osnet`) over-specialises on its training domain (mAP collapses cross-domain),
// so `reid_osnet_ain` (instance-norm, domain-generalisable) is preferred when present,
// and `reid_clip` (CLIP-ReID, GPU) above that. Each carries its own preprocessing +
// cosine match floor (deep spaces match at a lower cosine than the HSV histogram).

const IN_MEAN: [f32; 3] = [0.485, 0.456, 0.406]; // ImageNet (torchreid/OSNet/most ReID)
const IN_STD:  [f32; 3] = [0.229, 0.224, 0.225];

struct ReidBackbone {
    skill: &'static str,
    label: &'static str,   // shown in People → Tracked
    w: u32,
    h: u32,
    mean: [f32; 3],
    std:  [f32; 3],
    floor: f32,            // cosine match floor for this embedding space
}

// Priority order — first installed wins. Only the verified OSNet ONNX
// (anriha/osnet_x0_25_msmt17, 256×128 ImageNet-norm, 512-d) ships today; extra
// backbones can be added here as their ONNX exports are verified.
const REID_BACKBONES: &[ReidBackbone] = &[
    ReidBackbone { skill: "reid_osnet", label: "Deep (OSNet)", w: 128, h: 256, mean: IN_MEAN, std: IN_STD, floor: 0.62 },
];

fn active_backbone(data_dir: &Path) -> Option<&'static ReidBackbone> {
    REID_BACKBONES.iter()
        .find(|b| data_dir.join("skills").join(b.skill).join("model.onnx").exists())
}

/// Human label for the active Re-ID backbone (People → Tracked caption).
pub(crate) fn active_reid_label(data_dir: &Path) -> String {
    active_backbone(data_dir).map(|b| b.label.to_string())
        .unwrap_or_else(|| "Color histogram".into())
}

/// Which Re-ID backbone is currently driving cross-camera body matching — drives
/// the People → Tracked caption so the soft-signal quality is legible to the user.
#[tauri::command]
pub async fn reid_backend_status(
    state: tauri::State<'_, std::sync::Arc<crate::AppState>>,
) -> Result<String, String> {
    Ok(active_reid_label(&state.data_dir))
}

/// Cosine match floor for the active deep backbone (HSV keeps its own threshold).
pub(crate) fn active_deep_floor(data_dir: &Path) -> f32 {
    active_backbone(data_dir).map(|b| b.floor).unwrap_or(0.62)
}

// Cached deep session: (session, input_name, skill_id, batch). `batch` is the
// model's required batch size, probed on first use (0 = not yet probed). Some
// real exports (e.g. the public OSNet ONNX) are compiled with a FIXED batch
// (16), not dynamic — feeding [1,...] is rejected, so we must tile the single
// crop to the model's batch and read row 0. Keyed by skill id → switching
// backbones reloads cleanly.
static REID_SESSION: OnceLock<Mutex<Option<(OrtSession, String, &'static str, usize)>>> = OnceLock::new();

// Batch sizes to probe, in order. 1 = dynamic/batch-1 exports; 16 = the common
// fixed-batch OSNet export. First one that runs is cached.
const REID_BATCH_CANDIDATES: &[usize] = &[1, 16];

/// QUALITY GATE for trusting a person crop with DURABLE identity (cross-camera match +
/// gallery storage). Tiny, low-confidence, or non-person-shaped (wide/merged) boxes give
/// unreliable embeddings — the main source of "by appearance" mistakes — so they're kept
/// OUT of identity (the short-term frame-to-frame tracker still uses them for association).
/// Grounded in Re-ID practice: usable crops sit well above the ~30px detection floor and
/// keep a standing-person aspect.
pub(crate) fn crop_is_reliable(box_: &[f32; 4], score: f32) -> bool {
    let w = (box_[2] - box_[0]).max(0.0);
    let h = (box_[3] - box_[1]).max(0.0);
    score >= 0.5 && h >= 64.0 && w >= 24.0 && h >= w * 1.1
}

/// Body descriptor for Re-ID: deep embedding when a skill is installed, else the
/// HSV histogram. Caller doesn't need to know which.
pub(crate) fn body_descriptor(data_dir: &Path, jpeg: &[u8], box_: &[f32; 4]) -> Option<Vec<f32>> {
    if let Some(b) = active_backbone(data_dir) {
        if let Some(v) = compute_body_descriptor_deep(data_dir, b, jpeg, box_) {
            return Some(v);
        }
        // Deep model present but inference failed → fall through to HSV (never lose tracking).
    }
    compute_body_descriptor(jpeg, box_)
}

fn compute_body_descriptor_deep(data_dir: &Path, b: &ReidBackbone, jpeg: &[u8], box_: &[f32; 4]) -> Option<Vec<f32>> {
    let model_path = data_dir.join("skills").join(b.skill).join("model.onnx");
    let cell = REID_SESSION.get_or_init(|| Mutex::new(None));
    let mut guard = cell.lock().ok()?;
    let need_reload = guard.as_ref().map(|(_, _, s, _)| *s) != Some(b.skill);
    if need_reload {
        // Enrichment lane = CPU (event cadence; keeps the GPU lock YOLO-only
        // and avoids a per-model TRT engine build that would block YOLO).
        match crate::inference::build_ort_session_cpu(&model_path) {
            Ok(s) => {
                let input = s.inputs().first().map(|i| i.name().to_string())
                    .unwrap_or_else(|| "images".into());
                *guard = Some((s, input, b.skill, 0)); // batch 0 = probe on first run
            }
            Err(e) => { tracing::debug!("{} load failed: {e}", b.skill); return None; }
        }
    }
    let (session, input_name, _, batch) = guard.as_mut()?;

    let img = image::load_from_memory(jpeg).ok()?;
    let (iw, ih) = (img.width() as f32, img.height() as f32);
    let x1 = box_[0].max(0.0).min(iw - 1.0) as u32;
    let y1 = box_[1].max(0.0).min(ih - 1.0) as u32;
    let x2 = box_[2].max(0.0).min(iw) as u32;
    let y2 = box_[3].max(0.0).min(ih) as u32;
    if x2 <= x1 || y2 <= y1 { return None; }
    let crop = img.crop_imm(x1, y1, x2 - x1, y2 - y1)
        .resize_exact(b.w, b.h, image::imageops::FilterType::Triangle).to_rgb8();
    let (w, h) = (b.w as usize, b.h as usize);
    let plane = w * h;
    // One image worth of CHW input.
    let mut chw = vec![0.0f32; 3 * plane];
    for y in 0..h {
        for x in 0..w {
            let p = crop.get_pixel(x as u32, y as u32);
            let d = y * w + x;
            for c in 0..3 {
                chw[c * plane + d] = ((p[c] as f32 / 255.0) - b.mean[c]) / b.std[c];
            }
        }
    }

    // Probe the model's required batch on first use (some exports are fixed-batch);
    // then reuse the cached size. We tile the single crop across the batch and read
    // row 0 — every row is identical, so the first embedding is the one we want.
    let candidates: Vec<usize> = if *batch == 0 { REID_BATCH_CANDIDATES.to_vec() } else { vec![*batch] };
    for &bs in &candidates {
        let mut data = Vec::with_capacity(bs * chw.len());
        for _ in 0..bs { data.extend_from_slice(&chw); }
        let Ok(tensor) = Tensor::<f32>::from_array(([bs, 3, h, w], data)) else { continue };
        let Ok(outputs) = ({ let _t = crate::inference::infer_timer("reid");
            session.run(ort::inputs![input_name.as_str() => tensor]) }) else { continue };
        let Ok((_, raw)) = outputs[0].try_extract_tensor::<f32>() else { continue };
        if raw.is_empty() { continue; }
        // Row 0 = first (len / batch) floats (the per-image embedding).
        let dim = (raw.len() / bs).max(1);
        let mut v = raw[..dim.min(raw.len())].to_vec();
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if n > 0.0 { v.iter_mut().for_each(|x| *x /= n); }
        *batch = bs; // remember the working batch
        return if v.is_empty() { None } else { Some(v) };
    }
    None
}

/// Query body_embeddings for the person whose descriptor best matches `desc`.
/// Returns (person_id, similarity) if a match above `threshold` is found.
pub(crate) async fn query_body_reid(
    db: &SqlitePool,
    desc: &[f32],
    threshold: f32,
    deep_floor: f32,
) -> Option<(String, f32)> {
    // Load recent embeddings (last 7 days, up to 1000)
    let rows: Vec<(String, Vec<u8>)> = sqlx::query_as(
        "SELECT person_id, descriptor FROM body_embeddings
         WHERE seen_at > datetime('now', '-7 days')
         ORDER BY seen_at DESC LIMIT 1000"
    ).fetch_all(db).await.ok()?;

    // (C) FastReID-style robustness: instead of trusting a single nearest row,
    // collect every same-space similarity PER PERSON and score each person by the
    // MEAN OF ITS TOP-K — so one lucky frame can't win, and a person with several
    // consistent matches beats a one-off near-duplicate.
    const TOP_K: usize = 3;
    let mut per_person: std::collections::HashMap<String, Vec<f32>> = std::collections::HashMap::new();
    for (pid, blob) in &rows {
        if blob.len() % 4 != 0 { continue; }
        let stored: Vec<f32> = blob.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        // Only compare descriptors from the SAME space (deep OSNet 512-d vs HSV
        // 144-d) — a model switch leaves both in the table; mixing them is garbage.
        if stored.len() != desc.len() { continue; }
        per_person.entry(pid.clone()).or_default().push(cosine_sim(desc, &stored));
    }

    // Hard-negatives: never auto-resolve to a known person (kp_<id>) whose appearance the
    // user has corrected away from — so a fixed mislabel can't silently come back.
    let negs = load_body_negatives(db).await;

    let mut best_pid = String::new();
    let mut best_score = 0.0f32;
    for (pid, mut sims) in per_person {
        if let Some(known_id) = pid.strip_prefix("kp_") {
            if is_body_negative(desc, known_id, &negs, deep_floor) { continue; }
        }
        sims.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let k = sims.len().min(TOP_K);
        if k == 0 { continue; }
        let score = sims[..k].iter().sum::<f32>() / k as f32;
        if score > best_score { best_score = score; best_pid = pid; }
    }

    // Deep embeddings (≥256-d) match at a lower cosine than the HSV histogram, so
    // use the active backbone's space-appropriate floor (passed by the caller) and
    // ignore the HSV-tuned `threshold` for them.
    let effective = if desc.len() >= 256 { deep_floor } else { threshold };
    if best_score >= effective && !best_pid.is_empty() { Some((best_pid, best_score)) } else { None }
}

/// Crop a person's bounding box from a frame JPEG and re-encode it as a small
/// base64 JPEG, so the People → Tracked view can show the actual person. Body
/// Re-ID has no model crop otherwise (the descriptor is a bare histogram), which
/// is why those cards used to render a blank placeholder. Long edge capped at
/// 256 px — these are list thumbnails, not evidence frames.
pub(crate) fn crop_person_jpeg(jpeg: &[u8], box_: &[f32; 4]) -> Option<String> {
    let img = image::load_from_memory(jpeg).ok()?;
    let (iw, ih) = (img.width() as f32, img.height() as f32);
    let x1 = box_[0].max(0.0).min(iw - 1.0) as u32;
    let y1 = box_[1].max(0.0).min(ih - 1.0) as u32;
    let x2 = box_[2].max(0.0).min(iw) as u32;
    let y2 = box_[3].max(0.0).min(ih) as u32;
    if x2 <= x1 || y2 <= y1 { return None; }
    let mut crop = img.crop_imm(x1, y1, x2 - x1, y2 - y1);
    if crop.width() > 256 || crop.height() > 256 {
        crop = crop.resize(256, 256, image::imageops::FilterType::Triangle);
    }
    let mut jpeg_out = Vec::with_capacity(8192);
    let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg_out, 80);
    enc.encode_image(&crop).ok()?;
    Some(base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &jpeg_out))
}

/// Store a new body descriptor. Uses person_id from Re-ID match or auto-generated.
/// `thumb` is an optional base64 JPEG crop of the person (stored on the first
/// sighting of a new person so the Tracked view has a preview).
/// Clothing colors from a person detection: dominant color of the TORSO band
/// (18–52% of bbox height — shirt/jacket) and the LEGS band (55–92% — pants/
/// skirt), sides trimmed 18% against background bleed. Same HSV plurality-vote
/// core as the vehicle classifier (`alpr::dominant_color_in_band`), same honest
/// abstain — a band with no dominant color contributes nothing rather than a
/// guess. Returns compact JSON like {"top":"blue","bottom":"black"}, or None
/// when neither band is confident. Words only — no imagery is derived/stored.
pub(crate) fn classify_person_colors(jpeg: &[u8], bbox: &[f32; 4]) -> Option<String> {
    let img = image::load_from_memory(jpeg).ok()?.to_rgb8();
    let top    = crate::alpr::dominant_color_in_band(&img, *bbox, (0.18, 0.52, 0.18));
    let bottom = crate::alpr::dominant_color_in_band(&img, *bbox, (0.55, 0.92, 0.18));
    if top.is_none() && bottom.is_none() { return None; }
    let mut o = serde_json::Map::new();
    if let Some((c, _)) = top    { o.insert("top".into(),    c.into()); }
    if let Some((c, _)) = bottom { o.insert("bottom".into(), c.into()); }
    Some(serde_json::Value::Object(o).to_string())
}

pub(crate) async fn store_body_embedding(
    db: &SqlitePool,
    person_id: &str,
    desc: &[f32],
    cam_id: u8,
    event_id: Option<&str>,
    thumb: Option<&str>,
    attrs: Option<&str>,
    // How this row's identity was decided ("body_reid" for a cross-camera body
    // match) + that match's score. None for anonymous tracks (no decision made).
    provenance: Option<(&str, Option<f32>)>,
) {
    let blob: Vec<u8> = desc.iter().flat_map(|v| v.to_le_bytes()).collect();
    let id = uuid::Uuid::new_v4().to_string();
    let (method, score) = match provenance {
        Some((m, s)) => (Some(m), s.map(|v| v as f64)),
        None => (None, None),
    };
    sqlx::query(
        "INSERT INTO body_embeddings(id, person_id, descriptor, cam_id, event_id, thumbnail_b64, attrs, match_method, match_score)
         VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)"
    ).bind(&id).bind(person_id).bind(&blob).bind(cam_id as i64).bind(event_id).bind(thumb).bind(attrs)
     .bind(method).bind(score)
     .execute(db).await.ok();
}

/// Cap on a known person's auto-labelled body gallery (keeps it current + bounded).
const KP_GALLERY_CAP: i64 = 30;

/// Face↔body fusion (Avigilon/BriefCam-style): attach a body descriptor seen
/// co-occurring with a recognised face to that KNOWN person's appearance gallery,
/// so future distant / cross-camera bodies resolve to their NAME. Stored under a
/// stable `kp_<known_id>` person_id with `known_person_id` set. De-duped against
/// recent samples (cosine ≥ 0.92) and pruned to the most-recent CAP — so it tracks
/// the current outfit (the face keeps re-anchoring it; the 7-day query window ages
/// out stale clothing automatically).
pub(crate) async fn link_body_to_known(
    db: &SqlitePool,
    known_id: &str,
    desc: &[f32],
    cam_id: u8,
    thumb: Option<&str>,
    attrs: Option<&str>,
    // The consensus-confirmed FACE score that anchors this fusion — provenance
    // so a poisoned gallery is traceable back to the face match that fed it.
    face_score: f32,
) {
    let pid = format!("kp_{known_id}");
    // Skip near-identical recent samples (a lingering person mustn't flood the gallery).
    let rows: Vec<(Vec<u8>,)> = sqlx::query_as(
        "SELECT descriptor FROM body_embeddings WHERE person_id=? ORDER BY seen_at DESC LIMIT ?"
    ).bind(&pid).bind(KP_GALLERY_CAP).fetch_all(db).await.unwrap_or_default();
    for (blob,) in &rows {
        if blob.len() != desc.len() * 4 { continue; }
        let stored: Vec<f32> = blob.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        if cosine_sim(desc, &stored) >= 0.92 { return; } // already represented
    }
    let blob: Vec<u8> = desc.iter().flat_map(|v| v.to_le_bytes()).collect();
    let id = uuid::Uuid::new_v4().to_string();
    let _ = sqlx::query(
        "INSERT INTO body_embeddings(id, person_id, descriptor, cam_id, event_id, thumbnail_b64, known_person_id, attrs, match_method, match_score)
         VALUES(?, ?, ?, ?, NULL, ?, ?, ?, 'fusion', ?)"
    ).bind(&id).bind(&pid).bind(&blob).bind(cam_id as i64).bind(thumb).bind(known_id).bind(attrs)
     .bind(face_score as f64)
     .execute(db).await;
    // Prune to the most-recent CAP for this person's gallery.
    let _ = sqlx::query(
        "DELETE FROM body_embeddings WHERE person_id=? AND id NOT IN
           (SELECT id FROM body_embeddings WHERE person_id=? ORDER BY seen_at DESC LIMIT ?)"
    ).bind(&pid).bind(&pid).bind(KP_GALLERY_CAP).execute(db).await;
}

/// Delete anonymous (non-face-anchored) appearance fragments older than 2 days —
/// they can never match again once the outfit changes, so they're pure noise. Run
/// hourly from the inference loop (a one-time copy also runs at boot in `init_db`).
/// Face-linked galleries (`kp_*` / `known_person_id` set) are always kept.
pub(crate) async fn prune_anon_bodies(db: &SqlitePool) {
    let _ = sqlx::query(
        "DELETE FROM body_embeddings
          WHERE known_person_id IS NULL AND person_id LIKE 'body\\_%' ESCAPE '\\'
            AND seen_at < datetime('now','-2 days')"
    ).execute(db).await;
}

/// AUTOMATION (hourly): auto-label confident anonymous tracks. A person who was anonymous
/// at first sighting gets promoted to a KNOWN identity once enough gallery evidence accrues
/// — no human tap — by matching each recent `body_*` track's centroid against the named
/// galleries at a STRICT floor (well above the interactive suggestion) and honoring
/// hard-negatives. Conservative on purpose: needs ≥2 samples + high confidence, only links
/// to EXISTING people (never creates an identity), and a wrong promotion is still fixable
/// via "Wrong?" (which records a durable negative, so it won't recur). Tesla-style active
/// FACE OVERRIDES BODY: a consensus-confirmed face disagreed with the identity a
/// live tracklet was carrying (from a body match or an earlier bad fusion). The
/// face is the anchor biometric, so it wins — but surgically:
///   • the polluting samples (this camera, last 10 min — the current tracklet's
///     lifetime) MOVE to the face-confirmed person's gallery (they are verified
///     body shots of that person — GEFF-style face-verified enrichment), tagged
///     `face_override` so the conflict is traceable;
///   • when the displaced identity was a KNOWN person, those descriptors also
///     become hard negatives for them (this appearance is NOT that person);
///   • the rest of the displaced person's gallery is untouched (it may be legit).
/// The caller re-points the live tracklet cache afterwards.
pub(crate) async fn face_override_track(
    db: &SqlitePool,
    old_pid: &str,              // the tracklet's previous id: "kp_<id>" or "body_<uuid>"
    correct_known_id: &str,     // face-confirmed person
    cam_id: u8,
    face_score: f32,
) {
    let wrong_known: Option<String> = old_pid.strip_prefix("kp_")
        .filter(|w| *w != correct_known_id)
        .map(str::to_string);
    if old_pid.starts_with("kp_") && wrong_known.is_none() { return; } // same person — nothing to fix

    // The current tracklet's samples: this camera, recent. For an anonymous
    // track the whole track is this person, so take it all.
    let scope = if old_pid.starts_with("kp_") {
        "person_id=? AND cam_id=? AND seen_at > datetime('now','-10 minutes')"
    } else {
        "person_id=? AND cam_id>=?" // cam_id>=0 → whole anonymous track
    };
    let cam_bind: i64 = if old_pid.starts_with("kp_") { cam_id as i64 } else { 0 };

    // Hard negatives for the displaced KNOWN person (deep descriptors only).
    if let Some(wrong) = &wrong_known {
        let rows: Vec<(Vec<u8>,)> = sqlx::query_as(
            &format!("SELECT descriptor FROM body_embeddings WHERE {scope} LIMIT 30")
        ).bind(old_pid).bind(cam_bind).fetch_all(db).await.unwrap_or_default();
        for (blob,) in &rows {
            if blob.len() % 4 != 0 { continue; }
            let dim = (blob.len() / 4) as i64;
            if dim < 256 { continue; }
            let _ = sqlx::query(
                "INSERT INTO body_negatives(id, known_person_id, descriptor, dim) VALUES(?,?,?,?)"
            ).bind(uuid::Uuid::new_v4().to_string()).bind(wrong).bind(blob).bind(dim)
             .execute(db).await;
        }
    }

    // Move the samples to the face-confirmed person, provenance 'face_override'.
    let kp = format!("kp_{correct_known_id}");
    let _ = sqlx::query(&format!(
        "UPDATE body_embeddings SET person_id=?, known_person_id=?, match_method='face_override', match_score=? WHERE {scope}"
    )).bind(&kp).bind(correct_known_id).bind(face_score as f64)
      .bind(old_pid).bind(cam_bind).execute(db).await;
    let _ = sqlx::query(
        "DELETE FROM body_embeddings WHERE person_id=? AND id NOT IN
           (SELECT id FROM body_embeddings WHERE person_id=? ORDER BY seen_at DESC LIMIT ?)"
    ).bind(&kp).bind(&kp).bind(KP_GALLERY_CAP).execute(db).await;
    tracing::info!(
        "face override: track {} on cam{} re-attributed to known {} (face score {:.2}{})",
        old_pid, cam_id, correct_known_id, face_score,
        wrong_known.as_deref().map(|w| format!(", negatives recorded for {w}")).unwrap_or_default()
    );
}

/// AUTOMATION (hourly): SILENTLY reconcile ID-SPLITS — the hardest self-learning
/// problem in a multi-camera NVR. When a person walks behind a pillar / into a
/// shadow, the tracker loses them and assigns a NEW `body_*` id to the SAME
/// individual (the "In-Flight Identity Switch"). This clusters all recent anonymous
/// tracks by appearance — the SAME usearch kNN graph + Chinese Whispers primitives
/// as face discovery — at a STRICT floor, and merges fragments of one person into a
/// single canonical id (the EARLIEST — so Person_45 folds back into Person_12).
///
/// Safe to run unattended: (1) only anonymous `body_*` tracks (never face-anchored
/// or `kp_` known galleries); (2) a merge floor well ABOVE the matching floor, since
/// a merge is more consequential than a suggestion; (3) a spatio-temporal VETO — two
/// tracks visible on the SAME camera within 2 s are DIFFERENT people (the tracker
/// already separated them), so a cluster containing such a pair is never merged;
/// (4) body is a soft signal and a bad merge is user-splittable. This is the
/// blueprint's Stage-3 "silent correction": duplicate cards quietly consolidate.
pub(crate) async fn reconcile_anon_tracks(db: &SqlitePool, data_dir: &Path) {
    let base = active_deep_floor(data_dir);
    let merge_floor = (base + 0.20).max(0.80); // merging is consequential → very strict

    // Recent anonymous deep tracks → centroid + (cam, seen_at) sightings per track.
    let rows: Vec<(String, Vec<u8>, i64, String)> = sqlx::query_as(
        "SELECT person_id, descriptor, cam_id, seen_at FROM body_embeddings
          WHERE known_person_id IS NULL AND person_id LIKE 'body\\_%' ESCAPE '\\'
            AND seen_at > datetime('now','-1 day') ORDER BY seen_at DESC LIMIT 4000"
    ).fetch_all(db).await.unwrap_or_default();

    struct T { sum: Vec<f32>, n: f32, first: String, sightings: Vec<(i64, String)> }
    let mut tracks: std::collections::HashMap<String, T> = std::collections::HashMap::new();
    for (pid, blob, cam, seen) in rows {
        if blob.len() % 4 != 0 || blob.len() / 4 < 256 { continue; }
        let v: Vec<f32> = blob.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let e = tracks.entry(pid).or_insert_with(|| T { sum: vec![0.0; v.len()], n: 0.0, first: seen.clone(), sightings: Vec::new() });
        if e.sum.len() == v.len() { for (s, x) in e.sum.iter_mut().zip(&v) { *s += *x; } e.n += 1.0; }
        if seen < e.first { e.first = seen.clone(); }
        if e.sightings.len() < 60 { e.sightings.push((cam, seen)); }
    }
    // Keep tracks with ≥2 samples; build parallel id/centroid arrays for clustering.
    let ids: Vec<String> = tracks.iter().filter(|(_, t)| t.n >= 2.0).map(|(k, _)| k.clone()).collect();
    if ids.len() < 2 { return; }
    let centroids: Vec<Vec<f32>> = ids.iter().map(|id| {
        let t = &tracks[id];
        let mut c: Vec<f32> = t.sum.iter().map(|s| s / t.n).collect();
        let norm: f32 = c.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 { c.iter_mut().for_each(|x| *x /= norm); }
        c
    }).collect();

    // usearch kNN graph + Chinese Whispers over the track centroids.
    let edges = crate::vector_index::knn_graph(&centroids, 8, merge_floor);
    if edges.is_empty() { return; }
    let labels = crate::vector_index::chinese_whispers(ids.len(), &edges, 30);
    let mut groups: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
    for (i, &l) in labels.iter().enumerate() { groups.entry(l).or_default().push(i); }

    let mut merged = 0u64;
    for (_, idxs) in groups {
        if idxs.len() < 2 { continue; }
        // SPATIO-TEMPORAL VETO: any two tracks seen on the SAME camera within 2 s are
        // distinct people → don't merge this cluster.
        let mut impossible = false;
        'v: for a in 0..idxs.len() {
            for b in (a + 1)..idxs.len() {
                let (ta, tb) = (&tracks[&ids[idxs[a]]], &tracks[&ids[idxs[b]]]);
                for (ca, sa) in &ta.sightings {
                    for (cb, sb) in &tb.sightings {
                        if ca == cb && seen_within(sa, sb, 2) { impossible = true; break 'v; }
                    }
                }
            }
        }
        if impossible { continue; }
        // Canonical = earliest first-seen (fold newer fragments INTO the original).
        let canon_i = *idxs.iter().min_by(|&&a, &&b| tracks[&ids[a]].first.cmp(&tracks[&ids[b]].first)).unwrap();
        let canon = ids[canon_i].clone();
        for &j in &idxs {
            if j == canon_i { continue; }
            let _ = sqlx::query(
                "UPDATE body_embeddings SET person_id=? WHERE person_id=? AND known_person_id IS NULL"
            ).bind(&canon).bind(&ids[j]).execute(db).await;
            merged += 1;
        }
        // Prune the consolidated anonymous gallery (a little larger than a known one).
        let _ = sqlx::query(
            "DELETE FROM body_embeddings WHERE person_id=? AND id NOT IN
               (SELECT id FROM body_embeddings WHERE person_id=? ORDER BY seen_at DESC LIMIT ?)"
        ).bind(&canon).bind(&canon).bind(KP_GALLERY_CAP * 3).execute(db).await;
    }
    if merged > 0 { tracing::info!("reconcile: silently merged {merged} split body-track fragment(s)"); }
}

/// True if two `seen_at` stamps ("YYYY-MM-DD HH:MM:SS" UTC, or RFC3339) are within
/// `secs` of each other. Used by the spatio-temporal merge veto.
fn seen_within(a: &str, b: &str, secs: i64) -> bool {
    use chrono::{DateTime, NaiveDateTime};
    let parse = |s: &str| -> Option<i64> {
        NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S").map(|n| n.and_utc().timestamp())
            .or_else(|_| DateTime::parse_from_rfc3339(s).map(|d| d.timestamp()))
            .ok()
    };
    match (parse(a), parse(b)) {
        (Some(x), Some(y)) => (x - y).abs() <= secs,
        _ => false,
    }
}

/// "Train" a tracked body: bind one or more anonymous `body_*` groups to an
/// existing enrolled person. Retags every row of those groups into the person's
/// `kp_<known_id>` appearance gallery (so their PAST history is absorbed and FUTURE
/// same-outfit sightings resolve to the name), then prunes the gallery to cap.
/// Mirrors the face Train `assign_faces_to_person`. A durable identity needs a face,
/// so we only ever attach a body to an EXISTING known person (not create one).
#[tauri::command]
pub async fn assign_tracked_to_known(
    state: tauri::State<'_, std::sync::Arc<crate::AppState>>,
    body_person_ids: Vec<String>,
    known_id: String,
) -> Result<(), String> {
    // Validate the target exists.
    let exists: Option<(String,)> = sqlx::query_as("SELECT id FROM known_persons WHERE id=?")
        .bind(&known_id).fetch_optional(&state.db).await.map_err(|e| e.to_string())?;
    if exists.is_none() { return Err("unknown person".into()); }
    let kp = format!("kp_{known_id}");
    for bid in &body_person_ids {
        // Don't let a caller re-home a face-anchored or kp_ group by mistake.
        if bid.starts_with("kp_") { continue; }
        let _ = sqlx::query(
            "UPDATE body_embeddings SET person_id=?, known_person_id=? WHERE person_id=?"
        ).bind(&kp).bind(&known_id).bind(bid).execute(&state.db).await;
    }
    // Prune the (now larger) gallery to the most-recent cap.
    let _ = sqlx::query(
        "DELETE FROM body_embeddings WHERE person_id=? AND id NOT IN
           (SELECT id FROM body_embeddings WHERE person_id=? ORDER BY seen_at DESC LIMIT ?)"
    ).bind(&kp).bind(&kp).bind(KP_GALLERY_CAP).execute(&state.db).await;
    // Keep the roster's last-seen honest.
    let _ = sqlx::query("UPDATE known_persons SET last_seen_at=datetime('now') WHERE id=?")
        .bind(&known_id).execute(&state.db).await;
    Ok(())
}

/// Name a body group that has NO enrolled face yet — a "soft" identity. Creates a
/// `known_persons` row with an EMPTY face gallery (`embeddings = "[]"`, the "no face
/// yet" marker), then routes the body groups into its `kp_<id>` gallery via the same
/// path as `assign_tracked_to_known`. The face classifier skips people with too few
/// face embeddings, so soft identities never pollute face recognition; the row
/// auto-upgrades to a durable face identity the moment a face is linked to it.
#[tauri::command]
pub async fn name_tracked_group(
    state: tauri::State<'_, std::sync::Arc<crate::AppState>>,
    body_person_ids: Vec<String>,
    name: String,
    role: String,
) -> Result<String, String> {
    let name = name.trim();
    if name.is_empty() { return Err("name required".into()); }
    let id = uuid::Uuid::new_v4().to_string();
    let now = chrono::Utc::now().to_rfc3339();
    // Empty face gallery = soft (body-only) identity.
    sqlx::query(
        "INSERT INTO known_persons(id, name, role, embeddings, thumbnail, created_at, last_seen_at)
         VALUES(?,?,?,'[]',NULL,?,?)"
    ).bind(&id).bind(name).bind(&role).bind(&now).bind(&now)
     .execute(&state.db).await.map_err(|e| e.to_string())?;

    let kp = format!("kp_{id}");
    // Seed the new person's thumbnail from the first group's latest crop.
    if let Some(bid) = body_person_ids.first() {
        if let Ok(Some(thumb)) = sqlx::query_scalar::<_, String>(
            "SELECT thumbnail_b64 FROM body_embeddings WHERE person_id=? AND thumbnail_b64 IS NOT NULL ORDER BY seen_at DESC LIMIT 1"
        ).bind(bid).fetch_optional(&state.db).await {
            let _ = sqlx::query("UPDATE known_persons SET thumbnail=? WHERE id=?")
                .bind(&thumb).bind(&id).execute(&state.db).await;
        }
    }
    for bid in &body_person_ids {
        if bid.starts_with("kp_") { continue; }
        let _ = sqlx::query("UPDATE body_embeddings SET person_id=?, known_person_id=? WHERE person_id=?")
            .bind(&kp).bind(&id).bind(bid).execute(&state.db).await;
    }
    let _ = sqlx::query(
        "DELETE FROM body_embeddings WHERE person_id=? AND id NOT IN
           (SELECT id FROM body_embeddings WHERE person_id=? ORDER BY seen_at DESC LIMIT ?)"
    ).bind(&kp).bind(&kp).bind(KP_GALLERY_CAP).execute(&state.db).await;
    Ok(id)
}

/// Correction for body grouping: DETACH mis-grouped tracks from their assigned
/// identity — clears `known_person_id` and re-homes each group to a fresh anonymous
/// `body_*` id so it's eligible to be re-clustered/re-assigned correctly.
#[tauri::command]
pub async fn unname_tracked_group(
    state: tauri::State<'_, std::sync::Arc<crate::AppState>>,
    body_person_ids: Vec<String>,
) -> Result<(), String> {
    for bid in &body_person_ids {
        let fresh = format!("body_{}", uuid::Uuid::new_v4());
        let _ = sqlx::query(
            "UPDATE body_embeddings SET person_id=?, known_person_id=NULL WHERE person_id=?"
        ).bind(&fresh).bind(bid).execute(&state.db).await;
    }
    Ok(())
}

/// DURABLE body correction — the body-Re-ID analog of `correct_face`. Given a tracked
/// `body_person_id` that was mis-identified:
///   1. Records its descriptors as HARD NEGATIVES for `wrong_known_id` (if given), so
///      matching / clustering / "looks like X" never re-attribute this appearance to them.
///   2. Routes the track to `correct_known_id` (reassign + prune that person's gallery),
///      or — when no correct person is given — DETACHES it to a fresh anonymous track.
/// This is what makes "Wrong? / Not the same person" stick across future sightings.
#[tauri::command]
pub async fn correct_track(
    state: tauri::State<'_, std::sync::Arc<crate::AppState>>,
    body_person_id: String,
    wrong_known_id: Option<String>,
    correct_known_id: Option<String>,
) -> Result<(), String> {
    // 1. Hard negatives for the WRONG person (deep descriptors only — HSV is too weak to
    //    be a useful negative). Skip a no-op correction to the same person.
    if let Some(wrong) = wrong_known_id.as_deref().filter(|w| !w.is_empty()) {
        if correct_known_id.as_deref() != Some(wrong) {
            let rows: Vec<(Vec<u8>,)> = sqlx::query_as(
                "SELECT descriptor FROM body_embeddings WHERE person_id=? ORDER BY seen_at DESC LIMIT 30"
            ).bind(&body_person_id).fetch_all(&state.db).await.unwrap_or_default();
            for (blob,) in &rows {
                if blob.len() % 4 != 0 { continue; }
                let dim = (blob.len() / 4) as i64;
                if dim < 256 { continue; }
                let _ = sqlx::query(
                    "INSERT INTO body_negatives(id, known_person_id, descriptor, dim) VALUES(?,?,?,?)"
                ).bind(uuid::Uuid::new_v4().to_string()).bind(wrong).bind(blob).bind(dim)
                 .execute(&state.db).await;
            }
        }
    }

    // 2. Reassign to the correct person, or detach to a fresh anonymous track.
    match correct_known_id.as_deref().filter(|c| !c.is_empty()) {
        Some(correct) => {
            let kp = format!("kp_{correct}");
            let _ = sqlx::query("UPDATE body_embeddings SET person_id=?, known_person_id=? WHERE person_id=?")
                .bind(&kp).bind(correct).bind(&body_person_id).execute(&state.db).await;
            let _ = sqlx::query(
                "DELETE FROM body_embeddings WHERE person_id=? AND id NOT IN
                   (SELECT id FROM body_embeddings WHERE person_id=? ORDER BY seen_at DESC LIMIT ?)"
            ).bind(&kp).bind(&kp).bind(KP_GALLERY_CAP).execute(&state.db).await;
            let _ = sqlx::query("UPDATE known_persons SET last_seen_at=datetime('now') WHERE id=?")
                .bind(correct).execute(&state.db).await;
        }
        None => {
            let fresh = format!("body_{}", uuid::Uuid::new_v4());
            let _ = sqlx::query("UPDATE body_embeddings SET person_id=?, known_person_id=NULL WHERE person_id=?")
                .bind(&fresh).bind(&body_person_id).execute(&state.db).await;
        }
    }
    Ok(())
}

// ─── Self-grouping of tracked bodies (propose-then-confirm batch training) ─────

#[derive(serde::Serialize)]
pub struct TrackedCluster {
    pub cluster_id:     String,
    pub member_ids:     Vec<String>,   // the body_* person_ids grouped here
    pub track_count:    i64,
    pub sighting_count: i64,
    pub cameras:        Vec<i64>,
    pub last_seen:      String,
    /// `"@crop"` marker — served via GET /body/{rep_track_id}/crop.
    pub rep_thumbnail:  Option<String>,
    /// The track id whose crop represents this cluster (the crop URL key).
    pub rep_track_id:   Option<String>,
    /// Closest enrolled person by BODY appearance ("looks like X"), for one-tap confirm.
    pub suggested_name:  Option<String>,
    pub suggested_score: Option<f32>,
    /// The suggested person's ID — confirm actions bind to this, not the name.
    pub suggested_person_id: Option<String>,
    /// One representative crop per member track — so the card can EXPAND to show
    /// every photo in the group and the user can deselect any that don't belong
    /// before naming. Keyed by the track's `person_id` (the removal unit).
    pub samples: Vec<TrackSample>,
}

#[derive(serde::Serialize)]
pub struct TrackSample {
    pub person_id: String,  // the body_* track id (deselect this to drop it from the group)
    pub thumbnail: String,  // "@crop" marker — GET /body/:person_id/crop
    pub sightings: i64,
    pub last_seen: String,
}

/// Self-supervised grouping of ANONYMOUS body tracks (edge-AI NVRs' Milvus-cluster
/// idea, done locally): collapse each `body_*` track to a centroid, then greedy-
/// cluster the centroids at the active deep backbone's floor, so many fragmented
/// "Person N" that are really the same person surface as ONE proposed group the user
/// can batch-train in a tap. DEEP descriptors only (≥256-d) — the HSV histogram is
/// too weak to group reliably across tracks; returns empty when no deep Re-ID skill.
/// Propose-only: nothing is merged in the DB until the user confirms.
#[tauri::command]
pub async fn list_tracked_clusters(
    state: tauri::State<'_, std::sync::Arc<crate::AppState>>,
    days: Option<i64>,
) -> Result<Vec<TrackedCluster>, String> {
    let days = days.unwrap_or(7).clamp(1, 90);
    let floor = active_deep_floor(&state.data_dir);

    // Anonymous, deep, recent body sightings. Crops are NOT pulled — only a
    // has-crop flag; the UI fetches crops via GET /body/:track/crop.
    let rows: Vec<(String, Vec<u8>, i64, String, i64)> = sqlx::query_as(
        "SELECT person_id, descriptor, cam_id, seen_at,
                CASE WHEN thumbnail_b64 IS NOT NULL THEN 1 ELSE 0 END
           FROM body_embeddings
          WHERE known_person_id IS NULL AND person_id LIKE 'body\\_%' ESCAPE '\\'
            AND seen_at > datetime('now', ?)
          ORDER BY seen_at DESC LIMIT 3000"
    ).bind(format!("-{days} days")).fetch_all(&state.db).await.map_err(|e| e.to_string())?;

    // Named body galleries (+ hard-negatives) → negative-aware "looks like X" suggestion.
    let galleries = KnownBodyGalleries::load(&state.db).await;

    // Descriptor decode + centroids + the greedy O(tracks²) clustering are all
    // pure CPU over ≤3000 rows — run the whole assembly on the blocking pool so
    // the async runtime (UI IPC) stays responsive while this crunches.
    // ponytail: greedy O(n²) kept — bounded at 3000 rows; swap for knn_graph+CW if track counts grow.
    let mut out: Vec<TrackedCluster> = tokio::task::spawn_blocking(move || {
        // Collapse to one prototype per track (mean of its deep descriptors).
        struct Track { sum: Vec<f32>, n: f32, sightings: i64, cams: Vec<i64>, last: String, has_thumb: bool }
        let mut tracks: std::collections::HashMap<String, Track> = std::collections::HashMap::new();
        for (pid, blob, cam, seen, has_thumb) in rows {
            if blob.len() % 4 != 0 || blob.len() / 4 < 256 { continue; } // deep only
            let v: Vec<f32> = blob.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            let t = tracks.entry(pid).or_insert_with(|| Track {
                sum: vec![0.0; v.len()], n: 0.0, sightings: 0, cams: Vec::new(), last: String::new(), has_thumb: false,
            });
            if t.sum.len() == v.len() { for (s, x) in t.sum.iter_mut().zip(&v) { *s += *x; } t.n += 1.0; }
            t.sightings += 1;
            if !t.cams.contains(&cam) { t.cams.push(cam); }
            if seen > t.last { t.last = seen; }
            if has_thumb != 0 { t.has_thumb = true; }
        }
        // L2-normalised centroid per track.
        let mut units: Vec<(String, Vec<f32>, Track)> = Vec::new();
        for (pid, t) in tracks {
            if t.n < 1.0 { continue; }
            let mut c: Vec<f32> = t.sum.iter().map(|s| s / t.n).collect();
            let norm: f32 = c.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 { c.iter_mut().for_each(|x| *x /= norm); }
            units.push((pid, c, t));
        }

        // Greedy-cluster track centroids at the deep floor.
        struct Cl { centroid: Vec<f32>, n: f32, members: Vec<(String, Track)> }
        let mut clusters: Vec<Cl> = Vec::new();
        for (pid, c, t) in units {
            let (mut best, mut bi) = (-1.0f32, None);
            for (i, cl) in clusters.iter().enumerate() {
                if cl.centroid.len() != c.len() { continue; }
                let s = cosine_sim(&c, &cl.centroid);
                if s > best { best = s; bi = Some(i); }
            }
            match bi {
                Some(i) if best >= floor => {
                    let cl = &mut clusters[i];
                    for (k, x) in cl.centroid.iter_mut().enumerate() { *x = (*x * cl.n + c[k]) / (cl.n + 1.0); }
                    cl.n += 1.0;
                    cl.members.push((pid, t));
                }
                _ => clusters.push(Cl { centroid: c, n: 1.0, members: vec![(pid, t)] }),
            }
        }

        // Only groups worth confirming (≥2 tracks merged — a single track isn't a "group").
        clusters.into_iter().enumerate().filter_map(|(idx, cl)| {
            if cl.members.len() < 2 { return None; }
            let (suggested_person_id, suggested_name, suggested_score) = galleries.suggest(&cl.centroid, floor);
            let mut cameras: Vec<i64> = cl.members.iter().flat_map(|(_, t)| t.cams.clone()).collect();
            cameras.sort_unstable(); cameras.dedup();
            let sighting_count: i64 = cl.members.iter().map(|(_, t)| t.sightings).sum();
            let last_seen = cl.members.iter().map(|(_, t)| t.last.clone()).max().unwrap_or_default();
            // Representative = the most-recent member WITH a crop ('@crop' marker;
            // the URL is keyed by the member's track id).
            let rep_track_id = {
                let mut with_thumb: Vec<&(String, Track)> = cl.members.iter().filter(|(_, t)| t.has_thumb).collect();
                with_thumb.sort_by(|a, b| b.1.last.cmp(&a.1.last));
                with_thumb.first().map(|(p, _)| p.clone())
            };
            // One crop per member track (newest first), so the card expands to show the
            // group's photos and the user can deselect outliers. Capped to bound payload.
            let mut member_sorted: Vec<&(String, Track)> = cl.members.iter().collect();
            member_sorted.sort_by(|a, b| b.1.last.cmp(&a.1.last));
            let samples: Vec<TrackSample> = member_sorted.iter().filter(|(_, t)| t.has_thumb).map(|(pid, t)| {
                TrackSample {
                    person_id: pid.clone(), thumbnail: "@crop".into(),
                    sightings: t.sightings, last_seen: t.last.clone(),
                }
            }).take(40).collect();
            Some(TrackedCluster {
                cluster_id: format!("tc{idx}"),
                member_ids: cl.members.iter().map(|(p, _)| p.clone()).collect(),
                track_count: cl.members.len() as i64,
                sighting_count,
                cameras,
                last_seen,
                rep_thumbnail: rep_track_id.as_ref().map(|_| "@crop".to_string()),
                rep_track_id,
                suggested_name,
                suggested_score,
                suggested_person_id,
                samples,
            })
        }).collect::<Vec<_>>()
    }).await.map_err(|e| e.to_string())?;
    // Active-learning order: most-confident "looks like X" first (one-tap wins), then recency.
    out.sort_by(|a, b| b.suggested_score.partial_cmp(&a.suggested_score)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then(b.last_seen.cmp(&a.last_seen)));
    Ok(out)
}

// ─── Cross-camera tracked persons (body Re-ID) — read-only People view ─────────

#[derive(serde::Serialize)]
pub struct TrackedPerson {
    pub person_id:     String,
    pub label:         String,        // friendly "Person N" (stable, by first-seen order)
    pub sighting_count: i64,
    pub cameras:       Vec<i64>,
    pub first_seen:    String,
    pub last_seen:     String,
    pub thumbnail:     Option<String>,
    /// Resolved known-person NAME when this body was auto-labelled from a face
    /// (face↔body fusion). `None` = anonymous appearance track ("Person N").
    pub known_name:    Option<String>,
    /// The known person's ID — corrections bind to this, never the name string.
    pub known_person_id: Option<String>,
    /// For anonymous tracks: closest enrolled person by BODY appearance ("looks like
    /// Ravi"), so the user can confirm/train in one tap. `None` if nothing close.
    pub suggested_name: Option<String>,
    /// The suggested person's ID — confirm actions bind to this, not the name.
    pub suggested_person_id: Option<String>,
    /// Human-readable clothing line majority-voted over the track's sightings —
    /// e.g. "blue top · black bottom". None when no sighting had confident colors.
    pub outfit: Option<String>,
}

/// Majority-vote clothing attr JSONs (newest-first, ≤40) into one outfit line.
/// Top and bottom are voted INDEPENDENTLY (one occluded band shouldn't drop
/// the other). Pure CPU — callers batch-fetch the rows.
/// Majority colour per band across observations, with the winning vote count.
///
/// Returns `(top, bottom)`, each `Option<(colour, votes)>`. The caller decides
/// what confidence it needs: the People list is happy with one observation, an
/// event attribute written to the database is not — a single frame's colour is
/// noisy enough that lighting alone can flip red to brown.
pub(crate) fn vote_outfit_pair<'a>(
    jsons: impl Iterator<Item = &'a str>,
) -> (Option<(String, u32)>, Option<(String, u32)>) {
    let mut tops:    std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    let mut bottoms: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    for json in jsons.take(40) {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(json) else { continue };
        if let Some(t) = v.get("top").and_then(|x| x.as_str())    { *tops.entry(t.to_string()).or_default() += 1; }
        if let Some(b) = v.get("bottom").and_then(|x| x.as_str()) { *bottoms.entry(b.to_string()).or_default() += 1; }
    }
    (tops.into_iter().max_by_key(|(_, n)| *n),
     bottoms.into_iter().max_by_key(|(_, n)| *n))
}

/// Clothing colours for one event's dominant person, voted across the frames
/// available. Returns the JSON stored in `motion_events.outfit`, or `None`.
///
/// `frames` are decoded JPEG bytes — the thumbnail plus whatever strobe frames
/// the caller has. More frames means a stronger vote; one frame (the backfill
/// case) is accepted on its own because `dominant_color_in_band` already abstains
/// below a 0.25 plurality share, so a lone observation is not a coin flip. But a
/// colour seen once in poor light is still the weakest thing here — which is why
/// `retrieve`'s refinement loop drops the outfit filter first.
///
/// `None` is written as SQL NULL and must be read as UNKNOWN, never "not red".
pub(crate) fn outfit_for_event(frames: &[Vec<u8>], detections_json: &str) -> Option<String> {
    #[derive(serde::Deserialize)]
    struct Bx { xmin: f64, ymin: f64, xmax: f64, ymax: f64 }
    #[derive(serde::Deserialize)]
    struct Det {
        label: String,
        #[serde(rename = "box", alias = "bbox")]
        bx: Option<Bx>,
    }

    let dets: Vec<Det> = serde_json::from_str(detections_json).ok()?;
    // The largest person box — the subject of the event, same rule the vehicle
    // path uses for choosing which car to read a plate from.
    let person = dets.iter()
        .filter(|d| d.label.eq_ignore_ascii_case("person"))
        .filter_map(|d| d.bx.as_ref())
        .max_by(|a, b| {
            let area = |x: &Bx| (x.xmax - x.xmin).max(0.0) * (x.ymax - x.ymin).max(0.0);
            area(a).partial_cmp(&area(b)).unwrap_or(std::cmp::Ordering::Equal)
        })?;

    let observations: Vec<String> = frames.iter().take(4).filter_map(|jpeg| {
        let img = image::load_from_memory(jpeg).ok()?;
        // Boxes are stored normalised 0..1; the classifier wants pixels.
        let (iw, ih) = (img.width() as f64, img.height() as f64);
        let bbox = [(person.xmin * iw) as f32, (person.ymin * ih) as f32,
                    (person.xmax * iw) as f32, (person.ymax * ih) as f32];
        classify_person_colors(jpeg, &bbox)
    }).collect();
    if observations.is_empty() { return None; }

    let need = if observations.len() > 1 { 2 } else { 1 };
    let (top, bottom) = vote_outfit_pair(observations.iter().map(String::as_str));
    let mut o = serde_json::Map::new();
    if let Some((c, n)) = top    { if n >= need { o.insert("top".into(), c.into()); } }
    if let Some((c, n)) = bottom { if n >= need { o.insert("bottom".into(), c.into()); } }
    if o.is_empty() { return None; }
    Some(serde_json::Value::Object(o).to_string())
}

/// Human phrasing of the vote, for the People → Tracked list.
fn vote_outfit<'a>(jsons: impl Iterator<Item = &'a str>) -> Option<String> {
    match vote_outfit_pair(jsons) {
        (Some((t, _)), Some((b, _))) => Some(format!("{t} top · {b} bottom")),
        (Some((t, _)), None)         => Some(format!("{t} top")),
        (None, Some((b, _)))         => Some(format!("{b} bottom")),
        (None, None)                 => None,
    }
}

/// List body-Re-ID tracked persons worth showing — those with ≥2 sightings OR
/// seen on ≥2 cameras (drops the single-shot noise body Re-ID inevitably
/// creates). Numbered "Person N" by first-seen order (stable across refreshes);
/// returned newest-activity first. Appearance-based (HSV histogram) — a soft,
/// best-within-a-day cross-camera signal, complementary to face recognition.
#[tauri::command]
pub async fn list_tracked_persons(
    state: tauri::State<'_, std::sync::Arc<crate::AppState>>,
) -> Result<Vec<TrackedPerson>, String> {
    // NAMED (face-anchored) people are always shown; ANONYMOUS fragments only when
    // still appearance-valid (last seen ≤ 2 days) — otherwise the list is hundreds of
    // dead "Person N" the same person re-minted daily. Anonymous are capped below.
    //
    // This used to be an N+1 storm (4-5 awaited queries PER ROW × hundreds of
    // rows) that froze the People tab under load — now the same output comes
    // from 5 total queries + in-memory joins. The known name rides the main
    // query as a correlated scalar subquery on the aggregated known_id.
    // The name lookup joins on the AGGREGATED known_id, so the aggregation must
    // finish in an inner query first — SQLite rejects an outer aggregate inside
    // a correlated subquery's WHERE ("misuse of aggregate function MAX()"),
    // which blanked the whole Tracked view.
    let rows: Vec<(String, i64, String, String, Option<String>, Option<String>, Option<String>)> = sqlx::query_as(
        "SELECT t.person_id, t.sightings, t.first_seen, t.last_seen, t.cams, t.known_id,
                (SELECT name FROM known_persons k WHERE k.id = t.known_id) AS known_name
         FROM (
            SELECT person_id, COUNT(*) AS sightings, MIN(seen_at) AS first_seen,
                   MAX(seen_at) AS last_seen, GROUP_CONCAT(DISTINCT cam_id) AS cams,
                   MAX(known_person_id) AS known_id
              FROM body_embeddings
             GROUP BY person_id
            HAVING (COUNT(*) >= 2 OR COUNT(DISTINCT cam_id) >= 2)
               AND (MAX(known_person_id) IS NOT NULL OR MAX(seen_at) > datetime('now','-2 days'))
         ) t
         ORDER BY t.first_seen ASC"
    ).fetch_all(&state.db).await.map_err(|e| e.to_string())?;

    // Batch: which tracks have ANY crop to serve (body crop, or event-thumbnail
    // fallback — GET /body/:track/crop resolves the same preference order).
    let mut has_thumb: std::collections::HashSet<String> = sqlx::query_scalar::<_, String>(
        "SELECT DISTINCT person_id FROM body_embeddings WHERE thumbnail_b64 IS NOT NULL"
    ).fetch_all(&state.db).await.unwrap_or_default().into_iter().collect();
    for pid in sqlx::query_scalar::<_, String>(
        "SELECT DISTINCT b.person_id FROM body_embeddings b
          JOIN motion_events m ON m.id = b.event_id
         WHERE m.thumbnail IS NOT NULL"
    ).fetch_all(&state.db).await.unwrap_or_default() { has_thumb.insert(pid); }

    // Batch: latest deep descriptor per track (SQLite bare-column-with-MAX picks
    // the MAX(seen_at) row's descriptor — SQLite-specific semantics, fine here).
    let latest_desc: std::collections::HashMap<String, Vec<u8>> = sqlx::query_as::<_, (String, Vec<u8>, String)>(
        "SELECT person_id, descriptor, MAX(seen_at) FROM body_embeddings GROUP BY person_id"
    ).fetch_all(&state.db).await.unwrap_or_default()
     .into_iter().map(|(pid, d, _)| (pid, d)).collect();

    // Batch: clothing attrs newest-first; vote_outfit caps at 40 per track.
    let mut outfit_rows: std::collections::HashMap<String, Vec<String>> = std::collections::HashMap::new();
    for (pid, attrs) in sqlx::query_as::<_, (String, String)>(
        "SELECT person_id, attrs FROM body_embeddings
          WHERE attrs IS NOT NULL ORDER BY seen_at DESC"
    ).fetch_all(&state.db).await.unwrap_or_default() {
        let v = outfit_rows.entry(pid).or_default();
        if v.len() < 40 { v.push(attrs); }
    }

    // Named people's BODY galleries (deep only) → "looks like X" suggestion for
    // anonymous tracks. Loaded once; matched by top-3 mean cosine ≥ 0.6.
    // Named galleries + hard-negatives (negative-aware "looks like X").
    let galleries = KnownBodyGalleries::load(&state.db).await;
    let suggest_floor = active_deep_floor(&state.data_dir);

    let mut out: Vec<TrackedPerson> = Vec::with_capacity(rows.len());
    for (i, (person_id, sightings, first_seen, last_seen, cams, known_id, known_name)) in rows.into_iter().enumerate() {
        let cameras: Vec<i64> = cams.unwrap_or_default()
            .split(',')
            .filter_map(|s| s.trim().parse::<i64>().ok())
            .collect();
        // '@crop' marker — the actual bytes are served by GET /body/:track/crop
        // (body crop preferred, event thumbnail fallback), Chromium-cacheable.
        let thumbnail: Option<String> =
            has_thumb.contains(&person_id).then(|| "@crop".to_string());

        // "Looks like X" for anonymous tracks: match this body's latest deep descriptor
        // against the named galleries (top-3 mean), SUPPRESSING any person the user has
        // marked as a hard-negative for this appearance.
        let (suggested_person_id, suggested_name): (Option<String>, Option<String>) =
            if known_name.is_none() && !galleries.is_empty() {
                latest_desc.get(&person_id).and_then(|blob| {
                    if blob.len() % 4 != 0 || blob.len() / 4 < 256 { return None; }
                    let v: Vec<f32> = blob.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                    let (sid, sname, _) = galleries.suggest(&v, suggest_floor);
                    Some((sid, sname))
                }).unwrap_or((None, None))
            } else { (None, None) };

        // Outfit line: majority-vote the track's per-sighting clothing attrs
        // (top/bottom voted independently — a shadowed lower band on one frame
        // mustn't flip the whole outfit). "blue top · black bottom".
        let outfit = outfit_rows.get(&person_id)
            .and_then(|rows| vote_outfit(rows.iter().map(|s| s.as_str())));

        out.push(TrackedPerson {
            person_id,
            label: format!("Person {}", i + 1),
            sighting_count: sightings,
            cameras,
            first_seen,
            last_seen,
            thumbnail,
            known_name,
            known_person_id: known_id.filter(|k| !k.is_empty()),
            suggested_name,
            suggested_person_id,
            outfit,
        });
    }
    // Display newest-activity first; the "Person N" numbering stays fixed by first-seen.
    out.sort_by(|a, b| b.last_seen.cmp(&a.last_seen));
    // Keep ALL named (face-anchored) people, but cap anonymous fragments to the most
    // recent 60 so the Tracked tab can't be flooded between prunes.
    const ANON_CAP: usize = 60;
    let mut anon_kept = 0usize;
    out.retain(|p| {
        if p.known_name.is_some() { return true; }
        anon_kept += 1;
        anon_kept <= ANON_CAP
    });
    Ok(out)
}


#[cfg(test)]
mod clothing_tests {
    use super::*;

    /// Synthetic person crop: blue "shirt" band over black "pants" band on a
    /// gray background, person bbox in the middle of the frame.
    fn synthetic_person() -> (Vec<u8>, [f32; 4]) {
        let (w, h) = (200u32, 400u32);
        let mut img = image::RgbImage::from_pixel(w, h, image::Rgb([120, 120, 120]));
        // Person bbox: x 40..160, y 40..360 (head 40..100, torso 100..210, legs 210..340)
        for y in 100..210 { for x in 45..155 { img.put_pixel(x, y, image::Rgb([20, 60, 200])); } }   // blue shirt
        for y in 210..340 { for x in 45..155 { img.put_pixel(x, y, image::Rgb([15, 15, 18])); } }    // black pants
        let mut jpeg = Vec::new();
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg, 92)
            .encode_image(&image::DynamicImage::ImageRgb8(img)).unwrap();
        (jpeg, [40.0, 40.0, 160.0, 360.0])
    }

    /// The event-level bridge: one frame is enough for the backfill, and the
    /// result is the JSON `motion_events.outfit` stores.
    #[test]
    fn outfit_for_event_reads_the_largest_person() {
        let (jpeg, _) = synthetic_person();
        // Boxes are stored NORMALISED 0..1, which is what the caller must handle.
        let dets = r#"[
            {"label":"person","score":0.9,"box":{"xmin":0.2,"ymin":0.1,"xmax":0.8,"ymax":0.9}},
            {"label":"person","score":0.8,"box":{"xmin":0.0,"ymin":0.0,"xmax":0.05,"ymax":0.05}}
        ]"#;
        let out = outfit_for_event(&[jpeg], dets).expect("outfit");
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["top"], "blue");
        assert_eq!(v["bottom"], "black");
    }

    /// No person, no outfit — and an event with no detections must not panic.
    #[test]
    fn outfit_for_event_abstains_without_a_person() {
        let (jpeg, _) = synthetic_person();
        let cars = r#"[{"label":"car","score":0.9,"box":{"xmin":0.2,"ymin":0.1,"xmax":0.8,"ymax":0.9}}]"#;
        assert!(outfit_for_event(&[jpeg.clone()], cars).is_none());
        assert!(outfit_for_event(&[jpeg], "not json").is_none());
        assert!(outfit_for_event(&[], "[]").is_none());
    }

    /// With several frames the vote must agree, so one bad frame cannot name the
    /// colour. A single frame is accepted on its own — the backfill has no others.
    #[test]
    fn multi_frame_needs_agreement() {
        let three = ["{\"top\":\"red\"}", "{\"top\":\"red\"}", "{\"top\":\"blue\"}"];
        let (top, _) = vote_outfit_pair(three.into_iter());
        assert_eq!(top, Some(("red".to_string(), 2)));

        let (t, b) = vote_outfit_pair(["{\"bottom\":\"black\"}"].into_iter());
        assert_eq!(t, None);
        assert_eq!(b, Some(("black".to_string(), 1)));
    }

    #[test]
    fn blue_shirt_black_pants_classified() {
        let (jpeg, bbox) = synthetic_person();
        let attrs = classify_person_colors(&jpeg, &bbox).expect("colors detected");
        let v: serde_json::Value = serde_json::from_str(&attrs).unwrap();
        assert_eq!(v.get("top").and_then(|x| x.as_str()), Some("blue"), "attrs={attrs}");
        assert_eq!(v.get("bottom").and_then(|x| x.as_str()), Some("black"), "attrs={attrs}");
    }

    #[test]
    fn tiny_crop_abstains() {
        let img = image::RgbImage::from_pixel(30, 30, image::Rgb([200, 30, 30]));
        let mut jpeg = Vec::new();
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg, 90)
            .encode_image(&image::DynamicImage::ImageRgb8(img)).unwrap();
        assert!(classify_person_colors(&jpeg, &[0.0, 0.0, 20.0, 20.0]).is_none());
    }
}
