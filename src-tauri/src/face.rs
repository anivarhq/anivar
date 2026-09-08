//! Native face detection + recognition pipeline (standard).
//!
//! Two-stage:
//!   1. **Detector** — `yolov8n-face.onnx` (~7 MB). YOLO-style head augmented with 5
//!      facial landmarks per box: eyes, nose, mouth corners. Output `[1, 20, N]`
//!      where channels = [cx, cy, w, h, conf, lm1_x, lm1_y, lm1_score, ...,
//!      lm5_x, lm5_y, lm5_score].
//!   2. **Embedder** — both tiers are 512-d ArcFace ResNet100 (small = INT8
//!      quantised, large = FP32). The dimension is uniform across tiers so stored
//!      embeddings stay comparable when the user swaps tiers (see `embed_dim`).
//!      Input 1×3×112×112 RGB normalised to `[-1, 1]` (`(p - 127.5)/128`).
//!      Output is L2-normalised so cosine similarity = dot product.
//!
//! Thresholds copied from common defaults: detection 0.7, recognition 0.9,
//! unknown 0.8.
//!
//! Models are downloaded via the existing skill mechanism into
//! `<data_dir>/skills/face_small/` or `…/face_large/` as `detector.onnx` +
//! `embedder.onnx`. Lazy-loaded on first call to `try_load`.

use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

use ort::session::Session as OrtSession;
use sqlx::SqlitePool;
use tauri::State;

use crate::AppState;

/// Process-wide cache of the currently-loaded face model pair, keyed by `FaceModelSize`.
/// Built lazily on first use of `recognize_faces`. `std::sync::Mutex` is fine here:
/// callers release the guard before doing any `.await`.
static FACE_MODELS: OnceLock<Mutex<Option<FaceModels>>> = OnceLock::new();

/// Minimum face bbox area (px²) before recognition runs — mature NVRs' `min_area`
/// default. Tiny/distant faces yield garbage embeddings, so we skip them. ~22×22.
const MIN_FACE_AREA_PX: f32 = 500.0;
/// Higher bar to STORE a crop as a People/Train training candidate (~60×60).
/// Recognition still runs at MIN_FACE_AREA_PX (so distant enrolled faces are still
/// named), but we don't flood the Train tab with tiny, unidentifiable crops.
const MIN_STORE_FACE_AREA: f32 = 3600.0;

// ─── Public types ──────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaceModelSize { Small, Large }

impl FaceModelSize {
    pub fn from_setting(s: &str) -> Option<Self> {
        match s {
            "small" => Some(Self::Small),
            "large" => Some(Self::Large),
            _       => None,
        }
    }
    pub fn skill_id(self) -> &'static str {
        match self { Self::Small => "face_small", Self::Large => "face_large" }
    }
    /// Both tiers use a 512-d ArcFace-family embedder (INT8 quantised vs FP32).
    /// Keeping the dimension uniform means we can swap tiers without rebuilding
    /// stored embeddings — only accuracy/speed change.
    pub fn embed_dim(self) -> usize { 512 }
}

/// Is this tier's model actually on disk (BOTH onnx files present, not just the dir)?
fn tier_installed(data_dir: &Path, size: FaceModelSize) -> bool {
    let dir = data_dir.join("skills").join(size.skill_id());
    dir.join("detector.onnx").exists() && dir.join("embedder.onnx").exists()
}

/// The face tier to actually use. Prefers the configured `face_model` tier when
/// its files exist, otherwise falls back to whichever tier IS installed (Large
/// preferred for accuracy). Returns `None` only when neither tier is installed.
///
/// Critically: an installed model is treated as ENABLED. `face_model` defaults to
/// "off", which previously disabled everything even after the user installed a
/// model — so enrollment, live, and event recognition all silently did nothing.
pub fn active_face_tier(face_model: &str, data_dir: &Path) -> Option<FaceModelSize> {
    if let Some(t) = FaceModelSize::from_setting(face_model) {
        if tier_installed(data_dir, t) { return Some(t); }
    }
    if tier_installed(data_dir, FaceModelSize::Large) { return Some(FaceModelSize::Large); }
    if tier_installed(data_dir, FaceModelSize::Small) { return Some(FaceModelSize::Small); }
    None
}

#[derive(Clone, Debug)]
pub struct FaceDetection {
    pub bbox:      [f32; 4],         // x1, y1, x2, y2 (image coords)
    pub score:     f32,
    pub landmarks: [[f32; 2]; 5],    // left eye, right eye, nose, mouth-left, mouth-right
}

#[derive(Clone, Debug)]
pub struct FaceEmbedding {
    pub vector:  Vec<f32>,           // L2-normalised; len = 512 (both tiers, ArcFace ResNet100)
    pub quality: f32,                // 0..1, 1 = sharp (Laplacian variance normalised)
    /// JPEG (quality 85) of the aligned 112×112 crop, base64-encoded.
    /// Used by the Persons "Train" UI to show the user what was seen.
    pub thumbnail_b64: String,
}

#[derive(Clone, Debug)]
pub struct FaceMatch {
    pub person_id: String,
    pub name:      String,
    pub score:     f32,              // cosine similarity, higher = better match
    /// WHICH recognizer produced this decision — persisted as provenance so a
    /// wrong name is traceable: "face_cosine" | "face_classifier" | "unknown".
    pub method:    &'static str,
    /// Gap to the runner-up PERSON's best cosine (look-alike margin). None when
    /// there was no second candidate to compare against.
    pub margin:    Option<f32>,
}

pub struct FaceModels {
    detector: OrtSession,
    embedder: OrtSession,
    size:     FaceModelSize,
    det_input_name: String,
    emb_input_name: String,
    /// True when the embedder wants NHWC `[1,112,112,3]` input (e.g. the Large
    /// `arc.onnx`); false for standard NCHW `[1,3,112,112]` (the Small model).
    emb_nhwc: bool,
}

impl FaceModels {
    /// Try loading the configured model pair from `<data_dir>/skills/<skill_id>/`.
    /// Looks for `detector.onnx` + `embedder.onnx` — caller is responsible for
    /// downloading them via the skill system.
    pub fn try_load(data_dir: &Path, size: FaceModelSize) -> anyhow::Result<Self> {
        let dir = data_dir.join("skills").join(size.skill_id());
        let det_path = dir.join("detector.onnx");
        let emb_path = dir.join("embedder.onnx");
        if !det_path.exists() {
            anyhow::bail!("face detector model not found at {}", det_path.display());
        }
        if !emb_path.exists() {
            anyhow::bail!("face embedder model not found at {}", emb_path.display());
        }
        // ENRICHMENT LANE = CPU (mature NVRs' model). Two reasons this beats the
        // GPU: (1) it keeps the realtime GPU lock YOLO-only (no starvation);
        // (2) putting these on TensorRT triggers a per-model ENGINE BUILD on
        // first detection that holds the shared GPU lock for MINUTES, wedging
        // YOLO (observed: yolo n=1 over a whole soak). Event-cadence CPU
        // inference (~face 250ms) is invisible; the recorder/postprocessor is
        // the thing that must never starve, and it lives on CPU too — validated
        // with a realistic camera count (synthetic local encoders inflate CPU).
        let detector = crate::inference::build_ort_session_cpu(&det_path)?;
        let mut embedder = crate::inference::build_ort_session_cpu(&emb_path)?;
        let det_input_name = detector.inputs().first()
            .map(|i| i.name().to_string()).unwrap_or_else(|| "images".to_string());
        let emb_input_name = embedder.inputs().first()
            .map(|i| i.name().to_string()).unwrap_or_else(|| "input".to_string());

        // Probe the embedder's expected input layout. ArcFace exports disagree:
        // the ONNX-model-zoo ResNet100 is NCHW [1,3,112,112], but garavv/arc.onnx
        // (our Large tier) is NHWC [1,112,112,3]. Try a dummy NCHW forward — if the
        // run errors (fixed-shape mismatch), the model wants NHWC. Decisive because
        // ArcFace input dims are fixed.
        let emb_nhwc = {
            let dummy = vec![0.0f32; 3 * 112 * 112];
            match ort::value::Tensor::<f32>::from_array(([1usize, 3, 112, 112], dummy)) {
                Ok(t) => { let _t = crate::inference::infer_timer("face_align");
                    embedder.run(ort::inputs![emb_input_name.as_str() => t]).is_err() }
                Err(_) => false,
            }
        };
        tracing::info!(
            "Face models loaded ({:?}) — det_in={} emb_in={} emb_layout={}",
            size, det_input_name, emb_input_name, if emb_nhwc { "NHWC" } else { "NCHW" }
        );
        Ok(Self { detector, embedder, size, det_input_name, emb_input_name, emb_nhwc })
    }

    pub fn size(&self) -> FaceModelSize { self.size }

    /// Detect faces in a JPEG. Returns boxes in **original image coordinates**.
    /// `min_conf` is the per-face confidence gate (mature NVRs' `detection_threshold`).
    pub fn detect(&mut self, jpeg: &[u8], min_conf: f32) -> anyhow::Result<Vec<FaceDetection>> {
        // Detect liberally — clamp the gate so a high setting (mature NVRs' 0.7, for
        // clean IP-cam frames) can't starve full-frame webcam detection. Identity
        // is still confirmed strictly by the recognition threshold downstream.
        let min_conf = min_conf.min(0.5);

        // ── Preprocess: LETTERBOX-resize to 640×640 (preserve aspect, pad gray
        // 114) RGB CHW [0,1] — exactly how YOLOv8-face was trained. Stretching
        // (the old resize_exact) distorts the face and collapses confidence. ───
        let img = image::load_from_memory(jpeg)?.to_rgb8();
        let (orig_w, orig_h) = (img.width() as f32, img.height() as f32);
        let scale = (640.0 / orig_w).min(640.0 / orig_h);
        let new_w = ((orig_w * scale).round() as u32).clamp(1, 640);
        let new_h = ((orig_h * scale).round() as u32).clamp(1, 640);
        let pad_x = ((640 - new_w) / 2) as f32;
        let pad_y = ((640 - new_h) / 2) as f32;
        let resized = image::imageops::resize(&img, new_w, new_h, image::imageops::FilterType::Triangle);

        // Gray-114 canvas, paste the resized frame centred (the pad stays gray).
        let mut data = vec![114.0f32 / 255.0; 3 * 640 * 640];
        let bytes = resized.as_raw();
        let (rw, rh) = (new_w as usize, new_h as usize);
        let (ox, oy) = (pad_x as usize, pad_y as usize);
        for y in 0..rh {
            for x in 0..rw {
                let s = (y * rw + x) * 3;
                let d = (oy + y) * 640 + (ox + x);
                data[d]                   = bytes[s    ] as f32 / 255.0;
                data[640 * 640 + d]       = bytes[s + 1] as f32 / 255.0;
                data[640 * 640 * 2 + d]   = bytes[s + 2] as f32 / 255.0;
            }
        }
        let tensor = ort::value::Tensor::<f32>::from_array(([1usize, 3, 640, 640], data))
            .map_err(|e| anyhow::anyhow!("face det tensor: {e}"))?;
        let ort_inputs = ort::inputs![self.det_input_name.as_str() => tensor];
        let outputs = { let _t = crate::inference::infer_timer("face_detect"); self.detector.run(ort_inputs) }
            .map_err(|e| anyhow::anyhow!("face det run: {e}"))?;
        let (shape, raw): (Vec<i64>, Vec<f32>) = match outputs[0].try_extract_tensor::<f32>() {
            Ok((s, d)) => (s.iter().copied().collect(), d.to_vec()),
            Err(e) => anyhow::bail!("face det extract: {e}"),
        };

        // YOLO-face output is [1, C, N] (channels-first) or [1, N, C] (channels-last).
        // C is the SMALL dim (a few), N the big one (thousands of anchors). Two real
        // export variants exist and BOTH must work:
        //   • C = 5  → cx, cy, w, h, conf  (detection-only; e.g. deepghs/yolo-face)
        //   • C ≥ 15 → + 5 × (x, y, score) keypoints (pose/landmark export)
        // The old code only recognised C == 20, so a 5-channel model was transposed
        // and decoded as garbage → 0 faces (the "No face" enrollment bug).
        let (channels, n, channels_first) = if shape.len() == 3 {
            let a = shape[1] as usize;
            let b = shape[2] as usize;
            if a <= b { (a, b, true) } else { (b, a, false) }
        } else {
            anyhow::bail!("unexpected face-det output shape: {:?}", shape);
        };
        let has_landmarks = channels >= 15;

        let get = |row: usize, col: usize| -> f32 {
            if channels_first { raw[row * n + col] } else { raw[col * channels + row] }
        };

        // ── Decode + NMS — un-letterbox: subtract pad, divide by scale ────────
        let mut cand: Vec<FaceDetection> = Vec::new();
        for i in 0..n {
            let conf = get(4, i);
            if conf < min_conf { continue; }
            let cx = (get(0, i) - pad_x) / scale;
            let cy = (get(1, i) - pad_y) / scale;
            let w  = get(2, i) / scale;
            let h  = get(3, i) / scale;
            let mut lm = [[0.0f32; 2]; 5];
            if has_landmarks {
                for k in 0..5 {
                    lm[k][0] = (get(5 + k * 3,     i) - pad_x) / scale;
                    lm[k][1] = (get(5 + k * 3 + 1, i) - pad_y) / scale;
                }
            }
            // Landmark-less detectors leave `lm` zeroed → align_and_embed falls back
            // to a square bbox crop instead of a 5-point warp.
            cand.push(FaceDetection {
                bbox: [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0],
                score: conf,
                landmarks: lm,
            });
        }
        let kept = nms(cand, 0.45);
        // Diagnostic: if this logs `0 faces` with a non-20-channel shape, the
        // detector model's output layout differs from what we parse.
        tracing::debug!(
            "face detect: out shape {:?} (channels={channels}, n={n}) → {} face(s)",
            shape, kept.len()
        );
        Ok(kept)
    }

    /// Align a face crop using 5-point landmarks and run the embedder.
    /// Returns an L2-normalised vector of length `embed_dim()`.
    pub fn align_and_embed(&mut self, jpeg: &[u8], det: &FaceDetection) -> anyhow::Result<FaceEmbedding> {
        let img = image::load_from_memory(jpeg)?.to_rgb8();
        // Prefer a 5-point landmark warp (best for ArcFace). When the detector has
        // no landmarks (5-channel export → `det.landmarks` all zero), fall back to a
        // centred square bbox crop. ArcFace is trained on aligned faces, so the crop
        // path is slightly less accurate — but it's self-consistent (enroll + match
        // use the same crop), so people still cluster + recognise.
        let has_lm = det.landmarks.iter().any(|p| p[0] != 0.0 || p[1] != 0.0);
        let aligned = if has_lm {
            warp_face_to_112(&img, &det.landmarks)
        } else {
            bbox_crop_112(&img, &det.bbox)
        };

        // Quality: Laplacian variance of the ALIGNED crop → [0,1]. The crop is a
        // bilinear warp (interpolation damps high-frequency detail), so its laplacian
        // runs ~5× lower than a raw image — normalise by 300, not 1500, or real
        // webcam faces score ~0.1 and get rejected by the enroll/quality gates.
        let quality = laplacian_variance(&aligned).min(300.0) / 300.0;

        // DISPLAY thumbnail = a natural, padded bounding-box crop of the face (what a
        // human recognises — forehead to chin with margin), NOT the tight aligned
        // warp. The warp is the EMBEDDER's input and looks distorted/unrecognisable
        // to a person, which is why Train cards looked "broken". Fall back to the
        // aligned crop only if the bbox is degenerate.
        let thumbnail_b64 = face_display_crop(&img, &det.bbox).unwrap_or_else(|| {
            let mut j = Vec::with_capacity(4096);
            let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut j, 85);
            enc.encode_image(&aligned).ok();
            base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &j)
        });

        // Run the embedder in its expected layout (NCHW or NHWC). Some ArcFace
        // exports are NHWC (e.g. the Large arc.onnx); feeding the wrong layout
        // makes ORT reject the shape — which previously surfaced as "No face".
        // Bulletproof: on a run error, retry the OTHER layout + remember it.
        let want = self.emb_nhwc;
        let name = self.emb_input_name.clone();
        let raw = match run_embedder(&mut self.embedder, &name, &aligned, want) {
            Ok(r) => r,
            Err(_) => {
                let r = run_embedder(&mut self.embedder, &name, &aligned, !want)?;
                self.emb_nhwc = !want;
                tracing::info!("face embedder layout corrected → {}", if !want { "NHWC" } else { "NCHW" });
                r
            }
        };
        if raw.is_empty() { anyhow::bail!("face emb produced an empty vector"); }
        if raw.len() != self.size.embed_dim() {
            tracing::warn!("face emb dim {} (expected {}) — using as-is", raw.len(), self.size.embed_dim());
        }
        // L2-normalise so cosine similarity = dot product.
        let mut v = raw;
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if n > 0.0 { v.iter_mut().for_each(|x| *x /= n); }
        Ok(FaceEmbedding { vector: v, quality, thumbnail_b64 })
    }
}

/// Fallback when the detector has no landmarks: a centred SQUARE crop of the face
/// box (with ~25% margin for forehead/chin) resized to 112×112. ArcFace prefers a
/// landmark-aligned warp, but a consistent square crop still yields usable,
/// self-consistent embeddings (the same crop is used to enroll and to match).
fn bbox_crop_112(img: &image::RgbImage, bbox: &[f32; 4]) -> image::RgbImage {
    let (iw, ih) = (img.width() as f32, img.height() as f32);
    let cx = (bbox[0] + bbox[2]) / 2.0;
    let cy = (bbox[1] + bbox[3]) / 2.0;
    let side = ((bbox[2] - bbox[0]).max(bbox[3] - bbox[1]) * 1.25).max(8.0);
    let half = side / 2.0;
    let x1 = (cx - half).max(0.0).min(iw - 1.0) as u32;
    let y1 = (cy - half).max(0.0).min(ih - 1.0) as u32;
    let x2 = (cx + half).max(1.0).min(iw) as u32;
    let y2 = (cy + half).max(1.0).min(ih) as u32;
    let w = x2.saturating_sub(x1).max(1);
    let h = y2.saturating_sub(y1).max(1);
    let crop = image::imageops::crop_imm(img, x1, y1, w, h).to_image();
    image::imageops::resize(&crop, 112, 112, image::imageops::FilterType::Triangle)
}

/// Preprocess an aligned 112×112 face + run the embedder in the given layout.
/// RGB, `(px-127.5)/128`. NCHW = `[1,3,112,112]`, NHWC = `[1,112,112,3]`.
fn run_embedder(
    emb: &mut OrtSession,
    input_name: &str,
    aligned: &image::RgbImage,
    nhwc: bool,
) -> anyhow::Result<Vec<f32>> {
    let plane = 112 * 112;
    let mut data = vec![0.0f32; 3 * plane];
    for y in 0..112usize {
        for x in 0..112usize {
            let pix = aligned.get_pixel(x as u32, y as u32);
            let r = (pix[0] as f32 - 127.5) / 128.0;
            let g = (pix[1] as f32 - 127.5) / 128.0;
            let b = (pix[2] as f32 - 127.5) / 128.0;
            if nhwc {
                let d = (y * 112 + x) * 3;
                data[d] = r; data[d + 1] = g; data[d + 2] = b;
            } else {
                let d = y * 112 + x;
                data[d] = r; data[plane + d] = g; data[2 * plane + d] = b;
            }
        }
    }
    let shape: [usize; 4] = if nhwc { [1, 112, 112, 3] } else { [1, 3, 112, 112] };
    let tensor = ort::value::Tensor::<f32>::from_array((shape, data))
        .map_err(|e| anyhow::anyhow!("face emb tensor: {e}"))?;
    let outputs = { let _t = crate::inference::infer_timer("face_embed"); emb.run(ort::inputs![input_name => tensor]) }
        .map_err(|e| anyhow::anyhow!("face emb run ({}): {e}", if nhwc { "NHWC" } else { "NCHW" }))?;
    let (_, raw): (Vec<i64>, Vec<f32>) = outputs[0].try_extract_tensor::<f32>()
        .map(|(s, d)| (s.iter().copied().collect::<Vec<_>>(), d.to_vec()))
        .map_err(|e| anyhow::anyhow!("face emb extract: {e}"))?;
    Ok(raw)
}

// ─── Track-consensus naming — THE single naming policy ────────────────────

/// Accumulated face-match votes for ONE identity across a track/event.
/// Both naming paths (event `sub_label` in agent::clip and live face→body
/// fusion in inference.rs) must decide through [`IdentityVote::confirmed`],
/// so a name can never be committed by one lucky frame and the live and event
/// verdicts can't disagree.
#[derive(Clone, Copy, Default, Debug)]
pub struct IdentityVote { pub n: u32, pub sum: f32, pub max: f32 }

impl IdentityVote {
    pub fn add(&mut self, score: f32) {
        self.n += 1;
        self.sum += score;
        if score > self.max { self.max = score; }
    }
    pub fn avg(&self) -> f32 { if self.n == 0 { 0.0 } else { self.sum / self.n as f32 } }
    /// standard consensus: recognized consistently (≥2 frames at a solid
    /// average) OR once with real clearance above the threshold.
    pub fn confirmed(&self, rec_threshold: f32) -> bool {
        (self.n >= 2 && self.avg() >= rec_threshold * 0.95) || self.max >= rec_threshold + 0.07
    }
}

// ─── Matching against the known_persons DB ────────────────────────────────

/// Walk every known person's stored embeddings and return the best cosine
/// similarity. standard: returns the match if `score >= rec_threshold`,
/// `("unknown", score)` if `score >= unknown_score`, or `None` otherwise.
pub async fn match_face(
    db: &SqlitePool,
    emb: &[f32],
    rec_threshold: f32,
    unknown_score: f32,
    class_conf: f32,
) -> Option<FaceMatch> {
    // ONE roster walk gives the cosine top-pick AND the runner-up — used by the
    // classifier's double-verification, the ambiguity guard, and provenance
    // (previously best_two ran a second time inside the guard).
    let (cosine_best, second) = best_two_known_matches(db, emb).await;
    let margin = match (&cosine_best, second) {
        (Some((_, _, b)), Some(s)) => Some(b - s),
        _ => None,
    };

    // Hybrid head: when a trained classifier covers the roster (≥2 well-sampled
    // people), prefer its calibrated decision — it discriminates look-alikes better
    // than raw cosine as the gallery grows. It returns None (→ cosine fallback) for
    // sparse rosters, low confidence, or its synthetic "unknown" reject class, so
    // 1–2-shot cold-start behaves exactly as before. `score` here is a probability.
    if let Some((pid, name, prob)) = crate::face_classifier::predict_person(db, emb, class_conf).await {
        // Edge-AI NVRs DOUBLE-VERIFICATION: a trained classifier can be overconfident
        // on a look-alike, so confirm its pick against the named person's ACTUAL
        // embeddings before trusting it. The confirming cosine must clear the FULL
        // recognition threshold — verifying at the near-miss floor (unknown_score)
        // let a look-alike at 0.4 cosine "confirm" a wrong classifier pick, which
        // then poisoned body galleries via fusion. A real match to an enrolled
        // person clears rec_threshold by definition of enrollment quality.
        let verified = match &cosine_best {
            Some((bid, _, bscore)) if bid == &pid => *bscore >= rec_threshold,
            _ => max_cosine_to_person(db, emb, &pid).await >= rec_threshold,
        };
        if verified {
            return Some(FaceMatch { person_id: pid, name, score: prob, method: "face_classifier", margin });
        }
    }
    let (best_id, best_name, best_score) = cosine_best?;
    if best_score >= rec_threshold {
        // Look-alike ambiguity guard (verification-margin practice): when a SECOND
        // person scores nearly as high, a confident-looking top score is not a
        // recognition — demote to the anonymous near-miss path instead of guessing.
        let ambiguous = second.is_some_and(|s| s >= unknown_score && (best_score - s) < 0.05);
        if ambiguous {
            return Some(FaceMatch { person_id: String::new(), name: "unknown".into(), score: best_score, method: "unknown", margin });
        }
        Some(FaceMatch { person_id: best_id, name: best_name, score: best_score, method: "face_cosine", margin })
    } else if best_score >= unknown_score {
        // Near-miss: a known person is the closest, but below the confident bar.
        // The live/event contract keeps this anonymous ("unknown"); the Train tab
        // surfaces the candidate separately via `best_known_match`.
        Some(FaceMatch { person_id: String::new(), name: "unknown".into(), score: best_score, method: "unknown", margin })
    } else {
        None
    }
}

/// Max ArcFace cosine of `emb` to ONE specific enrolled person's stored shots.
/// Powers `match_face`'s double-verification. Returns -1 when the person has no
/// comparable embeddings.
async fn max_cosine_to_person(db: &SqlitePool, emb: &[f32], pid: &str) -> f32 {
    let row: Option<(String,)> = sqlx::query_as("SELECT embeddings FROM known_persons WHERE id = ?")
        .bind(pid).fetch_optional(db).await.ok().flatten();
    let Some((embs_json,)) = row else { return -1.0 };
    let Ok(embs): Result<Vec<Vec<f32>>, _> = serde_json::from_str(&embs_json) else { return -1.0 };
    embs.iter().filter(|s| s.len() == emb.len())
        .map(|s| cosine_sim(emb, s))
        .fold(-1.0_f32, f32::max)
}

/// Top match PLUS the best score of any OTHER person — the runner-up margin is
/// the classic verification guard against look-alikes: a confident cosine that is
/// only a hair above the second-best PERSON is ambiguous, not a recognition.
pub async fn best_two_known_matches(db: &SqlitePool, emb: &[f32]) -> (Option<(String, String, f32)>, Option<f32>) {
    let Ok(rows) = sqlx::query_as::<_, (String, String, String)>(
        "SELECT id, name, embeddings FROM known_persons"
    ).fetch_all(db).await else { return (None, None) };

    // Per-PERSON max cosine (a person's own multiple shots must not fill both slots).
    let mut best: (f32, String, String) = (-1.0, String::new(), String::new());
    let mut second: f32 = -1.0;
    for (pid, pname, embs_json) in &rows {
        // `embeddings` column is JSON: Vec<Vec<f32>> — one row per enrolled shot.
        let Ok(embs): Result<Vec<Vec<f32>>, _> = serde_json::from_str(embs_json) else { continue };
        let mut person_max = -1.0_f32;
        for stored in &embs {
            if stored.len() != emb.len() { continue; }
            let score = cosine_sim(emb, stored);
            if score > person_max { person_max = score; }
        }
        if person_max > best.0 {
            second = best.0;
            best = (person_max, pid.clone(), pname.clone());
        } else if person_max > second {
            second = person_max;
        }
    }
    let top = if best.1.is_empty() { None } else { Some((best.1, best.2, best.0)) };
    (top, if second > -1.0 { Some(second) } else { None })
}

/// Detect + align + embed every face in a JPEG. ORT work runs under the global
/// model mutex, which is acquired and **released before return** (no lock is held
/// across an `.await`). Returns each detection paired with its embedding. Shared
/// by `recognize_faces` (events), `recognize_frame` (live), and `embed_face`
/// (enrollment) so all three use the identical 512-d ArcFace space.
fn detect_and_embed(
    state: &Arc<AppState>,
    jpeg: &[u8],
    size: FaceModelSize,
    det_thr: f32,
) -> Vec<(FaceDetection, FaceEmbedding)> {
    let cell = FACE_MODELS.get_or_init(|| Mutex::new(None));
    let mut guard = match cell.lock() { Ok(g) => g, Err(_) => return Vec::new() };
    // (Re)load if the configured size changed since the last call.
    let need_reload = guard.as_ref().map(|m| m.size()) != Some(size);
    if need_reload {
        match FaceModels::try_load(&state.data_dir, size) {
            Ok(m)  => *guard = Some(m),
            Err(e) => {
                // A load failure here silently disables ALL face recognition, so
                // make it visible (the People tab's health strip surfaces the same
                // via `face_debug`). Was `debug` — too quiet for a total outage.
                tracing::warn!("face models unavailable ({size:?}): {e} — recognition disabled until fixed");
                return Vec::new();
            }
        }
    }
    let models = guard.as_mut().unwrap();
    let dets = models.detect(jpeg, det_thr).unwrap_or_default();
    dets.into_iter()
        .filter_map(|d| models.align_and_embed(jpeg, &d).ok().map(|e| (d, e)))
        .collect()
}

/// One-shot face recognition for a single JPEG. Returns the list of confident
/// matches (standard — empty when face_model is "off" or models are missing).
///
/// Caller flow:
///   1. Settings → which model size to use (off / small / large)
///   2. ORT detect + align + embed under the global mutex (released before any await)
///   3. Async DB writes: store embedding, query known_persons, return matches
pub async fn recognize_faces(
    state: &Arc<AppState>,
    jpeg: &[u8],
    cam_id: u8,
    event_id: Option<&str>,
    person_boxes: &[[f32; 4]],
) -> Vec<FaceMatch> {
    // Snapshot the relevant settings then drop the read lock immediately.
    let (size, det_thr, rec_thr, unk_thr, quality_floor, class_conf) = {
        let s = state.settings.read().await;
        let Some(sz) = active_face_tier(&s.face_model, &state.data_dir) else { return Vec::new() };
        (sz, s.face_detection_threshold, s.face_recognition_threshold, s.face_unknown_score, s.face_quality_floor, s.face_class_confidence)
    };

    // Liveness/anti-spoofing gate (optional, off by default). Read once per frame.
    let liveness_on = state.settings.read().await.face_liveness;

    // ── Synchronous ORT work, mutex held only inside detect_and_embed ─────
    let faces = detect_and_embed(state, jpeg, size, det_thr);
    // Full-frame context for any crop we store this frame (computed ONCE, not per face).
    let context = if faces.is_empty() { None } else { make_context_b64(jpeg) };

    // ── Async DB work (no ORT lock held) ─────────────────────────────────
    // Pass 1: gate + MATCH every face first, with no side effects — the
    // per-frame identity-uniqueness check below needs the whole frame's
    // matches before anything is stored or counted.
    struct FrameFace<'a> {
        det:  &'a FaceDetection,
        emb:  &'a FaceEmbedding,
        pbox: Option<&'a [f32; 4]>,
        m:    Option<FaceMatch>,
    }
    let mut frame_faces: Vec<FrameFace> = Vec::new();
    for (det, emb) in &faces {
        // Size gate: skip tiny/distant faces (mature NVRs' min_area) — they embed poorly.
        let area = (det.bbox[2]-det.bbox[0]).max(0.0) * (det.bbox[3]-det.bbox[1]).max(0.0);
        if area < MIN_FACE_AREA_PX { continue; }
        // Quality gate: drop very blurry faces (mature NVRs' blur filter).
        // Floor is user-tunable via `settings.face_quality_floor`; 0.0 = keep all.
        if emb.quality < quality_floor { continue; }
        // Person-gate: only keep faces that sit on a DETECTED PERSON (drops
        // background clutter — e.g. a ceiling corner the detector mistook for a
        // face). When no person boxes are passed (live/event paths) we don't gate.
        let pbox = containing_person(&det.bbox, person_boxes);
        if !person_boxes.is_empty() && pbox.is_none() { continue; }
        let m = match_face(&state.db, &emb.vector, rec_thr, unk_thr, class_conf).await;
        frame_faces.push(FrameFace { det, emb, pbox, m });
    }

    // Pass 2: per-frame identity uniqueness — one person cannot be two faces at
    // once. Keep the highest-scoring face per identity; demote the rest to
    // unknown (see `duplicate_identity_indices`).
    let named: Vec<(usize, String, f32)> = frame_faces.iter().enumerate()
        .filter_map(|(i, f)| f.m.as_ref()
            .filter(|fm| !fm.person_id.is_empty() && fm.name != "unknown")
            .map(|fm| (i, fm.person_id.clone(), fm.score)))
        .collect();
    for i in duplicate_identity_indices(&named) {
        if let Some(fm) = frame_faces[i].m.as_mut() {
            tracing::info!("cam{cam_id}: '{}' matched two faces in one frame — demoting the weaker match to unknown", fm.name);
            fm.person_id = String::new();
            fm.name = "unknown".into();
        }
    }

    // Pass 3: side effects (liveness, crop storage, last-seen, sightings).
    let mut matches = Vec::new();
    for FrameFace { det, emb, pbox, m } in frame_faces {
        let mut m = m;
        let area = (det.bbox[2]-det.bbox[0]).max(0.0) * (det.bbox[3]-det.bbox[1]).max(0.0);
        // Tag the stored sighting with the matched known person so the People
        // view can render per-person galleries + a recent-recognitions feed;
        // unknowns stay person_id=NULL so they still surface in the Train tab.
        let mut matched_pid = m.as_ref()
            .filter(|fm| !fm.person_id.is_empty() && fm.name != "unknown")
            .map(|fm| fm.person_id.clone());
        // Liveness: a face that scores as a photo/screen presentation attack must
        // NOT be named — demote it to "unknown" so a printed photo or phone screen
        // can't impersonate a resident. Only worth the inference cost on a face we'd
        // otherwise name. Fail-open inside `is_spoof` (missing model → not a spoof).
        if liveness_on && matched_pid.is_some()
            && crate::liveness::is_spoof(&state.data_dir, jpeg, &det.bbox) {
            tracing::info!("liveness: demoted a spoofed face on cam {cam_id} to unknown");
            matched_pid = None;
            m = Some(FaceMatch { person_id: String::new(), name: "unknown".into(), score: 0.0, method: "unknown", margin: None });
        }
        // Store as a People/Train crop ONLY when the face is big enough to be
        // identifiable AND it's not a near-duplicate of a recent capture on this
        // camera. (Matching above still ran for smaller faces, so a distant enrolled
        // person is still named — we just don't flood Train with tiny near-dupes.)
        if area >= MIN_STORE_FACE_AREA && !is_recent_duplicate(cam_id, &emb.vector) {
            // DISPLAY crop = the Tracked-style PERSON crop (upright, head-to-torso,
            // recognisable — the same `crop_person_jpeg` the Tracked tab uses), so
            // Train never shows the tilted aligned warp. Fall back to the padded
            // face crop when there's no person box.
            let thumb = pbox
                .and_then(|pb| crate::reid::crop_person_jpeg(jpeg, pb))
                .unwrap_or_else(|| emb.thumbnail_b64.clone());
            // Provenance rides along only when the crop was actually NAMED.
            let prov = matched_pid.as_ref().and(m.as_ref())
                .map(|fm| (fm.method, fm.score, fm.margin));
            store_face_embedding(&state.db, &state.data_dir, matched_pid.as_deref(), emb, cam_id, event_id, &thumb, context.as_deref(), prov).await;
        }
        if let Some(pid) = &matched_pid {
            // Keep the roster's "last seen" honest on every recognition.
            let _ = sqlx::query("UPDATE known_persons SET last_seen_at=datetime('now') WHERE id=?")
                .bind(pid).execute(&state.db).await;
            // Record the sighting (throttled to one row per person+cam per minute).
            // This path is where MOST recognitions happen; previously only the
            // event-analysis pass logged sightings, so the roster's activity stats
            // (sightings/days/peak-hour/cameras) starved for anyone whose events
            // weren't LLM-analyzed — e.g. 1 recorded sighting across 4 present days.
            if sighting_due(pid, cam_id) {
                if let Some(fm) = m.as_ref() {
                    let _ = crate::correlation::insert_face_sighting(
                        &state.db, &fm.name, Some(pid.as_str()), cam_id as i64, event_id, fm.score, fm.method).await;
                }
            }
        }
        if let Some(m) = m { matches.push(m); }
    }
    matches
}

/// Persist a face embedding for later analysis / auto-enrolment. The DISPLAY crop
/// + context frame are offloaded to files (`blobstore`) so the DB stays lean — the
/// recognition `descriptor` is stored inline as always.
pub async fn store_face_embedding(
    db: &SqlitePool,
    data_dir: &std::path::Path,
    person_id: Option<&str>,
    emb: &FaceEmbedding,
    cam_id: u8,
    event_id: Option<&str>,
    thumbnail: &str,
    context: Option<&str>,
    // Provenance of the naming decision (None when stored unmatched/unknown):
    // which recognizer, its score, and the runner-up margin — so a wrong name
    // is traceable in People → "why this name?".
    provenance: Option<(&str, f32, Option<f32>)>,
) {
    // Depth-anonymized cams: identity data (descriptors + crops) is NEVER
    // persisted. In-memory matching still tagged the event by name upstream;
    // enrolled faces are untouched (face-data hard rule) — we just don't ADD.
    if crate::depth::is_anonymized(cam_id) {
        tracing::debug!("cam{cam_id}: face persistence skipped (depth anonymization ON)");
        return;
    }
    let blob: Vec<u8> = emb.vector.iter().flat_map(|v| v.to_le_bytes()).collect();
    let id = uuid::Uuid::new_v4().to_string();
    // Offload the display images to disk; store a compact `@file:` ref (or the
    // original base64 if offload fails — never loses the crop).
    let thumb_ref = crate::blobstore::store(data_dir, "faces", &id, "_t", thumbnail);
    let ctx_ref = context.map(|c| crate::blobstore::store(data_dir, "faces", &id, "_c", c));
    let (method, score, margin) = match provenance {
        Some((m, s, g)) => (Some(m), Some(s as f64), g.map(|v| v as f64)),
        None => (None, None, None),
    };
    sqlx::query(
        "INSERT INTO face_embeddings(id, person_id, descriptor, dim, quality, cam_id, event_id, thumbnail_b64, context_b64, match_method, match_score, match_margin)
         VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    ).bind(&id).bind(person_id).bind(&blob)
     .bind(emb.vector.len() as i64).bind(emb.quality as f64)
     .bind(cam_id as i64).bind(event_id)
     .bind(&thumb_ref)
     .bind(&ctx_ref)
     .bind(method).bind(score).bind(margin)
     .execute(db).await.ok();
}

/// Read-only recognition over a single JPEG — same matcher as `recognize_faces`
/// but **no DB writes** (no crops stored, no `last_seen` bump). Used to boost
/// event recall by recognising faces across several stroboscopic frames in
/// addition to the single peak-motion thumbnail, WITHOUT flooding People → Train
/// with one crop per frame. Returns confident + near-miss matches (the same
/// `match_face` contract as the thumbnail pass).
pub async fn recognize_faces_readonly(state: &Arc<AppState>, jpeg: &[u8]) -> Vec<FaceMatch> {
    let (size, det_thr, rec_thr, unk_thr, quality_floor, class_conf) = {
        let s = state.settings.read().await;
        let Some(sz) = active_face_tier(&s.face_model, &state.data_dir) else { return Vec::new() };
        (sz, s.face_detection_threshold, s.face_recognition_threshold, s.face_unknown_score, s.face_quality_floor, s.face_class_confidence)
    };
    let faces = detect_and_embed(state, jpeg, size, det_thr);
    let mut out = Vec::new();
    for (det, emb) in &faces {
        let area = (det.bbox[2]-det.bbox[0]).max(0.0) * (det.bbox[3]-det.bbox[1]).max(0.0);
        if area < MIN_FACE_AREA_PX { continue; }
        if emb.quality < quality_floor { continue; }
        if let Some(m) = match_face(&state.db, &emb.vector, rec_thr, unk_thr, class_conf).await { out.push(m); }
    }
    out
}

// ─── Enrollment + live recognition commands (unified ArcFace) ──────────────

/// One captured face for guided multi-angle enrollment — the 512-d ArcFace
/// embedding plus quality/size/crop so the UI can gate diversity + preview.
#[derive(serde::Serialize)]
pub struct FaceCapture {
    pub embedding:     Vec<f32>,
    pub quality:       f32,
    pub area:          f32,        // face bbox area in source pixels
    pub bbox:          [f32; 4],
    pub thumbnail_b64: String,
}

/// Embed the largest face in a base64 JPEG with the ArcFace model. Returns `None`
/// when no model is installed or no face is found. Drives the Enroll capture loop
/// (one embedding per angle) — the SAME space as recognition, so enrolled faces
/// match in live + events + search.
#[tauri::command]
pub async fn embed_face(
    state: State<'_, Arc<AppState>>,
    jpeg_b64: String,
) -> Result<Option<FaceCapture>, String> {
    let (size, det_thr) = {
        let s = state.settings.read().await;
        // Use whichever tier is actually installed (NOT a hardcoded Small — that
        // failed silently when only Large was present). Err clearly if none.
        let Some(sz) = active_face_tier(&s.face_model, &state.data_dir) else {
            return Err("No face model installed".into());
        };
        // Enrollment is guided (face in the oval) — detect liberally so a
        // moderate-confidence frame isn't dropped as "No face".
        (sz, 0.3_f32)
    };
    let jpeg = base64::Engine::decode(
        &base64::engine::general_purpose::STANDARD, jpeg_b64.trim(),
    ).map_err(|e| format!("bad jpeg: {e}"))?;

    let faces = detect_and_embed(state.inner(), &jpeg, size, det_thr);
    let best = faces.into_iter().max_by(|a, b| {
        let aa = (a.0.bbox[2]-a.0.bbox[0]).max(0.0) * (a.0.bbox[3]-a.0.bbox[1]).max(0.0);
        let ba = (b.0.bbox[2]-b.0.bbox[0]).max(0.0) * (b.0.bbox[3]-b.0.bbox[1]).max(0.0);
        aa.partial_cmp(&ba).unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(best.map(|(d, e)| {
        let area = (d.bbox[2]-d.bbox[0]).max(0.0) * (d.bbox[3]-d.bbox[1]).max(0.0);
        FaceCapture { embedding: e.vector, quality: e.quality, area, bbox: d.bbox, thumbnail_b64: e.thumbnail_b64 }
    }))
}

/// One recognised face in a live frame — name + score + box for the overlay.
#[derive(serde::Serialize)]
pub struct RecognizedFace {
    pub person_id: String,
    pub name:      String,
    pub score:     f32,
    pub bbox:      [f32; 4],
}

/// A located face match for face↔body fusion: the recognised identity plus its
/// bounding box, so the inference loop can attribute a name to the person box /
/// tracklet it sits in. `name == "unknown"` / empty `person_id` = no confident
/// identity (caller filters those out).
pub struct LocatedFace {
    pub person_id: String,
    pub name:      String,
    pub score:     f32,
    pub bbox:      [f32; 4],
}

/// READ-ONLY recognition that returns each face's identity **and bbox** (the one
/// thing the event-path recognizers don't expose). Reuses the SAME `detect_and_embed`
/// + `match_face` as everything else — performs **no DB writes** and does NOT modify
/// any existing recognition path. Used by the inference loop to auto-label bodies
/// from a co-occurring recognised face. Empty when no face model is installed.
pub async fn locate_faces(state: &Arc<AppState>, jpeg: &[u8]) -> Vec<LocatedFace> {
    let (size, det_thr, rec_thr, unk_thr, quality_floor, class_conf) = {
        let s = state.settings.read().await;
        let Some(sz) = active_face_tier(&s.face_model, &state.data_dir) else { return Vec::new() };
        (sz, s.face_detection_threshold, s.face_recognition_threshold, s.face_unknown_score, s.face_quality_floor, s.face_class_confidence)
    };
    let faces = detect_and_embed(state, jpeg, size, det_thr);
    let mut out = Vec::new();
    for (det, emb) in &faces {
        let area = (det.bbox[2]-det.bbox[0]).max(0.0) * (det.bbox[3]-det.bbox[1]).max(0.0);
        if area < MIN_FACE_AREA_PX { continue; }
        if emb.quality < quality_floor { continue; }
        if let Some(m) = match_face(&state.db, &emb.vector, rec_thr, unk_thr, class_conf).await {
            out.push(LocatedFace { person_id: m.person_id, name: m.name, score: m.score, bbox: det.bbox });
        }
    }
    // Per-frame identity uniqueness — this feeds the face↔body fusion, where a
    // duplicated identity would link the WRONG body to a person's gallery.
    let named: Vec<(usize, String, f32)> = out.iter().enumerate()
        .filter(|&(_, f)| !f.person_id.is_empty() && f.name != "unknown").map(|(i, f)| (i, f.person_id.clone(), f.score))
        .collect();
    for i in duplicate_identity_indices(&named) {
        out[i].person_id = String::new();
        out[i].name = "unknown".into();
    }
    out
}

/// Recognise all faces in a base64 JPEG (live overlay path). ArcFace — the same
/// 512-d space as enrollment + events, so the live overlay and event sub_labels
/// agree. Read-only (no DB writes): the People galleries / recent-recognitions
/// are fed by the event path, so live recognition stays cheap.
#[tauri::command]
pub async fn recognize_frame(
    state: State<'_, Arc<AppState>>,
    jpeg_b64: String,
) -> Result<Vec<RecognizedFace>, String> {
    let (size, det_thr, rec_thr, unk_thr, quality_floor, class_conf) = {
        let s = state.settings.read().await;
        let Some(sz) = active_face_tier(&s.face_model, &state.data_dir) else { return Ok(Vec::new()) };
        (sz, s.face_detection_threshold, s.face_recognition_threshold, s.face_unknown_score, s.face_quality_floor, s.face_class_confidence)
    };
    let jpeg = base64::Engine::decode(
        &base64::engine::general_purpose::STANDARD, jpeg_b64.trim(),
    ).map_err(|e| format!("bad jpeg: {e}"))?;

    let faces = detect_and_embed(state.inner(), &jpeg, size, det_thr);
    let mut out = Vec::new();
    for (det, emb) in &faces {
        let area = (det.bbox[2]-det.bbox[0]).max(0.0) * (det.bbox[3]-det.bbox[1]).max(0.0);
        if area < MIN_FACE_AREA_PX { continue; }
        if emb.quality < quality_floor { continue; }
        if let Some(m) = match_face(&state.db, &emb.vector, rec_thr, unk_thr, class_conf).await {
            out.push(RecognizedFace { person_id: m.person_id, name: m.name, score: m.score, bbox: det.bbox });
        }
    }
    // Per-frame identity uniqueness for the live overlay — never draw the same
    // person's name on two boxes at once.
    let named: Vec<(usize, String, f32)> = out.iter().enumerate()
        .filter(|&(_, f)| !f.person_id.is_empty() && f.name != "unknown").map(|(i, f)| (i, f.person_id.clone(), f.score))
        .collect();
    for i in duplicate_identity_indices(&named) {
        out[i].person_id = String::new();
        out[i].name = "unknown".into();
    }
    Ok(out)
}

/// Diagnostic: can the face pipeline actually RUN right now? Resolves the active
/// tier and tries to load both ONNX models. Returns `"ready: face_small|face_large"`
/// or a clear error (no model installed / load failure) so the UI can surface it
/// instead of capture silently doing nothing. The heavy ORT load runs off-thread.
#[tauri::command]
pub async fn face_pipeline_status(state: State<'_, Arc<AppState>>) -> Result<String, String> {
    let face_model = state.settings.read().await.face_model.clone();
    let Some(tier) = active_face_tier(&face_model, &state.data_dir) else {
        return Err("No face model installed — install it in the Cookbook or via the Enroll tab.".into());
    };
    let data_dir = state.data_dir.clone();
    let loaded = tokio::task::spawn_blocking(move || FaceModels::try_load(&data_dir, tier))
        .await.map_err(|e| e.to_string())?;
    match loaded {
        Ok(_)  => Ok(format!("ready: {}", tier.skill_id())),
        Err(e) => Err(format!("Face model failed to load ({}): {e}", tier.skill_id())),
    }
}

/// Per-stage face-pipeline diagnostic surfaced in the Enroll tab so the user can
/// SEE what's installed/loaded and exactly why detection fails (not installed vs
/// 0 faces vs low confidence vs embedder can't run).
#[derive(serde::Serialize, Default)]
pub struct FaceDebug {
    pub tier:               String,   // "face_small" | "face_large" | "none"
    pub detector_installed: bool,
    pub embedder_installed: bool,
    pub detector_loaded:    bool,
    pub embedder_loaded:    bool,
    pub img_w:              u32,
    pub img_h:              u32,
    pub raw_faces:          usize,    // detected at a very low threshold
    pub max_conf:           f32,
    pub embedded_faces:     usize,    // how many also embedded successfully
    pub enrolled_persons:   usize,    // people with stored face embeddings
    pub enrolled_dim_mismatch: usize, // …whose embeddings don't match the active model (need re-enroll)
    pub note:               String,
}

/// Inspect the face pipeline (+ optionally run it on one frame). Heavy ORT work
/// off-thread. Drives the Enroll status line / bundled-models dots.
#[tauri::command]
pub async fn face_debug(
    state: State<'_, Arc<AppState>>,
    jpeg_b64: Option<String>,
) -> Result<FaceDebug, String> {
    let face_model = state.settings.read().await.face_model.clone();
    let data_dir = state.data_dir.clone();
    let jpeg = jpeg_b64.and_then(|b| base64::Engine::decode(
        &base64::engine::general_purpose::STANDARD, b.trim()).ok());

    let mut dbg = tokio::task::spawn_blocking(move || {
        let mut dbg = FaceDebug::default();
        let tier = active_face_tier(&face_model, &data_dir);
        dbg.tier = tier.map(|t| t.skill_id().to_string()).unwrap_or_else(|| "none".into());
        let resolved = tier.unwrap_or(FaceModelSize::Small);
        let dir = data_dir.join("skills").join(resolved.skill_id());
        dbg.detector_installed = dir.join("detector.onnx").exists();
        dbg.embedder_installed = dir.join("embedder.onnx").exists();

        let Some(t) = tier else { dbg.note = "No face model installed".into(); return dbg; };
        let mut models = match FaceModels::try_load(&data_dir, t) {
            Ok(m)  => { dbg.detector_loaded = true; dbg.embedder_loaded = true; m }
            Err(e) => { dbg.note = format!("Model failed to load: {e}"); return dbg; }
        };

        match jpeg {
            Some(jpeg) => {
                if let Ok(im) = image::load_from_memory(&jpeg) { dbg.img_w = im.width(); dbg.img_h = im.height(); }
                match models.detect(&jpeg, 0.05) {
                    Ok(dets) => {
                        dbg.raw_faces = dets.len();
                        dbg.max_conf = dets.iter().map(|d| d.score).fold(0.0, f32::max);
                        let mut embed_err: Option<String> = None;
                        for d in &dets {
                            match models.align_and_embed(&jpeg, d) {
                                Ok(_)  => dbg.embedded_faces += 1,
                                Err(e) => if embed_err.is_none() { embed_err = Some(e.to_string()); },
                            }
                        }
                        // Always include the numbers so nothing is hidden.
                        let base = format!("{} face(s) @ {:.2}, {} embedded",
                            dbg.raw_faces, dbg.max_conf, dbg.embedded_faces);
                        dbg.note = if dbg.raw_faces == 0 {
                            "0 faces — improve lighting / center your face".into()
                        } else if dbg.embedded_faces == 0 {
                            format!("{base} — embed failed: {}", embed_err.unwrap_or_default())
                        } else {
                            base
                        };
                    }
                    Err(e) => dbg.note = format!("detect error: {e}"),
                }
            }
            None => dbg.note = "Models loaded".into(),
        }
        dbg
    }).await.map_err(|e| e.to_string())?;

    // Legacy-embedding guard: enrolled people whose stored vectors don't match the
    // active embedder's dimension are SILENTLY skipped by the matcher (see
    // `best_known_match`) — a stealth cause of "enrolled people stay unknown" after
    // an engine/model change. Surface a count so the UI can prompt a re-enroll.
    let active_dim = match dbg.tier.as_str() {
        "face_small" => Some(FaceModelSize::Small.embed_dim()),
        "face_large" => Some(FaceModelSize::Large.embed_dim()),
        _            => None,
    };
    if let Some(dim) = active_dim {
        let rows: Vec<(String,)> = sqlx::query_as("SELECT embeddings FROM known_persons")
            .fetch_all(&state.db).await.unwrap_or_default();
        for (json,) in &rows {
            if let Ok(embs) = serde_json::from_str::<Vec<Vec<f32>>>(json) {
                if embs.is_empty() { continue; }
                dbg.enrolled_persons += 1;
                // Stale only if NONE of the person's angles match the active dim.
                if !embs.iter().any(|e| e.len() == dim) { dbg.enrolled_dim_mismatch += 1; }
            }
        }
        if dbg.enrolled_dim_mismatch > 0 {
            let n = dbg.enrolled_dim_mismatch;
            let extra = format!("{n} enrolled {} need re-enrollment on the current model (embedding size changed)",
                if n == 1 { "person" } else { "people" });
            dbg.note = if dbg.note.is_empty() { extra } else { format!("{} · {extra}", dbg.note) };
        }
    }
    Ok(dbg)
}

// ─── Helpers ───────────────────────────────────────────────────────────────

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// The detected person box whose (slightly-expanded) area contains the face's
/// center — used to (a) gate out clutter that isn't on a person and (b) take a
/// Tracked-style person crop for display. `None` when the face is on no person.
fn containing_person<'a>(face_bbox: &[f32; 4], person_boxes: &'a [[f32; 4]]) -> Option<&'a [f32; 4]> {
    let fcx = (face_bbox[0] + face_bbox[2]) * 0.5;
    let fcy = (face_bbox[1] + face_bbox[3]) * 0.5;
    // SMALLEST containing box wins (same rule as the face↔body fusion loop): with
    // two OVERLAPPING people the face belongs to the nearer/tighter body, but
    // `.find()` used to return whichever box happened to be first — attributing
    // the face (and its stored display crop) to the WRONG person's body.
    person_boxes.iter()
        .filter(|p| {
            let (pw, ph) = (p[2] - p[0], p[3] - p[1]);
            let (ex, ey) = (pw * 0.15, ph * 0.15);
            fcx >= p[0] - ex && fcx <= p[2] + ex && fcy >= p[1] - ey && fcy <= p[3] + ey
        })
        .min_by(|a, b| {
            let aa = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
            let ba = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
            aa.partial_cmp(&ba).unwrap_or(std::cmp::Ordering::Equal)
        })
}

/// Per-frame identity uniqueness: ONE person cannot appear twice in a single
/// frame. Input = (index, person_id, score) of every CONFIDENTLY-NAMED face in
/// the frame; output = the indices that must be DEMOTED to unknown — for every
/// person_id matched by multiple faces, all but the highest-scoring face are
/// demoted. A duplicate identity in one frame is itself evidence of a false
/// match (a look-alike standing next to the person, or a held-up photo) — and
/// letting both through would poison the person's gallery, stats and the
/// face↔body fusion. Deterministic: on equal scores the earlier face wins.
fn duplicate_identity_indices(named: &[(usize, String, f32)]) -> Vec<usize> {
    let mut best: std::collections::HashMap<&str, (usize, f32)> = std::collections::HashMap::new();
    for (idx, pid, score) in named {
        match best.get(pid.as_str()) {
            Some((_, s)) if *s >= *score => {}
            _ => { best.insert(pid.as_str(), (*idx, *score)); }
        }
    }
    let mut out: Vec<usize> = named.iter()
        .filter(|(idx, pid, _)| best.get(pid.as_str()).map(|(bi, _)| bi != idx).unwrap_or(false))
        .map(|(idx, _, _)| *idx)
        .collect();
    out.sort_unstable();
    out
}

/// Rate-limit face-sighting rows: at most one per (person, camera) per 60 s.
/// `recognize_faces` runs several times a second while someone is in frame —
/// without this, the activity log (`face_sightings`, which powers the roster's
/// stats + cross-camera correlation) would gain thousands of rows per hour of
/// presence. Same static-Mutex pattern as [`is_recent_duplicate`].
fn sighting_due(person_id: &str, cam_id: u8) -> bool {
    use std::time::{Duration, Instant};
    const MIN_INTERVAL: Duration = Duration::from_secs(60);
    static LAST: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashMap<(String, u8), Instant>>
    > = std::sync::OnceLock::new();
    let cell = LAST.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let mut map = match cell.lock() { Ok(m) => m, Err(_) => return false };
    let now = Instant::now();
    match map.get(&(person_id.to_string(), cam_id)) {
        Some(t) if now.duration_since(*t) < MIN_INTERVAL => false,
        _ => {
            map.insert((person_id.to_string(), cam_id), now);
            // Bounded: (persons × cams) keys — tiny; still sweep stale entries.
            if map.len() > 512 { map.retain(|_, t| now.duration_since(*t) < MIN_INTERVAL); }
            true
        }
    }
}

/// True if `emb` is a near-duplicate (cosine ≥ 0.92) of a face captured on this
/// camera within the last 60 s — so a person lingering in front of a camera yields
/// a few DISTINCT angles instead of a near-identical crop every few seconds
/// (mature NVRs' "diversity over near-duplicates"). Records `emb` as recent on a miss.
fn is_recent_duplicate(cam_id: u8, emb: &[f32]) -> bool {
    use std::time::{Duration, Instant};
    const DEDUP: f32 = 0.92;
    const TTL: Duration = Duration::from_secs(60);
    static RECENT: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashMap<u8, Vec<(Vec<f32>, Instant)>>>
    > = std::sync::OnceLock::new();
    let cell = RECENT.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let mut map = match cell.lock() { Ok(m) => m, Err(_) => return false };
    let now = Instant::now();
    let v = map.entry(cam_id).or_default();
    v.retain(|(_, t)| now.duration_since(*t) < TTL);
    if v.iter().any(|(e, _)| cosine_sim(emb, e) >= DEDUP) { return true; }
    v.push((emb.to_vec(), now));
    if v.len() > 8 { let n = v.len() - 8; v.drain(0..n); }
    false
}

fn nms(mut dets: Vec<FaceDetection>, iou_thr: f32) -> Vec<FaceDetection> {
    dets.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    let mut keep: Vec<FaceDetection> = Vec::new();
    for d in dets {
        let dropped = keep.iter().any(|k| iou(&k.bbox, &d.bbox) > iou_thr);
        if !dropped { keep.push(d); }
        if keep.len() >= 20 { break; }
    }
    keep
}

fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = a[2].min(b[2]);
    let y2 = a[3].min(b[3]);
    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area_a = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let area_b = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    let union = area_a + area_b - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

/// Natural, padded face crop for DISPLAY (base64 JPEG) — forehead-to-chin with
/// ~30% margin so a human can actually recognise who it is. The aligned warp is
/// the embedder's input and looks distorted, so People/Train show THIS instead.
fn face_display_crop(img: &image::RgbImage, bbox: &[f32; 4]) -> Option<String> {
    let (iw, ih) = (img.width() as f32, img.height() as f32);
    let (bw, bh) = ((bbox[2] - bbox[0]).max(1.0), (bbox[3] - bbox[1]).max(1.0));
    let (px, py) = (bw * 0.3, bh * 0.3);
    let x1 = (bbox[0] - px).max(0.0) as u32;
    let y1 = (bbox[1] - py).max(0.0) as u32;
    let x2 = (bbox[2] + px).min(iw) as u32;
    let y2 = (bbox[3] + py).min(ih) as u32;
    if x2 <= x1 || y2 <= y1 { return None; }
    let mut crop = image::imageops::crop_imm(img, x1, y1, x2 - x1, y2 - y1).to_image();
    let maxd = crop.width().max(crop.height());
    if maxd > 200 {
        let s = 200.0 / maxd as f32;
        crop = image::imageops::resize(
            &crop,
            ((crop.width()  as f32 * s).round() as u32).max(1),
            ((crop.height() as f32 * s).round() as u32).max(1),
            image::imageops::FilterType::Triangle,
        );
    }
    let mut out = Vec::with_capacity(8192);
    let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 82);
    enc.encode_image(&crop).ok()?;
    Some(base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &out))
}

/// Downscaled JPEG of the WHOLE frame a face was captured in (base64), so the
/// Train UI can expand a face crop into its full scene (person + surroundings).
/// ≤720px longest side, q72 — ~40–80 KB.
fn make_context_b64(jpeg: &[u8]) -> Option<String> {
    let img = image::load_from_memory(jpeg).ok()?;
    let (w, h) = (img.width(), img.height());
    let maxd = w.max(h);
    let scaled = if maxd > 720 {
        let s = 720.0 / maxd as f32;
        img.resize(((w as f32 * s) as u32).max(1), ((h as f32 * s) as u32).max(1),
                   image::imageops::FilterType::Triangle)
    } else { img };
    let mut out = Vec::with_capacity(48 * 1024);
    let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 72);
    enc.encode_image(&scaled).ok()?;
    Some(base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &out))
}

/// Estimate sharpness as the variance of the Laplacian over a 112×112 face crop.
/// Higher = sharper. Same trick OpenCV uses for blur detection.
fn laplacian_variance(img: &image::RgbImage) -> f32 {
    let (w, h) = (img.width() as usize, img.height() as usize);
    if w < 3 || h < 3 { return 0.0; }
    let gray: Vec<f32> = img.pixels()
        .map(|p| 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32)
        .collect();
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    let mut n = 0usize;
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let c = gray[y * w + x];
            let lap = gray[(y - 1) * w + x] + gray[(y + 1) * w + x]
                + gray[y * w + x - 1] + gray[y * w + x + 1]
                - 4.0 * c;
            sum    += lap;
            sum_sq += lap * lap;
            n += 1;
        }
    }
    if n == 0 { return 0.0; }
    let mean = sum / n as f32;
    (sum_sq / n as f32) - mean * mean
}

/// 5-point landmark face alignment to 112×112.
///
/// Uses the InsightFace canonical template (the reference points every ArcFace
/// derivative is trained against):
///   left-eye  (38.29, 51.70), right-eye (73.53, 51.50),
///   nose      (56.03, 71.74),
///   mouth-L   (41.55, 92.36), mouth-R   (70.73, 92.20).
///
/// Solves the least-squares similarity transform (scale + rotation + translation)
/// that maps the detected landmarks onto the template, then samples the
/// transformed coordinates with bilinear interpolation.
fn warp_face_to_112(src: &image::RgbImage, landmarks: &[[f32; 2]; 5]) -> image::RgbImage {
    const TEMPLATE: [[f32; 2]; 5] = [
        [38.29, 51.70], [73.53, 51.50],
        [56.03, 71.74],
        [41.55, 92.36], [70.73, 92.20],
    ];

    // Compute similarity transform from `landmarks` (src) onto TEMPLATE (dst).
    let (sx_mean, sy_mean) = mean_pt(landmarks);
    let (tx_mean, ty_mean) = mean_pt(&TEMPLATE);
    // Umeyama least-squares similarity (scale + rotation, NO reflection):
    //   a = Σ<s,t>  (dot),   b = Σ(s × t)  (2D cross — sign AND magnitude give rotation)
    //   M = (1/Σ|s|²) · [[a, -b], [b, a]]
    // The previous hand-rolled cos/sin had a flipped rotation sign (used
    // `den.signum()` where the cross term needed the opposite sign) AND derived
    // |sin| from sqrt(1-cos²), which is numerically unstable near cos≈1. Together
    // they mis-rotated any non-upright face, warping it into the corner of the
    // 112×112 crop (black margins) — exactly the "corner crop" seen in Train.
    let (mut a, mut b) = (0.0_f32, 0.0_f32);
    let mut sx_var = 0.0_f32;
    for k in 0..5 {
        let sx = landmarks[k][0] - sx_mean;
        let sy = landmarks[k][1] - sy_mean;
        let tx = TEMPLATE[k][0] - tx_mean;
        let ty = TEMPLATE[k][1] - ty_mean;
        a += sx * tx + sy * ty;     // dot
        b += sx * ty - sy * tx;     // cross
        sx_var += sx * sx + sy * sy;
    }
    let inv = if sx_var > 1e-6 { 1.0 / sx_var } else { 0.0 };

    // dst = M * src + t  =>  src = M⁻¹ * (dst - t)
    let m11 =  a * inv;
    let m12 = -b * inv;
    let m21 =  b * inv;
    let m22 =  a * inv;
    let tx_off = tx_mean - (m11 * sx_mean + m12 * sy_mean);
    let ty_off = ty_mean - (m21 * sx_mean + m22 * sy_mean);

    // Inverse 2×2 (similarity → invertible as long as det != 0)
    let det = m11 * m22 - m12 * m21;
    let inv_det = if det.abs() > 1e-6 { 1.0 / det } else { 1.0 };
    let im11 =  m22 * inv_det;
    let im12 = -m12 * inv_det;
    let im21 = -m21 * inv_det;
    let im22 =  m11 * inv_det;

    // Sample dst (112×112) by mapping each pixel back to src.
    let (sw, sh) = (src.width() as f32, src.height() as f32);
    let mut out = image::RgbImage::new(112, 112);
    for dy in 0..112u32 {
        for dx in 0..112u32 {
            let xf = dx as f32 - tx_off;
            let yf = dy as f32 - ty_off;
            let sx = im11 * xf + im12 * yf;
            let sy = im21 * xf + im22 * yf;
            let pix = if sx >= 0.0 && sy >= 0.0 && sx < sw - 1.0 && sy < sh - 1.0 {
                bilinear(src, sx, sy)
            } else {
                image::Rgb([0, 0, 0])
            };
            out.put_pixel(dx, dy, pix);
        }
    }
    out
}

fn mean_pt(pts: &[[f32; 2]; 5]) -> (f32, f32) {
    let (mut sx, mut sy) = (0.0_f32, 0.0_f32);
    for p in pts { sx += p[0]; sy += p[1]; }
    (sx / 5.0, sy / 5.0)
}

fn bilinear(img: &image::RgbImage, x: f32, y: f32) -> image::Rgb<u8> {
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(img.width()  - 1);
    let y1 = (y0 + 1).min(img.height() - 1);
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let p00 = img.get_pixel(x0, y0);
    let p01 = img.get_pixel(x1, y0);
    let p10 = img.get_pixel(x0, y1);
    let p11 = img.get_pixel(x1, y1);
    let mut out = [0u8; 3];
    for c in 0..3 {
        let v = (1.0 - fx) * (1.0 - fy) * p00[c] as f32
              +        fx  * (1.0 - fy) * p01[c] as f32
              + (1.0 - fx) *        fy  * p10[c] as f32
              +        fx  *        fy  * p11[c] as f32;
        out[c] = v.clamp(0.0, 255.0) as u8;
    }
    image::Rgb(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicate_identity_keeps_highest_and_demotes_rest() {
        // Faces 0 and 2 both matched person "a" — the higher score (idx 2) wins.
        let named = vec![
            (0, "a".to_string(), 0.55),
            (1, "b".to_string(), 0.70),
            (2, "a".to_string(), 0.61),
        ];
        assert_eq!(duplicate_identity_indices(&named), vec![0]);
    }

    #[test]
    fn duplicate_identity_equal_scores_keeps_earlier() {
        let named = vec![(3, "a".to_string(), 0.5), (7, "a".to_string(), 0.5)];
        assert_eq!(duplicate_identity_indices(&named), vec![7]);
    }

    #[test]
    fn duplicate_identity_unique_frame_demotes_nothing() {
        let named = vec![(0, "a".to_string(), 0.9), (1, "b".to_string(), 0.9)];
        assert!(duplicate_identity_indices(&named).is_empty());
        assert!(duplicate_identity_indices(&[]).is_empty());
    }

    #[test]
    fn containing_person_prefers_smallest_box_in_overlaps() {
        // Face centred at (50,50). Both boxes contain it; the TIGHTER box
        // (the nearer person) must win, not whichever comes first.
        let face = [45.0, 45.0, 55.0, 55.0];
        let big   = [0.0, 0.0, 200.0, 200.0];
        let small = [40.0, 40.0, 80.0, 120.0];
        let boxes = vec![big, small];
        assert_eq!(containing_person(&face, &boxes), Some(&boxes[1]));
        // Order-independent.
        let boxes2 = vec![small, big];
        assert_eq!(containing_person(&face, &boxes2), Some(&boxes2[0]));
    }

    #[test]
    fn containing_person_none_when_outside_all() {
        let face = [500.0, 500.0, 520.0, 520.0];
        let boxes = vec![[0.0, 0.0, 100.0, 100.0]];
        assert_eq!(containing_person(&face, &boxes), None);
    }

    // ── IdentityVote — THE shared naming policy (events + live fusion) ──────

    #[test]
    fn consensus_one_weak_hit_never_confirms() {
        // The exact poisoning scenario the policy exists to stop: a single
        // barely-over-threshold frame must not commit a name.
        let mut v = IdentityVote::default();
        v.add(0.51);
        assert!(!v.confirmed(0.5));
    }

    #[test]
    fn consensus_two_solid_hits_confirm() {
        let mut v = IdentityVote::default();
        v.add(0.52);
        v.add(0.50);
        // avg 0.51 ≥ 0.5·0.95 = 0.475 with n≥2 → confirmed.
        assert!(v.confirmed(0.5));
    }

    #[test]
    fn consensus_one_very_high_hit_confirms() {
        // A single hit with REAL clearance (thr + 0.07) is enough.
        let mut v = IdentityVote::default();
        v.add(0.58);
        assert!(v.confirmed(0.5));
        let mut w = IdentityVote::default();
        w.add(0.56); // just under the clearance bar
        assert!(!w.confirmed(0.5));
    }

    #[test]
    fn consensus_two_weak_hits_do_not_confirm() {
        // The OLD live-fusion rule ("any 2 ticks") passed this — the new shared
        // policy must not: two near-floor scores are a look-alike, not a person.
        let mut v = IdentityVote::default();
        v.add(0.42);
        v.add(0.44);
        assert!(!v.confirmed(0.5));
    }
}
