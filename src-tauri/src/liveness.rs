//! Face liveness / anti-spoofing (presentation-attack detection).
//!
//! The closest *achievable* thing to "more than 2D" on an RGB CCTV camera: reject a
//! printed photo or a phone/tablet screen held up to the camera so it isn't
//! recognised as a resident. This is NOT 3D face recognition (impossible without a
//! depth sensor) — it's a single 2D classifier (Silent-Face / MiniFASNet family)
//! that scores how "live" a face crop looks.
//!
//! Optional + OFF by default (`settings.face_liveness`). When the `face_liveness`
//! skill isn't installed or the model fails, `is_spoof` returns `false` (fail-open —
//! never block recognition because of a missing/broken optional model).
//!
//! NOTE: the model URL + output convention are best-guess (like the audio/Re-ID
//! skills) and need runtime validation once the skill downloads. MiniFASNet
//! (Silent-Face-Anti-Spoofing) emits a 3-class softmax `[spoof, real, spoof]` where
//! index 1 is the live score; we treat `< LIVE_FLOOR` as a spoof.

use std::path::Path;
use std::sync::{Mutex, OnceLock};

use ort::session::Session as OrtSession;
use ort::value::Tensor;


const SKILL: &str = "face_liveness";
/// MiniFASNet canonical input.
const IN_SIZE: u32 = 80;
/// Live-probability floor below which a face is treated as a presentation attack.
const LIVE_FLOOR: f32 = 0.55;

static SESSION: OnceLock<Mutex<Option<(OrtSession, String)>>> = OnceLock::new();

fn installed(data_dir: &Path) -> bool {
    data_dir.join("skills").join(SKILL).join("model.onnx").exists()
}

/// `true` if the face crop at `bbox` looks like a presentation attack (photo /
/// screen). Fail-open: returns `false` when the model is absent or errors, so a
/// missing/broken optional model never blocks recognition.
pub(crate) fn is_spoof(data_dir: &Path, jpeg: &[u8], bbox: &[f32; 4]) -> bool {
    if !installed(data_dir) { return false; }
    match live_score(data_dir, jpeg, bbox) {
        Some(live) => live < LIVE_FLOOR,
        None => false,
    }
}

fn live_score(data_dir: &Path, jpeg: &[u8], bbox: &[f32; 4]) -> Option<f32> {
    let model_path = data_dir.join("skills").join(SKILL).join("model.onnx");
    let cell = SESSION.get_or_init(|| Mutex::new(None));
    let mut guard = cell.lock().ok()?;
    if guard.is_none() {
        // Enrichment lane = CPU (event cadence; keeps the GPU lock YOLO-only).
        match crate::inference::build_ort_session_cpu(&model_path) {
            Ok(s) => {
                let input = s.inputs().first().map(|i| i.name().to_string())
                    .unwrap_or_else(|| "input".into());
                *guard = Some((s, input));
            }
            Err(e) => { tracing::debug!("face_liveness load failed: {e}"); return None; }
        }
    }
    let (session, input_name) = guard.as_mut()?;

    let img = image::load_from_memory(jpeg).ok()?;
    let (iw, ih) = (img.width() as f32, img.height() as f32);
    // Pad the face box ~30% — anti-spoof cues (moiré, paper edge, screen bezel) live
    // around the face, not just on it.
    let bw = (bbox[2] - bbox[0]).max(1.0);
    let bh = (bbox[3] - bbox[1]).max(1.0);
    let x1 = (bbox[0] - bw * 0.3).max(0.0).min(iw - 1.0) as u32;
    let y1 = (bbox[1] - bh * 0.3).max(0.0).min(ih - 1.0) as u32;
    let x2 = (bbox[2] + bw * 0.3).max(0.0).min(iw) as u32;
    let y2 = (bbox[3] + bh * 0.3).max(0.0).min(ih) as u32;
    if x2 <= x1 || y2 <= y1 { return None; }
    let crop = img.crop_imm(x1, y1, x2 - x1, y2 - y1)
        .resize_exact(IN_SIZE, IN_SIZE, image::imageops::FilterType::Triangle).to_rgb8();

    let (w, h) = (IN_SIZE as usize, IN_SIZE as usize);
    let plane = w * h;
    let mut data = vec![0.0f32; 3 * plane];
    for y in 0..h {
        for x in 0..w {
            let p = crop.get_pixel(x as u32, y as u32);
            let d = y * w + x;
            // MiniFASNet uses raw [0,1] RGB, NCHW.
            for c in 0..3 { data[c * plane + d] = p[c] as f32 / 255.0; }
        }
    }
    let tensor = Tensor::<f32>::from_array(([1usize, 3, h, w], data)).ok()?;
    let outputs = { let _t = crate::inference::infer_timer("liveness");
        session.run(ort::inputs![input_name.as_str() => tensor]) }.ok()?;
    let (_, raw) = outputs[0].try_extract_tensor::<f32>().ok()?;
    let logits = raw.to_vec();
    if logits.len() < 2 { return None; }
    // Softmax, then take the "live" class. 3-class Silent-Face → index 1 is live;
    // for a 2-class head we assume index 1 is live as well.
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum <= 0.0 { return None; }
    Some(exps[1] / sum)
}
