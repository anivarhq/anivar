//! Pedestrian attributes — PaddleClas PULC `person_attribute` (PP-LCNet x1.0).
//!
//! Trained on PA-100K only (CC-BY 4.0) — which is why this model and not
//! PP-Human's own, whose weights also saw PETA and "academic research only" data.
//!
//! Converted with paddle2onnx 1.3.1 (opset 14) and VERIFIED 2026-09-15 against
//! Paddle inference: max |Δ| 5e-7. Input `x` `[N,3,256,192]` RGB, /255 then
//! ImageNet mean/std; output `[N,26]` sigmoid probabilities in the index order
//! below. That order comes from PaddleClas's `PersonAttribute` post-processor,
//! which disagrees with PP-Human's docs about the three age buckets — the code is
//! what the model was evaluated with.
//!
//! Runs once per track at flush over its best crops and AVERAGES them. PP-Human
//! re-predicts every frame with no smoothing; a track vote is steadier.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use ort::session::Session as OrtSession;
use ort::value::Tensor;

pub(crate) const SKILL: &str = "par_pulc";
pub(crate) const N: usize = 26;

/// Output index → attribute key (PaddleClas `PersonAttribute` layout). The
/// contract every hard-coded index in `evidence()` and people search relies on,
/// pinned by `the_index_table_matches_paddleclas`.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) const ATTRS: [&str; N] = [
    "hat", "glasses", "short_sleeve", "long_sleeve",
    "upper_stride", "upper_logo", "upper_plaid", "upper_splice",
    "lower_stripe", "lower_pattern", "long_coat", "trousers", "shorts", "skirt",
    "boots", "handbag", "shoulder_bag", "backpack", "holding",
    "age_under_18", "age_18_60", "age_over_60", "female",
    "front", "side", "back",
];

const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];
const W: u32 = 192;
const H: u32 = 256;

static SESSION: OnceLock<Mutex<Option<(OrtSession, String)>>> = OnceLock::new();

fn model_path(data_dir: &Path) -> PathBuf {
    data_dir.join("skills").join(SKILL).join("model.onnx")
}

pub(crate) fn is_installed(data_dir: &Path) -> bool {
    std::fs::metadata(model_path(data_dir)).map(|m| m.len() > 64 * 1024).unwrap_or(false)
}

#[cfg(test)]
fn index(key: &str) -> Option<usize> { ATTRS.iter().position(|a| *a == key) }

/// Decision threshold per attribute (PaddleClas: glasses 0.3, holding 0.6, else 0.5).
pub(crate) fn threshold(i: usize) -> f32 {
    match i { 1 => 0.3, 18 => 0.6, _ => 0.5 }
}

/// Per-crop probabilities from ONE batched run. `None` when the model isn't
/// installed or fails — attributes are then simply unknown, never guessed.
pub(crate) fn predict(data_dir: &Path, crops: &[&image::RgbImage]) -> Option<Vec<[f32; N]>> {
    if crops.is_empty() || !is_installed(data_dir) { return None; }
    let (w, h) = (W as usize, H as usize);
    let plane = w * h;
    let mut data = vec![0.0f32; crops.len() * 3 * plane];
    for (k, img) in crops.iter().enumerate() {
        let resized = image::imageops::resize(*img, W, H, image::imageops::FilterType::Triangle);
        let chw = &mut data[k * 3 * plane..(k + 1) * 3 * plane];
        for (x, y, p) in resized.enumerate_pixels() {
            let d = y as usize * w + x as usize;
            for c in 0..3 {
                chw[c * plane + d] = (p[c] as f32 / 255.0 - MEAN[c]) / STD[c];
            }
        }
    }

    let cell = SESSION.get_or_init(|| Mutex::new(None));
    let mut guard = cell.lock().ok()?;
    if guard.is_none() {
        match crate::inference::build_ort_session_cpu(&model_path(data_dir)) {
            Ok(s) => {
                let input = s.inputs().first().map(|i| i.name().to_string()).unwrap_or_else(|| "x".into());
                *guard = Some((s, input));
            }
            Err(e) => { tracing::debug!("attribute model load failed: {e}"); return None; }
        }
    }
    let (session, input_name) = guard.as_mut()?;
    let tensor = Tensor::<f32>::from_array(([crops.len(), 3, h, w], data)).ok()?;
    let outputs = { let _t = crate::inference::infer_timer("attributes");
        session.run(ort::inputs![input_name.as_str() => tensor]).ok()? };
    let (_, raw) = outputs[0].try_extract_tensor::<f32>().ok()?;
    if raw.len() != crops.len() * N { return None; }
    Some(raw.chunks_exact(N).map(|r| std::array::from_fn(|i| r[i])).collect())
}

/// Average over a track's crops.
pub(crate) fn mean(per_crop: &[[f32; N]]) -> Option<[f32; N]> {
    if per_crop.is_empty() { return None; }
    let n = per_crop.len() as f32;
    Some(std::array::from_fn(|i| per_crop.iter().map(|p| p[i]).sum::<f32>() / n))
}

/// What the person visibly wears or carries, as evidence chips. Deliberately
/// never gender, age or facing direction: those exist as search filters only —
/// a label printed on a person is a claim the model can't back up.
pub(crate) fn evidence(p: &[f32; N]) -> Vec<&'static str> {
    let mut out = Vec::new();
    for (i, label) in [(0, "hat"), (1, "glasses"), (10, "long coat"), (14, "boots"),
                       (15, "handbag"), (16, "shoulder bag"), (17, "backpack"), (18, "holding something")] {
        if p[i] > threshold(i) { out.push(label); }
    }
    if p[2].max(p[3]) > 0.5 { out.push(if p[3] > p[2] { "long sleeves" } else { "short sleeves" }); }
    let lower = [(11, "trousers"), (12, "shorts"), (13, "skirt or dress")];
    if let Some(&(i, label)) = lower.iter().max_by(|a, b| p[a.0].partial_cmp(&p[b.0]).unwrap_or(std::cmp::Ordering::Equal)) {
        if p[i] > 0.5 { out.push(label); }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_index_table_matches_paddleclas() {
        assert_eq!(ATTRS.len(), N);
        assert_eq!(index("female"), Some(22));
        assert_eq!(index("backpack"), Some(17));
        assert_eq!(index("age_under_18"), Some(19), "PaddleClas code order, not PP-Human's doc table");
        assert_eq!(index("back"), Some(25));
    }

    #[test]
    fn evidence_uses_per_attribute_thresholds_and_never_demographics() {
        let mut p = [0.0f32; N];
        p[1] = 0.35;   // glasses — clears its 0.3 bar
        p[18] = 0.55;  // holding — misses its 0.6 bar
        p[17] = 0.9;   // backpack
        p[3] = 0.8;    // long sleeves
        p[11] = 0.7;   // trousers
        p[22] = 0.99; p[21] = 0.99; p[25] = 0.99; // female / over 60 / back: never chips
        let e = evidence(&p);
        assert_eq!(e, vec!["glasses", "backpack", "long sleeves", "trousers"]);
    }

    #[test]
    fn a_track_vote_is_the_mean() {
        let mut a = [0.0f32; N]; a[0] = 1.0;
        let b = [0.0f32; N];
        assert!((mean(&[a, b]).unwrap()[0] - 0.5).abs() < 1e-6);
        assert!(mean(&[]).is_none());
    }
}
