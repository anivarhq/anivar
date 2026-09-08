//! License plate recognition (ALPR) — standard separate skill.
//!
//! Uses the global Mobile-ViT v2 OCR model from
//! <https://github.com/ankandrew/fast-plate-ocr>, distributed via the
//! `cnn-ocr-lp` release tag `arg-plates`. The model takes a cropped plate
//! (or vehicle) region at 140×70 RGB and emits classification logits over
//! 9 character slots × 37-class alphabet (`0..9 A..Z _`).  Decoding is a
//! simple per-slot argmax with the pad character stripped.
//!
//! Pipeline:
//!   1. YOLO 2026 returns `car` / `motorcycle` bbox during clip analysis.
//!   2. We crop the bbox region from the event thumbnail, resize to 140×70.
//!   3. ORT forward — `output[slot, class]` → max class per slot.
//!   4. Concatenate letters, strip `_`, return `Option<String>`.
//!
//! Models are user-installed via `alpr_global` / `alpr_european` /
//! `alpr_argentinian` skills (see `skillDownload.ts`). `find_alpr_model`
//! looks for `<data>/skills/alpr_{region}/model.onnx`, falling back to
//! global and then any installed alpr_* directory.

use std::path::Path;
use std::sync::{Mutex, OnceLock};

use ort::session::Session as OrtSession;


/// Characters in the model's alphabet. Index = class id. `_` is the padding
/// slot, dropped from the final output. Source: `global_mobile_vit_v2_ocr_config.yaml`.
const ALPHABET: &[u8; 37] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_";
const IMG_W: u32 = 140;
const IMG_H: u32 = 70;
const SLOTS: usize = 9;
const CLASSES: usize = 37;

/// Process-wide ALPR model cache. Built lazily — most events don't contain a
/// vehicle so we don't pay the load cost unless we have to.
static ALPR_MODEL: OnceLock<Mutex<Option<OrtSession>>> = OnceLock::new();

/// Locate the ALPR model file, region-aware (v7).
/// Lookup order:
///   1. `skills/alpr_{region}/model.onnx` for the requested region
///   2. `skills/alpr_global/model.onnx` (best default)
///   3. `skills/alpr/model.onnx` (v6 legacy install layout)
///   4. Any `skills/alpr_*/model.onnx` so existing installs aren't broken
pub(crate) fn find_alpr_model(data_dir: &Path, region: &str) -> Option<std::path::PathBuf> {
    let skills_dir = data_dir.join("skills");

    // 1. Requested region first.
    let preferred = skills_dir.join(format!("alpr_{}", region.trim().to_ascii_lowercase()))
        .join("model.onnx");
    if preferred.exists() { return Some(preferred); }

    // 2. Global fallback.
    let global = skills_dir.join("alpr_global").join("model.onnx");
    if global.exists() { return Some(global); }

    // 3. v6 legacy layout (single `skills/alpr/` directory).
    let legacy = skills_dir.join("alpr").join("model.onnx");
    if legacy.exists() { return Some(legacy); }

    // 4. Anything else that looks like an ALPR install.
    if let Ok(read) = std::fs::read_dir(&skills_dir) {
        for entry in read.flatten() {
            let name = entry.file_name();
            let name_s = name.to_string_lossy();
            if name_s.starts_with("alpr") {
                let candidate = entry.path().join("model.onnx");
                if candidate.exists() { return Some(candidate); }
            }
        }
    }
    None
}

/// Recognise a license plate inside a JPEG region. `bbox` is `[x1, y1, x2, y2]`
/// in pixel coordinates of the JPEG. Pass the vehicle bbox from YOLO — the
/// crop is then resized to the model's expected 140×70 input.
///
/// Returns `Some(("ABC1234", confidence))` when the model is installed AND a
/// plate-like string was decoded — confidence is the mean per-character softmax
/// probability (0..1). Empty / pad-only output → `None`.
pub fn recognize_plate(data_dir: &Path, region: &str, jpeg: &[u8], bbox: [f32; 4]) -> Option<(String, f32)> {
    let model_path = find_alpr_model(data_dir, region)?;

    // Lazy load.
    let cell = ALPR_MODEL.get_or_init(|| Mutex::new(None));
    let mut guard = cell.lock().ok()?;
    if guard.is_none() {
        // Enrichment lane = CPU (event cadence; keeps the GPU lock YOLO-only).
        match crate::inference::build_ort_session_cpu(&model_path) {
            Ok(s) => *guard = Some(s),
            Err(e) => { tracing::debug!("ALPR session load failed: {e}"); return None; }
        }
    }
    let session = guard.as_mut()?;
    let input_name = session.inputs().first()
        .map(|i| i.name().to_string())
        .unwrap_or_else(|| "input".to_string());

    // Mature NVRs run a dedicated plate DETECTOR and OCRs the plate crop; squeezing
    // the WHOLE vehicle into the OCR's 140×70 input leaves the plate a few pixels
    // tall, so recognition only ever fired on frame-filling close-ups. Until a
    // plate-detector skill lands, approximate the locator GEOMETRICALLY: plates
    // live in the lower-center of a vehicle box. Try the candidate regions and
    // keep the decode the OCR itself is most confident about.
    let img = image::load_from_memory(jpeg).ok()?;
    let (iw, ih) = (img.width() as f32, img.height() as f32);
    let bw = bbox[2] - bbox[0];
    let bh = bbox[3] - bbox[1];
    let pad = (bw * 0.10).max(8.0);
    // (x1, y1, x2, y2) in image pixels, clamped later.
    let candidates: [[f32; 4]; 3] = [
        // lower-center band — the plate zone on most vehicles (front or rear view)
        [bbox[0] + bw * 0.15, bbox[1] + bh * 0.55, bbox[2] - bw * 0.15, bbox[3]],
        // tight lower-middle — closer crop when the vehicle is small in frame
        [bbox[0] + bw * 0.25, bbox[1] + bh * 0.62, bbox[2] - bw * 0.25, bbox[3] - bh * 0.04],
        // full padded vehicle — the legacy behaviour; still best for close-ups
        [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
    ];
    let mut best: Option<(String, f32)> = None;
    for c in candidates {
        let x1 = c[0].max(0.0) as u32;
        let y1 = c[1].max(0.0) as u32;
        let x2 = c[2].min(iw - 1.0) as u32;
        let y2 = c[3].min(ih - 1.0) as u32;
        if x2 <= x1 + 8 || y2 <= y1 + 4 { continue; }
        let crop = img.crop_imm(x1, y1, x2 - x1, y2 - y1);
        if let Some((plate, conf)) = ocr_region(session, &input_name, &crop) {
            if best.as_ref().map_or(true, |(_, bc)| conf > *bc) { best = Some((plate, conf)); }
        }
    }
    best
}

/// Run the fast-plate-ocr CNN over one candidate crop → (plate, confidence).
fn ocr_region(session: &mut OrtSession, input_name: &str, crop: &image::DynamicImage) -> Option<(String, f32)> {
    let resized = crop.resize_exact(IMG_W, IMG_H, image::imageops::FilterType::Triangle).to_rgb8();

    // CHW float32 [0,1].
    let mut data = vec![0.0f32; 3 * (IMG_W * IMG_H) as usize];
    let plane = (IMG_W * IMG_H) as usize;
    let bytes = resized.as_raw();
    for y in 0..IMG_H as usize {
        for x in 0..IMG_W as usize {
            let s = (y * IMG_W as usize + x) * 3;
            let d = y * IMG_W as usize + x;
            data[d]              = bytes[s    ] as f32 / 255.0;
            data[plane + d]      = bytes[s + 1] as f32 / 255.0;
            data[plane * 2 + d]  = bytes[s + 2] as f32 / 255.0;
        }
    }
    let tensor = ort::value::Tensor::<f32>::from_array(([1usize, 3, IMG_H as usize, IMG_W as usize], data)).ok()?;
    let inputs = ort::inputs![input_name => tensor];
    let outputs = { let _t = crate::inference::infer_timer("alpr"); session.run(inputs) }.ok()?;
    let (shape, raw): (Vec<i64>, Vec<f32>) = match outputs[0].try_extract_tensor::<f32>() {
        Ok((s, d)) => (s.iter().copied().collect(), d.to_vec()),
        Err(_) => return None,
    };

    // Decode: per-slot argmax over the 37-class alphabet. The exact shape
    // depends on the export — handle both `[1, 9, 37]` and `[1, 9*37]`.
    let (slots, classes) = match shape.as_slice() {
        [1, s, c] => (*s as usize, *c as usize),
        [s, c]    => (*s as usize, *c as usize),
        _         => (SLOTS, CLASSES),
    };
    if slots == 0 || classes == 0 || raw.len() < slots * classes {
        return None;
    }

    let mut chars = String::with_capacity(slots);
    let mut conf_sum = 0.0f32;
    let mut conf_n = 0u32;
    for slot in 0..slots.min(SLOTS) {
        let row = &raw[slot * classes..(slot + 1) * classes];
        let (best_idx, best_logit) = row.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))?;
        let ch_idx = best_idx.min(ALPHABET.len() - 1);
        let ch = ALPHABET[ch_idx] as char;
        if ch != '_' {
            chars.push(ch);
            // Softmax probability of the chosen character (numerically stable).
            let maxv = *best_logit;
            let denom: f32 = row.iter().map(|&v| (v - maxv).exp()).sum::<f32>().max(1e-9);
            conf_sum += 1.0 / denom; // exp(best - max) == 1
            conf_n += 1;
        }
    }

    // Plates are at least 3 chars in practice. Anything shorter is OCR noise.
    if chars.len() < 3 { None } else {
        let conf = if conf_n > 0 { conf_sum / conf_n as f32 } else { 0.0 };
        Some((chars, conf))
    }
}


// ─── Vehicle color (HSV histogram voting — the lightweight ANPR-industry
//     standard; no model). Runs at clip-analysis where the frame + vehicle
//     bbox are already in hand. Returns None rather than guessing when no
//     color wins a clear plurality (dusk/sodium light stays honest). ─────────

/// Classify the dominant body color of a vehicle crop. `bbox` in pixels.
/// SHARED HSV plurality-vote color classifier over one BAND of a bbox — the
/// core behind vehicle body color AND person clothing colors (reid.rs). Band
/// fractions are relative to the bbox: (top_frac, bottom_frac, side_trim_frac).
/// Honest abstain: returns None when too small, too few samples, or no color
/// reaches the plurality floor — never a coin-flip label.
pub(crate) fn dominant_color_in_band(
    img: &image::RgbImage,
    bbox: [f32; 4],
    band: (f32, f32, f32),
) -> Option<(String, f32)> {
    let (w, h) = (img.width() as f32, img.height() as f32);
    let x0 = bbox[0].clamp(0.0, w - 2.0);
    let y0 = bbox[1].clamp(0.0, h - 2.0);
    let x1 = bbox[2].clamp(x0 + 1.0, w - 1.0);
    let y1 = bbox[3].clamp(y0 + 1.0, h - 1.0);
    let bw = x1 - x0;
    let bh = y1 - y0;
    if bw < 24.0 || bh < 24.0 { return None; } // too small to judge
    let (top_frac, bottom_frac, trim_frac) = band;
    let sy0 = (y0 + bh * top_frac) as u32;
    let sy1 = (y0 + bh * bottom_frac) as u32;
    let sx0 = (x0 + bw * trim_frac) as u32;
    let sx1 = (x1 - bw * trim_frac) as u32;
    if sy1 <= sy0 || sx1 <= sx0 { return None; }

    // 11 bins: black white silver gray red orange yellow green blue purple brown
    const NAMES: [&str; 11] = ["black","white","silver","gray","red","orange","yellow","green","blue","purple","brown"];
    let mut votes = [0u32; 11];
    let mut total = 0u32;
    let step = (((sx1 - sx0) * (sy1 - sy0)) as f32 / 4000.0).sqrt().max(1.0) as u32; // ~≤4k samples
    let mut y = sy0;
    while y < sy1 {
        let mut x = sx0;
        while x < sx1 {
            let px = img.get_pixel(x, y);
            let (r, g, b) = (px[0] as f32 / 255.0, px[1] as f32 / 255.0, px[2] as f32 / 255.0);
            let max = r.max(g).max(b);
            let min = r.min(g).min(b);
            let v = max;
            let s_ = if max > 0.0 { (max - min) / max } else { 0.0 };
            let d = max - min;
            let hdeg = if d == 0.0 { 0.0 } else if max == r {
                60.0 * (((g - b) / d) % 6.0)
            } else if max == g {
                60.0 * (((b - r) / d) + 2.0)
            } else {
                60.0 * (((r - g) / d) + 4.0)
            };
            let hdeg = if hdeg < 0.0 { hdeg + 360.0 } else { hdeg };
            let idx = if v < 0.16 { 0 }                       // black
                else if s_ < 0.18 {
                    if v > 0.72 { 1 } else if v > 0.42 { 2 } else { 3 } // white/silver/gray
                } else {
                    match hdeg {
                        hh if !(15.0..345.0).contains(&hh) => 4,                        // red
                        hh if hh < 45.0 => if v < 0.45 { 10 } else { 5 },           // brown / orange
                        hh if hh < 70.0 => 6,                                        // yellow
                        hh if hh < 165.0 => 7,                                       // green
                        hh if hh < 262.0 => 8,                                       // blue
                        _ => 9,                                                      // purple
                    }
                };
            votes[idx] += 1;
            total += 1;
            x += step;
        }
        y += step;
    }
    if total < 200 { return None; }
    let (best, &n) = votes.iter().enumerate().max_by_key(|(_, n)| **n)?;
    let share = n as f32 / total as f32;
    // Plurality floor: a real dominant color owns the band. Below it we
    // return None (unknown) — never a coin-flip label.
    if share < 0.25 { return None; }
    Some((NAMES[best].to_string(), share))
}

pub(crate) fn classify_vehicle_color(jpeg: &[u8], bbox: [f32; 4]) -> Option<(String, f32)> {
    let img = image::load_from_memory(jpeg).ok()?.to_rgb8();
    // BODY band: skip the top 35% (windows/roof glare) and bottom 10%
    // (shadow/road); trim 12% off each side (mirrors/background bleed).
    dominant_color_in_band(&img, bbox, (0.35, 0.90, 0.12))
}


#[cfg(test)]
mod color_tests {
    use super::*;

    fn jpeg_of(rgb: [u8; 3]) -> Vec<u8> {
        let img = image::RgbImage::from_pixel(200, 150, image::Rgb(rgb));
        let mut buf = Vec::new();
        let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, 90);
        enc.encode_image(&image::DynamicImage::ImageRgb8(img)).unwrap();
        buf
    }
    const BBOX: [f32; 4] = [10.0, 10.0, 190.0, 140.0];

    #[test]
    fn red_car_is_red() {
        let (c, _) = classify_vehicle_color(&jpeg_of([200, 25, 25]), BBOX).expect("color");
        assert_eq!(c, "red");
    }
    #[test]
    fn white_car_is_white() {
        let (c, _) = classify_vehicle_color(&jpeg_of([235, 235, 238]), BBOX).expect("color");
        assert_eq!(c, "white");
    }
    #[test]
    fn black_car_is_black() {
        let (c, _) = classify_vehicle_color(&jpeg_of([18, 18, 22]), BBOX).expect("color");
        assert_eq!(c, "black");
    }
    #[test]
    fn blue_car_is_blue() {
        let (c, _) = classify_vehicle_color(&jpeg_of([30, 80, 200]), BBOX).expect("color");
        assert_eq!(c, "blue");
    }
    #[test]
    fn tiny_crop_abstains() {
        assert!(classify_vehicle_color(&jpeg_of([200, 25, 25]), [0.0, 0.0, 12.0, 12.0]).is_none());
    }
}
