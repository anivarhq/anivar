//! Body keypoints — MoveNet SinglePose Lightning (Apache-2.0; trained on COCO +
//! Google's own "Active" set), top-down on ONE person crop. Pose runs only where
//! it earns its cost: behaviour candidates (person down, climbing) and the few
//! crops a track keeps for clothing colour.
//!
//! I/O VERIFIED 2026-09-15 with onnxruntime on the Xenova export: input int32
//! NHWC `[1,192,192,3]` (raw 0–255 RGB), output float `[1,1,17,3]` =
//! (y, x, score) normalised to the 192×192 input. ~6 ms on CPU.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use ort::session::Session as OrtSession;
use ort::value::Tensor;

pub(crate) const SKILL: &str = "pose_movenet";
const SIZE: u32 = 192;
/// Keypoints scoring below this are treated as not seen.
pub(crate) const MIN_KP: f32 = 0.3;

// COCO-17 keypoint indices.
const L_SHOULDER: usize = 5;
const R_SHOULDER: usize = 6;
const L_HIP: usize = 11;
const R_HIP: usize = 12;
const L_KNEE: usize = 13;
const R_KNEE: usize = 14;
const L_ANKLE: usize = 15;
const R_ANKLE: usize = 16;

/// 17 keypoints as (x, y, score), in the pixel space of the image passed in.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Pose { pub kp: [[f32; 3]; 17] }

static SESSION: OnceLock<Mutex<Option<(OrtSession, String)>>> = OnceLock::new();

fn model_path(data_dir: &Path) -> PathBuf {
    data_dir.join("skills").join(SKILL).join("model.onnx")
}

pub(crate) fn is_installed(data_dir: &Path) -> bool {
    std::fs::metadata(model_path(data_dir)).map(|m| m.len() > 64 * 1024).unwrap_or(false)
}

/// Scale and padding that letterbox (w, h) into SIZE×SIZE.
fn letterbox(w: u32, h: u32) -> (f32, f32, f32) {
    let scale = SIZE as f32 / w.max(h).max(1) as f32;
    let pad_x = ((SIZE as f32 - w as f32 * scale) / 2.0).floor();
    let pad_y = ((SIZE as f32 - h as f32 * scale) / 2.0).floor();
    (scale, pad_x, pad_y)
}

/// Keypoints for the person filling `img` (a person crop). `None` when the model
/// isn't installed or can't run — callers treat that as "no pose evidence".
pub(crate) fn estimate(data_dir: &Path, img: &image::RgbImage) -> Option<Pose> {
    if !is_installed(data_dir) { return None; }
    let (w, h) = img.dimensions();
    if w < 16 || h < 16 { return None; }
    let (scale, pad_x, pad_y) = letterbox(w, h);
    let rw = ((w as f32 * scale).round() as u32).clamp(1, SIZE);
    let rh = ((h as f32 * scale).round() as u32).clamp(1, SIZE);
    let resized = image::imageops::resize(img, rw, rh, image::imageops::FilterType::Triangle);
    let (ox, oy) = (pad_x as u32, pad_y as u32);
    let mut data = vec![0i32; (SIZE * SIZE * 3) as usize];
    for (x, y, p) in resized.enumerate_pixels() {
        let (px, py) = (x + ox, y + oy);
        if px >= SIZE || py >= SIZE { continue; }
        let i = ((py * SIZE + px) * 3) as usize;
        data[i] = p[0] as i32;
        data[i + 1] = p[1] as i32;
        data[i + 2] = p[2] as i32;
    }

    let cell = SESSION.get_or_init(|| Mutex::new(None));
    let mut guard = cell.lock().ok()?;
    if guard.is_none() {
        match crate::inference::build_ort_session_cpu(&model_path(data_dir)) {
            Ok(s) => {
                let input = s.inputs().first().map(|i| i.name().to_string())
                    .unwrap_or_else(|| "input".into());
                *guard = Some((s, input));
            }
            Err(e) => { tracing::debug!("pose model load failed: {e}"); return None; }
        }
    }
    let (session, input_name) = guard.as_mut()?;
    let tensor = Tensor::<i32>::from_array(([1usize, SIZE as usize, SIZE as usize, 3], data)).ok()?;
    let outputs = { let _t = crate::inference::infer_timer("pose");
        session.run(ort::inputs![input_name.as_str() => tensor]).ok()? };
    let (_, raw) = outputs[0].try_extract_tensor::<f32>().ok()?;
    if raw.len() < 51 { return None; }
    Some(Pose { kp: std::array::from_fn(|k| {
        let (yn, xn, s) = (raw[k * 3], raw[k * 3 + 1], raw[k * 3 + 2]);
        unletterbox(xn, yn, s, scale, pad_x, pad_y)
    }) })
}

fn unletterbox(xn: f32, yn: f32, score: f32, scale: f32, pad_x: f32, pad_y: f32) -> [f32; 3] {
    [(xn * SIZE as f32 - pad_x) / scale, (yn * SIZE as f32 - pad_y) / scale, score]
}

impl Pose {
    fn seen(&self, i: usize) -> Option<(f32, f32)> {
        let p = self.kp[i];
        (p[2] >= MIN_KP).then_some((p[0], p[1]))
    }

    /// Midpoint of a left/right pair, or whichever side is visible.
    fn mid(&self, a: usize, b: usize) -> Option<(f32, f32)> {
        match (self.seen(a), self.seen(b)) {
            (Some(p), Some(q)) => Some(((p.0 + q.0) / 2.0, (p.1 + q.1) / 2.0)),
            (p, q) => p.or(q),
        }
    }

    /// Torso angle from vertical in degrees: 0 = upright, 90 = lying flat.
    /// Rotation-invariant to the crop, so it compares across frames of a track.
    pub(crate) fn torso_angle(&self) -> Option<f32> {
        let (sx, sy) = self.mid(L_SHOULDER, R_SHOULDER)?;
        let (hx, hy) = self.mid(L_HIP, R_HIP)?;
        let (dx, dy) = ((hx - sx).abs(), (hy - sy).abs());
        if dx + dy < 2.0 { return None; }
        Some(dx.atan2(dy).to_degrees())
    }

    /// Shoulders→hips box, shrunk 15% — the shirt, without arms or background.
    pub(crate) fn torso_box(&self) -> Option<[f32; 4]> {
        self.box_of(&[L_SHOULDER, R_SHOULDER, L_HIP, R_HIP], 0.15)
    }

    /// Hips→ankles (or knees) box, shrunk 10% — the trousers/skirt.
    pub(crate) fn legs_box(&self) -> Option<[f32; 4]> {
        self.box_of(&[L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE], 0.10)
    }

    /// Height (smallest y) of the highest visible ankle.
    pub(crate) fn highest_ankle_y(&self) -> Option<f32> {
        [L_ANKLE, R_ANKLE].iter().filter_map(|&i| self.seen(i)).map(|p| p.1)
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    }

    fn box_of(&self, idx: &[usize], shrink: f32) -> Option<[f32; 4]> {
        let pts: Vec<(f32, f32)> = idx.iter().filter_map(|&i| self.seen(i)).collect();
        if pts.len() < 3 { return None; }
        let x0 = pts.iter().map(|p| p.0).fold(f32::MAX, f32::min);
        let x1 = pts.iter().map(|p| p.0).fold(f32::MIN, f32::max);
        let y0 = pts.iter().map(|p| p.1).fold(f32::MAX, f32::min);
        let y1 = pts.iter().map(|p| p.1).fold(f32::MIN, f32::max);
        let (w, h) = (x1 - x0, y1 - y0);
        if w < 2.0 || h < 2.0 { return None; }
        Some([x0 + w * shrink, y0 + h * shrink, x1 - w * shrink, y1 - h * shrink])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pose(points: &[(usize, f32, f32)]) -> Pose {
        let mut kp = [[0.0f32; 3]; 17];
        for &(i, x, y) in points { kp[i] = [x, y, 0.9]; }
        Pose { kp }
    }

    #[test]
    fn upright_and_lying_torsos_read_as_such() {
        let standing = pose(&[(L_SHOULDER, 40.0, 50.0), (R_SHOULDER, 60.0, 50.0),
                              (L_HIP, 42.0, 110.0), (R_HIP, 58.0, 110.0)]);
        assert!(standing.torso_angle().unwrap() < 10.0);
        let lying = pose(&[(L_SHOULDER, 20.0, 100.0), (R_SHOULDER, 22.0, 112.0),
                           (L_HIP, 90.0, 102.0), (R_HIP, 92.0, 114.0)]);
        assert!(lying.torso_angle().unwrap() > 80.0);
        // No hips seen → no angle, never a guess.
        let partial = pose(&[(L_SHOULDER, 40.0, 50.0), (R_SHOULDER, 60.0, 50.0)]);
        assert!(partial.torso_angle().is_none());
    }

    #[test]
    fn torso_box_sits_inside_the_keypoints() {
        let p = pose(&[(L_SHOULDER, 40.0, 50.0), (R_SHOULDER, 80.0, 50.0),
                       (L_HIP, 44.0, 150.0), (R_HIP, 76.0, 150.0)]);
        let b = p.torso_box().unwrap();
        assert!(b[0] > 40.0 && b[2] < 80.0 && b[1] > 50.0 && b[3] < 150.0);
        assert!(pose(&[(L_HIP, 1.0, 1.0)]).torso_box().is_none(), "two points are not a torso");
    }

    #[test]
    fn letterbox_mapping_round_trips() {
        let (w, h) = (100u32, 200u32);
        let (scale, px, py) = letterbox(w, h);
        // A point at (50, 120) in the crop, as the model would report it.
        let xn = (50.0 * scale + px) / SIZE as f32;
        let yn = (120.0 * scale + py) / SIZE as f32;
        let back = unletterbox(xn, yn, 1.0, scale, px, py);
        assert!((back[0] - 50.0).abs() < 0.01 && (back[1] - 120.0).abs() < 0.01);
    }
}
