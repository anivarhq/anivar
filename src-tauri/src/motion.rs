//! standard motion detection.
//!
//! * [`decode_to_gray_bytes`]    — JPEG bytes → grayscale buffer.
//! * [`box_blur_3x3`]            — 3×3 averaging filter applied BEFORE diffing.
//!                                  Suppresses JPEG / sensor / auto-exposure noise.
//! * [`compute_motion_masked`]   — per-pixel grayscale diff with the mask
//!                                  applied to the diff (not to the input frames).
//!                                  Returns `(motion_score, motion_regions)`.
//! * [`build_mask_buffer`]       — rasterises one or more polygon masks into a
//!                                  per-pixel boolean buffer at frame resolution.
//! * [`is_point_in_polygon`]    — ray-casting test used by mask filtering at
//!                                  YOLO-detection time too (cf. [`crate::filter_detections_by_masks`]).

/// Decode JPEG bytes to a grayscale buffer. The detection hot path passes the
/// bytes it already base64-decoded, so the frame is never decoded from base64 twice.
pub(crate) fn decode_to_gray_bytes(bytes: &[u8]) -> anyhow::Result<(u32, u32, Vec<u8>)> {
    let img = image::load_from_memory(bytes)?.grayscale();
    let (w, h) = (img.width(), img.height());
    Ok((w, h, img.to_luma8().into_raw()))
}

/// Box-sample a grayscale buffer down so its width is ≤ `max_w` (integer
/// factor, averaged blocks). Motion diffing needs nowhere near full frame
/// resolution — mature NVRs run motion on the low-res detect stream — and blur +
/// diff are O(pixels), so a 720p→320w downscale cuts that work ~8x. Regions
/// come out normalized either way, so downstream is unaffected.
pub(crate) fn downscale_gray(src: &[u8], w: u32, h: u32, max_w: u32) -> (u32, u32, Vec<u8>) {
    if w <= max_w || w == 0 || h == 0 || src.len() != (w * h) as usize {
        return (w, h, src.to_vec());
    }
    let f = w.div_ceil(max_w).max(2);
    let nw = (w / f).max(1);
    let nh = (h / f).max(1);
    let mut out = vec![0u8; (nw * nh) as usize];
    for y in 0..nh {
        for x in 0..nw {
            let mut sum: u32 = 0;
            for dy in 0..f {
                let row = (y * f + dy).min(h - 1) * w;
                for dx in 0..f {
                    sum += src[(row + (x * f + dx).min(w - 1)) as usize] as u32;
                }
            }
            out[(y * nw + x) as usize] = (sum / (f * f)) as u8;
        }
    }
    (nw, nh, out)
}

/// Ray-casting point-in-polygon test (mature NVRs' same algorithm).
/// `poly` points are normalized 0–1. `px`, `py` are also normalized.
pub(crate) fn is_point_in_polygon(px: f32, py: f32, poly: &[(f32, f32)]) -> bool {
    let n = poly.len();
    if n < 3 { return false; }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = poly[i];
        let (xj, yj) = poly[j];
        if ((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// 3x3 box blur to suppress JPEG / sensor / auto-exposure noise before diffing.
/// Without this, individual noisy pixels routinely exceed the diff threshold
/// and cumulative noise across the unmasked area drives motion_score over the
/// sensitivity setting -- even when nothing in the scene is actually moving.
pub(crate) fn box_blur_3x3(src: &[u8], w: u32, h: u32) -> Vec<u8> {
    let n = (w * h) as usize;
    if src.len() != n || w < 3 || h < 3 { return src.to_vec(); }
    let mut out = vec![0u8; n];
    let wu = w as usize;
    let hu = h as usize;
    for x in 0..wu { out[x] = src[x]; out[(hu - 1) * wu + x] = src[(hu - 1) * wu + x]; }
    for y in 0..hu { out[y * wu] = src[y * wu]; out[y * wu + wu - 1] = src[y * wu + wu - 1]; }
    for y in 1..hu - 1 {
        let r0 = (y - 1) * wu;
        let r1 = y * wu;
        let r2 = (y + 1) * wu;
        for x in 1..wu - 1 {
            let sum = src[r0 + x - 1] as u32 + src[r0 + x] as u32 + src[r0 + x + 1] as u32
                    + src[r1 + x - 1] as u32 + src[r1 + x] as u32 + src[r1 + x + 1] as u32
                    + src[r2 + x - 1] as u32 + src[r2 + x] as u32 + src[r2 + x + 1] as u32;
            out[r1 + x] = (sum / 9) as u8;
        }
    }
    out
}

/// standard motion detection: per-pixel grayscale diff with mask applied
/// to the diff itself. Masked pixels are skipped entirely -- they cannot
/// contribute to the motion score. Frames are box-blurred before diffing so
/// per-pixel JPEG / sensor noise doesn't register as motion.
pub(crate) fn compute_motion_masked(
    prev_raw: &[u8],
    curr_raw: &[u8],
    mask: &[bool],
    threshold: u8,
    lightning_threshold: f32,
    w: u32,
    h: u32,
) -> (f32, Vec<[f32; 4]>) {
    let n = curr_raw.len();
    if n == 0 || n != prev_raw.len() || n != (w * h) as usize { return (0.0, vec![]); }

    let prev_b = box_blur_3x3(prev_raw, w, h);
    let curr_b = box_blur_3x3(curr_raw, w, h);
    let prev = prev_b.as_slice();
    let curr = curr_b.as_slice();

    const COLS: usize = 8;
    const ROWS: usize = 6;
    let mut tile_fg    = [0u32; COLS * ROWS];
    let mut tile_total = [0u32; COLS * ROWS];

    let mut total_fg:       u32 = 0;
    let mut total_unmasked: u32 = 0;
    let mask_n = mask.len();

    for i in 0..n {
        if i < mask_n && mask[i] { continue; }
        total_unmasked += 1;
        let px = (i as u32) % w;
        let py = (i as u32) / w;
        let tc = ((px as usize * COLS) / w as usize).min(COLS - 1);
        let tr = ((py as usize * ROWS) / h as usize).min(ROWS - 1);
        let ti = tr * COLS + tc;
        tile_total[ti] += 1;
        let d = prev[i].abs_diff(curr[i]);
        if d > threshold {
            total_fg += 1;
            tile_fg[ti] += 1;
        }
    }
    if total_unmasked == 0 { return (0.0, vec![]); }
    let global = total_fg as f32 / total_unmasked as f32;

    // Mature NVRs `lightning_threshold`: when a huge fraction of the (unmasked) frame
    // changes AT ONCE, that's a lighting / IR day-night / auto-exposure shift, not
    // real motion — counting it would open a bogus event covering the whole frame.
    // Ignore this frame; the caller updates prev→curr each tick, so the next diff is
    // against the new baseline (mature NVRs' "recalibrate"). 0 disables the guard.
    if lightning_threshold > 0.0 && global > lightning_threshold {
        return (0.0, vec![]);
    }

    let mut active = [false; COLS * ROWS];
    let mut peak = 0.0f32;
    for ti in 0..(COLS * ROWS) {
        if tile_total[ti] < 20 { continue; }
        let s = tile_fg[ti] as f32 / tile_total[ti] as f32;
        if s > peak { peak = s; }
        if s > 0.08 { active[ti] = true; }
    }
    let motion_score = global.max(peak * 0.25);

    let mut regions: Vec<[f32; 4]> = Vec::new();
    if active.iter().any(|&a| a) {
        let ar: Vec<usize> = (0..ROWS).filter(|&r| (0..COLS).any(|c| active[r * COLS + c])).collect();
        let ac: Vec<usize> = (0..COLS).filter(|&c| (0..ROWS).any(|r| active[r * COLS + c])).collect();
        if !ar.is_empty() && !ac.is_empty() {
            let pad = 0.08f32;
            regions.push([
                (*ac.first().unwrap() as f32 / COLS as f32 - pad).max(0.0),
                (*ar.first().unwrap() as f32 / ROWS as f32 - pad).max(0.0),
                ((*ac.last().unwrap() + 1) as f32 / COLS as f32 + pad).min(1.0),
                ((*ar.last().unwrap() + 1) as f32 / ROWS as f32 + pad).min(1.0),
            ]);
        }
    }
    (motion_score, regions)
}


/// Build a per-pixel boolean mask from one or more polygons (normalised coords).
/// `true` = pixel is inside at least one polygon and should be skipped by motion detection.
pub(crate) fn build_mask_buffer(polys: &[Vec<(f32, f32)>], w: u32, h: u32) -> Vec<bool> {
    let n = (w * h) as usize;
    let mut mask = vec![false; n];
    if polys.is_empty() || w == 0 || h == 0 { return mask; }

    for poly in polys {
        if poly.len() < 3 { continue; }
        let min_x = poly.iter().map(|(x, _)| *x).fold(f32::INFINITY, f32::min).max(0.0);
        let max_x = poly.iter().map(|(x, _)| *x).fold(f32::NEG_INFINITY, f32::max).min(1.0);
        let min_y = poly.iter().map(|(_, y)| *y).fold(f32::INFINITY, f32::min).max(0.0);
        let max_y = poly.iter().map(|(_, y)| *y).fold(f32::NEG_INFINITY, f32::max).min(1.0);
        let col_start = (min_x * w as f32) as u32;
        let col_end   = ((max_x * w as f32) as u32 + 1).min(w);
        let row_start = (min_y * h as f32) as u32;
        let row_end   = ((max_y * h as f32) as u32 + 1).min(h);
        for row in row_start..row_end {
            let py = (row as f32 + 0.5) / h as f32;
            for col in col_start..col_end {
                let px = (col as f32 + 0.5) / w as f32;
                if is_point_in_polygon(px, py, poly) {
                    mask[(row * w + col) as usize] = true;
                }
            }
        }
    }
    mask
}