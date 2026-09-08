//! Server-side multi-object tracker — **StrongSORT-style** (Kalman motion model +
//! OSNet appearance, gated Hungarian assignment) — plus ground-plane homography.
//!
//! Foundation for stable cross-frame identity (so people aren't ID-swapped when
//! they cross/occlude), zone speed, and line-crossing. We already compute an OSNet
//! appearance embedding per person per frame in `inference.rs`; this tracker
//! consumes it as the primary association cue, with motion (Kalman + Mahalanobis
//! gate) and a second-pass IoU fallback for objects with no embedding (e.g. cars,
//! or the HSV fallback). Falls back to plain IoU behaviour when no appearance is
//! available, so it's never worse than the old greedy-IoU tracker.
//!
//! Per camera we keep a set of Kalman tracks; each carries an EMA appearance
//! feature, a bounded history of normalised bottom-centre points (ground-contact,
//! for homography → metres/second), and a confirmed/tentative lifecycle.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use crate::inference::iou;

const MAX_HISTORY: usize = 30;
const IOU_MATCH:   f32 = 0.30;     // second-pass IoU gate
const N_INIT:      u32 = 3;        // hits before a track is "confirmed"
const MAX_AGE:     u32 = 15;       // frames a track may coast unmatched (~5 s @ 3 fps)
const MAX_WALL:    Duration = Duration::from_secs(8); // hard wall-clock safety cap
const APPEAR_MAX_COST: f64 = 0.50; // reject appearance matches looser than cos 0.50
const CHI2_4DOF:   f64 = 9.4877;   // Mahalanobis 0.95 gate, 4 dof
const LAMBDA:      f64 = 0.98;     // appearance vs motion weight in the cost
const BIG:         f64 = 1e6;      // gated / impossible pair cost

// Kalman std weights (DeepSORT convention).
const SPW: f64 = 1.0 / 20.0;
const SVW: f64 = 1.0 / 160.0;

#[derive(Clone, Copy)]
pub struct TrackPoint { pub x: f32, pub y: f32, pub t: Instant } // normalised 0..1 bottom-centre

struct Track {
    id:        u64,
    label:     String,
    mean:      Mat,           // 8×1 Kalman state [cx,cy,a,h,vcx,vcy,va,vh]
    cov:       Mat,           // 8×8 covariance
    feature:   Option<Vec<f32>>, // L2-normalised EMA appearance (deep descriptors only)
    hits:      u32,
    time_since_update: u32,
    confirmed: bool,
    history:   Vec<TrackPoint>,
    last_seen: Instant,
}

struct CamTracker { tracks: Vec<Track>, next_id: u64 }

static TRACKERS: OnceLock<Mutex<HashMap<u8, CamTracker>>> = OnceLock::new();

/// One detection after tracking: its stable id, label, bbox, and a *clone* of the
/// recent normalised bottom-centre history (so callers can compute speed/crossing
/// without holding the tracker lock).
pub struct TrackedDet {
    pub track_id: u64,
    pub label:    String,
    /// Current bounding box — populated for every result, available to per-box consumers
    /// (zone/overlay logic). The current pipeline reads `history` (track motion), not the
    /// box, so it isn't read yet.
    #[allow(dead_code)]
    pub bbox:     [f32; 4],
    pub history:  Vec<TrackPoint>,
}

/// Update a camera's tracker with this frame's detections (pixel `[x1,y1,x2,y2]`).
/// `descriptors` is parallel to `dets` — the per-person OSNet/HSV appearance vector
/// (or `None`). Returns one `TrackedDet` per input detection, in the same order.
pub fn update(
    cam_id: u8,
    dets: &[(String, f32, [f32; 4])],
    descriptors: &[Option<Vec<f32>>],
    orig_w: f32,
    orig_h: f32,
) -> Vec<TrackedDet> {
    let now = Instant::now();
    let cell = TRACKERS.get_or_init(|| Mutex::new(HashMap::new()));
    let Ok(mut map) = cell.lock() else {
        // Lock poisoned → degrade to untracked (never panic the inference loop).
        return dets.iter().map(|(l, _, b)| TrackedDet { track_id: 0, label: l.clone(), bbox: *b, history: Vec::new() }).collect();
    };
    let ct = map.entry(cam_id).or_insert_with(|| CamTracker { tracks: Vec::new(), next_id: 1 });

    // ── Predict every track forward one step ─────────────────────────────────
    for t in ct.tracks.iter_mut() {
        kalman_predict(&mut t.mean, &mut t.cov);
        t.time_since_update += 1;
    }

    let nd = dets.len();
    let mut result: Vec<Option<TrackedDet>> = (0..nd).map(|_| None).collect();
    let mut det_used = vec![false; nd];
    let mut track_matched = vec![false; ct.tracks.len()];

    // Snapshot per-track projection + feature for cost building (immutable phase).
    struct Snap { proj_mean: [f64; 4], sinv: Option<Mat>, feat: Option<Vec<f32>>, label: String }
    let snaps: Vec<Snap> = ct.tracks.iter().map(|t| {
        // Project for gating with conf=0 (full R → slightly looser, conservative gate).
        let (pm, s) = kalman_project(&t.mean, &t.cov, 0.0);
        Snap { proj_mean: pm, sinv: s.inv(), feat: t.feature.clone(), label: t.label.clone() }
    }).collect();

    // ── Pass 1: appearance-gated Hungarian over tracks-with-feature × dets ────
    let track_with_feat: Vec<usize> = (0..ct.tracks.len()).filter(|&i| snaps[i].feat.is_some()).collect();
    if !track_with_feat.is_empty() && nd > 0 {
        let rows = track_with_feat.len();
        let n = rows.max(nd); // pad to square
        let mut cost = vec![vec![BIG; n]; n];
        for (ri, &ti) in track_with_feat.iter().enumerate() {
            let s = &snaps[ti];
            let Some(sinv) = &s.sinv else { continue };
            let tf = s.feat.as_ref().unwrap();
            for (di, (label, _score, b)) in dets.iter().enumerate() {
                if &s.label != label { continue; }
                let Some(df) = descriptors.get(di).and_then(|d| d.as_ref()) else { continue };
                if df.len() != tf.len() || df.len() < 256 { continue; } // deep features only
                let z = bbox_to_z(b);
                let maha = mahalanobis(&s.proj_mean, sinv, &z);
                if maha > CHI2_4DOF { continue; } // motion gate
                let app = 1.0 - cos32(tf, df) as f64;
                if app > APPEAR_MAX_COST { continue; }
                cost[ri][di] = LAMBDA * app + (1.0 - LAMBDA) * (maha / CHI2_4DOF);
            }
        }
        let assign = hungarian(&cost);
        for (ri, &ti) in track_with_feat.iter().enumerate() {
            let di = assign[ri];
            if di >= nd || cost[ri][di] >= BIG { continue; }
            apply_match(ct, ti, di, dets, descriptors, orig_w, orig_h, now, &mut result);
            track_matched[ti] = true;
            det_used[di] = true;
        }
    }

    // ── Pass 2: IoU fallback for unmatched tracks × unmatched dets ────────────
    for ti in 0..ct.tracks.len() {
        if track_matched[ti] { continue; }
        let pbox = mean_to_bbox(&ct.tracks[ti].mean);
        let mut best: Option<usize> = None;
        let mut best_iou = IOU_MATCH;
        for di in 0..nd {
            if det_used[di] || dets[di].0 != ct.tracks[ti].label { continue; }
            let v = iou(&dets[di].2, &pbox);
            if v >= best_iou { best_iou = v; best = Some(di); }
        }
        if let Some(di) = best {
            apply_match(ct, ti, di, dets, descriptors, orig_w, orig_h, now, &mut result);
            track_matched[ti] = true;
            det_used[di] = true;
        }
    }

    // ── New tracks for still-unmatched detections ────────────────────────────
    for di in 0..nd {
        if det_used[di] { continue; }
        let (label, _score, b) = &dets[di];
        let z = bbox_to_z(b);
        let (mean, cov) = kalman_initiate(&z);
        let id = ct.next_id; ct.next_id += 1;
        let mut feature: Option<Vec<f32>> = None;
        if let Some(df) = descriptors.get(di).and_then(|d| d.as_ref()) {
            if df.len() >= 256 { let mut v = df.clone(); l2(&mut v); feature = Some(v); }
        }
        let mut history = Vec::new();
        push_history(&mut history, b, orig_w, orig_h, now);
        let out_hist = history.clone();
        ct.tracks.push(Track {
            id, label: label.clone(), mean, cov, feature,
            hits: 1, time_since_update: 0, confirmed: N_INIT <= 1, history, last_seen: now,
        });
        result[di] = Some(TrackedDet { track_id: id, label: label.clone(), bbox: *b, history: out_hist });
    }

    // ── Lifecycle: drop stale tracks ─────────────────────────────────────────
    ct.tracks.retain(|t| t.time_since_update <= MAX_AGE && now.duration_since(t.last_seen) < MAX_WALL);

    // Build output (one per det, in order). Any None (shouldn't happen) → untracked.
    result.into_iter().enumerate().map(|(di, r)| r.unwrap_or_else(|| TrackedDet {
        track_id: 0, label: dets[di].0.clone(), bbox: dets[di].2, history: Vec::new(),
    })).collect()
}

/// Apply a (track, det) match: Kalman-correct, EMA the appearance, push history,
/// confirm if mature, and record the output row.
#[allow(clippy::too_many_arguments)]
fn apply_match(
    ct: &mut CamTracker, ti: usize, di: usize,
    dets: &[(String, f32, [f32; 4])], descriptors: &[Option<Vec<f32>>],
    orig_w: f32, orig_h: f32, now: Instant,
    result: &mut [Option<TrackedDet>],
) {
    let (label, score, b) = &dets[di];
    let z = bbox_to_z(b);
    {
        let t = &mut ct.tracks[ti];
        kalman_update(&mut t.mean, &mut t.cov, &z, *score);
        if let Some(df) = descriptors.get(di).and_then(|d| d.as_ref()) {
            if df.len() >= 256 { ema_feature(&mut t.feature, df); }
        }
        t.hits += 1;
        t.time_since_update = 0;
        t.last_seen = now;
        if t.hits >= N_INIT { t.confirmed = true; }
        push_history(&mut t.history, b, orig_w, orig_h, now);
    }
    let t = &ct.tracks[ti];
    result[di] = Some(TrackedDet { track_id: t.id, label: label.clone(), bbox: *b, history: t.history.clone() });
}

fn push_history(history: &mut Vec<TrackPoint>, b: &[f32; 4], orig_w: f32, orig_h: f32, now: Instant) {
    let x = ((b[0] + b[2]) * 0.5 / orig_w.max(1.0)).clamp(0.0, 1.0);
    let y = (b[3] / orig_h.max(1.0)).clamp(0.0, 1.0);
    history.push(TrackPoint { x, y, t: now });
    if history.len() > MAX_HISTORY { history.remove(0); }
}

fn ema_feature(feat: &mut Option<Vec<f32>>, new: &[f32]) {
    match feat {
        Some(f) if f.len() == new.len() => {
            for i in 0..f.len() { f[i] = 0.9 * f[i] + 0.1 * new[i]; }
            l2(f);
        }
        _ => { let mut v = new.to_vec(); l2(&mut v); *feat = Some(v); }
    }
}

fn l2(v: &mut [f32]) {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 { v.iter_mut().for_each(|x| *x /= n); }
}

fn cos32(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

// ─── Box ↔ Kalman measurement conversions ─────────────────────────────────────

/// `[x1,y1,x2,y2]` → measurement `[cx, cy, a, h]` (a = w/h aspect, h = height).
fn bbox_to_z(b: &[f32; 4]) -> [f64; 4] {
    let w = (b[2] - b[0]).max(1.0) as f64;
    let h = (b[3] - b[1]).max(1.0) as f64;
    [((b[0] + b[2]) * 0.5) as f64, ((b[1] + b[3]) * 0.5) as f64, w / h, h]
}

fn mean_to_bbox(mean: &Mat) -> [f32; 4] {
    let cx = mean.at(0, 0); let cy = mean.at(1, 0);
    let a = mean.at(2, 0).max(0.01); let h = mean.at(3, 0).max(1.0);
    let w = a * h;
    [(cx - w * 0.5) as f32, (cy - h * 0.5) as f32, (cx + w * 0.5) as f32, (cy + h * 0.5) as f32]
}

// ─── Kalman filter (8-state constant-velocity, DeepSORT-style) ────────────────

fn motion_mat() -> Mat { let mut f = Mat::ident(8); for i in 0..4 { f.set(i, i + 4, 1.0); } f }

fn kalman_initiate(z: &[f64; 4]) -> (Mat, Mat) {
    let h = z[3].max(1.0);
    let mut mean = Mat::new(8, 1);
    mean.set(0, 0, z[0]); mean.set(1, 0, z[1]); mean.set(2, 0, z[2]); mean.set(3, 0, z[3]);
    let std = [2.0*SPW*h, 2.0*SPW*h, 1e-2, 2.0*SPW*h, 10.0*SVW*h, 10.0*SVW*h, 1e-5, 10.0*SVW*h];
    let mut cov = Mat::new(8, 8);
    for i in 0..8 { cov.set(i, i, std[i] * std[i]); }
    (mean, cov)
}

fn kalman_predict(mean: &mut Mat, cov: &mut Mat) {
    let h = mean.at(3, 0).max(1.0);
    let qs = [SPW*h, SPW*h, 1e-2, SPW*h, SVW*h, SVW*h, 1e-5, SVW*h];
    let mut q = Mat::new(8, 8);
    for i in 0..8 { q.set(i, i, qs[i] * qs[i]); }
    let f = motion_mat();
    *mean = f.mul(mean);
    *cov = f.mul(cov).mul(&f.t()).add(&q);
}

/// Project state to measurement space: returns (proj_mean[4], S=4×4 innovation cov).
/// `conf` (detection confidence, 0..1) scales R via NSA — higher conf → tighter R.
fn kalman_project(mean: &Mat, cov: &Mat, conf: f32) -> ([f64; 4], Mat) {
    let h = mean.at(3, 0).max(1.0);
    let rs = [SPW*h, SPW*h, 1e-1, SPW*h];
    let nsa = (1.0 - conf as f64).clamp(1e-2, 1.0); // never zero R
    let mut s = Mat::new(4, 4);
    for i in 0..4 { for j in 0..4 { s.set(i, j, cov.at(i, j)); } } // top-left 4×4 of cov
    for i in 0..4 { s.set(i, i, s.at(i, i) + rs[i] * rs[i] * nsa); }
    ([mean.at(0,0), mean.at(1,0), mean.at(2,0), mean.at(3,0)], s)
}

fn kalman_update(mean: &mut Mat, cov: &mut Mat, z: &[f64; 4], conf: f32) {
    let (pm, s) = kalman_project(mean, cov, conf);
    let Some(sinv) = s.inv() else { return };
    // PH^T = first 4 columns of cov (8×4).
    let mut pht = Mat::new(8, 4);
    for i in 0..8 { for j in 0..4 { pht.set(i, j, cov.at(i, j)); } }
    let k = pht.mul(&sinv); // 8×4 Kalman gain
    // innovation y = z - proj_mean (4×1)
    let mut y = Mat::new(4, 1);
    for i in 0..4 { y.set(i, 0, z[i] - pm[i]); }
    *mean = mean.add(&k.mul(&y));
    // cov = cov - K (H cov), where H cov = first 4 rows of cov (4×8).
    let mut hcov = Mat::new(4, 8);
    for i in 0..4 { for j in 0..8 { hcov.set(i, j, cov.at(i, j)); } }
    *cov = cov.sub(&k.mul(&hcov));
}

/// Mahalanobis² of measurement `z` against a projected track (mean[4], S^{-1}).
fn mahalanobis(proj_mean: &[f64; 4], sinv: &Mat, z: &[f64; 4]) -> f64 {
    let d = [z[0]-proj_mean[0], z[1]-proj_mean[1], z[2]-proj_mean[2], z[3]-proj_mean[3]];
    let mut acc = 0.0;
    for i in 0..4 { for j in 0..4 { acc += d[i] * sinv.at(i, j) * d[j]; } }
    acc
}

// ─── Minimal dense matrix (small, fixed use — no extra crates) ────────────────

#[derive(Clone)]
struct Mat { r: usize, c: usize, d: Vec<f64> }

impl Mat {
    fn new(r: usize, c: usize) -> Mat { Mat { r, c, d: vec![0.0; r * c] } }
    fn ident(n: usize) -> Mat { let mut m = Mat::new(n, n); for i in 0..n { m.set(i, i, 1.0); } m }
    #[inline] fn at(&self, i: usize, j: usize) -> f64 { self.d[i * self.c + j] }
    #[inline] fn set(&mut self, i: usize, j: usize, v: f64) { self.d[i * self.c + j] = v; }
    fn t(&self) -> Mat {
        let mut m = Mat::new(self.c, self.r);
        for i in 0..self.r { for j in 0..self.c { m.set(j, i, self.at(i, j)); } }
        m
    }
    fn mul(&self, o: &Mat) -> Mat {
        let mut m = Mat::new(self.r, o.c);
        for i in 0..self.r {
            for k in 0..self.c {
                let a = self.at(i, k);
                if a == 0.0 { continue; }
                for j in 0..o.c { m.d[i * o.c + j] += a * o.at(k, j); }
            }
        }
        m
    }
    fn add(&self, o: &Mat) -> Mat { let mut m = self.clone(); for k in 0..m.d.len() { m.d[k] += o.d[k]; } m }
    fn sub(&self, o: &Mat) -> Mat { let mut m = self.clone(); for k in 0..m.d.len() { m.d[k] -= o.d[k]; } m }
    /// Square-matrix inverse via Gauss-Jordan with partial pivoting. None if singular.
    fn inv(&self) -> Option<Mat> {
        if self.r != self.c { return None; }
        let n = self.r;
        let mut a = self.clone();
        let mut inv = Mat::ident(n);
        for col in 0..n {
            let mut piv = col;
            for r in (col + 1)..n { if a.at(r, col).abs() > a.at(piv, col).abs() { piv = r; } }
            if a.at(piv, col).abs() < 1e-12 { return None; }
            if piv != col {
                for j in 0..n { a.d.swap(col * n + j, piv * n + j); inv.d.swap(col * n + j, piv * n + j); }
            }
            let pv = a.at(col, col);
            for j in 0..n { a.d[col * n + j] /= pv; inv.d[col * n + j] /= pv; }
            for r in 0..n {
                if r == col { continue; }
                let f = a.at(r, col);
                if f == 0.0 { continue; }
                for j in 0..n { a.d[r * n + j] -= f * a.at(col, j); inv.d[r * n + j] -= f * inv.at(col, j); }
            }
        }
        Some(inv)
    }
}

// ─── Hungarian (Kuhn–Munkres) optimal assignment, O(n³), square cost ──────────
// Minimises total cost. `cost` is n×n; returns `row → col`. Standard e-maxx form.
fn hungarian(cost: &[Vec<f64>]) -> Vec<usize> {
    let n = cost.len();
    if n == 0 { return Vec::new(); }
    let inf = f64::INFINITY;
    let mut u = vec![0.0f64; n + 1];
    let mut v = vec![0.0f64; n + 1];
    let mut p = vec![0usize; n + 1]; // p[j] = row (1-indexed) matched to col j
    let mut way = vec![0usize; n + 1];
    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![inf; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = inf;
            let mut j1 = 0usize;
            for j in 1..=n {
                if !used[j] {
                    let cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                    if cur < minv[j] { minv[j] = cur; way[j] = j0; }
                    if minv[j] < delta { delta = minv[j]; j1 = j; }
                }
            }
            for j in 0..=n {
                if used[j] { u[p[j]] += delta; v[j] -= delta; }
                else { minv[j] -= delta; }
            }
            j0 = j1;
            if p[j0] == 0 { break; }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 { break; }
        }
    }
    let mut row_col = vec![usize::MAX; n];
    for j in 1..=n { if p[j] != 0 { row_col[p[j] - 1] = j - 1; } }
    row_col
}

// ─── 4-point homography (image quad → real rectangle in metres) ───────────────

/// Solve the 3×3 homography (`h8 == 1`, 8 unknowns) mapping the four normalised
/// image points `quad` to the real-world rectangle `[0,0]–[w_m,h_m]` (metres), in
/// the corner order top-left, top-right, bottom-right, bottom-left. Returns the 8
/// free coefficients `[h0..h7]`. None if degenerate.
pub fn homography(quad: &[(f32, f32); 4], w_m: f32, h_m: f32) -> Option<[f32; 8]> {
    let dst = [(0.0, 0.0), (w_m, 0.0), (w_m, h_m), (0.0, h_m)];
    // 8×8 system A·h = b.
    let mut a = [[0.0f64; 8]; 8];
    let mut b = [0.0f64; 8];
    for i in 0..4 {
        let (x, y) = (quad[i].0 as f64, quad[i].1 as f64);
        let (xp, yp) = (dst[i].0 as f64, dst[i].1 as f64);
        a[i * 2]     = [x, y, 1.0, 0.0, 0.0, 0.0, -x * xp, -y * xp];
        b[i * 2]     = xp;
        a[i * 2 + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -x * yp, -y * yp];
        b[i * 2 + 1] = yp;
    }
    let h = solve8(&mut a, &mut b)?;
    Some([h[0] as f32, h[1] as f32, h[2] as f32, h[3] as f32, h[4] as f32, h[5] as f32, h[6] as f32, h[7] as f32])
}

/// Map a normalised image point through the homography to real-world metres.
pub fn apply_h(h: &[f32; 8], x: f32, y: f32) -> (f32, f32) {
    let d = h[6] * x + h[7] * y + 1.0;
    if d.abs() < 1e-6 { return (0.0, 0.0); }
    ((h[0] * x + h[1] * y + h[2]) / d, (h[3] * x + h[4] * y + h[5]) / d)
}

/// Gaussian elimination with partial pivoting for an 8×8 system. Returns the
/// solution vector, or None if singular.
fn solve8(a: &mut [[f64; 8]; 8], b: &mut [f64; 8]) -> Option<[f64; 8]> {
    for col in 0..8 {
        // Pivot.
        let mut piv = col;
        for r in (col + 1)..8 { if a[r][col].abs() > a[piv][col].abs() { piv = r; } }
        if a[piv][col].abs() < 1e-9 { return None; }
        a.swap(col, piv); b.swap(col, piv);
        // Eliminate.
        for r in 0..8 {
            if r == col { continue; }
            let f = a[r][col] / a[col][col];
            if f == 0.0 { continue; }
            for c in col..8 { a[r][c] -= f * a[col][c]; }
            b[r] -= f * b[col];
        }
    }
    let mut x = [0.0f64; 8];
    for i in 0..8 { x[i] = b[i] / a[i][i]; }
    Some(x)
}

/// Point-in-polygon (normalised coords), ray-casting. `poly` is the quad/zone.
pub fn point_in_poly(x: f32, y: f32, poly: &[(f32, f32)]) -> bool {
    let n = poly.len();
    if n < 3 { return false; }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = poly[i];
        let (xj, yj) = poly[j];
        if ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi + f32::EPSILON) + xi) {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Median ground speed (km/h) of a track inside a speed zone, from its history
/// transformed to metres via `h`. Uses only points still inside the quad and the
/// most recent ~1.5 s. None if too little motion/history.
pub fn track_speed_kmh(history: &[TrackPoint], h: &[f32; 8], quad: &[(f32, f32)]) -> Option<f32> {
    let now = Instant::now();
    let pts: Vec<&TrackPoint> = history.iter()
        .filter(|p| now.duration_since(p.t) <= Duration::from_millis(1500) && point_in_poly(p.x, p.y, quad))
        .collect();
    if pts.len() < 2 { return None; }
    let mut speeds: Vec<f32> = Vec::new();
    for w in pts.windows(2) {
        let dt = w[1].t.duration_since(w[0].t).as_secs_f32();
        if dt <= 0.01 { continue; }
        let (x0, y0) = apply_h(h, w[0].x, w[0].y);
        let (x1, y1) = apply_h(h, w[1].x, w[1].y);
        let d = ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt(); // metres
        speeds.push(d / dt * 3.6); // m/s → km/h
    }
    if speeds.is_empty() { return None; }
    speeds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(speeds[speeds.len() / 2])
}

/// If the last two history points cross the segment `a`→`b` (normalised), return
/// the direction: `1` for the a→b side, `-1` for b→a. None otherwise.
pub fn line_cross(history: &[TrackPoint], a: (f32, f32), b: (f32, f32)) -> Option<i8> {
    if history.len() < 2 { return None; }
    let p0 = history[history.len() - 2];
    let p1 = history[history.len() - 1];
    let side = |px: f32, py: f32| (b.0 - a.0) * (py - a.1) - (b.1 - a.1) * (px - a.0);
    let s0 = side(p0.x, p0.y);
    let s1 = side(p1.x, p1.y);
    if s0 == 0.0 || s1 == 0.0 || (s0 > 0.0) == (s1 > 0.0) { return None; } // no side change
    // Ensure the crossing is within the segment span (not the infinite line).
    if !segments_intersect((p0.x, p0.y), (p1.x, p1.y), a, b) { return None; }
    Some(if s1 > 0.0 { 1 } else { -1 })
}

fn segments_intersect(p: (f32, f32), q: (f32, f32), a: (f32, f32), b: (f32, f32)) -> bool {
    let d = |o: (f32, f32), x: (f32, f32), y: (f32, f32)|
        (y.0 - x.0) * (o.1 - x.1) - (y.1 - x.1) * (o.0 - x.0);
    let d1 = d(p, a, b); let d2 = d(q, a, b);
    let d3 = d(a, p, q); let d4 = d(b, p, q);
    ((d1 > 0.0) != (d2 > 0.0)) && ((d3 > 0.0) != (d4 > 0.0))
}
