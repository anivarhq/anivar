//! Person tracks — one row per person's continuous presence on one camera.
//!
//! This is PP-Human's "ID-based" unit, and the thing the People section was
//! missing: identity, clothing, attributes and appearance used to be decided per
//! FRAME and scattered across `face_embeddings`, `body_embeddings` and one crop
//! per event. The inference loop now feeds per-frame observations here; when a
//! track goes quiet it is voted once (identity consensus, quality-weighted
//! clothing colours, appearance mean) and written to `person_tracks` — the table
//! Today, Search and person alerts read.
//!
//! Cheap by construction: observing is a map update; a crop is cut only when it
//! would enter the track's top-3 by quality, at most once per `CROP_EVERY`; all
//! decoding and voting happens once, at flush, off the async runtime.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};

use crate::face::IdentityVote;
use crate::AppState;

/// Crops kept per track (PP-Human MTMCT keeps the top 5; 3 is plenty for a vote).
const TOP_CROPS: usize = 3;
/// A track unseen this long has ended (matches tracking.rs `MAX_WALL`).
const QUIET: Duration = Duration::from_secs(8);
/// Someone lingering for ten minutes is written out in ten-minute rows, so Today
/// shows them before they leave and one row never spans an afternoon.
const SPLIT: Duration = Duration::from_secs(600);
/// Minimum spacing between crops of one track — a crop costs a frame decode.
const CROP_EVERY: Duration = Duration::from_millis(700);
/// Observations below this are tracker noise, not a person worth a row.
const MIN_OBSERVATIONS: u32 = 3;
/// Crops below this quality never vote on clothing colour or attributes (see
/// `quality`). A wide box cut off by the frame edge scores ~0.25, a small
/// standing person ~0.4.
const MIN_VOTE_QUALITY: f32 = 0.3;

use crate::alpr::COLOR_NAMES;

/// One person detection this frame, after tracking.
pub(crate) struct Observation<'a> {
    pub cam_id: u8,
    pub track_id: u64,
    pub event_id: Option<&'a str>,
    /// Pixel box `[x1, y1, x2, y2]`.
    pub bbox: [f32; 4],
    pub score: f32,
    pub frame_w: f32,
    pub frame_h: f32,
    /// Largest IoU with any OTHER person box this frame (occlusion proxy).
    pub overlap: f32,
    /// The tracklet's body identity (`body_*` / `kp_<id>`), when one is committed.
    pub body_id: Option<&'a str>,
    pub body_score: Option<f32>,
    /// Appearance descriptor from a RELIABLE crop only (never a tiny/blurry one).
    pub descriptor: Option<&'a [f32]>,
}

#[derive(Clone)]
pub(crate) struct Crop { pub quality: f32, pub b64: String }

pub(crate) struct Agg {
    pub cam_id: u8,
    pub track_id: u64,
    pub event_id: Option<String>,
    pub started: DateTime<Utc>,
    pub last: DateTime<Utc>,
    first_seen: Instant,
    last_seen: Instant,
    last_crop: Option<Instant>,
    pub observations: u32,
    pub crops: Vec<Crop>,
    reid_sum: Vec<f32>,
    reid_n: u32,
    pub body_id: Option<String>,
    pub body_score: Option<f32>,
    /// person_id → (name, vote) from faces located inside this track's box.
    pub faces: HashMap<String, (String, IdentityVote)>,
    /// Behaviours confirmed on this track ("loitering", "running", …).
    pub behaviours: Vec<&'static str>,
}

impl Agg {
    fn new(cam_id: u8, track_id: u64, now: Instant) -> Self {
        let t = Utc::now();
        Agg {
            cam_id, track_id, event_id: None, started: t, last: t,
            first_seen: now, last_seen: now, last_crop: None, observations: 0,
            crops: Vec::new(), reid_sum: Vec::new(), reid_n: 0,
            body_id: None, body_score: None, faces: HashMap::new(), behaviours: Vec::new(),
        }
    }

    /// Mean appearance descriptor, L2-normalised; `None` with no reliable crop.
    pub fn reid_mean(&self) -> Option<Vec<f32>> {
        if self.reid_n == 0 { return None; }
        let mut v = self.reid_sum.clone();
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if n <= 0.0 { return None; }
        v.iter_mut().for_each(|x| *x /= n);
        Some(v)
    }
}

type Key = (u8, u64);
static TRACKS: OnceLock<Mutex<HashMap<Key, Agg>>> = OnceLock::new();

fn tracks() -> &'static Mutex<HashMap<Key, Agg>> {
    TRACKS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// How good a crop is for identity, colour and attributes, 0..1. Tall, confident,
/// un-occluded, fully-in-frame boxes win; a truncated or crowded box is kept only
/// when nothing better turns up.
pub(crate) fn quality(bbox: &[f32; 4], score: f32, frame_w: f32, frame_h: f32, overlap: f32) -> f32 {
    let w = (bbox[2] - bbox[0]).max(0.0);
    let h = (bbox[3] - bbox[1]).max(0.0);
    if w < 8.0 || h < 16.0 { return 0.0; }
    let size = (h / 192.0).min(1.0);
    let edge = 2.0;
    let truncated = bbox[0] <= edge || bbox[1] <= edge || bbox[2] >= frame_w - edge || bbox[3] >= frame_h - edge;
    let trunc = if truncated { 0.6 } else { 1.0 };
    // A standing person is taller than wide; a wide box is a merge or a crouch.
    let shape = if h >= w * 1.1 { 1.0 } else { 0.5 };
    let clear = (1.0 - overlap).clamp(0.0, 1.0);
    score.clamp(0.0, 1.0) * size * trunc * shape * clear
}

/// Insert keeping the best `TOP_CROPS`, best first.
fn push_crop(crops: &mut Vec<Crop>, c: Crop) {
    let at = crops.iter().position(|x| x.quality < c.quality).unwrap_or(crops.len());
    crops.insert(at, c);
    crops.truncate(TOP_CROPS);
}

/// Record one observation. Cuts a crop from `frame_jpeg` only when it would make
/// the track's top-3 — the decode happens outside the lock.
pub(crate) fn observe(o: Observation, frame_jpeg: &[u8]) {
    let now = Instant::now();
    let q = quality(&o.bbox, o.score, o.frame_w, o.frame_h, o.overlap);
    let wants_crop = {
        let Ok(mut map) = tracks().lock() else { return };
        let a = map.entry((o.cam_id, o.track_id)).or_insert_with(|| Agg::new(o.cam_id, o.track_id, now));
        a.last_seen = now;
        a.last = Utc::now();
        a.observations += 1;
        if let Some(e) = o.event_id { a.event_id = Some(e.to_string()); }
        if let Some(b) = o.body_id {
            a.body_id = Some(b.to_string());
            if o.body_score.is_some() { a.body_score = o.body_score; }
        }
        if let Some(d) = o.descriptor {
            if a.reid_sum.is_empty() { a.reid_sum = vec![0.0; d.len()]; }
            if a.reid_sum.len() == d.len() {
                a.reid_sum.iter_mut().zip(d).for_each(|(s, x)| *s += x);
                a.reid_n += 1;
            }
        }
        let beats = a.crops.len() < TOP_CROPS || a.crops.last().is_some_and(|c| q > c.quality);
        let spaced = a.last_crop.map_or(true, |t| now.duration_since(t) >= CROP_EVERY);
        let want = q > 0.0 && beats && spaced;
        if want { a.last_crop = Some(now); }
        want
    };
    if !wants_crop { return; }
    let Some(b64) = crate::reid::crop_person_jpeg(frame_jpeg, &o.bbox) else { return };
    if let Ok(mut map) = tracks().lock() {
        if let Some(a) = map.get_mut(&(o.cam_id, o.track_id)) {
            push_crop(&mut a.crops, Crop { quality: q, b64 });
        }
    }
}

/// A face located inside this track's box (named or not — the vote decides).
pub(crate) fn note_face(cam_id: u8, track_id: u64, person_id: &str, name: &str, score: f32) {
    if person_id.is_empty() || track_id == 0 { return; }
    let Ok(mut map) = tracks().lock() else { return };
    if let Some(a) = map.get_mut(&(cam_id, track_id)) {
        let e = a.faces.entry(person_id.to_string()).or_insert_with(|| (name.to_string(), IdentityVote::default()));
        e.1.add(score);
    }
}

/// Record a confirmed behaviour on the track, for its row and for search.
pub(crate) fn note_behaviour(cam_id: u8, track_id: u64, what: &'static str) {
    let Ok(mut map) = tracks().lock() else { return };
    if let Some(a) = map.get_mut(&(cam_id, track_id)) {
        if !a.behaviours.contains(&what) { a.behaviours.push(what); }
    }
}

/// Who this track is RIGHT NOW, if a face has reached consensus — the same
/// decision `resolve_identity` makes at flush, available while the person is
/// still in view (alert rules such as "unfamiliar people only" need it live).
pub(crate) fn known_person(cam_id: u8, track_id: u64, rec_threshold: f32) -> Option<(String, String)> {
    let map = tracks().lock().ok()?;
    let a = map.get(&(cam_id, track_id))?;
    a.faces.iter()
        .filter(|(_, (_, v))| v.confirmed(rec_threshold))
        .max_by(|x, y| x.1.1.avg().partial_cmp(&y.1.1.avg()).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(pid, (name, _))| (pid.clone(), name.clone()))
}

/// Remove and return tracks that ended (quiet) or ran past `SPLIT`. A split track
/// keeps going under the same key with its identity carried over.
pub(crate) fn take_due() -> Vec<Agg> {
    let now = Instant::now();
    let Ok(mut map) = tracks().lock() else { return Vec::new() };
    let due: Vec<Key> = map.iter()
        .filter(|(_, a)| now.duration_since(a.last_seen) >= QUIET || now.duration_since(a.first_seen) >= SPLIT)
        .map(|(k, _)| *k)
        .collect();
    let mut out = Vec::with_capacity(due.len());
    for k in due {
        let Some(a) = map.remove(&k) else { continue };
        if now.duration_since(a.last_seen) < QUIET {
            // Split: continue the same presence in a fresh row.
            let mut next = Agg::new(a.cam_id, a.track_id, now);
            next.event_id = a.event_id.clone();
            next.body_id = a.body_id.clone();
            next.body_score = a.body_score;
            next.faces = a.faces.clone();
            map.insert(k, next);
        }
        out.push(a);
    }
    out
}

/// Who this track was, decided once. A consensus-confirmed FACE names the track;
/// a body match to a known person only PROPOSES (it rides in `body_person_id`,
/// and the UI says "Maybe X?") — body similarity never commits a name, and a
/// look-alike stranger must never be treated as a resident by alert rules.
pub(crate) fn resolve_identity(
    faces: &HashMap<String, (String, IdentityVote)>,
    body_id: Option<&str>,
    body_score: Option<f32>,
    rec_threshold: f32,
) -> (Option<String>, &'static str, Option<f32>) {
    let best = faces.iter()
        .filter(|(_, (_, v))| v.confirmed(rec_threshold))
        .max_by(|a, b| a.1.1.avg().partial_cmp(&b.1.1.avg()).unwrap_or(std::cmp::Ordering::Equal));
    if let Some((pid, (_, v))) = best {
        return (Some(pid.clone()), "face", Some(v.avg()));
    }
    match body_id {
        Some(b) if b.starts_with("kp_") => (None, "body_reid", body_score),
        _ => (None, "none", None),
    }
}

/// Colour name for a band's 11-bin share vector: a clear winner or nothing.
/// NULL means UNKNOWN, never "not red".
pub(crate) fn name_from_shares(shares: &[f32; 11]) -> Option<&'static str> {
    let mut idx: Vec<usize> = (0..11).collect();
    idx.sort_by(|&a, &b| shares[b].partial_cmp(&shares[a]).unwrap_or(std::cmp::Ordering::Equal));
    let (top, second) = (shares[idx[0]], shares[idx[1]]);
    (top >= 0.35 && top - second >= 0.10).then(|| COLOR_NAMES[idx[0]])
}

/// Quality-weighted mean of per-crop colour shares for one band.
/// Crops whose band abstained (too small/too few pixels) contribute nothing.
pub(crate) fn vote_shares(per_crop: &[(f32, Option<[f32; 11]>)]) -> Option<[f32; 11]> {
    let mut acc = [0.0f32; 11];
    let mut wsum = 0.0f32;
    for (q, s) in per_crop {
        let Some(s) = s else { continue };
        let w = q.max(0.05);
        for i in 0..11 { acc[i] += w * s[i]; }
        wsum += w;
    }
    if wsum <= 0.0 { return None; }
    acc.iter_mut().for_each(|x| *x /= wsum);
    Some(acc)
}

/// Clothing colours of a track from its crops: (top name, bottom name, 22 shares).
/// With the pose model installed the shirt is read from the shoulder→hip box and
/// the trousers from hip→ankle, instead of fixed fractions of a box that also
/// holds arms, bags and background; without it (or when pose misses) the fixed
/// torso/legs bands stand in.
fn vote_colors(data_dir: &std::path::Path, crops: &[(f32, image::RgbImage)]) -> (Option<&'static str>, Option<&'static str>, Option<Vec<f32>>) {
    use crate::alpr::color_shares_in_band as shares;
    let mut tops = Vec::new();
    let mut bottoms = Vec::new();
    for (quality, img) in crops {
        let full = [0.0, 0.0, img.width() as f32, img.height() as f32];
        let pose = crate::pose::estimate(data_dir, img);
        let whole = (0.0, 1.0, 0.0);
        let top = pose.as_ref().and_then(|p| p.torso_box()).and_then(|b| shares(img, b, whole))
            .or_else(|| shares(img, full, (0.18, 0.52, 0.18)));
        let bottom = pose.as_ref().and_then(|p| p.legs_box()).and_then(|b| shares(img, b, whole))
            .or_else(|| shares(img, full, (0.55, 0.92, 0.18)));
        tops.push((*quality, top));
        bottoms.push((*quality, bottom));
    }
    let top = vote_shares(&tops);
    let bottom = vote_shares(&bottoms);
    if top.is_none() && bottom.is_none() { return (None, None, None); }
    let mut blob = Vec::with_capacity(22);
    blob.extend_from_slice(&top.unwrap_or([0.0; 11]));
    blob.extend_from_slice(&bottom.unwrap_or([0.0; 11]));
    (top.as_ref().and_then(name_from_shares), bottom.as_ref().and_then(name_from_shares), Some(blob))
}

/// Write finished tracks. Voting (image decode) runs on the blocking pool.
pub(crate) async fn flush(state: Arc<AppState>, due: Vec<Agg>) {
    let due: Vec<Agg> = due.into_iter()
        .filter(|a| a.observations >= MIN_OBSERVATIONS && !a.crops.is_empty())
        .collect();
    if due.is_empty() { return; }
    let rec_thr = state.settings.read().await.face_recognition_threshold;
    let dd = state.data_dir.clone();
    let voted = tokio::task::spawn_blocking(move || {
        due.into_iter().map(|a| {
            // Decode each kept crop ONCE; colour and attributes both read it. Only
            // person-shaped, un-truncated crops vote: on real footage a wide box cut
            // off by the bottom of a desk webcam's frame read as "female, handbag,
            // seen from the back". Unknown is the honest answer for those.
            let imgs: Vec<(f32, image::RgbImage)> = a.crops.iter()
                .filter(|c| c.quality >= MIN_VOTE_QUALITY)
                .filter_map(|c| {
                let bytes = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, &c.b64).ok()?;
                Some((c.quality, image::load_from_memory(&bytes).ok()?.to_rgb8()))
            }).collect();
            let colors = vote_colors(&dd, &imgs);
            let refs: Vec<&image::RgbImage> = imgs.iter().map(|(_, i)| i).collect();
            let par = crate::par::predict(&dd, &refs).and_then(|v| crate::par::mean(&v));
            (a, colors, par)
        }).collect::<Vec<_>>()
    }).await.unwrap_or_default();

    for (a, (top, bottom, colors), par) in voted {
        let id = uuid::Uuid::new_v4().to_string();
        let (known, method, id_score) = resolve_identity(&a.faces, a.body_id.as_deref(), a.body_score, rec_thr);
        let crop = crate::blobstore::store(&state.data_dir, "tracks", &id, "", &a.crops[0].b64);
        let reid = a.reid_mean().map(|v| crate::embed::vec_to_blob(&v));
        let colors = colors.map(|v| crate::embed::vec_to_blob(&v));
        let res = sqlx::query(
            "INSERT INTO person_tracks(id, cam_id, event_id, started_at, ended_at, body_person_id,
                known_person_id, identity_method, identity_score, crop, quality,
                top_color, bottom_color, colors, reid, behaviours, par)
             VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
        )
        .bind(&id).bind(a.cam_id as i64).bind(&a.event_id)
        .bind(a.started.to_rfc3339()).bind(a.last.to_rfc3339())
        .bind(&a.body_id).bind(&known).bind(method).bind(id_score.map(|s| s as f64))
        .bind(&crop).bind(a.crops[0].quality as f64)
        .bind(top).bind(bottom).bind(colors).bind(reid)
        .bind((!a.behaviours.is_empty()).then(|| serde_json::to_string(&a.behaviours).unwrap_or_default()))
        .bind(par.map(|p| crate::embed::vec_to_blob(&p)))
        .execute(&state.db).await;
        match res {
            Ok(_) => embed_track_clip(&state, &id, &a.crops[0].b64).await,
            Err(e) => {
                tracing::warn!("person_tracks: insert failed for cam{} track {}: {e}", a.cam_id, a.track_id);
                crate::blobstore::delete(&state.data_dir, &crop);
            }
        }
    }
}

/// CLIP image embedding of a track's best crop, so the words people search with
/// that structure can't capture ("hoodie", "umbrella") still rank tracks. No-op
/// without a search model; the backfill in jobs.rs catches up after an install.
pub(crate) async fn embed_track_clip(state: &Arc<AppState>, id: &str, b64: &str) {
    let model = state.settings.read().await.search_model.clone();
    if !crate::embed::is_installed(&state.data_dir, &model) { return; }
    let Ok(jpeg) = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, b64) else { return };
    let (dd, m) = (state.data_dir.clone(), model.clone());
    let v = tokio::task::spawn_blocking(move || {
        crate::embed::with_model(&dd, &m, |enc| enc.encode_image(&jpeg).ok()).flatten()
    }).await.ok().flatten();
    if let Some(v) = v {
        let _ = sqlx::query("UPDATE person_tracks SET clip=?, clip_model=? WHERE id=?")
            .bind(crate::embed::vec_to_blob(&v)).bind(&model).bind(id).execute(&state.db).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_clear_tall_box_beats_a_cropped_crowded_one() {
        let clear = quality(&[100.0, 50.0, 180.0, 300.0], 0.9, 1280.0, 720.0, 0.0);
        let truncated = quality(&[0.0, 50.0, 80.0, 300.0], 0.9, 1280.0, 720.0, 0.0);
        let crowded = quality(&[100.0, 50.0, 180.0, 300.0], 0.9, 1280.0, 720.0, 0.6);
        let wide = quality(&[100.0, 50.0, 400.0, 300.0], 0.9, 1280.0, 720.0, 0.0);
        assert!(clear > truncated && clear > crowded && clear > wide);
        assert_eq!(quality(&[0.0, 0.0, 4.0, 4.0], 0.9, 1280.0, 720.0, 0.0), 0.0);
    }

    #[test]
    fn only_person_shaped_uncut_crops_vote() {
        // From real footage: a 769×305 "person" box cut off by the bottom edge of a
        // 1280×720 frame — attributes on it came out as nonsense.
        let cut = quality(&[511.0, 415.0, 1280.0, 719.0], 0.84, 1280.0, 720.0, 0.0);
        assert!(cut < MIN_VOTE_QUALITY, "truncated wide box must not vote: {cut}");
        let small_standing = quality(&[600.0, 300.0, 640.0, 396.0], 0.8, 1280.0, 720.0, 0.0);
        assert!(small_standing >= MIN_VOTE_QUALITY, "a small standing person still votes: {small_standing}");
    }

    #[test]
    fn top_crops_stay_best_first_and_bounded() {
        let mut v = Vec::new();
        for q in [0.2, 0.9, 0.5, 0.1, 0.7] { push_crop(&mut v, Crop { quality: q, b64: String::new() }); }
        let qs: Vec<f32> = v.iter().map(|c| c.quality).collect();
        assert_eq!(qs, vec![0.9, 0.7, 0.5]);
    }

    #[test]
    fn a_confirmed_face_names_the_track_and_a_body_match_only_proposes() {
        let mut faces = HashMap::new();
        let mut v = IdentityVote::default(); v.add(0.60); v.add(0.58);
        faces.insert("p1".to_string(), ("Ravi".to_string(), v));
        let (known, method, _) = resolve_identity(&faces, Some("kp_p2"), Some(0.7), 0.5);
        assert_eq!((known.as_deref(), method), (Some("p1"), "face"));

        // One weak face hit is not consensus — the body proposal stands, unnamed.
        let mut weak = HashMap::new();
        let mut w = IdentityVote::default(); w.add(0.51);
        weak.insert("p1".to_string(), ("Ravi".to_string(), w));
        let (known, method, _) = resolve_identity(&weak, Some("kp_p2"), Some(0.7), 0.5);
        assert_eq!((known, method), (None, "body_reid"));

        let (known, method, _) = resolve_identity(&HashMap::new(), Some("body_ab12"), None, 0.5);
        assert_eq!((known, method), (None, "none"));
    }

    #[test]
    fn colour_needs_a_clear_winner() {
        let mut s = [0.0f32; 11];
        s[8] = 0.6; s[0] = 0.3; // blue over black by 0.30
        assert_eq!(name_from_shares(&s), Some("blue"));
        let mut tie = [0.0f32; 11];
        tie[4] = 0.40; tie[1] = 0.35; // red vs white — too close to call
        assert_eq!(name_from_shares(&tie), None);
    }

    #[test]
    fn the_sharper_crop_outvotes_the_blurry_one() {
        let mut blue = [0.0f32; 11]; blue[8] = 1.0;
        let mut red = [0.0f32; 11]; red[4] = 1.0;
        let v = vote_shares(&[(0.9, Some(blue)), (0.2, Some(red)), (0.8, None)]).unwrap();
        assert_eq!(name_from_shares(&v), Some("blue"));
        assert!(vote_shares(&[(0.5, None)]).is_none());
    }
}
