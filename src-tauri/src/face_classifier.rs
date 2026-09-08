//! Hybrid face matching: a trained linear classifier on top of the 512-d ArcFace
//! embeddings, used in addition to raw cosine nearest-neighbour.
//!
//! Why: cosine-NN (see `face.rs::match_face`) is great for 1–2-shot cold-start but
//! gets fuzzy as the roster grows and people start to look alike. Edge-AI NVRs' edge
//! over a pure cosine system is exactly this — they train an SVM on the ArcFace
//! vectors. We do the lightweight, robust version: an L2-regularised **multinomial
//! logistic regression** (softmax) over the L2-normalised 512-d vectors, with a
//! synthetic **"unknown"** class sampled from un-tagged sightings so the model can
//! *reject* instead of always naming the nearest person.
//!
//! Cold-start preserved: the classifier only activates once ≥2 people each have
//! ≥`MIN_SHOTS` enrolled embeddings. Below that we never train, the cache stays
//! empty, and `match_face` falls through to the unchanged cosine path. The score it
//! returns is a **calibrated probability** (0..1), NOT a cosine — don't compare it
//! to the 0.5/0.4 cosine thresholds (cf. the 0.9/0.8 footgun in `state.rs`).

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;
use tauri::State;
use tokio::sync::RwLock;

use crate::AppState;

/// A person needs at least this many enrolled embeddings to become a classifier
/// class. Fewer-shot people stay on the cosine path (the classifier would overfit).
const MIN_SHOTS: usize = 3;
/// We need at least this many distinct well-sampled people before a classifier is
/// worth training at all (with one class there's nothing to discriminate).
const MIN_CLASSES: usize = 2;
/// Cap on un-tagged "unknown" negatives pulled in to form the reject class.
const MAX_NEGATIVES: usize = 250;
/// Skip the synthetic unknown class below this many negatives (too few to be a
/// meaningful boundary — rely on the probability gate instead).
const MIN_NEGATIVES: usize = 12;
/// ArcFace embedding dimension this classifier operates on.
const DIM: usize = 512;

// ── Model ────────────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Clone)]
pub struct ClassEntry {
    /// Empty `person_id` marks the synthetic "unknown"/reject class.
    pub person_id: String,
    pub name: String,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ClassifierModel {
    pub dim: usize,
    pub classes: Vec<ClassEntry>,
    /// Row-major `[num_classes × (dim + 1)]`; the last column of each row is bias.
    pub weights: Vec<f32>,
    pub trained_at: String,
}

impl ClassifierModel {
    /// Softmax over the linear scores. Returns `(class_index, probability)` of the
    /// argmax, or `None` if the embedding dim doesn't match (model/tier mismatch).
    fn predict(&self, emb: &[f32]) -> Option<(usize, f32)> {
        if emb.len() != self.dim { return None; }
        let stride = self.dim + 1;
        let c = self.classes.len();
        let mut logits = vec![0.0f32; c];
        for k in 0..c {
            let row = &self.weights[k * stride..k * stride + stride];
            let mut s = row[self.dim]; // bias
            for d in 0..self.dim { s += row[d] * emb[d]; }
            logits[k] = s;
        }
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for l in logits.iter_mut() { *l = (*l - max).exp(); sum += *l; }
        if sum <= 0.0 { return None; }
        let mut best = (0usize, 0.0f32);
        for (k, l) in logits.iter().enumerate() {
            let p = l / sum;
            if p > best.1 { best = (k, p); }
        }
        Some(best)
    }

    /// Number of real person classes (excludes the synthetic "unknown" class).
    fn person_class_count(&self) -> usize {
        self.classes.iter().filter(|c| !c.person_id.is_empty()).count()
    }
}

// ── Process-wide cache (avoids a DB hit per recognised face) ───────────────────

struct Cache { loaded: bool, model: Option<Arc<ClassifierModel>> }
static MODEL: std::sync::OnceLock<RwLock<Cache>> = std::sync::OnceLock::new();
fn cache() -> &'static RwLock<Cache> {
    MODEL.get_or_init(|| RwLock::new(Cache { loaded: false, model: None }))
}

/// Get the cached model, loading from the DB the first time. `None` = no classifier
/// is active (roster too small / never trained) → caller uses cosine.
async fn get_model(db: &SqlitePool) -> Option<Arc<ClassifierModel>> {
    {
        let c = cache().read().await;
        if c.loaded { return c.model.clone(); }
    }
    let loaded = load_from_db(db).await;
    let mut c = cache().write().await;
    c.loaded = true;
    c.model = loaded.map(Arc::new);
    c.model.clone()
}

async fn load_from_db(db: &SqlitePool) -> Option<ClassifierModel> {
    let row: Option<(String,)> = sqlx::query_as(
        "SELECT model_json FROM face_classifier_model WHERE id = 1"
    ).fetch_optional(db).await.ok().flatten();
    let (json,) = row?;
    serde_json::from_str::<ClassifierModel>(&json).ok()
}

async fn set_cache(model: Option<Arc<ClassifierModel>>) {
    let mut c = cache().write().await;
    c.loaded = true;
    c.model = model;
}

// ── Public matching entry point (called from face.rs::match_face) ──────────────

/// The hybrid head's verdict for one embedding. `Some((person_id, name, prob))`
/// only when a trained model confidently names a real person at `>= min_prob`;
/// `None` (reject / no model / unknown class / low confidence) means the caller
/// should fall back to the cosine path. `prob` is a calibrated probability.
pub async fn predict_person(
    db: &SqlitePool,
    emb: &[f32],
    min_prob: f32,
) -> Option<(String, String, f32)> {
    let model = get_model(db).await?;
    let (idx, prob) = model.predict(emb)?;
    let cls = model.classes.get(idx)?;
    if cls.person_id.is_empty() { return None; } // landed on the reject class
    if prob < min_prob { return None; }
    Some((cls.person_id.clone(), cls.name.clone(), prob))
}

// ── Training ───────────────────────────────────────────────────────────────────

fn parse_blob(blob: &[u8], dim: usize) -> Option<Vec<f32>> {
    if blob.len() != dim * 4 { return None; }
    Some(blob.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

/// Gather (person classes with ≥MIN_SHOTS 512-d embeddings) + a sample of un-tagged
/// negatives for the reject class. Returns `None` if too few classes to bother.
async fn gather(db: &SqlitePool) -> Option<(Vec<(String, String, Vec<Vec<f32>>)>, Vec<Vec<f32>>)> {
    let rows: Vec<(String, String, String)> = sqlx::query_as(
        "SELECT id, name, embeddings FROM known_persons"
    ).fetch_all(db).await.ok()?;

    let mut people: Vec<(String, String, Vec<Vec<f32>>)> = Vec::new();
    for (pid, name, embs_json) in rows {
        let Ok(embs): Result<Vec<Vec<f32>>, _> = serde_json::from_str(&embs_json) else { continue };
        let good: Vec<Vec<f32>> = embs.into_iter().filter(|e| e.len() == DIM).collect();
        if good.len() >= MIN_SHOTS { people.push((pid, name, good)); }
    }
    if people.len() < MIN_CLASSES { return None; }

    // CURATED hard negatives first — descriptors the user explicitly corrected
    // away from a person ("this is NOT X"). These are the highest-value reject
    // samples (they sit right on a confused boundary), so they lead the pool and
    // are never crowded out by random untagged faces. This is what turns a
    // correction into a lasting accuracy gain.
    let hard_rows: Vec<(Vec<u8>, i64)> = sqlx::query_as(
        "SELECT descriptor, dim FROM face_negatives WHERE dim = 512 ORDER BY created_at DESC LIMIT ?"
    ).bind(MAX_NEGATIVES as i64).fetch_all(db).await.unwrap_or_default();
    let mut negatives: Vec<Vec<f32>> = hard_rows.into_iter()
        .filter_map(|(b, d)| parse_blob(&b, d as usize)).collect();

    // Top up with recent untagged "stranger" faces (the original reject pool).
    if negatives.len() < MAX_NEGATIVES {
        let neg_rows: Vec<(Vec<u8>, i64)> = sqlx::query_as(
            "SELECT descriptor, dim FROM face_embeddings
              WHERE person_id IS NULL AND dim = 512
              ORDER BY seen_at DESC LIMIT ?"
        ).bind((MAX_NEGATIVES - negatives.len()) as i64).fetch_all(db).await.unwrap_or_default();
        negatives.extend(neg_rows.into_iter().filter_map(|(b, d)| parse_blob(&b, d as usize)));
    }

    Some((people, negatives))
}

/// Train a softmax classifier. Classes are balanced by upsampling minority classes
/// to the largest class count (edge-AI NVRs' "upsampling policy" for imbalance).
fn train_model(
    people: &[(String, String, Vec<Vec<f32>>)],
    negatives: &[Vec<f32>],
    now: String,
) -> ClassifierModel {
    let mut classes: Vec<ClassEntry> = people.iter()
        .map(|(pid, name, _)| ClassEntry { person_id: pid.clone(), name: name.clone() })
        .collect();
    // Build per-class sample lists (indices into a flat sample pool come later).
    let mut class_samples: Vec<Vec<&Vec<f32>>> = people.iter()
        .map(|(_, _, embs)| embs.iter().collect()).collect();
    let use_unknown = negatives.len() >= MIN_NEGATIVES;
    if use_unknown {
        classes.push(ClassEntry { person_id: String::new(), name: "unknown".into() });
        class_samples.push(negatives.iter().collect());
    }

    // Upsample every class to the max count so no person dominates the boundary.
    let max_count = class_samples.iter().map(|s| s.len()).max().unwrap_or(1).max(1);
    let mut samples: Vec<(&Vec<f32>, usize)> = Vec::new();
    for (ci, list) in class_samples.iter().enumerate() {
        if list.is_empty() { continue; }
        for i in 0..max_count {
            samples.push((list[i % list.len()], ci));
        }
    }

    let c = classes.len();
    let stride = DIM + 1;
    let mut w = vec![0.0f32; c * stride];

    // Full-batch gradient descent on cross-entropy + L2. Cheap: runs off the
    // request path (async retrain), a few hundred samples × C classes × ~250 iters.
    const ITERS: usize = 250;
    const LR: f32 = 0.5;
    const L2: f32 = 1e-3;
    let n = samples.len().max(1) as f32;
    let mut logits = vec![0.0f32; c];
    let mut grad = vec![0.0f32; c * stride];
    for _ in 0..ITERS {
        for g in grad.iter_mut() { *g = 0.0; }
        for (emb, label) in &samples {
            // forward
            for k in 0..c {
                let row = &w[k * stride..k * stride + stride];
                let mut s = row[DIM];
                for d in 0..DIM { s += row[d] * emb[d]; }
                logits[k] = s;
            }
            let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for l in logits.iter_mut() { *l = (*l - max).exp(); sum += *l; }
            for l in logits.iter_mut() { *l /= sum; }
            // backward: dL/dlogit_k = p_k - 1{k==label}
            for k in 0..c {
                let err = logits[k] - if k == *label { 1.0 } else { 0.0 };
                let base = k * stride;
                for d in 0..DIM { grad[base + d] += err * emb[d]; }
                grad[base + DIM] += err; // bias
            }
        }
        // average + L2 (not on bias) → step
        for k in 0..c {
            let base = k * stride;
            for d in 0..DIM {
                let g = grad[base + d] / n + L2 * w[base + d];
                w[base + d] -= LR * g;
            }
            w[base + DIM] -= LR * (grad[base + DIM] / n);
        }
    }

    ClassifierModel { dim: DIM, classes, weights: w, trained_at: now }
}

/// (Re)train the classifier from the current roster and persist it. Clears the
/// model (back to pure cosine) when the roster is too small. Updates the cache.
/// Cheap + idempotent — safe to call after every enroll / confirm / delete.
pub async fn retrain(db: &SqlitePool) -> ClassifierStatus {
    match gather(db).await {
        Some((people, negatives)) => {
            let now = chrono::Utc::now().to_rfc3339();
            let model = train_model(&people, &negatives, now.clone());
            let json = serde_json::to_string(&model).unwrap_or_default();
            let _ = sqlx::query(
                "INSERT INTO face_classifier_model(id, model_json, trained_at) VALUES(1, ?, ?)
                 ON CONFLICT(id) DO UPDATE SET model_json = excluded.model_json, trained_at = excluded.trained_at"
            ).bind(&json).bind(&now).execute(db).await;
            let status = ClassifierStatus {
                active: true,
                trained_people: model.person_class_count(),
                has_reject_class: model.classes.iter().any(|c| c.person_id.is_empty()),
                trained_at: Some(model.trained_at.clone()),
                person_ids: model.classes.iter().filter(|c| !c.person_id.is_empty())
                    .map(|c| c.person_id.clone()).collect(),
            };
            set_cache(Some(Arc::new(model))).await;
            status
        }
        None => {
            // Too few well-sampled people → drop any stale model, revert to cosine.
            let _ = sqlx::query("DELETE FROM face_classifier_model WHERE id = 1").execute(db).await;
            set_cache(None).await;
            ClassifierStatus::inactive()
        }
    }
}

// ── Status (for the People UI) ─────────────────────────────────────────────────

#[derive(Serialize, Clone)]
pub struct ClassifierStatus {
    /// True when a trained model is active (recognition uses "smart match").
    pub active: bool,
    /// Number of people the classifier discriminates (≥MIN_SHOTS each).
    pub trained_people: usize,
    /// Whether a synthetic reject/"unknown" class was trained in.
    pub has_reject_class: bool,
    pub trained_at: Option<String>,
    /// Which known-person ids are covered, so the Roster can show a "Trained ✓" chip.
    pub person_ids: Vec<String>,
}

impl ClassifierStatus {
    fn inactive() -> Self {
        ClassifierStatus { active: false, trained_people: 0, has_reject_class: false, trained_at: None, person_ids: Vec::new() }
    }
}

/// Report whether the hybrid classifier is active and who it covers, without
/// retraining. Loads the cached/persisted model.
#[tauri::command]
pub async fn face_classifier_status(state: State<'_, Arc<AppState>>) -> Result<ClassifierStatus, String> {
    match get_model(&state.db).await {
        Some(m) => Ok(ClassifierStatus {
            active: true,
            trained_people: m.person_class_count(),
            has_reject_class: m.classes.iter().any(|c| c.person_id.is_empty()),
            trained_at: Some(m.trained_at.clone()),
            person_ids: m.classes.iter().filter(|c| !c.person_id.is_empty())
                .map(|c| c.person_id.clone()).collect(),
        }),
        None => Ok(ClassifierStatus::inactive()),
    }
}

/// Force a retrain (Roster "Retrain" affordance) and return the new status.
#[tauri::command]
pub async fn retrain_face_classifier(state: State<'_, Arc<AppState>>) -> Result<ClassifierStatus, String> {
    Ok(retrain(&state.db).await)
}
