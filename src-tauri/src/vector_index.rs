//! usearch-backed ANN index — an **acceleration layer** over the embedding BLOBs
//! we already keep in SQLite. Each logical space (e.g. semantic-search vectors for
//! one CLIP model) is a single HNSW index, keyed by the SQLite `rowid` of the
//! source row so a hit maps straight back to its record.
//!
//! Design goals (single-location appliance — one process, no service):
//!   * **Lean:** the in-process, native equivalent of a vector DB. Indexes live in
//!     RAM, rebuilt from SQLite on boot. Millions of vectors, sub-ms search — no
//!     Milvus, no Docker, +1 crate.
//!   * **Zero-regression:** purely additive. If an index is empty, missing, or an
//!     op fails, callers fall back to the existing brute-force cosine scan. Search
//!     is therefore never *worse* than before — only faster at scale.
//!   * **Correct:** cosine metric over the same L2-normalised f32 vectors, full
//!     `F32` precision (no quantisation) so ranked results match the brute-force
//!     path within floating-point noise.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use usearch::{new_index, Index, IndexOptions, MetricKind, ScalarKind};

/// Registry of named indexes. `Index` is `Send + Sync`; we still guard the *map*
/// with a `Mutex` so `ensure`/insert races are safe. Per-op locking is fine here:
/// adds happen on event close and searches are user-initiated — never a per-frame
/// hot path.
static REG: OnceLock<Mutex<HashMap<String, Index>>> = OnceLock::new();
fn reg() -> &'static Mutex<HashMap<String, Index>> { REG.get_or_init(|| Mutex::new(HashMap::new())) }

/// Ensure index `name` exists with dimension `dim`. Recreated if the dimension
/// changed (e.g. the user switched search models to a different vector size).
/// Returns false if creation failed (caller then no-ops / falls back).
fn ensure_locked(map: &mut HashMap<String, Index>, name: &str, dim: usize) -> bool {
    if let Some(ix) = map.get(name) {
        if ix.dimensions() == dim { return true; }
        map.remove(name); // dim mismatch → rebuild from scratch
    }
    let opts = IndexOptions {
        dimensions: dim,
        metric: MetricKind::Cos,
        quantization: ScalarKind::F32,
        ..Default::default()
    };
    match new_index(&opts) {
        Ok(ix) => {
            let _ = ix.reserve(1024);
            map.insert(name.to_string(), ix);
            true
        }
        Err(e) => { tracing::debug!("vector_index '{name}': create failed: {e}"); false }
    }
}

/// Add or update one vector under `key` (the source row's `rowid`). Idempotent —
/// an existing key is replaced, so re-embedding an event refreshes its vector
/// without duplicating it.
pub fn upsert(name: &str, dim: usize, key: u64, vector: &[f32]) {
    if dim == 0 || vector.len() != dim { return; }
    let Ok(mut map) = reg().lock() else { return };
    if !ensure_locked(&mut map, name, dim) { return; }
    let Some(ix) = map.get(name) else { return };
    // usearch requires capacity be reserved before it's exceeded.
    if ix.size() + 1 > ix.capacity() {
        let _ = ix.reserve((ix.capacity() * 2).max(1024));
    }
    let _ = ix.remove(key); // replace-if-present
    if let Err(e) = ix.add(key, vector) {
        tracing::debug!("vector_index '{name}': add failed: {e}");
    }
}

/// Remove a vector (e.g. when its source event is deleted). Not yet wired to a
/// delete path — stale keys are already harmless (the rowid→id join drops them and
/// a boot rebuild clears them) — but this is the hook for explicit pruning next.
#[allow(dead_code)]
pub fn remove(name: &str, key: u64) {
    if let Ok(map) = reg().lock() {
        if let Some(ix) = map.get(name) { let _ = ix.remove(key); }
    }
}

/// Nearest `k` by cosine similarity, best-first, as `(key, similarity)` pairs.
/// Returns empty when the index is absent/empty — the signal for the caller to
/// fall back to the brute-force scan.
pub fn search(name: &str, dim: usize, query: &[f32], k: usize) -> Vec<(u64, f32)> {
    if dim == 0 || query.len() != dim || k == 0 { return Vec::new(); }
    let Ok(map) = reg().lock() else { return Vec::new() };
    let Some(ix) = map.get(name) else { return Vec::new() };
    let n = ix.size();
    if n == 0 { return Vec::new(); }
    match ix.search(query, k.min(n)) {
        // usearch `Cos` distance = 1 − cosine_similarity ⇒ similarity = 1 − distance.
        Ok(m) => m.keys.into_iter().zip(m.distances)
            .map(|(key, dist)| (key, 1.0 - dist))
            .collect(),
        Err(e) => { tracing::debug!("vector_index '{name}': search failed: {e}"); Vec::new() }
    }
}

/// Number of vectors currently indexed under `name` (0 if absent). Diagnostics +
/// the forthcoming face/body index phase.
#[allow(dead_code)]
pub fn len(name: &str) -> usize {
    reg().lock().ok().and_then(|m| m.get(name).map(|ix| ix.size())).unwrap_or(0)
}

// ─── Graph clustering primitives (self-supervised face discovery) ─────────────

/// Build a kNN similarity graph over `vectors` with a **transient** usearch index:
/// for each vector, its `k` nearest neighbours with cosine ≥ `min_sim` become
/// undirected, de-duplicated edges `(i, j, similarity)`. O(n·k·log n) — the scalable
/// alternative to O(n²) all-pairs that lets us cluster the WHOLE face gallery
/// instead of a recent-400 cap. The index is dropped when this returns.
pub fn knn_graph(vectors: &[Vec<f32>], k: usize, min_sim: f32) -> Vec<(usize, usize, f32)> {
    let n = vectors.len();
    if n < 2 { return Vec::new(); }
    let dim = vectors[0].len();
    if dim == 0 { return Vec::new(); }
    let opts = IndexOptions { dimensions: dim, metric: MetricKind::Cos, quantization: ScalarKind::F32, ..Default::default() };
    let Ok(ix) = new_index(&opts) else { return Vec::new() };
    if ix.reserve(n).is_err() { return Vec::new(); }
    for (i, v) in vectors.iter().enumerate() {
        if v.len() == dim { let _ = ix.add(i as u64, v); }
    }
    let want = (k + 1).min(n); // +1 because the vector matches itself
    let mut edges: Vec<(usize, usize, f32)> = Vec::new();
    let mut seen: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
    for (i, v) in vectors.iter().enumerate() {
        if v.len() != dim { continue; }
        let Ok(m) = ix.search(v, want) else { continue };
        for (key, dist) in m.keys.into_iter().zip(m.distances) {
            let j = key as usize;
            if j == i || j >= n { continue; }
            let sim = 1.0 - dist; // usearch Cos distance → similarity
            if sim < min_sim { continue; }
            let e = if i < j { (i, j) } else { (j, i) };
            if seen.insert(e) { edges.push((e.0, e.1, sim)); }
        }
    }
    edges
}

/// **Chinese Whispers** graph clustering (Biemann 2006) — the best-in-class,
/// hyperparameter-robust face-clustering algorithm (dlib uses it). Each node starts
/// in its own class, then repeatedly adopts the highest edge-weight-sum class in its
/// neighbourhood; it discovers the cluster COUNT automatically and resists the
/// chaining that plagues single-linkage. Returns a class label per node (nodes in
/// the same cluster share a label). O(iters · E).
pub fn chinese_whispers(n: usize, edges: &[(usize, usize, f32)], iters: usize) -> Vec<usize> {
    use rand::seq::SliceRandom;
    let mut adj: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
    for &(a, b, w) in edges {
        if a < n && b < n { adj[a].push((b, w)); adj[b].push((a, w)); }
    }
    let mut labels: Vec<usize> = (0..n).collect();
    let mut order: Vec<usize> = (0..n).collect();
    let mut rng = rand::thread_rng();
    for _ in 0..iters {
        order.shuffle(&mut rng);
        let mut changed = false;
        for &node in &order {
            if adj[node].is_empty() { continue; }
            let mut scores: std::collections::HashMap<usize, f32> = std::collections::HashMap::new();
            for &(nb, w) in &adj[node] {
                *scores.entry(labels[nb]).or_insert(0.0) += w;
            }
            if let Some((&best, _)) = scores.iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            {
                if labels[node] != best { labels[node] = best; changed = true; }
            }
        }
        if !changed { break; } // converged early
    }
    labels
}

// ─── Domain helpers ──────────────────────────────────────────────────────────

/// Index name for the semantic-search vectors of a given CLIP model. (Different
/// models live in different vector spaces + dimensions, so each gets its own.)
pub fn events_index(model: &str) -> String { format!("events:{model}") }

/// Rebuild the semantic-search ANN index from every stored event embedding.
/// Called once on boot (embeddings are small; a few MB total at our scale).
pub async fn build_events(db: &sqlx::SqlitePool) {
    let rows: Vec<(i64, Vec<u8>, i64, String)> = sqlx::query_as(
        "SELECT rowid, descriptor, dim, model FROM event_embeddings"
    ).fetch_all(db).await.unwrap_or_default();
    let n = rows.len();
    for (rowid, blob, dim, model) in rows {
        let v = crate::embed::blob_to_vec(&blob);
        upsert(&events_index(&model), dim as usize, rowid as u64, &v);
    }
    if n > 0 {
        tracing::info!("vector_index: loaded {n} event embedding(s) into the ANN index");
    }
}
