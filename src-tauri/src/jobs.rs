//! Durable background jobs — **Apalis** with a **SQLite** backend (shares the app
//! DB pool). The lean, embedded equivalent of a Celery/worker tier: a persisted,
//! retryable queue for the *deferred* work that must survive a crash/restart.
//!
//! Scope (single-location appliance — one process):
//!   * **ONLY deferred work** lives here — event embedding, and (later) clip export,
//!     pruning, retention. The real-time per-frame detection pipeline stays in the
//!     in-memory drop-don't-queue path; a DB-backed queue would choke on frames.
//!   * **Durability model:** a job is persisted the moment it's enqueued, so a
//!     crash between "event closed" and "embedding done" no longer loses the work
//!     (the old `tokio::spawn` did). Apalis re-enqueues jobs orphaned by a dead
//!     worker (`reenqueue_orphaned_after`), and a periodic **self-healing backfill**
//!     re-queues any event still missing its embedding — belt-and-suspenders
//!     eventual consistency without a fragile per-job retry layer.
//!   * **Zero-regression:** if the queue can't be created, callers fall back to the
//!     original `tokio::spawn(embed_event)` path (see `AppState.embed_jobs: Option`).

use std::sync::Arc;
use std::time::Duration;

use apalis::prelude::*;
use apalis_sql::{sqlite::SqliteStorage, Config};
use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;

use crate::AppState;

/// One unit of durable work: (re)embed an event's thumbnail + summary into the
/// semantic-search vector space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedEventJob {
    pub event_id: String,
}

/// The concrete storage type for embedding jobs. `Clone` + `Send` + `Sync` (wraps
/// the pool + config), so it lives happily in `AppState` and is cloned per enqueue.
pub type EmbedStorage = SqliteStorage<EmbedEventJob>;

/// Create (and migrate) the Apalis SQLite storage on the app's own pool. Idempotent
/// — `setup` creates the `apalis`-namespaced job tables if absent. A calm 5 s poll
/// keeps idle overhead negligible on an appliance (embedding within ~5 s of an event
/// is plenty fresh for search).
pub async fn make_storage(db: &SqlitePool) -> Result<EmbedStorage, sqlx::Error> {
    SqliteStorage::setup(db).await?;
    // DELIBERATELY not renamed with the app. This string is the queue's identity
    // in the `apalis` tables of the user's existing database: rows enqueued under
    // it would never be polled again if it changed, stranding every pending
    // embedding job at upgrade. It is internal and invisible — not worth the loss.
    let cfg = Config::new("securecam::embed").set_poll_interval(Duration::from_secs(5));
    Ok(SqliteStorage::new_with_config(db.clone(), cfg))
}

/// Enqueue a durable embedding job. Cheap — clones the storage handle (shares the
/// pool) and persists one row. Best-effort: a failure just logs (the backfill will
/// catch the event later).
pub async fn enqueue_embed(storage: &EmbedStorage, event_id: String) {
    let mut s = storage.clone();
    if let Err(e) = s.push(EmbedEventJob { event_id }).await {
        tracing::debug!("jobs: enqueue_embed failed: {e}");
    }
}

/// The worker: run one embedding job. `embed_event` already no-ops safely when the
/// search skill isn't installed or the event has no thumbnail, so this is a thin
/// durable wrapper. Returns `Ok` — the self-healing backfill provides retry, so a
/// soft failure here doesn't need to poison the job.
async fn run_embed_event(job: EmbedEventJob, state: Data<Arc<AppState>>) -> Result<(), Error> {
    crate::agent::embed_event((*state).clone(), job.event_id).await;
    Ok(())
}

/// Run the Apalis monitor (workers) for the app's lifetime. Spawn on a background
/// task; it polls the SQLite queue and processes jobs with bounded concurrency.
pub async fn run_workers(storage: EmbedStorage, state: Arc<AppState>) {
    let monitor = Monitor::new().register(
        WorkerBuilder::new("embed-events")
            .concurrency(1) // the CLIP encoder is mutex-serialized anyway; 1 blocking worker

            .data(state)
            .backend(storage)
            .build_fn(run_embed_event),
    );
    if let Err(e) = monitor.run().await {
        tracing::warn!("jobs: apalis monitor stopped: {e}");
    }
}

/// Fill in `motion_events.outfit` for person events recorded before the column
/// existed, oldest-visible first, 200 per tick.
///
/// Single-frame only — there is no clip to strobe after the fact — so this is a
/// weaker signal than the live path, which votes across up to four frames. That
/// is the honest trade for having any answer at all about last week.
///
/// Idempotent by construction: the `outfit IS NULL` filter excludes everything
/// already written. Events whose colours can't be read stay NULL and are
/// retried each tick; at 200/tick that costs a scan, not a stampede.
async fn backfill_outfits(state: &Arc<AppState>) {
    let rows: Vec<(String, Option<String>, Option<String>)> = sqlx::query_as(
        "SELECT id, thumbnail, detections FROM motion_events
          WHERE outfit IS NULL AND thumbnail IS NOT NULL AND thumbnail != ''
            AND detections LIKE '%\"label\":\"person\"%'
          ORDER BY started_at DESC LIMIT 200"
    ).fetch_all(&state.db).await.unwrap_or_default();
    if rows.is_empty() { return; }

    let mut written = 0usize;
    for (id, thumb, dets) in &rows {
        let (Some(thumb), Some(dets)) = (thumb, dets) else { continue };
        // Thumbnails may be "@file:" blobstore refs rather than inline base64.
        let resolved = crate::blobstore::resolve(&state.data_dir, thumb);
        let Ok(jpeg) = base64::Engine::decode(
            &base64::engine::general_purpose::STANDARD, &resolved) else { continue };
        if let Some(o) = crate::reid::outfit_for_event(&[jpeg], dets) {
            let _ = sqlx::query("UPDATE motion_events SET outfit=? WHERE id=?")
                .bind(&o).bind(id).execute(&state.db).await;
            written += 1;
        }
    }
    if written > 0 {
        tracing::info!("jobs: outfit backfill wrote {written} of {} candidates", rows.len());
    }
}

/// Self-healing backfill: every 10 min, find events that have a thumbnail but no
/// embedding for the active search model and enqueue them. This is the app-level
/// retry + a catch-up for events that closed while the search skill was
/// uninstalled (install it later → they get indexed automatically).
pub async fn run_backfill_scheduler(storage: EmbedStorage, state: Arc<AppState>) {
    // One-shot re-embed (v1 → object-crop): image embeddings used to encode the
    // FULL frame; embed_event now crops the dominant detection box (mature NVRs
    // parity — "red shirt" queries need the object, not the scene). Drop the
    // stale image embeddings for events that HAVE a croppable box; the regular
    // backfill below re-embeds them gradually (200 / 10 min). Text embeddings
    // and no-detection events are untouched. Flag-guarded: runs exactly once.
    const REEMBED_FLAG: &str = "embed_object_crop_v1";
    if crate::agent::memory::read_memory(&state.db, REEMBED_FLAG).await.is_none() {
        let res = sqlx::query(
            "DELETE FROM event_embeddings WHERE kind='image' AND event_id IN (
                SELECT id FROM motion_events
                 WHERE thumbnail IS NOT NULL AND thumbnail != ''
                   AND (detections LIKE '%\"label\":\"person\"%'  OR detections LIKE '%\"label\":\"car\"%'
                     OR detections LIKE '%\"label\":\"truck\"%'   OR detections LIKE '%\"label\":\"bus\"%'
                     OR detections LIKE '%\"label\":\"motorcycle\"%' OR detections LIKE '%\"label\":\"bicycle\"%'
                     OR detections LIKE '%\"label\":\"dog\"%'     OR detections LIKE '%\"label\":\"cat\"%'))"
        ).execute(&state.db).await;
        if let Ok(r) = res {
            tracing::info!("jobs: object-crop re-embed queued — dropped {} stale image embeddings", r.rows_affected());
        }
        crate::agent::memory::write_memory(&state.db, REEMBED_FLAG, "done").await;
    }

    let mut tick = tokio::time::interval(Duration::from_secs(600));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        tick.tick().await;

        // Clothing colours for events recorded before the `outfit` column
        // existed. Deliberately BEFORE the search-model gate below: this needs no
        // model and no network, so it must not be skipped on installs that have
        // no search skill — which is the default. Without it, "who was in the red
        // jacket last Tuesday" is blind to everything already recorded.
        backfill_outfits(&state).await;

        let model = state.settings.read().await.search_model.clone();
        if !crate::embed::is_installed(&state.data_dir, &model) { continue; }
        // Catch events missing EITHER embedding kind they should have (image if a
        // thumbnail exists, text if an ai_summary exists) — so a re-embed also
        // backfills text for events that only got an image (e.g. after the CLIP
        // text-padding fix). embed_event upserts both, so re-running is idempotent.
        let rows: Vec<(String,)> = sqlx::query_as(
            "SELECT m.id FROM motion_events m
             WHERE (m.thumbnail IS NOT NULL AND NOT EXISTS (
                        SELECT 1 FROM event_embeddings e WHERE e.event_id = m.id AND e.model = ? AND e.kind = 'image'))
                OR (m.ai_summary IS NOT NULL AND NOT EXISTS (
                        SELECT 1 FROM event_embeddings e WHERE e.event_id = m.id AND e.model = ? AND e.kind = 'text'))
             ORDER BY m.started_at DESC
             LIMIT 200"
        ).bind(&model).bind(&model).fetch_all(&state.db).await.unwrap_or_default();
        if rows.is_empty() { continue; }
        tracing::info!("jobs: backfill enqueuing {} event(s) missing embeddings", rows.len());
        for (id,) in rows { enqueue_embed(&storage, id).await; }
    }
}
