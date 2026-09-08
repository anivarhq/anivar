//! Known-persons enrollment, listing, and face-recognition commands.

use std::sync::Arc;

use chrono::Utc;
use serde::{Deserialize, Serialize};
use tauri::State;
use uuid::Uuid;

use crate::AppState;


/// Fire-and-forget retrain of the hybrid face classifier after a roster change.
/// Training is cheap and these mutations are user-paced, so a debounce isn't worth
/// it; spawned with a cloned pool so the command returns immediately. The mutation's
/// awaits have already committed by the time this reads the DB.
fn schedule_classifier_retrain(db: &sqlx::SqlitePool) {
    let db = db.clone();
    tokio::spawn(async move { crate::face_classifier::retrain(&db).await; });
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct KnownPerson {
    pub id: String,
    pub name: String,
    pub role: String,
    /// JSON: Vec<Vec<f32>>. Listings ship a stub "[]" — the full arrays are
    /// matcher-internal (~190KB/person) and the UI only ever needed the COUNT,
    /// which now travels separately (`embedding_count`).
    pub embeddings: String,
    pub embedding_count: usize,
    pub thumbnail: Option<String>,
    pub created_at: String,
    pub last_seen_at: Option<String>,
}

#[tauri::command]
pub async fn enroll_person(
    state: State<'_, Arc<AppState>>,
    name: String,
    role: String,
    embedding: Vec<f32>,   // single 512-d ArcFace descriptor (same space as embed_face)
    thumbnail: Option<String>,
) -> Result<KnownPerson, String> {
    let id = Uuid::new_v4().to_string();
    let now = Utc::now().to_rfc3339();
    // Embeddings stored as array of arrays — supports adding more later
    let embeddings = serde_json::to_string(&vec![embedding]).map_err(|e| e.to_string())?;
    sqlx::query(
        "INSERT INTO known_persons(id, name, role, embeddings, thumbnail, created_at) VALUES(?,?,?,?,?,?)"
    )
    .bind(&id).bind(&name).bind(&role).bind(&embeddings)
    .bind(&thumbnail).bind(&now)
    .execute(&state.db).await.map_err(|e| e.to_string())?;

    schedule_classifier_retrain(&state.db);
    Ok(KnownPerson { id, name, role, embeddings, embedding_count: 1, thumbnail, created_at: now, last_seen_at: None })
}

/// Enroll a person from MULTIPLE captured angle embeddings at once (guided
/// multi-angle capture). Every angle lands in the person's `embeddings` array so
/// recognition matches them from any pose. Embeddings come from `embed_face`
/// (512-d ArcFace) — the same space as live + event recognition.
#[tauri::command]
pub async fn enroll_person_multi(
    state: State<'_, Arc<AppState>>,
    name: String,
    role: String,
    embeddings: Vec<Vec<f32>>,
    thumbnail: Option<String>,
) -> Result<KnownPerson, String> {
    if embeddings.is_empty() { return Err("no face angles captured".into()); }
    let id = Uuid::new_v4().to_string();
    let now = Utc::now().to_rfc3339();
    let n_angles = embeddings.len();
    let embeddings_json = serde_json::to_string(&embeddings).map_err(|e| e.to_string())?;
    sqlx::query(
        "INSERT INTO known_persons(id, name, role, embeddings, thumbnail, created_at) VALUES(?,?,?,?,?,?)"
    )
    .bind(&id).bind(&name).bind(&role).bind(&embeddings_json)
    .bind(&thumbnail).bind(&now)
    .execute(&state.db).await.map_err(|e| e.to_string())?;

    schedule_classifier_retrain(&state.db);
    Ok(KnownPerson { id, name, role, embeddings: embeddings_json, embedding_count: n_angles, thumbnail, created_at: now, last_seen_at: None })
}

#[tauri::command]
pub async fn add_person_embedding(
    state: State<'_, Arc<AppState>>,
    id: String,
    embedding: Vec<f32>,
) -> Result<(), String> {
    let row: Option<(String,)> = sqlx::query_as("SELECT embeddings FROM known_persons WHERE id=?")
        .bind(&id).fetch_optional(&state.db).await.map_err(|e| e.to_string())?;
    let (emb_json,) = row.ok_or("person not found")?;
    let mut embs: Vec<Vec<f32>> = serde_json::from_str(&emb_json).unwrap_or_default();
    // Diversity gate + 30-shot cap (see push_enrolled_shot). A near-duplicate
    // angle is a silent no-op — the person is already covered from that pose.
    if push_enrolled_shot(&mut embs, embedding) {
        let updated = serde_json::to_string(&embs).map_err(|e| e.to_string())?;
        sqlx::query("UPDATE known_persons SET embeddings=? WHERE id=?")
            .bind(&updated).bind(&id)
            .execute(&state.db).await.map_err(|e| e.to_string())?;
        schedule_classifier_retrain(&state.db);
    }
    Ok(())
}

#[tauri::command]
pub async fn list_known_persons(state: State<'_, Arc<AppState>>) -> Result<Vec<KnownPerson>, String> {
    let rows = sqlx::query_as::<_, (String, String, String, String, Option<String>, String, Option<String>)>(
        "SELECT id, name, role, embeddings, thumbnail, created_at, last_seen_at FROM known_persons ORDER BY name ASC"
    )
    .fetch_all(&state.db).await.map_err(|e| e.to_string())?;

    Ok(rows.into_iter().map(|(id, name, role, embeddings, thumbnail, created_at, last_seen_at)| {
        // Ship the COUNT, not the arrays: the UI only ever displayed
        // `JSON.parse(embeddings).length`, but the raw JSON is ~190KB per
        // enrolled person — ~1MB of IPC per People refresh for one number.
        let embedding_count = serde_json::from_str::<Vec<serde_json::Value>>(&embeddings)
            .map(|v| v.len()).unwrap_or(0);
        KnownPerson { id, name, role, embeddings: "[]".into(), embedding_count, thumbnail, created_at, last_seen_at }
    }).collect())
}

/// Rename a person and/or change their role. `face_sightings` rows are keyed by
/// NAME, so they follow the rename — history and activity stats keep continuity
/// instead of orphaning (the old workaround was delete + re-enroll, which lost
/// the enrolled gallery AND left orphans). The classifier stores display names
/// per class, so a retrain is scheduled to refresh them.
#[tauri::command]
pub async fn rename_person(
    state: State<'_, Arc<AppState>>,
    id: String,
    name: String,
    role: Option<String>,
) -> Result<(), String> {
    let name = name.trim().to_string();
    if name.is_empty() { return Err("name cannot be empty".into()); }
    let old: Option<String> = sqlx::query_scalar("SELECT name FROM known_persons WHERE id=?")
        .bind(&id).fetch_optional(&state.db).await.ok().flatten();
    let Some(old) = old else { return Err("person not found".into()) };
    match role.as_deref().map(str::trim) {
        Some(r) if !r.is_empty() => {
            sqlx::query("UPDATE known_persons SET name=?, role=? WHERE id=?")
                .bind(&name).bind(r).bind(&id)
                .execute(&state.db).await.map_err(|e| e.to_string())?;
        }
        _ => {
            sqlx::query("UPDATE known_persons SET name=? WHERE id=?")
                .bind(&name).bind(&id)
                .execute(&state.db).await.map_err(|e| e.to_string())?;
        }
    }
    if old != name {
        let _ = sqlx::query("UPDATE face_sightings SET person_name=? WHERE person_name=?")
            .bind(&name).bind(&old).execute(&state.db).await;
        // The EVENT label has to move too.
        //
        // `motion_events.sub_label` stores the recognised person as a name string
        // with no id, and `get_person_events` falls back to matching it with
        // `LIKE '%name%'`. Renaming without rewriting it orphaned every past
        // event: the person's history went quiet, and worse, the stale label
        // could then fuzzy-match a DIFFERENT person whose name contains the old
        // one. `forget_person` already clears this column; rename must maintain
        // it for the same reason.
        let _ = sqlx::query("UPDATE motion_events SET sub_label=? WHERE sub_label=?")
            .bind(&name).bind(&old).execute(&state.db).await;
    }
    schedule_classifier_retrain(&state.db);
    Ok(())
}

#[tauri::command]
pub async fn delete_person(state: State<'_, Arc<AppState>>, id: String) -> Result<(), String> {
    // Deleting ONLY the roster row used to strand every associated row as an
    // invisible orphan: linked face crops (person_id set → excluded from Train
    // AND from every gallery), hard negatives for a boundary that no longer
    // exists, body-gallery links (excluded from matching by the JOIN, excluded
    // from pruning by known_person_id being set = kept forever), and sightings
    // that polluted stats. Clean up every association — conservatively:
    // face CROPS are returned to the Train pool, never destroyed.
    let name: Option<String> = sqlx::query_scalar("SELECT name FROM known_persons WHERE id=?")
        .bind(&id).fetch_optional(&state.db).await.ok().flatten();

    sqlx::query("DELETE FROM known_persons WHERE id=?")
        .bind(&id).execute(&state.db).await.map_err(|e| e.to_string())?;
    // Linked face crops → back to the unlabeled (Train) pool.
    let _ = sqlx::query("UPDATE face_embeddings SET person_id=NULL WHERE person_id=?")
        .bind(&id).execute(&state.db).await;
    // Hard negatives reference this person's decision boundary — gone with them.
    let _ = sqlx::query("DELETE FROM face_negatives WHERE person_id=?")
        .bind(&id).execute(&state.db).await;
    // Body-gallery links → rejoin the anonymous pool under a fresh track id so
    // they age out naturally via prune_anon_bodies (2-day retention).
    let _ = sqlx::query(
        "UPDATE body_embeddings
            SET known_person_id=NULL,
                person_id='body_' || substr(lower(hex(randomblob(4))),1,8)
          WHERE known_person_id=?"
    ).bind(&id).execute(&state.db).await;
    // Sightings log: a removed person's activity must not linger in stats.
    if let Some(n) = &name {
        let _ = sqlx::query("DELETE FROM face_sightings WHERE person_name=?")
            .bind(n).execute(&state.db).await;
    }
    schedule_classifier_retrain(&state.db);
    Ok(())
}

/// ERASE a person — every biometric trace, not just the label.
///
/// Distinct from [`delete_person`] on purpose, because they answer different
/// questions. `delete_person` says "this label was wrong": it unlinks and returns
/// the face crops to the training pool so they can be relabelled. That is the
/// right behaviour for a mistake and the WRONG behaviour for "delete me".
///
/// Face descriptors are biometric data — special-category personal data under
/// GDPR Article 9 and the UK DPA, where the erasure right is not satisfied by
/// unlinking a name. This deletes the descriptors, the crops, the sightings and
/// the body-appearance vectors outright.
///
/// What it deliberately does NOT touch: recorded footage. Video is bulk
/// recording governed by the retention setting, not per-person data, and a
/// command that silently deleted hours of unrelated recording because one face
/// appeared in it would be its own disaster. Footage is erased by retention or
/// by the explicit range-delete in Review.
///
/// Returns the number of biometric rows removed, so the caller can tell the user
/// what actually happened rather than showing an unconditional "done".
#[tauri::command]
pub async fn forget_person(
    state: State<'_, Arc<AppState>>,
    id: String,
) -> Result<u64, String> {
    let removed = erase_biometrics(&state.db, &state.data_dir, &id).await;
    schedule_classifier_retrain(&state.db);
    Ok(removed)
}

/// The erasure itself, free of `AppState` so it can be tested against an
/// in-memory database. Every DELETE here names a real column — a typo'd one
/// fails at runtime and the caller reports success over data still on disk,
/// which is why `erasure_leaves_nothing_behind` exists.
pub(crate) async fn erase_biometrics(
    db: &sqlx::SqlitePool,
    data_dir: &std::path::Path,
    id: &str,
) -> u64 {
    let name: Option<String> = sqlx::query_scalar("SELECT name FROM known_persons WHERE id=?")
        .bind(id).fetch_optional(db).await.ok().flatten();

    // Face and body crops live out-of-DB as "@file:" blob refs (see blobstore.rs);
    // drop the files before the rows that point at them, or they become
    // unreachable garbage that sweep_orphans has to find later.
    let face_refs: Vec<(Option<String>, Option<String>)> = sqlx::query_as(
        "SELECT thumbnail_b64, context_b64 FROM face_embeddings WHERE person_id=?"
    ).bind(id).fetch_all(db).await.unwrap_or_default();
    let body_refs: Vec<(Option<String>,)> = sqlx::query_as(
        "SELECT thumbnail_b64 FROM body_embeddings WHERE known_person_id=?"
    ).bind(id).fetch_all(db).await.unwrap_or_default();
    for r in face_refs.iter().flat_map(|(a, b)| [a, b])
        .chain(body_refs.iter().map(|(a,)| a))
        .flatten()
    {
        crate::blobstore::delete(data_dir, r);
    }

    let mut removed = 0u64;
    // The descriptors themselves — the biometric identifiers.
    //
    // NOT in this list: face_sightings, which has no person_id column at all
    // (db.rs:83) — it is keyed by person_name, and is handled by name below.
    for sql in [
        "DELETE FROM face_embeddings WHERE person_id=?",
        "DELETE FROM face_negatives  WHERE person_id=?",
        "DELETE FROM body_embeddings WHERE known_person_id=?",
        "DELETE FROM body_negatives  WHERE known_person_id=?",
        "DELETE FROM known_persons   WHERE id=?",
    ] {
        match sqlx::query(sql).bind(id).execute(db).await {
            Ok(r)  => removed += r.rows_affected(),
            // Loud, because a silent failure here means data the user asked us to
            // destroy is still on disk while the UI says it is gone.
            Err(e) => tracing::error!("forget_person: {sql} failed: {e}"),
        }
    }
    if let Some(n) = &name {
        match sqlx::query("DELETE FROM face_sightings WHERE person_name=?")
            .bind(n).execute(db).await
        {
            Ok(r)  => removed += r.rows_affected(),
            Err(e) => tracing::error!("forget_person: face_sightings failed: {e}"),
        }
        // And the name must stop appearing on past events, or the identification
        // survives the erasure in every list the agent can read.
        let _ = sqlx::query("UPDATE motion_events SET sub_label=NULL, sub_label_score=NULL \
                             WHERE sub_label=? COLLATE NOCASE")
            .bind(n).execute(db).await;
    }

    // The trained classifier still encodes this face in its weights until it is
    // rebuilt from what remains.
    let _ = sqlx::query("DELETE FROM face_classifier_model").execute(db).await;

    tracing::info!("forget_person: erased {removed} biometric row(s) for {}",
        name.as_deref().unwrap_or(id));
    removed
}

#[tauri::command]
pub async fn mark_person_seen(state: State<'_, Arc<AppState>>, id: String) -> Result<(), String> {
    let now = Utc::now().to_rfc3339();
    sqlx::query("UPDATE known_persons SET last_seen_at=? WHERE id=?")
        .bind(&now).bind(&id).execute(&state.db).await.map_err(|e| e.to_string())?;
    Ok(())
}

// ─── Unknown-face training (standard "tag from recent events") ───────

#[derive(Debug, Serialize)]
pub struct UnknownFace {
    pub id:             String,
    pub thumbnail_b64:  String,        // 112×112 aligned JPEG
    pub quality:        f32,
    pub cam_id:         i64,
    pub event_id:       Option<String>,
    pub seen_at:        String,
    /// Closest enrolled person (Train-tab "looks like {name}") when the near-miss
    /// cosine is in [unknown_score, rec_threshold). None when nobody is close.
    pub suggested_name:  Option<String>,
    pub suggested_score: Option<f32>,
    /// The suggested person's ID — the UI binds confirm actions to THIS, never
    /// to the name string (renames / duplicate names mis-resolve by name).
    pub suggested_person_id: Option<String>,
}

/// Load every enrolled person's (id, name, parsed embeddings) once so a listing
/// can suggest candidates in-memory without re-querying per face.
async fn load_known(db: &sqlx::SqlitePool) -> Vec<(String, String, Vec<Vec<f32>>)> {
    let rows: Vec<(String, String, String)> = sqlx::query_as("SELECT id, name, embeddings FROM known_persons")
        .fetch_all(db).await.unwrap_or_default();
    rows.into_iter()
        .filter_map(|(id, name, j)| serde_json::from_str::<Vec<Vec<f32>>>(&j).ok().map(|e| (id, name, e)))
        .collect()
}

/// Best (id, name, score) for `emb` over preloaded known persons; returns the
/// suggestion only when the best cosine clears `floor` (the unknown-score band).
/// The ID is the authoritative identity — the UI must bind confirm/correct
/// actions to it, never to the display name (renames/dup names mis-resolve).
fn best_suggestion(known: &[(String, String, Vec<Vec<f32>>)], emb: &[f32], floor: f32)
    -> (Option<String>, Option<String>, Option<f32>)
{
    let mut best = (-1.0f32, String::new(), String::new());
    for (id, name, embs) in known {
        for stored in embs {
            if stored.len() != emb.len() { continue; }
            let s = cos(emb, stored);
            if s > best.0 { best = (s, id.clone(), name.clone()); }
        }
    }
    if best.0 >= floor && !best.1.is_empty() {
        (Some(best.1), Some(best.2), Some(best.0))
    } else { (None, None, None) }
}

/// Returns the N most-recent face sightings that aren't linked to a known
/// person yet — what the agent has seen but couldn't identify.
/// `min_quality` filters out blurry frames (default 0.20 — keep the floor low
/// because in practice most candid frames are below the laplacian sweet spot).
#[tauri::command]
pub async fn list_recent_unknown_faces(
    state: State<'_, Arc<AppState>>,
    limit: Option<i64>,
    days: Option<i64>,
    min_quality: Option<f32>,
) -> Result<Vec<UnknownFace>, String> {
    let limit = limit.unwrap_or(60).clamp(1, 500);
    let days  = days.unwrap_or(14).clamp(1, 365);
    let q_min = min_quality.unwrap_or(0.10);
    let floor = state.settings.read().await.face_unknown_score;
    let known = load_known(&state.db).await;
    let rows: Vec<(String, Option<String>, f64, i64, Option<String>, String, Vec<u8>, i64)> = sqlx::query_as(
        "SELECT id, thumbnail_b64, quality, cam_id, event_id, seen_at, descriptor, dim
           FROM face_embeddings
          WHERE person_id IS NULL
            AND thumbnail_b64 IS NOT NULL
            AND quality >= ?
            AND seen_at > datetime('now', ?)
          ORDER BY quality DESC, seen_at DESC
          LIMIT ?"
    )
    .bind(q_min as f64)
    .bind(format!("-{} days", days))
    .bind(limit)
    .fetch_all(&state.db).await.map_err(|e| e.to_string())?;

    // Descriptor decode + cosine over every enrolled gallery is CPU work (60 ×
    // 30-shot loops), and the old inline blobstore::resolve was a blocking fs
    // read per row on the async runtime — crops are now '@crop' markers served
    // by GET /face/:id/crop instead.
    tokio::task::spawn_blocking(move || {
        rows.into_iter().filter_map(|(id, thumb, quality, cam_id, event_id, seen_at, blob, dim)| {
            thumb.as_deref()?; // no stored crop → skip (matches old behavior)
            let (suggested_person_id, suggested_name, suggested_score) = if blob.len() == dim as usize * 4 {
                let v: Vec<f32> = blob.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                best_suggestion(&known, &v, floor)
            } else { (None, None, None) };
            Some(UnknownFace {
                id,
                thumbnail_b64: "@crop".into(),
                quality: quality as f32,
                cam_id,
                event_id,
                seen_at,
                suggested_name,
                suggested_score,
                suggested_person_id,
            })
        }).collect()
    }).await.map_err(|e| e.to_string())
}

/// One stored face crop for a person — used by the per-person detail gallery so
/// the user can SEE (and curate) what the cameras matched to this identity.
#[derive(Debug, Serialize)]
pub struct FaceShot {
    pub id:            String,
    pub thumbnail_b64: String,
    pub quality:       f32,
    pub cam_id:        i64,
    pub seen_at:       String,
}

/// Every stored face crop linked to one known person, newest first.
#[tauri::command]
pub async fn list_person_faces(
    state: State<'_, Arc<AppState>>,
    person_id: String,
    limit: Option<i64>,
) -> Result<Vec<FaceShot>, String> {
    let limit = limit.unwrap_or(60).clamp(1, 500);
    let rows: Vec<(String, Option<String>, f64, i64, String)> = sqlx::query_as(
        "SELECT id, thumbnail_b64, quality, cam_id, seen_at
           FROM face_embeddings
          WHERE person_id = ? AND thumbnail_b64 IS NOT NULL
          ORDER BY seen_at DESC
          LIMIT ?"
    )
    .bind(&person_id).bind(limit)
    .fetch_all(&state.db).await.map_err(|e| e.to_string())?;

    Ok(rows.into_iter().filter_map(|(id, thumb, quality, cam_id, seen_at)| {
        Some(FaceShot { id, thumbnail_b64: crate::blobstore::resolve(&state.data_dir, thumb.as_deref()?), quality: quality as f32, cam_id, seen_at })
    }).collect())
}

/// Drop one stored face crop (mature NVRs' "remove low-quality before reprocessing").
/// Only removes the runtime sighting row; the person's enrolled embeddings are
/// kept in `known_persons.embeddings`.
#[tauri::command]
pub async fn delete_face_embedding(
    state: State<'_, Arc<AppState>>,
    id: String,
) -> Result<(), String> {
    // Remove the offloaded display blobs too (no orphaned files on disk).
    if let Ok(Some((t, c))) = sqlx::query_as::<_, (Option<String>, Option<String>)>(
        "SELECT thumbnail_b64, context_b64 FROM face_embeddings WHERE id = ?"
    ).bind(&id).fetch_optional(&state.db).await {
        if let Some(t) = t { crate::blobstore::delete(&state.data_dir, &t); }
        if let Some(c) = c { crate::blobstore::delete(&state.data_dir, &c); }
    }
    sqlx::query("DELETE FROM face_embeddings WHERE id = ?")
        .bind(&id).execute(&state.db).await.map_err(|e| e.to_string())?;
    Ok(())
}

/// Wipe ALL un-tagged (stranger) face crops — the Train backlog — in one tap.
/// Enrolled people's linked faces (`person_id` set) are kept. Cameras re-capture
/// clean bbox crops via the continuous-capture loop. Returns rows removed.
#[tauri::command]
pub async fn clear_unknown_faces(state: State<'_, Arc<AppState>>) -> Result<u64, String> {
    // Collect the blob refs first, then delete the files off-thread (can be
    // thousands) so we don't leave orphaned crops on disk.
    let refs: Vec<(Option<String>, Option<String>)> = sqlx::query_as(
        "SELECT thumbnail_b64, context_b64 FROM face_embeddings WHERE person_id IS NULL"
    ).fetch_all(&state.db).await.unwrap_or_default();
    if !refs.is_empty() {
        let data_dir = state.data_dir.clone();
        tokio::task::spawn_blocking(move || {
            for (t, c) in &refs {
                if let Some(t) = t { crate::blobstore::delete(&data_dir, t); }
                if let Some(c) = c { crate::blobstore::delete(&data_dir, c); }
            }
        }).await.ok();
    }
    let res = sqlx::query("DELETE FROM face_embeddings WHERE person_id IS NULL")
        .execute(&state.db).await.map_err(|e| e.to_string())?;
    Ok(res.rows_affected())
}

/// Hourly retention for the per-person recognition LOG (`face_embeddings` rows
/// with `person_id` set): keep the newest 500 per person, delete older rows and
/// their offloaded display blobs. Without this the log grew unbounded — one
/// frequently-seen resident accumulated 9,883 crops (94% of the table).
/// User-approved standard hygiene. NEVER touches: unlinked (Train) rows,
/// `known_persons.embeddings` (the matcher input), or any enrolled identity.
/// Takes OWNED args so callers can `tokio::spawn` it — the first prune after a
/// bloated history deletes thousands of rows + ~2× that in blob files, and must
/// never stall the inference tick that triggered it.
pub(crate) async fn prune_linked_face_crops(db: sqlx::SqlitePool, data_dir: std::path::PathBuf) {
    const KEEP_PER_PERSON: i64 = 500;
    let pids: Vec<(String,)> = sqlx::query_as(
        "SELECT person_id FROM face_embeddings WHERE person_id IS NOT NULL
          GROUP BY person_id HAVING COUNT(*) > ?"
    ).bind(KEEP_PER_PERSON).fetch_all(&db).await.unwrap_or_default();
    for (pid,) in pids {
        // Rows older than the newest KEEP_PER_PERSON for this person.
        let refs: Vec<(String, Option<String>, Option<String>)> = sqlx::query_as(
            "SELECT id, thumbnail_b64, context_b64 FROM face_embeddings
              WHERE person_id=? AND id NOT IN (
                SELECT id FROM face_embeddings WHERE person_id=?
                 ORDER BY seen_at DESC LIMIT ?)"
        ).bind(&pid).bind(&pid).bind(KEEP_PER_PERSON).fetch_all(&db).await.unwrap_or_default();
        if refs.is_empty() { continue; }
        let n = refs.len();
        let ids: Vec<String> = refs.iter().map(|(id, _, _)| id.clone()).collect();
        // Offloaded display blobs go too (same pattern as clear_unknown_faces) —
        // deleting rows but not files would strand thousands of crops on disk.
        let blobs: Vec<(Option<String>, Option<String>)> =
            refs.into_iter().map(|(_, t, c)| (t, c)).collect();
        let dd = data_dir.clone();
        tokio::task::spawn_blocking(move || {
            for (t, c) in &blobs {
                if let Some(t) = t { crate::blobstore::delete(&dd, t); }
                if let Some(c) = c { crate::blobstore::delete(&dd, c); }
            }
        }).await.ok();
        // NOT error-swallowed: a transient SQLITE_BUSY on one chunk (seen on the
        // write-heavy boot window) previously vanished silently, leaving a slice
        // of old rows behind until the next hourly pass. Log + retry once.
        for chunk in ids.chunks(500) {
            let ph = vec!["?"; chunk.len()].join(",");
            let sql = format!("DELETE FROM face_embeddings WHERE id IN ({ph})");
            for attempt in 0..2u8 {
                let mut q = sqlx::query(&sql);
                for id in chunk { q = q.bind(id); }
                match q.execute(&db).await {
                    Ok(_) => break,
                    Err(e) if attempt == 0 => {
                        tracing::warn!("face-crop retention: chunk delete failed ({e}) — retrying once");
                        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
                    }
                    Err(e) => tracing::warn!("face-crop retention: chunk delete failed twice ({e}) — will heal on next hourly pass"),
                }
            }
        }
        tracing::info!("face-crop retention: pruned {n} old linked crop(s) for person {pid} (kept newest {KEEP_PER_PERSON})");
    }
}

/// The full source frame a captured face came from (downscaled JPEG, base64) for
/// the Train UI's click-to-expand. Falls back to the face crop for rows stored
/// before context was captured.
#[tauri::command]
pub async fn get_face_context(
    state: State<'_, Arc<AppState>>,
    face_id: String,
) -> Result<Option<String>, String> {
    let row: Option<(Option<String>,)> = sqlx::query_as(
        "SELECT COALESCE(context_b64, thumbnail_b64) FROM face_embeddings WHERE id=?"
    ).bind(&face_id).fetch_optional(&state.db).await.map_err(|e| e.to_string())?;
    Ok(row.and_then(|(c,)| c).map(|c| crate::blobstore::resolve(&state.data_dir, &c)))
}

/// A recent recognition of a KNOWN person — the mature NVRs "Recent Recognitions"
/// feed. Surfaces that face recognition is working + who's been around lately.
#[derive(Debug, Serialize)]
pub struct Recognition {
    /// face_embeddings.id — needed so a WRONG recognition can be corrected.
    pub id:            String,
    pub person_id:     String,
    pub name:          String,
    pub role:          String,
    pub thumbnail_b64: String,
    pub quality:       f32,
    pub cam_id:        i64,
    pub seen_at:       String,
    /// Naming provenance — WHICH recognizer, its score, runner-up margin, and the
    /// event it came from. NULL on rows stored before traceability shipped.
    pub match_method:  Option<String>,
    pub match_score:   Option<f32>,
    pub match_margin:  Option<f32>,
    pub event_id:      Option<String>,
}

/// Most-recent recognitions of enrolled people across all cameras.
#[tauri::command]
pub async fn list_recent_recognitions(
    state: State<'_, Arc<AppState>>,
    limit: Option<i64>,
    days: Option<i64>,
) -> Result<Vec<Recognition>, String> {
    let limit = limit.unwrap_or(40).clamp(1, 200);
    let days  = days.unwrap_or(7).clamp(1, 365);
    let rows: Vec<(String, String, String, String, Option<String>, f64, i64, String, Option<String>, Option<f64>, Option<f64>, Option<String>)> = sqlx::query_as(
        "SELECT f.id, k.id, k.name, k.role, f.thumbnail_b64, f.quality, f.cam_id, f.seen_at,
                f.match_method, f.match_score, f.match_margin, f.event_id
           FROM face_embeddings f
           JOIN known_persons k ON k.id = f.person_id
          WHERE f.person_id IS NOT NULL
            AND f.thumbnail_b64 IS NOT NULL
            AND f.seen_at > datetime('now', ?)
          ORDER BY f.seen_at DESC
          LIMIT ?"
    )
    .bind(format!("-{} days", days)).bind(limit)
    .fetch_all(&state.db).await.map_err(|e| e.to_string())?;

    Ok(rows.into_iter().filter_map(|(id, person_id, name, role, thumb, quality, cam_id, seen_at, match_method, match_score, match_margin, event_id)| {
        Some(Recognition {
            id, person_id, name, role,
            thumbnail_b64: crate::blobstore::resolve(&state.data_dir, thumb.as_deref()?),
            quality: quality as f32, cam_id, seen_at,
            match_method,
            match_score: match_score.map(|v| v as f32),
            match_margin: match_margin.map(|v| v as f32),
            event_id,
        })
    }).collect())
}

/// Attach an unmatched face embedding to an existing `known_persons` row.
/// Copies the embedding vector into the person's JSON-encoded `embeddings`
/// array so future matches improve, then marks the row as linked.
#[tauri::command]
pub async fn assign_face_to_person(
    state: State<'_, Arc<AppState>>,
    face_id:   String,
    person_id: String,
) -> Result<(), String> {
    // Pull the embedding blob + capture quality.
    let (descriptor, dim, quality): (Vec<u8>, i64, f64) = sqlx::query_as(
        "SELECT descriptor, dim, quality FROM face_embeddings WHERE id=? AND person_id IS NULL"
    ).bind(&face_id).fetch_one(&state.db).await
        .map_err(|e| format!("face not found or already linked: {e}"))?;

    if descriptor.len() != (dim as usize) * 4 {
        return Err("descriptor blob size mismatch".into());
    }
    let vec: Vec<f32> = descriptor.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // QUALITY GATE: a blurry confirmation still TAGS the row (below) but must
    // not become matcher input — soft shots dilute recognition for everyone.
    let q_floor = state.settings.read().await.face_quality_floor;
    let sharp_enough = quality as f32 >= q_floor;
    if !sharp_enough {
        tracing::info!("assign_face: tagged but not enrolled (quality {quality:.2} < floor {q_floor:.2})");
    }

    // Append to the person's embedding array (diversity gate + 30-shot cap —
    // a near-duplicate shot tags the row below but doesn't grow the gallery).
    let (mut embs_json,): (String,) = sqlx::query_as(
        "SELECT embeddings FROM known_persons WHERE id=?"
    ).bind(&person_id).fetch_one(&state.db).await
        .map_err(|e| format!("person not found: {e}"))?;
    let mut embs: Vec<Vec<f32>> = serde_json::from_str(&embs_json).unwrap_or_default();
    if sharp_enough && push_enrolled_shot(&mut embs, vec) {
        embs_json = serde_json::to_string(&embs).map_err(|e| e.to_string())?;
        sqlx::query("UPDATE known_persons SET embeddings=?, last_seen_at=datetime('now') WHERE id=?")
            .bind(&embs_json).bind(&person_id)
            .execute(&state.db).await.map_err(|e| e.to_string())?;
    } else {
        sqlx::query("UPDATE known_persons SET last_seen_at=datetime('now') WHERE id=?")
            .bind(&person_id)
            .execute(&state.db).await.map_err(|e| e.to_string())?;
    }

    sqlx::query("UPDATE face_embeddings SET person_id=? WHERE id=?")
        .bind(&person_id).bind(&face_id)
        .execute(&state.db).await.map_err(|e| e.to_string())?;
    schedule_classifier_retrain(&state.db);
    Ok(())
}

/// CORRECT a recognition: re-tag a face that's ALREADY labelled (the pipeline or a
/// previous tap matched it to the wrong person) to the right person — or to "not
/// them" (`correct_person_id = None`). Unlike `assign_face_to_person` (which only
/// accepts un-tagged rows), this is the "fix a mistake" path:
///   1. If it was wrongly attributed to someone, store the descriptor as a HARD
///      NEGATIVE for that person so the classifier stops repeating the error.
///   2. Re-tag the face row to the correct person (or NULL).
///   3. If a correct person is given, append the descriptor to their gallery.
///   4. Retrain the hybrid head.
#[tauri::command]
pub async fn correct_face(
    state: State<'_, Arc<AppState>>,
    face_id: String,
    correct_person_id: Option<String>,
) -> Result<(), String> {
    // Current label + descriptor + capture quality of the face being corrected.
    let (descriptor, dim, wrong_person, quality): (Vec<u8>, i64, Option<String>, f64) = sqlx::query_as(
        "SELECT descriptor, dim, person_id, quality FROM face_embeddings WHERE id=?"
    ).bind(&face_id).fetch_one(&state.db).await
        .map_err(|e| format!("face not found: {e}"))?;
    if descriptor.len() != (dim as usize) * 4 {
        return Err("descriptor blob size mismatch".into());
    }
    let vec: Vec<f32> = descriptor.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    // 1. Record a hard negative for whoever it was WRONGLY attributed to (skip if it
    //    was unlabeled, or if the "correction" is a no-op to the same person).
    if let Some(wrong) = wrong_person.as_deref() {
        if !wrong.is_empty() && correct_person_id.as_deref() != Some(wrong) {
            let _ = sqlx::query(
                "INSERT INTO face_negatives(id, person_id, descriptor, dim) VALUES(?,?,?,?)"
            ).bind(Uuid::new_v4().to_string()).bind(wrong).bind(&descriptor).bind(dim)
             .execute(&state.db).await;
            // Drop the bad sample from the wrong person's enrolled gallery if it's there
            // (best-effort exact-match removal), so it stops pulling matches in.
            if let Ok((embs_json,)) = sqlx::query_as::<_, (String,)>(
                "SELECT embeddings FROM known_persons WHERE id=?"
            ).bind(wrong).fetch_one(&state.db).await {
                let mut embs: Vec<Vec<f32>> = serde_json::from_str(&embs_json).unwrap_or_default();
                let before = embs.len();
                embs.retain(|e| e.len() != vec.len() || e.iter().zip(&vec).any(|(a, b)| (a - b).abs() > 1e-6));
                if embs.len() != before {
                    if let Ok(j) = serde_json::to_string(&embs) {
                        let _ = sqlx::query("UPDATE known_persons SET embeddings=? WHERE id=?")
                            .bind(j).bind(wrong).execute(&state.db).await;
                    }
                }
            }
        }
    }

    // 2 + 3. Re-tag the face row and, when a correct person is given, grow their gallery.
    match correct_person_id.as_deref() {
        Some(correct) if !correct.is_empty() => {
            if let Ok((embs_json,)) = sqlx::query_as::<_, (String,)>(
                "SELECT embeddings FROM known_persons WHERE id=?"
            ).bind(correct).fetch_one(&state.db).await {
                let mut embs: Vec<Vec<f32>> = serde_json::from_str(&embs_json).unwrap_or_default();
                // QUALITY GATE + shared append policy (diversity gate, newest-30
                // cap): a blurry correction re-tags the row but must not become
                // matcher input.
                let q_floor = state.settings.read().await.face_quality_floor;
                let sharp_enough = quality as f32 >= q_floor;
                if !sharp_enough {
                    tracing::info!("correct_face: re-tagged but not enrolled (quality {quality:.2} < floor {q_floor:.2})");
                }
                if sharp_enough && push_enrolled_shot(&mut embs, vec) {
                    let j = serde_json::to_string(&embs).map_err(|e| e.to_string())?;
                    sqlx::query("UPDATE known_persons SET embeddings=?, last_seen_at=datetime('now') WHERE id=?")
                        .bind(j).bind(correct).execute(&state.db).await.map_err(|e| e.to_string())?;
                } else {
                    sqlx::query("UPDATE known_persons SET last_seen_at=datetime('now') WHERE id=?")
                        .bind(correct).execute(&state.db).await.map_err(|e| e.to_string())?;
                }
            } else {
                return Err("correct person not found".into());
            }
            sqlx::query("UPDATE face_embeddings SET person_id=? WHERE id=?")
                .bind(correct).bind(&face_id).execute(&state.db).await.map_err(|e| e.to_string())?;
        }
        _ => {
            // "Not them" — return the face to the unlabeled pool.
            sqlx::query("UPDATE face_embeddings SET person_id=NULL WHERE id=?")
                .bind(&face_id).execute(&state.db).await.map_err(|e| e.to_string())?;
        }
    }

    schedule_classifier_retrain(&state.db);
    Ok(())
}

/// Cosine similarity of two L2-normalised embeddings (dot product).
fn cos(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() { return -1.0; }
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Enrolled-gallery append policy — ONE rule for every path that grows a
/// person's matcher-input shots (`known_persons.embeddings`), mature NVRs-calibrated:
///  * DIVERSITY GATE: skip when the new shot is a near-duplicate (cosine ≥ 0.92,
///    the same bar as guided enrollment) of an existing shot — mature NVRs: "train
///    no more than 4-6 similar images … to avoid over-fitting". Tagging 20
///    near-identical crops must not crowd out genuinely diverse angles.
///  * CAP: keep the newest 30 shots (mature NVRs' 20-30 sweet spot). Previously
///    only 2 of the 4 append paths capped, so galleries grew unbounded.
/// Returns whether the shot was appended (callers treat a skip as success —
/// the face ROW still gets tagged; only the redundant matcher append is skipped).
const ENROLLED_DIVERSITY: f32 = 0.92;
const ENROLLED_CAP: usize = 30;
fn push_enrolled_shot(embs: &mut Vec<Vec<f32>>, v: Vec<f32>) -> bool {
    if embs.iter().any(|e| cos(e, &v) >= ENROLLED_DIVERSITY) { return false; }
    embs.push(v);
    if embs.len() > ENROLLED_CAP {
        let n = embs.len() - ENROLLED_CAP;
        embs.drain(0..n);
    }
    true
}

/// A cluster of unrecognised face sightings that look like the same person, so
/// the user can name a whole individual at once instead of one frame at a time.
#[derive(Debug, Clone, Serialize)]
pub struct UnknownCluster {
    pub cluster_id:    String,
    /// `"@crop"` marker — the representative face, served via /face/{rep_id}/crop.
    pub rep_thumbnail: String,
    /// Face id of the representative (highest-quality) member — the crop URL key.
    pub rep_id:        String,
    pub count:         i64,
    pub cameras:       Vec<i64>,
    pub last_seen:     String,
    /// First time this recurring stranger was seen (min seen_at across the cluster).
    pub first_seen:    String,
    /// How many DISTINCT days this person has appeared — a strong "regular" signal.
    pub days_active:   i64,
    /// Human-readable when-they-show-up pattern, e.g. "Evenings", "Mornings",
    /// "Overnight", or "Any time" — derived from the sighting hour histogram.
    pub time_pattern:  String,
    pub face_ids:      Vec<String>,
    /// Closest enrolled person to the cluster centroid (Train "looks like {name}").
    pub suggested_name:  Option<String>,
    pub suggested_score: Option<f32>,
    /// The suggested person's ID — bind confirm actions to this, not the name.
    pub suggested_person_id: Option<String>,
    /// A spread of member faces (id + crop, ≤9) so the modal can show the faces
    /// grouped here AND expand each into its full frame via `get_face_context`.
    pub samples: Vec<FaceSample>,
}

#[derive(Debug, Clone, Serialize)]
pub struct FaceSample {
    pub id:        String,
    /// `"@crop"` marker — fetch via GET /face/:id/crop (URL-served, cacheable).
    pub thumbnail: String,
}

/// Discover RECURRING unknown people across the WHOLE untagged gallery — the
/// Edge-AI NVRs "AutoGroup" done right. Instead of greedy-clustering only the recent
/// 400, we build a kNN similarity graph with usearch and run **Chinese Whispers**
/// (Biemann 2006 — the best-in-class, chaining-resistant face-clustering algorithm),
/// so a stranger seen 20× over 3 weeks is grouped as ONE person, not scattered.
/// ArcFace (512-d) rows only; clusters below `MIN_SIGHTINGS` dropped as noise; most-
/// recurring first. Read-only — never mutates face data (hard rule).
#[tauri::command]
pub async fn list_unknown_clusters(
    state: State<'_, Arc<AppState>>,
    days: Option<i64>,
    min_quality: Option<f32>,
) -> Result<Vec<UnknownCluster>, String> {
    let days  = days.unwrap_or(30).clamp(1, 365); // wider window now that we scale
    let q_min = min_quality.unwrap_or(0.10);
    // Link two unknowns at the SAME bar recognition uses ("two unknowns are one
    // person" == "a face is recognized"), so clustering never disagrees with
    // recognition. `floor` surfaces the centroid's closest enrolled candidate.
    let (join, floor) = {
        let s = state.settings.read().await;
        (s.face_recognition_threshold, s.face_unknown_score)
    };

    // ── Fingerprint cache: clustering 20k descriptors per People-tab visit is
    // absurd — one cheap COUNT+MAX query decides whether ANYTHING could have
    // changed (new faces, tags, deletes all shift it; settings/params are part
    // of the key). Hit → cached clone, miss → recompute + store.
    let window = format!("-{} days", days);
    let fp_core: Option<String> = sqlx::query_scalar(
        "SELECT COUNT(*) || '|' || COALESCE(MAX(seen_at),'')
           FROM face_embeddings
          WHERE person_id IS NULL AND thumbnail_b64 IS NOT NULL
            AND dim = 512 AND quality >= ?
            AND seen_at > datetime('now', ?)"
    ).bind(q_min as f64).bind(&window)
     .fetch_optional(&state.db).await.ok().flatten();
    let fingerprint = format!("{}|{join}|{floor}|{days}|{q_min}", fp_core.unwrap_or_default());
    {
        let cache = state.unknown_clusters_cache.lock().await;
        if let Some((fp, cached)) = cache.as_ref() {
            if *fp == fingerprint {
                tracing::debug!("unknown-clusters: cache hit");
                return Ok(cached.clone());
            }
        }
    }
    tracing::info!("unknown-clusters: cache miss — reclustering");

    let known = load_known(&state.db).await;
    // Whole untagged gallery in-window (no thumbnails — crops are URL-served
    // via /face/:id/crop, so the payload carries markers only). Capped for
    // safety; usearch + Chinese Whispers handle this scale in O(n·k).
    const MAX_FACES: i64 = 20_000;
    let rows: Vec<(String, Vec<u8>, i64, f64, i64, String)> = sqlx::query_as(
        "SELECT id, descriptor, dim, quality, cam_id, seen_at
           FROM face_embeddings
          WHERE person_id IS NULL AND thumbnail_b64 IS NOT NULL
            AND dim = 512 AND quality >= ?
            AND seen_at > datetime('now', ?)
          ORDER BY seen_at DESC
          LIMIT ?"
    ).bind(q_min as f64).bind(&window).bind(MAX_FACES)
     .fetch_all(&state.db).await.map_err(|e| e.to_string())?;

    struct Meta { id: String, quality: f32, cam_id: i64, seen_at: String }
    // Descriptor decode (up to ~40MB of blobs) + kNN graph + Chinese Whispers
    // are ALL CPU-bound — one spawn_blocking keeps the async runtime (and the
    // UI's IPC commands) responsive while this crunches.
    let (vectors, meta, labels) = tokio::task::spawn_blocking(move || {
        let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(rows.len());
        let mut meta: Vec<Meta> = Vec::with_capacity(rows.len());
        for (id, blob, dim, quality, cam_id, seen_at) in rows {
            if blob.len() != dim as usize * 4 { continue; }
            let v: Vec<f32> = blob.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            vectors.push(v);
            meta.push(Meta { id, quality: quality as f32, cam_id, seen_at });
        }
        let labels = if vectors.len() < 2 { Vec::new() } else {
            let edges = crate::vector_index::knn_graph(&vectors, 16, join);
            crate::vector_index::chinese_whispers(vectors.len(), &edges, 30)
        };
        (vectors, meta, labels)
    }).await.map_err(|e| e.to_string())?;
    if vectors.len() < 2 {
        let mut cache = state.unknown_clusters_cache.lock().await;
        *cache = Some((fingerprint, Vec::new()));
        return Ok(Vec::new());
    }

    // Group member indices by cluster label.
    let mut groups: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
    for (i, &lab) in labels.iter().enumerate() { groups.entry(lab).or_default().push(i); }

    const MIN_SIGHTINGS: usize = 3; // a genuinely RECURRING stranger, not a one-off

    let dim = vectors[0].len();
    let mut out: Vec<UnknownCluster> = Vec::new();
    for (label, idxs) in groups {
        if idxs.len() < MIN_SIGHTINGS { continue; }

        // Centroid → "looks like {enrolled}" suggestion.
        let mut centroid = vec![0.0f32; dim];
        for &i in &idxs { for d in 0..dim { centroid[d] += vectors[i][d]; } }
        for x in centroid.iter_mut() { *x /= idxs.len() as f32; }
        let (suggested_person_id, suggested_name, suggested_score) = best_suggestion(&known, &centroid, floor);

        // Rich spatial + temporal metadata.
        let mut cameras: Vec<i64> = idxs.iter().map(|&i| meta[i].cam_id).collect();
        cameras.sort_unstable(); cameras.dedup();
        let seens: Vec<&str> = idxs.iter().map(|&i| meta[i].seen_at.as_str()).collect();
        let first_seen = seens.iter().min().map(|s| s.to_string()).unwrap_or_default();
        let last_seen  = seens.iter().max().map(|s| s.to_string()).unwrap_or_default();
        let mut days_set: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for s in &seens { days_set.insert(&s[..s.len().min(10)]); } // YYYY-MM-DD prefix
        let days_active = days_set.len() as i64;
        let time_pattern = time_pattern_of(&seens);

        // Representative + sample crops: highest-quality members, shipped as
        // '@crop' markers (URL-served + Chromium-cached via /face/:id/crop) —
        // no per-sample DB round-trips or blocking fs reads anymore.
        let mut ranked = idxs.clone();
        ranked.sort_by(|&a, &b| meta[b].quality.partial_cmp(&meta[a].quality).unwrap_or(std::cmp::Ordering::Equal));
        let samples: Vec<FaceSample> = ranked.iter().take(9)
            .map(|&i| FaceSample { id: meta[i].id.clone(), thumbnail: "@crop".into() })
            .collect();
        let rep_id = ranked.first().map(|&i| meta[i].id.clone()).unwrap_or_default();

        out.push(UnknownCluster {
            cluster_id: format!("c{label}"),
            rep_thumbnail: "@crop".into(),
            rep_id,
            count: idxs.len() as i64,
            cameras,
            last_seen,
            first_seen,
            days_active,
            time_pattern,
            face_ids: idxs.iter().map(|&i| meta[i].id.clone()).collect(),
            suggested_name,
            suggested_score,
            suggested_person_id,
            samples,
        });
    }
    // Most-recurring first (strongest "you should label me"), then most-recent.
    out.sort_by(|a, b| b.count.cmp(&a.count).then(b.last_seen.cmp(&a.last_seen)));
    let mut cache = state.unknown_clusters_cache.lock().await;
    *cache = Some((fingerprint, out.clone()));
    Ok(out)
}

/// Summarise WHEN a recurring person tends to appear, from their sighting
/// timestamps → a friendly label. Buckets by LOCAL hour (timestamps are UTC; we
/// convert). "Any time" when no 6-hour window holds a clear majority.
fn time_pattern_of(seens: &[&str]) -> String {
    use chrono::{Local, NaiveDateTime, TimeZone, Timelike, Utc};
    let mut buckets = [0u32; 4]; // 0=overnight(0-6) 1=morning(6-12) 2=afternoon(12-18) 3=evening(18-24)
    let mut total = 0u32;
    for s in seens {
        let naive = NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S")
            .or_else(|_| NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S"))
            .or_else(|_| chrono::DateTime::parse_from_rfc3339(s).map(|d| d.naive_utc()));
        if let Ok(ndt) = naive {
            let h = Utc.from_utc_datetime(&ndt).with_timezone(&Local).hour();
            buckets[(h / 6).min(3) as usize] += 1;
            total += 1;
        }
    }
    if total == 0 { return "Any time".into(); }
    let (bi, &bmax) = buckets.iter().enumerate().max_by_key(|(_, &c)| c).unwrap();
    if (bmax as f32) < 0.55 * total as f32 { return "Any time".into(); }
    ["Overnight", "Mornings", "Afternoons", "Evenings"][bi].to_string()
}

/// One clip-linked sighting of a person: the EVENT a face/body was captured in,
/// with camera (location), time, thumbnail, and the `event_id` needed to play the
/// clip (NVR export works regardless of `clip_path`).
#[derive(Debug, Serialize)]
pub struct PersonSighting {
    pub event_id:   String,
    pub cam_id:     i64,
    pub seen_at:    String,
    pub thumbnail:  String,
    pub ai_summary: Option<String>,
}

/// The clip-linked sighting history for a recurring person's cluster — every event
/// its faces appeared in, newest first, with WHERE (camera) + WHEN + a thumbnail +
/// the event_id to play the footage. Powers "show me the clips + last-seen location
/// of this stranger".
#[tauri::command]
pub async fn get_person_sightings(
    state: State<'_, Arc<AppState>>,
    face_ids: Vec<String>,
) -> Result<Vec<PersonSighting>, String> {
    if face_ids.is_empty() { return Ok(Vec::new()); }
    // The events those face crops were captured in.
    let ph = face_ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
    let sql = format!("SELECT DISTINCT event_id FROM face_embeddings WHERE id IN ({ph}) AND event_id IS NOT NULL");
    let mut q = sqlx::query_scalar::<_, String>(&sql);
    for id in &face_ids { q = q.bind(id); }
    let event_ids: Vec<String> = q.fetch_all(&state.db).await.map_err(|e| e.to_string())?;
    if event_ids.is_empty() { return Ok(Vec::new()); }
    let ph2 = event_ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
    let sql2 = format!(
        "SELECT id, cam_id, started_at, thumbnail, ai_summary FROM motion_events
          WHERE id IN ({ph2}) ORDER BY started_at DESC"
    );
    let mut q2 = sqlx::query_as::<_, (String, Option<i64>, String, Option<String>, Option<String>)>(&sql2);
    for id in &event_ids { q2 = q2.bind(id); }
    let rows = q2.fetch_all(&state.db).await.map_err(|e| e.to_string())?;
    Ok(rows.into_iter().map(|(id, cam, seen, thumb, ai)| PersonSighting {
        event_id:   id,
        cam_id:     cam.unwrap_or(0),
        seen_at:    seen,
        thumbnail:  crate::blobstore::resolve(&state.data_dir, &thumb.unwrap_or_default()),
        ai_summary: ai,
    }).collect())
}

/// One video event a person appears in — the standard "events for this
/// person" record. `event` is the STANDARD MotionEvent shape (thumbnail is the
/// tiny '@thumb' marker, resolved by the frontend via /footage/:id/thumbnail —
/// inlining 500 full thumbnails shipped ~25MB over IPC). `person_crop` is the
/// Mature NVRs object-crop: the face crop of THIS person in THIS event (small),
/// so previews can show WHO, not just the scene.
#[derive(Debug, Serialize)]
pub struct PersonEvent {
    pub event:       crate::MotionEvent,
    pub person_crop: Option<String>,
}

/// Every motion event a person appears in, newest first.
/// Identity is EITHER an enrolled `person_id` (three UNIONed sources — id-keyed
/// face_sightings, linked face_embeddings crops, legacy sub_label LIKE name) OR
/// a set of unknown-cluster `face_ids` (the Train-tab strangers), so unknown
/// people's events are filterable too. Powers the PersonDetail timeline and the
/// Review "People" person-mode feed.
#[tauri::command]
pub async fn get_person_events(
    state: State<'_, Arc<AppState>>,
    person_id: Option<String>,
    face_ids: Option<Vec<String>>,
    days: Option<i64>,
    limit: Option<i64>,
) -> Result<Vec<PersonEvent>, String> {
    let days  = days.unwrap_or(30).clamp(1, 365);
    let limit = limit.unwrap_or(100).clamp(1, 500);
    let window = format!("-{days} days");

    // Best face crop per event for THIS identity (quality-ranked; first wins).
    // Crops are small aligned JPEGs — safe to inline, unlike event thumbnails.
    let mut crop_by_event: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    let crop_rows: Vec<(Option<String>, Option<String>)> = match (&person_id, &face_ids) {
        (Some(pid), _) => sqlx::query_as(
            "SELECT event_id, thumbnail_b64 FROM face_embeddings
              WHERE person_id=? AND event_id IS NOT NULL AND thumbnail_b64 IS NOT NULL
              ORDER BY quality DESC LIMIT 400"
        ).bind(pid).fetch_all(&state.db).await.unwrap_or_default(),
        (None, Some(ids)) if !ids.is_empty() => {
            let ph = ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
            let sql = format!(
                "SELECT event_id, thumbnail_b64 FROM face_embeddings
                  WHERE id IN ({ph}) AND event_id IS NOT NULL AND thumbnail_b64 IS NOT NULL
                  ORDER BY quality DESC"
            );
            let mut q = sqlx::query_as(&sql);
            for id in ids { q = q.bind(id); }
            q.fetch_all(&state.db).await.unwrap_or_default()
        }
        _ => Vec::new(),
    };
    for (ev, thumb) in crop_rows {
        if let (Some(ev), Some(t)) = (ev, thumb) {
            crop_by_event.entry(ev).or_insert(t);
        }
    }

    // Event rows — standard MotionEvent columns with the '@thumb' marker
    // (same convention as get_motion_events; see events_cmds.rs).
    const EV_SELECT: &str = "SELECT m.id, m.started_at, m.ended_at, m.duration_secs, m.peak_score, m.clip_path, \
        CASE WHEN m.thumbnail IS NOT NULL AND m.thumbnail <> '' THEN '@thumb' END AS thumbnail, \
        m.detections, m.ai_summary, m.cam_id, m.event_category, m.recognized_plate, m.dominant_label, m.sub_label, m.first_object_at \
        FROM motion_events m";
    type EvRow = (String, String, Option<String>, Option<f64>, f64, Option<String>, Option<String>, Option<String>, Option<String>, Option<i64>, Option<String>, Option<String>, Option<String>, Option<String>, Option<String>);
    let rows: Vec<EvRow> = match (&person_id, &face_ids) {
        (Some(pid), _) => {
            // Display name for the legacy sub_label fallback (empty disables it).
            let name: String = sqlx::query_scalar("SELECT name FROM known_persons WHERE id=?")
                .bind(pid).fetch_optional(&state.db).await.ok().flatten().unwrap_or_default();
            let sql = format!(
                "{EV_SELECT}
                  WHERE m.started_at > datetime('now', ?1)
                    AND (
                      m.id IN (SELECT event_id FROM face_sightings  WHERE person_id=?2 AND event_id IS NOT NULL)
                      OR m.id IN (SELECT event_id FROM face_embeddings WHERE person_id=?2 AND event_id IS NOT NULL)
                      OR (?3 != '' AND m.sub_label LIKE '%' || ?3 || '%')
                    )
                  ORDER BY m.started_at DESC LIMIT ?4"
            );
            sqlx::query_as(&sql)
                .bind(&window).bind(pid).bind(&name).bind(limit)
                .fetch_all(&state.db).await.map_err(|e| e.to_string())?
        }
        (None, Some(ids)) if !ids.is_empty() => {
            let ph = ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
            let sql = format!(
                "{EV_SELECT}
                  WHERE m.id IN (SELECT event_id FROM face_embeddings WHERE id IN ({ph}) AND event_id IS NOT NULL)
                  ORDER BY m.started_at DESC LIMIT {limit}"
            );
            let mut q = sqlx::query_as(&sql);
            for id in ids { q = q.bind(id); }
            q.fetch_all(&state.db).await.map_err(|e| e.to_string())?
        }
        _ => return Err("get_person_events needs person_id or face_ids".into()),
    };

    Ok(rows.into_iter().map(|(id, started_at, ended_at, duration_secs, peak_score, clip_path, thumbnail, detections, ai_summary, cam_id, event_category, recognized_plate, dominant_label, sub_label, first_object_at)| {
        let person_crop = crop_by_event.get(&id)
            .map(|t| crate::blobstore::resolve(&state.data_dir, t));
        PersonEvent {
            event: crate::MotionEvent {
                id, started_at, ended_at, duration_secs, peak_score: peak_score as f32,
                clip_path, thumbnail, detections, ai_summary,
                cam_id: cam_id.map(|c| c as u8),
                event_category, recognized_plate, dominant_label, sub_label, first_object_at,
                top_speed_kmh: None,
            },
            person_crop,
        }
    }).collect())
}

/// A recognised vehicle (by licence plate) with its sighting summary — the Vehicles
/// section's card, mirroring a person: plate, optional friendly name (from
/// `settings.known_plates` "PLATE=Name"), how often/where/when seen, a thumbnail, and
/// the event ids to review the clips.
#[derive(Debug, Serialize)]
pub struct Vehicle {
    pub plate:      String,
    pub name:       Option<String>,
    pub count:      i64,
    pub cameras:    Vec<i64>,
    pub first_seen: String,
    pub last_seen:  String,
    pub thumbnail:  String,        // most-recent event thumbnail (bare base64)
    pub event_ids:  Vec<String>,   // for the per-vehicle clip/sighting view
}

/// List detected vehicles (ALPR): every recognised plate in-window, grouped, with
/// its friendly name (if in `known_plates`), sighting count, cameras (locations),
/// first/last seen, a thumbnail, and event ids. Most-seen first.
#[tauri::command]
pub async fn list_vehicles(
    state: State<'_, Arc<AppState>>,
    days: Option<i64>,
) -> Result<Vec<Vehicle>, String> {
    let days = days.unwrap_or(30).clamp(1, 365);
    let rows: Vec<(String, Option<i64>, String, Option<String>, String)> = sqlx::query_as(
        "SELECT recognized_plate, cam_id, started_at, thumbnail, id FROM motion_events
          WHERE recognized_plate IS NOT NULL AND recognized_plate <> ''
            AND started_at > datetime('now', ?)
          ORDER BY started_at DESC"
    ).bind(format!("-{days} days")).fetch_all(&state.db).await.map_err(|e| e.to_string())?;

    let known = state.settings.read().await.known_plates.clone();
    let name_of = |plate: &str| -> Option<String> {
        for line in known.lines() {
            if let Some((p, n)) = line.split_once('=') {
                if p.trim().eq_ignore_ascii_case(plate.trim()) {
                    let n = n.trim();
                    if !n.is_empty() { return Some(n.to_string()); }
                }
            }
        }
        None
    };

    let mut map: std::collections::HashMap<String, Vehicle> = std::collections::HashMap::new();
    for (plate, cam, started, thumb, eid) in rows {
        let cam = cam.unwrap_or(0);
        let e = map.entry(plate.clone()).or_insert_with(|| Vehicle {
            plate: plate.clone(), name: name_of(&plate), count: 0, cameras: Vec::new(),
            first_seen: started.clone(), last_seen: started.clone(),
            thumbnail: String::new(), event_ids: Vec::new(),
        });
        e.count += 1;
        if !e.cameras.contains(&cam) { e.cameras.push(cam); }
        if started < e.first_seen { e.first_seen = started.clone(); }
        if started > e.last_seen  { e.last_seen  = started.clone(); }
        if e.thumbnail.is_empty() { // rows are newest-first → first crop = most recent
            e.thumbnail = crate::blobstore::resolve(&state.data_dir, &thumb.unwrap_or_default());
        }
        if e.event_ids.len() < 50 { e.event_ids.push(eid); }
    }
    let mut out: Vec<Vehicle> = map.into_values().collect();
    for v in &mut out { v.cameras.sort_unstable(); }
    out.sort_by(|a, b| b.count.cmp(&a.count).then(b.last_seen.cmp(&a.last_seen)));
    Ok(out)
}

/// Per-person activity pattern for the People roster — turns raw face sightings
/// into the answers a user actually wants: how often, which cameras, WHEN. This is
/// the "self-learned" picture of each person's routine (edge-AI-style entity
/// knowledge, surfaced instead of buried in rows).
#[derive(Debug, Serialize)]
pub struct PersonStats {
    /// The identity this row belongs to. `None` only for pre-migration sightings
    /// that were written before `face_sightings.person_id` existed and could not
    /// be resolved back to a roster entry by name.
    pub person_id:     Option<String>,
    pub name:          String,
    pub sightings_30d: i64,
    pub days_active:   i64,
    /// Local hour-of-day this person is most often seen (0-23), None if too few data.
    pub peak_hour:     Option<u8>,
    /// Sightings per local hour of day, 0..23.
    ///
    /// The histogram was always computed here and then thrown away, with only its
    /// argmax (`peak_hour`) crossing the IPC boundary — so the UI could say "most
    /// often around 18:00" but never draw the shape. "Home every evening" and
    /// "here once at 18:00 and never again" produced the same single number.
    pub hours:         [u32; 24],
    pub cameras:       Vec<i64>,
    pub last_seen:     Option<String>,
}

/// Fold raw sightings into per-identity stats.
///
/// Keyed by `person_id`, NOT by display name. This grouped by `person_name`, so
/// two people called "Alex" merged into a single row and each was shown the
/// other's sightings, peak hour and camera list — the name-binding this codebase
/// forbids everywhere else. The frontend had grown a workaround that suppressed
/// stats entirely for any shared name; that workaround exists only because of
/// this function and goes away with it.
///
/// `person_id` is NULL on rows written before the column existed (`db.rs:412`),
/// so those resolve their name against the roster first. Only a name matching no
/// roster entry at all falls back to being keyed by name — that keeps genuinely
/// old history visible instead of silently dropping it.
///
/// Pure and separate from the command so the id/NULL/unknown-name split is
/// testable without a database.
fn aggregate_person_stats(
    rows: Vec<(Option<String>, String, String, i64)>,
    roster: &[(String, String)],
) -> Vec<PersonStats> {
    use std::collections::{HashMap, HashSet};
    struct Acc { id: Option<String>, name: String, n: i64, days: HashSet<String>,
                 hours: [u32; 24], cams: HashSet<i64>, last: Option<String> }
    let mut by: HashMap<String, Acc> = HashMap::new();
    for (pid, name, seen, cam) in rows {
        let resolved = pid.filter(|v| !v.is_empty()).or_else(|| roster.iter()
            .find(|(_, n)| n.eq_ignore_ascii_case(&name))
            .map(|(id, _)| id.clone()));
        let key = resolved.clone().unwrap_or_else(|| format!("name:{}", name.to_lowercase()));
        let a = by.entry(key).or_insert(Acc {
            id: resolved, name: name.clone(), n: 0, days: HashSet::new(),
            hours: [0; 24], cams: HashSet::new(), last: None });
        a.n += 1;
        a.days.insert(seen.chars().take(10).collect());
        a.cams.insert(cam);
        if a.last.is_none() { a.last = Some(seen.clone()); } // rows are newest-first
        let local = chrono::DateTime::parse_from_rfc3339(&seen)
            .map(|d| d.with_timezone(&chrono::Local))
            .or_else(|_| chrono::NaiveDateTime::parse_from_str(&seen, "%Y-%m-%d %H:%M:%S")
                .map(|n| n.and_utc().with_timezone(&chrono::Local)));
        if let Ok(t) = local { a.hours[chrono::Timelike::hour(&t) as usize] += 1; }
    }
    by.into_values().map(|a| {
        let peak = if a.n >= 3 {
            a.hours.iter().enumerate().max_by_key(|(_, c)| **c).map(|(h, _)| h as u8)
        } else { None };
        let mut cams: Vec<i64> = a.cams.into_iter().collect();
        cams.sort_unstable();
        PersonStats {
            person_id: a.id, name: a.name, sightings_30d: a.n,
            days_active: a.days.len() as i64,
            peak_hour: peak, hours: a.hours, cameras: cams, last_seen: a.last,
        }
    }).collect()
}

/// Aggregate sighting patterns for every enrolled person (last 30 days).
#[tauri::command]
pub async fn get_person_stats(
    state: State<'_, Arc<AppState>>,
) -> Result<Vec<PersonStats>, String> {
    let rows: Vec<(Option<String>, String, String, i64)> = sqlx::query_as(
        "SELECT person_id, person_name, seen_at, camera_id FROM face_sightings
          WHERE seen_at > datetime('now','-30 days')
          ORDER BY seen_at DESC LIMIT 5000"
    ).fetch_all(&state.db).await.map_err(|e| e.to_string())?;

    let roster: Vec<(String, String)> = sqlx::query_as("SELECT id, name FROM known_persons")
        .fetch_all(&state.db).await.unwrap_or_default();

    let mut out = aggregate_person_stats(rows, &roster);
    out.sort_by_key(|p| std::cmp::Reverse(p.sightings_30d));
    Ok(out)
}

/// Per-sound pattern summary for the Audio tab (last 7 days): what's being heard,
/// how often, and when — the difference between a raw event list and INSIGHT.
#[derive(Debug, Serialize)]
pub struct AudioStats {
    pub sound:      String,
    pub count_7d:   i64,
    pub last_heard: String,
    /// Local hour-of-day this sound peaks (0-23), None if too few events.
    pub peak_hour:  Option<u8>,
}

#[tauri::command]
pub async fn get_audio_stats(
    state: State<'_, Arc<AppState>>,
) -> Result<Vec<AudioStats>, String> {
    let rows: Vec<(Option<String>, String)> = sqlx::query_as(
        "SELECT dominant_label, started_at FROM motion_events
          WHERE event_category='audio' AND started_at > datetime('now','-7 days')
          ORDER BY started_at DESC LIMIT 2000"
    ).fetch_all(&state.db).await.map_err(|e| e.to_string())?;

    use std::collections::HashMap;
    struct Acc { n: i64, hours: [u32; 24], last: Option<String> }
    let mut by: HashMap<String, Acc> = HashMap::new();
    for (sound, at) in rows {
        let key = sound.unwrap_or_else(|| "sound".into());
        let a = by.entry(key).or_insert(Acc { n: 0, hours: [0; 24], last: None });
        a.n += 1;
        if a.last.is_none() { a.last = Some(at.clone()); }
        if let Ok(t) = chrono::DateTime::parse_from_rfc3339(&at).map(|d| d.with_timezone(&chrono::Local)) {
            a.hours[chrono::Timelike::hour(&t) as usize] += 1;
        }
    }
    let mut out: Vec<AudioStats> = by.into_iter().map(|(sound, a)| AudioStats {
        peak_hour: if a.n >= 3 { a.hours.iter().enumerate().max_by_key(|(_, c)| **c).map(|(h, _)| h as u8) } else { None },
        last_heard: a.last.unwrap_or_default(),
        count_7d: a.n,
        sound,
    }).collect();
    out.sort_by_key(|p| std::cmp::Reverse(p.count_7d));
    Ok(out)
}

/// One detected audio event — a sustained sound (dog bark, speech, alarm, glass…)
/// heard on a camera's mic. Mirrors an event row: what sound, where (camera), when,
/// how long, and how confident. The Audio section renders these as a recent feed.
/// One vehicle sighting — a motion event categorised 'vehicle', with the
/// attribute enrichment (type, body color, plate, owner, speed).
#[derive(Debug, Serialize)]
pub struct VehicleEvent {
    pub id:            String,
    pub vtype:         String,          // dominant_label: car/truck/bus/motorcycle/bicycle
    pub cam_id:        Option<i64>,
    pub started_at:    String,
    pub ended_at:      Option<String>,
    pub duration_secs: Option<f64>,
    pub score:         f32,
    pub thumbnail:     Option<String>,  // '@thumb' marker
    pub plate:         Option<String>,
    pub plate_score:   Option<f64>,
    pub owner:         Option<String>,  // known-plate friendly name (sub_label)
    pub color:         Option<String>,  // HSV-voted body color
    pub speed_kmh:     Option<f64>,
}

/// Whitelist a CSV of values into an `AND col IN (…)` clause (multi-select
/// dropdowns send "car,truck"). Non-whitelisted entries are dropped, so the
/// interpolation stays injection-free.
fn csv_in_clause(raw: Option<&str>, allowed: &[&str], col: &str) -> String {
    let vals: Vec<&str> = raw.unwrap_or("").split(',')
        .filter(|v| allowed.contains(v)).collect();
    if vals.is_empty() { return String::new(); }
    let list = vals.iter().map(|v| format!("'{v}'")).collect::<Vec<_>>().join(",");
    format!(" AND {col} IN ({list})")
}

/// Multi-select camera filter: CSV of integer cam ids → `AND cam_id IN (…)`.
fn cams_in_clause(cams: Option<&str>) -> String {
    let ids: Vec<i64> = cams.unwrap_or("").split(',')
        .filter_map(|s| s.trim().parse().ok()).collect();
    if ids.is_empty() { return String::new(); }
    let list = ids.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(",");
    format!(" AND cam_id IN ({list})")
}

/// List vehicle sightings, keyset-paged, with server-side type/color/plate/
/// camera filters (whitelisted clauses) + optional day window (`from`/`to` —
/// the Review toolbar's date applies here too).
#[tauri::command]
pub async fn list_vehicle_events(
    state: State<'_, Arc<AppState>>,
    days:  Option<i64>,
    before: Option<String>,
    limit: Option<i64>,
    vtype: Option<String>,   // CSV: "car,truck"
    color: Option<String>,   // CSV
    plate: Option<String>,   // CSV of with/known/unknown, OR-joined
    cams:  Option<String>,   // CSV of cam ids
    from:  Option<String>,   // UTC ISO day bounds
    to:    Option<String>,
) -> Result<Vec<VehicleEvent>, String> {
    let days  = days.unwrap_or(365).clamp(1, 3650);
    let limit = limit.unwrap_or(60).clamp(10, 100);
    let before = before.filter(|b| !b.is_empty()).unwrap_or_else(|| "9999".to_string());

    const TYPES:  [&str; 5]  = ["car", "truck", "bus", "motorcycle", "bicycle"];
    const COLORS: [&str; 11] = ["black","white","silver","gray","red","orange","yellow","green","blue","purple","brown"];
    let mut clauses = csv_in_clause(vtype.as_deref(), &TYPES, "dominant_label");
    clauses.push_str(&csv_in_clause(color.as_deref(), &COLORS, "vehicle_color"));
    clauses.push_str(&cams_in_clause(cams.as_deref()));
    // Plate filter: multi-select OR-joins the fixed sub-clauses.
    let plate_ors: Vec<&str> = plate.as_deref().unwrap_or("").split(',').filter_map(|p| match p {
        "with"    => Some("(recognized_plate IS NOT NULL AND recognized_plate != '')"),
        "known"   => Some("(recognized_plate IS NOT NULL AND recognized_plate != '' AND sub_label IS NOT NULL AND sub_label != recognized_plate)"),
        "unknown" => Some("(recognized_plate IS NOT NULL AND recognized_plate != '' AND (sub_label IS NULL OR sub_label = recognized_plate))"),
        _ => None,
    }).collect();
    if !plate_ors.is_empty() { clauses.push_str(&format!(" AND ({})", plate_ors.join(" OR "))); }
    let from_clause = if from.is_some() { " AND started_at >= ?" } else { "" };
    let to_clause   = if to.is_some()   { " AND started_at < ?"  } else { "" };
    let sql = format!(
        "SELECT id, dominant_label, cam_id, started_at, ended_at, duration_secs, peak_score,
                CASE WHEN thumbnail IS NOT NULL AND thumbnail != '' THEN '@thumb' ELSE NULL END,
                recognized_plate, plate_score, sub_label, vehicle_color, top_speed_kmh
           FROM motion_events
          WHERE event_category='vehicle' AND started_at > datetime('now', ?)
            AND started_at < ? {from_clause} {to_clause} {clauses}
          ORDER BY started_at DESC
          LIMIT ?");
    let mut q = sqlx::query_as(&sql).bind(format!("-{days} days")).bind(&before);
    if let Some(f) = &from { q = q.bind(f); }
    if let Some(t) = &to   { q = q.bind(t); }
    let rows: Vec<(String, Option<String>, Option<i64>, String, Option<String>, Option<f64>, f64, Option<String>, Option<String>, Option<f64>, Option<String>, Option<String>, Option<f64>)> =
        q.bind(limit)
            .fetch_all(&state.db).await.map_err(|e| e.to_string())?;
    Ok(rows.into_iter().map(|(id, dl, cam_id, started_at, ended_at, duration_secs, score, thumbnail, plate, plate_score, sub_label, color, speed)| {
        // sub_label is the OWNER name only when it differs from the raw plate.
        let owner = match (&sub_label, &plate) {
            (Some(sl), Some(pl)) if sl != pl => Some(sl.clone()),
            (Some(sl), None) => Some(sl.clone()),
            _ => None,
        };
        VehicleEvent {
            id,
            vtype: dl.unwrap_or_else(|| "vehicle".into()),
            cam_id, started_at, ended_at, duration_secs,
            score: score as f32,
            thumbnail, plate, plate_score, owner, color,
            speed_kmh: speed,
        }
    }).collect())
}

#[derive(Debug, Serialize)]
pub struct AudioClass { pub l: String, pub s: f32 }

#[derive(Debug, Serialize)]
pub struct AudioEvent {
    pub id:            String,
    pub sound:         String,          // dominant_label (the recognised sound)
    pub cam_id:        Option<i64>,
    pub started_at:    String,
    pub ended_at:      Option<String>,
    pub duration_secs: Option<f64>,
    pub score:         f32,             // peak_score = detection confidence 0..1
    pub ai_summary:    Option<String>,
    /// '@thumb' marker when a stored frame exists (served via /footage/:id/thumbnail).
    pub thumbnail:     Option<String>,
    pub loudness_db:   Option<f64>,
    /// YAMNet semantic flag: any top class in the high-pitch/alert set.
    pub high_pitch:    bool,
    /// Top YAMNet classes of the loudest window (label + confidence).
    pub classes:       Vec<AudioClass>,
}

/// List recent AUDIO events (YAMNet sound detection): rows in `motion_events` tagged
/// `event_category='audio'`, newest first. Bounded by `days` + a hard row cap so the
/// feed payload stays small. The dedicated Audio section reads this.
#[tauri::command]
pub async fn list_audio_events(
    state: State<'_, Arc<AppState>>,
    days:  Option<i64>,
    before: Option<String>,
    limit: Option<i64>,
    category: Option<String>, // CSV of categories, OR-joined ("human,alarm")
    cams:  Option<String>,    // CSV of cam ids
    from:  Option<String>,    // UTC ISO day bounds
    to:    Option<String>,
) -> Result<Vec<AudioEvent>, String> {
    // KEYSET pagination (O(log n) via the started_at index — never OFFSET):
    // `before` = the oldest loaded row's started_at; each page continues from it.
    let days  = days.unwrap_or(365).clamp(1, 3650);
    let limit = limit.unwrap_or(60).clamp(10, 100);
    let before = before.filter(|b| !b.is_empty()).unwrap_or_else(|| "9999".to_string());
    // Server-side category filter so every view pages a DENSE stream. Patterns
    // are compile-time constants (no injection surface). Multi-select = union.
    let mut cat_ors: Vec<String> = Vec::new();
    for cat in category.as_deref().unwrap_or("").split(',').filter(|c| !c.is_empty() && *c != "all") {
        if cat == "high_pitch" {
            cat_ors.push("json_extract(audio_meta, '$.high_pitch') = 1".to_string());
        } else if let Some(pats) = crate::audio::category_label_patterns(cat) {
            cat_ors.extend(pats.iter().map(|p| format!("LOWER(dominant_label) LIKE '%{p}%'")));
        }
    }
    let mut cat_clause = if cat_ors.is_empty() { String::new() }
        else { format!("AND ({})", cat_ors.join(" OR ")) };
    cat_clause.push_str(&cams_in_clause(cams.as_deref()));
    let from_clause = if from.is_some() { " AND started_at >= ?" } else { "" };
    let to_clause   = if to.is_some()   { " AND started_at < ?"  } else { "" };
    let sql = format!(
        "SELECT id, dominant_label, cam_id, started_at, ended_at, duration_secs, peak_score, ai_summary,
                CASE WHEN thumbnail IS NOT NULL AND thumbnail != '' THEN '@thumb' ELSE NULL END,
                loudness_db, COALESCE(audio_meta, json_extract(attributes, '$.audio'))
           FROM motion_events
          WHERE event_category='audio' AND started_at > datetime('now', ?)
            AND started_at < ? {from_clause} {to_clause} {cat_clause}
          ORDER BY started_at DESC
          LIMIT ?");
    let mut q = sqlx::query_as(&sql).bind(format!("-{days} days")).bind(&before);
    if let Some(f) = &from { q = q.bind(f); }
    if let Some(t) = &to   { q = q.bind(t); }
    let rows: Vec<(String, Option<String>, Option<i64>, String, Option<String>, Option<f64>, f64, Option<String>, Option<String>, Option<f64>, Option<String>)> =
        q.bind(limit)
            .fetch_all(&state.db).await.map_err(|e| e.to_string())?;

    Ok(rows.into_iter().map(|(id, sound, cam_id, started_at, ended_at, duration_secs, score, ai_summary, thumbnail, loudness_db, attributes)| {
        // Parse the YAMNet enrichment (audio_meta; tolerate the legacy wrapped shape).
        let (high_pitch, classes) = attributes.as_deref()
            .and_then(|a| serde_json::from_str::<serde_json::Value>(a).ok())
            .map(|v| v.get("audio").cloned().unwrap_or(v))
            .map(|a| {
                let hp = a.get("high_pitch").and_then(|b| b.as_bool()).unwrap_or(false);
                let cls = a.get("classes").and_then(|c| c.as_array()).map(|arr| {
                    arr.iter().filter_map(|e| Some(AudioClass {
                        l: e.get("l")?.as_str()?.to_string(),
                        s: e.get("s")?.as_f64()? as f32,
                    })).collect::<Vec<_>>()
                }).unwrap_or_default();
                (hp, cls)
            }).unwrap_or((false, Vec::new()));
        AudioEvent {
            id,
            sound: sound.unwrap_or_else(|| "sound".to_string()),
            cam_id, started_at, ended_at, duration_secs,
            score: score as f32,
            ai_summary,
            thumbnail, loudness_db, high_pitch, classes,
        }
    }).collect())
}

/// Assign many unrecognised faces (a whole cluster) to one person at once:
/// append each embedding to the person + tag the rows. Caps the person's stored
/// embeddings at 30 (most recent) so a big cluster can't bloat matching.
#[tauri::command]
pub async fn assign_faces_to_person(
    state: State<'_, Arc<AppState>>,
    face_ids:  Vec<String>,
    person_id: String,
) -> Result<(), String> {
    if face_ids.is_empty() { return Ok(()); }
    let (embs_json,): (String,) = sqlx::query_as(
        "SELECT embeddings FROM known_persons WHERE id=?"
    ).bind(&person_id).fetch_one(&state.db).await
        .map_err(|e| format!("person not found: {e}"))?;
    let mut embs: Vec<Vec<f32>> = serde_json::from_str(&embs_json).unwrap_or_default();

    let q_floor = state.settings.read().await.face_quality_floor;
    for fid in &face_ids {
        let row: Option<(Vec<u8>, i64, f64)> = sqlx::query_as(
            "SELECT descriptor, dim, quality FROM face_embeddings WHERE id=? AND person_id IS NULL"
        ).bind(fid).fetch_optional(&state.db).await.map_err(|e| e.to_string())?;
        let Some((descriptor, dim, quality)) = row else { continue };
        if descriptor.len() != dim as usize * 4 { continue; }
        let v: Vec<f32> = descriptor.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        // Shared append policy (diversity gate + newest-30 cap) behind the
        // QUALITY GATE: blurry crops tag their row but never become matcher input.
        if quality as f32 >= q_floor {
            push_enrolled_shot(&mut embs, v);
        } else {
            tracing::info!("assign_faces: tagged but not enrolled (quality {quality:.2} < floor {q_floor:.2})");
        }
        sqlx::query("UPDATE face_embeddings SET person_id=? WHERE id=?")
            .bind(&person_id).bind(fid).execute(&state.db).await.map_err(|e| e.to_string())?;
    }
    let updated = serde_json::to_string(&embs).map_err(|e| e.to_string())?;
    sqlx::query("UPDATE known_persons SET embeddings=?, last_seen_at=datetime('now') WHERE id=?")
        .bind(&updated).bind(&person_id).execute(&state.db).await.map_err(|e| e.to_string())?;
    schedule_classifier_retrain(&state.db);
    Ok(())
}

/// Create a brand-new known person from an unmatched face. Hands off to the
/// existing `known_persons` schema — same as `enroll_person`, but seeded from
/// a stored embedding instead of one captured live in the browser.
#[tauri::command]
pub async fn create_person_from_face(
    state: State<'_, Arc<AppState>>,
    face_id: String,
    name:    String,
    role:    String,
) -> Result<KnownPerson, String> {
    let (descriptor, dim, thumb): (Vec<u8>, i64, Option<String>) = sqlx::query_as(
        "SELECT descriptor, dim, thumbnail_b64 FROM face_embeddings WHERE id=? AND person_id IS NULL"
    ).bind(&face_id).fetch_one(&state.db).await
        .map_err(|e| format!("face not found or already linked: {e}"))?;
    // Resolve the crop to real base64 before it becomes the person's enrolled
    // thumbnail (known_persons.thumbnail stays inline, not a file ref).
    let thumb = thumb.map(|t| crate::blobstore::resolve(&state.data_dir, &t));

    if descriptor.len() != (dim as usize) * 4 {
        return Err("descriptor blob size mismatch".into());
    }
    let vec: Vec<f32> = descriptor.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let id  = Uuid::new_v4().to_string();
    let now = Utc::now().to_rfc3339();
    let embeddings = serde_json::to_string(&vec![vec]).map_err(|e| e.to_string())?;

    sqlx::query(
        "INSERT INTO known_persons(id, name, role, embeddings, thumbnail, created_at, last_seen_at)
         VALUES(?,?,?,?,?,?,?)"
    )
    .bind(&id).bind(&name).bind(&role).bind(&embeddings)
    .bind(&thumb).bind(&now).bind(&now)
    .execute(&state.db).await.map_err(|e| e.to_string())?;

    sqlx::query("UPDATE face_embeddings SET person_id=? WHERE id=?")
        .bind(&id).bind(&face_id)
        .execute(&state.db).await.map_err(|e| e.to_string())?;

    schedule_classifier_retrain(&state.db);
    Ok(KnownPerson {
        id,
        name,
        role,
        embeddings,
        embedding_count: 1,
        thumbnail: thumb,
        created_at: now.clone(),
        last_seen_at: Some(now),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two people, one name. They must NOT merge.
    ///
    /// `get_person_stats` grouped by `person_name`, so a household with two
    /// people called "Alex" showed each of them the other's sightings, peak hour
    /// and camera list. It was invisible because the numbers looked plausible.
    /// The frontend's answer was to hide stats for any duplicated name, which
    /// traded a wrong answer for no answer; keying by id gives both of them a
    /// right one.
    #[test]
    fn same_name_two_people_do_not_merge() {
        let roster = vec![("p1".to_string(), "Alex".to_string()),
                          ("p2".to_string(), "Alex".to_string())];
        let rows = vec![
            (Some("p1".into()), "Alex".into(), "2026-08-20T09:00:00Z".into(), 0i64),
            (Some("p1".into()), "Alex".into(), "2026-08-21T09:00:00Z".into(), 0),
            (Some("p2".into()), "Alex".into(), "2026-08-21T22:00:00Z".into(), 3),
        ];
        let out = aggregate_person_stats(rows, &roster);
        assert_eq!(out.len(), 2, "two ids must stay two rows");
        let p1 = out.iter().find(|s| s.person_id.as_deref() == Some("p1")).unwrap();
        let p2 = out.iter().find(|s| s.person_id.as_deref() == Some("p2")).unwrap();
        assert_eq!(p1.sightings_30d, 2);
        assert_eq!(p2.sightings_30d, 1);
        assert_eq!(p1.cameras, vec![0], "p1 was never on camera 3");
        assert_eq!(p2.cameras, vec![3]);
    }

    /// Rows predating the `person_id` migration (`db.rs:412`) carry NULL, and
    /// must still land on the right identity by resolving their name against the
    /// roster — otherwise an id-only query silently drops all history older than
    /// the migration while the UI reports a confident zero.
    #[test]
    fn pre_migration_rows_resolve_by_name() {
        let roster = vec![("p1".to_string(), "Dana".to_string())];
        let rows = vec![
            (None,                "Dana".into(), "2026-08-20T09:00:00Z".into(), 1i64),
            (Some("p1".into()),   "Dana".into(), "2026-08-21T09:00:00Z".into(), 1),
            // Someone who left the roster: keyed by name so the history is still
            // visible rather than vanishing.
            (None,                "Ghost".into(), "2026-08-21T10:00:00Z".into(), 2),
        ];
        let out = aggregate_person_stats(rows, &roster);
        let dana = out.iter().find(|s| s.person_id.as_deref() == Some("p1")).unwrap();
        assert_eq!(dana.sightings_30d, 2, "the NULL row must fold into the id");
        let ghost = out.iter().find(|s| s.person_id.is_none()).unwrap();
        assert_eq!(ghost.name, "Ghost");
    }

    /// The hour histogram is the point of shipping `hours` at all: `peak_hour`
    /// alone cannot tell "home every evening" from "here once at 18:00".
    #[test]
    fn the_hour_histogram_survives_serialisation() {
        let roster = vec![("p1".to_string(), "Sam".to_string())];
        let rows: Vec<_> = (0..4)
            .map(|i| (Some("p1".to_string()), "Sam".to_string(),
                      format!("2026-08-2{i}T18:30:00Z"), 0i64))
            .collect();
        let out = aggregate_person_stats(rows, &roster);
        let s = &out[0];
        assert_eq!(s.hours.iter().sum::<u32>(), 4, "every sighting lands in a bucket");
        let peak = s.peak_hour.expect("4 sightings clears the >=3 floor") as usize;
        assert_eq!(s.hours[peak], 4, "peak_hour must be the argmax of hours");
    }

    /// Erasure must actually erase. Every DELETE in `erase_biometrics` names a
    /// column by hand, and a wrong one fails at runtime while the UI still says
    /// "erased" — the exact failure mode that makes a compliance claim false.
    /// (It caught one: `face_sightings` is keyed by NAME — it does also carry a
    /// `person_id` from a later migration, but the erasure path deletes by name
    /// because pre-migration rows have that column NULL.)
    #[tokio::test]
    async fn erasure_leaves_nothing_behind() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        crate::db::init_db(&pool).await.unwrap();

        // Two people, so we also prove erasure is surgical: Bob must survive.
        for (id, name) in [("p1", "Alice"), ("p2", "Bob")] {
            sqlx::query("INSERT INTO known_persons(id, name, embeddings, created_at)
                         VALUES(?,?,'[]',datetime('now'))")
                .bind(id).bind(name).execute(&pool).await.unwrap();
            sqlx::query("INSERT INTO face_embeddings(id, person_id, descriptor, dim) VALUES(?,?,x'00',1)")
                .bind(format!("f_{id}")).bind(id).execute(&pool).await.unwrap();
            sqlx::query("INSERT INTO face_negatives(id, person_id, descriptor, dim) VALUES(?,?,x'00',1)")
                .bind(format!("n_{id}")).bind(id).execute(&pool).await.unwrap();
            sqlx::query("INSERT INTO body_embeddings(id, person_id, known_person_id, descriptor)
                         VALUES(?,?,?,x'00')")
                .bind(format!("b_{id}")).bind(format!("body_{id}")).bind(id)
                .execute(&pool).await.unwrap();
            sqlx::query("INSERT INTO body_negatives(id, known_person_id, descriptor, dim)
                         VALUES(?,?,x'00',1)")
                .bind(format!("bn_{id}")).bind(id).execute(&pool).await.unwrap();
            sqlx::query("INSERT INTO face_sightings(id, person_name) VALUES(?,?)")
                .bind(format!("s_{id}")).bind(name).execute(&pool).await.unwrap();
        }
        // A past event identified as Alice — the name must stop appearing there
        // too, or the identification outlives the erasure everywhere it is read.
        sqlx::query("INSERT INTO motion_events(id, started_at, sub_label) VALUES('e1', datetime('now'), 'Alice')")
            .execute(&pool).await.unwrap();

        let n = erase_biometrics(&pool, std::path::Path::new("."), "p1").await;
        assert!(n >= 5, "expected every Alice row counted, got {n}");

        for (table, col, val) in [
            ("face_embeddings", "person_id",       "p1"),
            ("face_negatives",  "person_id",       "p1"),
            ("body_embeddings", "known_person_id", "p1"),
            ("body_negatives",  "known_person_id", "p1"),
            ("known_persons",   "id",              "p1"),
            ("face_sightings",  "person_name",     "Alice"),
        ] {
            let left: i64 = sqlx::query_scalar(&format!("SELECT COUNT(*) FROM {table} WHERE {col}=?"))
                .bind(val).fetch_one(&pool).await
                .unwrap_or_else(|e| panic!("{table}.{col} — schema drift: {e}"));
            assert_eq!(left, 0, "{table} still holds Alice");

            // Surgical: the same query shape must still find Bob.
            let bob = if col == "person_name" { "Bob" } else { "p2" };
            let kept: i64 = sqlx::query_scalar(&format!("SELECT COUNT(*) FROM {table} WHERE {col}=?"))
                .bind(bob).fetch_one(&pool).await.unwrap();
            assert_eq!(kept, 1, "{table} lost Bob");
        }

        let label: Option<String> = sqlx::query_scalar("SELECT sub_label FROM motion_events WHERE id='e1'")
            .fetch_one(&pool).await.unwrap();
        assert_eq!(label, None, "the name survived on a past event");
    }
}
