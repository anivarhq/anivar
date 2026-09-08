//! On-disk blob store for DISPLAY images (face crops + context frames), so they
//! don't bloat the relational DB. On this box those two face columns were 462 MB
//! of a 533 MB database and caused a 3.7 s slow query.
//!
//! **No schema change.** A column holds EITHER legacy inline base64 OR a compact
//! `@file:<rel>` reference; `resolve` returns bare base64 for both, so the frontend
//! contract is unchanged. Files store the *decoded* JPEG (≈25% smaller than base64).
//!
//! **Hard-rule-safe.** This relocates DISPLAY images only — never the recognition
//! `descriptor`. A lost file degrades to a missing thumbnail; recognition is
//! unaffected. `store` falls back to returning the original base64 on ANY failure,
//! so an image is never lost by trying to offload it.

use std::path::{Path, PathBuf};

use base64::Engine as _;

const PREFIX: &str = "@file:";

fn root(data_dir: &Path) -> PathBuf { data_dir.join("blobs") }

/// True if `value` is a file reference rather than inline base64.
pub fn is_ref(value: &str) -> bool { value.starts_with(PREFIX) }

/// Offload bare-base64 JPEG `b64` to `blobs/<subdir>/<id><suffix>.jpg` and return
/// its `@file:` reference. Returns `b64` unchanged if it's empty, already a ref, or
/// on any decode/IO failure (never loses the image).
pub fn store(data_dir: &Path, subdir: &str, id: &str, suffix: &str, b64: &str) -> String {
    if b64.is_empty() || is_ref(b64) { return b64.to_string(); }
    let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(b64) else {
        return b64.to_string();
    };
    let rel = format!("{subdir}/{id}{suffix}.jpg");
    let path = root(data_dir).join(&rel);
    if let Some(parent) = path.parent() {
        if std::fs::create_dir_all(parent).is_err() { return b64.to_string(); }
    }
    if std::fs::write(&path, &bytes).is_err() { return b64.to_string(); }
    format!("{PREFIX}{rel}")
}

/// Resolve a stored value to **bare base64** (the frontend adds the `data:` prefix,
/// matching how these columns were always stored). File refs are read from disk;
/// inline base64 is returned as-is. A missing file yields "" (recognition
/// unaffected).
pub fn resolve(data_dir: &Path, value: &str) -> String {
    match value.strip_prefix(PREFIX) {
        Some(rel) => match std::fs::read(root(data_dir).join(rel)) {
            Ok(bytes) => base64::engine::general_purpose::STANDARD.encode(&bytes),
            Err(_) => String::new(),
        },
        None => value.to_string(),
    }
}

/// Resolve a stored value to raw JPEG **bytes** (for HTTP endpoints — skips the
/// base64 round-trip `resolve` does). File refs read from disk; inline base64
/// decoded. Empty vec when missing/undecodable.
pub fn resolve_bytes(data_dir: &Path, value: &str) -> Vec<u8> {
    match value.strip_prefix(PREFIX) {
        Some(rel) => std::fs::read(root(data_dir).join(rel)).unwrap_or_default(),
        None => base64::engine::general_purpose::STANDARD.decode(value.trim()).unwrap_or_default(),
    }
}

/// Delete the backing file for a ref (no-op for inline values) — for row deletion.
pub fn delete(data_dir: &Path, value: &str) {
    if let Some(rel) = value.strip_prefix(PREFIX) {
        let _ = std::fs::remove_file(root(data_dir).join(rel));
    }
}

/// Safety-net sweep: delete `faces/` blob files that NO `face_embeddings` row
/// references (orphaned by a missed delete path or a crash mid-delete). Conservative
/// — skips files modified in the last hour so it can never race a just-written crop
/// whose row INSERT is still in flight. Runs the file scan off the async executor.
pub async fn sweep_orphans(db: &sqlx::SqlitePool, data_dir: &Path) {
    // Build the referenced-set from the DB (async), then hand the file walk to a
    // blocking thread (a full `faces/` dir can hold tens of thousands of files).
    let rows: Vec<(Option<String>, Option<String>)> = sqlx::query_as(
        "SELECT thumbnail_b64, context_b64 FROM face_embeddings
          WHERE thumbnail_b64 LIKE '@file:%' OR context_b64 LIKE '@file:%'"
    ).fetch_all(db).await.unwrap_or_default();
    let mut referenced = std::collections::HashSet::new();
    for (t, c) in &rows {
        for v in [t, c].into_iter().flatten() {
            if let Some(rel) = v.strip_prefix(PREFIX) { referenced.insert(rel.to_string()); }
        }
    }
    // SAFETY: never mass-delete. An empty set means either no offloaded crops or a
    // transient query failure — in both cases do nothing rather than risk wiping
    // every blob. (Deliberate deletions are handled by `delete` at the call sites.)
    if referenced.is_empty() { return; }
    let dir = root(data_dir).join("faces");
    let removed = tokio::task::spawn_blocking(move || {
        let Ok(entries) = std::fs::read_dir(&dir) else { return 0u64 };
        let cutoff = std::time::SystemTime::now() - std::time::Duration::from_secs(3600);
        let mut removed = 0u64;
        for e in entries.flatten() {
            let Some(fname) = e.file_name().to_str().map(|s| s.to_string()) else { continue };
            if referenced.contains(&format!("faces/{fname}")) { continue; }
            // Age guard — never delete a crop written in the last hour (INSERT may
            // still be in flight, or a concurrent capture is mid-write).
            if let Ok(m) = e.metadata() {
                if m.modified().map(|t| t > cutoff).unwrap_or(true) { continue; }
            }
            if std::fs::remove_file(e.path()).is_ok() { removed += 1; }
        }
        removed
    }).await.unwrap_or(0);
    if removed > 0 { tracing::info!("blobstore: swept {removed} orphaned blob file(s)"); }
}

/// One-time, **idempotent** migration: relocate existing inline face crops +
/// context frames to files. Runs as a gentle background boot task. Cursor-paged by
/// id (always terminates), touches ONLY the display columns (never `descriptor`/
/// `person_id` — hard-rule-safe), writes the file before updating the row, and
/// falls back to leaving a crop inline if its offload fails. Fast-exits on every
/// later boot once nothing inline remains, then VACUUMs to reclaim the freed space.
pub async fn migrate_face_thumbnails(db: &sqlx::SqlitePool, data_dir: &Path) {
    let pending: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM face_embeddings
          WHERE (thumbnail_b64 IS NOT NULL AND thumbnail_b64 NOT LIKE '@file:%')
             OR (context_b64   IS NOT NULL AND context_b64   NOT LIKE '@file:%')"
    ).fetch_one(db).await.unwrap_or(0);
    if pending == 0 { return; }
    tracing::info!("blobstore: offloading {pending} inline face crop(s) to disk…");

    let mut last_id = String::new();
    let mut moved: u64 = 0;
    loop {
        let rows: Vec<(String, Option<String>, Option<String>)> = sqlx::query_as(
            "SELECT id, thumbnail_b64, context_b64 FROM face_embeddings
              WHERE id > ? ORDER BY id LIMIT 200"
        ).bind(&last_id).fetch_all(db).await.unwrap_or_default();
        if rows.is_empty() { break; }
        for (id, thumb, ctx) in &rows {
            last_id = id.clone();
            let nt = thumb.as_ref().filter(|t| !t.is_empty() && !is_ref(t))
                .map(|t| store(data_dir, "faces", id, "_t", t));
            let nc = ctx.as_ref().filter(|c| !c.is_empty() && !is_ref(c))
                .map(|c| store(data_dir, "faces", id, "_c", c));
            if nt.is_some() || nc.is_some() {
                let _ = sqlx::query(
                    "UPDATE face_embeddings
                        SET thumbnail_b64 = COALESCE(?, thumbnail_b64),
                            context_b64   = COALESCE(?, context_b64)
                      WHERE id = ?"
                ).bind(nt.as_deref()).bind(nc.as_deref()).bind(id).execute(db).await;
                moved += 1;
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(30)).await; // don't hog the DB
    }
    if moved > 0 {
        tracing::info!("blobstore: offloaded {moved} face crop(s); running VACUUM to reclaim disk…");
        let _ = sqlx::query("VACUUM").execute(db).await;
        tracing::info!("blobstore: VACUUM complete — database compacted");
    }
}
