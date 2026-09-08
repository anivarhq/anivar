//! Nightly online database backups via `VACUUM INTO` — SQLite's official
//! mechanism for a consistent, compacted copy taken WITHOUT blocking readers
//! or writers. The DB holds the face identities and full event history
//! (hard rule: face data must never be lost); before this, a single disk
//! fault or corruption event lost everything with no recovery path.
//!
//! Policy: first backup ~15 min after boot (so a fresh install is protected
//! the same day), then every 24 h. Keep the newest 3, verify each copy with
//! `PRAGMA integrity_check` before trusting it, log every outcome.

use std::sync::Arc;

use crate::AppState;

const KEEP: usize = 3;
const BACKUP_PREFIX: &str = "anivar-";
/// Every prefix used before the current one. Still listed and rotated so a rename
/// never strands the copies a user already has. Backups keep showing up in the
/// list and keep rotating correctly across renames.
const LEGACY_BACKUP_PREFIXES: &[&str] = &["nivar-", "anvil-", "securecam-"];

pub(crate) async fn run_backup_loop(state: Arc<AppState>) {
    tokio::time::sleep(std::time::Duration::from_secs(900)).await;
    let mut tick = tokio::time::interval(std::time::Duration::from_secs(24 * 3600));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        backup_once(&state).await;
        tick.tick().await;
    }
}

pub(crate) async fn backup_once(state: &Arc<AppState>) {
    let dir = state.data_dir.join("backups");
    if tokio::fs::create_dir_all(&dir).await.is_err() {
        tracing::warn!("backup: cannot create {}", dir.display());
        return;
    }
    let name = format!("{BACKUP_PREFIX}{}.db", chrono::Local::now().format("%Y%m%d"));
    let dest = dir.join(&name);
    // VACUUM INTO refuses to overwrite; a same-day rerun replaces the stale copy.
    let _ = tokio::fs::remove_file(&dest).await;

    let dest_sql = dest.to_string_lossy().replace('\'', "''");
    let started = std::time::Instant::now();
    if let Err(e) = sqlx::query(&format!("VACUUM INTO '{dest_sql}'")).execute(&state.db).await {
        tracing::warn!("backup: VACUUM INTO failed: {e}");
        return;
    }

    // Reclaim the LIVE database's WAL while we're in the nightly window.
    // VACUUM INTO only writes the copy — it never checkpoints the source, so
    // without this the -wal file sits at its burst high-water mark forever
    // (journal_size_limit only applies AT checkpoint time).
    match sqlx::query_scalar::<_, i64>("PRAGMA wal_checkpoint(TRUNCATE)")
        .fetch_one(&state.db).await
    {
        Ok(0) => tracing::info!("backup: WAL checkpoint-truncated"),
        Ok(_) => tracing::debug!("backup: WAL checkpoint deferred (writer busy) — next night retries"),
        Err(e) => tracing::debug!("backup: WAL checkpoint skipped: {e}"),
    }

    // Trust nothing: verify the COPY before counting it as a backup.
    match integrity_ok(&dest).await {
        Ok(true) => {
            let mb = tokio::fs::metadata(&dest).await.map(|m| m.len() / 1_048_576).unwrap_or(0);
            tracing::info!("backup: {} written + integrity ok ({mb} MB, {:.1}s)",
                name, started.elapsed().as_secs_f32());
        }
        Ok(false) => {
            tracing::warn!("backup: {} FAILED integrity_check — discarding", name);
            let _ = tokio::fs::remove_file(&dest).await;
            return;
        }
        Err(e) => {
            tracing::warn!("backup: could not verify {}: {e}", name);
            return;
        }
    }

    // Rotation: keep the newest KEEP by filename (they embed the date).
    if let Ok(mut rd) = tokio::fs::read_dir(&dir).await {
        // Backups from every past name still count. Sort by the embedded DATE, not
        // the filename: with several prefixes in play "anvil-20260810" sorts BEFORE
        // "securecam-20260809", and rotating on that order would delete the NEWER
        // backup and keep the stale one.
        let mut names: Vec<(String, String)> = Vec::new(); // (date, filename)
        while let Ok(Some(ent)) = rd.next_entry().await {
            let n = ent.file_name().to_string_lossy().to_string();
            let ours = n.starts_with(BACKUP_PREFIX)
                || LEGACY_BACKUP_PREFIXES.iter().any(|p| n.starts_with(p));
            if !(ours && n.ends_with(".db")) { continue; }
            let date = n.rsplit_once('-').map(|(_, d)| d.trim_end_matches(".db").to_string())
                        .unwrap_or_default();
            names.push((date, n));
        }
        names.sort();
        while names.len() > KEEP {
            let (_, victim) = names.remove(0);
            let _ = tokio::fs::remove_file(dir.join(&victim)).await;
            tracing::info!("backup: rotated out {victim}");
        }
    }
}

async fn integrity_ok(path: &std::path::Path) -> anyhow::Result<bool> {
    use sqlx::ConnectOptions as _;
    let mut conn = sqlx::sqlite::SqliteConnectOptions::new()
        .filename(path)
        .read_only(true)
        .connect().await?;
    let verdict: String = sqlx::query_scalar("PRAGMA integrity_check")
        .fetch_one(&mut conn).await?;
    Ok(verdict.eq_ignore_ascii_case("ok"))
}
