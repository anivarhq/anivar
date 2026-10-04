//! In-app updates, through tauri-plugin-updater.
//!
//! The plugin was registered for months and never called: "Check for Updates"
//! read the GitHub API and handed the user a download link, and that link was
//! an `<a target="_blank">`, which the webview drops (no new-window handler),
//! so in practice nothing happened. This drives the plugin itself:
//!
//!   check   → `latest.json` from the newest GitHub release (tauri.conf.json
//!             `plugins.updater.endpoints`)
//!   download→ the platform's installer, verified against the minisign key in
//!             `plugins.updater.pubkey` before a byte of it is used
//!   install → Windows: the NSIS installer runs in passive mode, the plugin
//!             exits this process, and the installer relaunches the app with
//!             the same arguments (so `--headless` survives). macOS/Linux: the
//!             bundle is replaced in place and we restart.
//!
//! ffmpeg/go2rtc children die with the process through the Job Object, the
//! same as any other exit. The installer must not: it is let out of the job
//! just before it starts (`proc::let_next_children_outlive_us`).

use crate::state::AppState;
use serde_json::json;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tauri::{AppHandle, Emitter};
use tauri_plugin_updater::{Update, UpdaterExt};

/// One install at a time: a click on "Install and restart" while the
/// automatic path is mid-download would otherwise fetch the installer twice
/// and race two installers against each other.
static INSTALLING: AtomicBool = AtomicBool::new(false);

fn check_err(e: impl std::fmt::Display) -> String {
    format!("Couldn't check for updates: {e}")
}

async fn find_update(app: &AppHandle) -> Result<Option<Update>, String> {
    app.updater_builder()
        .on_before_exit(|| {
            // Said out loud for the same reason the tray's Quit is: otherwise an
            // update and a crash leave the same trace, a log that just stops.
            tracing::info!("update: handing over to the installer — exiting");
            #[cfg(windows)]
            crate::proc::let_next_children_outlive_us();
        })
        .build()
        .map_err(check_err)?
        .check()
        .await
        .map_err(check_err)
}

/// Settings → Updates → "Check for Updates".
#[tauri::command]
pub async fn update_check(app: AppHandle) -> Result<serde_json::Value, String> {
    let current = app.package_info().version.to_string();
    Ok(match find_update(&app).await? {
        Some(u) => json!({
            "available": true,
            "current":   current,
            "latest":    u.version,
            "notes":     u.body.clone().unwrap_or_default(),
            "date":      u.date.map(|d| d.date().to_string()),
        }),
        None => json!({ "available": false, "current": current }),
    })
}

/// Settings → Updates → "Install and restart". On success this never returns:
/// the process exits into the installer (Windows) or restarts (macOS/Linux).
#[tauri::command]
pub async fn update_install(app: AppHandle) -> Result<(), String> {
    if INSTALLING.swap(true, Ordering::SeqCst) {
        return Err("An update is already being installed".into());
    }
    let result = async {
        let update = find_update(&app).await?
            .ok_or_else(|| "Already on the latest version".to_string())?;
        let bytes = download(&app, &update).await?;
        install(&app, &update, bytes)
    }.await;
    INSTALLING.store(false, Ordering::SeqCst);
    result
}

/// Downloads and verifies the installer, reporting progress to the UI as
/// `update:progress {downloaded, total}` — at most every 512 KB, not per
/// chunk, so a 60 MB installer is ~120 events rather than thousands.
async fn download(app: &AppHandle, update: &Update) -> Result<Vec<u8>, String> {
    tracing::info!("update: downloading {} (from {})", update.version, update.current_version);
    let (mut got, mut sent) = (0u64, 0u64);
    let bytes = update.download(
        |chunk, total| {
            got += chunk as u64;
            if got - sent >= 512 * 1024 || Some(got) == total {
                sent = got;
                let _ = app.emit("update:progress", json!({ "downloaded": got, "total": total }));
            }
        },
        || {},
    ).await.map_err(|e| format!("Couldn't download the update: {e}"))?;
    tracing::info!("update: {} downloaded and its signature verified", update.version);
    Ok(bytes)
}

fn install(app: &AppHandle, update: &Update, bytes: Vec<u8>) -> Result<(), String> {
    tracing::info!("update: installing {}", update.version);
    let _ = app.emit("update:installing", json!({ "latest": update.version }));
    update.install(bytes).map_err(|e| format!("Couldn't install the update: {e}"))?;
    // Windows never gets here (the plugin exits into the installer). On macOS
    // and Linux the new version is on disk and only runs after a restart.
    app.restart()
}

/// True while any camera has an event open (`ended_at IS NULL` — see db.rs,
/// "in progress"). The automatic path never restarts through one of those:
/// a restart mid-event would cut off exactly the footage someone cares about.
async fn event_in_progress(db: &sqlx::SqlitePool) -> bool {
    sqlx::query_scalar::<_, i64>("SELECT EXISTS(SELECT 1 FROM motion_events WHERE ended_at IS NULL)")
        .fetch_one(db).await
        .map(|n| n != 0)
        .unwrap_or(true) // can't tell → assume busy; the wait below is bounded
}

/// Background check, 2 minutes after launch and every 6 hours after.
///
/// `auto_update_check` (default on): tell the UI an update exists.
/// `auto_update_install` (default off): also download it, wait for a quiet
/// moment (no event in progress, checked each minute, for up to 6 hours), and
/// install. The settings are re-read at every step, so turning either off
/// takes effect before the next network call or the install.
pub(crate) fn spawn_auto_update(state: Arc<AppState>) {
    tauri::async_runtime::spawn(async move {
        tokio::time::sleep(Duration::from_secs(120)).await;
        loop {
            auto_update_once(&state).await;
            tokio::time::sleep(Duration::from_secs(6 * 3600)).await;
        }
    });
}

async fn auto_update_once(state: &AppState) {
    let settings = || async { let s = state.settings.read().await; (s.auto_update_check, s.auto_update_install) };
    if !settings().await.0 { return; }
    let app = &state.app_handle;
    let update = match find_update(app).await {
        Ok(Some(u)) => u,
        Ok(None) => return,
        Err(e) => { tracing::warn!("update: {e}"); return; }
    };
    let auto = settings().await.1;
    let _ = app.emit("update:available", json!({ "latest": update.version, "auto": auto }));
    if !auto { return; }
    if INSTALLING.swap(true, Ordering::SeqCst) { return; }

    let bytes = match download(app, &update).await {
        Ok(b) => b,
        Err(e) => { tracing::warn!("update: {e}"); INSTALLING.store(false, Ordering::SeqCst); return; }
    };
    for _ in 0..360 {
        if !event_in_progress(&state.db).await { break; }
        tokio::time::sleep(Duration::from_secs(60)).await;
    }
    if !settings().await.1 {
        tracing::info!("update: automatic install switched off while waiting — not installing");
        INSTALLING.store(false, Ordering::SeqCst);
        return;
    }
    if let Err(e) = install(app, &update, bytes) {
        tracing::warn!("update: {e}");
        INSTALLING.store(false, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::event_in_progress;

    #[tokio::test]
    async fn install_waits_while_an_event_is_open() {
        let db = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        sqlx::query("CREATE TABLE motion_events (id INTEGER PRIMARY KEY, started_at TEXT, ended_at TEXT)")
            .execute(&db).await.unwrap();
        assert!(!event_in_progress(&db).await, "no events → quiet");

        sqlx::query("INSERT INTO motion_events (started_at, ended_at) VALUES ('t0', 't1')")
            .execute(&db).await.unwrap();
        assert!(!event_in_progress(&db).await, "only finished events → quiet");

        sqlx::query("INSERT INTO motion_events (started_at, ended_at) VALUES ('t2', NULL)")
            .execute(&db).await.unwrap();
        assert!(event_in_progress(&db).await, "an open event → wait");

        db.close().await;
        assert!(event_in_progress(&db).await, "unreadable DB → assume busy");
    }
}
