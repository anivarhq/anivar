//! Tauri app boot — the one-time `.setup(|app| ...)` work: system-tray menu, DB pool, settings load + decrypt, AppState construction, agent / inference / NVR / mDNS / HTTP-server task spawns.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use sqlx::sqlite::SqlitePoolOptions;
use tauri::{Emitter, Manager};
use tokio::sync::{broadcast, watch, Mutex, RwLock};

use crate::{
    AppState, CameraInventory,
    agent,
    crypto::{decrypt_settings_secrets, load_or_create_master_key},
    db::{init_db, load_or_create_auth_token, load_settings_from_db},
    ensure_ffmpeg,
    inference::run_inference_loop,
    nvr_pipes::{cleanup_orphaned_nvr_temps, prune_orphaned_nvr_segments, reindex_existing_nvr_segments, spawn_hls_pipe, spawn_nvr_pipe},
    start_http_server,
};

/// Kill orphaned app-spawned helper processes left by a FORCE-killed previous run.
///
/// When the app is killed ungracefully (Task Manager / `taskkill /F`), a child it
/// spawned can survive AND keep the HTTP/stream server's listen-socket handle it
/// inherited. That pins port 8882 under the dead parent's PID, so the new run
/// can't bind it and the stream server stays DOWN → BLANK camera (and dead clips
/// /snapshots). Every such child is spawned by us from our own data dir, so any
/// instance alive at boot is an orphan: kill it before we bind. Windows-only (the
/// inherited-handle quirk is Windows-specific).
#[cfg(windows)]
fn kill_orphaned_app_children() {
    let _ = crate::proc::std_cmd("powershell")
        .args([
            "-NoProfile", "-Command",
            // Both identifiers: a zombie spawned by the PRE-rename build still has
            // the old data-dir path on its command line, and it is exactly as
            // capable of pinning the stream port. Drop the legacy arm once no
            // pre-rename build can plausibly still be installed.
            "Get-CimInstance Win32_Process -Filter \"Name='ffmpeg.exe'\" | \
             Where-Object { $_.CommandLine -match 'com\\.(anivar\\.app|nivar\\.app|anvil\\.nvr|securecam\\.app)' -and \
               -not (Get-Process -Id $_.ParentProcessId -ErrorAction SilentlyContinue) } | \
             ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }",
            // ffmpeg (2nd statement above): a force-killed run leaves its RECORDER
            // ffmpeg alive — the dshow mic input keeps it running after the frame
            // pipe dies — so it encodes forever, holds .tmp.mp4 locks, and eats
            // CPU/disk (six accumulated zombies measurably crippled the machine).
            // Parent-dead filter protects a legitimately-running instance's children.
        ])
        .output();
}
#[cfg(not(windows))]
fn kill_orphaned_app_children() {}

/// Show the control-panel window, RECREATING it if it was destroyed (e.g. a
/// WebView2 renderer crash while the recording backend kept running under the
/// `RunEvent::ExitRequested` guard in `lib.rs`). Lets the appliance recover its UI
/// from the tray without a full restart. Mirrors the `main` window's tauri.conf.json.
pub(crate) fn open_control_panel(app: &tauri::AppHandle) {
    if let Some(w) = app.get_webview_window("main") {
        let _ = w.unminimize();
        let _ = w.show();
        let _ = w.set_focus();
        return;
    }
    match tauri::WebviewWindowBuilder::new(app, "main", tauri::WebviewUrl::App("index.html".into()))
        .title("Anivar")
        .inner_size(1280.0, 800.0)
        .min_inner_size(960.0, 600.0)
        .resizable(true)
        .decorations(false)
        .build()
    {
        Ok(w) => {
            let _ = w.show();
            let _ = w.set_focus();
            tracing::info!("Control-panel window recreated after a UI teardown");
        }
        Err(e) => tracing::error!("Failed to recreate control-panel window: {e}"),
    }
}

/// The app database, inside the data dir. `migrate_legacy_db` moves any earlier
/// name onto this one.
pub(crate) const DB_FILENAME: &str = "anivar.db";

/// Every data directory this app has ever used, NEWEST FIRST.
///
/// A list rather than a single constant because there have now been two renames
/// (SecureCam → Anvil NVR → Nivar → Anivar) and an install may sit on any. Newest
/// first matters: an upgrade from Anvil must find `com.anvil.nvr` before it looks
/// for the older SecureCam directory.
///
/// Add to the FRONT on the next rename. Do not remove entries — the only thing
/// that can still find an old install is its name.
const LEGACY_DATA_DIRS: &[&str] = &["com.nivar.app", "com.anvil.nvr", "com.securecam.app"];

/// Database stems this app has used, newest first. Paired with the same list in
/// `migrate_legacy_db`, which renames the `-wal`/`-shm` sidecars alongside.
const LEGACY_DB_STEMS: &[&str] = &["nivar", "anvil", "securecam"];

/// One-time move of an earlier data directory onto the current one.
///
/// All of them live under the same parent (`%APPDATA%` on Windows), so this is a
/// same-volume rename: atomic and instant even though a real install is several
/// GB of footage. Copying instead would stall boot for minutes and demand that
/// much free space again.
///
/// No-ops unless the destination is absent-or-empty *and* a legacy directory
/// exists, so it is safe on a fresh install and safe to run on every boot.
fn migrate_legacy_data_dir(new: &Path) {
    // Destination already holds REAL data → this install is already ours.
    //
    // `crash.log` does not count. The panic hook builds this path by hand and
    // creates the directory, so a panic anywhere in the first boot before this
    // point leaves a lone crash.log behind — and reading that as occupancy would
    // strand a multi-GB legacy install permanently, on every boot after.
    let occupied = new.read_dir().map(|it| {
        it.flatten().any(|e| e.file_name() != std::ffi::OsStr::new("crash.log"))
    }).unwrap_or(false);
    if occupied { return; }
    let Some(parent) = new.parent() else { return };
    for legacy in LEGACY_DATA_DIRS {
        let old = parent.join(legacy);
        if !old.is_dir() || old == new { continue; }
        // Windows MoveFileEx refuses an existing destination, even an empty one —
        // and refuses a non-empty one outright, so the stray log must go first.
        let _ = std::fs::remove_file(new.join("crash.log"));
        if new.exists() { let _ = std::fs::remove_dir(new); }
        match std::fs::rename(&old, new) {
            Ok(()) => tracing::info!("migrated data directory {} → {}", old.display(), new.display()),
            Err(e) => tracing::error!(
                "data migration from {} failed ({e}) — starting with an empty data dir", old.display()),
        }
        return;                     // one hop is enough: the newest match wins
    }
}

/// Rename an earlier database onto [`DB_FILENAME`], in place.
///
/// Runs on every boot and is idempotent, so a directory that moved but whose
/// database rename failed is repaired on the next launch instead of silently
/// starting on an empty database next to several GB of footage.
///
/// The `-wal`/`-shm` sidecars are renamed WITH the database: SQLite pairs them
/// by filename, and a dirty WAL holds committed transactions not yet
/// checkpointed into the `.db`. Moving the `.db` alone discards them.
fn migrate_legacy_db(dir: &Path) {
    if dir.join(DB_FILENAME).exists() { return; }       // already migrated
    let stem = DB_FILENAME.trim_end_matches(".db");
    for legacy in LEGACY_DB_STEMS {
        if !dir.join(format!("{legacy}.db")).exists() { continue; }
        for suffix in ["", "-wal", "-shm"] {
            let from = dir.join(format!("{legacy}.db{suffix}"));
            if !from.exists() { continue; }
            let to = dir.join(format!("{stem}.db{suffix}"));
            if let Err(e) = std::fs::rename(&from, &to) {
                tracing::error!("database migration {} → {} failed: {e}",
                    from.display(), to.display());
            }
        }
        tracing::info!("database migrated {legacy}.db → {DB_FILENAME}");
        return;
    }
}

/// One-time application setup invoked from `tauri::Builder::setup`.
pub(crate) fn setup_app(app: &mut tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    // System tray menu
    let quit   = tauri::menu::MenuItem::with_id(app, "quit",   "Quit Anivar", true, None::<&str>)?;
    let show   = tauri::menu::MenuItem::with_id(app, "show",   "Open Anivar", true, None::<&str>)?;
    let status = tauri::menu::MenuItem::with_id(app, "status", "Recording…",     false, None::<&str>)?;
    let menu   = tauri::menu::Menu::with_items(app, &[&status, &show, &quit])?;
    let tray_icon = app.default_window_icon().cloned();
    let mut tray_builder = tauri::tray::TrayIconBuilder::new()
        .menu(&menu)
        .tooltip("Anivar — Recording");
    if let Some(icon) = tray_icon { tray_builder = tray_builder.icon(icon); }
    let _tray = tray_builder
        .on_menu_event(|app, event| match event.id().as_ref() {
            "show" => open_control_panel(app),
            "quit" => std::process::exit(0),
            _ => {}
        })
        // Left-click the tray icon to open the control panel — the expected
        // behaviour for an always-on appliance running windowless in the tray.
        // Recreates the window if a prior UI teardown destroyed it.
        .on_tray_icon_event(|tray, event| {
            if let tauri::tray::TrayIconEvent::Click {
                button: tauri::tray::MouseButton::Left,
                button_state: tauri::tray::MouseButtonState::Up, ..
            } = event {
                open_control_panel(tray.app_handle());
            }
        })
        .build(app)?;

    // ── Headless / appliance mode ────────────────────────────────────────────
    // When launched with `--headless` (e.g. autostart at boot on a dedicated NVR
    // box) we HIDE the window and run as a background service — just the tray +
    // the full recording/detection/agent backend. Normal launches keep the
    // window (config default visible) so the GUI can never fail to appear.
    let headless = std::env::args().any(|a| a == "--headless");
    if let Some(w) = app.get_webview_window("main") {
        if headless {
            w.hide().ok();
            tracing::info!("Starting in HEADLESS mode — running as a background NVR service (tray only)");
        } else {
            w.show().ok(); w.set_focus().ok();
        }
    }

    let data_dir = app.path().app_data_dir().unwrap_or_else(|_| PathBuf::from("."));
    // BEFORE create_dir_all: the rename below needs the destination not to exist
    // (Windows MoveFileEx refuses an existing directory, even an empty one).
    migrate_legacy_data_dir(&data_dir);
    std::fs::create_dir_all(&data_dir).ok();
    // Everything worth reading after the fact (EP canary, pack provisioning,
    // capture failures) is logged from here on — point the file sink at the data
    // dir before any of it runs.
    crate::logfile::set_dir(&data_dir);
    tracing::info!("Anivar {} starting — logs at {}",
        env!("CARGO_PKG_VERSION"), data_dir.join("logs").display());

    // NOTHING IS BUNDLED, BY DESIGN. There used to be a first-run seed here that
    // copied an ffmpeg binary and a default detector out of the installer's resource
    // dir. Both were removed deliberately: shipping them means DISTRIBUTING them, and
    // the detector was an Ultralytics YOLO (AGPL-3.0) while the ffmpeg build was GPL.
    // Downloading them on the user's instruction keeps that relationship between the
    // user and the upstream author, and keeps this Apache-2.0 app clear of both.
    //
    // ffmpeg therefore arrives via `ensure_ffmpeg` (provision.rs) on first use, and a
    // detector only ever exists because the user chose one in Arsenal. See
    // THIRD-PARTY-NOTICES.md.

    // Wipe the live-HLS dir SYNCHRONOUSLY, before the webview exists: HLS is
    // ephemeral view state, but `append_list` makes a new capture continue the
    // previous session's playlist (six-figure media sequence, dead .ts refs) —
    // the live player then error/re-attach loops (whole-UI flicker). This must
    // run before the frontend can race a capture start (the async NVR task's
    // wipe was too late — the UI's start_dshow_camera beat it by seconds).
    if let Ok(rd) = std::fs::read_dir(data_dir.join("hls")) {
        for ent in rd.flatten() { let _ = std::fs::remove_file(ent.path()); }
    }

    migrate_legacy_db(&data_dir);
    let db_url = format!("sqlite://{}?mode=rwc", data_dir.join(DB_FILENAME).display());
    // One broadcast channel per camera slot — up to 16 cameras for home/professional use
    const NUM_CAM_SLOTS: usize = 16;
    let frame_txs: Arc<Vec<broadcast::Sender<Arc<Vec<u8>>>>> = Arc::new(
        (0..NUM_CAM_SLOTS).map(|_| broadcast::channel::<Arc<Vec<u8>>>(16).0).collect()
    );
    let frame_txs_for_server = Arc::clone(&frame_txs);
    let (camera_state_tx, _) = broadcast::channel::<bool>(4);
    let camera_state_tx_clone = camera_state_tx.clone();

    let pool = tauri::async_runtime::block_on(async {
        // 4 connections starved the async runtime: the frontend loads every panel at
        // once (many concurrent reads) on top of the post-processor, backfill jobs and
        // agent — with only 4 slots, tasks pile up on `acquire()`, the tokio workers
        // stall, and DB-dependent work WEDGES (segment indexing + the NVR watchdog stop,
        // People/list commands time out → "nothing works") while the memory-served live
        // feed keeps working. WAL already allows concurrent readers, so give the pool
        // real headroom; `acquire_timeout` makes a starved acquire ERROR (freeing the
        // worker) instead of hanging the runtime forever.
        // 32 (was 16): the 9-camera soak showed pool acquires stretching to 8 s
        // under sustained multi-cam event load (every reader/writer path per cam
        // multiplies). WAL takes concurrent readers happily; more slots keeps
        // acquire latency flat at high camera counts. Nothing failed at 16 —
        // this is headroom, not a fix.
        //
        // Per-connection pragmas MUST be set here, on the connect options — the
        // old one-shot `PRAGMA …` in init_db reached exactly ONE pooled
        // connection; the other 31 ran at SQLite defaults (verified live:
        // synchronous=FULL — a full fsync per commit, forfeiting the WAL win —
        // and a 2 MB page cache). sqlx applies ConnectOptions to EVERY
        // connection it opens. (busy_timeout needs no line: sqlx defaults it
        // to 5 s per connection.)
        use std::str::FromStr as _;
        let opts = sqlx::sqlite::SqliteConnectOptions::from_str(&db_url)
            .expect("DB url parse failed")
            .journal_mode(sqlx::sqlite::SqliteJournalMode::Wal)
            .synchronous(sqlx::sqlite::SqliteSynchronous::Normal)
            // 8 MB page cache per connection. Was 32 MB — × max_connections(32)
            // that was a 1 GB commit ceiling that never shrank once a burst
            // opened the full pool. 8 MB/conn (256 MB ceiling) still covers the
            // hot working set; the OS file cache backs the rest.
            .pragma("cache_size", "-8192")
            // Cap the -wal file's on-disk footprint: after a checkpoint, SQLite
            // truncates the WAL back to this size instead of leaving it at its
            // burst high-water mark forever (multi-cam write bursts grow it).
            .pragma("journal_size_limit", "8388608"); // 8 MB
        SqlitePoolOptions::new()
            .max_connections(32)
            .min_connections(2)
            // Reclaim burst connections (and their page caches) after a quiet
            // minute — without this, one 32-connection burst committed its full
            // page-cache footprint for the life of the process.
            .idle_timeout(std::time::Duration::from_secs(60))
            .acquire_timeout(std::time::Duration::from_secs(20))
            .connect_with(opts).await.expect("DB connect failed")
    });

    let (mut settings, raw_token) = tauri::async_runtime::block_on(async {
        // Permanent proof-in-logs that per-connection pragmas apply (the old
        // one-shot PRAGMA reached 1 of 32 connections; the rest ran FULL/2MB).
        let sync_mode: i64 = sqlx::query_scalar("PRAGMA synchronous")
            .fetch_one(&pool).await.unwrap_or(-1);
        let cache: i64 = sqlx::query_scalar("PRAGMA cache_size")
            .fetch_one(&pool).await.unwrap_or(0);
        tracing::info!("db: per-connection pragmas synchronous={sync_mode} (1=NORMAL) cache_size={cache} (-32768=32MB)");
        init_db(&pool).await.expect("DB init failed");
        let s = load_settings_from_db(&pool).await;
        crate::depth::refresh_anon_cams(&s.depth_anonymize);
        crate::depth::set_depth_variant(&s.depth_model);

        // Periodic inference-latency digest: per-model avg/p95 every 5 min so
        // EP changes (DirectML vs TensorRT-RTX) are provable from logs alone.
        tauri::async_runtime::spawn(async {
            let mut iv = tokio::time::interval(std::time::Duration::from_secs(300));
            iv.tick().await;
            loop {
                iv.tick().await;
                let rows = crate::inference::infer_stats_snapshot();
                if rows.is_empty() { continue; }
                let line: Vec<String> = rows.iter()
                    .map(|r| format!("{}={:.1}ms(p95 {:.1}, n={})", r.model, r.avg_ms, r.p95_ms, r.count))
                    .collect();
                tracing::info!("infer stats [{}]: {}", crate::inference::active_accelerator(), line.join("  "));
            }
        });

        // NVIDIA Performance Pack (TensorRT-RTX): activate if previously installed.
        // Canary-verified; on any failure we stay on DirectML exactly as before.
        #[cfg(windows)]
        {
            let dd = data_dir.clone();
            // Blocking canary (builds a throwaway session builder) — cheap, but
            // keep it off the async path.
            tauri::async_runtime::spawn_blocking(move || {
                crate::trtx_runtime::activate_if_provisioned(&dd);
                // Record the full picture right after the canaries settle: which
                // lane won, and the concrete reason every other one didn't. This
                // line is the first thing to read when someone asks "is my GPU
                // being used?" — see logs/anivar.log.
                crate::system_cmds::log_accel_report(&dd);
            });
        }
        // Linux NVIDIA lane: fetch + dlopen the managed CUDA/cuDNN runtime, then
        // report. Linux has no DirectML, so this is the only route off the CPU.
        // Best-effort — a failure leaves the CPU provider working exactly as before.
        #[cfg(target_os = "linux")]
        {
            let dd = data_dir.clone();
            tauri::async_runtime::spawn(async move {
                if crate::inference::has_nvidia_adapter() {
                    if let Err(e) = crate::cuda_runtime::ensure_cuda_runtime(&dd).await {
                        tracing::warn!("CUDA runtime unavailable: {e} — inference stays on CPU");
                    }
                } else {
                    tracing::info!("No NVIDIA driver detected — inference runs on CPU");
                }
                crate::system_cmds::log_accel_report(&dd);
            });
        }
        #[cfg(all(not(windows), not(target_os = "linux")))]
        {
            let dd = data_dir.clone();
            tauri::async_runtime::spawn_blocking(move || crate::system_cmds::log_accel_report(&dd));
        }
        #[cfg(feature = "openvino")]
        {
            let dd = data_dir.clone();
            tauri::async_runtime::spawn(async move {
                if let Err(e) = crate::openvino_runtime::ensure_openvino_runtime(&dd).await {
                    tracing::warn!("OpenVINO runtime unavailable: {e} — falling back");
                }
            });
        }
        let t = load_or_create_auth_token(&pool).await;
        (s, t)
    });
    // Decrypt sensitive fields loaded from DB (encrypted by save_settings)
    let startup_key = load_or_create_master_key(&data_dir);
    decrypt_settings_secrets(&startup_key, &mut settings);
    let lan_access_for_server = settings.lan_access;

    // Apply the ONNX hardware-acceleration preference before any model loads.
    // On-device LLM: point the in-process engine at the tier the user picked.
    // Nothing is loaded here — the worker loads on first request and unloads when
    // idle. Re-applied on every settings save, so a tier switch needs no restart.
    crate::agent::local_llm::set_tier(&data_dir, &settings.local_llm_tier);
    crate::inference::set_gpu_inference(settings.inference_device != "cpu");

    // CUDA build variant: provision the NVIDIA runtime (download once on first
    // launch) and put it on the DLL path BEFORE any ORT session loads, so the CUDA
    // EP can bind cudart/cublas/cudnn. Best-effort — falls back to DirectML/CPU.
    //
    // Gated on an actual NVIDIA adapter, the way the Linux lane above already is.
    // Without that check EVERY machine fetched NVIDIA's redistributables, including
    // AMD and Intel ones that can never load them -- which is what forced this into
    // a separate "NVIDIA edition" installer rather than simply being what the one
    // installer does when it finds an NVIDIA card.
    #[cfg(feature = "cuda")]
    if settings.inference_device != "cpu" && crate::inference::has_nvidia_adapter() {
        let cuda_dd = data_dir.clone();
        if let Err(e) = tauri::async_runtime::block_on(crate::cuda_runtime::ensure_cuda_runtime(&cuda_dd)) {
            tracing::warn!("CUDA runtime setup failed ({e}) — using DirectML/CPU instead");
        }
    }

    let port = settings.stream_port;
    // Shared token: AppState and the HTTP server both hold a reference so
    // revoke_token() takes effect in the middleware immediately.
    let auth_token: Arc<RwLock<String>> = Arc::new(RwLock::new(raw_token));
    let auth_token_for_server = Arc::clone(&auth_token);

    // Revocation watch channel — sender stays in AppState, receiver in StreamState
    let (revoke_tx, revoke_rx) = watch::channel::<u64>(0);

    let app_handle = app.handle().clone();
    let app_handle_for_server = app_handle.clone();
    let camera_active = Arc::new(RwLock::new(false));
    let camera_active_for_server = Arc::clone(&camera_active);

    let pool_for_server = pool.clone();
    let data_dir_for_server = data_dir.clone();
    // Durable job queue (Apalis on the app's own SQLite pool). Created before the
    // AppState so the storage handle can live in it; `None` → callers fall back to
    // an in-process spawn (zero regression).
    let embed_jobs = match tauri::async_runtime::block_on(crate::jobs::make_storage(&pool)) {
        Ok(s) => Some(s),
        Err(e) => { tracing::warn!("jobs: durable queue unavailable ({e}); using in-process spawn"); None }
    };

    // Restore restart-surviving share links (see active_shares below).
    let (restored_shares, restored_generation) =
        tauri::async_runtime::block_on(crate::share_cmds::load_persisted_shares(&pool));
    if !restored_shares.is_empty() {
        tracing::info!("shares: restored {} unexpired share link(s) across restart", restored_shares.len());
    }

    let state = Arc::new(AppState {
        frame_txs,
        cam_states: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
        settings: RwLock::new(settings),
        recording_active: Mutex::new(false),
        clip_txs: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
        capture_handles: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
        db: pool,
        data_dir,
        auth_token,
        revoke_tx,
        camera_active,
        camera_state_tx,
        app_handle,
        agent_last_run: RwLock::new(None),
        latest_frames: Arc::new(RwLock::new(HashMap::new())),
        camera_inventory: Arc::new(RwLock::new(CameraInventory::default())),
        scene_objects: Arc::new(RwLock::new(HashMap::new())),
        scene_last_update: Arc::new(RwLock::new(HashMap::new())),
        behavior_events: Arc::new(RwLock::new(HashMap::new())),
        pending_escalations: Arc::new(RwLock::new(HashMap::new())),
        nvr_processes: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
        hls_processes: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
        rtsp_processes: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
        capture_keys:   Arc::new(tokio::sync::Mutex::new(HashMap::new())),
        stopped_usb_captures: Arc::new(tokio::sync::Mutex::new(Vec::new())),
        capture_start_lock: Arc::new(tokio::sync::Mutex::new(())),
        audio_processes: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
        nvr_pipe_txs:   Arc::new(tokio::sync::Mutex::new(HashMap::new())),
        hls_pipe_txs:   Arc::new(tokio::sync::Mutex::new(HashMap::new())),
        infer_queue: Arc::new(crate::state::InferQueue::new(NUM_CAM_SLOTS)),
        unknown_clusters_cache: tokio::sync::Mutex::new(None),
        last_shown: RwLock::new(None),
        latest_detections: Arc::new(RwLock::new(HashMap::new())),
        hw_encoder: Arc::new(std::sync::RwLock::new("libx264".to_string())),
        hw_decoder: Arc::new(std::sync::RwLock::new(String::new())),
        embed_jobs,
        master_key: startup_key,
        auth: crate::auth::AuthState::default(),
        client_sessions: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
        kick_txs: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
        // v9: shared inference-loop status — populated by run_inference_loop,
        // read by get_inference_status command so CameraView can pull-correct
        // its YOLO badge on mount instead of being stuck at "loading".
        inference_status: Arc::new(RwLock::new(crate::state::InferenceStatusSnapshot::default())),
        // v11 share infra: tunnel + outstanding shares + generation counter.
        // RESTORED from the DB: memory-only state meant every app restart broke
        // all outstanding links ("share not found or expired" on a link minted
        // minutes earlier) because the cookie gate reconstructs tokens from
        // this list. Timed shares + the generation now survive restarts;
        // expiry=0 ("until app restart") entries intentionally don't.
        active_shares:    Arc::new(RwLock::new(restored_shares)),
        share_generation: Arc::new(RwLock::new(restored_generation)),
    });

    app.manage(Arc::clone(&state));

    // Spawn Guardian agent loop
    let state_for_agent = Arc::clone(&state);
    tauri::async_runtime::spawn(async move {
        agent::run_agent_loop(state_for_agent).await;
    });

    // Spawn Telegram polling loop
    let state_for_tg = Arc::clone(&state);
    tauri::async_runtime::spawn(async move {
        agent::run_telegram_loop(state_for_tg).await;
    });

    // (The managed-Ollama watchdog lived here. It downloaded a server, spawned
    //  `ollama serve`, and restarted it every 60 s — which is how a 6 GB
    //  llama-server outlived the app on a 16 GB machine. The model now runs
    //  inside this process; there is no child to watch.)

    // Spawn Rust ONNX inference — runs on blocking thread pool (tract uses Rc, not Send)
    let state_for_infer = Arc::clone(&state);
    tauri::async_runtime::spawn(async move {
        // run_inference_loop contains blocking CPU work; use spawn_blocking
        // so we don't starve the tokio async executor
        let _ = tokio::task::spawn_blocking(move || {
            let rt = tokio::runtime::Handle::current();
            rt.block_on(run_inference_loop(state_for_infer));
        }).await;
    });

    // One-shot: backfill server-side review segments for recent events so the
    // Review feed + timeline have grouped items immediately on first run after
    // this feature ships. Guarded by a settings flag inside; near-free thereafter.
    {
        let state_rs = Arc::clone(&state);
        tauri::async_runtime::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_secs(8)).await;
            crate::review_segments::backfill_review_segments(&state_rs.db).await;
            // Rebuild the derived table from motion_events. The upsert can only
            // merge, never split, so runaway segments already on disk (a whole
            // busy period, or an audio event bridging two video events, as one
            // card) can only be fixed by re-deriving. Flag-guarded, runs once.
            crate::review_segments::rebuild_review_segments(&state_rs.db).await;
        });
    }

    // Auto-start NVR+HLS pipes when camera activates (done inside async spawn)
    {
        let state2 = Arc::clone(&state);
        tauri::async_runtime::spawn(async move {
            // Delay so HTTP server is ready and settings are loaded
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            let nvr_enabled = state2.settings.read().await.nvr_enabled;
            if !nvr_enabled { return; }
            let seg_mins = state2.settings.read().await.nvr_segment_mins;
            // Clean up any .tmp.mp4 files orphaned by the previous session
            let nvr_dir = state2.data_dir.join("nvr");
            cleanup_orphaned_nvr_temps(&nvr_dir).await;
            // (HLS dir is wiped synchronously at setup start — before the
            // frontend can race a capture spawn. Never wipe it here: a capture
            // may already be writing the fresh playlist.)
            // Sweep leaked `_concat_*.txt` manifests in the data dir (the streaming
            // path leaks one when the client aborts mid-stream). Harmless but they pile up.
            if let Ok(mut rd) = tokio::fs::read_dir(&state2.data_dir).await {
                while let Ok(Some(ent)) = rd.next_entry().await {
                    if let Some(n) = ent.file_name().to_str() {
                        if n.starts_with("_concat_") && n.ends_with(".txt") {
                            let _ = tokio::fs::remove_file(ent.path()).await;
                        }
                    }
                }
            }
            // Re-index any .mp4 files that exist on disk but aren't in the DB
            // (happens after app restart or DB clear)
            reindex_existing_nvr_segments(&nvr_dir, &state2.db).await;
            // Drop DB rows whose file is gone so the DB-only listing matches disk.
            prune_orphaned_nvr_segments(&state2.db).await;
            let enc = state2.hw_encoder.read().unwrap().clone();
            // Every ENABLED camera (all 16 slots) gets its server-side bring-up.
            // (This query was `cam_id < 4` — the single hard blocker that left
            // slots 4-15 with no recording/HLS/mic until the UI touched them.)
            // Fall back to cam0 (the USB slot) if nothing is configured yet.
            let rows: Vec<(i64, String, String, String)> = sqlx::query_as(
                "SELECT cam_id, source_type, source_url, device_id FROM camera_configs \
                 WHERE enabled=1 AND cam_id < 16 ORDER BY cam_id"
            ).fetch_all(&state2.db).await.unwrap_or_default();
            // NATIVE (USB) cams: the capture ffmpeg records DIRECTLY (one clock =
            // A/V sync) — boot starts it server-side so recording no longer waits
            // for the UI to open the Live tab. LEGACY cams (browser push / nokhwa
            // fallback) keep the pipe recorder + pre-registered mic. NETWORK cams
            // (rtsp/mjpeg/onvif) get their relay + copy-recorder started here too —
            // previously they recorded nothing until the UI opened the Live tab.
            let mut native:  Vec<(u8, String)> = Vec::new();
            let mut network: Vec<(u8, String)> = Vec::new();
            let mut legacy:  Vec<u8> = Vec::new();
            for (cam, ty, url, dev) in &rows {
                if ty == "native" {
                    let device = if !url.is_empty() { url.clone() } else { dev.clone() };
                    if device.is_empty() { legacy.push(*cam as u8); }
                    else { native.push((*cam as u8, device)); }
                } else if ty == "browser" {
                    legacy.push(*cam as u8);
                } else if !url.is_empty() {
                    // rtsp / mjpeg / onvif — anything with a stream URL
                    network.push((*cam as u8, url.clone()));
                }
            }
            // NOTE: no "empty ⇒ assume cam0" fallback. It used to push slot 0 here,
            // so an install whose cameras were all REMOVED still brought up a
            // recorder + mic tap for a camera that doesn't exist (device light on,
            // empty segments). No configured cameras ⇒ nothing starts.
            if let Ok(ff) = crate::ensure_ffmpeg(&state2.data_dir).await {
                if let Some(aargs) = crate::dshow::audio_input_args(&ff).await {
                    for cam_id in &legacy {
                        crate::nvr_pipes::set_nvr_mic(*cam_id, Some(aargs.clone()));
                    }
                    if !legacy.is_empty() {
                        tracing::info!("NVR: mic registered for legacy cams {legacy:?} before recorder start");
                    }
                }
            }
            tracing::info!("NVR boot: native(capture-records)={:?} network(relay+copy)={:?} legacy(pipe)={legacy:?}",
                native.iter().map(|(c, _)| *c).collect::<Vec<_>>(),
                network.iter().map(|(c, _)| *c).collect::<Vec<_>>());
            // Spawns are STAGGERED (400 ms apart) so 16 cameras can't thundering-
            // herd the GPU/disk/USB bus with simultaneous ffmpeg startups.
            let stagger = std::time::Duration::from_millis(400);
            for (cam_id, device) in &native {
                let st = state2.app_handle.state::<Arc<AppState>>();
                if let Err(e) = crate::dshow::start_usb_capture(st, *cam_id, device.clone()).await {
                    tracing::warn!("boot: usb capture cam{cam_id} failed to start: {e}");
                }
                tokio::time::sleep(stagger).await;
            }
            for (cam_id, url) in &network {
                let st = state2.app_handle.state::<Arc<AppState>>();
                if let Err(e) = crate::rtsp::start_rtsp_relay(st, *cam_id, url.clone()).await {
                    tracing::warn!("boot: network relay cam{cam_id} failed to start: {e}");
                }
                tokio::time::sleep(stagger).await;
            }
            for cam_id in &legacy {
                let cam_id = *cam_id;
                if let Ok((tx, child)) = spawn_nvr_pipe(cam_id, &state2.data_dir, seg_mins, &enc, state2.app_handle.clone(), state2.db.clone()).await {
                    state2.nvr_pipe_txs.lock().await.insert(cam_id, tx);
                    state2.nvr_processes.lock().await.insert(cam_id, child);
                }
                tokio::time::sleep(stagger).await;
            }
            // Legacy HLS encoders ONLY for cams not serving their own HLS:
            // native cams write HLS from the capture tee, network cams from the
            // copy-recorder. Browser/legacy cams (and the unconfigured cam0
            // fallback) still need the pipe encoder. Depth-anonymized cams get
            // theirs inside spawn_capture.
            let hls_targets: Vec<u8> = legacy.clone(); // (no cam0 fallback — see above)
            for cam_id in hls_targets {
                if state2.hls_pipe_txs.lock().await.contains_key(&cam_id) { continue; }
                // Keep the CHILD (was `_`): spawn_hls_pipe sets kill_on_drop(true), so
                // dropping it here killed the HLS ffmpeg immediately → /hls/ produced
                // nothing. Store it so the live HLS stream actually runs.
                if let Ok((tx, child)) = spawn_hls_pipe(cam_id, &state2.data_dir, &enc).await {
                    state2.hls_pipe_txs.lock().await.insert(cam_id, tx);
                    state2.hls_processes.lock().await.insert(cam_id, child);
                }
            }

            // ── Watchdog ──────────────────────────────────────────────────────
            // The NVR ffmpeg pipe can exit mid-session (frame starvation, crash),
            // which silently stops recording — observed today: it died at 08:28
            // and never came back. Poll every 30s; if a cam's child has exited,
            // respawn the pipe so recording self-heals.
            let state3 = Arc::clone(&state2);
            tauri::async_runtime::spawn(async move {
                let mut interval = tokio::time::interval(std::time::Duration::from_secs(30));
                interval.tick().await; // skip the immediate first tick
                // Consecutive hardware-encoder deaths per cam → downgrade to software.
                let mut death_counts: HashMap<u8, u8> = HashMap::new();
                // Stall-check grace: when the watchdog first sees a cam (or just
                // respawned it), MAX(started_at) in the DB is still the PREVIOUS
                // session/process's last segment — >75s stale by definition. The
                // old code stall-killed the healthy fresh capture over that
                // (corrupting its in-flight segment, twice per boot). No stall
                // verdicts until a cam has been observed 90s.
                let mut watch_since: HashMap<u8, std::time::Instant> = HashMap::new();
                loop {
                    interval.tick().await;

                    // ── USB-capture watchdog (dead OR recording-stalled) ──────
                    // The merged capture ffmpeg IS the recorder for native cams;
                    // it also feeds live view + detection, so healing it heals
                    // all three (previously a dead capture killed live/detection
                    // forever — only the pointless empty-pipe recorder respawned).
                    let nvr_on = state3.settings.read().await.nvr_enabled;
                    let usb_cams: Vec<(u8, String)> = state3.capture_keys.lock().await.iter()
                        .filter_map(|(c, k)| k.strip_prefix("usb:").map(|d| (*c, d.to_string())))
                        .collect();
                    for (cam, device) in usb_cams {
                        let dead = {
                            let mut procs = state3.rtsp_processes.lock().await;
                            match procs.get_mut(&cam) {
                                Some(child) => matches!(child.try_wait(), Ok(Some(_)) | Err(_)),
                                None => true,
                            }
                        };
                        let in_grace = {
                            let now = std::time::Instant::now();
                            watch_since.entry(cam).or_insert(now).elapsed().as_secs() < 90
                        };
                        let stalled = if nvr_on && !dead && !in_grace {
                            let last: Option<String> = sqlx::query_scalar(
                                "SELECT MAX(started_at) FROM nvr_segments WHERE cam_id=?"
                            ).bind(cam as i64).fetch_optional(&state3.db).await.ok().flatten();
                            last.as_deref()
                                .and_then(|t| chrono::DateTime::parse_from_rfc3339(t).ok())
                                .map(|dt| (chrono::Utc::now().timestamp() - dt.timestamp()) > 75)
                                .unwrap_or(false)
                        } else { false };
                        if !(dead || stalled) { continue; }
                        let enabled = sqlx::query_scalar::<_, i64>(
                            "SELECT enabled FROM camera_configs WHERE cam_id=?")
                            .bind(cam as i64).fetch_optional(&state3.db).await.ok().flatten()
                            .map(|e| e != 0).unwrap_or(true);
                        if !enabled { continue; }
                        tracing::warn!("capture watchdog: cam{cam} {} — respawning",
                            if dead { "process died" } else { "recording stalled >75s" });
                        if let Some(mut child) = state3.rtsp_processes.lock().await.remove(&cam) {
                            let _ = child.kill().await;
                        }
                        state3.capture_keys.lock().await.remove(&cam);
                        let deaths = { let c = death_counts.entry(cam).or_insert(0); *c += 1; *c };
                        if deaths >= 2 {
                            crate::nvr_pipes::force_software_encoder(cam);
                            tracing::warn!("capture watchdog: cam{cam} died {deaths}× — switching to CPU encoder");
                        }
                        let st = state3.app_handle.state::<Arc<AppState>>();
                        match crate::dshow::start_usb_capture(st, cam, device).await {
                            Ok(())  => tracing::info!("capture watchdog: cam{cam} respawned"),
                            Err(e)  => tracing::warn!("capture watchdog: cam{cam} respawn failed: {e}"),
                        }
                        watch_since.insert(cam, std::time::Instant::now());
                        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                    }

                    // ── RTSP/network watchdog (relay dead, recorder dead, or stalled) ──
                    // The copy-recorder used to be an untracked local — when it died,
                    // an RTSP cam silently stopped recording FOREVER. Now: relay dead,
                    // recorder dead, or no indexed segment >75s ⇒ full chain respawn
                    // (stop_rtsp_relay tears down relay+recorder+audio, then a fresh
                    // start rebuilds them).
                    let rtsp_cams: Vec<(u8, String)> = state3.capture_keys.lock().await.iter()
                        .filter_map(|(c, k)| k.strip_prefix("rtsp:").map(|u| (*c, u.to_string())))
                        .collect();
                    for (cam, url) in rtsp_cams {
                        let relay_dead = {
                            let mut procs = state3.rtsp_processes.lock().await;
                            match procs.get_mut(&cam) {
                                Some(child) => matches!(child.try_wait(), Ok(Some(_)) | Err(_)),
                                None => true,
                            }
                        };
                        let rec_dead = if nvr_on {
                            let mut procs = state3.nvr_processes.lock().await;
                            match procs.get_mut(&cam) {
                                Some(child) => matches!(child.try_wait(), Ok(Some(_)) | Err(_)),
                                None => true,
                            }
                        } else { false };
                        let in_grace = {
                            let now = std::time::Instant::now();
                            watch_since.entry(cam).or_insert(now).elapsed().as_secs() < 90
                        };
                        let stalled = if nvr_on && !rec_dead && !in_grace {
                            let last: Option<String> = sqlx::query_scalar(
                                "SELECT MAX(started_at) FROM nvr_segments WHERE cam_id=?"
                            ).bind(cam as i64).fetch_optional(&state3.db).await.ok().flatten();
                            last.as_deref()
                                .and_then(|t| chrono::DateTime::parse_from_rfc3339(t).ok())
                                .map(|dt| (chrono::Utc::now().timestamp() - dt.timestamp()) > 75)
                                .unwrap_or(false)
                        } else { false };
                        if !(relay_dead || rec_dead || stalled) { continue; }
                        let enabled = sqlx::query_scalar::<_, i64>(
                            "SELECT enabled FROM camera_configs WHERE cam_id=?")
                            .bind(cam as i64).fetch_optional(&state3.db).await.ok().flatten()
                            .map(|e| e != 0).unwrap_or(true);
                        if !enabled { continue; }
                        tracing::warn!("rtsp watchdog: cam{cam} {} — respawning chain",
                            if relay_dead { "relay died" } else if rec_dead { "recorder died" } else { "stalled >75s" });
                        let st = state3.app_handle.state::<Arc<AppState>>();
                        crate::rtsp::stop_rtsp_relay(st, cam).await.ok();
                        let st2 = state3.app_handle.state::<Arc<AppState>>();
                        match crate::rtsp::start_rtsp_relay(st2, cam, url).await {
                            Ok(())  => tracing::info!("rtsp watchdog: cam{cam} respawned"),
                            Err(e)  => tracing::warn!("rtsp watchdog: cam{cam} respawn failed: {e}"),
                        }
                        watch_since.insert(cam, std::time::Instant::now());
                        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                    }

                    if !state3.settings.read().await.nvr_enabled { continue; }
                    let seg_mins = state3.settings.read().await.nvr_segment_mins;
                    let enc = state3.hw_encoder.read().unwrap().clone();
                    // Which started cams have a dead child? RTSP-keyed cams are
                    // EXCLUDED — nvr_processes holds their copy-recorder, whose
                    // respawn path is the rtsp watchdog above (spawn_nvr_pipe
                    // would wrongly give them a JPEG pipe recorder).
                    let rtsp_keyed: std::collections::HashSet<u8> = state3.capture_keys.lock().await.iter()
                        .filter(|(_, k)| k.starts_with("rtsp:")).map(|(c, _)| *c).collect();
                    let mut dead: Vec<u8> = Vec::new();
                    {
                        let mut procs = state3.nvr_processes.lock().await;
                        for (cam, child) in procs.iter_mut() {
                            if rtsp_keyed.contains(cam) { continue; }
                            // try_wait()=Ok(Some(_)) → process has exited.
                            if matches!(child.try_wait(), Ok(Some(_)) | Err(_)) {
                                dead.push(*cam);
                            }
                        }
                        for cam in &dead { procs.remove(cam); }
                    }
                    for cam in dead {
                        state3.nvr_pipe_txs.lock().await.remove(&cam);
                        // A recorder that exited while muxing a mic almost certainly died
                        // BECAUSE of the mic (device contention/dropout on a shared laptop
                        // cam). Drop the mic so the respawn records VIDEO-ONLY and reliably;
                        // audio-in-recording must never break recording itself.
                        crate::nvr_pipes::set_nvr_mic(cam, None);
                        // After repeated deaths the HARDWARE ENCODER is the likely culprit
                        // (NVENC session-limit / GPU contention with pinned inference) —
                        // downgrade this cam to CPU (libx264), which always works.
                        let deaths = { let c = death_counts.entry(cam).or_insert(0); *c += 1; *c };
                        if deaths >= 2 {
                            crate::nvr_pipes::force_software_encoder(cam);
                            tracing::warn!("NVR watchdog: cam{} died {}× — switching to CPU encoder (libx264)", cam, deaths);
                        }
                        tracing::warn!("NVR watchdog: cam{} pipe died — respawning", cam);
                        if let Ok((tx, child)) = spawn_nvr_pipe(
                            cam, &state3.data_dir, seg_mins, &enc,
                            state3.app_handle.clone(), state3.db.clone()).await
                        {
                            state3.nvr_pipe_txs.lock().await.insert(cam, tx);
                            state3.nvr_processes.lock().await.insert(cam, child);
                            tracing::info!("NVR watchdog: cam{} pipe respawned", cam);
                        }
                    }

                    // STALL detection (the real fix for the "empty footage" gaps):
                    // an ffmpeg pipe can be ALIVE but wedged/starved — producing NO
                    // segments — which `try_wait()` above cannot see, so recording
                    // silently stops for minutes while inference keeps running. If a
                    // started cam hasn't INDEXED a segment in >75s, force-respawn the
                    // pipe so recording self-heals. Normal cadence is a segment every
                    // ~10s (≤~20s to index), so 75s is a safe 3× margin.
                    let started: Vec<u8> = state3.nvr_pipe_txs.lock().await.keys().copied().collect();
                    for cam in started {
                        // A cam disabled at runtime should stop being watched — kill its
                        // idle pipe and don't respawn (else we'd thrash a sourceless cam).
                        let enabled = sqlx::query_scalar::<_, i64>("SELECT enabled FROM camera_configs WHERE cam_id=?")
                            .bind(cam as i64).fetch_optional(&state3.db).await.ok().flatten()
                            .map(|e| e != 0).unwrap_or(true);
                        if !enabled {
                            if let Some(mut child) = state3.nvr_processes.lock().await.remove(&cam) { let _ = child.kill().await; }
                            state3.nvr_pipe_txs.lock().await.remove(&cam);
                            continue;
                        }
                        let last: Option<String> = sqlx::query_scalar(
                            "SELECT MAX(started_at) FROM nvr_segments WHERE cam_id=?"
                        ).bind(cam as i64).fetch_optional(&state3.db).await.ok().flatten();
                        // None = no segment yet (just-started cam) → don't thrash.
                        let stalled = last.as_deref()
                            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                            .map(|dt| (chrono::Utc::now().timestamp() - dt.timestamp()) > 75)
                            .unwrap_or(false);
                        if !stalled { continue; }
                        tracing::warn!("NVR watchdog: cam{} STALLED (no segment >75s) — force-respawning", cam);
                        if let Some(mut child) = state3.nvr_processes.lock().await.remove(&cam) {
                            let _ = child.kill().await;
                        }
                        state3.nvr_pipe_txs.lock().await.remove(&cam);
                        if let Ok((tx, child)) = spawn_nvr_pipe(
                            cam, &state3.data_dir, seg_mins, &enc,
                            state3.app_handle.clone(), state3.db.clone()).await
                        {
                            state3.nvr_pipe_txs.lock().await.insert(cam, tx);
                            state3.nvr_processes.lock().await.insert(cam, child);
                            tracing::info!("NVR watchdog: cam{} respawned after stall", cam);
                        }
                    }
                }
            });
        });
    }


    // Build the semantic-search ANN index (usearch) from stored embeddings so
    // search is O(log n) from the first query. Async — never blocks boot; search
    // falls back to brute-force until this finishes.
    {
        let state_vec = Arc::clone(&state);
        tauri::async_runtime::spawn(async move {
            crate::vector_index::build_events(&state_vec.db).await;
        });
    }

    // One-time: offload inline face crops/context frames to disk to slim the DB
    // (was 462MB of 533MB here) and kill the thumbnail slow-query. Idempotent +
    // gentle; fast-exits once done. Display images only — recognition untouched.
    {
        let state_bs = Arc::clone(&state);
        tauri::async_runtime::spawn(async move {
            crate::blobstore::migrate_face_thumbnails(&state_bs.db, &state_bs.data_dir).await;
        });
    }

    // Safety-net: periodically sweep orphaned face blob files (a missed delete path
    // or a crash mid-delete). First pass 30 min after boot (lets the migration
    // finish), then every 6 h. Conservative — never mass-deletes.
    {
        let state_sw = Arc::clone(&state);
        tauri::async_runtime::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_secs(1800)).await;
            let mut tick = tokio::time::interval(std::time::Duration::from_secs(6 * 3600));
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                tick.tick().await;
                crate::blobstore::sweep_orphans(&state_sw.db, &state_sw.data_dir).await;
            }
        });
    }

    // Nightly DB backup (VACUUM INTO + integrity check + keep-3 rotation). The DB
    // holds the face identities — the one dataset that must never be lost.
    {
        let state_bk = Arc::clone(&state);
        tauri::async_runtime::spawn(async move {
            crate::db_backup::run_backup_loop(state_bk).await;
        });
    }

    // Crash auto-restart: re-assert the keep-alive task at boot when enabled, so
    // it always points at the CURRENT exe path (installers can move it).
    {
        let state_ka = Arc::clone(&state);
        tauri::async_runtime::spawn(async move {
            if state_ka.settings.read().await.relaunch_after_crash {
                if let Err(e) = crate::system_cmds::set_keepalive_task(true).await {
                    tracing::warn!("keepalive task re-assert failed: {e}");
                }
            }
        });
    }

    // Footage retention: enforce `retention_days` — the one unbounded growth source
    // (~7 GB/day here). First pass 10 min after boot, then every 6 h. Video only —
    // never touches face data (hard rule).
    {
        let state_ret = Arc::clone(&state);
        tauri::async_runtime::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_secs(600)).await;
            let mut tick = tokio::time::interval(std::time::Duration::from_secs(6 * 3600));
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                tick.tick().await;
                // prune_old_footage also runs identity consolidation (keep recurring
                // "regulars", drop one-off strangers whose clip aged out) at its end —
                // coupled to actual footage deletion, so nothing prunes on a bare timer.
                crate::nvr_recording::prune_old_footage(&state_ret).await;
            }
        });
    }

    // Scrub previews (nvr_preview.rs) are NOT generated: nothing consumes them.
    //
    // They exist to make DRAGGING a scrubber cheap. The only scrubber that ever
    // dragged them was the in-video one, and that was deleted — the timeline is
    // the single scrub surface now (Frigate's shape), and dragging it pans the
    // viewport rather than seeking. Generating an hour of preview per camera per
    // hour to feed nothing is pure cost, so the loop is not spawned.
    //
    // The module, the `/nvr-preview` route and `list_previews` all still work.
    // Re-spawn this if the timeline grows a draggable handlebar; `prune_previews`
    // still runs from `prune_old_footage`, so anything already generated ages out.

    // Durable-job workers: process persisted embedding jobs + a self-healing
    // backfill that re-queues any event still missing its embedding. Both run for
    // the app's lifetime on background tasks (no-op if the queue is unavailable).
    if let Some(storage) = state.embed_jobs.clone() {
        let s_work = Arc::clone(&state);
        tauri::async_runtime::spawn(crate::jobs::run_workers(storage.clone(), s_work));
        let s_bf = Arc::clone(&state);
        tauri::async_runtime::spawn(crate::jobs::run_backfill_scheduler(storage, s_bf));
        tracing::info!("jobs: durable embedding queue online (Apalis/SQLite)");
    }

    // Detect hardware video encoder (runs async, updates state when done)
    {
        let state_enc = Arc::clone(&state);
        tauri::async_runtime::spawn(async move {
            if let Ok(ffmpeg) = ensure_ffmpeg(&state_enc.data_dir).await {
                let (enc, has_qsv) = crate::hw::detect_hw_encoders(&ffmpeg).await;
                tracing::info!("Hardware encoder: {} (qsv overflow lane: {})", enc, has_qsv);
                crate::nvr_pipes::set_qsv_available(has_qsv);
                *state_enc.hw_encoder.write().unwrap() = enc;
                state_enc.app_handle.emit("hw_encoder:detected",
                    state_enc.hw_encoder.read().unwrap().clone()).ok();
                // Hardware DECODER for the live detection stream (offloads H.264/H.265
                // decode off the CPU — the NVR's biggest CPU cost, à la mature NVRs).
                let dec = crate::hw::detect_hw_decoder(&ffmpeg).await;
                tracing::info!("Hardware decoder: {}", if dec.is_empty() { "software (none detected)" } else { &dec });
                *state_enc.hw_decoder.write().unwrap() = dec;
            }
        });
    }

    // Advertise the HTTP server via DNS-SD mDNS so Android NsdManager
    // can discover it instantly — no subnet scanning needed.
    {
        let state_mdns = Arc::clone(&state);
        tauri::async_runtime::spawn(async move {
            let device_name = state_mdns.settings.read().await.device_name.clone();
            let name = if device_name.is_empty() {
                hostname::get().ok()
                    .and_then(|h| h.into_string().ok())
                    .unwrap_or_else(|| "Anivar".to_string())
            } else { device_name };
            let port_mdns = state_mdns.settings.read().await.stream_port;
            tokio::task::spawn_blocking(move || {
                let host = format!("{}.local.", hostname::get()
                    .ok().and_then(|h| h.into_string().ok())
                    .unwrap_or_else(|| "anivar".to_string()));
                match mdns_sd::ServiceDaemon::new() {
                    Ok(mdns) => {
                        match mdns_sd::ServiceInfo::new(
                            "_anivar._tcp.local.",
                            &name, &host, "", port_mdns, None,
                        ) {
                            Ok(info) => {
                                mdns.register(info).ok();
                                tracing::info!("mDNS: advertising _anivar._tcp on port {}", port_mdns);
                                std::mem::forget(mdns); // keep daemon alive for process lifetime
                            }
                            Err(e) => tracing::warn!("mDNS service info error: {e}"),
                        }
                    }
                    Err(e) => tracing::warn!("mDNS daemon error: {e}"),
                }
            }).await.ok();
        });
    }

    let client_sessions_for_server = Arc::clone(&state.client_sessions);
    let kick_txs_for_server = Arc::clone(&state.kick_txs);

    // Free the stream port from any orphaned child left by a force-killed run
    // BEFORE we try to bind it (otherwise: blank camera). See fn docs above.
    kill_orphaned_app_children();

    tauri::async_runtime::spawn(async move {
        start_http_server(
            frame_txs_for_server, port, auth_token_for_server,
            app_handle_for_server, camera_active_for_server,
            camera_state_tx_clone,
            pool_for_server, data_dir_for_server, revoke_rx,
            client_sessions_for_server, kick_txs_for_server,
            lan_access_for_server,
        );
    });

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bound that matters: an existing install must survive the rename with
    /// its database INTACT — including the `-wal`, whose uncheckpointed commits
    /// are lost if the sidecars don't travel with the `.db`.
    /// Runs the whole legacy chain, one hop per entry: an install sitting on ANY
    /// past name must arrive intact. Parameterised so adding a future rename to
    /// `LEGACY_DATA_DIRS` is covered here automatically.
    #[test]
    /// The CURRENT names must never appear in their own legacy lists.
    ///
    /// That is the documented failure of this codebase's second rename: a bulk
    /// find/replace rewrote the legacy entry into the current identifier, which
    /// orphans the real install AND asks the migration to rename a directory
    /// onto itself. It is invisible in review because both literals look right
    /// in isolation.
    ///
    /// Cheap to assert, and it fails loudly on the next rename if someone
    /// prepends the new name instead of the outgoing one.
    #[test]
    fn current_names_are_not_in_their_own_legacy_lists() {
        assert!(!LEGACY_DATA_DIRS.contains(&crate::BUNDLE_ID),
            "BUNDLE_ID {} is in LEGACY_DATA_DIRS - the migration would rename a              directory onto itself", crate::BUNDLE_ID);
        let stem = DB_FILENAME.trim_end_matches(".db");
        assert!(!LEGACY_DB_STEMS.contains(&stem),
            "DB_FILENAME stem {stem} is in LEGACY_DB_STEMS - the migration would              rename the database onto itself");
        assert!(!crate::LEGACY_WEBVIEW_IDS.contains(&crate::BUNDLE_ID),
            "BUNDLE_ID {} is in LEGACY_WEBVIEW_IDS", crate::BUNDLE_ID);
    }

    fn migrates_every_legacy_data_dir_and_db() {
        let stem = DB_FILENAME.trim_end_matches(".db");
        for (i, (legacy_dir, legacy_stem)) in
            LEGACY_DATA_DIRS.iter().zip(LEGACY_DB_STEMS.iter()).enumerate()
        {
            let root = std::env::temp_dir()
                .join(format!("anivar_mig_{}_{i}", std::process::id()));
            let _ = std::fs::remove_dir_all(&root);
            let old = root.join(legacy_dir);
            let new = root.join(crate::BUNDLE_ID);

            // An install on that older name: database trio plus some "footage".
            std::fs::create_dir_all(old.join("nvr")).unwrap();
            std::fs::write(old.join(format!("{legacy_stem}.db")), b"main").unwrap();
            std::fs::write(old.join(format!("{legacy_stem}.db-wal")), b"uncheckpointed").unwrap();
            std::fs::write(old.join(format!("{legacy_stem}.db-shm")), b"shm").unwrap();
            std::fs::write(old.join("nvr").join("cam0.mp4"), b"video").unwrap();

            migrate_legacy_data_dir(&new);
            migrate_legacy_db(&new);

            assert!(!old.exists(), "{legacy_dir}: legacy dir should be gone after the move");
            assert_eq!(std::fs::read(new.join(DB_FILENAME)).unwrap(), b"main",
                       "{legacy_dir}: database must arrive");
            assert_eq!(std::fs::read(new.join(format!("{stem}.db-wal"))).unwrap(), b"uncheckpointed",
                       "{legacy_dir}: the dirty WAL must travel with the database");
            assert_eq!(std::fs::read(new.join(format!("{stem}.db-shm"))).unwrap(), b"shm");
            assert!(new.join("nvr").join("cam0.mp4").exists(),
                    "{legacy_dir}: footage should come across");

            // Idempotent: a second boot must not disturb the migrated install.
            migrate_legacy_data_dir(&new);
            migrate_legacy_db(&new);
            assert_eq!(std::fs::read(new.join(DB_FILENAME)).unwrap(), b"main");

            let _ = std::fs::remove_dir_all(&root);
        }
    }

    /// A panic during the first boot drops a `crash.log` into the brand-new data
    /// directory (the hook builds that path by hand). If that counted as "this
    /// install already has data", the several-GB legacy install would be skipped
    /// on this boot AND every boot after it — permanent, silent stranding.
    #[test]
    fn a_stray_crash_log_does_not_strand_the_legacy_install() {
        let root = std::env::temp_dir().join(format!("anivar_crashlog_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let old = root.join(LEGACY_DATA_DIRS[0]);
        let new = root.join(crate::BUNDLE_ID);

        std::fs::create_dir_all(old.join("nvr")).unwrap();
        std::fs::write(old.join(format!("{}.db", LEGACY_DB_STEMS[0])), b"main").unwrap();
        std::fs::write(old.join("nvr").join("cam0.mp4"), b"video").unwrap();

        // The first boot panicked before setup_app reached the migration.
        std::fs::create_dir_all(&new).unwrap();
        std::fs::write(new.join("crash.log"), b"panicked at ...").unwrap();

        migrate_legacy_data_dir(&new);
        migrate_legacy_db(&new);

        assert!(!old.exists(), "the legacy dir must still be claimed");
        assert_eq!(std::fs::read(new.join(DB_FILENAME)).unwrap(), b"main");
        assert!(new.join("nvr").join("cam0.mp4").exists(), "footage must come across");
        let _ = std::fs::remove_dir_all(&root);
    }

    /// The two chains are walked in lockstep, so they must stay the same length —
    /// otherwise a rename adds a directory with no database stem beside it.
    #[test]
    fn legacy_chains_are_parallel() {
        assert_eq!(LEGACY_DATA_DIRS.len(), LEGACY_DB_STEMS.len(),
                   "every legacy data dir needs its matching db stem");
    }

    /// A fresh install has no legacy dir: migration must be a silent no-op, not
    /// an error path that leaves a half-made directory behind.
    #[test]
    fn fresh_install_is_untouched() {
        let root = std::env::temp_dir().join(format!("anivar_fresh_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let new = root.join(crate::BUNDLE_ID);
        std::fs::create_dir_all(&new).unwrap();

        migrate_legacy_data_dir(&new);
        migrate_legacy_db(&new);

        assert!(new.is_dir());
        assert!(!new.join(DB_FILENAME).exists(), "no db should be conjured");
        let _ = std::fs::remove_dir_all(&root);
    }
}
