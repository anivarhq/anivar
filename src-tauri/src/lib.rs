// ── Crate-wide lint policy ───────────────────────────────────────────────────
// Five clippy lints are allowed deliberately. Each was reviewed rather than
// blanket-suppressed, and in each case applying the suggested fix would make the
// code worse, not better:
//
//   needless_range_loop (8 sites) — numeric kernels where the loop variable is
//     index ARITHMETIC, not merely a position: the landmark decoder reads
//     `get(5 + k * 3, i)`. An iterator still needs `k`, so `enumerate()` adds a
//     binding without removing the arithmetic.
//
//   type_complexity (30 sites) — almost entirely `sqlx::query_as` row tuples like
//     `Vec<(String, f32, Option<String>, ...)>`. The "fix" is a named type alias
//     per query, used exactly once, which adds indirection between the SQL and the
//     shape it returns instead of removing any.
//
//   too_many_arguments (7 sites) — functions such as `store_face_embedding` and
//     `start_http_server` that genuinely take that many independent domain values.
//     Bundling them into a struct purely to satisfy a count of 7 moves the argument
//     list somewhere else; it does not simplify the call.
//
//   doc_overindented_list_items / doc_lazy_continuation (22 sites) — the module
//     doc comments align their continuation lines into columns on purpose, the same
//     deliberate alignment used throughout this codebase (and the reason there is no
//     rustfmt gate). Reflowing them to satisfy the lint would break the alignment.
//
// Everything else is expected to be clean: `cargo clippy --lib` should report zero
// warnings, so a NEW lint is visible immediately instead of being lost in noise.
#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::doc_overindented_list_items)]
#![allow(clippy::doc_lazy_continuation)]
#![allow(clippy::needless_range_loop)]

mod agent;
pub use agent::{AgentAlert, AgentStatus};



mod crypto;

mod auth;
mod auth_cmds;
pub use auth_cmds::{
    auth_status, set_login_password, set_login_required, set_2fa_enabled, set_remember_device,
    login, login_verify_otp, auth_resume, lock, logout, request_recovery, recovery_reset,
};

mod hw;
pub use hw::detect_hw_encoder;

mod onvif;

mod state;
pub use state::*;


mod server;
pub use server::start_http_server;


/// Constant-time byte comparison to prevent timing-based token oracle attacks.
pub(crate) fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() { return false; }
    a.iter().zip(b.iter()).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

mod http_handlers;

mod capture;
use capture::run_capture_loop;

// v12: per-event clip writer removed. Event "clips" are now virtual slices of
// the continuous NVR recording, served by `/footage/:id/clip` via ffmpeg-concat
// (mature NVRs' `events/{id}/clip.mp4` model).

mod footage;

mod tailscale;
pub use tailscale::{tailscale_status, tailscale_enable};

mod share_security;



/// One contract for every artifact we download at runtime (see the module docs).
mod provision;
mod ffmpeg;
/// Rolling on-disk log — the app runs detached, so console output reaches nobody.
mod logfile;
pub use ffmpeg::ensure_ffmpeg;


mod boot;
#[cfg(any(feature = "cuda", target_os = "linux"))]
mod cuda_runtime;

mod motion;
mod motion_lifecycle;

mod face;
pub use face::{embed_face, recognize_frame, face_pipeline_status, face_debug};
mod face_classifier;
pub use face_classifier::{retrain_face_classifier, face_classifier_status};
mod liveness;
mod alpr;

mod recommend;


mod db;
pub use db::probe_mjpeg_url;

// ─── Tauri Commands ───────────────────────────────────────────────────────────

mod tunnel_cmds;
pub use tunnel_cmds::get_stream_info;

mod share_cmds;
pub use share_cmds::{generate_share_link, revoke_all_shares, list_active_shares};

mod search_cmds;
pub use search_cmds::check_for_update;

mod inference_cmds;
pub use inference_cmds::{stream_frame, process_frame, get_inference_status, get_camera_snapshot};
pub(crate) use inference_cmds::process_frame_inner;

mod frontend_cmds;
pub use frontend_cmds::{update_scene_objects, read_clip_frames, save_clip_blob};

mod nvr_pipes;
pub(crate) use nvr_pipes::{spawn_nvr_pipe, spawn_hls_pipe};

mod nvr_stream;
mod nvr_vod;
mod nvr_preview;
mod depth;
mod go2rtc;
#[cfg(windows)]
mod trtx_runtime;
#[cfg(feature = "openvino")]
mod openvino_runtime;

mod hls;


mod reid;
pub use reid::{list_tracked_persons, reid_backend_status, assign_tracked_to_known, list_tracked_clusters, name_tracked_group, unname_tracked_group, correct_track};

mod inference;
pub use inference::run_inference_loop;

mod embed;
mod vector_index;
mod jobs;
mod blobstore;
mod db_backup;
mod audio;
mod audio_cmds;
pub use audio_cmds::analyze_audio_window;
mod tracking;
mod timeline;
pub use timeline::get_event_timeline;


mod nvr_recording;
pub use nvr_recording::{start_nvr, stop_nvr, save_nvr_segment, get_nvr_segments, list_nvr_recordings, get_events_in_range, get_event_markers, list_recorded_days, search_events, find_similar_events, reindex_semantic_search, list_bookmarked_events};

mod rtsp;
pub use rtsp::{start_rtsp_relay, stop_rtsp_relay, probe_stream};
mod dshow;
pub use dshow::{list_dshow_cameras, start_dshow_camera};

mod cam_config;
pub use cam_config::{get_camera_configs, set_camera_config, CameraConfig};

mod native_cam_cmds;
pub use native_cam_cmds::{get_active_cameras, get_camera_telemetry, get_camera_inventory, report_browser_cameras, list_native_cameras, start_native_camera, stop_native_camera, stop_all_native_cameras};

mod events_cmds;
pub use events_cmds::{get_storage_info, get_settings, save_settings, get_motion_events, keep_alive_event, store_detections, store_ai_summary};

mod agent_cmds;
pub use agent_cmds::{get_agent_alerts, delete_agent_alert, clear_all_agent_alerts, set_alert_feedback, get_reflection_prompt, report_behavior_events, get_agent_memory, set_agent_memory, list_agent_memory, delete_agent_memory, get_agent_status, analyze_snapshot, chat_app, get_chat_log, clear_chat_log, trigger_agent_now, query_events, explore_events};

mod agent_data_cmds;
pub use agent_data_cmds::{search_clips, list_alert_conditions, create_alert_condition, delete_alert_condition, toggle_alert_condition, read_memory_file, write_memory_file, read_all_memory_files, report_crowd_count, delete_motion_event, delete_events, clear_nvr_recordings, clear_all_events, purge_orphaned_clips, delete_footage_in_range};

mod persons;
pub use persons::{enroll_person, enroll_person_multi, add_person_embedding, list_known_persons, delete_person,
            forget_person, rename_person, mark_person_seen, list_recent_unknown_faces, assign_face_to_person, create_person_from_face, list_person_faces, delete_face_embedding, list_recent_recognitions, list_unknown_clusters, get_person_sightings, get_person_events, list_vehicles, list_audio_events, get_person_stats, get_audio_stats, assign_faces_to_person, clear_unknown_faces, get_face_context, correct_face};
mod correlation;
pub use correlation::{record_face_sighting, get_camera_correlations, detect_anomalies};

mod agent_tools;
pub use agent_tools::{get_person_history, search_similar_events, trigger_alarm, send_telegram_test, telegram_connect, set_auth_password};

mod system_cmds;
pub use system_cmds::{GpuInfo, list_gpus, set_preferred_gpu, revoke_token, disconnect_client, get_local_ip, DiscoveredCamera, discover_cameras, recommend_face_model, list_installed_skills, SystemMetrics, get_system_metrics};

mod ai_provider;
pub use ai_provider::{list_provider_models, test_ai_provider};


mod hw_onvif;
pub use hw_onvif::{get_hw_encoder, fix_firewall, discover_onvif, get_onvif_streams, get_onvif_device_info, discover_and_configure_onvif};

/// User-downloadable AI models. Moved out of `hw_onvif` (a camera-discovery
/// file) — nothing about installing a model relates to ONVIF.
mod skills;
pub use skills::{check_skill_installed, download_skill, remove_skill};

mod review_segments;
pub use review_segments::{get_review_segments, set_review_segment_reviewed};
pub use nvr_preview::list_previews;

mod bookmarks;
pub use bookmarks::{add_bookmark, remove_bookmark, list_bookmark_ids};

mod proc;

mod hostinfo;

// ─── App Entry ───────────────────────────────────────────────────────────────

/// Where the durable crash log lives — the app's roaming data dir (same place the
/// DB + skills live), falling back to the OS temp dir. Kept dependency-free so it
/// works from inside the panic hook.
fn crash_log_path() -> std::path::PathBuf {
    #[cfg(windows)]
    let base = std::env::var_os("APPDATA").map(std::path::PathBuf::from);
    #[cfg(not(windows))]
    let base = std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".local/share"));
    // MUST match `identifier` in tauri.conf.json — this path is built by hand
    // (the panic hook cannot reach Tauri's `app_data_dir()`), so the two cannot
    // be derived from one another and will silently diverge if only one changes.
    let dir = base
        .map(|b| b.join(BUNDLE_ID))
        .unwrap_or_else(std::env::temp_dir);
    let _ = std::fs::create_dir_all(&dir);
    dir.join("crash.log")
}

/// The bundle identifier. MUST match `identifier` in `tauri.conf.json`.
///
/// Tauri derives the app-data directory from that value, but the panic hook and
/// the WebView2 migration below build their paths BY HAND — the hook cannot reach
/// Tauri's `app_data_dir()`, and the migration has to run before the webview
/// exists. Nothing derives one from the other, so they diverge silently if only
/// one is changed.
pub(crate) const BUNDLE_ID: &str = "com.anivar.app";

/// Every bundle identifier this app has used, NEWEST FIRST — the WebView2 twin of
/// `boot::LEGACY_DATA_DIRS`.
///
/// This was an inline array literal inside the function below while its four
/// siblings were all named constants, and that asymmetry is exactly the shape of
/// the bug recorded in the comment there: a scripted find/replace skips things
/// that look like config and rewrites things that look like code. Naming it puts
/// it where a rename author will actually look.
///
/// Add to the FRONT on the next rename. Never edit an existing entry and never
/// remove one — the only thing that can still find an old install is its name.
/// Present on Windows, where `migrate_legacy_webview_profile` uses it, and in
/// test builds everywhere, where `current_names_are_not_in_their_own_legacy_lists`
/// asserts the invariant. Without the `test` arm the guard-rail would only run on
/// Windows; without the `windows` arm clippy calls it dead on macOS and Linux and
/// `-D warnings` fails the build, which is exactly what it did.
#[cfg(any(windows, test))]
pub(crate) const LEGACY_WEBVIEW_IDS: &[&str] = &["com.nivar.app", "com.anvil.nvr", "com.securecam.app"];

/// Delete `HKCU\Run` autostart entries left behind by a previous product name.
///
/// `tauri_plugin_autostart` registers under the CURRENT app name, so after a
/// rename the old value survives pointing at an executable that no longer
/// exists. Windows then fails it silently at every logon, and the user has an
/// app that used to start itself and quietly stopped.
///
/// This is the gap the other chains already covered: `LEGACY_TASKS` does exactly
/// this for the KeepAlive scheduled task. Delete rather than migrate — the plugin
/// re-registers the current name on its own when autostart is enabled.
///
/// Add to the FRONT on the next rename, like every other legacy list.
#[cfg(windows)]
fn clear_legacy_autostart() {
    const LEGACY_AUTOSTART: &[&str] = &["Nivar", "Anvil NVR", "SecureCam"];
    for name in LEGACY_AUTOSTART {
        // Through `proc` — a raw spawn here got its OWN console window from a
        // release build that has none to inherit, so every launch flashed three
        // terminals (one per legacy name) before the app even appeared.
        let _ = crate::proc::std_cmd("reg")
            .args(["delete", r"HKCU\Software\Microsoft\Windows\CurrentVersion\Run",
                   "/v", name, "/f"])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
    }
}
#[cfg(not(windows))]
fn clear_legacy_autostart() {}

/// Move the pre-rename WebView2 profile onto the current identifier.
///
/// WebView2 keys its user-data folder by BUNDLE IDENTIFIER under `%LOCALAPPDATA%`,
/// so every rename orphans it — separately from, and in addition to, the
/// app-data migration in `boot.rs`. What lives there is the
/// frontend's `localStorage`: the saved `cam_source_<id>` entries that auto-start
/// each camera, plus layout and UI preferences. Losing it silently means the app
/// opens with the right database but no cameras running, which reads as data loss
/// even though nothing in the database was touched.
///
/// Same-volume rename, so it is instant. No-ops unless the destination is
/// absent-or-empty, so it is safe on a fresh install and on every later boot.
#[cfg(windows)]
fn migrate_legacy_webview_profile() {
    let Some(base) = std::env::var_os("LOCALAPPDATA").map(std::path::PathBuf::from) else { return };
    let new = base.join(BUNDLE_ID).join("EBWebView");
    if new.read_dir().map(|mut d| d.next().is_some()).unwrap_or(false) { return; }
    // Newest legacy first — an upgrade must not reach past a nearer profile to an
    // older one. Mirrors LEGACY_DATA_DIRS in boot.rs.
    // NOTE: these are LEGACY identifiers and must never be bulk-renamed with the
    // rest of the app — a blanket find/replace did exactly that once, replacing
    // the Anvil entry with the CURRENT id, which both orphans the real profile
    // and asks the code to move a directory onto itself. See LEGACY_WEBVIEW_IDS.
    for legacy in LEGACY_WEBVIEW_IDS {
        let old = base.join(legacy).join("EBWebView");
        if !old.is_dir() { continue; }
        if let Some(parent) = new.parent() { let _ = std::fs::create_dir_all(parent); }
        // Windows MoveFileEx refuses an existing destination, even an empty one.
        if new.exists() { let _ = std::fs::remove_dir(&new); }
        match std::fs::rename(&old, &new) {
            Ok(()) => tracing::info!("migrated WebView2 profile {} → {}", old.display(), new.display()),
            Err(e) => tracing::warn!(
                "WebView2 profile migration failed ({e}) — cameras may need re-adding once"),
        }
        return;
    }
}
#[cfg(not(windows))]
fn migrate_legacy_webview_profile() {}

/// Install a panic hook that records the EXACT panic site (thread, location,
/// message, full backtrace) to a flushed `crash.log` BEFORE the process can die.
///
/// Motivation: the app was intermittently "closing itself" with no diagnostic —
/// Windows reported `0xc0000409` (a Rust `abort()`), but nothing reached the normal
/// logs because a panic that crosses an FFI boundary (ORT / WebView2 / camera
/// callbacks) aborts immediately. The default hook only writes to stderr, which is
/// lost when the process vanishes. This hook ALSO appends to a durable file and
/// flushes it, so the next occurrence names the culprit. A silent crash.log after a
/// close means the fault was in NATIVE code (e.g. concurrent DirectML), not Rust —
/// itself a useful signal (see `inference::gpu_infer_guard`).
fn install_crash_handler() {
    // Capture backtraces even if the user never set the env var (the default hook
    // reads this; our own capture uses `force_capture` regardless).
    if std::env::var_os("RUST_BACKTRACE").is_none() {
        std::env::set_var("RUST_BACKTRACE", "1");
    }
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let thread = std::thread::current();
        let tname = thread.name().unwrap_or("<unnamed>").to_string();
        let loc = info
            .location()
            .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
            .unwrap_or_else(|| "<unknown location>".into());
        let msg = info
            .payload()
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| info.payload().downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "<non-string panic payload>".into());
        let bt = std::backtrace::Backtrace::force_capture();
        let when = chrono::Local::now().to_rfc3339();
        let record = format!(
            "\n===== PANIC @ {when} =====\nthread : {tname}\nlocation: {loc}\nmessage : {msg}\nbacktrace:\n{bt}\n============================\n"
        );
        // 1) structured log (best-effort; may be lost if the process aborts).
        tracing::error!(target: "panic", thread = %tname, location = %loc, message = %msg,
            "PANIC — full backtrace written to crash.log");
        // 2) durable, flushed file — survives an immediate abort.
        {
            use std::io::Write;
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(crash_log_path())
            {
                let _ = f.write_all(record.as_bytes());
                let _ = f.flush();
            }
        }
        // 3) preserve the original console output.
        default_hook(info);
    }));
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Console AND a rolling file in the data dir. The app normally runs detached
    // with no console attached, so console-only logging meant every boot line —
    // EP canary results, pack provisioning, camera failures — was written to
    // nobody. The file sink stays inert until `logfile::set_dir` runs in boot.
    {
        use tracing_subscriber::fmt::writer::MakeWriterExt;
        tracing_subscriber::fmt()
            .with_writer(std::io::stdout.and(logfile::FileWriter))
            .with_ansi(false) // the file is the primary reader now
            .init();
    }
    install_crash_handler();
    // MUST run before anything spawns children: ties every child process's
    // lifetime to ours (see proc.rs) so force-kills can't breed zombie ffmpegs.
    proc::adopt_kill_on_close_job();
    // An always-on NVR must come back from ANY unexpected death (fast-fail
    // aborts from the GPU stack bypass the panic hook entirely) — Windows
    // relaunches us via WER. Deliberate tray-quit is unaffected.
    proc::register_crash_restart();
    // UI compositing on the iGPU, models on the discrete GPU — must run before
    // the Builder spawns WebView2 so it applies to THIS launch.
    system_cmds::pin_webview_gpu_power_saving();
    // Same deal: the WebView2 profile must be in place BEFORE the Builder creates
    // the webview, or WebView2 makes a fresh empty one and we can never move it.
    migrate_legacy_webview_profile();
    clear_legacy_autostart();

    // Cap the tokio runtime at 8 workers. The default (= logical cores, 28 on
    // a modern desktop) committed ~20 idle worker stacks for a workload that is
    // I/O + spawn_blocking bound — CPU-heavy work (ORT, JPEG, ffmpeg) runs on
    // the blocking pool / child processes, not these workers. Must be set
    // before tauri::Builder touches the runtime.
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(8)
        .enable_all()
        .build()
        .expect("tokio runtime build failed");
    tauri::async_runtime::set(rt.handle().clone());
    // Keep the runtime alive for the process lifetime (set() stores a handle).
    std::mem::forget(rt);

    tauri::Builder::default()
        // MUST be first: if the app is already running, a 2nd launch just focuses the
        // existing window (recreating it if it was hidden to the tray) and the new
        // process exits — instead of a rival instance fighting over port 8882 / the DB
        // / the cameras. This is why launching again "crashed": two NVRs at once.
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            boot::open_control_panel(app);
        }))
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        // tauri-plugin-shell intentionally NOT loaded: nothing (frontend or backend)
        // uses it, and it's a command-execution surface. The backend spawns its own
        // processes via `proc.rs` (std/tokio Command), so shell exposure = pure risk.
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_autostart::init(
            tauri_plugin_autostart::MacosLauncher::LaunchAgent, Some(vec!["--headless"])
        ))
        // System tray: closing the window hides UI but keeps NVR recording alive.
        // That's deliberate (appliance), but it USED TO BE SILENT — the window
        // vanished while cameras kept running, so "I closed it, why is my webcam
        // still on?" had no answer on screen. Tell the user once per session.
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                window.hide().ok();
                api.prevent_close();
                static NOTIFIED: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if !NOTIFIED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    use tauri_plugin_notification::NotificationExt;
                    window.notification().builder()
                        .title("Anivar is still recording")
                        .body("Closing the window keeps cameras running in the background. To stop everything, right-click the tray icon and choose Quit.")
                        .show().ok();
                }
            }
        })
        .setup(boot::setup_app)
        .invoke_handler(tauri::generate_handler![
            persons::list_vehicle_events,
            system_cmds::trtx_status,
            system_cmds::accel_report,
            system_cmds::install_trtx_pack,
            system_cmds::import_trtx_sdk,
            system_cmds::benchmark_inference,
            system_cmds::nvr_disk_projection,
            check_skill_installed,
            download_skill,
            remove_skill,
            stream_frame,
            process_frame,
            get_inference_status,
            get_camera_snapshot,
            get_stream_info,
            generate_share_link,
            revoke_all_shares,
            list_active_shares,
            probe_mjpeg_url,
            check_for_update,
            get_settings,
            save_settings,
            get_motion_events,
            delete_motion_event,
            delete_events,
            clear_all_events,
            purge_orphaned_clips,
            delete_footage_in_range,
            revoke_token,
            disconnect_client,
            get_local_ip,
            get_storage_info,
            discover_cameras,
            get_system_metrics,
            list_native_cameras,
            read_clip_frames,
            save_clip_blob,
            start_nvr,
            stop_nvr,
            save_nvr_segment,
            get_nvr_segments,
            list_nvr_recordings,
            list_recorded_days,
            get_events_in_range,
            get_event_markers,
            search_events,
            find_similar_events,
            reindex_semantic_search,
            list_tracked_persons,
            list_tracked_clusters,
            name_tracked_group,
            unname_tracked_group,
            correct_track,
            reid_backend_status,
            assign_tracked_to_known,
            analyze_audio_window,
            get_event_timeline,
            clear_nvr_recordings,
            get_active_cameras,
            get_camera_telemetry,
            get_camera_configs,
            set_camera_config,
            start_rtsp_relay,
            probe_stream,
            list_dshow_cameras,
            start_dshow_camera,
            stop_rtsp_relay,
            get_camera_inventory,
            report_browser_cameras,
            update_scene_objects,
            start_native_camera,
            stop_native_camera,
            stop_all_native_cameras,
            store_detections,
            store_ai_summary,
            keep_alive_event,
            get_agent_alerts,
            delete_agent_alert,
            clear_all_agent_alerts,
            set_alert_feedback,
            get_reflection_prompt,
            report_behavior_events,
            get_agent_memory,
            set_agent_memory,
            list_agent_memory,
            delete_agent_memory,
            get_agent_status,
            trigger_agent_now,
            chat_app,
            get_chat_log,
            clear_chat_log,
            query_events,
            explore_events,
            search_clips,
            list_alert_conditions,
            create_alert_condition,
            delete_alert_condition,
            toggle_alert_condition,
            read_memory_file,
            write_memory_file,
            read_all_memory_files,
            report_crowd_count,
            analyze_snapshot,
            enroll_person,
            enroll_person_multi,
            embed_face,
            recognize_frame,
            face_pipeline_status,
            face_debug,
            retrain_face_classifier,
            face_classifier_status,
            add_person_embedding,
            list_known_persons,
            delete_person,
            rename_person,
            mark_person_seen,
            list_recent_unknown_faces,
            assign_face_to_person,
            list_unknown_clusters,
            get_person_sightings,
            get_person_events,
            list_vehicles,
            list_audio_events,
            get_person_stats,
            get_audio_stats,
            assign_faces_to_person,
            list_person_faces,
            delete_face_embedding,
            clear_unknown_faces,
            get_face_context,
            list_recent_recognitions,
            create_person_from_face,
            correct_face,
            send_telegram_test,
            telegram_connect,
            tailscale_status,
            tailscale_enable,
            set_auth_password,
            auth_status,
            set_login_password,
            set_login_required,
            set_2fa_enabled,
            set_remember_device,
            login,
            login_verify_otp,
            auth_resume,
            lock,
            logout,
            request_recovery,
            recovery_reset,
            list_gpus,
            set_preferred_gpu,
            recommend_face_model,
            list_installed_skills,
            get_person_history,
            search_similar_events,
            trigger_alarm,
            record_face_sighting,
            get_camera_correlations,
            detect_anomalies,
            list_provider_models,
            test_ai_provider,
            get_hw_encoder,
            fix_firewall,
            discover_onvif,
            get_onvif_streams,
            get_onvif_device_info,
            discover_and_configure_onvif,
            get_review_segments,
            list_previews,
            set_review_segment_reviewed,
            add_bookmark,
            remove_bookmark,
            list_bookmark_ids,
            list_bookmarked_events,
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app_handle, event| {
            // RESILIENCE: an always-on NVR must survive its GUI dying. A WebView2
            // renderer/host crash (seen as the app "closing itself" — GPU oversubscription
            // on a hybrid laptop could take the webview + ffmpeg children down together)
            // must NOT end the process. A DELIBERATE quit goes tray → std::process::exit(0),
            // which bypasses this handler entirely, so everything here only ever catches
            // UNEXPECTED UI loss.
            match event {
                // The window was destroyed out from under us (not a user close — those are
                // intercepted as CloseRequested → hide). Auto-REBUILD it so the UI self-heals
                // instead of leaving a headless process the user thinks has "closed itself".
                // Throttled so a webview that crashes on load can't spin in a rebuild loop.
                tauri::RunEvent::WindowEvent { label, event: tauri::WindowEvent::Destroyed, .. }
                    if label == "main" =>
                {
                    use std::sync::atomic::{AtomicI64, Ordering};
                    static LAST_REBUILD: AtomicI64 = AtomicI64::new(0);
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_secs() as i64).unwrap_or(0);
                    if now - LAST_REBUILD.load(Ordering::Relaxed) >= 10 {
                        LAST_REBUILD.store(now, Ordering::Relaxed);
                        tracing::warn!("main window destroyed unexpectedly — NVR backend alive, rebuilding UI");
                        boot::open_control_panel(app_handle);
                    } else {
                        tracing::warn!("main window destroyed again within 10s — leaving headless (reopen from tray)");
                    }
                }
                // Backstop: if Tauri still tries to exit (e.g. rebuild failed → 0 windows),
                // refuse — keep recording. The user reopens from the tray.
                tauri::RunEvent::ExitRequested { api, .. } => {
                    api.prevent_exit();
                    tracing::warn!("exit requested with no window — prevented; NVR still recording, reopen from the tray");
                }
                _ => {}
            }
        });
}
