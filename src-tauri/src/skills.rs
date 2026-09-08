//! Skills — the user-downloadable AI models (detector, face, ALPR, CLIP, depth,
//! audio, Re-ID) that live under `<data_dir>/skills/<id>/`.
//!
//! These commands used to live in `hw_onvif.rs`, a camera-discovery file, purely
//! because that is where they were first written. Nothing about downloading a
//! model relates to ONVIF, and the misfiling actively misleads: looking for the
//! model installer in `system_cmds.rs` (the obvious place) finds nothing. The
//! filesystem scan that reports install state moved here from `recommend.rs` for
//! the same reason — one home for skills.
//!
//! Download mechanics (timeouts, retry, status checks, atomic publish) come from
//! [`crate::provision`]; this file owns only skill-specific policy.

use std::path::Path;
use std::sync::Arc;

use serde::Serialize;
use tauri::State;

use crate::AppState;
use crate::provision::{self, Requirement};

/// Smallest plausible weights file. A model that "downloaded" to fewer bytes than
/// this is an HTML error page, a 0-byte stub from a killed download, or a
/// truncated transfer — all of which used to count as INSTALLED and then failed at
/// session-build time with an opaque ONNX parse error the user couldn't act on.
pub(crate) const MIN_MODEL_BYTES: u64 = 64 * 1024;

fn is_real_model(path: &std::path::Path) -> bool {
    Requirement::MinSize(MIN_MODEL_BYTES).met(path)
}

/// Returns true if the skill model file is present in <data_dir>/skills/<skill_id>/
/// For YOLO26: checks for model.onnx (the onnx-community/yolo26x-ONNX export).
///
/// `min_bytes` lets the caller demand a floor bigger than [`MIN_MODEL_BYTES`].
/// The on-device LLM needs it: its weights were replaced by a larger model at the
/// SAME path (`skills/local_llm/model.gguf`), and a superseded 230 MB file
/// satisfies every "is a weight file present" test there is — so without a floor
/// the upgrade is never offered and the user keeps the old brain forever.
#[tauri::command]
pub async fn check_skill_installed(
    skill_id: String,
    min_bytes: Option<u64>,
    state: State<'_, Arc<AppState>>,
) -> Result<bool, String> {
    let skill_dir = state.data_dir.join("skills").join(&skill_id);
    if let Some(floor) = min_bytes.filter(|b| *b > MIN_MODEL_BYTES) {
        // A specific floor means a specific file: don't let the generic
        // "any weight file will do" fallback below wave through the old one.
        for name in &["model.onnx", "model.gguf"] {
            if Requirement::MinSize(floor).met(&skill_dir.join(name)) { return Ok(true); }
        }
        return Ok(false);
    }
    // Weight formats we ship. `.gguf` is NOT optional: the on-device LLM skill
    // (`local_llm`) is a GGUF, and while this only looked for `.onnx` it reported
    // "not installed" for a present, working 229 MB model — which hid the ready
    // state AND the button that selects it, leaving no way to pick on-device.
    for name in &["model.onnx", "model.gguf", "yolo26n.onnx", "yolo26s.onnx", "yolo26m.onnx", "yolo26x.onnx"] {
        if is_real_model(&skill_dir.join(name)) { return Ok(true); }
    }
    // Fallback: accept any weight file in the skill directory. In-progress
    // downloads live under `.part` and never match the extension test.
    if let Ok(mut dir) = tokio::fs::read_dir(&skill_dir).await {
        while let Ok(Some(entry)) = dir.next_entry().await {
            let p = entry.path();
            let ext = p.extension().and_then(|e| e.to_str());
            if matches!(ext, Some("onnx") | Some("gguf")) && is_real_model(&p) {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// Destinations with a download in flight — one writer per file, process-wide.
static IN_FLIGHT: std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<std::path::PathBuf>>>
    = std::sync::OnceLock::new();

/// RAII claim on a download destination; released even if the command errors out.
struct DownloadGuard(std::path::PathBuf);

impl DownloadGuard {
    fn acquire(dest: &std::path::Path) -> Result<Self, String> {
        let set = IN_FLIGHT.get_or_init(Default::default);
        let mut g = set.lock().unwrap_or_else(|p| p.into_inner());
        if !g.insert(dest.to_path_buf()) {
            return Err("that download is already running".into());
        }
        Ok(Self(dest.to_path_buf()))
    }
}

impl Drop for DownloadGuard {
    fn drop(&mut self) {
        if let Some(set) = IN_FLIGHT.get() {
            set.lock().unwrap_or_else(|p| p.into_inner()).remove(&self.0);
        }
    }
}

/// Download a skill package from `url` into <data_dir>/skills/<skill_id>/.
/// Emits "skill:progress" events as the download progresses. When `filename`
/// is supplied, the blob is written under that name; otherwise the URL's
/// trailing path component is used (legacy behaviour for single-file skills).
#[tauri::command]
pub async fn download_skill(
    skill_id: String,
    url: String,
    filename: Option<String>,
    state: State<'_, Arc<AppState>>,
    app: tauri::AppHandle,
) -> Result<(), String> {
    use tauri::Emitter;

    let skills_dir = state.data_dir.join("skills").join(&skill_id);
    tokio::fs::create_dir_all(&skills_dir).await.map_err(|e| e.to_string())?;

    let filename = filename
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| url.split('/').next_back().unwrap_or("skill.bin").to_string());
    // Never let a supplied name escape the skill directory (`../../ffmpeg.exe`).
    let filename = std::path::Path::new(&filename)
        .file_name().and_then(|f| f.to_str()).unwrap_or("skill.bin").to_string();
    let dest = skills_dir.join(&filename);

    // One download per destination. Two clicks on "Install" (or an install racing
    // an auto-install) used to point two streams at the same file and interleave
    // them into an unloadable model.
    let _guard = DownloadGuard::acquire(&dest)?;

    // Emit an immediate 0-byte tick so the UI flips from "queued" to
    // "downloading" the moment we start.
    let tick = |downloaded: u64, total: Option<u64>| {
        let _ = app.emit("skill:progress", serde_json::json!({
            "skill_id": &skill_id,
            "percent": total.filter(|t| *t > 0)
                .map(|t| ((downloaded * 100 / t).min(100)) as u8).unwrap_or(0),
            "downloaded": downloaded,
            "total": total,
        }));
    };
    tick(0, None);

    // Throttled progress: emit on whichever fires first — a 64 KB delta or 250 ms
    // wall-clock. Keeps the bar smooth without flooding the event channel.
    // Streaming, atomic publish, status checks, the content-length match and
    // partial-file cleanup all live in `provision::fetch_to_file`.
    let mut last_bytes = 0u64;
    let mut last_at = std::time::Instant::now();
    let got = provision::fetch_to_file(&provision::client(), &url, &dest, |downloaded, total| {
        if downloaded.saturating_sub(last_bytes) >= 64 * 1024
            || last_at.elapsed().as_millis() >= 250
        {
            tick(downloaded, total);
            last_bytes = downloaded;
            last_at = std::time::Instant::now();
        }
    }).await.map_err(|e| format!("download failed: {e}"))?;

    // A 404/403 page is a valid HTTP response — it just isn't a model. The size
    // floor is what stops one being reported as a successful install.
    if got < MIN_MODEL_BYTES {
        let _ = tokio::fs::remove_file(&dest).await;
        return Err(format!("server returned only {got} bytes — not a model file"));
    }

    tick(got, Some(got)); // final 100%
    tracing::info!("Skill '{}' downloaded to {:?} ({} bytes)", skill_id, dest, got);
    Ok(())
}

/// Remove a skill — deletes <data_dir>/skills/<skill_id>/ (plus any LEGACY
/// directory that `list_installed_skills` folds into this id).
///
/// `list_installed_skills` reports the pre-tier `skills/yolo26/` install on the
/// `yolo26x` row, and the v6 `skills/alpr/` install on `alpr_global`. Without
/// also clearing those legacy dirs here, the UI would show XLarge / Global as
/// installed (with a size) but "Remove" would silently delete nothing — the
/// exact bug behind "YOLO26 XLarge can't be deleted".
#[tauri::command]
pub async fn remove_skill(
    skill_id: String,
    state: State<'_, Arc<AppState>>,
) -> Result<(), String> {
    let skills = state.data_dir.join("skills");
    let mut targets = vec![skills.join(&skill_id)];
    match skill_id.as_str() {
        "yolo26x"     => targets.push(skills.join("yolo26")),
        "alpr_global" => targets.push(skills.join("alpr")),
        _ => {}
    }

    // The on-device LLM is memory-MAPPED by llama.cpp while loaded, and Windows
    // refuses to delete a mapped file — uninstalling within the 180 s idle window
    // after any chat would fail with a sharing violation. Release it first. The
    // worker may be mid-generation, so give it a moment to act on the message.
    // ANY llm tier, not just the original id — a hardcoded `== "local_llm"`
    // would let removing the Fast or Vision tier hit the sharing violation the
    // guard exists to prevent.
    if crate::agent::local_llm::TIERS.iter().any(|(_, id, _)| *id == skill_id) {
        crate::agent::local_llm::unload();
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    }

    let mut removed_any = false;
    for path in targets {
        if path.exists() {
            tokio::fs::remove_dir_all(&path).await.map_err(|e| e.to_string())?;
            removed_any = true;
        }
    }
    tracing::info!("Skill '{}' removed (cleared dirs: {})", skill_id, removed_any);
    Ok(())
}

// ── Installed-skill filesystem scan ──────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct SkillStatus {
    pub id:               String,
    pub name:             String,
    pub installed:        bool,
    pub size_on_disk_mb:  u64,
}

/// Walk `<data>/skills/` and report which of our known skill IDs have files
/// on disk. Doesn't (yet) report which is "active" — that's up to the caller
/// since `active` depends on settings the recommend module doesn't read.
pub(crate) fn list_installed_skills(data_dir: &Path) -> Vec<SkillStatus> {
    // One entry per YOLO 2026 variant + the two face tiers + ALPR regions.
    // The legacy `yolo26/` directory is folded into `yolo26x` when reporting
    // so existing installs keep showing up as "Installed" on the XLarge row.
    // The legacy `alpr/` directory is folded into `alpr_global` (v7 split).
    const SKILLS: &[(&str, &str)] = &[
        ("yolo26n",         "YOLO26 — Nano"),
        ("yolo26s",         "YOLO26 — Small"),
        ("yolo26m",         "YOLO26 — Medium"),
        ("yolo26l",         "YOLO26 — Large"),
        ("yolo26x",         "YOLO26 — XLarge"),
        ("face_small",      "Face Recognition — Small"),
        ("face_large",      "Face Recognition — Large"),
        ("alpr_global",     "License Plates — Global"),
        ("alpr_european",   "License Plates — European"),
        ("alpr_argentinian","License Plates — Argentinian"),
        ("mobileclip_s0",   "Semantic Search — MobileCLIP-S0"),
        ("clip_b32",        "Semantic Search — CLIP ViT-B/32"),
        ("jina_clip",       "Semantic Search — Jina-CLIP"),
        ("audio_yamnet",    "Audio Detection — YAMNet"),
        ("reid_osnet",      "Person Re-ID — OSNet"),
        ("depth_anything",  "Depth Anonymization — Depth-Anything-v2"),
        // The on-device language model. Absent from this table it had NO uninstall
        // path at all: "Installed models" is the only surface that renders a Remove
        // button, and it renders exactly what this table lists. Sizing below uses
        // walk_dir_size, which doesn't care that this one is a .gguf and not .onnx.
        ("local_llm_fast",   "On-device AI — Fast (LFM2.5-350M)"),
        ("local_llm",        "On-device AI — Balanced (LFM2.5-1.2B)"),
        ("local_llm_vision", "On-device AI — Vision (LFM2.5-VL-1.6B)"),
    ];
    SKILLS.iter().map(|(id, name)| {
        let dir = data_dir.join("skills").join(id);
        // Legacy fallback: pre-tier installs landed in `skills/yolo26/`. If we
        // see the XL row and the new directory doesn't exist, treat the legacy
        // directory as the XLarge install.
        let probe_dir = if *id == "yolo26x" && !dir.exists() {
            data_dir.join("skills").join("yolo26")
        } else if *id == "alpr_global" && !dir.exists() {
            // v6 → v7 migration: a single `skills/alpr/` install is treated as
            // the Global variant so users don't see it as "uninstalled" after upgrade.
            data_dir.join("skills").join("alpr")
        } else {
            dir
        };
        let (installed, size) = if probe_dir.exists() {
            let bytes = walk_dir_size(&probe_dir);
            (bytes > 0, bytes / (1024 * 1024))
        } else {
            (false, 0)
        };
        SkillStatus {
            id: (*id).to_string(),
            name: (*name).to_string(),
            installed,
            size_on_disk_mb: size,
        }
    }).collect()
}

fn walk_dir_size(p: &Path) -> u64 {
    let Ok(rd) = std::fs::read_dir(p) else { return 0 };
    rd.flatten().fold(0u64, |acc, e| {
        let Ok(meta) = e.metadata() else { return acc };
        if meta.is_dir() {
            acc + walk_dir_size(&e.path())
        } else {
            acc + meta.len()
        }
    })
}
