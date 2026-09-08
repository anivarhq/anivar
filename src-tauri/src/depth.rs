//! Depth Map Anonymization — edge-AI NVRs' privacy model, enforced SERVER-SIDE.
//!
//! When enabled for a camera, the ONLY pixels that ever persist or leave the
//! process are a colorized depth map (near = warm, far = cool): recordings,
//! live streams, snapshots, thumbnails, Telegram sends and share links all
//! carry depth frames. Raw camera pixels exist exclusively in RAM for the
//! local detection stack (YOLO / faces / plates keep full fidelity — events
//! still get labeled and named), matching edge-AI NVRs' "all analytics local,
//! identity never stored" guarantee.
//!
//! Model: Depth-Anything-v2-small (`skills/depth_anything/model.onnx`,
//! installable from Arsenal — same weights the old cosmetic browser overlay
//! used via transformers.js, now run through ORT with the DirectML session
//! rules shared with every other model).

use std::collections::HashSet;
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use crate::AppState;
use crate::inference::{build_ort_session, gpu_infer_guard};
use ort::session::Session as OrtSession;

// ─── Mode registry (refreshed at boot + on settings save) ─────────────────────

static ANON_CAMS: RwLock<Option<HashSet<u8>>> = RwLock::new(None);

// Depth SPEED PRESET (settings.depth_model): "fast" | "balanced" (default) |
// "quality". The efficiency lever is RESOLUTION, not quantization: the model
// input is fully dynamic (accepts any H×W divisible by 14), and lowering it is
// a big, genuine speedup — measured 518→252 ≈ 4.5× fewer ViT tokens — with no
// quality cost that matters for a privacy depth map (it's JPEG'd + upscaled to
// the frame anyway). We ship ONE model file (FP16, ~50 MB — half the old FP32,
// and faster on GPU); the preset only changes the inference resolution.
// (INT8/Q4 quantization was rejected: smaller download but SLOWER on our GPU —
// those ops target CPU/NPU and don't accelerate on DirectML/TensorRT.)
static DEPTH_PRESET: RwLock<Option<String>> = RwLock::new(None);

/// Record the chosen speed preset. Resolution is read live per-frame, so a
/// preset change takes effect immediately (no restart needed).
pub(crate) fn set_depth_variant(preset: &str) {
    *DEPTH_PRESET.write().unwrap() = Some(preset.to_string());
}

/// Inference resolution for the current preset. Multiples of 14 (DINOv2 patch
/// grid). Legacy value "fp16"/unset → balanced.
fn depth_resolution() -> u32 {
    match DEPTH_PRESET.read().unwrap().as_deref() {
        Some("fast")    => 252, // 14×18 — ~4.5× faster than 518
        Some("quality") => 518, // 14×37 — model's native training size
        _               => 378, // 14×27 — balanced default (~1.7× faster)
    }
}

/// Parse `settings.depth_anonymize` (JSON `{"<camId>": true}`) into the registry.
pub(crate) fn refresh_anon_cams(depth_anonymize_json: &str) {
    let mut set = HashSet::new();
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(depth_anonymize_json) {
        if let Some(map) = v.as_object() {
            for (k, on) in map {
                if on.as_bool() == Some(true) {
                    if let Ok(cam) = k.parse::<u8>() { set.insert(cam); }
                }
            }
        }
    }
    *ANON_CAMS.write().unwrap() = Some(set);
}

/// Is depth anonymization ON for this camera? (false until first refresh)
pub(crate) fn is_anonymized(cam: u8) -> bool {
    ANON_CAMS.read().unwrap().as_ref().map(|s| s.contains(&cam)).unwrap_or(false)
}

// ─── Model ─────────────────────────────────────────────────────────────────────

/// Resolve the depth model file. Prefer FP16 (~50 MB, faster on GPU); fall back
/// to the legacy FP32 `model.onnx` if that's what's on disk, so existing installs
/// and the fail-closed contract keep working (a missing model degrades to "no
/// model" — a blank anonymized feed — never to leaking raw pixels). Returns the
/// FP16 path even when absent so `model_installed` reports false correctly.
pub(crate) fn model_path(data_dir: &std::path::Path) -> std::path::PathBuf {
    let dir = data_dir.join("skills").join("depth_anything");
    let fp16 = dir.join("model_fp16.onnx");
    if fp16.exists() { return fp16; }
    let fp32 = dir.join("model.onnx");
    if fp32.exists() { return fp32; }
    fp16
}
pub(crate) fn model_installed(data_dir: &std::path::Path) -> bool {
    // `.exists()` accepted a truncated download or a saved HTML error page, which
    // then reported "installed" and produced a permanently blank feed. Use the
    // project's own install check instead — the same one provisioning uses.
    crate::provision::Requirement::MinSize(crate::skills::MIN_MODEL_BYTES)
        .met(&model_path(data_dir))
}

/// Lazy singleton session, with a retry cooldown on failure.
///
/// This was `OnceLock<Option<Mutex<..>>>`, which cached **failure permanently**:
/// installing the depth skill after boot left the feed blank until the app was
/// restarted — the same defect `run_inference_loop` had. Success is still cached
/// for the life of the process; only failure is retried, and only after a
/// cooldown, so a genuinely broken model still cannot spin the GPU in a loop.
static SESSION: OnceLock<Mutex<OrtSession>> = OnceLock::new();
static NEXT_TRY: Mutex<Option<std::time::Instant>> = Mutex::new(None);
const SESSION_RETRY: std::time::Duration = std::time::Duration::from_secs(30);

fn session(data_dir: &std::path::Path) -> Option<&'static Mutex<OrtSession>> {
    if let Some(s) = SESSION.get() { return Some(s); }

    // Cooldown gate. Returning None here keeps the fail-closed behaviour intact:
    // the worker leaves `latest_depth` unset and emits nothing, so a missing model
    // blanks the feed rather than leaking raw pixels.
    {
        let mut next = NEXT_TRY.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(t) = *next {
            if std::time::Instant::now() < t { return None; }
        }
        *next = Some(std::time::Instant::now() + SESSION_RETRY);
    }

    let dir = data_dir.join("skills").join("depth_anything");
    // Try FP16 first (smaller + faster); if it's absent OR fails to load on this
    // runtime, fall back to the legacy FP32 model.onnx.
    for p in [dir.join("model_fp16.onnx"), dir.join("model.onnx")] {
        if !p.exists() { continue; }
        match build_ort_session(&p) {
            Ok(s) => {
                tracing::info!("depth: Depth-Anything-v2 session ready — {} ({})",
                    p.file_name().and_then(|f| f.to_str()).unwrap_or("model.onnx"),
                    crate::inference::active_accelerator());
                let _ = SESSION.set(Mutex::new(s));
                return SESSION.get();
            }
            Err(e) => tracing::warn!("depth: {} load failed: {e} — trying fallback",
                p.file_name().and_then(|f| f.to_str()).unwrap_or("?")),
        }
    }
    None
}


/// JPEG in → colorized-depth JPEG out (same dimensions). Blocking (GPU + image
/// work) — call from `spawn_blocking`. Runs under `gpu_infer_guard` (DirectML
/// sessions must never run concurrently — hard rule).
pub(crate) fn depth_anonymize_jpeg(data_dir: &std::path::Path, jpeg: &[u8]) -> Option<Vec<u8>> {
    let sess = session(data_dir)?;
    let img = image::load_from_memory(jpeg).ok()?.to_rgb8();
    let (w, h) = (img.width(), img.height());

    // Preprocess: resize to the PRESET resolution (dynamic model input),
    // ImageNet-normalized NCHW f32. Lower res = fewer ViT tokens = faster.
    let size = depth_resolution();
    let small = image::imageops::resize(&img, size, size, image::imageops::FilterType::Triangle);
    const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const STD:  [f32; 3] = [0.229, 0.224, 0.225];
    let hw = (size * size) as usize;
    let mut input = vec![0f32; 3 * hw];
    for (i, px) in small.pixels().enumerate() {
        for c in 0..3 {
            input[c * hw + i] = (px.0[c] as f32 / 255.0 - MEAN[c]) / STD[c];
        }
    }

    let tensor = ort::value::Tensor::<f32>::from_array(
        ([1usize, 3, size as usize, size as usize], input)).ok()?;
    let depth: Vec<f32> = {
        let mut s = sess.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let input_name: String = s.inputs().first()
            .map(|i| i.name().to_string()).unwrap_or_else(|| "pixel_values".into());
        let outputs = {
            let _t = crate::inference::infer_timer("depth");
            let _gpu = gpu_infer_guard();
            s.run(ort::inputs![input_name.as_str() => tensor]).ok()?
        };
        let (_shape, raw): (Vec<i64>, Vec<f32>) = outputs[0].try_extract_tensor::<f32>()
            .ok().map(|(s0, d)| (s0.to_vec(), d.to_vec()))?;
        raw
    };
    if depth.len() < hw { return None; }

    // Normalize (model emits relative inverse depth: larger = closer) → turbo LUT.
    let (mut lo, mut hi) = (f32::MAX, f32::MIN);
    for &v in &depth[..hw] {
        if v < lo { lo = v; }
        if v > hi { hi = v; }
    }
    let range = (hi - lo).max(1e-6);
    let lut = turbo_lut();
    let mut colored = image::RgbImage::new(size, size);
    for (i, px) in colored.pixels_mut().enumerate() {
        let t = ((depth[i] - lo) / range * 255.0) as usize;
        *px = image::Rgb(lut[t.min(255)]);
    }

    // Back to the frame's native size; encode JPEG.
    let out = image::imageops::resize(&colored, w, h, image::imageops::FilterType::Triangle);
    let mut buf = Vec::with_capacity(64 * 1024);
    let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, 80);
    enc.encode_image(&image::DynamicImage::ImageRgb8(out)).ok()?;
    Some(buf)
}

/// Google's Turbo colormap, 256-entry LUT built once from the published
/// polynomial approximation (near = red/warm at t=1 after our inverse-depth
/// normalization maps CLOSER → HIGHER t).
fn turbo_lut() -> &'static [[u8; 3]; 256] {
    static LUT: OnceLock<[[u8; 3]; 256]> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut lut = [[0u8; 3]; 256];
        for (i, e) in lut.iter_mut().enumerate() {
            let x = i as f32 / 255.0;
            let r = 34.61 + x * (1172.33 + x * (-10793.56 + x * (33300.12 + x * (-38394.49 + x * 14825.05))));
            let g = 23.31 + x * (557.33 + x * (1225.33 + x * (-3574.96 + x * (1073.77 + x * 707.56))));
            let b = 27.2 + x * (3211.1 + x * (-15327.97 + x * (27814.0 + x * (-22569.18 + x * 6838.66))));
            *e = [r.clamp(0.0, 255.0) as u8, g.clamp(0.0, 255.0) as u8, b.clamp(0.0, 255.0) as u8];
        }
        lut
    })
}

// ─── Per-camera anonymization worker ──────────────────────────────────────────

/// Paced fan-out: emits the latest depth frame to every EXTERNAL consumer at a
/// steady 30 fps CFR (frame_txs //stream/share, nvr pipe → recording, hls pipe
/// → live view, latest_frames → snapshots/Telegram/agent). Inference runs at
/// its own pace on the newest raw frame (drop-don't-queue); emitted duplicates
/// keep the CFR-stamped recorder/HLS timelines honest. Ends when the capture
/// reader drops the raw sender (respawn creates a fresh worker).
pub(crate) fn spawn_depth_worker(
    state: Arc<AppState>,
    cam: u8,
    mut raw_rx: tokio::sync::watch::Receiver<Option<Arc<Vec<u8>>>>,
) {
    tokio::spawn(async move {
        let latest_depth: Arc<Mutex<Option<Arc<Vec<u8>>>>> = Arc::new(Mutex::new(None));
        let busy = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mut tick = tokio::time::interval(std::time::Duration::from_millis(33));
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        let mut ticks: u64 = 0;
        let mut produced: u64 = 0;
        let mut last_rate_log = std::time::Instant::now();
        tracing::info!("cam{cam} DEPTH ANONYMIZATION ON — raw pixels are never persisted or served");
        loop {
            tick.tick().await;
            ticks += 1;
            // Reader gone (capture died/respawned) → this worker retires.
            if raw_rx.has_changed().is_err() { return; }

            // Kick an inference if a new raw frame arrived and the GPU lane is free.
            if raw_rx.has_changed().unwrap_or(false)
                && !busy.swap(true, std::sync::atomic::Ordering::AcqRel)
            {
                let raw = raw_rx.borrow_and_update().clone();
                match raw {
                    Some(jpeg) => {
                        let dd = state.data_dir.clone();
                        let slot = Arc::clone(&latest_depth);
                        let busy2 = Arc::clone(&busy);
                        tokio::task::spawn_blocking(move || {
                            if let Some(d) = depth_anonymize_jpeg(&dd, &jpeg) {
                                *slot.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
                                    = Some(Arc::new(d));
                            }
                            busy2.store(false, std::sync::atomic::Ordering::Release);
                        });
                    }
                    None => { busy.store(false, std::sync::atomic::Ordering::Release); }
                }
            }

            let frame = latest_depth.lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner).clone();
            let Some(frame) = frame else { continue }; // model warming up
            produced += 1;
            let _ = state.frame_txs[cam as usize].send(Arc::clone(&frame));
            if let Some(tx) = state.nvr_pipe_txs.lock().await.get(&cam) { tx.try_send(Arc::clone(&frame)).ok(); } // drop-on-full
            if let Some(tx) = state.hls_pipe_txs.lock().await.get(&cam) { tx.try_send(Arc::clone(&frame)).ok(); } // drop-on-full
            if ticks % 5 == 0 { // ~160ms cadence, mirrors fan_out_frame's throttle
                state.latest_frames.write().await.insert(cam, frame.as_ref().clone());
            }
            if last_rate_log.elapsed().as_secs() >= 300 {
                tracing::info!("cam{cam} depth worker: emitting {} fps (CFR-paced)", produced / 300);
                produced = 0;
                last_rate_log = std::time::Instant::now();
            }
        }
    });
}
