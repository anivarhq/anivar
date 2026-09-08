//! YOLO26 ONNX inference engine.
//!
//! In-process YOLO object detection via Microsoft ONNX Runtime (the `ort`
//! crate). Loads the user-installed `model.onnx` skill from the application
//! data directory and runs a single shared inference loop driven by frames
//! pushed from every active camera capture.
//!
//! * [`find_yolo26_model`]              — locate `<data_dir>/skills/yolo26/model.onnx`
//! * [`decode_yolo_output`]            — parse raw output tensor into per-class detections
//! * [`parse_motion_masks_for_cam`]   - read motion-mask polygons for a given cam
//! * [`filter_detections_by_masks`]   - drop detections whose bbox-bottom-center is inside a mask
//! * [`run_inference_loop`]            — the long-running task driven by `state.infer_queue`
//! * [`build_ort_session`]             — ORT session builder with optimisation + thread tuning
//! * [`preprocess_jpeg_for_yolo`]     — JPEG -> normalised 1x3x640x640 f32 input tensor

use std::sync::Arc;

use ort::session::{Session as OrtSession, builder::GraphOptimizationLevel};
use tauri::Emitter;

use crate::{AppState, SceneObject};
use crate::motion::is_point_in_polygon;
use crate::reid::{body_descriptor, query_body_reid, store_body_embedding, crop_person_jpeg};

/// Mature NVRs two-tier scoring: detections must first pass `min_score`
/// (yolo_confidence_threshold) just to be considered, then a HIGHER confirm
/// threshold to actually confirm/classify a tracked object for the event. This
/// split cuts mislabels while still surfacing weaker detections in the overlay.
pub(crate) const YOLO_CONFIRM_THRESHOLD: f32 = 0.65;

/// Find the installed YOLO26 ONNX model for the requested variant.
/// Variant tiers map to dedicated skill subdirectories:
///   nano   → `skills/yolo26n/model.onnx`
///   small  → `skills/yolo26s/model.onnx`
///   medium → `skills/yolo26m/model.onnx`
///   large  → `skills/yolo26l/model.onnx`
///   xlarge → `skills/yolo26x/model.onnx` (also accepts legacy `skills/yolo26/model.onnx`)
///
/// If the requested variant isn't installed we fall back to ANY installed
/// variant — so the user keeps detection working while a heavier/lighter
/// tier downloads. `None` only if nothing is installed at all.
pub(crate) fn find_yolo26_model(data_dir: &std::path::Path, variant: &str) -> Option<std::path::PathBuf> {
    let order = match variant {
        // Try the requested tier first, then walk outward (smallest → largest).
        "nano"   => ["yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x"],
        "small"  => ["yolo26s", "yolo26n", "yolo26m", "yolo26l", "yolo26x"],
        "medium" => ["yolo26m", "yolo26s", "yolo26l", "yolo26n", "yolo26x"],
        "large"  => ["yolo26l", "yolo26m", "yolo26x", "yolo26s", "yolo26n"],
        _        => ["yolo26x", "yolo26l", "yolo26m", "yolo26s", "yolo26n"], // xlarge / unknown
    };
    let skills = data_dir.join("skills");
    for id in &order {
        let p = skills.join(id).join("model.onnx");
        if p.exists() { return Some(p); }
    }
    // Legacy fallback: pre-tier installs that landed in `skills/yolo26/`.
    let legacy = skills.join("yolo26").join("model.onnx");
    if legacy.exists() { return Some(legacy); }
    None
}

/// YOLO26 output decoder. Handles BOTH common ONNX export layouts:
///   • Ultralytics native:    [1, 84, 8400]  (channels-first) — used by `yolo export format=onnx`
///   • Channels-last:         [1, 8400, 84]  (some converters)
/// `shape` is the actual output shape from ORT.
///
/// Rows 0-3 = cx,cy,w,h in **PIXEL coordinates of 640×640 input** (NOT normalized).
/// Rows 4-83 = 80 COCO class scores, already sigmoid-activated (0-1).
pub(crate) fn decode_yolo_output(data: &[f32], shape: &[i64], orig_w: f32, orig_h: f32, conf: f32)
    -> Vec<(String, f32, [f32; 4])>
{
    const NAMES: &[&str] = &[
        "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
        "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
        "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
        "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
        "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
        "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
        "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
        "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
        "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
        "book","clock","vase","scissors","teddy bear","hair drier","toothbrush",
    ];

    // Determine layout from shape:
    //   [1, 84, N]   → channels-first: data[c*N + a]
    //   [1, N, 84]   → channels-last:  data[a*84 + c]
    let (n_anchors, channels_first) = match shape {
        [1, c, n] if *c == 84 => (*n as usize, true),
        [1, n, c] if *c == 84 => (*n as usize, false),
        // Single batch dimension omitted (sometimes seen)
        [c, n] if *c == 84 => (*n as usize, true),
        [n, c] if *c == 84 => (*n as usize, false),
        _ => return Vec::new(), // unknown layout — bail
    };

    // YOLO11/26 output bbox in pixels of 640x640 input — need to scale to original
    let scale_x = orig_w / 640.0;
    let scale_y = orig_h / 640.0;

    let mut dets: Vec<(String, f32, [f32; 4])> = Vec::new();
    let read = |c: usize, a: usize| -> f32 {
        if channels_first { data[c * n_anchors + a] }
        else              { data[a * 84 + c] }
    };

    for a in 0..n_anchors {
        let cx = read(0, a);
        let cy = read(1, a);
        let w  = read(2, a);
        let h  = read(3, a);

        // Find highest-confidence class
        let mut best_score = 0.0f32;
        let mut best_class = 0usize;
        for c in 4..84 {
            let s = read(c, a);
            if s > best_score { best_score = s; best_class = c - 4; }
        }

        if best_score < conf { continue; }
        let label = NAMES.get(best_class).copied().unwrap_or("unknown").to_string();

        let x1 = ((cx - w / 2.0) * scale_x).max(0.0);
        let y1 = ((cy - h / 2.0) * scale_y).max(0.0);
        let x2 = ((cx + w / 2.0) * scale_x).min(orig_w);
        let y2 = ((cy + h / 2.0) * scale_y).min(orig_h);
        dets.push((label, best_score, [x1, y1, x2, y2]));
    }

    // Simple greedy NMS by IoU — keeps highest-score box, drops overlapping duplicates
    dets.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut kept: Vec<(String, f32, [f32; 4])> = Vec::new();
    for d in dets {
        let mut suppress = false;
        for k in &kept {
            if k.0 == d.0 && iou(&k.2, &d.2) > 0.5 { suppress = true; break; }
        }
        if !suppress { kept.push(d); }
        if kept.len() >= 50 { break; }
    }
    kept
}

/// DETR-style decoder for the `onnx-community/yolo26{n,s,m,l,x}-ONNX` exports.
/// These models emit TWO output tensors instead of one:
///   • `logits` — shape `[1, 300, 80]`, raw class logits (pre-sigmoid).
///   • `pred_boxes` — shape `[1, 300, 4]`, normalised cx,cy,w,h in `[0, 1]`.
///
/// Reference: edge-AI NVRs `lib/env_config.py::_OnnxCoreMLModel.__call__` — same
/// math here, just in Rust. Sigmoid the logits, argmax per proposal for the
/// class, un-normalise the box back to original image pixels, NMS.
///
/// Returns the same `Vec<(label, conf, [x1,y1,x2,y2])>` as the classic decoder
/// so callers don't have to branch beyond format detection.
pub(crate) fn decode_yolo_output_detr(
    logits: &[f32], logits_shape: &[i64],
    boxes: &[f32], boxes_shape: &[i64],
    orig_w: f32, orig_h: f32, conf: f32,
) -> Vec<(String, f32, [f32; 4])> {
    const NAMES: &[&str] = &[
        "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
        "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
        "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
        "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
        "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
        "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
        "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
        "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
        "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
        "book","clock","vase","scissors","teddy bear","hair drier","toothbrush",
    ];

    // Expected shapes (batch dim may be elided):
    //   logits: [1, N, 80] or [N, 80]
    //   boxes:  [1, N, 4]  or [N, 4]
    let (n_props, n_classes) = match logits_shape {
        [1, n, c] => (*n as usize, *c as usize),
        [n, c]    => (*n as usize, *c as usize),
        _ => return Vec::new(),
    };
    let n_boxes = match boxes_shape {
        [1, n, 4] => *n as usize,
        [n, 4]    => *n as usize,
        _ => return Vec::new(),
    };
    if n_props != n_boxes || n_classes < 1 { return Vec::new(); }
    if logits.len() < n_props * n_classes || boxes.len() < n_props * 4 {
        return Vec::new();
    }

    // We preprocessed at 640×640 with letterbox preserved (resize_exact, not
    // padded). Until we add padding, scale boxes by orig/640 — matches the
    // existing classic decoder's assumption.
    let scale_x = orig_w;
    let scale_y = orig_h;

    let mut dets: Vec<(String, f32, [f32; 4])> = Vec::new();
    for i in 0..n_props {
        // Per-proposal class probabilities — sigmoid then argmax.
        let off = i * n_classes;
        let mut best_score = 0.0f32;
        let mut best_class = 0usize;
        for c in 0..n_classes.min(NAMES.len()) {
            let z = logits[off + c];
            // Stable sigmoid.
            let p = if z >= 0.0 { 1.0 / (1.0 + (-z).exp()) } else { let e = z.exp(); e / (1.0 + e) };
            if p > best_score { best_score = p; best_class = c; }
        }
        if best_score < conf { continue; }

        let bo = i * 4;
        let cx = boxes[bo]     * scale_x;
        let cy = boxes[bo + 1] * scale_y;
        let bw = boxes[bo + 2] * scale_x;
        let bh = boxes[bo + 3] * scale_y;

        let label = NAMES.get(best_class).copied().unwrap_or("unknown").to_string();
        let x1 = (cx - bw / 2.0).max(0.0);
        let y1 = (cy - bh / 2.0).max(0.0);
        let x2 = (cx + bw / 2.0).min(orig_w);
        let y2 = (cy + bh / 2.0).min(orig_h);
        dets.push((label, best_score, [x1, y1, x2, y2]));
    }

    // Same NMS shape as the classic decoder.
    dets.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut kept: Vec<(String, f32, [f32; 4])> = Vec::new();
    for d in dets {
        let mut suppress = false;
        for k in &kept {
            if k.0 == d.0 && iou(&k.2, &d.2) > 0.5 { suppress = true; break; }
        }
        if !suppress { kept.push(d); }
        if kept.len() >= 50 { break; }
    }
    kept
}

/// A named zone polygon (standard). Zones don't suppress anything — they
/// tag detections with the named region they fell in, so the agent can say
/// "entered front_porch" instead of generic "in the frame".
#[derive(Debug, Clone)]
pub(crate) struct NamedZone {
    pub name:    String,
    pub polygon: Vec<(f32, f32)>,
}

/// Internal: parse all polygons of a given `type` for a camera. Empty when
/// the JSON is empty / unparseable / no entries for the cam.
fn parse_polys_of_type(camera_masks_json: &str, cam_id: u8, type_filter: &str) -> Vec<Vec<(f32, f32)>> {
    if camera_masks_json.is_empty() { return Vec::new(); }
    let Ok(map) = serde_json::from_str::<serde_json::Value>(camera_masks_json) else { return Vec::new() };
    let Some(arr) = map.get(cam_id.to_string()).and_then(|v| v.as_array()) else { return Vec::new() };
    let mut polys: Vec<Vec<(f32, f32)>> = Vec::new();
    for mask in arr {
        if mask.get("type").and_then(|t| t.as_str()) != Some(type_filter) { continue; }
        let Some(pts_str) = mask.get("points").and_then(|p| p.as_str()) else { continue };
        let nums: Vec<f32> = pts_str.split(',').filter_map(|s| s.trim().parse().ok()).collect();
        if nums.len() < 6 { continue; }
        polys.push(nums.chunks(2).filter(|c| c.len() == 2).map(|c| (c[0], c[1])).collect());
    }
    polys
}

/// Parse motion-mask polygons (`type == "motion"`) — applied to raw motion
/// pixels in `compute_motion_masked`. Mature NVRs' `motion: mask:`.
pub(crate) fn parse_motion_masks_for_cam(camera_masks_json: &str, cam_id: u8) -> Vec<Vec<(f32, f32)>> {
    parse_polys_of_type(camera_masks_json, cam_id, "motion")
}

/// Parse object-mask polygons (`type == "object"`) — applied to YOLO
/// detections via `filter_detections_by_masks`. Mature NVRs'
/// `objects: filters: <label>: mask:`. Separate from motion masks so users
/// can suppress YOLO false-positives without also blinding the motion sensor.
pub(crate) fn parse_object_masks_for_cam(camera_masks_json: &str, cam_id: u8) -> Vec<Vec<(f32, f32)>> {
    parse_polys_of_type(camera_masks_json, cam_id, "object")
}

/// Parse named zones (`type == "zone"`) — informational only. Used by
/// `analyze_event_clip` to tag motion events with zones-entered metadata
/// and by the agent's clip-analysis context. Zones do NOT suppress motion
/// or detections.
pub(crate) fn parse_zones_for_cam(camera_masks_json: &str, cam_id: u8) -> Vec<NamedZone> {
    if camera_masks_json.is_empty() { return Vec::new(); }
    let Ok(map) = serde_json::from_str::<serde_json::Value>(camera_masks_json) else { return Vec::new() };
    let Some(arr) = map.get(cam_id.to_string()).and_then(|v| v.as_array()) else { return Vec::new() };
    let mut zones: Vec<NamedZone> = Vec::new();
    for mask in arr {
        if mask.get("type").and_then(|t| t.as_str()) != Some("zone") { continue; }
        let name = mask.get("name").and_then(|n| n.as_str()).unwrap_or("zone").to_string();
        let Some(pts_str) = mask.get("points").and_then(|p| p.as_str()) else { continue };
        let nums: Vec<f32> = pts_str.split(',').filter_map(|s| s.trim().parse().ok()).collect();
        if nums.len() < 6 { continue; }
        let polygon: Vec<(f32, f32)> = nums.chunks(2).filter(|c| c.len() == 2)
            .map(|c| (c[0], c[1])).collect();
        zones.push(NamedZone { name, polygon });
    }
    zones
}

/// Drop detections whose bbox bottom-center falls inside any mask polygon.
/// Matches mature NVRs' object-filter-mask semantics. Bbox is pixel coords against orig_w/orig_h.
pub(crate) fn filter_detections_by_masks(
    dets: Vec<(String, f32, [f32; 4])>,
    polys: &[Vec<(f32, f32)>],
    orig_w: f32,
    orig_h: f32,
) -> Vec<(String, f32, [f32; 4])> {
    if polys.is_empty() { return dets; }
    dets.into_iter().filter(|(_, _, b)| {
        let cx_norm = (b[0] + b[2]) * 0.5 / orig_w.max(1.0);
        let by_norm = b[3] / orig_h.max(1.0); // bottom-center
        // Drop if bottom-center is inside ANY mask polygon
        !polys.iter().any(|poly| is_point_in_polygon(cx_norm, by_norm, poly))
    }).collect()
}

pub(crate) fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = a[2].min(b[2]);
    let y2 = a[3].min(b[3]);
    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area_a = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let area_b = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    let union = area_a + area_b - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

/// How often the arm loop re-checks for a newly installed detector model.
/// A directory stat; cheap enough to run forever, short enough that installing
/// the skill feels immediate.
const MODEL_POLL_INTERVAL: std::time::Duration = std::time::Duration::from_secs(15);

/// Backoff between ONNX session build attempts after a load failure.
const SESSION_RETRY_INTERVAL: std::time::Duration = std::time::Duration::from_secs(30);

/// Background task: reads frames from the watch channel, runs YOLO26 inference,
/// stores results in scene_objects, and emits `detections:update` to frontend.
/// Waits for the YOLO26 skill to be installed rather than exiting, so installing
/// it at any point after boot arms detection without an app restart.
pub async fn run_inference_loop(state: Arc<AppState>) {
    // Give the app a moment to start before loading the model
    tokio::time::sleep(std::time::Duration::from_secs(10)).await;

    // ARM LOOP — do not `return` when the model is missing.
    //
    // boot.rs spawns this task exactly once, so returning here meant a user who
    // installed YOLO from Arsenal at any point after boot got no detection at all
    // until they restarted the app, with nothing in the UI saying so. That is
    // precisely what first run looks like: the model is installed *after* launch.
    //
    // Polling a directory every 15 s costs a stat. The settings read is inside the
    // loop on purpose so switching variant in Settings is picked up too.
    let model_path = {
        let mut announced = false;
        loop {
            let variant = state.settings.read().await.yolo_variant.clone();
            if let Some(p) = find_yolo26_model(&state.data_dir, &variant) { break p; }
            if !announced {
                announced = true;
                tracing::info!("YOLO26 skill not installed — watching for it. Install via Agent → Skills.");
                // v9: mirror to shared pull-state so the frontend can read it on mount.
                {
                    let mut s = state.inference_status.write().await;
                    s.state = "not_installed".to_string();
                    s.variant = None;
                    s.fps = 0;
                    s.since = Some(chrono::Utc::now().to_rfc3339());
                }
                state.app_handle.emit("inference:status", serde_json::json!({
                    "status": "not_installed"
                })).ok();
            }
            tokio::time::sleep(MODEL_POLL_INTERVAL).await;
        }
    };

    let variant = model_path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("model.onnx")
        .to_string();

    tracing::info!("YOLO26: loading {:?}…", model_path);
    {
        let mut s = state.inference_status.write().await;
        s.state = "loading".to_string();
        s.variant = Some(variant.clone());
        s.since = Some(chrono::Utc::now().to_rfc3339());
    }
    state.app_handle.emit("inference:status", serde_json::json!({
        "status": "loading", "variant": variant
    })).ok();

    // Build ORT session — Microsoft ONNX Runtime, full operator support for YOLO26x
    // Same reasoning as the arm loop above: a failed session build used to end the
    // task permanently. A model still finishing its download, or a GPU busy with
    // another process, both recover on their own — so retry rather than retire.
    let mut session = loop {
    match build_ort_session(&model_path) {
        Ok(sess) => {
            tracing::info!("YOLO26 ready — {} loaded via ORT (CPU)", variant);
            {
                let mut s = state.inference_status.write().await;
                s.state = "ready".to_string();
                s.variant = Some(variant.clone());
                s.since = Some(chrono::Utc::now().to_rfc3339());
            }
            state.app_handle.emit("inference:status", serde_json::json!({
                "status": "ready", "variant": variant, "backend": "ort"
            })).ok();
            // Keep legacy event for backwards compat
            state.app_handle.emit("inference:ready", serde_json::json!({
                "backend": "ort", "model": "yolo26", "variant": variant
            })).ok();
            break sess;
        }
        Err(e) => {
            let msg = e.to_string();
            tracing::warn!("YOLO26 failed to load — {msg}. Retrying in {}s.",
                           SESSION_RETRY_INTERVAL.as_secs());
            {
                let mut s = state.inference_status.write().await;
                s.state = "error".to_string();
                s.since = Some(chrono::Utc::now().to_rfc3339());
            }
            state.app_handle.emit("inference:status", serde_json::json!({
                "status": "error", "message": msg
            })).ok();
            tokio::time::sleep(SESSION_RETRY_INTERVAL).await;
        }
    }
    };

    // Discover the model's actual input tensor name — never assume "images".
    // Different YOLO export tools use different names ("images", "input", "input.1", etc.).
    let input_name: String = session.inputs().first()
        .map(|i| i.name().to_string())
        .unwrap_or_else(|| "images".to_string());
    tracing::info!(
        "YOLO26 model — input[0]={:?}  outputs={}",
        input_name, session.outputs().len()
    );

    let infer_q = Arc::clone(&state.infer_queue);
    let mut rr_cursor: usize = 0;
    // Heartbeat: emit inference:tick every ~5s so the UI knows the engine is alive
    let mut tick_frames: u32 = 0;
    let mut tick_last = std::time::Instant::now();

    // v9: wall-clock heartbeat so the badge resolves even when no camera is
    // pushing frames yet. Re-emits `inference:status:ready` every 5s using the
    // current shared snapshot; CameraView listeners that mounted late catch up.
    let state_for_heartbeat = state.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(5));
        interval.tick().await; // first tick fires immediately; skip it
        loop {
            interval.tick().await;
            let snap = state_for_heartbeat.inference_status.read().await.clone();
            if snap.state == "ready" {
                state_for_heartbeat.app_handle.emit("inference:status", serde_json::json!({
                    "status": "ready",
                    "variant": snap.variant,
                    "backend": "ort"
                })).ok();
            }
        }
    });

    loop {
        // Fair intake: drain per-camera slots round-robin so every active camera
        // gets detector time (per-cam rate limiting happens at push — see
        // InferQueue). The notified future is created BEFORE the empty check so
        // a push landing in between still wakes us (no lost wakeup).
        let notified = infer_q.notify.notified();
        let Some((cam_id, frame)) = infer_q.pop_rr(&mut rr_cursor) else {
            notified.await;
            continue;
        };

        tick_frames += 1;
        // Emit a heartbeat every 5 seconds so the UI can confirm the engine is alive
        if tick_last.elapsed().as_secs() >= 5 {
            let fps = tick_frames / tick_last.elapsed().as_secs().max(1) as u32;
            // v9: also write fps to the shared pull-state so the UI can read it via
            // get_inference_status without depending on the event firing.
            {
                let mut s = state.inference_status.write().await;
                s.fps = fps;
            }
            state.app_handle.emit("inference:tick", serde_json::json!({
                "fps": fps, "cam_id": cam_id
            })).ok();
            tick_frames = 0;
            tick_last   = std::time::Instant::now();
        }

        // Decode JPEG → 640×640 CHW float32 Vec
        let input_data = match preprocess_jpeg_for_yolo(&frame) {
            Ok(d) => d,
            Err(_) => continue,
        };

        // Original image dimensions for bbox rescaling
        let (orig_w, orig_h) = match image::load_from_memory(&frame) {
            Ok(img) => (img.width() as f32, img.height() as f32),
            Err(_)  => (640.0, 640.0),
        };

        // Create ORT tensor from raw Vec<f32> — official tuple API, no ndarray required:
        //   Tensor::from_array(([shape...], vec_data))
        let tensor = match ort::value::Tensor::<f32>::from_array(([1usize, 3, 640, 640], input_data)) {
            Ok(t) => t,
            Err(e) => { tracing::warn!("YOLO26 tensor create error: {e}"); continue; }
        };
        // Use the runtime-discovered input name — never hard-code "images"
        let ort_inputs = ort::inputs![input_name.as_str() => tensor];
        // Guards scoped to the run itself: the timer and the DirectML serialising
        // lock drop as this block ends, before the match arms run. (Written as a
        // block bound to a name rather than a block-in-match-scrutinee, which is
        // both a clippy lint and something  cannot parse.)
        let run_result = {
            let _t = infer_timer("yolo");
            let _gpu = gpu_infer_guard();
            session.run(ort_inputs)
        };
        let outputs = match run_result {
            Ok(o) => o,
            Err(e) => { tracing::warn!("YOLO26 inference error: {e}"); continue; }
        };
        // try_extract_tensor returns (&Shape, &[f32]) — destructure tuple, no .view() needed
        let (raw_shape, raw): (Vec<i64>, Vec<f32>) = match outputs[0].try_extract_tensor::<f32>() {
            Ok((shape, data)) => (shape.iter().copied().collect(), data.to_vec()),
            Err(e) => { tracing::warn!("YOLO26 output extraction error: {e}"); continue; }
        };

        // One-shot diagnostic: log the output tensor shape + score statistics on
        // first run so we can see whether the model is producing usable output.
        static FIRST_RUN: std::sync::Once = std::sync::Once::new();
        FIRST_RUN.call_once(|| {
            let max_score = raw.iter().cloned().fold(f32::MIN, f32::max);
            let min_score = raw.iter().cloned().fold(f32::MAX, f32::min);
            tracing::info!(
                "YOLO26 first-frame diagnostics: outputs={}  shape0={:?}  raw_len={}  min={:.4}  max={:.4}",
                outputs.len(), raw_shape, raw.len(), min_score, max_score
            );
        });

        // ── Format detection ─────────────────────────────────────────────────
        // The onnx-community YOLO 2026 exports use a DETR-style head: TWO
        // output tensors (logits [1, 300, 80] + pred_boxes [1, 300, 4]
        // normalised). Ultralytics-native single-tensor exports keep the
        // classic [1, 84, 8400] layout. Branch by output count + class-dim
        // sniff so we don't have to track the model variant separately.
        let is_detr_layout = outputs.len() >= 2
            && raw_shape.last().copied() == Some(80);

        // ── User-tunable confidence + class-filter snapshot (v7) ─────────────
        // Read both at once so we hold the settings lock minimally. Class filter
        // is a comma-separated whitelist; empty → no filter.
        let (yolo_conf, class_filter_csv) = {
            let s = state.settings.read().await;
            (s.yolo_confidence_threshold, s.yolo_class_filter.clone())
        };
        // Clamp the threshold to sane bounds in case settings.json was edited by hand.
        let yolo_conf = yolo_conf.clamp(0.05, 0.95);
        let class_filter: Option<std::collections::HashSet<String>> = if class_filter_csv.trim().is_empty() {
            None
        } else {
            Some(class_filter_csv.split(',').map(|s| s.trim().to_ascii_lowercase())
                .filter(|s| !s.is_empty()).collect())
        };

        let detections = if is_detr_layout {
            // Second output is the box tensor.
            let (boxes_shape, boxes): (Vec<i64>, Vec<f32>) =
                match outputs[1].try_extract_tensor::<f32>() {
                    Ok((shape, data)) => (shape.iter().copied().collect(), data.to_vec()),
                    Err(e) => { tracing::warn!("YOLO26 box-output extraction error: {e}"); continue; }
                };
            decode_yolo_output_detr(&raw, &raw_shape, &boxes, &boxes_shape, orig_w, orig_h, yolo_conf)
        } else {
            // The decoder applies NMS (IoU>0.5) and caps at 50 boxes per frame, so spam is bounded.
            decode_yolo_output(&raw, &raw_shape, orig_w, orig_h, yolo_conf)
        };

        // ── Apply object masks to YOLO detections ────────────────────────────
        // v8: standard separation. `object` masks suppress YOLO detections
        // (their bottom-center inside the polygon → drop). `motion` masks
        // suppress raw motion pixels in `compute_motion_masked` and no longer
        // touch YOLO. Backwards-compat: if a camera has no object masks drawn,
        // we fall back to its motion masks so users who already configured
        // "mask the bench" keep their suppression. Once they draw a dedicated
        // object mask the fallback is disabled per-camera.
        let camera_masks_json = state.settings.read().await.camera_masks.clone();
        let object_polys = parse_object_masks_for_cam(&camera_masks_json, cam_id);
        let mask_polys = if object_polys.is_empty() {
            parse_motion_masks_for_cam(&camera_masks_json, cam_id)
        } else {
            object_polys
        };
        let detections = filter_detections_by_masks(detections, &mask_polys, orig_w, orig_h);

        // ── Class filter (v7) ───────────────────────────────────────────────
        // Drop any detection whose label isn't in the user's whitelist. Matches
        // case-insensitively against COCO class names (e.g. `"person,car"`).
        let detections: Vec<(String, f32, [f32; 4])> = if let Some(ref filter) = class_filter {
            detections.into_iter()
                .filter(|(label, _, _)| filter.contains(&label.to_ascii_lowercase()))
                .collect()
        } else {
            detections
        };

        // ── Body Re-ID: appearance descriptor per detected person ────────────────
        // OSNet (or HSV fallback) embedding — feeds BOTH the StrongSORT-style tracker
        // (appearance association) AND the tracklet-consistent identity step after
        // tracking. No DB work here; matching/storage is keyed to the tracklet below.
        let (reid_threshold, face_rec_threshold) = {
            let s = state.settings.read().await;
            (s.reid_threshold, s.face_recognition_threshold)
        };
        let reid_deep_floor = crate::reid::active_deep_floor(&state.data_dir);
        let mut descriptors: Vec<Option<Vec<f32>>> = Vec::with_capacity(detections.len());
        for (label, _score, box_) in &detections {
            descriptors.push(if label == "person" {
                body_descriptor(&state.data_dir, &frame, box_)
            } else { None });
        }

        // ── Continuous face capture (mature NVRs `save_attempts`) — throttled per camera ──
        // Populates People → Train and recognises known people live, server-side, even
        // with no UI open. No-op unless a face model is installed (gated inside
        // recognize_faces via active_face_tier). ~3 s/camera when a person is present.
        // Throttle face work to ~3 s/cam (shared by the Train-capture pass AND the
        // face↔body fusion pass below, so the face model runs at most twice per tick).
        let face_due = detections.iter().any(|(l, _, _)| l == "person") && {
            let cell = FACE_CAPTURE_THROTTLE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
            match cell.lock() {
                Ok(mut m) => {
                    let now = std::time::Instant::now();
                    let d = m.get(&cam_id).map_or(true, |t| now.duration_since(*t).as_secs() >= 3);
                    if d { m.insert(cam_id, now); }
                    d
                }
                Err(_) => false,
            }
        };
        if detections.iter().any(|(l, _, _)| l == "person")
            && face_due {
                // Only capture faces that sit on a detected person, and crop the
                // PERSON box (Tracked-style) for the Train thumbnail.
                let person_boxes: Vec<[f32; 4]> = detections.iter()
                    .filter(|(l, _, _)| l == "person")
                    .map(|(_, _, b)| *b)
                    .collect();
                let _ = crate::face::recognize_faces(&state, &frame, cam_id, None, &person_boxes).await;
                // Bound stranger growth (save_attempts): keep the most-recent ~150
                // unknowns so the Train tab stays a readable, recent set (crops are
                // now area-gated + de-duped in recognize_faces, so this is plenty).
                let _ = sqlx::query(
                    "DELETE FROM face_embeddings WHERE person_id IS NULL AND id NOT IN \
                     (SELECT id FROM face_embeddings WHERE person_id IS NULL ORDER BY seen_at DESC LIMIT 150)"
                ).execute(&state.db).await;
            }

        // Store in scene_objects for Guardian agent
        let objects: Vec<SceneObject> = detections.iter()
            .map(|(label, score, _)| SceneObject { label: label.clone(), score: *score })
            .collect();
        let now_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();
        state.scene_objects.write().await.insert(cam_id, objects.clone());
        state.scene_last_update.write().await.insert(cam_id, now_secs);

        // ── Multi-object tracking (foundation for speed + line-crossing) ─────────
        // Assigns a stable track_id per detection + records normalised bottom-centre
        // history. Then evaluate any speed zones + line tripwires for this camera.
        let tracked = crate::tracking::update(cam_id, &detections, &descriptors, orig_w, orig_h);
        evaluate_zone_analytics(&state, cam_id, &tracked).await;
        evaluate_loitering(&state, cam_id, &detections, &tracked).await;

        // ── Tracklet-consistent body identity (B) ───────────────────────────────
        // One body identity per tracklet — query_body_reid ONCE when a track first
        // appears (cross-camera aware), cache it for the tracklet, and grow the
        // gallery periodically (not every frame). Stabilises person_id in events +
        // the live overlay and removes a per-frame DB scan per person.
        let mut person_ids: Vec<Option<String>> = Vec::with_capacity(detections.len());
        // Per-frame identity uniqueness for BODIES (mirror of the face pass's
        // duplicate_identity_indices): one known person cannot be two bodies in one
        // frame. First claim (cached or higher up the loop) wins; a later FRESH
        // query resolving to an already-claimed kp_ gets an anonymous id instead.
        let mut claimed_kp: std::collections::HashSet<String> = std::collections::HashSet::new();
        for (i, t) in tracked.iter().enumerate() {
            if detections.get(i).is_some_and(|(l, _, _)| l == "person") {
                if let Some((p, _)) = track_body_id_get(cam_id, t.track_id) {
                    if p.starts_with("kp_") { claimed_kp.insert(p); }
                }
            }
        }
        for (i, (label, score, box_)) in detections.iter().enumerate() {
            if label != "person" { person_ids.push(None); continue; }
            let Some(desc) = descriptors.get(i).and_then(|d| d.clone()) else { person_ids.push(None); continue; };
            let track_id = tracked.get(i).map(|t| t.track_id).unwrap_or(0);
            // QUALITY GATE: only a reliable crop is trusted for cross-camera identity +
            // the durable gallery. A tracklet's identity is DEFERRED until its first
            // reliable crop, so a tiny/blurry first frame can't lock in a wrong match.
            let reliable = crate::reid::crop_is_reliable(box_, *score);
            let pid: Option<(String, Option<f32>)> = match track_body_id_get(cam_id, track_id) {
                Some(p) => Some(p),
                None if reliable => {
                    let matched = query_body_reid(&state.db, &desc, reid_threshold + 0.35, reid_deep_floor).await;
                    let (p, mscore) = match matched {
                        // Same-frame uniqueness: this known person is already embodied
                        // by another track in this frame — two simultaneous bodies on
                        // one camera are two people, so the weaker claim stays anonymous.
                        Some((p, _)) if p.starts_with("kp_") && claimed_kp.contains(&p) => {
                            tracing::info!("cam{cam_id}: body match {p} already claimed in this frame — keeping track {track_id} anonymous");
                            (format!("body_{}", &uuid::Uuid::new_v4().to_string()[..8]), None)
                        }
                        Some((p, s)) => (p, Some(s)),
                        None => (format!("body_{}", &uuid::Uuid::new_v4().to_string()[..8]), None),
                    };
                    if p.starts_with("kp_") { claimed_kp.insert(p.clone()); }
                    track_body_id_set(cam_id, track_id, &p, mscore);
                    Some((p, mscore))
                }
                None => None, // no reliable crop yet → don't commit an identity
            };
            // Grow the gallery ONLY from reliable crops (~1 s/track, bounded to 20).
            if let Some((pid, mscore)) = &pid {
                if reliable && track_store_due(cam_id, track_id) {
                    let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM body_embeddings WHERE person_id=?")
                        .bind(pid).fetch_one(&state.db).await.unwrap_or(0);
                    if count < 20 {
                        let thumb = if count == 0 { crop_person_jpeg(&frame, box_) } else { None };
                        // Clothing colors (torso/legs dominant HSV, honest abstain) —
                        // gives the track a human-readable outfit line in the UI.
                        let attrs = crate::reid::classify_person_colors(&frame, box_);
                        // Provenance: a kp_ identity here came from a body-Re-ID match.
                        let prov = pid.starts_with("kp_").then_some(("body_reid", *mscore));
                        store_body_embedding(&state.db, pid, &desc, cam_id, None, thumb.as_deref(), attrs.as_deref(), prov).await;
                    }
                }
            }
            person_ids.push(pid.map(|(p, _)| p));
        }
        // Hourly hygiene: prune stale anonymous fragments, then reconcile ID-SPLITS.
        // (Body-based AUTO-promotion was removed: clothing similarity must never
        // COMMIT a name — industry consensus (mature NVRs/GEFF). Body matches surface
        // as one-tap SUGGESTIONS in People → Tracked instead; only a face or a
        // human commits an identity.)
        if body_prune_due() {
            crate::reid::prune_anon_bodies(&state.db).await;
            // Reconcile ID-SPLITS: fold anonymous track fragments of the SAME unknown
            // person back into one (usearch + Chinese Whispers, strict floor,
            // spatio-temporal veto). Anonymous-only — never touches named identities.
            crate::reid::reconcile_anon_tracks(&state.db, &state.data_dir).await;
            // Face recognition-log retention: newest 500 linked crops per person
            // (user-approved standard hygiene; enrolled identities untouched).
            // DETACHED: the first prune after a bloated history removes thousands
            // of rows + blob files — the detection tick must not wait on it.
            tokio::spawn(crate::persons::prune_linked_face_crops(
                state.db.clone(), state.data_dir.clone()));
        }

        // ── Face↔body fusion (Avigilon/BriefCam-style auto-label) ───────────────
        // When a face is recognised up close, attribute its NAME to the person box
        // it sits in (→ that box's body descriptor + tracklet), so the body can be
        // recognised LATER at a distance / on another camera. READ-ONLY w.r.t. face
        // recognition: `locate_faces` reuses the same recognizer, no face path changed.
        // Commit only after the same (track → name) pairing is confirmed on ≥2 ticks,
        // so a single misrecognition can't poison a person's appearance gallery.
        if face_due {
            let located = crate::face::locate_faces(&state, &frame).await;
            // Owner map: which person box does EACH located face (named or not)
            // sit in? A box containing ≥2 faces has an AMBIGUOUS owner — someone
            // holding a photo/phone of a face, or two people in a tight embrace —
            // and fusing a name into that body's gallery would poison it.
            let owner_of = |bbox: &[f32; 4]| -> Option<usize> {
                let fcx = (bbox[0] + bbox[2]) / 2.0;
                let fcy = (bbox[1] + bbox[3]) / 2.0;
                let mut best: Option<usize> = None;
                let mut best_area = f32::MAX;
                for (i, (label, _, b)) in detections.iter().enumerate() {
                    if label != "person" { continue; }
                    if fcx >= b[0] && fcx <= b[2] && fcy >= b[1] && fcy <= b[3] {
                        let area = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
                        if area < best_area { best_area = area; best = Some(i); }
                    }
                }
                best
            };
            let mut faces_per_box: std::collections::HashMap<usize, u32> = std::collections::HashMap::new();
            for lf in &located {
                if let Some(i) = owner_of(&lf.bbox) { *faces_per_box.entry(i).or_default() += 1; }
            }
            for lf in &located {
                if lf.person_id.is_empty() || lf.name == "unknown" { continue; }
                // Smallest person box whose area contains the face centre = the body it belongs to.
                let Some(idx) = owner_of(&lf.bbox) else { continue };
                // Ambiguous owner (≥2 faces in this one person box) → skip this tick.
                if faces_per_box.get(&idx).copied().unwrap_or(0) >= 2 { continue; }
                // Only fuse a RELIABLE body crop into the named gallery — a tiny/blurry
                // crop here would poison that person's appearance gallery.
                if !crate::reid::crop_is_reliable(&detections[idx].2, detections[idx].1) { continue; }
                let Some(desc) = descriptors.get(idx).and_then(|d| d.clone()) else { continue };
                let track_id = tracked.get(idx).map(|t| t.track_id).unwrap_or(0);
                // Tracker-coupled consensus before committing — the SAME score-aware
                // policy that names events (face::IdentityVote), not just "2 ticks of
                // any match": a sustained look-alike at 0.5x scores no longer poisons
                // the person's appearance gallery.
                if confirm_face_track(cam_id, track_id, &lf.person_id, lf.score, face_rec_threshold) {
                    // FACE OVERRIDES BODY: if this tracklet already carries a different
                    // identity (from a body match or an earlier bad fusion), the
                    // consensus-confirmed face wins — re-point the tracklet and
                    // re-attribute its recent samples, with negatives + provenance.
                    let kp_pid = format!("kp_{}", lf.person_id);
                    if let Some((cached, _)) = track_body_id_get(cam_id, track_id) {
                        if cached != kp_pid {
                            tracing::info!(
                                "cam{cam_id}: face '{}' (score {:.2}) contradicts track {track_id}'s body identity {cached} — face wins",
                                lf.name, lf.score
                            );
                            crate::reid::face_override_track(&state.db, &cached, &lf.person_id, cam_id, lf.score).await;
                            track_body_id_set(cam_id, track_id, &kp_pid, None);
                        }
                    } else {
                        // Track had no identity yet — anchor it to the face directly.
                        track_body_id_set(cam_id, track_id, &kp_pid, None);
                    }
                    let thumb = crop_person_jpeg(&frame, &detections[idx].2);
                    let attrs = crate::reid::classify_person_colors(&frame, &detections[idx].2);
                    crate::reid::link_body_to_known(&state.db, &lf.person_id, &desc, cam_id, thumb.as_deref(), attrs.as_deref(), lf.score).await;
                }
            }
        }

        // Emit to frontend — boxes + person_id from Re-ID + track_id (no frame data crosses IPC).
        // Bounding boxes stored as NORMALISED 0-1 coordinates (divided by frame dimensions)
        // so the annotation function can scale to any target image size regardless of
        // which resolution the camera was capturing at.
        let boxes: Vec<serde_json::Value> = detections.iter().zip(person_ids.iter()).zip(tracked.iter())
            .map(|(((label, score, box_), pid), td)| serde_json::json!({
                "label": label, "score": score,
                "person_id": pid,
                "track_id": td.track_id,
                "box": {
                    "xmin": box_[0] / orig_w,
                    "ymin": box_[1] / orig_h,
                    "xmax": box_[2] / orig_w,
                    "ymax": box_[3] / orig_h,
                }
            })).collect();
        state.app_handle.emit("detections:update",
            serde_json::json!({ "cam_id": cam_id, "detections": boxes })).ok();

        // Update latest_detections for live analysis (doesn't wait for event end)
        state.latest_detections.write().await.insert(cam_id, boxes.clone());

        // ── Detection Accumulator (mature NVRs ReviewSegment pattern) ────────────────
        // If a motion event is currently open for this camera, accumulate the
        // detections into a per-camera ring buffer.  One entry per object class
        // — we keep only the highest-confidence detection per label to bound DB size.
        // The buffer is flushed atomically to motion_events.detections when the
        // event closes, so no race between inference timing and event end.
        if !boxes.is_empty() {
            // Stationary-object grace window (mature NVRs stationary model): an object
            // only holds the event open while the scene is actually MOVING. Read
            // before locking cam_states so we don't await under that lock.
            let stationary_window = std::time::Duration::from_secs_f32(
                (state.settings.read().await.record_post_buffer_secs as f32).max(2.0)
            );
            let mut cs_map = state.cam_states.lock().await;
            let cs = cs_map.entry(cam_id).or_default();
            if cs.motion_active.is_some() {
                // v9: per-frame "any tracked class seen?" sentinel. Drives
                // both the standard close (reset object_absence_frames
                // when true; increment when false) and the activity-burst
                // detector (a new class shows up → fresh alert).
                let mut any_tracked_seen = false;
                // Confirmed tracked labels this tick (score >= confirm tier) —
                // used to classify the open event in REAL TIME (event_category +
                // dominant_label), standard, not only after clip analysis.
                let mut confirmed_labels: Vec<String> = Vec::new();
                for det in &boxes {
                    let label = det["label"].as_str().unwrap_or("").to_string();
                    let score = det["score"].as_f64().unwrap_or(0.0);
                    let is_tracked = matches!(label.as_str(),
                        "person" | "car" | "truck" | "bus" | "motorcycle" | "bicycle" |
                        "train" | "boat" | "airplane" |
                        "cat" | "dog" | "bird" | "horse" | "sheep" | "cow" |
                        "elephant" | "bear" | "zebra" | "giraffe" |
                        "backpack" | "suitcase" | "handbag"
                    );
                    // Mature NVRs two-tier: detections already passed min_score
                    // (yolo_confidence_threshold) upstream. Only a score above the
                    // higher CONFIRM tier actually confirms/classifies the event.
                    if is_tracked && score >= YOLO_CONFIRM_THRESHOLD as f64 {
                        cs.event_object_confirmed = true;
                        any_tracked_seen = true;
                        confirmed_labels.push(label.clone());
                        cs.classes_seen.insert(label.clone());
                    }
                    if let Some(existing) = cs.detection_buffer.iter_mut()
                        .find(|d| d["label"].as_str() == Some(label.as_str()))
                    {
                        if score > existing["score"].as_f64().unwrap_or(0.0) {
                            *existing = det.clone();
                        }
                    } else {
                        cs.detection_buffer.push(det.clone());
                    }
                }
                // Mature NVRs stationary model: a tracked object only HOLDS the event
                // open while the scene is moving. If an object is seen but motion
                // has been gone for `stationary_window`, treat it as STATIONARY
                // (parked car, still person, static false-positive) and let the
                // close timer climb — otherwise such objects kept events open
                // forever ("long events with nobody in frame"). Movement proxy:
                // recent hysteresis-confirmed motion (`last_motion_at`, set by the
                // lifecycle on sustained motion).
                let moving = cs.last_motion_at
                    .map(|t| t.elapsed() < stationary_window)
                    .unwrap_or(false);
                if any_tracked_seen && moving {
                    cs.last_object_seen_at = Some(std::time::Instant::now());
                    cs.object_absence_frames = 0;
                } else {
                    cs.object_absence_frames = cs.object_absence_frames.saturating_add(1);
                }
                // ── Early classification (real-time, mature NVRs per-label) ──────
                // Write event_category (bucket) + dominant_label (specific class)
                // onto the open row as soon as we have a confirmed object — so the
                // Review UI shows the real object immediately. Only on the first
                // confirm or when the dominant class changes, to bound DB writes.
                if !confirmed_labels.is_empty() {
                    let label_refs: Vec<&str> = confirmed_labels.iter().map(|s| s.as_str()).collect();
                    let category = crate::agent::categorise_labels(&label_refs);
                    let dominant = crate::agent::dominant_label(&label_refs);
                    if cs.last_dominant.as_deref() != Some(dominant.as_str()) && !dominant.is_empty() {
                        cs.last_dominant = Some(dominant.clone());
                        if let Some(eid) = cs.motion_active.clone() {
                            let db = state.db.clone();
                            // COALESCE keeps first_object_at write-once: it's set
                            // the first time any object is confirmed (the object's
                            // real onset) and never overwritten as the dominant
                            // class changes. footage_clip trims the empty
                            // motion→object lead-in to this timestamp.
                            let first_seen = chrono::Utc::now().to_rfc3339();
                            tokio::spawn(async move {
                                let _ = sqlx::query(
                                    "UPDATE motion_events SET event_category=?, dominant_label=?, first_object_at=COALESCE(first_object_at, ?)                                      WHERE id=? AND COALESCE(event_category,'') NOT IN ('audio','fall','crossing')")
                                    .bind(category).bind(&dominant).bind(&first_seen).bind(&eid)
                                    .execute(&db).await;
                            });
                        }
                    }
                }
                // (v28) The "🆕 New activity — X just appeared" burst alert was
                // removed — it fired for essentially every detection and spammed
                // the chat + Telegram. Real threat alerts come from clip analysis.
                // ── v9 mid-event re-analysis (long event → fresh agent alert) ─
                let re_interval = state.settings.read().await.re_analysis_interval_secs;
                if re_interval > 0 {
                    let should_reanalyse = cs.last_analysis_at
                        .map(|t| t.elapsed().as_secs() as u32 >= re_interval)
                        .unwrap_or_else(|| {
                            cs.event_opened_at
                                .map(|t| t.elapsed().as_secs() as u32 >= re_interval)
                                .unwrap_or(false)
                        });
                    if should_reanalyse {
                        cs.last_analysis_at = Some(std::time::Instant::now());
                        if let Some(event_id) = cs.motion_active.clone() {
                            let state_for_reanalyse = state.clone();
                            tokio::spawn(async move {
                                // analyze_event_clip already updates the row;
                                // this just makes the agent re-summarise mid-event.
                                crate::agent::analyze_event_clip(state_for_reanalyse, event_id).await;
                            });
                        }
                    }
                }
            }
        }
    }
}

// ─── Loitering (track-dwell, uses the tracker) ────────────────────────────────
//
// A loiter alert requires the SAME tracked person to remain in view for the
// configured dwell time (mature NVRs zone-loitering semantics: `loitering_time` is
// per-object dwell). The old check used the open motion EVENT's age — any
// long-running event (wind, a parked car holding it open) fired "person has
// been in frame for Ns" with no person present at all.

struct LoiterTrack { first: std::time::Instant, last: std::time::Instant, alerted: bool }

static LOITER_TRACKS: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<(u8, u64), LoiterTrack>>> =
    std::sync::OnceLock::new();

/// A track's dwell resets once it's been out of view this long.
const LOITER_GONE_SECS: u64 = 60;

async fn evaluate_loitering(
    state: &Arc<AppState>,
    cam_id: u8,
    detections: &[(String, f32, [f32; 4])],
    tracked: &[crate::tracking::TrackedDet],
) {
    let (on, thr_secs) = {
        let s = state.settings.read().await;
        (s.loitering_detection, s.loitering_threshold_secs)
    };
    if !on || thr_secs == 0 { return; }
    let now = std::time::Instant::now();
    let mut fire: Option<f32> = None;
    {
        let cell = LOITER_TRACKS.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
        let Ok(mut map) = cell.lock() else { return };
        map.retain(|_, t| now.duration_since(t.last).as_secs() < LOITER_GONE_SECS);
        for (i, td) in tracked.iter().enumerate() {
            // track_id 0 = the tracker's "unmatched" fallback — no continuity.
            if td.track_id == 0 { continue; }
            if detections.get(i).map_or(true, |(l, _, _)| l != "person") { continue; }
            let t = map.entry((cam_id, td.track_id))
                .or_insert(LoiterTrack { first: now, last: now, alerted: false });
            t.last = now;
            let dwell = now.duration_since(t.first).as_secs_f32();
            if !t.alerted && dwell >= thr_secs as f32 {
                t.alerted = true; // once per track — re-arms only via a fresh track
                fire = Some(dwell);
            }
        }
    }
    if let Some(dwell) = fire {
        tracing::warn!("LOITER: same person tracked ~{dwell:.0}s on cam{cam_id}");
        crate::agent::dispatch_intelligence_alert(
            state, "loitering",
            &format!("Same person has stayed in view for ~{dwell:.0}s"),
            cam_id, None,
        ).await;
    }
}

// ─── Zone analytics: speed + line-crossing (uses the tracker) ─────────────────

/// Speed zones for a camera: (name, normalised quad, image→metres homography).
fn parse_speed_zones_for_cam(masks_json: &str, cam_id: u8) -> Vec<(String, Vec<(f32, f32)>, [f32; 8])> {
    let mut out = Vec::new();
    let Ok(map) = serde_json::from_str::<serde_json::Value>(masks_json) else { return out };
    let Some(arr) = map.get(cam_id.to_string()).and_then(|v| v.as_array()) else { return out };
    for m in arr {
        if m.get("type").and_then(|t| t.as_str()) != Some("speed") { continue; }
        let name = m.get("name").and_then(|n| n.as_str()).unwrap_or("speed").to_string();
        let Some(pts) = m.get("points").and_then(|p| p.as_str()) else { continue };
        let nums: Vec<f32> = pts.split(',').filter_map(|s| s.trim().parse().ok()).collect();
        if nums.len() < 8 { continue; }
        let quad: Vec<(f32, f32)> = nums.chunks(2).take(4).filter(|c| c.len() == 2).map(|c| (c[0], c[1])).collect();
        if quad.len() != 4 { continue; }
        let w_m = m.get("width_m").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
        let h_m = m.get("height_m").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
        if w_m <= 0.0 || h_m <= 0.0 { continue; }
        let q4 = [quad[0], quad[1], quad[2], quad[3]];
        if let Some(h) = crate::tracking::homography(&q4, w_m, h_m) {
            out.push((name, quad, h));
        }
    }
    out
}

/// Line tripwires for a camera: (name, point a, point b) in normalised coords.
fn parse_lines_for_cam(masks_json: &str, cam_id: u8) -> Vec<(String, (f32, f32), (f32, f32))> {
    let mut out = Vec::new();
    let Ok(map) = serde_json::from_str::<serde_json::Value>(masks_json) else { return out };
    let Some(arr) = map.get(cam_id.to_string()).and_then(|v| v.as_array()) else { return out };
    for m in arr {
        if m.get("type").and_then(|t| t.as_str()) != Some("line") { continue; }
        let name = m.get("name").and_then(|n| n.as_str()).unwrap_or("line").to_string();
        let Some(pts) = m.get("points").and_then(|p| p.as_str()) else { continue };
        let nums: Vec<f32> = pts.split(',').filter_map(|s| s.trim().parse().ok()).collect();
        if nums.len() >= 4 { out.push((name, (nums[0], nums[1]), (nums[2], nums[3]))); }
    }
    out
}

/// For every tracked object this frame, update zone speed + line-crossing. Cheap
/// no-op when the camera has no speed/line zones.
async fn evaluate_zone_analytics(state: &Arc<AppState>, cam_id: u8, tracked: &[crate::tracking::TrackedDet]) {
    if tracked.is_empty() { return; }
    let (masks, speed_alert) = {
        let s = state.settings.read().await;
        (s.camera_masks.clone(), s.speed_alert_kmh)
    };
    if masks.is_empty() { return; }
    let speed_zones = parse_speed_zones_for_cam(&masks, cam_id);
    let lines = parse_lines_for_cam(&masks, cam_id);
    if speed_zones.is_empty() && lines.is_empty() { return; }

    for td in tracked {
        let Some(last) = td.history.last() else { continue };
        for (_name, quad, h) in &speed_zones {
            if !crate::tracking::point_in_poly(last.x, last.y, quad) { continue; }
            if let Some(kmh) = crate::tracking::track_speed_kmh(&td.history, h, quad) {
                if kmh > 1.0 && kmh < 300.0 {
                    update_event_top_speed(state, cam_id, kmh).await;
                    if speed_alert > 0.0 && kmh >= speed_alert {
                        crate::agent::dispatch_intelligence_alert(
                            state, "speeding", &format!("{} at ~{:.0} km/h", td.label, kmh), cam_id, None).await;
                    }
                }
            }
        }
        for (lname, a, b) in &lines {
            if let Some(dir) = crate::tracking::line_cross(&td.history, *a, *b) {
                fire_crossing_event(state, cam_id, &td.label, lname, dir).await;
            }
        }
    }
}

async fn update_event_top_speed(state: &Arc<AppState>, cam_id: u8, kmh: f32) {
    // Exclude audio events: a sustained open audio event must not absorb a vehicle's
    // speed (it isn't a tracked object). COALESCE so motion-only events (NULL category)
    // still match — `NULL != 'audio'` is NULL (not true), which would wrongly skip them.
    let _ = sqlx::query(
        "UPDATE motion_events SET top_speed_kmh = MAX(COALESCE(top_speed_kmh, 0), ?)
         WHERE id = (SELECT id FROM motion_events WHERE cam_id=? AND ended_at IS NULL
                     AND COALESCE(event_category,'') != 'audio' ORDER BY started_at DESC LIMIT 1)"
    ).bind(kmh).bind(cam_id as i64).execute(&state.db).await;
}

/// Per-(camera,line) cooldown so bbox jitter on the line can't spam events.
static CROSS_COOLDOWN: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<(u8, String), std::time::Instant>>> = std::sync::OnceLock::new();

/// Per-camera throttle for continuous face capture (mature NVRs `save_attempts`).
static FACE_CAPTURE_THROTTLE: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<u8, std::time::Instant>>> = std::sync::OnceLock::new();

/// Global throttle (~hourly) for pruning stale anonymous body fragments.
static BODY_PRUNE_AT: std::sync::OnceLock<std::sync::Mutex<Option<std::time::Instant>>> = std::sync::OnceLock::new();

/// True at most once per hour, process-wide — gates `reid::prune_anon_bodies`.
fn body_prune_due() -> bool {
    let cell = BODY_PRUNE_AT.get_or_init(|| std::sync::Mutex::new(None));
    match cell.lock() {
        Ok(mut g) => {
            let now = std::time::Instant::now();
            let due = g.map_or(true, |t| now.duration_since(t).as_secs() >= 3600);
            if due { *g = Some(now); }
            due
        }
        Err(_) => false,
    }
}

/// Confirmation memory for face↔body auto-labelling:
/// `(cam, track_id) → (person_id, accumulated votes, last touch)`.
static FACE_TRACK_CONFIRM: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<(u8, u64), (String, crate::face::IdentityVote, std::time::Instant)>>> = std::sync::OnceLock::new();

/// Tracklet → body identity cache: a confirmed tracklet keeps ONE body person_id
/// (re-arbitrated only by a consensus-confirmed face — face overrides body).
/// `(cam, track_id) → (body_person_id, match score if from a body query, last touch)`.
static TRACK_BODY_ID: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<(u8, u64), (String, Option<f32>, std::time::Instant)>>> = std::sync::OnceLock::new();
/// Per-tracklet store throttle so the gallery grows ~1×/s, not every frame.
static TRACK_STORE_AT: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<(u8, u64), std::time::Instant>>> = std::sync::OnceLock::new();

/// Evict the oldest half of a `(key → (.., Instant))`-shaped map when over `cap`.
/// The old `.clear()` bound wiped EVERY live tracklet's confirmation state on a
/// busy multi-camera session, resetting identities mid-track; evicting the stale
/// half keeps active tracks intact.
fn evict_oldest_half<K: Clone + std::hash::Hash + Eq>(
    m: &mut std::collections::HashMap<K, impl TouchStamp>, cap: usize,
) {
    if m.len() <= cap { return; }
    let mut stamps: Vec<(K, std::time::Instant)> = m.iter().map(|(k, v)| (k.clone(), v.stamp())).collect();
    stamps.sort_by_key(|(_, t)| *t);
    for (k, _) in stamps.into_iter().take(m.len() / 2) { m.remove(&k); }
}
trait TouchStamp { fn stamp(&self) -> std::time::Instant; }
impl TouchStamp for std::time::Instant { fn stamp(&self) -> std::time::Instant { *self } }
impl TouchStamp for (String, Option<f32>, std::time::Instant) { fn stamp(&self) -> std::time::Instant { self.2 } }
impl TouchStamp for (String, crate::face::IdentityVote, std::time::Instant) { fn stamp(&self) -> std::time::Instant { self.2 } }

/// Cached body identity for a tracklet (None for untracked `track_id == 0`).
/// Returns (person_id, body-match score if the id came from a body query).
fn track_body_id_get(cam_id: u8, track_id: u64) -> Option<(String, Option<f32>)> {
    if track_id == 0 { return None; }
    let cell = TRACK_BODY_ID.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    cell.lock().ok().and_then(|mut m| {
        m.get_mut(&(cam_id, track_id)).map(|v| { v.2 = std::time::Instant::now(); (v.0.clone(), v.1) })
    })
}

fn track_body_id_set(cam_id: u8, track_id: u64, pid: &str, score: Option<f32>) {
    if track_id == 0 { return; }
    let cell = TRACK_BODY_ID.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    if let Ok(mut m) = cell.lock() {
        evict_oldest_half(&mut m, 2048);
        m.insert((cam_id, track_id), (pid.to_string(), score, std::time::Instant::now()));
    }
}

/// True at most ~1×/s per tracklet (always for untracked, matching the old cadence).
fn track_store_due(cam_id: u8, track_id: u64) -> bool {
    if track_id == 0 { return true; }
    let cell = TRACK_STORE_AT.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    match cell.lock() {
        Ok(mut m) => {
            evict_oldest_half(&mut m, 2048);
            let now = std::time::Instant::now();
            let due = m.get(&(cam_id, track_id)).map_or(true, |t| now.duration_since(*t).as_secs_f32() >= 1.0);
            if due { m.insert((cam_id, track_id), now); }
            due
        }
        Err(_) => true,
    }
}

/// Accumulate a face vote for this tracklet and return true once the pairing
/// passes the SHARED consensus policy (`face::IdentityVote::confirmed` — the
/// same rule that names events), so live fusion and event naming can't disagree
/// and two weak ticks of a look-alike can no longer poison a body gallery.
/// Resets if the identity flips mid-track.
fn confirm_face_track(cam_id: u8, track_id: u64, person_id: &str, score: f32, rec_threshold: f32) -> bool {
    let cell = FACE_TRACK_CONFIRM.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let Ok(mut m) = cell.lock() else { return false };
    evict_oldest_half(&mut m, 1024);
    let now = std::time::Instant::now();
    let e = m.entry((cam_id, track_id))
        .or_insert_with(|| (person_id.to_string(), crate::face::IdentityVote::default(), now));
    if e.0 != person_id { *e = (person_id.to_string(), crate::face::IdentityVote::default(), now); } // flipped → restart
    e.1.add(score);
    e.2 = now;
    e.1.confirmed(rec_threshold)
}

async fn fire_crossing_event(state: &Arc<AppState>, cam_id: u8, label: &str, line: &str, dir: i8) {
    // Debounce (2 s) per camera+line.
    {
        let cell = CROSS_COOLDOWN.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
        if let Ok(mut map) = cell.lock() {
            let key = (cam_id, line.to_string());
            let now = std::time::Instant::now();
            if let Some(t) = map.get(&key) {
                if now.duration_since(*t).as_secs() < 2 { return; }
            }
            map.insert(key, now);
        }
    }
    let arrow = if dir > 0 { "→" } else { "←" };
    let dom = format!("{} crossed {} {}", label, line, arrow);
    let id = uuid::Uuid::new_v4().to_string();
    let now = chrono::Utc::now().to_rfc3339();
    let _ = sqlx::query(
        "INSERT INTO motion_events(id, started_at, ended_at, duration_secs, peak_score, event_category, dominant_label, cam_id)
         VALUES(?,?,?,1.0,0.8,'crossing',?,?)"
    ).bind(&id).bind(&now).bind(&now).bind(&dom).bind(cam_id as i64).execute(&state.db).await;
    crate::timeline::log(&state.db, &id, cam_id as i64,
        crate::timeline::class::CROSSING, Some("line"), Some(&dom), None).await;
    crate::review_segments::upsert_review_segment(&state.db, &id).await;
    tracing::info!("CROSSING: cam{} {}", cam_id, dom);
    state.app_handle.emit("agent:analyzed", ()).ok();
    crate::agent::dispatch_intelligence_alert(state, "line crossing", &dom, cam_id, None).await;
}

/// Hardware-acceleration preference for ONNX inference, set once at boot from
/// `settings.inference_device` ("auto"/"gpu" → true, "cpu" → false). Read by
/// every `build_ort_session` so YOLO + face + ALPR + CLIP + audio all share one
/// switch. Default true (best-effort GPU with automatic CPU fallback).
static GPU_INFERENCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Set the global GPU-inference preference (called at boot + on settings change).
pub(crate) fn set_gpu_inference(on: bool) {
    GPU_INFERENCE.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Global GPU-inference serializer.
///
/// ONNX Runtime sessions are individually thread-safe, but the **DirectML EP is
/// NOT safe to drive concurrently on one device**: running two sessions (YOLO +
/// face + ReID + ALPR + audio, each on its own thread / `spawn_blocking`) at the
/// same instant intermittently corrupts the shared GPU device state and Windows
/// `__fastfail`s the whole process with `0xc0000409` (STATUS_STACK_BUFFER_OVERRUN)
/// — the "the application closed itself" crash (3× on 2026-06-28, same offset).
/// ORT's own docs (and this file's `build_ort_session_cpu` note) already flag the
/// milder "Add node … parameter is incorrect" from the same race.
///
/// Every GPU `Run()` takes this lock so only one inference touches the device at a
/// time. On CPU the guard is a **no-op** (CPU sessions parallelise safely, and we
/// don't want to bottleneck audio/clip work behind detection there).
static GPU_INFER_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Acquire the global GPU-inference lock for the duration of one `Run()`. Returns
/// `None` (no serialization) when inference is on CPU. Scope it TIGHTLY around the
/// `.run()` call only — never hold it across another guarded run (std `Mutex` is
/// non-reentrant → nesting would deadlock) or across an `.await`.
/// Which GPU EP family is serving GPU sessions. The serializing lock exists
/// ONLY for DirectML (concurrent DML Run corrupts device state → 0xc0000409).
/// TensorRT-RTX sessions are independently streamed — concurrency is safe and
/// is precisely the point (YOLO + depth + faces overlap on the GPU).
static GPU_EP_NVRTX: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// Called from the Windows-only TensorRT-RTX path; dead on other targets.
#[allow(dead_code)]
pub(crate) fn set_active_gpu_ep_nvrtx() {
    GPU_EP_NVRTX.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// Is an NVIDIA adapter present? (drives TRT-RTX pack eligibility)
#[cfg(windows)]
pub(crate) fn has_nvidia_adapter() -> bool {
    discrete_dml_adapter().map(|(_, name)| {
        let n = name.to_lowercase();
        n.contains("nvidia") || n.contains("geforce") || n.contains("rtx") || n.contains("quadro")
    }).unwrap_or(false)
}
/// Linux: the proprietary driver publishes `/proc/driver/nvidia/version` once its
/// kernel module is loaded, which is exactly the precondition for the CUDA EP.
/// A file probe (no `nvidia-smi` subprocess) keeps this cheap enough to call from
/// every session build; the answer can't change without a reboot, so it's cached.
#[cfg(target_os = "linux")]
pub(crate) fn has_nvidia_adapter() -> bool {
    static HAS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *HAS.get_or_init(|| {
        std::path::Path::new("/proc/driver/nvidia/version").exists()
            || std::path::Path::new("/dev/nvidiactl").exists()
    })
}

#[cfg(not(any(windows, target_os = "linux")))]
#[allow(dead_code)] // no NVIDIA lane on these targets; stub keeps callers uniform
pub(crate) fn has_nvidia_adapter() -> bool { false }

#[cfg(test)]
mod ep_smoke {
    /// Build a real ORT session on whatever hardware this machine has, and report
    /// the execution provider that won.
    ///
    /// This is the only automated proof that the per-OS EP chain WORKS rather than
    /// merely compiles — macOS CoreML and the Linux CUDA lane have no hardware in
    /// this project, so CI runners are where they get exercised. Skipped unless
    /// `SC_SMOKE_MODEL` points at an .onnx file (CI downloads one for the test; the
    /// app itself ships no model).
    #[test]
    fn builds_a_session_on_this_platform() {
        let Ok(model) = std::env::var("SC_SMOKE_MODEL") else {
            eprintln!("SC_SMOKE_MODEL unset — skipping EP smoke test");
            return;
        };
        let path = std::path::PathBuf::from(&model);
        assert!(path.is_file(), "SC_SMOKE_MODEL={model} is not a file");

        // Reproduce boot's NVIDIA activation when pointed at a real data dir, so
        // this exercises the TensorRT/CUDA lane rather than always falling to
        // DirectML. Unset in CI (no packs there) → plain default chain.
        #[cfg(windows)]
        if let Ok(dd) = std::env::var("SC_SMOKE_DATA_DIR") {
            let armed = crate::trtx_runtime::activate_if_provisioned(std::path::Path::new(&dd));
            eprintln!("EP smoke: NVIDIA activation → {armed} ({:?})", crate::trtx_runtime::active_ep());
        }

        let session = super::build_ort_session(&path);
        assert!(session.is_ok(), "session build failed: {:?}", session.err().map(|e| e.to_string()));

        // Whatever armed, the app must be able to name it — a blank label means
        // the reporting path is broken even if inference works.
        let ep = super::active_accelerator();
        assert!(!ep.is_empty(), "no execution provider label was recorded");
        eprintln!("EP smoke: active execution provider = {ep}");
    }
}

pub(crate) fn gpu_infer_guard() -> Option<std::sync::MutexGuard<'static, ()>> {
    if !GPU_INFERENCE.load(std::sync::atomic::Ordering::Relaxed) {
        return None; // CPU inference — no GPU contention to serialize
    }
    if GPU_EP_NVRTX.load(std::sync::atomic::Ordering::Relaxed) {
        return None; // TensorRT-RTX active — concurrent GPU sessions are safe
    }
    Some(GPU_INFER_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner))
}

/// Human label for the accelerator the last session selected (shown in the UI).
static ACTIVE_EP: std::sync::RwLock<Option<String>> = std::sync::RwLock::new(None);

// ─── Per-model inference telemetry ────────────────────────────────────────────
// Every session.run site is timed (lock-wait INCLUDED — that's the honest
// end-to-end latency the pipeline actually pays, and it makes the DirectML
// serialization cost visible next to concurrency-capable EPs). Snapshot is
// surfaced in SystemMetrics → Telemetry panel and by `benchmark_inference`.

struct InferAcc {
    count: u64,
    sum_ms: f64,
    /// Last 128 samples (ring) for an approximate p95.
    ring: Vec<f32>,
    ring_pos: usize,
}

static INFER_STATS: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<&'static str, InferAcc>>> =
    std::sync::OnceLock::new();

/// RAII timer: created before the gpu guard, records on drop (after run).
pub(crate) struct InferTimer { model: &'static str, start: std::time::Instant }
pub(crate) fn infer_timer(model: &'static str) -> InferTimer {
    InferTimer { model, start: std::time::Instant::now() }
}
impl Drop for InferTimer {
    fn drop(&mut self) {
        let ms = self.start.elapsed().as_secs_f64() * 1000.0;
        let mut g = INFER_STATS.get_or_init(Default::default)
            .lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let acc = g.entry(self.model).or_insert_with(|| InferAcc {
            count: 0, sum_ms: 0.0, ring: Vec::with_capacity(128), ring_pos: 0,
        });
        acc.count += 1;
        acc.sum_ms += ms;
        if acc.ring.len() < 128 { acc.ring.push(ms as f32); }
        else { acc.ring[acc.ring_pos] = ms as f32; acc.ring_pos = (acc.ring_pos + 1) % 128; }
    }
}

#[derive(serde::Serialize, Clone, Debug)]
pub struct InferStatRow {
    pub model: String,
    pub count: u64,
    pub avg_ms: f32,
    pub p95_ms: f32,
}

pub(crate) fn infer_stats_snapshot() -> Vec<InferStatRow> {
    let g = INFER_STATS.get_or_init(Default::default)
        .lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut out: Vec<InferStatRow> = g.iter().map(|(m, a)| {
        let mut ring = a.ring.clone();
        ring.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
        let p95 = if ring.is_empty() { 0.0 }
                  else { ring[((ring.len() as f32 * 0.95) as usize).min(ring.len() - 1)] };
        InferStatRow {
            model: m.to_string(),
            count: a.count,
            avg_ms: (a.sum_ms / a.count.max(1) as f64) as f32,
            p95_ms: p95,
        }
    }).collect();
    out.sort_by(|a, b| a.model.cmp(&b.model));
    out
}

/// The execution provider currently in use, e.g. "DirectML (GPU)",
/// "CoreML (Apple Neural Engine)", or "CPU". Surfaced in the System Monitor.
pub(crate) fn active_accelerator() -> String {
    ACTIVE_EP.read().ok().and_then(|g| g.clone()).unwrap_or_else(|| "CPU".to_string())
}

/// Like [`active_accelerator`] but distinguishes "nothing has loaded a model yet"
/// (`None`) from "the CPU provider is genuinely in use". The accelerator report
/// needs that difference — reporting CPU before the first session exists would
/// have users chasing a GPU problem they don't have.
pub(crate) fn active_accelerator_opt() -> Option<String> {
    ACTIVE_EP.read().ok().and_then(|g| g.clone())
}

/// True when GPU inference is enabled in settings (the user can turn it off).
pub(crate) fn gpu_inference_enabled() -> bool {
    GPU_INFERENCE.load(std::sync::atomic::Ordering::Relaxed)
}

/// Load an ONNX model into an ORT session (Microsoft ONNX Runtime), auto-selecting
/// the best execution provider for THIS device.
///
/// The EP chain is per-OS and **best-effort**: ORT runs unsupported nodes on CPU and,
/// if a hardware provider can't initialise, silently falls back — so a missing or
/// incompatible accelerator never breaks inference (worst case = CPU).
///   • Windows → DirectML (any DX12 GPU: NVIDIA / AMD / Intel; no vendor toolkit)
///   • macOS   → CoreML  (Apple Neural Engine + GPU; built into the OS, no extra deps)
///   • Linux/other → CPU (XNNPACK) today; CUDA/ROCm/OpenVINO are a managed-deps phase
pub(crate) fn build_ort_session(model_path: &std::path::Path) -> anyhow::Result<OrtSession> {
    build_ort_session_opts(model_path, false)
}

/// The DXGI adapter index DirectML should bind, chosen ONCE at first use and cached.
///
/// On a hybrid laptop (this app's target hardware) DXGI adapter **0 is the Intel
/// iGPU** — weak, 2 GB shared, already driving the display + WebView2 + video decode.
/// `DirectMLExecutionProvider::default()` binds adapter 0, so inference piled onto the
/// iGPU and oversubscribed it (cascading crashes + slow inference) while a powerful
/// discrete GPU sat idle. We enumerate DXGI and pick the **discrete** adapter (NVIDIA
/// `0x10DE` / AMD `0x1002`, biggest dedicated VRAM), returning its index so it can be
/// passed to `with_device_id`. `None` → no discrete GPU found → keep ORT's own
/// high-performance auto-selection.
#[cfg(windows)]
fn discrete_dml_adapter() -> Option<(i32, String)> {
    use std::sync::OnceLock;
    static CHOICE: OnceLock<Option<(i32, String)>> = OnceLock::new();
    CHOICE
        .get_or_init(|| {
            use windows::Win32::Graphics::Dxgi::{
                CreateDXGIFactory1, IDXGIFactory1, DXGI_ADAPTER_FLAG_SOFTWARE,
            };
            unsafe {
                let factory: IDXGIFactory1 = CreateDXGIFactory1().ok()?;
                let mut best: Option<(i32, String, u64)> = None; // (index, name, dedicated VRAM)
                let mut i = 0u32;
                while let Ok(adapter) = factory.EnumAdapters1(i) {
                    if let Ok(desc) = adapter.GetDesc1() {
                        let is_software =
                            (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32) != 0;
                        let is_discrete = desc.VendorId == 0x10DE || desc.VendorId == 0x1002;
                        let vram = desc.DedicatedVideoMemory as u64;
                        if !is_software && is_discrete {
                            let end = desc.Description.iter().position(|&c| c == 0)
                                .unwrap_or(desc.Description.len());
                            let name = String::from_utf16_lossy(&desc.Description[..end]);
                            if best.as_ref().map_or(true, |(_, _, v)| vram > *v) {
                                best = Some((i as i32, name, vram));
                            }
                        }
                    }
                    i += 1;
                }
                best.map(|(idx, name, _)| (idx, name))
            }
        })
        .clone()
}

/// Build a session pinned to the **CPU** execution provider, regardless of the
/// global GPU setting. For BACKGROUND models (the CLIP semantic-search embedder)
/// that must not contend with the real-time detector on a single GPU: concurrent
/// DirectML inference from two sessions raises "Add node … parameter is incorrect",
/// so background embedding stays on CPU — which also frees the GPU for detection.
pub(crate) fn build_ort_session_cpu(model_path: &std::path::Path) -> anyhow::Result<OrtSession> {
    build_ort_session_opts(model_path, true)
}

/// Prefer an FP16-quantized sibling (`model.fp16.onnx` next to `model.onnx`) when
/// running on GPU. FP16 is the industry default for GPU-class NVR inference
/// (mature NVRs ship FP16 OpenVINO models; TensorRT auto-calibrates to FP16; INT8 is
/// reserved for Coral-class edge TPUs and costs real accuracy). Converted with
/// `keep_io_types=True`, so tensors stay f32 at the boundary — zero code changes.
/// CPU sessions keep FP32 (CPUs execute fp16 slower, not faster).
fn prefer_fp16_sibling(model_path: &std::path::Path, force_cpu: bool) -> std::path::PathBuf {
    if force_cpu || !GPU_INFERENCE.load(std::sync::atomic::Ordering::Relaxed) {
        return model_path.to_path_buf();
    }
    let sibling = model_path.with_extension("fp16.onnx");
    if sibling.is_file() {
        tracing::info!("Using FP16 model: {:?}", sibling.file_name().unwrap_or_default());
        sibling
    } else {
        model_path.to_path_buf()
    }
}

fn build_ort_session_opts(model_path: &std::path::Path, force_cpu: bool) -> anyhow::Result<OrtSession> {
    // Try the FP16 sibling first (GPU only); if the EP rejects the fp16 graph for
    // any reason, fall back to the original FP32 file so a bad conversion can
    // never take a detector down — worst case is the old precision, never a failure.
    let preferred = prefer_fp16_sibling(model_path, force_cpu);
    if preferred != model_path {
        match build_ort_session_at(&preferred, force_cpu) {
            Ok(s) => return Ok(s),
            Err(e) => tracing::warn!("FP16 model {:?} failed to load ({e}) — falling back to FP32", preferred.file_name().unwrap_or_default()),
        }
    }
    build_ort_session_at(model_path, force_cpu)
}

fn build_ort_session_at(model_path: &std::path::Path, force_cpu: bool) -> anyhow::Result<OrtSession> {
    // ort errors carry non-Send pointers so we must .map_err before returning anyhow::Result
    let mut builder = OrtSession::builder()
        .map_err(|e| anyhow::anyhow!("ORT init: {e}"))?;

    #[allow(unused_mut)]
    let mut ep_label = "CPU".to_string();

    if !force_cpu && GPU_INFERENCE.load(std::sync::atomic::Ordering::Relaxed) {
        #[cfg(windows)]
        {
            use ort::execution_providers::{DirectMLExecutionProvider, CPUExecutionProvider, ExecutionProviderDispatch};
            use ort::ep::directml::{PerformancePreference, DeviceFilter};

            let mut eps: Vec<ExecutionProviderDispatch> = Vec::new();

            // NVIDIA lane — whichever EP passed its boot canary (trtx_runtime
            // activation order: TensorRT-RTX → classic TensorRT → CUDA). All
            // three provider DLLs ship in the cu12 build; their runtimes come
            // from the managed pack.
            //
            // CRITICAL: DirectML CANNOT share a session with TensorRT/CUDA —
            // ORT hard-errors "DML EP can only be used with CPU EPs" and the
            // whole session fails to load. So a session is EITHER an NVIDIA
            // lane (NVIDIA EP → CPU) OR the DirectML lane (DML → CPU), never
            // both. `nvidia_active` selects which.
            let nvidia_active = crate::trtx_runtime::active_ep();
            match nvidia_active {
                Some(crate::trtx_runtime::NvEp::Nvrtx) => {
                    let cache = crate::trtx_runtime::cache_dir()
                        .map(|c| c.to_string_lossy().into_owned()).unwrap_or_default();
                    eps.push(ort::ep::nvrtx::NVRTX::default()
                        .with_device_id(0)
                        .with_runtime_cache_path(cache)
                        .build());
                    ep_label = crate::trtx_runtime::NvEp::Nvrtx.label().to_string();
                }
                Some(crate::trtx_runtime::NvEp::Tensorrt) => {
                    // Engine + timing caches make the minutes-long TRT engine
                    // build a ONE-TIME cost per model+driver; fp16 suits our
                    // quantized model set.
                    let cache = crate::trtx_runtime::cache_dir()
                        .map(|c| c.to_string_lossy().into_owned()).unwrap_or_default();
                    eps.push(ort::ep::tensorrt::TensorRT::default()
                        .with_device_id(0)
                        .with_fp16(true)
                        .with_engine_cache(true)
                        .with_engine_cache_path(&cache)
                        .with_timing_cache(true)
                        .with_timing_cache_path(&cache)
                        .build());
                    ep_label = crate::trtx_runtime::NvEp::Tensorrt.label().to_string();
                }
                Some(crate::trtx_runtime::NvEp::Cuda) => {
                    eps.push(ort::ep::cuda::CUDA::default().with_device_id(0).build());
                    ep_label = crate::trtx_runtime::NvEp::Cuda.label().to_string();
                }
                None => {
                    // Intel OpenVINO (source-built ORT variants only).
                    #[cfg(feature = "openvino")]
                    {
                        eps.push(ort::ep::openvino::OpenVINO::default().build());
                        ep_label = "OpenVINO (Intel)".to_string();
                    }
                    // DirectML lane. Pin to the DISCRETE GPU by adapter index —
                    // the `HighPerformance` hint is unreliable on hybrid laptops
                    // and DML defaulted to adapter 0 (the Intel iGPU).
                    let dml = match discrete_dml_adapter() {
                        Some((idx, name)) => {
                            ep_label = format!("DirectML → {name} (adapter {idx})");
                            tracing::info!("DirectML pinned to discrete GPU: {name} (DXGI adapter {idx})");
                            DirectMLExecutionProvider::default().with_device_id(idx)
                        }
                        None => {
                            ep_label = "DirectML (GPU)".to_string();
                            DirectMLExecutionProvider::default()
                                .with_device_filter(DeviceFilter::Gpu)
                                .with_performance_preference(PerformancePreference::HighPerformance)
                        }
                    };
                    eps.push(dml.build());
                }
            }
            // CPU is the universal last-resort fallback in every lane.
            eps.push(CPUExecutionProvider::default().build());

            builder = builder
                .with_execution_providers(eps)
                .map_err(|e| anyhow::anyhow!("ORT EP: {e}"))?
                // DirectML REQUIRES memory-pattern OFF + SEQUENTIAL execution, or ORT
                // raises "Add node … parameter is incorrect" at run time (ORT DML docs).
                // Harmless for the CUDA/CPU providers.
                .with_memory_pattern(false)
                .map_err(|e| anyhow::anyhow!("ORT mem-pattern: {e}"))?
                .with_parallel_execution(false)
                .map_err(|e| anyhow::anyhow!("ORT exec-mode: {e}"))?;
        }

        #[cfg(target_os = "macos")]
        {
            use ort::ep::{CoreML, CPU};
            use ort::ep::coreml::{ComputeUnits, ModelFormat};
            // Cache the compiled CoreML model next to the .onnx so we don't recompile
            // it on every session load (CoreML's first-compile is the slow part).
            let cache_dir = model_path.parent().map(|p| p.join("coreml_cache"));
            let mut coreml = CoreML::default()
                .with_compute_units(ComputeUnits::All)      // ANE + GPU + CPU — let CoreML pick the fastest
                .with_model_format(ModelFormat::MLProgram)  // newer format: more ops on-device, faster (macOS 12+)
                .with_static_input_shapes(true);            // our detector inputs are fixed-size → better ANE perf
            if let Some(dir) = &cache_dir {
                std::fs::create_dir_all(dir).ok();
                coreml = coreml.with_model_cache_dir(dir.to_string_lossy());
            }
            builder = builder
                .with_execution_providers([coreml.build(), CPU::default().build()])
                .map_err(|e| anyhow::anyhow!("ORT EP: {e}"))?;
            ep_label = "CoreML (Apple Neural Engine)".to_string();
        }

        // Linux: ROCm EP under the `rocm` feature (AMD, system ROCm required —
        // EXPERIMENTAL/UNVERIFIED: no AMD/Linux hardware in this project to test).
        #[cfg(all(target_os = "linux", feature = "rocm"))]
        {
            builder = builder
                .with_execution_providers([ort::ep::rocm::ROCm::default().build()])
                .map_err(|e| anyhow::anyhow!("ORT EP: {e}"))?;
            ep_label = "ROCm (AMD)".to_string();
        }

        // Linux NVIDIA lane. There is no DirectML on Linux, so without this every
        // Linux install ran detection on the CPU no matter what GPU was fitted.
        // The runtime libraries come from `cuda_runtime.rs` (downloaded once,
        // dlopened at boot); if they're absent the EP fails to register and ORT
        // falls through to CPU exactly as before — never a hard failure.
        //
        // UNVERIFIED ON HARDWARE: no Linux GPU machine in this project. CI builds
        // it and runs a CPU smoke test; `accel_report` is what a Linux user runs
        // to confirm the lane actually armed on their box.
        #[cfg(all(target_os = "linux", not(feature = "rocm")))]
        {
            use ort::ep::{CPU, ExecutionProviderDispatch};
            let mut eps: Vec<ExecutionProviderDispatch> = Vec::new();
            if has_nvidia_adapter() {
                eps.push(ort::ep::cuda::CUDA::default().with_device_id(0).build());
                ep_label = "CUDA (NVIDIA)".to_string();
            }
            eps.push(CPU::default().build());
            builder = builder
                .with_execution_providers(eps)
                .map_err(|e| anyhow::anyhow!("ORT EP: {e}"))?;
        }
        // Other targets fall through to the built-in CPU (XNNPACK) provider.
    }

    // Only primary (GPU-capable) sessions set the globally-reported accelerator —
    // a forced-CPU background model must not flip the Telemetry panel to "CPU".
    if !force_cpu {
        if let Ok(mut g) = ACTIVE_EP.write() { *g = Some(ep_label.clone()); }
    }
    tracing::info!("ORT accelerator: {ep_label}  (model {:?})",
        model_path.file_name().unwrap_or_default());

    let session = builder
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow::anyhow!("ORT opt-level: {e}"))?
        .with_intra_threads(4)
        .map_err(|e| anyhow::anyhow!("ORT threads: {e}"))?
        .commit_from_file(model_path)
        .map_err(|e| anyhow::anyhow!("ORT load model: {e}"))?;
    Ok(session)
}

/// Decode JPEG → resize to 640×640 → CHW float32 Vec normalised [0,1].
/// Returns flat Vec of length 1×3×640×640 in channel-first row-major order.
pub(crate) fn preprocess_jpeg_for_yolo(jpeg: &[u8]) -> anyhow::Result<Vec<f32>> {
    let img   = image::load_from_memory(jpeg)?
        .resize_exact(640, 640, image::imageops::FilterType::Triangle);
    let rgb   = img.to_rgb8();
    let bytes = rgb.as_raw(); // HWC u8, length = 640*640*3

    // Transpose HWC u8 → CHW f32, normalise to [0,1]
    // Layout: channel-first [3, 640, 640] row-major; each channel plane is PLANE elems.
    const PLANE: usize = 640 * 640;
    let mut data = vec![0.0f32; 3 * PLANE];
    for y in 0..640usize {
        for x in 0..640usize {
            let src = (y * 640 + x) * 3;
            let dst = y * 640 + x;
            data[dst]             = bytes[src    ] as f32 / 255.0; // R (channel 0)
            data[PLANE + dst]     = bytes[src + 1] as f32 / 255.0; // G (channel 1)
            data[2 * PLANE + dst] = bytes[src + 2] as f32 / 255.0; // B (channel 2)
        }
    }
    Ok(data)
}

