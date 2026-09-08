//! Frame ingest pipeline: fast-path `stream_frame` (broadcast only), `process_frame` (motion + event state machine), shared `process_frame_inner` used by RTSP/NVR.

use std::sync::Arc;

use base64::{engine::general_purpose::STANDARD as B64, Engine};
use tauri::{Emitter, State};

use crate::{
    AppState, FrameResult,
    inference::parse_motion_masks_for_cam,
    motion::{build_mask_buffer, compute_motion_masked, decode_to_gray_bytes},
};


/// Latest captured frame for a camera as base64 JPEG, over IPC.
///
/// Lets the in-app UI (e.g. face enrollment "Live capture") grab frames from a
/// camera that's owned SERVER-SIDE (the dshow USB capture / RTSP relay) — which the
/// browser's `getUserMedia` can no longer open because ffmpeg holds the device
/// exclusively. Reuses `latest_frames` (populated by `fan_out_frame`), so there's no
/// second device open, no port/CORS dance, and no disruption to recording.
///
/// **Depth anonymization is enforced here.** This used to hand back the RAW frame
/// with no check, while every other consumer — the recorder (`dshow.rs`), face
/// enrolment (`face.rs`), WebRTC (`go2rtc.rs`) and event thumbnails
/// (`motion_lifecycle.rs`) — all gate on `is_anonymized`. That made this the one
/// hole in the promise: raw pixels reached the chat's snapshot cards, the agent's
/// `[SNAPSHOT]` tag and Telegram for a camera the user had explicitly anonymised.
///
/// Fails CLOSED, matching `dshow.rs`: if anonymization is on and the depth model
/// is missing, return nothing rather than the raw frame. A blank tile is a bug
/// report; a leaked face is not recoverable.
#[tauri::command]
pub async fn get_camera_snapshot(
    state: State<'_, Arc<AppState>>,
    cam_id: Option<u8>,
) -> Result<Option<String>, String> {
    let cam = cam_id.unwrap_or(0).min(15);
    let frame = state.latest_frames.read().await.get(&cam).cloned();
    let Some(jpeg) = frame.filter(|b| !b.is_empty()) else { return Ok(None) };

    if crate::depth::is_anonymized(cam) {
        let dir = state.data_dir.clone();
        // Depth inference is CPU-bound; keep it off the async runtime.
        let anon = tokio::task::spawn_blocking(move || {
            crate::depth::depth_anonymize_jpeg(&dir, &jpeg)
        }).await.map_err(|e| e.to_string())?;
        return Ok(match anon {
            Some(d) => Some(B64.encode(&d)),
            None => {
                tracing::warn!("cam{cam}: snapshot suppressed — anonymization is on but \
                                the depth model produced nothing (fail-closed)");
                None
            }
        });
    }
    Ok(Some(B64.encode(&jpeg)))
}

/// Fast path: push a raw JPEG frame to all live-stream subscribers.
/// No motion detection, no DB work — returns in microseconds so the
/// capture interval is never delayed by recording logic.
#[tauri::command]
pub async fn stream_frame(
    state: State<'_, Arc<AppState>>,
    frame_b64: String,
    cam_id: Option<u8>,
) -> Result<(), String> {
    let cam = cam_id.unwrap_or(0).min(15);
    let idx = cam as usize;
    let jpeg = Arc::new(B64.decode(frame_b64.trim()).unwrap_or_default());

    // Store latest frame for Telegram snapshot requests
    state.latest_frames.write().await.insert(cam, jpeg.as_ref().clone());

    // Broadcast live frame to mobile WebSocket viewers
    let _ = state.frame_txs[idx].send(Arc::clone(&jpeg));

    // v12: pre-motion ring buffer + clip_txs sink removed. Event clips are
    // now virtual slices of the continuous NVR recording.
    let _ = jpeg;

    Ok(())
}

/// Cheap per-frame fan-out — runs at FULL capture rate. Feeds the NVR recorder, the
/// HLS encoder, the ONNX inference queue (latest-wins), and the latest-frame snapshot
/// cache. Every line here is just a channel send / lock, so this never bottlenecks the
/// capture. The EXPENSIVE work (JPEG decode + motion diff in `process_frame_inner`) is
/// gated separately by the caller (drop-don't-queue) — that keeps RECORDING smooth and
/// real-time even when detection can't keep up (e.g. heavy frames in a debug build),
/// which is exactly mature NVRs' split between the recording and detection pipelines.
pub(crate) async fn fan_out_frame(state: &Arc<AppState>, cam: u8, jpeg_arc: &Arc<Vec<u8>>) {
    if jpeg_arc.is_empty() { return; }
    // `latest_frames` is the snapshot store the agent / HTTP snapshot / People preview
    // read from — they only ever want a RECENT frame, not every one. Storing every
    // capture frame here deep-clones a full JPEG into the map at the camera's full
    // rate. Throttle to ~6 fps so that per-frame copy (and the write-lock churn) is
    // decoupled from capture; recording + inference below stay full-rate.
    {
        use std::collections::HashMap;
        use std::sync::Mutex;
        use std::time::Instant;
        static LAST_SNAP: Mutex<Option<HashMap<u8, Instant>>> = Mutex::new(None);
        let now = Instant::now();
        let due = {
            let mut g = LAST_SNAP.lock().unwrap();
            let map = g.get_or_insert_with(HashMap::new);
            match map.get(&cam) {
                Some(t) if now.duration_since(*t).as_millis() < 160 => false,
                _ => { map.insert(cam, now); true }
            }
        }; // std mutex guard dropped here — never held across the await below
        if due {
            state.latest_frames.write().await.insert(cam, jpeg_arc.as_ref().clone());
        }
    }
    if let Some(tx) = state.nvr_pipe_txs.lock().await.get(&cam) { tx.try_send(Arc::clone(jpeg_arc)).ok(); } // drop-on-full
    if let Some(tx) = state.hls_pipe_txs.lock().await.get(&cam) { tx.try_send(Arc::clone(jpeg_arc)).ok(); } // drop-on-full
    state.infer_queue.push(cam, Arc::clone(jpeg_arc));
}

/// Inner motion-detection logic shared by the Tauri command and the RTSP relay task.
/// Decodes the JPEG, computes motion, updates the local DB. Does NOT touch
/// `frame_tx` — streaming is handled entirely by `stream_frame`.
///
/// `feed_pipes`: when true (browser path) this also fans out to NVR/HLS/inference at
/// the end. The server-side capture readers pass FALSE and call `fan_out_frame`
/// themselves at full rate, so recording isn't throttled by detection speed.
pub(crate) async fn process_frame_inner(state: &Arc<AppState>, frame_b64: String, cam: u8, _ts: i64, feed_pipes: bool) -> Result<FrameResult, String> {

    let jpeg_bytes = B64.decode(frame_b64.trim()).unwrap_or_default();
    if jpeg_bytes.is_empty() {
        return Ok(FrameResult { motion_detected: false, motion_score: 0.0, recording: false, event_id: None, motion_regions: vec![] });
    }

    let (threshold, sensitivity, record_on_motion) = {
        let s = state.settings.read().await;
        (s.motion_threshold, s.sensitivity, s.record_on_motion)
    };

    // v12: pre-motion ring buffer + clip_txs sink removed. Frames flow
    // through the NVR pipe (always-on) so any event window can be sliced
    // from the continuous recording at playback time.

    // ── MOTION DETECTION with region bounding boxes ───────────────────────────
    // Timed into the shared inference digest ("motion=…ms") so the cost is
    // provable from logs like every model.
    let motion_timer = crate::inference::infer_timer("motion");
    let (w0, h0, full_gray) = decode_to_gray_bytes(&jpeg_bytes).map_err(|e| e.to_string())?;
    // Diff at ≤320 px wide (mature NVRs run motion low-res): blur + diff are
    // O(pixels), and full 720p buys nothing for a coarse motion score.
    let (w, h, curr_gray) = crate::motion::downscale_gray(&full_gray, w0, h0, 320);
    drop(full_gray);

    // standard motion detection: simple frame-to-frame grayscale diff with
    // the mask applied to the diff itself. Masked pixels are guaranteed to be
    // skipped — no transients, no learned state, no surprises.
    let motion_polys: Vec<Vec<(f32, f32)>> = {
        let masks_json = state.settings.read().await.camera_masks.clone();
        parse_motion_masks_for_cam(&masks_json, cam)
    };
    let mask_buf = build_mask_buffer(&motion_polys, w, h);
    let lightning = state.settings.read().await.motion_lightning_threshold;

    // One-shot diagnostic per (cam, frame-size, mask-coverage) — proves the mask
    // is actually being loaded and applied for THIS camera.
    {
        use std::sync::Mutex;
        use std::collections::HashMap;
        static LAST_LOGGED: Mutex<Option<HashMap<u8, (u32, u32, usize, usize)>>> = Mutex::new(None);
        let masked_count = mask_buf.iter().filter(|&&m| m).count();
        let snap = (w, h, motion_polys.len(), masked_count);
        let mut guard = LAST_LOGGED.lock().unwrap();
        let map = guard.get_or_insert_with(HashMap::new);
        if map.get(&cam) != Some(&snap) {
            map.insert(cam, snap);
            tracing::info!(
                "Mask diag cam{cam}: {}×{}  polys={}  masked_pixels={}  ({}% of frame)",
                w, h, motion_polys.len(), masked_count,
                if (w * h) > 0 { masked_count * 100 / (w as usize * h as usize) } else { 0 }
            );
        }
    }

    let (motion_score, motion_regions) = {
        let mut cs_map = state.cam_states.lock().await;
        let cs = cs_map.entry(cam).or_default();
        let (score, regions) = if let Some(prev) = cs.prev_frame.as_ref() {
            if cs.prev_dims == (w, h) {
                compute_motion_masked(prev, &curr_gray, &mask_buf, threshold, lightning, w, h)
            } else { (0.0, vec![]) }
        } else { (0.0, vec![]) };
        cs.prev_frame = Some(curr_gray);
        cs.prev_dims  = (w, h);
        (score, regions)
    };
    drop(motion_timer); // decode + downscale + diff measured; lifecycle excluded
    // ── v9: unified motion-event lifecycle ──────────────────────────────────
    // The browser/MJPEG path used to have its own state machine here with a
    // hardcoded 15s post-buffer that ignored the user's `record_post_buffer_secs`
    // setting. v9 routes all camera kinds through `tick_motion_event` so the
    // hysteresis + standard object-driven close work the same way
    // regardless of how frames arrive.
    let lifecycle_settings = crate::motion_lifecycle::LifecycleSettings::snapshot(state).await;
    let lifecycle_result = crate::motion_lifecycle::tick_motion_event(
        state, cam, motion_score, &jpeg_bytes, &lifecycle_settings, 15.0,
    ).await;
    let motion_detected = lifecycle_result.motion_detected;
    let event_id = lifecycle_result.event_id;
    // Tell the frontend a new motion event just opened so the timeline / event
    // list refresh INSTANTLY instead of waiting for the 30s poll.
    if let Some(opened_id) = lifecycle_result.just_opened {
        state.app_handle.emit("event:opened", serde_json::json!({
            "id": opened_id, "cam_id": cam,
        })).ok();
    }
    let _ = (record_on_motion, lifecycle_result.just_closed, lifecycle_result.just_dropped, sensitivity);

    // ── Fan out to NVR / HLS / inference ─────────────────────────────────────
    // Browser path feeds the pipes here; server-side capture readers fed them at
    // full rate already (feed_pipes=false) so recording isn't throttled by this
    // (possibly slow) detection pass.
    if feed_pipes {
        // Only the browser path fans out from here; build the shared Arc lazily so the
        // server-side capture path (feed_pipes=false) never pays for this clone.
        fan_out_frame(state, cam, &Arc::new(jpeg_bytes)).await;
    }

    let recording = state.clip_txs.lock().await.contains_key(&cam);
    Ok(FrameResult { motion_detected, motion_score, recording, event_id, motion_regions })
}

/// Tauri command wrapper — delegates to the shared inner function.
#[tauri::command]
pub async fn process_frame(
    state: State<'_, Arc<AppState>>,
    frame_b64: String,
    cam_id: Option<u8>,
    timestamp_ms: i64,
) -> Result<FrameResult, String> {
    process_frame_inner(&state, frame_b64, cam_id.unwrap_or(0).min(15), timestamp_ms, true).await
}

/// v9: pull-based query for the YOLO badge. `CameraView` calls this on mount
/// so the badge resolves regardless of whether the inference loop's one-shot
/// `inference:status` event fired before the listener was attached.
#[tauri::command]
pub async fn get_inference_status(
    state: State<'_, Arc<AppState>>,
) -> Result<crate::state::InferenceStatusSnapshot, String> {
    Ok(state.inference_status.read().await.clone())
}
