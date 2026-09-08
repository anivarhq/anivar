//! Native (nokhwa) camera capture helpers + the per-camera background capture loop.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use tauri::Emitter;

use crate::{AppState, FrameResult};
// v12: per-event clip writer removed.
use crate::inference::parse_motion_masks_for_cam;
use crate::motion::{build_mask_buffer, compute_motion_masked};


/// Encode a raw RGB24 buffer to JPEG using the `image` crate.
pub(crate) fn encode_jpeg_rgb(rgb: &[u8], width: u32, height: u32, quality: u8) -> anyhow::Result<Vec<u8>> {
    let img = image::RgbImage::from_raw(width, height, rgb.to_vec())
        .ok_or_else(|| anyhow::anyhow!("RGB buffer too small for {}×{}", width, height))?;
    let mut buf = Vec::with_capacity((width * height) as usize);
    let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, quality);
    enc.encode_image(&img)?;
    Ok(buf)
}

/// Convert RGB24 to 8-bit grayscale in one pass (luma approximation).
pub(crate) fn rgb_to_gray_direct(rgb: &[u8]) -> Vec<u8> {
    rgb.chunks_exact(3)
        .map(|p| ((p[0] as u32 * 77 + p[1] as u32 * 150 + p[2] as u32 * 29) >> 8) as u8)
        .collect()
}

// ─── Native capture loop ─────────────────────────────────────────────────────

/// Spawns a dedicated OS thread that captures frames from a physical camera via nokhwa,
/// encodes them as JPEG, and broadcasts to WebSocket viewers — entirely in Rust with no
/// JS/IPC overhead.  Also performs inline motion detection and recording management.
#[cfg(any(windows, target_os = "macos", target_os = "linux"))]
pub(crate) async fn run_capture_loop(
    cam_id: u8,
    device_index: u32,
    cancel_flag: Arc<AtomicBool>,
    state: Arc<AppState>,
) {
    use tokio::sync::mpsc::unbounded_channel as ubch;

    // Bridge: blocking OS thread → async tokio task (UnboundedSender is Send + non-async send)
    let (raw_tx, mut raw_rx) = ubch::<(u32, u32, Vec<u8>)>();
    let cancel_clone = Arc::clone(&cancel_flag);

    // ── Dedicated OS thread for nokhwa (blocking I/O) ────────────────────────
    let app_handle_thread = state.app_handle.clone();
    std::thread::spawn(move || {
        #[cfg(any(windows, target_os = "macos", target_os = "linux"))]
        {
            use nokhwa::{Camera, pixel_format::RgbFormat,
                utils::{CameraIndex, CameraFormat, FrameFormat, RequestedFormat, RequestedFormatType}};

            // Open the camera, then ENUMERATE its real formats and pick the best one.
            // The old approach (a fixed candidate list + `Closest`) silently matched a
            // **1-FPS** variant → the feed was a glitchy/laggy slideshow. Instead we
            // open permissively, list `compatible_camera_formats()`, and select the
            // highest resolution (≤1080p to bound CPU/USB) with a REAL frame rate
            // (≥15fps), preferring MJPEG (hardware-compressed → low CPU) over raw YUYV,
            // then higher fps. This fixes BOTH the 1-FPS lag AND the under-resolution.
            let mut last_err = String::new();

            // PHASE 1 — probe-open permissively JUST to enumerate the real formats,
            // then DROP the probe (releases the device). The in-place set_camera_*
            // methods are no-ops on MediaFoundation (format stays 1fps), so we must
            // re-open with the chosen format below.
            let formats: Vec<CameraFormat> = {
                let mut probe: Option<Camera> = None;
                for init in [RequestedFormatType::AbsoluteHighestResolution,
                             RequestedFormatType::AbsoluteHighestFrameRate] {
                    match Camera::new(CameraIndex::Index(device_index),
                                      RequestedFormat::new::<RgbFormat>(init)) {
                        Ok(c) => { probe = Some(c); break; }
                        Err(e) => { last_err = e.to_string(); }
                    }
                }
                match probe {
                    Some(mut c) => c.compatible_camera_formats().unwrap_or_default(),
                    None => {
                        if !last_err.is_empty() {
                            tracing::warn!("camera {device_index}: format probe failed, using candidate fallback: {last_err}");
                        }
                        Vec::new()
                    }
                }
            };

            // PHASE 2 — pick the BEST format: highest res ≤1080p with a REAL fps (≥15),
            // preferring MJPEG (hardware-compressed) over raw YUYV, then higher fps.
            let best: Option<CameraFormat> = {
                let usable: Vec<CameraFormat> = formats.into_iter()
                    .filter(|f| matches!(f.format(), FrameFormat::MJPEG | FrameFormat::YUYV)
                        && f.resolution().width() <= 1920 && f.resolution().height() <= 1080)
                    .collect();
                usable.iter().filter(|f| f.frame_rate() >= 15)
                    .max_by_key(|f| (
                        f.resolution().width() as u64 * f.resolution().height() as u64,
                        if f.format() == FrameFormat::MJPEG { 1u32 } else { 0 },
                        f.frame_rate(),
                    ))
                    .copied()
                    .or_else(|| usable.iter().max_by_key(|f| f.frame_rate()).copied())
            };

            // Let the probe handle fully release before re-opening (MediaFoundation).
            std::thread::sleep(std::time::Duration::from_millis(200));

            // PHASE 3 — OPEN with the EXACT chosen format (negotiated at open time, the
            // only reliable way to actually get e.g. 640×480@30 instead of 1fps).
            let req_type = match best {
                Some(b) => { tracing::info!("cam{} opening chosen format: {:?}", device_index, b); RequestedFormatType::Exact(b) }
                None    => RequestedFormatType::AbsoluteHighestFrameRate,
            };
            let mut cam = match Camera::new(CameraIndex::Index(device_index),
                                            RequestedFormat::new::<RgbFormat>(req_type)) {
                Ok(c) => c,
                Err(e) => {
                    // Fallback to a plain 640×480@30 request, else give up.
                    last_err = e.to_string();
                    match Camera::new(CameraIndex::Index(device_index), RequestedFormat::new::<RgbFormat>(
                        RequestedFormatType::Closest(CameraFormat::new_from(640, 480, FrameFormat::YUYV, 30)))) {
                        Ok(c) => c,
                        Err(e2) => {
                            let friendly_msg = if last_err.contains("preempted") || last_err.contains("0xC00D3EA3")
                                || e2.to_string().contains("0xC00D3EA3") {
                                format!("Camera {} is being used by another application. Close other camera apps and try again.", device_index)
                            } else {
                                format!("Camera {} failed to open: {} / {}", device_index, last_err, e2)
                            };
                            tracing::error!("{}", friendly_msg);
                            app_handle_thread.emit(&format!("native:camera_error:{}", cam_id), friendly_msg).ok();
                            return;
                        }
                    }
                }
            };
            tracing::info!("cam{} final format: {:?}", device_index, cam.camera_format());

            if let Err(e) = cam.open_stream() {
                let friendly_msg = if e.to_string().contains("preempted") || e.to_string().contains("0xC00D3EA3") {
                    format!("Camera {} stream is being used by another application. Close other camera apps and try again.", device_index)
                } else {
                    format!("Camera {} stream error: {}", device_index, e)
                };
                tracing::error!("{}", friendly_msg);
                app_handle_thread.emit(
                    &format!("native:camera_error:{}", cam_id), friendly_msg
                ).ok();
                return;
            }
            tracing::info!("cam{} opened: {:?}", device_index, cam.camera_format());
            app_handle_thread.emit(&format!("native:camera_opened:{}", cam_id), ()).ok();
            // Hard cap: never send frames faster than 20 FPS regardless of camera rate.
            // This prevents the async loop from being overwhelmed with YUYV conversions.
            const FRAME_BUDGET: std::time::Duration = std::time::Duration::from_millis(50);
            let mut last_sent = std::time::Instant::now() - FRAME_BUDGET;
            loop {
                if cancel_clone.load(Ordering::Relaxed) { break; }
                match cam.frame() {
                    Ok(frame) => {
                        let now = std::time::Instant::now();
                        if now.duration_since(last_sent) < FRAME_BUDGET { continue; }
                        if let Ok(rgb) = frame.decode_image::<RgbFormat>() {
                            let (w, h) = (rgb.width(), rgb.height());
                            if raw_tx.send((w, h, rgb.into_raw())).is_err() { break; }
                            last_sent = now;
                        }
                    }
                    Err(e) => {
                        tracing::warn!("cam{} frame error: {}", device_index, e);
                        let friendly_msg = if e.to_string().contains("preempted") || e.to_string().contains("0xC00D3EA3") {
                            "Camera is being used by another application. Close other camera apps (like Zoom, Teams, or Camera) and try again.".to_string()
                        } else {
                            format!("Camera feed lost: {}", e)
                        };
                        app_handle_thread.emit(
                            &format!("native:camera_error:{}", cam_id),
                            friendly_msg
                        ).ok();
                        break;
                    }
                }
            }
            let _ = cam.stop_stream();
        }
    });

    // ── Async processing loop ─────────────────────────────────────────────────
    loop {
        if cancel_flag.load(Ordering::Relaxed) { break; }

        // Wait for the next frame, then drain any additional queued frames so we
        // always encode the freshest data. This eliminates the display delay caused
        // by a backlog of unprocessed frames piling up in the channel.
        let mut frame = match raw_rx.recv().await {
            Some(f) => f,
            None    => break,
        };
        while let Ok(newer) = raw_rx.try_recv() { frame = newer; }
        let (w, h, raw) = frame;

        // Native is a LOCAL camera (localhost stream + on-disk recording), so JPEG
        // bandwidth isn't a concern — floor the quality at 85 so the live feed +
        // recording look crisp. (The default stream_quality of 60 made native look
        // blocky vs the browser's raw video.) The downstream H.264 NVR encode also
        // benefits from a higher-quality source frame.
        let quality = state.settings.read().await.stream_quality.max(85);

        // Encode once — used for broadcast, pre-buffer, and recording
        let jpeg = match encode_jpeg_rgb(&raw, w, h, quality) {
            Ok(j)  => Arc::new(j),
            Err(e) => { tracing::warn!("JPEG encode: {}", e); continue; }
        };

        // 1. Broadcast to all WebSocket viewers for this camera slot
        let _ = state.frame_txs[cam_id as usize].send(Arc::clone(&jpeg));

        // 1b. Feed YOLO26 inference loop (same as browser camera path)
        state.infer_queue.push(cam_id, Arc::clone(&jpeg));

        // v12: pre-buffer + clip_txs feeding removed with the per-event clip
        // writer. NVR continuous recording (below) is the only sink now;
        // event clips are virtual slices served by /footage/:id/clip.

        // 3b. Feed NVR continuous-recording pipe (always-on)
        // Without this, native cameras never appear in NVR segments — only browser cameras do.
        {
            let nvr_txs = state.nvr_pipe_txs.lock().await;
            if let Some(tx) = nvr_txs.get(&cam_id) { tx.try_send(Arc::clone(&jpeg)).ok(); } // drop-on-full
        }

        // 3c. Feed the live HLS pipe (hardware-decoded `<video>` live view). The HLS
        // pipe was ONLY ever fed by the now-dead browser `process_frame` push, so for
        // server-side USB cameras it starved → `/hls/camN.m3u8` produced nothing and
        // the live view had to use the CPU/GPU-heavy MJPEG `<img>`. Feeding it here
        // (same as NVR above) makes the efficient HLS feed actually work.
        {
            let hls_txs = state.hls_pipe_txs.lock().await;
            if let Some(tx) = hls_txs.get(&cam_id) { tx.try_send(Arc::clone(&jpeg)).ok(); } // drop-on-full
        }

        // 4. Motion detection — standard frame-diff with mask applied to the
        //    diff. Masked pixels never contribute to the motion score.
        let gray = rgb_to_gray_direct(&raw);
        // ── Compute raw motion score (frame-diff with masks) ─────────────
        // The lifecycle module handles everything downstream — open / sustain
        // / close, hysteresis, standard object-driven close, drop-on-no-
        // object — so this block ONLY computes the score.
        let motion_score = {
            let s = state.settings.read().await;
            let thr = s.motion_threshold;
            let lightning = s.motion_lightning_threshold;
            let camera_masks_json = s.camera_masks.clone();
            drop(s);

            let polys = parse_motion_masks_for_cam(&camera_masks_json, cam_id);
            let mask_buf = build_mask_buffer(&polys, w, h);

            let mut cs_map = state.cam_states.lock().await;
            let cs = cs_map.entry(cam_id).or_default();
            let score = if let Some(prev) = cs.prev_frame.as_ref() {
                if cs.prev_dims == (w, h) {
                    let (s, _) = compute_motion_masked(prev, &gray, &mask_buf, thr, lightning, w, h);
                    s
                } else { 0.0 }
            } else { 0.0 };
            cs.prev_frame = Some(gray);
            cs.prev_dims  = (w, h);
            score
        };

        // 5. v9: unified motion-event lifecycle. One source of truth shared
        //    with the browser/MJPEG path (inference_cmds.rs::process_frame_inner).
        let lifecycle_settings = crate::motion_lifecycle::LifecycleSettings::snapshot(&state).await;
        let lifecycle_result = crate::motion_lifecycle::tick_motion_event(
            &state, cam_id, motion_score, &jpeg, &lifecycle_settings, 20.0,
        ).await;
        let motion_detected = lifecycle_result.motion_detected;
        let event_id = lifecycle_result.event_id.clone();
        let record_on_motion = state.settings.read().await.record_on_motion;

        // (Loitering moved to inference.rs::evaluate_loitering — track-dwell based,
        // not event-age based, so it requires an actual person to persist.)

        // (Old inline state machine replaced by tick_motion_event call above.)
        let _just_closed = lifecycle_result.just_closed;
        let _just_dropped = lifecycle_result.just_dropped;

        // v12: per-event clip recording removed — events now reference virtual
        // slices of the continuous NVR recording (see footage::footage_clip).
        // The `record_on_motion` setting is now informational only.
        let _ = record_on_motion;

        // 7. Push FrameResult to the desktop UI via Tauri event
        // `recording` always false post-v12 (no per-event recorder). NVR runs
        // continuously regardless.
        let result = FrameResult { motion_detected, motion_score, recording: false, event_id, motion_regions: vec![] };
        state.app_handle.emit(&format!("frame:result:{}", cam_id), &result).ok();
    }

    // Cleanup: remove OUR handle so the slot can be restarted — but ONLY if it's
    // still ours (a newer start may have already replaced it; don't clobber that).
    {
        let mut handles = state.capture_handles.lock().await;
        if handles.get(&cam_id).map(|h| Arc::ptr_eq(&h.cancel, &cancel_flag)).unwrap_or(false) {
            handles.remove(&cam_id);
        }
    }
    state.app_handle.emit("native:camera_stopped", cam_id).ok();
}
