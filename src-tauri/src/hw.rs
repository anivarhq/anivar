//! Hardware H.264 encoder detection (ffmpeg-based).
//!
//! Probes ffmpeg's encoder list and picks the best available accelerator —
//! NVENC (NVIDIA), QSV (Intel Quick Sync), AMF (AMD), VideoToolbox (macOS),
//! V4L2 (Linux). Falls back to the software libx264 encoder otherwise.

// ─── Hardware encoder detection ───────────────────────────────────────────────

/// Probe ffmpeg for available hardware H.264 encoders.
/// Returns the best encoder name for use in -c:v argument.
pub async fn detect_hw_encoder(ffmpeg: &std::path::Path) -> String {
    let Ok(out) = crate::proc::tokio_cmd(ffmpeg)
        .args(["-hide_banner", "-encoders"])
        .output().await
    else { return "libx264".to_string(); };
    let s = String::from_utf8_lossy(&out.stdout);
    // Check in preference order: NVIDIA → Intel → AMD → Apple → CPU
    if s.contains(" h264_nvenc ")         { return "h264_nvenc".to_string(); }
    if s.contains(" h264_qsv ")           { return "h264_qsv".to_string(); }
    if s.contains(" h264_amf ")           { return "h264_amf".to_string(); }
    if s.contains(" h264_videotoolbox ")  { return "h264_videotoolbox".to_string(); }
    if s.contains(" h264_vaapi ")         { return "h264_vaapi".to_string(); }
    "libx264".to_string()
}

/// Probe once, return (best encoder, whether Intel QSV is ALSO present).
/// QSV matters even when NVENC wins: it's the overflow lane when a camera
/// count exceeds the NVENC session budget (iGPU sessions are separate silicon).
pub async fn detect_hw_encoders(ffmpeg: &std::path::Path) -> (String, bool) {
    let Ok(out) = crate::proc::tokio_cmd(ffmpeg)
        .args(["-hide_banner", "-encoders"])
        .output().await
    else { return ("libx264".to_string(), false); };
    let s = String::from_utf8_lossy(&out.stdout);
    let best = if s.contains(" h264_nvenc ")        { "h264_nvenc" }
        else if s.contains(" h264_qsv ")            { "h264_qsv" }
        else if s.contains(" h264_amf ")            { "h264_amf" }
        else if s.contains(" h264_videotoolbox ")   { "h264_videotoolbox" }
        else if s.contains(" h264_vaapi ")          { "h264_vaapi" }
        else                                        { "libx264" };
    (best.to_string(), s.contains(" h264_qsv "))
}

/// Probe ffmpeg for the best available hardware VIDEO DECODER (`-hwaccel`).
///
/// Decode is the single heaviest CPU cost in an NVR. Mature NVRs' #1 optimization is
/// to hardware-decode the detection stream so the CPU isn't software-decoding
/// H.264/H.265 on every frame. We pick the OS-appropriate accelerator that runs on
/// a dedicated fixed-function decode block (so it does NOT contend with GPU
/// inference running on the same adapter's compute units):
///   • Windows → d3d11va (any GPU's decode engine; needs no device config; falls
///               back to software per-stream if a codec isn't hw-decodable)
///   • macOS   → videotoolbox (Apple media engine)
///   • Linux   → cuda (NVDEC) / vaapi (Intel+AMD) / qsv
/// Returns the `-hwaccel` value, or "" when none is available (software decode).
pub async fn detect_hw_decoder(ffmpeg: &std::path::Path) -> String {
    let Ok(out) = crate::proc::tokio_cmd(ffmpeg)
        .args(["-hide_banner", "-hwaccels"])
        .output().await
    else { return String::new(); };
    let s = String::from_utf8_lossy(&out.stdout).to_lowercase();
    let has = |name: &str| s.lines().any(|l| l.trim() == name);
    // OS-preferred order — first reliable match wins. d3d11va leads on Windows
    // because it's the most robust (auto-falls-back to software for unsupported
    // codecs) and uses the GPU's decode block, not its inference compute units.
    let prefs: &[&str] = if cfg!(target_os = "windows") {
        &["d3d11va", "cuda", "qsv", "dxva2"]
    } else if cfg!(target_os = "macos") {
        &["videotoolbox"]
    } else {
        &["cuda", "vaapi", "qsv"]
    };
    for p in prefs { if has(p) { return (*p).to_string(); } }
    String::new()
}

/// ffmpeg INPUT flags (placed BEFORE `-i`) for the detected hardware decoder.
/// Empty ⇒ software decode. We intentionally don't set `-hwaccel_output_format`,
/// so the decoder hands system-memory frames to the downstream CPU filters
/// (fps/scale) — that keeps the graph simple and lets ffmpeg fall back cleanly.
pub(crate) fn hw_decode_args(decoder: &str) -> Vec<String> {
    if decoder.is_empty() { return Vec::new(); }
    let mut a = vec!["-hwaccel".to_string(), decoder.to_string()];
    if decoder == "vaapi" {
        a.extend(["-vaapi_device".into(), "/dev/dri/renderD128".into()]);
    }
    a
}

/// Build ffmpeg encoder args for the detected hardware encoder.
pub(crate) fn hw_encoder_args(encoder: &str) -> Vec<String> {
    let mut a = match encoder {
        "h264_nvenc" => vec![
            "-c:v".into(), "h264_nvenc".into(),
            "-preset".into(), "p4".into(),
            "-cq".into(), "23".into(),
            "-gpu".into(), "any".into(),
        ],
        "h264_qsv" => vec![
            "-c:v".into(), "h264_qsv".into(),
            "-preset".into(), "faster".into(),
            "-global_quality".into(), "23".into(),
        ],
        "h264_amf" => vec![
            "-c:v".into(), "h264_amf".into(),
            "-quality".into(), "speed".into(),
            "-rc".into(), "cqp".into(),
            "-qp_i".into(), "23".into(),
            "-qp_p".into(), "23".into(),
        ],
        "h264_videotoolbox" => vec![
            "-c:v".into(), "h264_videotoolbox".into(),
            "-b:v".into(), "2M".into(),
        ],
        "h264_vaapi" => vec![
            "-vaapi_device".into(), "/dev/dri/renderD128".into(),
            "-c:v".into(), "h264_vaapi".into(),
            "-qp".into(), "23".into(),
        ],
        _ => vec![
            "-c:v".into(), "libx264".into(),
            "-preset".into(), "ultrafast".into(),
            "-crf".into(), "23".into(),
            // Cap encode threads: x264 defaults to ~1.5× logical cores of
            // frame-threads (42 on a 28-core box), each holding frame +
            // lookahead buffers — one recorder ballooned to 110 threads /
            // 750 MB. Two threads comfortably encode 720p30 ultrafast.
            "-threads".into(), "2".into(),
        ],
    };
    // Force 4:2:0 chroma on the recorded stream. USB webcams feed MJPEG that decodes
    // to yuvj444p (H.264 High 4:4:4 Predictive) — which Chromium/WebView2 and mobile
    // H.264 decoders CANNOT decode. That made recorded footage play black in the
    // Events/NVR views and Telegram reject it as "can't play this format". 4:2:0 is
    // universally decodable (mature NVRs record 4:2:0 for exactly this reason). The RTSP
    // recorder already does this; the USB path never did. Skipped for VAAPI, whose
    // encoder consumes GPU surfaces (nv12/hwupload), not a `-pix_fmt` conversion.
    if encoder != "h264_vaapi" {
        a.extend(["-pix_fmt".into(), "yuv420p".into()]);
    }
    a
}

