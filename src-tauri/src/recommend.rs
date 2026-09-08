//! Host probe + the two hints that survived the recommendation engine.
//!
//! **The "hardware-aware model picks" feature was REMOVED (2026-07-28).** Arsenal
//! used to open with a hardware stat card and four scored picks (Face / YOLO /
//! VLM / agent LLM) under the banner "Powered by llmfit-core" — a library that
//! had already been dropped from the build, so the panel advertised a scoring
//! engine that no longer existed. The picks were also noise next to the plain
//! install / remove actions people actually use, so the ladders, the scoring,
//! the `CookbookReport` bundle and its commands are gone.
//!
//! Hardware facts come from [`crate::hostinfo`] (sysinfo + DXGI + NVML, all
//! in-process — no subprocesses).
//!
//! What remains:
//!
//!   • [`detect_host`] — total/available RAM, CPU cores, GPU name, VRAM,
//!     inference backend. Feeds the face hint below.
//!   • [`face_recommendation`] — Off / Small / Large tier for this host, behind
//!     the "Auto" button in Settings → Face recognition (a real, used action).
//!   • [`list_installed_skills`] — filesystem scan of `<data>/skills/` so the
//!     UI can render install + active state per skill.

use serde::Serialize;

/// What we return to the frontend after detecting the host. The shape mirrors
/// the host facts that are useful in the Settings UI
/// (we don't surface the multi-GPU detail — one machine, one recommendation).
#[derive(Debug, Clone, Serialize)]
pub struct HostSpecs {
    pub total_ram_gb:     f64,
    pub available_ram_gb: f64,
    pub cpu_name:         String,
    pub cpu_cores:        usize,
    pub gpu_name:         Option<String>,
    pub gpu_vram_gb:      Option<f64>,
    pub gpu_backend:      String,        // "CUDA" | "Metal" | "ROCm" | "Vulkan" | "CPU (x86)" | …
    pub unified_memory:   bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct FaceRecommendation {
    /// `"off"` | `"small"` | `"large"` — the value to write into Settings.face_model.
    pub tier:   String,
    /// Short human-readable justification for the UI tooltip.
    pub reason: String,
    /// Detected hardware (re-used to render a "your machine" card).
    pub host:   HostSpecs,
}

/// Probe the host with OUR in-process detectors (sysinfo + DXGI + NVML) —
/// llmfit-core was removed entirely: its `SystemSpecs::detect()` ran raw
/// `nvidia-smi` subprocesses (flashing console windows), and its big general
/// LLM catalog was overkill. Mature NVRs ship a small CURATED model set
/// with simple hardware rules — that is what `vlm_recommendation` /
/// `agent_llm_recommendation` do now.
pub(crate) fn probe_host() -> HostSpecs {
    let cm = crate::hostinfo::cpu_mem();

    // Apple Silicon: no DXGI/NVML — the GPU is on-die with UNIFIED memory, which
    // is exactly what the recommendation rules key on ("Metal" + unified_memory).
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        return HostSpecs {
            total_ram_gb:     cm.total_ram_gb,
            available_ram_gb: cm.available_ram_gb,
            cpu_name:         cm.cpu_name,
            cpu_cores:        cm.cpu_cores,
            gpu_name:         Some("Apple Silicon GPU".into()),
            gpu_vram_gb:      Some(cm.total_ram_gb), // unified: RAM is VRAM
            gpu_backend:      "Metal".into(),
            unified_memory:   true,
        };
    }

    #[allow(unreachable_code)]
    let adapters = crate::hostinfo::gpu_adapters();
    // Best DISCRETE adapter drives the recommendation (an integrated GPU shares
    // system RAM and should not count as VRAM for model-fit purposes).
    let best = adapters.iter()
        .filter(|a| a.is_discrete())
        .max_by(|a, b| a.dedicated_vram_gb.total_cmp(&b.dedicated_vram_gb));
    let gpu_backend = match best.map(|b| b.vendor_id) {
        Some(0x10DE) => "CUDA",
        Some(0x1002) => "Vulkan",   // Windows AMD without ROCm
        Some(0x8086) => "SYCL",     // Intel Arc discrete
        Some(_)      => "Vulkan",
        None         => "CPU (x86)",
    };
    HostSpecs {
        total_ram_gb:     cm.total_ram_gb,
        available_ram_gb: cm.available_ram_gb,
        cpu_name:         cm.cpu_name,
        cpu_cores:        cm.cpu_cores,
        gpu_name:         best.map(|b| b.name.clone()),
        gpu_vram_gb:      best.map(|b| b.dedicated_vram_gb),
        gpu_backend:      gpu_backend.to_string(),
        unified_memory:   false,
    }
}

/// Probe the host — in-process, subprocess-free. Cheap; runs once per UI open.
pub(crate) fn detect_host() -> HostSpecs {
    probe_host()
}

/// Pick the face-recognition tier that's most likely to give real-time
/// performance on this host. Rules (intentionally simple — mature NVRs' own
/// docs make the same Small/Large split):
///
///   • Discrete GPU with ≥ 4 GB VRAM (CUDA / ROCm / Vulkan)  → "large"
///   • Apple Silicon with ≥ 16 GB unified memory             → "large"
///   • Otherwise                                             → "small"
///
/// No GPU at all still picks "small" — never "off" — because the small tier
/// is genuinely usable on CPU (mature NVRs ship it that way too).
pub(crate) fn recommend_face_tier(host: &HostSpecs) -> (&'static str, String) {
    // Strong discrete GPU
    let strong_dgpu = matches!(host.gpu_backend.as_str(), "CUDA" | "ROCm" | "Vulkan")
        && host.gpu_vram_gb.unwrap_or(0.0) >= 4.0;
    if strong_dgpu {
        return ("large", format!(
            "{} with {:.1} GB VRAM — full-precision ArcFace runs real-time.",
            host.gpu_name.clone().unwrap_or_else(|| "Discrete GPU".into()),
            host.gpu_vram_gb.unwrap_or(0.0),
        ));
    }
    // Apple Silicon — unified memory acts as VRAM for Metal/MLX backends.
    let apple = host.gpu_backend == "Metal" && host.unified_memory;
    if apple && host.total_ram_gb >= 16.0 {
        return ("large", format!(
            "Apple Silicon with {:.0} GB unified memory — Metal can run FP32 ArcFace.",
            host.total_ram_gb,
        ));
    }
    // Default: small tier is CPU-real-time and only 37 MB on disk.
    let why = if host.gpu_vram_gb.is_some() {
        format!(
            "{} ({:.1} GB VRAM) — under 4 GB, Small tier (INT8) keeps latency low.",
            host.gpu_name.clone().unwrap_or_else(|| "GPU".into()),
            host.gpu_vram_gb.unwrap_or(0.0),
        )
    } else {
        format!(
            "{} cores / {:.0} GB RAM, no discrete GPU — Small tier runs on CPU at ~80 ms/face.",
            host.cpu_cores, host.total_ram_gb,
        )
    };
    ("small", why)
}

/// Full one-shot recommendation: detect + pick tier + bundle the reasoning.
pub(crate) fn face_recommendation() -> FaceRecommendation {
    let host = detect_host();
    let (tier, reason) = recommend_face_tier(&host);
    FaceRecommendation { tier: tier.to_string(), reason, host }
}
