//! System-level Tauri commands: GPU picker, token revocation, session kick, local IP, LAN camera discovery, ntfy topic, hardware-aware recommendations.

use std::sync::Arc;

use serde::Serialize;
use tauri::State;
use uuid::Uuid;

use crate::AppState;
use crate::recommend;


/// Recommend the face-recognition tier ("small" | "large") for this host,
/// plus a human-readable justification. Frontend exposes this as an "Auto"
/// button next to the Off/Small/Large selector.
#[tauri::command]
pub async fn recommend_face_model() -> Result<recommend::FaceRecommendation, String> {
    tokio::task::spawn_blocking(recommend::face_recommendation)
        .await
        .map_err(|e| e.to_string())
}

/// List which skill packages are installed on disk + their footprint. Used by
/// the Cookbook to render install state and "uninstall to reclaim space" CTAs.
#[tauri::command]
pub async fn list_installed_skills(state: State<'_, Arc<AppState>>) -> Result<Vec<crate::skills::SkillStatus>, String> {
    let data_dir = state.data_dir.clone();
    tokio::task::spawn_blocking(move || crate::skills::list_installed_skills(&data_dir))
        .await
        .map_err(|e| e.to_string())
}


#[derive(Debug, Clone, Serialize)]
pub struct GpuInfo {
    pub name: String,
    pub is_discrete: bool,
}

#[tauri::command]
pub async fn list_gpus() -> Result<Vec<GpuInfo>, String> {
    // In-process DXGI enumeration (was a PowerShell CIM query — subprocess, slow,
    // and a console-window risk). DXGI is the same source D3D/Task Manager use.
    Ok(crate::hostinfo::gpu_adapters().into_iter().map(|a| GpuInfo {
        is_discrete: a.is_discrete() || gpu_is_discrete(&a.name),
        name: a.name,
    }).collect())
}

/// Live host utilization for the System Monitor — CPU (overall + per-core), RAM,
/// and GPU (integrated or discrete) usage + VRAM. Sampled from Windows performance
/// counters in a single `Get-Counter` call (the same source Task Manager uses), so
/// it covers integrated GPUs that `nvidia-smi` can't. Static names/specs come from
/// `list_gpus`; this is the live layer the UI polls.
#[derive(Debug, Clone, Default, Serialize)]
pub struct GpuMetric {
    pub name:        String,
    pub util:        f32,   // 3D-engine %, 0 when idle (e.g. an Optimus dGPU powered down)
    pub mem_mb:      u64,   // dedicated GPU memory in use (MB)
    pub is_discrete: bool,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct SystemMetrics {
    pub cpu_total:    f32,        // overall CPU %
    pub per_core:     Vec<f32>,   // per-logical-core %
    pub mem_used_mb:  u64,
    pub mem_total_mb: u64,
    pub gpus:         Vec<GpuMetric>, // EVERY adapter (integrated + discrete), live or idle
    pub accelerator:  String,    // active ONNX EP: "DirectML (GPU)" / "CoreML (Apple Neural Engine)" / "CPU"
    /// Per-model inference latency (rolling avg + p95 incl. GPU-lock wait).
    pub infer_stats:  Vec<crate::inference::InferStatRow>,
}

/// Discrete-GPU heuristic by adapter name (Intel UHD/Iris = integrated; Arc = discrete).
fn gpu_is_discrete(name: &str) -> bool {
    let n = name.to_lowercase();
    n.contains("nvidia") || n.contains("geforce") || n.contains("rtx") || n.contains("gtx")
        || n.contains("radeon rx") || n.contains("rx 6") || n.contains("rx 7") || n.contains("arc")
}

#[tauri::command]
pub async fn get_system_metrics() -> Result<SystemMetrics, String> {
    // Fully IN-PROCESS (mature NVRs model: psutil/NVML library reads, no
    // subprocesses). The old implementation shelled a ~1s PowerShell `Get-Counter`
    // script per poll — slow, locale-fragile ("telemetry offline" on any parse
    // hiccup), and a console-window risk. This is a few library calls in <5 ms.
    let (cpu_total, per_core, mem_used_mb, mem_total_mb) =
        tokio::task::spawn_blocking(crate::hostinfo::live_cpu_mem)
            .await.map_err(|e| e.to_string())?;

    // NVIDIA GPUs: live util/VRAM via NVML (what nvidia-smi itself reads).
    // Other adapters (integrated Intel/AMD): listed from DXGI with idle stats —
    // Windows has no clean in-process counter for iGPU 3D load, and the discrete
    // card is the one doing the inference/encode work anyway.
    let nvidia = crate::hostinfo::nvidia_live();
    let mut gpus: Vec<GpuMetric> = nvidia.iter().map(|(name, util, mem_mb)| GpuMetric {
        name: name.clone(), util: *util, mem_mb: *mem_mb, is_discrete: true,
    }).collect();
    for a in crate::hostinfo::gpu_adapters() {
        let already = gpus.iter().any(|g| g.name.eq_ignore_ascii_case(&a.name));
        if !already {
            gpus.push(GpuMetric {
                is_discrete: a.is_discrete() || gpu_is_discrete(&a.name),
                name: a.name, util: 0.0, mem_mb: 0,
            });
        }
    }

    Ok(SystemMetrics {
        cpu_total, per_core, mem_used_mb, mem_total_mb, gpus,
        accelerator: crate::inference::active_accelerator(),
        infer_stats: crate::inference::infer_stats_snapshot(),
    })
}

/// Pin the WebView2 UI compositor to the POWER-SAVING GPU (the iGPU) at every
/// launch. The whole-app-lag root cause: a stale `GpuPreference=2` registry
/// entry put UI compositing on the SAME discrete GPU that Ollama + DirectML
/// detection + NVENC saturate — when a model generated, the compositor starved
/// and every panel janked (the edge-AI NVRs lesson: the UI must never share the
/// model GPU). Our inference is unaffected: DirectML pins the discrete adapter
/// explicitly by DXGI index, and the app's own preference is untouched.
/// Registry entries are PER EXE PATH and WebView2 updates change the version
/// dir, so we sweep every installed version under both install roots, every
/// launch. Runs before the Builder so it applies to THIS launch.
#[cfg(windows)]
pub fn pin_webview_gpu_power_saving() {
    use windows::core::HSTRING;
    use windows::Win32::System::Registry::{RegSetKeyValueW, HKEY_CURRENT_USER, REG_SZ};

    let mut exes: Vec<String> = Vec::new();
    let roots = [
        std::path::PathBuf::from(r"C:\Program Files (x86)\Microsoft\EdgeWebView\Application"),
        std::path::PathBuf::from(std::env::var("LOCALAPPDATA").unwrap_or_default())
            .join(r"Microsoft\EdgeWebView\Application"),
    ];
    for root in roots {
        let Ok(dirs) = std::fs::read_dir(&root) else { continue };
        for d in dirs.flatten() {
            let exe = d.path().join("msedgewebview2.exe");
            if exe.is_file() {
                exes.push(exe.to_string_lossy().to_string());
            }
        }
    }
    if exes.is_empty() {
        tracing::warn!("webview gpu pin: no msedgewebview2.exe found under EdgeWebView roots");
        return;
    }

    let subkey = HSTRING::from(r"Software\Microsoft\DirectX\UserGpuPreferences");
    // 1 = power-saving (iGPU). UTF-16 with trailing NUL, byte length for cbData.
    let val: Vec<u16> = "GpuPreference=1;\0".encode_utf16().collect();
    for exe in exes {
        let name = HSTRING::from(exe.as_str());
        let rc = unsafe {
            RegSetKeyValueW(
                HKEY_CURRENT_USER,
                &subkey,
                &name,
                REG_SZ.0,
                Some(val.as_ptr() as *const std::ffi::c_void),
                (val.len() * 2) as u32,
            )
        };
        if rc.is_ok() {
            tracing::info!("webview gpu pin: {exe} -> power-saving (iGPU)");
        } else {
            tracing::warn!("webview gpu pin failed for {exe}: {rc:?}");
        }
    }
}
#[cfg(not(windows))]
pub fn pin_webview_gpu_power_saving() {}

#[tauri::command]
pub async fn set_preferred_gpu(gpu_name: String) -> Result<(), String> {
    // WebView2 GPU preference is a Windows registry concept — no-op elsewhere
    // (macOS/Linux compositors pick the GPU themselves).
    #[cfg(not(windows))]
    { let _ = gpu_name; Ok(()) }
    #[cfg(windows)]
    { set_preferred_gpu_windows(gpu_name).await }
}

#[cfg(windows)]
async fn set_preferred_gpu_windows(gpu_name: String) -> Result<(), String> {
    // Find msedgewebview2.exe (highest version)
    let find_ps = r#"
$dirs = Get-ChildItem 'C:\Program Files (x86)\Microsoft\EdgeWebView\Application' -Directory -ErrorAction SilentlyContinue
$exe = $dirs | Sort-Object Name -Descending | ForEach-Object {
    $p = Join-Path $_.FullName 'msedgewebview2.exe'
    if (Test-Path $p) { $p; break }
} | Select-Object -First 1
$exe
"#;
    let find_out = crate::proc::tokio_cmd("powershell")
        .args(["-NoProfile", "-NonInteractive", "-Command", find_ps.trim()])
        .output()
        .await
        .map_err(|e| format!("PowerShell error: {e}"))?;

    let webview_exe = String::from_utf8_lossy(&find_out.stdout).trim().to_string();
    if webview_exe.is_empty() {
        return Err("msedgewebview2.exe not found".to_string());
    }

    // GpuPreference: 0=default, 1=integrated, 2=high performance (discrete)
    let gpu_lower = gpu_name.to_lowercase();
    let pref_value = if gpu_name.is_empty() {
        "GpuPreference=0;".to_string()
    } else if gpu_lower.contains("nvidia") || gpu_lower.contains("radeon rx")
        || gpu_lower.contains("rx 6") || gpu_lower.contains("rx 7")
    {
        "GpuPreference=2;".to_string()
    } else {
        "GpuPreference=1;".to_string()
    };

    let set_ps = format!(
        r#"$key = 'HKCU:\Software\Microsoft\DirectX\UserGpuPreferences'
if (-not (Test-Path $key)) {{ New-Item -Path $key -Force | Out-Null }}
Set-ItemProperty -Path $key -Name '{webview_exe}' -Value '{pref_value}' -Type String"#,
        webview_exe = webview_exe.replace('\'', "\\'"),
        pref_value = pref_value,
    );

    let set_out = crate::proc::tokio_cmd("powershell")
        .args(["-NoProfile", "-NonInteractive", "-Command", &set_ps])
        .output()
        .await
        .map_err(|e| format!("PowerShell error: {e}"))?;

    if !set_out.status.success() {
        let err = String::from_utf8_lossy(&set_out.stderr);
        return Err(format!("Registry write failed: {err}"));
    }

    Ok(())
}

/// Revoke the current auth token and generate a new one.
/// All existing phone sessions are disconnected instantly (watch channel notifies WS handlers).
/// The user must re-scan the QR code from the desktop app to reconnect.
#[tauri::command]
pub async fn revoke_token(state: State<'_, Arc<AppState>>) -> Result<(), String> {
    let new_token = Uuid::new_v4().simple().to_string();
    // Persist to DB first so the new token survives restart
    sqlx::query("INSERT OR REPLACE INTO settings(key,value) VALUES('auth_token',?)")
        .bind(&new_token).execute(&state.db).await.map_err(|e| e.to_string())?;
    // Update in-memory state (shared Arc — HTTP middleware sees it instantly)
    *state.auth_token.write().await = new_token;
    // Signal all open WebSocket connections to close themselves
    let gen = *state.revoke_tx.borrow() + 1;
    state.revoke_tx.send(gen).ok();
    Ok(())
}

/// Forcibly disconnect a single remote viewer by their session UUID.
/// The WS handler receives the kick signal and closes the socket gracefully.
#[tauri::command]
pub async fn disconnect_client(id: String, state: State<'_, Arc<AppState>>) -> Result<(), String> {
    let sender = state.kick_txs.lock().await.remove(&id);
    if let Some(tx) = sender {
        tx.send(()).ok();
        Ok(())
    } else {
        Err(format!("No active session: {}", id))
    }
}

#[tauri::command]
pub async fn get_local_ip() -> Result<String, String> {
    local_ip_address::local_ip().map(|ip| ip.to_string()).map_err(|e| e.to_string())
}

#[derive(Debug, serde::Serialize, serde::Deserialize, Clone, Default)]
pub struct DiscoveredCamera {
    pub ip: String,
    pub port: u16,
    pub kind: String,  // "rtsp" | "mjpeg" | "http" | "onvif"
    pub url: String,
    pub name: String,
}

#[tauri::command]
pub async fn discover_cameras(state: State<'_, Arc<AppState>>) -> Result<Vec<DiscoveredCamera>, String> {
    use std::net::SocketAddr;
    use tokio::time::timeout;
    use std::time::Duration;

    let local_ip = local_ip_address::local_ip()
        .map(|ip| ip.to_string())
        .map_err(|e| e.to_string())?;

    let parts: Vec<&str> = local_ip.split('.').collect();
    if parts.len() != 4 {
        return Err("Could not determine subnet".to_string());
    }
    let subnet = format!("{}.{}.{}", parts[0], parts[1], parts[2]);

    // Common camera ports: 554/8554 (RTSP), 8080/80 (HTTP/MJPEG), 8000 (ONVIF)
    let port_kinds: Vec<(u16, &str)> = vec![
        (554,  "rtsp"),
        (8554, "rtsp"),
        (8080, "mjpeg"),
        (80,   "http"),
        (8000, "onvif"),
    ];

    let mut handles = Vec::new();

    for i in 1u32..=254 {
        let ip = format!("{}.{}", subnet, i);
        if ip == local_ip { continue; }

        for &(port, kind) in &port_kinds {
            let ip2  = ip.clone();
            let kind2 = kind.to_string();
            handles.push(tokio::spawn(async move {
                let addr: SocketAddr = format!("{}:{}", ip2, port)
                    .parse().map_err(|_| ())?;
                timeout(Duration::from_millis(250), tokio::net::TcpStream::connect(addr))
                    .await.map_err(|_| ())?.map_err(|_| ())?;

                let url = match kind2.as_str() {
                    "rtsp" => format!("rtsp://{}:{}/", ip2, port),
                    _      => format!("http://{}:{}/", ip2, port),
                };
                Ok::<DiscoveredCamera, ()>(DiscoveredCamera {
                    name: format!("Camera @ {}", ip2),
                    ip: ip2,
                    port,
                    kind: kind2,
                    url,
                })
            }));
        }
    }

    let mut found: Vec<DiscoveredCamera> = Vec::new();
    for h in handles {
        if let Ok(Ok(cam)) = h.await {
            // De-dupe: prefer lower port per IP, skip pure HTTP if we already have RTSP
            let already = found.iter().any(|c| c.ip == cam.ip && c.kind == cam.kind);
            if !already {
                found.push(cam);
            }
        }
    }

    // Sort by IP then port
    found.sort_by(|a, b| {
        let ia: Vec<u8> = a.ip.split('.').filter_map(|s| s.parse().ok()).collect();
        let ib: Vec<u8> = b.ip.split('.').filter_map(|s| s.parse().ok()).collect();
        ia.cmp(&ib).then(a.port.cmp(&b.port))
    });

    state.camera_inventory.write().await.network = found.clone();
    Ok(found)
}

// ─── Hardware acceleration: NVIDIA Performance Pack + benchmarks ─────────────

#[derive(serde::Serialize)]
pub struct TrtxStatus {
    pub supported: bool,       // windows + NVIDIA adapter present
    pub provisioned: bool,     // TRT-RTX pack files on disk
    pub trt_provisioned: bool, // classic TensorRT pack files on disk
    pub active: bool,          // some NVIDIA EP passed its canary
    /// Which EP is armed: "nvrtx" | "tensorrt" | "cuda" | "none"
    pub active_ep: String,
    /// TensorRT-RTX runtime version our shipped provider links (e.g. "1.3"),
    /// read from its PE import table. The UI names THIS in the SDK picker rather
    /// than a hardcoded number that rots on the next ORT bump. Empty if unknown.
    pub trtx_required_runtime: String,
}

#[cfg(windows)]
fn nv_status(data_dir: &std::path::Path) -> TrtxStatus {
    let ep = crate::trtx_runtime::active_ep();
    TrtxStatus {
        supported: crate::inference::has_nvidia_adapter(),
        provisioned: crate::trtx_runtime::is_provisioned(data_dir),
        trt_provisioned: crate::trtx_runtime::is_trt_provisioned(data_dir),
        active: ep.is_some(),
        active_ep: match ep {
            Some(crate::trtx_runtime::NvEp::Nvrtx) => "nvrtx".into(),
            Some(crate::trtx_runtime::NvEp::Tensorrt) => "tensorrt".into(),
            Some(crate::trtx_runtime::NvEp::Cuda) => "cuda".into(),
            None => "none".into(),
        },
        trtx_required_runtime: crate::trtx_runtime::trtx_required_version().unwrap_or_default(),
    }
}

/// One candidate execution provider and why it is (or isn't) in use.
#[derive(serde::Serialize, Clone)]
pub struct AccelRow {
    pub ep: String,
    /// `armed` = running inference · `ready` = usable, another lane won ·
    /// `unavailable` = cannot run here · `n/a` = not applicable to this machine.
    pub state: String,
    pub detail: String,
}

fn row(ep: &str, state: &str, detail: impl Into<String>) -> AccelRow {
    AccelRow { ep: ep.into(), state: state.into(), detail: detail.into() }
}

/// Why each accelerator is or isn't running — the answer to "is my GPU being
/// used, and if not, what exactly is missing?" on ANY device.
///
/// Every row is derived from a real probe (adapter enumeration, the pack's own
/// file list, the provider DLL's linked soname, the registration canary result),
/// never from a hardcoded assumption. Before this existed the app logged its
/// fall-throughs to a console nobody was attached to, which is how a TensorRT-RTX
/// lane that could never load went unnoticed through several releases.
// data_dir is consumed by the Windows-only accelerator rows below.
#[allow(unused_variables)]
pub(crate) fn accel_rows(data_dir: &std::path::Path) -> Vec<AccelRow> {
    let mut rows = Vec::new();

    if !crate::inference::gpu_inference_enabled() {
        rows.push(row("GPU inference", "n/a", "disabled in Settings — everything runs on CPU"));
    }
    let active_label = crate::inference::active_accelerator_opt();
    let pending = active_label.is_none();

    #[cfg(windows)]
    {
        let nvidia = crate::inference::has_nvidia_adapter();
        let ep = crate::trtx_runtime::active_ep();

        // ── TensorRT-RTX ─────────────────────────────────────────────────────
        let required = crate::trtx_runtime::trtx_required_dlls();
        rows.push(if matches!(ep, Some(crate::trtx_runtime::NvEp::Nvrtx)) {
            row("TensorRT-RTX", "armed", "RTX-optimized JIT; concurrent GPU sessions enabled")
        } else if !nvidia {
            row("TensorRT-RTX", "n/a", "no NVIDIA adapter")
        } else if crate::trtx_runtime::trtx_requirement_met(data_dir) {
            row("TensorRT-RTX", "unavailable", "runtime present but registration failed on this driver")
        } else if required.is_empty() {
            row("TensorRT-RTX", "unavailable", "pack not installed")
        } else {
            let need = crate::trtx_runtime::trtx_required_version().unwrap_or_default();
            row("TensorRT-RTX", "unavailable", format!(
                "needs runtime {need}.x ({}) — NVIDIA publishes no public {need}.x wheel; \
                 import the TensorRT-for-RTX SDK to enable this lane",
                required.join(" + ")))
        });

        // ── Classic TensorRT + CUDA (share one pack) ─────────────────────────
        let trt_pack = crate::trtx_runtime::is_trt_provisioned(data_dir);
        rows.push(if matches!(ep, Some(crate::trtx_runtime::NvEp::Tensorrt)) {
            row("TensorRT", "armed", "engine-cached FP16 inference")
        } else if !nvidia {
            row("TensorRT", "n/a", "no NVIDIA adapter")
        } else if !trt_pack {
            row("TensorRT", "unavailable", "NVIDIA performance pack not installed")
        } else if ep.is_some() {
            row("TensorRT", "ready", "pack installed; another NVIDIA lane won the canary")
        } else {
            row("TensorRT", "unavailable", "pack installed but the EP failed to register on this driver")
        });

        rows.push(if matches!(ep, Some(crate::trtx_runtime::NvEp::Cuda)) {
            row("CUDA", "armed", "NVIDIA-native inference")
        } else if !nvidia {
            row("CUDA", "n/a", "no NVIDIA adapter")
        } else if !trt_pack {
            row("CUDA", "unavailable", "shares the NVIDIA performance pack — not installed")
        } else {
            row("CUDA", "ready", "available as the NVIDIA fallback lane")
        });

        // ── DirectML — the universal Windows lane (any DX12 GPU) ─────────────
        rows.push(if active_label.as_deref().is_some_and(|l| l.starts_with("DirectML")) {
            row("DirectML", "armed", active_label.clone().unwrap_or_default())
        } else if ep.is_some() {
            row("DirectML", "ready", "not attempted — an NVIDIA lane is active (ORT forbids mixing DirectML with CUDA/TensorRT in one session)")
        } else if pending {
            row("DirectML", "ready", "no model loaded yet — the lane is selected on first inference")
        } else {
            row("DirectML", "unavailable", "GPU lane did not initialise; running on CPU")
        });
    }

    #[cfg(target_os = "linux")]
    {
        let nvidia = crate::inference::has_nvidia_adapter();
        let provisioned = data_dir.join("cuda").join("libcudnn.so.9").exists()
            || std::fs::read_dir(data_dir.join("cuda"))
                .map(|d| d.filter_map(|e| e.ok())
                    .any(|e| e.file_name().to_string_lossy().starts_with("libcudnn")))
                .unwrap_or(false);
        rows.push(if active_label.as_deref().is_some_and(|l| l.starts_with("CUDA")) {
            row("CUDA", "armed", "NVIDIA-native inference")
        } else if !nvidia {
            row("CUDA", "n/a", "no NVIDIA driver (/proc/driver/nvidia/version absent)")
        } else if !provisioned {
            row("CUDA", "unavailable", "managed CUDA/cuDNN runtime not downloaded yet")
        } else if pending {
            row("CUDA", "ready", "runtime present — lane is selected on first inference")
        } else {
            row("CUDA", "unavailable", "runtime present but the EP failed to register (driver too old?)")
        });
        rows.push(row("DirectML", "n/a", "Windows-only API"));
    }

    #[cfg(target_os = "macos")]
    rows.push(if active_label.as_deref().is_some_and(|l| l.starts_with("CoreML")) {
        row("CoreML", "armed", "Apple Neural Engine + GPU")
    } else if pending {
        row("CoreML", "ready", "no model loaded yet — the lane is selected on first inference")
    } else {
        row("CoreML", "unavailable", "CoreML session build failed; running on CPU")
    });

    rows.push(match &active_label {
        Some(l) if l == "CPU" => row("CPU", "armed", "universal fallback"),
        Some(_) => row("CPU", "ready", "per-node fallback for unsupported operators"),
        None => row("CPU", "ready", "always available"),
    });

    rows
}

/// Log the accelerator table once, at boot, into the persistent log.
pub(crate) fn log_accel_report(data_dir: &std::path::Path) {
    for r in accel_rows(data_dir) {
        tracing::info!("accel: {:<14} {:<12} {}", r.ep, r.state, r.detail);
    }
}

#[tauri::command]
pub async fn accel_report(state: State<'_, Arc<AppState>>) -> Result<Vec<AccelRow>, String> {
    Ok(accel_rows(&state.data_dir))
}

#[tauri::command]
pub async fn trtx_status(state: State<'_, Arc<AppState>>) -> Result<TrtxStatus, String> {
    #[cfg(windows)]
    { Ok(nv_status(&state.data_dir)) }
    #[cfg(not(windows))]
    {
        let _ = state;
        Ok(TrtxStatus { supported: false, provisioned: false, trt_provisioned: false, active: false, active_ep: "none".into(), trtx_required_runtime: String::new() })
    }
}

/// Download + activate the classic TensorRT pack (~1.85 GB public wheels,
/// SHA-pinned, one-time). Sessions created AFTER activation use the new EP;
/// existing cached sessions keep their current EP until the app restarts —
/// the UI says so. The first model load after activation builds TensorRT
/// engines (minutes, once per model; cached thereafter).
#[tauri::command]
pub async fn install_trtx_pack(app: tauri::AppHandle, state: State<'_, Arc<AppState>>) -> Result<TrtxStatus, String> {
    #[cfg(windows)]
    {
        // Both NVIDIA payloads, behind this ONE consent. The CUDA runtime used to
        // be fetched at boot with no prompt at all (~1.8 GB), which meant a first
        // launch downloaded most of a TensorRT pack's worth of libraries before
        // the user had agreed to anything — and then this button asked about the
        // rest. One button, one agreement, everything or nothing.
        //
        // CUDA first: it is the smaller pack and it gives the CUDA EP as a working
        // fallback lane if TensorRT's canary later fails on this GPU.
        #[cfg(feature = "cuda")]
        let cuda_ok = match crate::cuda_runtime::ensure_cuda_runtime(&state.data_dir).await {
            Ok(_) => true,
            Err(e) => { tracing::warn!("CUDA runtime setup failed ({e}) — continuing to TensorRT"); false }
        };
        #[cfg(not(feature = "cuda"))]
        let cuda_ok = false;

        crate::trtx_runtime::ensure_trt_pack(&state.data_dir, &app).await.map_err(|e| e.to_string())?;
        let dd = state.data_dir.clone();
        let ok = tokio::task::spawn_blocking(move || crate::trtx_runtime::activate_if_provisioned(&dd))
            .await.map_err(|e| e.to_string())?;
        if !ok && !cuda_ok {
            return Err("Pack downloaded but no NVIDIA runtime passed its canary on this GPU — staying on DirectML".into());
        }
        Ok(nv_status(&state.data_dir))
    }
    #[cfg(not(windows))]
    {
        let _ = (app, state);
        Err("NVIDIA Performance Pack is Windows-only".into())
    }
}

/// Import a user-downloaded TensorRT-for-RTX SDK (zip or extracted folder).
/// ort rc.12 pairs with SDK 1.3 (free NVIDIA developer download). The canary
/// decides activation; a wrong file can never crash inference.
#[tauri::command]
pub async fn import_trtx_sdk(state: State<'_, Arc<AppState>>, path: String) -> Result<TrtxStatus, String> {
    #[cfg(windows)]
    {
        let src = std::path::PathBuf::from(&path);
        let msg = crate::trtx_runtime::import_trtx_sdk(&state.data_dir, &src)
            .await.map_err(|e| e.to_string())?;
        tracing::info!("TRT-RTX SDK import: {msg}");
        let dd = state.data_dir.clone();
        let ok = tokio::task::spawn_blocking(move || crate::trtx_runtime::activate_if_provisioned(&dd))
            .await.map_err(|e| e.to_string())?;
        if !ok {
            return Err(format!("{msg}, but the runtime failed its activation canary — ort rc.12 needs the TensorRT-for-RTX 1.3 SDK specifically"));
        }
        Ok(nv_status(&state.data_dir))
    }
    #[cfg(not(windows))]
    {
        let _ = (state, path);
        Err("NVIDIA Performance Pack is Windows-only".into())
    }
}

/// Live-workload benchmark: snapshots the per-model latency table, waits
/// `seconds`, snapshots again, and returns the DELTA (only inferences that
/// actually ran in the window). Honest numbers from the real pipeline.
#[tauri::command]
pub async fn benchmark_inference(seconds: Option<u64>) -> Result<Vec<crate::inference::InferStatRow>, String> {
    let secs = seconds.unwrap_or(20).clamp(5, 120);
    let before: std::collections::HashMap<String, (u64, f32)> =
        crate::inference::infer_stats_snapshot().into_iter()
            .map(|r| (r.model.clone(), (r.count, r.avg_ms))).collect();
    tokio::time::sleep(std::time::Duration::from_secs(secs)).await;
    let after = crate::inference::infer_stats_snapshot();
    Ok(after.into_iter().filter_map(|r| {
        let (c0, a0) = before.get(&r.model).copied().unwrap_or((0, 0.0));
        let dc = r.count.saturating_sub(c0);
        if dc == 0 { return None; }
        // window average from cumulative sums: (sumAfter - sumBefore) / dc
        let sum_after = r.avg_ms as f64 * r.count as f64;
        let sum_before = a0 as f64 * c0 as f64;
        Some(crate::inference::InferStatRow {
            model: r.model,
            count: dc,
            avg_ms: ((sum_after - sum_before) / dc as f64) as f32,
            p95_ms: r.p95_ms,
        })
    }).collect())
}

// ─── NVR disk projection ─────────────────────────────────────────────────────

/// Measured recording rate + projected retention, so the Storage settings can
/// show "at your camera count the 50 GB cap holds ~N days" instead of letting
/// retention silently collapse when cameras are added (16 cams ≈ 16× the write
/// rate against the same cap; the disk guard then deletes continuously).
#[derive(Serialize)]
pub struct DiskProjection {
    /// GB/day measured from segments indexed in the last 24 h (real rate, not an estimate).
    pub gb_per_day: f64,
    /// The `nvr_max_gb` cap the disk guard prunes to.
    pub cap_gb: u32,
    /// The user's retention_days setting (what they THINK they keep).
    pub retention_days_setting: u32,
    /// Days of footage the cap can actually hold at the measured rate.
    pub projected_days_at_cap: f64,
    /// Cameras that produced at least one segment in the last 24 h.
    pub recording_cams: i64,
    /// Active retain mode ("always" | "motion_only" | "events_only") — the
    /// measured rate reflects it once pruning has cycled.
    pub record_mode: String,
}

#[tauri::command]
pub async fn nvr_disk_projection(state: State<'_, Arc<AppState>>) -> Result<DiskProjection, String> {
    // strftime with an explicit 'T' separator: rows store RFC3339 ("…T…"), and
    // datetime('now') emits a SPACE — the mixed-format string compare silently
    // widens the window by up to a day.
    let (bytes_24h, cams): (i64, i64) = sqlx::query_as(
        "SELECT COALESCE(SUM(size_bytes),0), COUNT(DISTINCT cam_id) FROM nvr_segments \
         WHERE started_at > strftime('%Y-%m-%dT%H:%M:%S','now','-24 hours')")
        .fetch_one(&state.db).await.map_err(|e| e.to_string())?;
    let s = state.settings.read().await;
    let gb_per_day = bytes_24h as f64 / 1e9;
    let projected = if gb_per_day > 0.01 { (s.nvr_max_gb as f64 / gb_per_day).min(9999.0) } else { 9999.0 };
    Ok(DiskProjection {
        gb_per_day,
        cap_gb: s.nvr_max_gb,
        retention_days_setting: s.retention_days,
        projected_days_at_cap: projected,
        recording_cams: cams,
        record_mode: s.nvr_record_mode.clone(),
    })
}

// ─── Crash auto-restart (keep-alive Scheduled Task) ──────────────────────────

/// Register or remove the "Anivar KeepAlive" Windows Scheduled Task: every
/// 5 minutes it starts the app `--headless` IF the process isn't already
/// running (idempotent — never double-launches). This is the recovery lane for
/// hard NATIVE crashes (GPU driver aborts like 0xc0000409) that the in-process
/// panic hook can't catch, and for plain "someone closed it". Windows-native:
/// no service, no extra daemon.
#[cfg(windows)]
pub(crate) async fn set_keepalive_task(enabled: bool) -> Result<(), String> {
    const TASK: &str = "Anivar KeepAlive";
    /// Task names used before the current one. A task registered by an older
    /// build survives the rename and keeps trying to launch an exe that no longer
    /// exists, so every past name is removed on BOTH paths here. Boot re-asserts
    /// this function whenever keepalive is on, which makes the cleanup automatic.
    const LEGACY_TASKS: &[&str] = &["Nivar KeepAlive", "Anvil NVR KeepAlive", "SecureCam KeepAlive"];
    for legacy in LEGACY_TASKS {
        let _ = crate::proc::tokio_cmd("schtasks")
            .args(["/Delete", "/TN", legacy, "/F"])
            .output().await;
    }

    if !enabled {
        let _ = crate::proc::tokio_cmd("schtasks")
            .args(["/Delete", "/TN", TASK, "/F"])
            .output().await;
        tracing::info!("keepalive: scheduled task removed");
        return Ok(());
    }
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    // Derive the process name from the binary rather than hardcoding it: a
    // literal here silently stops matching the moment the binary is renamed,
    // and a keepalive that never matches relaunches a SECOND copy every 5 min.
    let proc_name = exe.file_stem().unwrap_or_default().to_string_lossy().to_string();
    // The action re-checks at run time, so a normal user-quit stays quit for at
    // most one 5-min window — acceptable for an appliance, documented in the UI.
    let ps = format!(
        "if(-not(Get-Process -Name '{proc_name}' -ErrorAction SilentlyContinue)){{Start-Process -FilePath '{}' -ArgumentList '--headless'}}",
        exe.display()
    );
    let tr = format!("powershell.exe -NoProfile -WindowStyle Hidden -Command \"{ps}\"");
    let out = crate::proc::tokio_cmd("schtasks")
        .args(["/Create", "/F", "/TN", TASK, "/SC", "MINUTE", "/MO", "5", "/TR", &tr])
        .output().await.map_err(|e| e.to_string())?;
    if out.status.success() {
        tracing::info!("keepalive: scheduled task registered ({})", exe.display());
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&out.stderr).trim().to_string())
    }
}

#[cfg(not(windows))]
pub(crate) async fn set_keepalive_task(_enabled: bool) -> Result<(), String> {
    Err("crash auto-restart is Windows-only for now".into())
}
