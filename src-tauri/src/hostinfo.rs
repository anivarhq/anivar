//! In-process host probing — CPU/RAM (sysinfo), GPU adapters (DXGI), NVIDIA live
//! stats (NVML). This is how mature NVRs do telemetry (psutil + NVML bindings
//! read OS APIs in-process): **no subprocesses**. The previous implementation shelled
//! out to PowerShell `Get-Counter` (slow ~1s, locale-fragile, "telemetry offline"
//! when parsing hiccuped) and a dependency ran raw `nvidia-smi` (flashing console
//! windows on every Arsenal open). Everything here is a library call.

use std::sync::{Mutex, OnceLock};

// ─── CPU + RAM (sysinfo — the Rust psutil) ───────────────────────────────────────

/// Static host facts for the recommendation card / model-fit scoring.
pub struct HostCpuMem {
    pub total_ram_gb:     f64,
    pub available_ram_gb: f64,
    pub cpu_name:         String,
    pub cpu_cores:        usize,
}

fn sys() -> &'static Mutex<sysinfo::System> {
    static SYS: OnceLock<Mutex<sysinfo::System>> = OnceLock::new();
    SYS.get_or_init(|| Mutex::new(sysinfo::System::new_all()))
}

pub fn cpu_mem() -> HostCpuMem {
    let mut s = sys().lock().unwrap_or_else(|e| e.into_inner());
    s.refresh_memory();
    HostCpuMem {
        total_ram_gb:     s.total_memory() as f64 / 1024.0 / 1024.0 / 1024.0,
        available_ram_gb: s.available_memory() as f64 / 1024.0 / 1024.0 / 1024.0,
        cpu_name:         s.cpus().first().map(|c| c.brand().trim().to_string()).unwrap_or_default(),
        cpu_cores:        s.cpus().len(),
    }
}

/// Live CPU/RAM sample for the Telemetry panel. sysinfo computes usage as the delta
/// since the previous refresh, so a persistent `System` behind the OnceLock gives
/// real percentages from the second poll onward (first poll reads 0 — harmless).
pub fn live_cpu_mem() -> (f32, Vec<f32>, u64, u64) {
    let mut s = sys().lock().unwrap_or_else(|e| e.into_inner());
    s.refresh_cpu_usage();
    s.refresh_memory();
    let per_core: Vec<f32> = s.cpus().iter().map(|c| c.cpu_usage()).collect();
    let total = s.global_cpu_usage();
    let used_mb  = (s.total_memory() - s.available_memory()) / 1024 / 1024;
    let total_mb = s.total_memory() / 1024 / 1024;
    (total, per_core, used_mb, total_mb)
}

// ─── GPU adapters (DXGI — names/VRAM/vendor, no drivers/tools needed) ────────────

pub struct GpuAdapter {
    pub name:              String,
    pub vendor_id:         u32,   // 0x10DE NVIDIA · 0x1002 AMD · 0x8086 Intel
    pub dedicated_vram_gb: f64,
}

impl GpuAdapter {
    /// Discrete = a dedicated-VRAM part (integrated GPUs share system RAM).
    pub fn is_discrete(&self) -> bool {
        (self.vendor_id == 0x10DE || self.vendor_id == 0x1002) && self.dedicated_vram_gb >= 0.5
            || self.dedicated_vram_gb >= 1.0
    }
}

#[cfg(windows)]
pub fn gpu_adapters() -> Vec<GpuAdapter> {
    use windows::Win32::Graphics::Dxgi::{
        CreateDXGIFactory1, IDXGIFactory1, DXGI_ADAPTER_FLAG_SOFTWARE,
    };
    let mut out = Vec::new();
    unsafe {
        let Ok(factory) = CreateDXGIFactory1::<IDXGIFactory1>() else { return out };
        let mut i = 0u32;
        while let Ok(adapter) = factory.EnumAdapters1(i) {
            if let Ok(desc) = adapter.GetDesc1() {
                let software = (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32) != 0;
                if !software {
                    let end = desc.Description.iter().position(|&c| c == 0)
                        .unwrap_or(desc.Description.len());
                    out.push(GpuAdapter {
                        name:              String::from_utf16_lossy(&desc.Description[..end]),
                        vendor_id:         desc.VendorId,
                        dedicated_vram_gb: desc.DedicatedVideoMemory as f64 / 1024.0 / 1024.0 / 1024.0,
                    });
                }
            }
            i += 1;
        }
    }
    out
}

/// Non-Windows: no DXGI, so enumerate NVIDIA GPUs via NVML (works on Linux; on
/// macOS NVML init simply fails → empty, and `probe_host` handles Apple Silicon
/// with its Metal/unified-memory special case instead).
#[cfg(not(windows))]
pub fn gpu_adapters() -> Vec<GpuAdapter> {
    nvidia_live_full()
        .into_iter()
        .map(|(name, total_mb)| GpuAdapter {
            name,
            vendor_id: 0x10DE,
            dedicated_vram_gb: total_mb as f64 / 1024.0,
        })
        .collect()
}

/// (name, total VRAM MB) per NVIDIA device — used by the non-Windows adapter path.
#[cfg(not(windows))]
fn nvidia_live_full() -> Vec<(String, u64)> {
    let Some(nv) = nvml() else { return Vec::new() };
    let Ok(count) = nv.device_count() else { return Vec::new() };
    (0..count).filter_map(|i| {
        let dev = nv.device_by_index(i).ok()?;
        Some((dev.name().ok()?, dev.memory_info().map(|m| m.total / 1024 / 1024).unwrap_or(0)))
    }).collect()
}

// ─── NVIDIA live stats (NVML — what nvidia-smi itself uses, minus the process) ──

fn nvml() -> Option<&'static nvml_wrapper::Nvml> {
    static NVML: OnceLock<Option<nvml_wrapper::Nvml>> = OnceLock::new();
    NVML.get_or_init(|| nvml_wrapper::Nvml::init().ok()).as_ref()
}

/// (name, 3D-util %, dedicated memory in use MB) per NVIDIA GPU. Empty when no
/// NVIDIA driver — callers fall back to the DXGI adapter list with idle stats.
pub fn nvidia_live() -> Vec<(String, f32, u64)> {
    let Some(nv) = nvml() else { return Vec::new() };
    let Ok(count) = nv.device_count() else { return Vec::new() };
    (0..count).filter_map(|i| {
        let dev  = nv.device_by_index(i).ok()?;
        let name = dev.name().ok()?;
        let util = dev.utilization_rates().map(|u| u.gpu as f32).unwrap_or(0.0);
        let mem  = dev.memory_info().map(|m| m.used / 1024 / 1024).unwrap_or(0);
        Some((name, util, mem))
    }).collect()
}
