//! Managed NVIDIA TensorRT-RTX runtime — the "NVIDIA Performance Pack".
//!
//! ORT's NV TensorRT RTX execution provider (compiled into our Windows build
//! via ort's `nvrtx` feature) JIT-compiles models into RTX-optimized engines —
//! NVIDIA's replacement for both classic TensorRT (~2.6 GB of deps) and
//! DirectML on RTX GPUs (~+50% throughput claimed). The EP dlopens the
//! TensorRT-for-RTX runtime at session build; those DLLs (~200 MB) are NOT
//! bundled — this module downloads them once from NVIDIA's PyPI index
//! (SHA-256 pinned), exactly like the cuda/ffmpeg/cloudflared managed
//! binaries. Without the pack, the EP registration fails gracefully and the
//! chain falls through to DirectML — byte-identical to pre-pack behavior.
//!
//! A successful REGISTER CANARY (not just file presence) gates activation:
//! the gpu_infer_guard serialization is only relaxed when TensorRT-RTX is
//! genuinely the active EP, because the concurrency crash it avoids is
//! DirectML-specific (0xc0000409 under concurrent Run).

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::{anyhow, Result};

/// Pinned runtime wheel (NVIDIA PyPI index, verified by SHA-256).
#[allow(dead_code)] // TRT-RTX lane is unreachable on the pinned ort; kept for the re-arm
const TRTX_WHEEL_URL: &str =
    "https://pypi.nvidia.com/tensorrt-rtx-cu12-libs/tensorrt_rtx_cu12_libs-1.5.0.114-py3-none-win_amd64.whl";
#[allow(dead_code)]
const TRTX_WHEEL_SHA256: &str = "bb573f3f45e4fc20c060d87a8261a1bb04be5c161d4491244cfa66562fb1e4ae";
/// cudart (small, resolved from pypi.org like cuda_runtime does) — TRT-RTX is
/// otherwise self-contained (no cuDNN/cuBLAS — that's its selling point).
#[allow(dead_code)]
const CUDART_PKG: &str = "nvidia-cuda-runtime-cu12";

// ─── Classic TensorRT pack (PUBLIC wheels — no NVIDIA login) ─────────────────
//
// ONNX Runtime 1.24's TensorRT EP pairs with TensorRT 10.9 + CUDA 12 + cuDNN 9
// (official requirements). All four runtimes ship as public wheels; every URL is
// SHA-256 pinned. ~1.85 GB download, one-time. The TensorRT EP also needs the
// CUDA EP's runtime for unsupported-node fallback, so cuDNN/cuBLAS ride along —
// which makes the plain CUDA EP a free second lane from the same pack.
/// `marker` is the lowercase DLL-name prefix that wheel contributes. It is both
/// the per-wheel resume point (already extracted → skip the re-download) and one
/// fifth of the completeness test in [`is_trt_provisioned`].
const TRT_PACK_WHEELS: &[(&str, &str, &str, &str)] = &[
    // (label, url, sha256, marker)
    ("TensorRT 10.9 libs",
     "https://pypi.nvidia.com/tensorrt-cu12-libs/tensorrt_cu12_libs-10.9.0.34-py2.py3-none-win_amd64.whl",
     "e43d38cc380d615bf8afab637c265f01a9452dc23deb9b9dd47b7662edf24531",
     "nvinfer"),
    ("cuDNN 9",
     "https://files.pythonhosted.org/packages/29/28/2c9a2a97a8b3fedcf74a14f38fd5edfae12274380a829fdc6b16ce29be4c/nvidia_cudnn_cu12-9.24.0.43-py3-none-win_amd64.whl",
     "cbd41a0ab084422c936dc9fb2fc89be5ea9a85bc421c6f23d0243bdfc945fbef",
     "cudnn"),
    ("cuBLAS 12",
     "https://files.pythonhosted.org/packages/20/e2/fc9a0e985249d873150276d5afb02e39a66817fedbf1a385724393e505ed/nvidia_cublas_cu12-12.9.2.10-py3-none-win_amd64.whl",
     "623f43027d40d44ceadf0043f002bd25cf353e8f13ce90b9a87057019f560661",
     "cublas"),
    // cuFFT — the CUDA EP's onnxruntime_providers_cuda.dll hard-imports
    // cufft64_11.dll (verified by PE import scan); without it every NVIDIA EP
    // fails to load ("cufft64_11.dll is missing", error 126).
    ("cuFFT 11",
     "https://files.pythonhosted.org/packages/20/ee/29955203338515b940bd4f60ffdbc073428f25ef9bfbce44c9a066aedc5c/nvidia_cufft_cu12-11.4.1.4-py3-none-win_amd64.whl",
     "8e5bfaac795e93f80611f807d42844e8e27e340e0cde270dcb6c65386d795b80",
     "cufft"),
    ("CUDA runtime",
     "https://files.pythonhosted.org/packages/59/df/e7c3a360be4f7b93cee39271b792669baeb3846c58a4df6dfcf187a7ffab/nvidia_cuda_runtime_cu12-12.9.79-py3-none-win_amd64.whl",
     "8e018af8fa02363876860388bd10ccb89eb9ab8fb0aa749aaf58430a9f7c4891",
     "cudart"),
];

/// Pack wheels scatter their DLLs (`tensorrt_rtx_libs/`, `bin/`, …), so unlike
/// the CUDA lib packs we keep every DLL wherever it sits in the archive.
fn is_pack_dll(entry: &str) -> bool { entry.to_lowercase().ends_with(".dll") }

/// Which NVIDIA execution provider passed its canary at activation.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum NvEp { Nvrtx, Tensorrt, Cuda }

impl NvEp {
    pub(crate) fn label(self) -> &'static str {
        match self {
            NvEp::Nvrtx => "TensorRT-RTX (NVIDIA)",
            NvEp::Tensorrt => "TensorRT (NVIDIA)",
            NvEp::Cuda => "CUDA (NVIDIA)",
        }
    }
}

/// Set once activation succeeds: (pack dir, engine-cache dir).
static ACTIVE: OnceLock<(PathBuf, PathBuf)> = OnceLock::new();
/// The NVIDIA EP that won the canary (None = DirectML/CPU as before).
static ACTIVE_EP: OnceLock<NvEp> = OnceLock::new();

#[allow(dead_code)] // read by diagnostics; kept beside the EP state it reports
pub(crate) fn is_active() -> bool { ACTIVE_EP.get().is_some() }
pub(crate) fn active_ep() -> Option<NvEp> { ACTIVE_EP.get().copied() }
pub(crate) fn cache_dir() -> Option<PathBuf> { ACTIVE.get().map(|(_, c)| c.clone()) }

fn pack_dir(data_dir: &Path) -> PathBuf { data_dir.join("trtx") }
fn trt_pack_dir(data_dir: &Path) -> PathBuf { data_dir.join("trt") }

/// Lowercase DLL names present in a pack dir.
fn dll_names(dir: &Path) -> Vec<String> {
    std::fs::read_dir(dir)
        .map(|d| d.filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().to_lowercase())
            .collect())
        .unwrap_or_default()
}

/// Classic-TRT pack provisioned = **every** wheel's marker DLL is on disk.
///
/// Testing only for nvinfer (the FIRST wheel) meant a download that died after
/// wheel 1 reported "provisioned": `ensure_trt_pack` then short-circuited on
/// every later attempt while the EP kept failing its canary for the missing
/// cuDNN/cuBLAS/cuFFT — a permanently half-installed pack with no way back to
/// DirectML-or-repair from the UI. Presence-of-all also means an ALREADY
/// complete legacy install still reads as provisioned (no forced re-download).
pub(crate) fn is_trt_provisioned(data_dir: &Path) -> bool {
    let names = dll_names(&trt_pack_dir(data_dir));
    !names.is_empty()
        && TRT_PACK_WHEELS.iter().all(|(_, _, _, marker)| names.iter().any(|n| n.starts_with(marker)))
}

/// Disk the extracted pack needs. The wheels are ~1.85 GB compressed but expand
/// to **4.6 GB of DLLs** (measured: `nvinfer_builder_resource_10.dll` alone is
/// 1.97 GB). Checking up front turns "the install died at 97%" into a sentence
/// the user can act on before spending the bandwidth.
const TRT_PACK_DISK_BYTES: u64 = 5_200_000_000;

/// Approximate grand total of all five wheels (~1.85 GB), used as the denominator
/// for the onboarding progress bar so it advances smoothly without a HEAD round-trip.
const TRT_PACK_TOTAL_BYTES: u64 = 1_986_000_000;

/// Download + verify + extract the classic TensorRT pack (one-time, ~1.85 GB).
/// Every wheel is SHA-256 pinned; a mismatch aborts the install (any wheels
/// already written stay put and are skipped when the user retries — the pack
/// only counts as provisioned once ALL markers are on disk).
///
/// Emits throttled `accel:progress` events (`{percent, downloaded, total, label,
/// step, steps}`) so the onboarding "Enable max performance" step can show a live
/// bar for the multi-minute download instead of a blind spinner.
pub(crate) async fn ensure_trt_pack(data_dir: &Path, app: &tauri::AppHandle) -> Result<PathBuf> {
    use tauri::Emitter;

    let dir = trt_pack_dir(data_dir);
    tokio::fs::create_dir_all(&dir).await.ok();
    if is_trt_provisioned(data_dir) { return Ok(dir); }

    // One installer at a time. Two clicks on "Enable max performance" used to run
    // two 1.85 GB downloads that File::create the same DLLs on top of each other.
    static INSTALLING: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
    let _lock = INSTALLING.try_lock()
        .map_err(|_| anyhow!("the performance pack is already installing — watch the progress bar"))?;
    // Whoever held the lock may have JUST finished it.
    if is_trt_provisioned(data_dir) { return Ok(dir); }

    // Space check BEFORE the bandwidth. Nothing is more annoying than a 1.85 GB
    // download that dies during extraction because the volume was full.
    if let Some(free) = crate::provision::free_space_bytes(&dir) {
        if free < TRT_PACK_DISK_BYTES {
            anyhow::bail!(
                "the NVIDIA performance pack needs ~{:.1} GB free on {} (the wheels are 1.85 GB \
                 but expand to ~4.6 GB of runtime DLLs) — only {:.1} GB available. Free some \
                 space and try again.",
                TRT_PACK_DISK_BYTES as f64 / 1e9,
                dir.display(),
                free as f64 / 1e9,
            );
        }
    }

    tracing::info!("TensorRT pack: downloading TRT 10.9 + CUDA 12 + cuDNN 9 runtimes (~1.85 GB, one-time)…");
    // Total-deadline timeouts kill a slow-but-healthy 1.85 GB transfer; bound the
    // connect and the per-read stall instead.
    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(30))
        .read_timeout(std::time::Duration::from_secs(180))
        .build()?;

    let steps = TRT_PACK_WHEELS.len() as u32;
    let mut cumulative: u64 = 0;
    let emit = |downloaded: u64, label: &str, step: u32| {
        let pct = ((downloaded.saturating_mul(100) / TRT_PACK_TOTAL_BYTES).min(100)) as u8;
        let _ = app.emit("accel:progress", serde_json::json!({
            "percent": pct,
            "downloaded": downloaded,
            "total": TRT_PACK_TOTAL_BYTES,
            "label": label,
            "step": step,
            "steps": steps,
        }));
    };
    emit(0, "Starting…", 0);

    let present = dll_names(&dir);
    for (i, (label, url, sha, marker)) in TRT_PACK_WHEELS.iter().enumerate() {
        let step = (i as u32) + 1;
        // Resume: a wheel already extracted by a previous (failed) attempt is not
        // re-downloaded, so retrying a pack that died on wheel 4 costs one wheel,
        // not 1.85 GB again.
        // ponytail: marker-DLL granularity — a crash mid-extract of one wheel can
        // leave that wheel short; delete the trt/ folder to force a clean pull.
        if present.iter().any(|n| n.starts_with(marker)) {
            tracing::info!("TensorRT pack: {label} already present — skipping");
            cumulative += TRT_PACK_TOTAL_BYTES / steps as u64;
            emit(cumulative.min(TRT_PACK_TOTAL_BYTES), label, step);
            continue;
        }
        tracing::info!("TensorRT pack: fetching {label}…");
        // Stream into memory (extract_dlls needs the whole zip) while emitting
        // throttled byte progress — 64 KB delta OR 250 ms, whichever first.
        let base = cumulative;
        let mut last_emit_bytes = 0u64;
        let mut last_emit_at = std::time::Instant::now();
        let buf = crate::provision::fetch(&client, url, label, |got| {
            if got.saturating_sub(last_emit_bytes) >= 64 * 1024
                || last_emit_at.elapsed().as_millis() >= 250
            {
                emit(base + got, label, step);
                last_emit_bytes = got;
                last_emit_at = std::time::Instant::now();
            }
        }).await?;
        cumulative = base + buf.len() as u64;
        crate::provision::verify_sha256(&buf, sha, label)?;
        let n = crate::provision::unpack_zip(&buf, &dir, is_pack_dll)?;
        tracing::info!("TensorRT pack: {label} — extracted {n} dll(s)");
        emit(cumulative, label, step);
    }
    if !is_trt_provisioned(data_dir) {
        // Name what's actually missing — "no nvinfer dll" was misleading once the
        // completeness test covered all five wheels.
        let have = dll_names(&dir);
        let missing: Vec<&str> = TRT_PACK_WHEELS.iter()
            .map(|(_, _, _, m)| *m)
            .filter(|m| !have.iter().any(|h| h.starts_with(m)))
            .collect();
        return Err(anyhow!(
            "TensorRT pack: install incomplete after extraction — missing {}. \
             Re-run the install; wheels already on disk are skipped.",
            missing.join(", ")
        ));
    }
    emit(TRT_PACK_TOTAL_BYTES, "Finalizing…", steps);
    Ok(dir)
}

/// Provisioned = the TensorRT-RTX runtime DLL is on disk.
pub(crate) fn is_provisioned(data_dir: &Path) -> bool {
    std::fs::read_dir(pack_dir(data_dir))
        .map(|d| d.filter_map(|e| e.ok())
            .any(|e| e.file_name().to_string_lossy().to_lowercase().starts_with("tensorrt_rtx")))
        .unwrap_or(false)
}

/// Download + verify + extract the runtime pack (one-time, ~200 MB).
#[allow(dead_code)] // entry point for the TRT-RTX lane; see the note on TRTX_WHEEL_URL
pub(crate) async fn ensure_trtx_pack(data_dir: &Path) -> Result<PathBuf> {
    let dir = pack_dir(data_dir);
    tokio::fs::create_dir_all(&dir).await.ok();
    if trtx_requirement_met(data_dir) { return Ok(dir); }

    // REFUSE TO DOWNLOAD A RUNTIME OUR OWN BINARY CANNOT LOAD.
    //
    // The shipped provider links an exact soname; the pinned wheel ships another.
    // We used to fetch ~200 MB anyway, write it to disk, report "provisioned",
    // and then fall silently through to the 1.85 GB classic pack. pip would have
    // said "no matching distribution" and stopped — so do that.
    if let Some(need) = trtx_required_version() {
        let offered = TRTX_WHEEL_URL
            .rsplit('/').next().unwrap_or("")
            .split('-').nth(1).unwrap_or("");           // tensorrt_rtx_cu12_libs-1.5.0.114-...
        if !offered.starts_with(&format!("{need}.")) {
            anyhow::bail!(
                "TensorRT-RTX needs runtime {need}.x (our provider links {}), but the public \
                 index only offers {offered} — NVIDIA never published a {need}.x wheel. \
                 Install the matching TensorRT-for-RTX SDK via 'Import SDK', or use the \
                 TensorRT pack instead. Skipping a download that could not be loaded.",
                trtx_required_dlls().join(", ")
            );
        }
    }

    tracing::info!("TRT-RTX pack: downloading TensorRT-for-RTX runtime (~200 MB, one-time)…");
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(1800))
        .build()?;

    // Main wheel — SHA-256 pinned.
    let bytes = crate::provision::fetch(&client, TRTX_WHEEL_URL, "TRT-RTX wheel", |_| {}).await?;
    crate::provision::verify_sha256(&bytes, TRTX_WHEEL_SHA256, "TRT-RTX wheel")?;
    let n = crate::provision::unpack_zip(&bytes, &dir, is_pack_dll)?;
    tracing::info!("TRT-RTX pack: extracted {n} runtime dll(s)");

    // cudart companion (few MB; latest win_amd64 from pypi.org JSON).
    if let Some((url, sha)) = crate::provision::resolve_wheel(&client, CUDART_PKG).await {
        if let Ok(b) = crate::provision::fetch(&client, &url, CUDART_PKG, |_| {}).await {
            let ok = sha.map(|s| crate::provision::verify_sha256(&b, &s, CUDART_PKG).is_ok()).unwrap_or(true);
            if ok {
                let n = crate::provision::unpack_zip(&b, &dir, is_pack_dll).unwrap_or(0);
                tracing::info!("TRT-RTX pack: extracted {n} cudart dll(s)");
            }
        }
    }
    if !is_provisioned(data_dir) {
        return Err(anyhow!("TRT-RTX pack: no tensorrt_rtx dll found after extraction"));
    }
    Ok(dir)
}

/// Activate at boot / after install: put pack DLLs on the loader path, then run
/// REGISTER CANARIES in preference order — NVRTX → classic TensorRT → CUDA.
/// Only an EP whose registration actually succeeds on this machine is armed;
/// everything else falls through to DirectML exactly as before. The DirectML
/// serialization lock is relaxed ONLY for NVRTX (documented concurrent Run);
/// TensorRT/CUDA keep the lock until proven (upstream reports same-GPU
/// multi-session crashes) — even locked, TRT drains the queue ~4x faster.
pub(crate) fn activate_if_provisioned(data_dir: &Path) -> bool {
    if ACTIVE_EP.get().is_some() { return true; }
    if !crate::inference::has_nvidia_adapter() {
        if is_provisioned(data_dir) || is_trt_provisioned(data_dir) {
            tracing::info!("NVIDIA pack present but no NVIDIA adapter — staying on DirectML");
        }
        return false;
    }

    // 1) TensorRT-RTX (SDK import / future unblocked wheel) — best: JIT-fast,
    //    concurrent sessions documented. Gated on the provider's OWN linked
    //    soname being present, not merely "a tensorrt_rtx file exists": a pack
    //    holding 1.5 DLLs for a provider that links 1.3 can never register, and
    //    burning a canary on it just hides the real reason behind a warning.
    if trtx_requirement_met(data_dir) {
        let dir = pack_dir(data_dir);
        crate::provision::make_libs_loadable(&dir);
        let cache = data_dir.join("trtx_cache");
        std::fs::create_dir_all(&cache).ok();
        let ok = (|| -> Result<bool> {
            use ort::ep::ExecutionProvider;
            let mut b = ort::session::Session::builder()?;
            let ep = ort::ep::nvrtx::NVRTX::default()
                .with_device_id(0)
                .with_runtime_cache_path(cache.to_string_lossy());
            Ok(ep.register(&mut b).is_ok())
        })().unwrap_or(false);
        if ok {
            let _ = ACTIVE.set((dir, cache));
            let _ = ACTIVE_EP.set(NvEp::Nvrtx);
            crate::inference::set_active_gpu_ep_nvrtx();
            tracing::info!("TensorRT-RTX ACTIVE — RTX-optimized inference, concurrent GPU sessions enabled");
            return true;
        }
        tracing::warn!("TRT-RTX pack present but EP registration failed — trying classic TensorRT. Known cause: ort rc.12's provider bridge links tensorrt_rtx_1_3.dll (login-gated SDK); public wheels are 1.4+.");
    }

    // 2) Classic TensorRT (public pack) — engine-cached, fp16.
    if is_trt_provisioned(data_dir) {
        let dir = trt_pack_dir(data_dir);
        crate::provision::make_libs_loadable(&dir);
        let ecache = data_dir.join("trt_cache");
        std::fs::create_dir_all(&ecache).ok();
        let trt_ok = (|| -> Result<bool> {
            use ort::ep::ExecutionProvider;
            let mut b = ort::session::Session::builder()?;
            let ep = ort::ep::tensorrt::TensorRT::default().with_device_id(0);
            Ok(ep.register(&mut b).is_ok())
        })().unwrap_or(false);
        if trt_ok {
            let _ = ACTIVE.set((dir, ecache));
            let _ = ACTIVE_EP.set(NvEp::Tensorrt);
            tracing::info!("TensorRT ACTIVE (classic, TRT 10.9) — engine-cached inference; first model load builds engines (one-time, minutes)");
            return true;
        }
        tracing::warn!("TensorRT pack present but TRT EP registration failed — trying CUDA EP");

        // 3) CUDA EP — same pack carries cudart/cuBLAS/cuDNN.
        let cuda_ok = (|| -> Result<bool> {
            use ort::ep::ExecutionProvider;
            let mut b = ort::session::Session::builder()?;
            let ep = ort::ep::cuda::CUDA::default().with_device_id(0);
            Ok(ep.register(&mut b).is_ok())
        })().unwrap_or(false);
        if cuda_ok {
            let _ = ACTIVE.set((dir, ecache));
            let _ = ACTIVE_EP.set(NvEp::Cuda);
            tracing::info!("CUDA EP ACTIVE — NVIDIA-native inference (TensorRT EP unavailable on this driver)");
            return true;
        }
        tracing::warn!("NVIDIA pack present but no NVIDIA EP passed its canary — staying on DirectML");
    }
    false
}

/// Import a user-downloaded TensorRT-RTX SDK (zip or extracted folder): copy its
/// DLLs into the RTX pack dir, then re-run activation. ort rc.12 needs the 1.3
/// SDK (free NVIDIA developer login). Canary decides; a bad import can't crash.
pub(crate) async fn import_trtx_sdk(data_dir: &Path, source: &Path) -> Result<String> {
    let dir = pack_dir(data_dir);
    tokio::fs::create_dir_all(&dir).await.ok();
    let mut copied = 0usize;
    if source.is_file() && source.extension().is_some_and(|e| e.eq_ignore_ascii_case("zip")) {
        let bytes = tokio::fs::read(source).await?;
        copied = crate::provision::unpack_zip(&bytes, &dir, is_pack_dll)?;
    } else if source.is_dir() {
        // Recursive walk: the SDK layout nests DLLs under lib/ or bin/.
        let mut stack = vec![source.to_path_buf()];
        while let Some(d) = stack.pop() {
            let Ok(rd) = std::fs::read_dir(&d) else { continue };
            for ent in rd.flatten() {
                let p = ent.path();
                if p.is_dir() { stack.push(p); }
                else if p.extension().is_some_and(|e| e.eq_ignore_ascii_case("dll")) {
                    if let Some(name) = p.file_name() {
                        std::fs::copy(&p, dir.join(name))?;
                        copied += 1;
                    }
                }
            }
        }
    } else {
        return Err(anyhow!("select the downloaded SDK .zip or its extracted folder"));
    }
    if copied == 0 {
        return Err(anyhow!("no DLLs found in the selected SDK"));
    }
    if !is_provisioned(data_dir) {
        return Err(anyhow!("copied {copied} dll(s) but no tensorrt_rtx runtime among them — is this the TensorRT-for-RTX SDK?"));
    }
    Ok(format!("imported {copied} dll(s)"))
}

// ─── Self-describing runtime requirements ────────────────────────────────────
//
// A provider DLL already states which runtime it needs, in its PE import table —
// exactly as a Python wheel states it in `requires_dist`. Hardcoding that
// requirement in Rust is the `pip install --no-deps <hand-typed-url>` mistake: it
// silently rots the moment ORT is bumped, which is how we ended up downloading a
// TensorRT-RTX 1.5 wheel for a provider that imports `tensorrt_rtx_1_3.dll` and
// then falling through to a 1.85 GB pack without a word. So: read it, don't guess.

/// DLL names in `dll`'s PE import directory. `None` if the file isn't a readable
/// PE — callers then skip the requirement check rather than blocking a lane.
pub(crate) fn pe_imports(dll: &Path) -> Option<Vec<String>> {
    let d = std::fs::read(dll).ok()?;
    let g32 = |o: usize| -> Option<u32> {
        Some(u32::from_le_bytes(d.get(o..o + 4)?.try_into().ok()?))
    };
    let g16 = |o: usize| -> Option<u16> {
        Some(u16::from_le_bytes(d.get(o..o + 2)?.try_into().ok()?))
    };

    let pe = g32(0x3c)? as usize;
    if d.get(pe..pe + 4)? != b"PE\0\0" { return None; }
    let nsec = g16(pe + 6)? as usize;
    let optsz = g16(pe + 20)? as usize;
    let opt = pe + 24;
    // DataDirectory sits after the optional header's fixed part: 112 bytes for
    // PE32+ (0x20b), 96 for PE32.
    let dd = opt + if g16(opt)? == 0x20b { 112 } else { 96 };
    let (imp_rva, _size) = (g32(dd + 8)?, g32(dd + 12)?); // DataDirectory[1] = imports
    if imp_rva == 0 { return Some(Vec::new()); }

    // Section table → RVA-to-file-offset mapping.
    let sec_base = opt + optsz;
    let mut secs: Vec<(u32, u32, u32)> = Vec::with_capacity(nsec);
    for i in 0..nsec {
        let s = sec_base + i * 40;
        secs.push((g32(s + 12)?, g32(s + 16)?, g32(s + 20)?)); // va, raw size, raw ptr
    }
    let rva2off = |rva: u32| -> Option<usize> {
        secs.iter()
            .find(|(va, sz, _)| rva >= *va && rva < va.saturating_add(*sz))
            .map(|(va, _, ptr)| (ptr + (rva - va)) as usize)
    };

    let mut names = Vec::new();
    let mut o = rva2off(imp_rva)?;
    loop {
        let desc = d.get(o..o + 20)?;
        if desc.iter().all(|b| *b == 0) { break; } // null terminator entry
        let name_rva = g32(o + 12)?;
        if name_rva != 0 {
            if let Some(no) = rva2off(name_rva) {
                let end = d[no..].iter().position(|b| *b == 0).map(|e| no + e)?;
                if let Ok(s) = std::str::from_utf8(&d[no..end]) { names.push(s.to_string()); }
            }
        }
        o += 20;
        if names.len() > 512 { break; } // malformed-file guard
    }
    Some(names)
}

/// The TensorRT-RTX runtime DLLs our shipped provider actually links, e.g.
/// `["tensorrt_rtx_1_3.dll", "tensorrt_onnxparser_rtx_1_3.dll"]`.
/// Empty when the provider can't be read (then we don't gate on it).
pub(crate) fn trtx_required_dlls() -> Vec<String> {
    static REQ: OnceLock<Vec<String>> = OnceLock::new();
    REQ.get_or_init(|| {
        let Some(dir) = std::env::current_exe().ok().and_then(|e| e.parent().map(Path::to_path_buf))
        else { return Vec::new() };
        let provider = dir.join("onnxruntime_providers_nv_tensorrt_rtx.dll");
        pe_imports(&provider)
            .unwrap_or_default()
            .into_iter()
            .filter(|n| {
                let l = n.to_lowercase();
                l.starts_with("tensorrt_rtx") || l.starts_with("tensorrt_onnxparser_rtx")
            })
            .collect()
    }).clone()
}

/// Runtime version our RTX provider links, as `major.minor` (e.g. `"1.3"` from
/// `tensorrt_rtx_1_3.dll`). `None` when the provider can't be read.
pub(crate) fn trtx_required_version() -> Option<String> {
    let dll = trtx_required_dlls().into_iter().find(|n| n.to_lowercase().starts_with("tensorrt_rtx"))?;
    // tensorrt_rtx_1_3.dll → 1.3
    let stem = dll.to_lowercase();
    let stem = stem.strip_suffix(".dll")?.strip_prefix("tensorrt_rtx_")?;
    let (maj, min) = stem.split_once('_')?;
    Some(format!("{maj}.{min}"))
}

/// Is every runtime DLL the RTX provider links present in the pack dir?
/// This — not "some tensorrt_rtx file exists" — is what decides whether the lane
/// can possibly load.
pub(crate) fn trtx_requirement_met(data_dir: &Path) -> bool {
    let req = trtx_required_dlls();
    if req.is_empty() { return is_provisioned(data_dir); } // can't read provider → old behaviour
    let have = dll_names(&pack_dir(data_dir));
    req.iter().all(|r| have.iter().any(|h| h == &r.to_lowercase()))
}

#[cfg(test)]
mod pack_tests {
    use super::*;

    /// The regression that made a half-downloaded pack permanent: nvinfer alone
    /// used to read as "provisioned", so `ensure_trt_pack` short-circuited while
    /// the EP kept failing for want of cuDNN/cuBLAS/cuFFT. Every wheel's marker
    /// must be present — this fails if someone adds a wheel without a marker.
    /// The PE parser is the thing that keeps our "what runtime do we need?" answer
    /// honest across ORT bumps, so it is checked against the provider we actually
    /// ship. If this ever fails after an ort upgrade, the required TensorRT-RTX
    /// version changed — which is exactly the event we want to be told about,
    /// because it decides whether a public wheel can satisfy the lane at all.
    #[test]
    fn provider_declares_its_tensorrt_rtx_runtime() {
        let dll = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("binaries/onnx/onnxruntime_providers_nv_tensorrt_rtx.dll");
        // "Present" is not "usable" — the same rule the installer follows. The DLLs
        // are gitignored, and CI creates EMPTY placeholders so tauri-build's resource
        // check passes; a bare `is_file()` let a 0-byte file through to the PE parser,
        // which then panicked. Skip unless there is a real provider to inspect.
        if !crate::provision::Requirement::MinSize(64 * 1024).met(&dll) {
            return; // no real provider vendored in this checkout
        }
        let imports = pe_imports(&dll).expect("provider DLL should parse as PE");
        let rtx: Vec<_> = imports.iter()
            .filter(|n| n.to_lowercase().starts_with("tensorrt_rtx"))
            .collect();
        assert!(!rtx.is_empty(), "RTX provider must link a tensorrt_rtx runtime, got: {imports:?}");
        // Sanity on the version extractor that gates the download.
        let v = rtx[0].to_lowercase();
        let v = v.strip_suffix(".dll").unwrap().strip_prefix("tensorrt_rtx_").unwrap();
        assert!(v.contains('_'), "expected major_minor in {rtx:?}");
    }

    #[test]
    fn partial_pack_is_not_provisioned() {
        let data = std::env::temp_dir().join(format!("sc_trt_pack_test_{}", std::process::id()));
        let pack = data.join("trt");
        std::fs::create_dir_all(&pack).unwrap();

        assert!(!is_trt_provisioned(&data), "empty pack dir must not be provisioned");

        let mut markers = TRT_PACK_WHEELS.iter().map(|(_, _, _, m)| *m);
        let first = markers.next().unwrap();
        std::fs::write(pack.join(format!("{first}64_1.dll")), b"stub").unwrap();
        assert!(!is_trt_provisioned(&data), "one wheel is not an install");

        for m in markers {
            std::fs::write(pack.join(format!("{m}64_1.dll")), b"stub").unwrap();
        }
        assert!(is_trt_provisioned(&data), "all markers present = provisioned");

        std::fs::remove_dir_all(&data).ok();
    }
}
