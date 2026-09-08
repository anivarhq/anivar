//! Managed NVIDIA CUDA + cuDNN runtime.
//!
//! ORT's CUDA execution provider needs the CUDA 12 + cuDNN 9 runtime libraries at
//! load time, which most machines don't have. Rather than bundle ~1.5 GB into the
//! installer we fetch NVIDIA's public redistributable wheels once, on first launch.
//!
//! Compiled for the Windows `cuda` build variant AND for Linux, where it is the
//! ONLY way to reach the GPU: Linux has no DirectML, so without a managed CUDA
//! runtime every Linux install silently ran detection on the CPU.
//!
//! The download/verify/extract/preload mechanics live in [`crate::provision`] —
//! this file is now just the package list and the marker set. It previously
//! carried private copies of `resolve_wheel_url`, `extract_dlls` and
//! `prepend_dll_path`, which is how the Unix `LD_LIBRARY_PATH` no-op survived
//! here long after it was understood elsewhere.
#![cfg(any(feature = "cuda", target_os = "linux"))]

use std::path::{Path, PathBuf};

use anyhow::Result;

use crate::provision::{self, Requirement};

/// NVIDIA redistributable wheels carrying the runtime libraries the ORT CUDA EP
/// links. cuDNN is the big one; cuBLAS/cudart are small.
const WHEELS: &[&str] = &[
    "nvidia-cuda-runtime-cu12", // cudart
    "nvidia-cublas-cu12",       // cublas + cublasLt (cublas statically imports cublasLt)
    "nvidia-cudnn-cu12",        // cuDNN 9 dispatch family
];

/// Every wheel's marker must be present, not just the last one: a run that died
/// after cuBLAS used to count as provisioned forever while the EP kept failing.
#[cfg(windows)]
const MARKERS: &[&str] = &["cudart", "cublas", "cudnn"];
#[cfg(not(windows))]
const MARKERS: &[&str] = &["libcudart", "libcublas", "libcudnn"];

/// Ensure the CUDA + cuDNN runtime is present and loadable, downloading it once
/// if needed. Cheap no-op afterwards. Best-effort by contract: on failure the
/// CUDA EP simply fails to register and ORT falls back, so a download hiccup
/// never breaks inference.
pub async fn ensure_cuda_runtime(data_dir: &Path) -> Result<PathBuf> {
    let dir = data_dir.join("cuda");
    provision::ensure_lib_pack(WHEELS, &dir, &Requirement::Markers(MARKERS), "CUDA runtime").await?;
    provision::make_libs_loadable(&dir);
    Ok(dir)
}
