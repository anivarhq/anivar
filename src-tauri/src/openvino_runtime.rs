//! Managed Intel OpenVINO runtime for the `openvino` build variant.
//!
//! ORT's OpenVINO EP (compiled only in `--features openvino` builds — it needs an
//! ONNX Runtime binary built with OpenVINO, which the prebuilt dists don't
//! include) loads Intel's runtime libraries at session build. Same managed-wheel
//! pattern as [`crate::cuda_runtime`]: fetch Intel's `openvino` wheel once,
//! extract the libraries, make them loadable. Best-effort — on failure the EP
//! registration fails gracefully and the chain falls through to DirectML/CPU.
//!
//! This file used to carry its own copies of `resolve_wheel_url`, `extract_dlls`
//! and `prepend_dll_path`. Those copies were where a `win_amd64`-only resolver
//! (making the module accidentally Windows-only), a whole-transfer timeout, and
//! the dead `LD_LIBRARY_PATH` branch all outlived their fixes elsewhere. The
//! mechanics now live once, in [`crate::provision`].
#![cfg(feature = "openvino")]

use std::path::{Path, PathBuf};

use anyhow::Result;

use crate::provision::{self, Requirement};

const WHEELS: &[&str] = &["openvino"];

/// The runtime's own library is the marker — `openvino.dll` / `libopenvino.so`.
const MARKERS: &[&str] = if cfg!(windows) { &["openvino"] } else { &["libopenvino"] };

pub(crate) async fn ensure_openvino_runtime(data_dir: &Path) -> Result<PathBuf> {
    let dir = data_dir.join("openvino");
    provision::ensure_lib_pack(WHEELS, &dir, &Requirement::Markers(MARKERS), "OpenVINO runtime").await?;
    provision::make_libs_loadable(&dir);
    Ok(dir)
}
