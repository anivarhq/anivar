//! One contract for every artifact the app downloads at runtime.
//!
//! We ship a small app and fetch the big pieces on demand — ffmpeg, go2rtc,
//! Ollama, ONNX Runtime GPU libraries, detection models. That had grown into
//! eight hand-rolled implementations of the same four steps, with three literal
//! copies of `resolve_wheel_url` / `extract_dlls` / `prepend_dll_path` and about
//! seven different answers to "is it installed?" (`>=1MB`, `>1MB`, `>0`,
//! `>=64KB`, and three marker scans).
//!
//! That duplication was not cosmetic. Fixing "a partial artifact counts as
//! installed" took five separate edits, and copies kept defects already repaired
//! elsewhere — `openvino_runtime` still had the `LD_LIBRARY_PATH` no-op long
//! after `cuda_runtime` was fixed.
//!
//! The rules encoded here, once:
//!   * timeouts bound the CONNECT and the per-read stall, never the whole
//!     transfer (a total deadline kills a slow-but-healthy 2 GB download),
//!   * every response is status-checked, so an HTML error page never lands on
//!     disk as a model,
//!   * transient failures retry with backoff; integrity failures never do,
//!   * files are published by atomic rename, so an interrupted run cannot leave
//!     something that later looks installed,
//!   * "installed" is one type with two variants, and both reject partial state.
//!
//! Callers whose orchestration is genuinely their own (multi-wheel packs with
//! progress events and resume) use the primitives directly rather than being
//! forced through a one-size abstraction.

use std::path::Path;

use anyhow::{anyhow, Result};

// ── Policy ───────────────────────────────────────────────────────────────────

/// The one HTTP policy for managed downloads.
///
/// `reqwest`'s `timeout()` is a deadline for the WHOLE request including the
/// body, which is why a 1.3 GB model on a slow link used to die at exactly ten
/// minutes. Bound the handshake and the per-read stall instead: slow-but-alive
/// keeps going, dead-but-open aborts.
pub(crate) fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(30))
        .read_timeout(std::time::Duration::from_secs(180))
        // reqwest sends NO User-Agent unless you set one, and several CDNs answer
        // a UA-less request with an interstitial rather than the file — HTTP 200,
        // matching content-length, an HTML body. Nothing upstream of the skill
        // download's shape check would notice.
        .user_agent(concat!("Anivar/", env!("CARGO_PKG_VERSION")))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new())
}

/// Number of attempts for a transient failure (network reset, 5xx, stall).
const ATTEMPTS: u32 = 3;

/// Download into memory, retrying transient failures.
///
/// `on_progress` receives THIS attempt's running byte count, so a retry that
/// restarts the transfer rewinds the caller's bar instead of double-counting.
/// Buffers because zip extraction needs `Seek`; use [`fetch_to_file`] for
/// anything large enough that RAM matters.
pub(crate) async fn fetch(
    client: &reqwest::Client,
    url: &str,
    label: &str,
    mut on_progress: impl FnMut(u64),
) -> Result<Vec<u8>> {
    use futures::StreamExt;
    let mut last_err = None;
    for attempt in 1..=ATTEMPTS {
        let attempt_res: Result<Vec<u8>> = async {
            let resp = client.get(url).send().await?.error_for_status()?;
            let mut buf = Vec::with_capacity(resp.content_length().unwrap_or(0) as usize);
            let mut stream = resp.bytes_stream();
            let mut got: u64 = 0;
            while let Some(chunk) = stream.next().await {
                let chunk = chunk?;
                got += chunk.len() as u64;
                on_progress(got);
                buf.extend_from_slice(&chunk);
            }
            Ok(buf)
        }.await;
        match attempt_res {
            Ok(b) => return Ok(b),
            Err(e) => {
                tracing::warn!("{label}: download attempt {attempt}/{ATTEMPTS} failed: {e}");
                last_err = Some(e);
                if attempt < ATTEMPTS {
                    tokio::time::sleep(std::time::Duration::from_secs(2 * attempt as u64)).await;
                }
            }
        }
    }
    Err(last_err.unwrap_or_else(|| anyhow!("{label}: download failed")))
}

/// Stream a download straight to `dest` via a `.part` file, published by rename.
///
/// Verifies the byte count against `content-length` when the server provides
/// one: a stream can end EARLY without erroring (proxy/CDN cut), which is how a
/// truncated model used to end up looking installed. Returns the byte count.
pub(crate) async fn fetch_to_file(
    client: &reqwest::Client,
    url: &str,
    dest: &Path,
    mut on_progress: impl FnMut(u64, Option<u64>),
) -> Result<u64> {
    use futures::StreamExt;
    use tokio::io::AsyncWriteExt;

    if let Some(parent) = dest.parent() { tokio::fs::create_dir_all(parent).await.ok(); }
    let part = dest.with_extension("part");

    let resp = client.get(url).send().await?.error_for_status()?;
    let total = resp.content_length();
    let mut file = tokio::fs::File::create(&part).await?;
    let mut stream = resp.bytes_stream();
    let mut got: u64 = 0;

    // Any mid-stream failure takes the partial file with it — leaving it behind
    // is what makes a stub look like an install.
    let outcome: Result<()> = async {
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            file.write_all(&chunk).await?;
            got += chunk.len() as u64;
            on_progress(got, total);
        }
        file.flush().await?;
        Ok(())
    }.await;
    drop(file);

    let verdict = outcome.and_then(|()| match total {
        Some(t) if got != t => Err(anyhow!("incomplete download — got {got} of {t} bytes")),
        _ => Ok(()),
    });
    if let Err(e) = verdict {
        let _ = tokio::fs::remove_file(&part).await;
        return Err(e);
    }

    // Windows cannot rename over an open/existing file reliably.
    let _ = tokio::fs::remove_file(dest).await;
    tokio::fs::rename(&part, dest).await?;
    Ok(got)
}

/// Hex SHA-256 of a byte slice.
pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(bytes))
}

/// Hex SHA-256 of a file on disk.
/// Part of the provisioning contract's public surface. No caller today — the
/// current artifacts verify from memory — but digest-checking an already-installed
/// file is exactly what this module exists to make available.
#[allow(dead_code)]
pub(crate) fn sha256_of_file(path: &Path) -> Result<String> {
    Ok(sha256_hex(&std::fs::read(path)?))
}

/// Integrity gate. Plain comparison is correct here: this detects corruption and
/// tampering of PUBLIC artifacts, not a secret, so timing is irrelevant.
pub(crate) fn verify_sha256(bytes: &[u8], expected: &str, label: &str) -> Result<()> {
    let got = sha256_hex(bytes);
    if got != expected {
        return Err(anyhow!(
            "{label}: SHA-256 mismatch (expected {expected}, got {got}) — refusing to install"
        ));
    }
    Ok(())
}

/// Write bytes to `dest` atomically: `.part` first, then rename.
/// Kept alongside `fetch_to_file` (which streams and renames) for callers that
/// already hold the bytes. Covered by `install_atomic_publishes_cleanly`.
#[allow(dead_code)]
pub(crate) async fn install_atomic(bytes: &[u8], dest: &Path) -> Result<()> {
    if let Some(parent) = dest.parent() { tokio::fs::create_dir_all(parent).await.ok(); }
    let part = dest.with_extension("part");
    tokio::fs::write(&part, bytes).await?;
    let _ = tokio::fs::remove_file(dest).await;
    tokio::fs::rename(&part, dest).await?;
    Ok(())
}

/// Extract every zip entry whose path satisfies `keep`, FLATTENED into `dest`
/// (archives nest payloads under version directories we don't want to mirror).
/// Returns how many files were written.
pub(crate) fn unpack_zip(archive: &[u8], dest: &Path, keep: impl Fn(&str) -> bool) -> Result<usize> {
    std::fs::create_dir_all(dest)?;
    let mut zip = zip::ZipArchive::new(std::io::Cursor::new(archive))?;
    let mut count = 0;
    for i in 0..zip.len() {
        let mut f = zip.by_index(i)?;
        if !f.is_file() { continue; }
        let name = f.name().to_string();
        if !keep(&name) { continue; }
        let Some(fname) = Path::new(&name).file_name() else { continue };
        let mut out = std::fs::File::create(dest.join(fname))?;
        std::io::copy(&mut f, &mut out)?;
        count += 1;
    }
    Ok(count)
}

/// Platform wheel tag we can actually load — picking the wrong one silently
/// installs libraries this process can never open.
#[cfg(windows)]
#[allow(dead_code)] // callers (cuda_runtime / openvino_runtime) are cfg-gated
pub(crate) const WHEEL_TAG: &str = "win_amd64";
#[cfg(not(windows))]
pub(crate) const WHEEL_TAG: &str = "manylinux";

/// Resolve a PyPI package's wheel for this platform: `(url, sha256)`.
///
/// The digest comes from the index alongside the URL — the same pairing pip
/// relies on — so a caller without a compiled-in pin still verifies what it got.
#[allow(dead_code)] // callers (cuda_runtime / openvino_runtime) are cfg-gated
pub(crate) async fn resolve_wheel(
    client: &reqwest::Client,
    pkg: &str,
) -> Option<(String, Option<String>)> {
    let json: serde_json::Value = client
        .get(format!("https://pypi.org/pypi/{pkg}/json"))
        .send().await.ok()?
        .json().await.ok()?;
    let files = json["urls"].as_array()?;
    let pick = files.iter()
        .find(|f| f["filename"].as_str().is_some_and(|n|
            n.contains(WHEEL_TAG) && (cfg!(windows) || n.contains("x86_64"))))
        .or_else(|| files.iter().find(|f| f["filename"].as_str().is_some_and(|n| n.contains(WHEEL_TAG))))?;
    let url = pick["url"].as_str()?.to_string();
    let sha = pick["digests"]["sha256"].as_str().map(String::from);
    Some((url, sha))
}

// ── "Is it installed?" — one definition ──────────────────────────────────────

/// The single answer to "is this artifact usable?", replacing the seven ad-hoc
/// rules this module retired. Both variants reject PARTIAL state, which is the
/// bug class that kept resurfacing: `Path::exists()` is never an install check.
pub(crate) enum Requirement<'a> {
    /// A single file that must be at least this many bytes. A truncated download
    /// or an HTML error page never reaches the floor.
    MinSize(u64),
    /// A directory that must contain a file starting with EVERY marker. Testing
    /// one marker meant a pack that died after its first wheel reported itself
    /// installed forever while the runtime kept failing to load.
    /// Constructed only by the cuda/openvino runtime packs, whose callers are
    /// feature-gated — so this reads as "never constructed" on a default build.
    #[allow(dead_code)]
    Markers(&'a [&'a str]),
}

impl Requirement<'_> {
    pub(crate) fn met(&self, path: &Path) -> bool {
        match self {
            Requirement::MinSize(min) => {
                std::fs::metadata(path).map(|m| m.is_file() && m.len() >= *min).unwrap_or(false)
            }
            Requirement::Markers(markers) => {
                let names: Vec<String> = match std::fs::read_dir(path) {
                    Ok(rd) => rd.filter_map(|e| e.ok())
                        .map(|e| e.file_name().to_string_lossy().to_lowercase())
                        .collect(),
                    Err(_) => return false,
                };
                !markers.is_empty()
                    && markers.iter().all(|m| names.iter().any(|n| n.starts_with(&m.to_lowercase())))
            }
        }
    }
}

// ── Native library packs (the 3× duplicated case) ────────────────────────────

/// Download PyPI wheels and flatten their native libraries into `dir`.
///
/// This is the shape `cuda_runtime` and `openvino_runtime` had each reimplemented.
/// Wheels already satisfying the requirement are skipped, so a retry after a
/// failure costs the remaining wheels rather than the whole set.
#[allow(dead_code)] // callers (cuda_runtime / openvino_runtime) are cfg-gated
pub(crate) async fn ensure_lib_pack(
    pkgs: &[&str],
    dir: &Path,
    requirement: &Requirement<'_>,
    label: &str,
) -> Result<()> {
    if requirement.met(dir) { return Ok(()); }
    tokio::fs::create_dir_all(dir).await.ok();
    let client = client();
    for pkg in pkgs {
        let (url, sha) = resolve_wheel(&client, pkg).await
            .ok_or_else(|| anyhow!("{label}: no {WHEEL_TAG} wheel published for {pkg}"))?;
        tracing::info!("{label}: fetching {pkg}");
        let bytes = fetch(&client, &url, pkg, |_| {}).await?;
        if let Some(sha) = sha {
            verify_sha256(&bytes, &sha, pkg)?;
        } else {
            tracing::warn!("{label}: {pkg} published no sha256 — installing unverified");
        }
        let n = unpack_zip(&bytes, dir, is_native_lib)?;
        tracing::info!("{label}: extracted {n} librar{} from {pkg}", if n == 1 { "y" } else { "ies" });
    }
    if !requirement.met(dir) {
        return Err(anyhow!("{label}: install incomplete after extraction — see {}", dir.display()));
    }
    Ok(())
}

/// Native libraries inside a Python wheel: Windows keeps DLLs under `bin/`,
/// Linux keeps versioned `.so` files under `lib/`.
#[allow(dead_code)] // callers (cuda_runtime / openvino_runtime) are cfg-gated
pub(crate) fn is_native_lib(entry: &str) -> bool {
    let lower = entry.to_lowercase();
    if cfg!(windows) {
        lower.ends_with(".dll") && lower.contains("/bin/")
    } else {
        lower.contains("/lib/") && lower.contains(".so")
    }
}

// ── Making a directory of native libraries loadable ──────────────────────────

/// Windows: prepend to the DLL search path.
#[cfg(windows)]
pub(crate) fn make_libs_loadable(dir: &Path) {
    let d = dir.display().to_string();
    let cur = std::env::var("PATH").unwrap_or_default();
    if !cur.split(';').any(|p| p == d) {
        std::env::set_var("PATH", format!("{d};{cur}"));
    }
}

/// Unix: `dlopen` each library by absolute path with `RTLD_GLOBAL`.
///
/// Setting `LD_LIBRARY_PATH` here — which two copies of this function used to do
/// — **cannot work**: glibc's loader reads it once at process start, so mutating
/// it mid-process has no effect on any later `dlopen`. Preloading instead makes
/// ORT's own `dlopen("libcudnn.so.9")` resolve to the already-resident soname.
#[cfg(target_os = "linux")]
pub(crate) fn make_libs_loadable(dir: &Path) {
    use libloading::os::unix::{Library, RTLD_GLOBAL, RTLD_LAZY};

    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let mut pending: Vec<std::path::PathBuf> = rd.flatten()
        .map(|e| e.path())
        .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.contains(".so")))
        .collect();
    pending.sort();

    // Fixed point: libcudnn needs libcublas, which needs libcudart… and readdir
    // order is arbitrary. Sweep until a pass loads nothing new, so any dependency
    // order resolves without hardcoding one.
    let mut loaded = 0usize;
    loop {
        let before = pending.len();
        pending.retain(|p| match unsafe { Library::open(Some(p), RTLD_LAZY | RTLD_GLOBAL) } {
            Ok(lib) => { std::mem::forget(lib); loaded += 1; false } // resident for process life
            Err(_) => true,                                          // unmet dep — retry next pass
        });
        if pending.len() == before { break; }
    }
    if !pending.is_empty() {
        tracing::warn!("preload: {} of {} libraries unresolved (first: {:?})",
            pending.len(), loaded + pending.len(), pending.first());
    }
    tracing::info!("preloaded {loaded} shared libraries from {}", dir.display());
}

/// macOS and friends: no managed native-library packs exist there (CoreML is
/// part of the OS), so there is nothing to preload.
#[cfg(not(any(windows, target_os = "linux")))]
#[allow(dead_code)] // no-op stub for targets with no preload step
pub(crate) fn make_libs_loadable(_dir: &Path) {}

/// Free bytes on the volume holding `path`; `None` when undeterminable (callers
/// then proceed rather than blocking an install on a failed probe).
/// Used by the Windows-only accelerator-pack preflight, so this reads as dead
/// code on other targets.
#[allow(dead_code)]
pub(crate) fn free_space_bytes(path: &Path) -> Option<u64> {
    let disks = sysinfo::Disks::new_with_refreshed_list();
    disks.list().iter()
        .filter(|d| path.starts_with(d.mount_point()))
        // Longest matching mount point wins (C:\ vs C:\data).
        .max_by_key(|d| d.mount_point().as_os_str().len())
        .map(|d| d.available_space())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::PathBuf;

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("sc_prov_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// A zip in, the right flattened files out — including the nested layout the
    /// wheels actually use.
    #[test]
    fn unpack_zip_flattens_and_filters() {
        let dir = tmpdir("zip");
        let mut buf = Vec::new();
        {
            let mut z = zip::ZipWriter::new(std::io::Cursor::new(&mut buf));
            let opts: zip::write::FileOptions<()> = zip::write::FileOptions::default();
            for name in ["pkg/bin/keep_me.dll", "pkg/doc/skip.txt", "pkg/bin/also.dll"] {
                z.start_file(name, opts).unwrap();
                z.write_all(b"payload").unwrap();
            }
            z.finish().unwrap();
        }
        let n = unpack_zip(&buf, &dir, |e| e.ends_with(".dll")).unwrap();
        assert_eq!(n, 2, "only the DLLs should be extracted");
        assert!(dir.join("keep_me.dll").is_file(), "entries must be flattened, not nested");
        assert!(dir.join("also.dll").is_file());
        assert!(!dir.join("skip.txt").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Publishing is atomic and leaves no scratch file behind.
    #[tokio::test]
    async fn install_atomic_publishes_cleanly() {
        let dir = tmpdir("atomic");
        let dest = dir.join("thing.bin");
        install_atomic(b"hello", &dest).await.unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), b"hello");
        assert!(!dest.with_extension("part").exists(), "no .part may survive a success");
        // Replacing an existing file must work (Windows rename-over).
        install_atomic(b"replaced", &dest).await.unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), b"replaced");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The bug class this module exists to retire: partial state must never read
    /// as installed.
    #[test]
    fn requirement_rejects_partial_state() {
        let dir = tmpdir("req");

        let stub = dir.join("model.onnx");
        std::fs::write(&stub, b"<html>404</html>").unwrap();
        assert!(!Requirement::MinSize(64 * 1024).met(&stub), "an error page is not a model");
        std::fs::write(&stub, vec![0u8; 64 * 1024]).unwrap();
        assert!(Requirement::MinSize(64 * 1024).met(&stub));
        assert!(!Requirement::MinSize(1).met(&dir), "a directory is not a file");

        let pack = dir.join("pack");
        std::fs::create_dir_all(&pack).unwrap();
        let req = Requirement::Markers(&["nvinfer", "cudnn", "cublas"]);
        assert!(!req.met(&pack), "empty dir is not provisioned");
        std::fs::write(pack.join("nvinfer_10.dll"), b"x").unwrap();
        assert!(!req.met(&pack), "one marker of three is a HALF install");
        std::fs::write(pack.join("cudnn64_9.dll"), b"x").unwrap();
        std::fs::write(pack.join("cublas64_12.dll"), b"x").unwrap();
        assert!(req.met(&pack), "all markers present = provisioned");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
