//! ffmpeg binary resolution.
//!
//! [`ensure_ffmpeg`] checks the system `PATH` first; if ffmpeg isn't installed
//! it downloads a pre-built binary (BtbN/FFmpeg-Builds, ~50 MB) into the
//! application data directory. Returns the resolved path either way.
//!
//! Used by NVR recording, clip post-processing, stroboscopic frame extraction,
//! and the live MJPEG transcoder.
//!
//! The answer is resolved ONCE per process ([`RESOLVED`]): every recording,
//! playback and probe path calls this, and each call used to spawn
//! `ffmpeg -version` just to re-derive the same path — a process spawn per HLS
//! range request on the playback hot path.

use std::path::{Path, PathBuf};

use tokio::sync::OnceCell;

// Uses BtbN/FFmpeg-Builds — the most reliable source for all platforms.
// Downloads only the essential ffmpeg binary (~50MB), not the full toolkit.

/// Process-wide resolution cache. `get_or_try_init` also SERIALIZES first-time
/// resolution, so N concurrent callers can't each start their own download.
static RESOLVED: OnceCell<PathBuf> = OnceCell::const_new();

/// A real ffmpeg build is tens of MB. Anything smaller in the data dir is a
/// truncated/failed download from an older build — ignore it and re-fetch
/// instead of handing every caller a binary that can't execute.
const MIN_FFMPEG_BYTES: u64 = 1_000_000;

/// Resolve the ffmpeg binary path, downloading it if not present.
/// Checks system PATH first — if ffmpeg is already installed, use it.
/// Otherwise downloads the appropriate pre-built binary into the data directory.
pub async fn ensure_ffmpeg(data_dir: &Path) -> anyhow::Result<PathBuf> {
    RESOLVED
        .get_or_try_init(|| resolve(data_dir))
        .await.cloned()
}

async fn resolve(data_dir: &Path) -> anyhow::Result<PathBuf> {
    // 1. Check if ffmpeg is already on system PATH
    if let Ok(output) = crate::proc::tokio_cmd("ffmpeg").args(["-version"]).output().await {
        if output.status.success() {
            return Ok(PathBuf::from("ffmpeg")); // system ffmpeg available
        }
    }

    // 2. Check if we already downloaded it (a truncated file doesn't count).
    #[cfg(target_os = "windows")]
    let bin_name = "ffmpeg.exe";
    #[cfg(not(target_os = "windows"))]
    let bin_name = "ffmpeg";

    let bin_path = data_dir.join(bin_name);
    if crate::provision::Requirement::MinSize(MIN_FFMPEG_BYTES).met(&bin_path) {
        return Ok(bin_path);
    }
    if bin_path.exists() {
        tracing::warn!("ffmpeg at {} is truncated — re-downloading", bin_path.display());
        let _ = tokio::fs::remove_file(&bin_path).await;
    }

    // 3. Download the appropriate pre-built binary
    tracing::info!("ffmpeg not found — downloading pre-built binary…");
    download_ffmpeg(&bin_path).await?;
    tracing::info!("ffmpeg downloaded to {}", bin_path.display());
    Ok(bin_path)
}

#[cfg(test)]
mod cold_path {
    /// Exercise the REAL cold install — fetch → unpack → atomic publish →
    /// `-version` — into a scratch dir, so the shared `provision` primitives are
    /// proven on the path that only runs when the binary is missing (installers
    /// bundle ffmpeg, so it is otherwise never taken and never noticed if broken).
    ///
    /// `#[ignore]`d: it downloads ~50 MB. Run deliberately with
    /// `cargo test --lib cold_path -- --ignored --nocapture`.
    #[tokio::test]
    #[ignore]
    async fn downloads_and_runs_ffmpeg() {
        let dir = std::env::temp_dir().join(format!("sc_ffmpeg_cold_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let dest = dir.join(if cfg!(windows) { "ffmpeg.exe" } else { "ffmpeg" });

        super::download_ffmpeg(&dest).await.expect("cold ffmpeg install should succeed");

        assert!(super::super::provision::Requirement::MinSize(super::MIN_FFMPEG_BYTES).met(&dest));
        assert!(!dest.with_extension("part").exists(), "no .part may survive");
        let out = crate::proc::std_cmd(&dest).arg("-version").output().unwrap();
        assert!(out.status.success(), "downloaded ffmpeg must execute");
        eprintln!("cold install OK: {}", String::from_utf8_lossy(&out.stdout).lines().next().unwrap_or(""));
        let _ = std::fs::remove_dir_all(&dir);
    }
}

pub(crate) async fn download_ffmpeg(dest: &Path) -> anyhow::Result<()> {
    // NOTE: this is now the ONLY way ffmpeg arrives. Installers no longer bundle
    // it — the build we use is GPL, and shipping it would mean distributing it.
    // Fetching on the user's behalf, from upstream, keeps this Apache-2.0 app
    // clear of those terms (see THIRD-PARTY-NOTICES.md). Every user hits this path
    // exactly once, on first use.
    //
    // These URLs used to point at an `n7.1-latest` stable line and **404'd**: BtbN
    // now publishes only `master` builds, and never published macOS at all. The
    // rot was invisible precisely because this path is the fallback — nobody runs
    // it until the day they need it. `cold_path::downloads_and_runs_ffmpeg` is the
    // check that catches it next time. `latest` is BtbN's permanent tag (dated
    // `autobuild-*` tags get pruned); GPL so libx264 is included.
    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    let url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip";
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    let url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz";

    // BtbN ships no macOS binaries. Rather than keep a URL that can only 404,
    // say what will actually work. (The .dmg bundles ffmpeg, so this is a
    // source-build path.)
    #[cfg(target_os = "macos")]
    {
        let _ = dest;
        anyhow::bail!(
            "ffmpeg not found and no macOS download source is configured — install it with \
             `brew install ffmpeg` (shipped macOS builds bundle ffmpeg; this fallback only \
             runs for source builds)"
        );
    }

    #[cfg(not(target_os = "macos"))]
    {

    let dir = dest.parent().unwrap_or_else(|| Path::new(".")).to_path_buf();
    tokio::fs::create_dir_all(&dir).await.ok();

    // Retry/timeout/status policy is shared — see `provision::fetch`.
    tracing::info!("Downloading ffmpeg from {}", url);
    let bytes = crate::provision::fetch(&crate::provision::client(), url, "ffmpeg", |_| {}).await?;
    tracing::info!("ffmpeg archive downloaded ({:.1} MB), extracting…", bytes.len() as f64 / 1_048_576.0);

    // Extract straight out of the archive in-process. The previous PowerShell
    // Expand-Archive path broke on any data-dir path containing a quote (a legal
    // Windows username: C:\Users\O'Brien\…) and left the whole ~200 MB extracted
    // tree behind next to the binary.
    let tmp = dest.with_extension("part");
    #[cfg(target_os = "windows")]
    {
        // The BtbN archive nests the binary under `<build>/bin/`; unpack_zip
        // flattens, so extract into the parent and then position it.
        let (dir2, tmp2) = (dir.clone(), tmp.clone());
        tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
            let n = crate::provision::unpack_zip(&bytes, &dir2, |e| e.ends_with("bin/ffmpeg.exe"))?;
            anyhow::ensure!(n == 1, "ffmpeg.exe not found in archive");
            std::fs::rename(dir2.join("ffmpeg.exe"), &tmp2)?;
            Ok(())
        }).await??;
    }

    #[cfg(not(target_os = "windows"))]
    {
        // tar.xz — no pure-Rust xz decoder in the tree, so shell out to tar
        // (args are passed as a vector, never interpolated into a shell string).
        let archive_path = dir.join("ffmpeg_archive.tar.xz");
        tokio::fs::write(&archive_path, &bytes).await?;
        let status = crate::proc::tokio_cmd("tar")
            .args(["-xJf", &archive_path.to_string_lossy(), "-C", &dir.to_string_lossy(),
                   "--strip-components=2", "--wildcards", "*/bin/ffmpeg"])
            .status().await;
        let _ = tokio::fs::remove_file(&archive_path).await;
        anyhow::ensure!(status.map(|s| s.success()).unwrap_or(false), "ffmpeg tar extraction failed");
        let extracted = dir.join("ffmpeg");
        anyhow::ensure!(extracted.exists(), "ffmpeg missing after tar extraction");
        std::fs::rename(&extracted, &tmp)?;
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o755))?;
    }

    // Publish atomically — a crash/kill mid-extract can never leave a truncated
    // `ffmpeg.exe` that every later call happily hands out.
    anyhow::ensure!(
        tokio::fs::metadata(&tmp).await.map(|m| m.len() >= MIN_FFMPEG_BYTES).unwrap_or(false),
        "extracted ffmpeg is too small — download was truncated"
    );
    tokio::fs::rename(&tmp, dest).await?;

    #[cfg(target_os = "windows")]
    {
        // Remove Zone.Identifier ADS so Windows doesn't block execution
        let ads = format!("{}:Zone.Identifier", dest.to_str().unwrap_or(""));
        let _ = std::fs::remove_file(ads);
    }

    // Verify it actually runs
    let check = crate::proc::tokio_cmd(dest).args(["-version"]).output().await;
    if !check.map(|o| o.status.success()).unwrap_or(false) {
        let _ = tokio::fs::remove_file(dest).await;
        anyhow::bail!("ffmpeg binary failed to run after download");
    }

    Ok(())
    }
}
