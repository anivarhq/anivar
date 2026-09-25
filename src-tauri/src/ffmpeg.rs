//! ffmpeg binary resolution.
//!
//! [`ensure_ffmpeg`] uses an ffmpeg the machine already has if there is one.
//! Otherwise it installs a pinned build — ffmpeg and ffprobe together — into
//! the application data directory, verified by SHA-256 exactly as go2rtc is,
//! and returns the resolved path either way.
//!
//! Used by NVR recording, clip post-processing, stroboscopic frame extraction,
//! and the live MJPEG transcoder. ffprobe, installed beside it, is what the
//! Add Camera "Test connection" and clip-duration probes run.
//!
//! The answer is resolved ONCE per process ([`RESOLVED`]): every recording,
//! playback and probe path calls this, and each call used to spawn
//! `ffmpeg -version` just to re-derive the same path — a process spawn per HLS
//! range request on the playback hot path.

use std::path::{Path, PathBuf};

use tokio::sync::OnceCell;

/// Process-wide resolution cache. `get_or_try_init` also SERIALIZES first-time
/// resolution, so N concurrent callers can't each start their own download.
static RESOLVED: OnceCell<PathBuf> = OnceCell::const_new();

/// A real ffmpeg build is tens of MB. Anything smaller in the data dir is a
/// truncated/failed download from an older build — ignore it and re-fetch
/// instead of handing every caller a binary that can't execute.
const MIN_FFMPEG_BYTES: u64 = 1_000_000;

#[cfg(windows)]
const FFMPEG: &str = "ffmpeg.exe";
#[cfg(not(windows))]
const FFMPEG: &str = "ffmpeg";
#[cfg(windows)]
const FFPROBE: &str = "ffprobe.exe";
#[cfg(not(windows))]
const FFPROBE: &str = "ffprobe";

/// Records which pinned build is installed. When the pin in this file moves,
/// the recorded value stops matching and the next launch replaces the pair.
const PIN_MARKER: &str = "ffmpeg.pin";

/// Resolve the ffmpeg binary path, installing it if not present.
pub async fn ensure_ffmpeg(data_dir: &Path) -> anyhow::Result<PathBuf> {
    RESOLVED
        .get_or_try_init(|| resolve(data_dir))
        .await.cloned()
}

async fn resolve(data_dir: &Path) -> anyhow::Result<PathBuf> {
    // 1. An ffmpeg the machine already has.
    if runs(Path::new("ffmpeg")).await {
        return Ok(PathBuf::from("ffmpeg"));
    }

    // 2. macOS package managers install where an app opened from Finder cannot
    //    see: GUI apps get the system PATH, which has neither Homebrew's
    //    directory nor MacPorts'. A user who ran `brew install ffmpeg` was told
    //    to do exactly that, and still had no recording.
    #[cfg(target_os = "macos")]
    for dir in ["/opt/homebrew/bin", "/usr/local/bin", "/opt/local/bin"] {
        let candidate = Path::new(dir).join("ffmpeg");
        if runs(&candidate).await {
            return Ok(candidate);
        }
    }

    // 3. Our own install, if it is complete and from the current pin.
    let bin = data_dir.join(FFMPEG);
    let Some(pin) = pin() else {
        anyhow::bail!(
            "ffmpeg not found, and there is no pinned build for {}/{} — install ffmpeg \
             with your system's package manager",
            std::env::consts::OS, std::env::consts::ARCH
        );
    };
    if installed(data_dir, &pin) {
        return Ok(bin);
    }

    // 4. Install the pinned build. An older, unverified ffmpeg from a previous
    //    version is replaced — but if that cannot happen (offline, say), keep
    //    using it rather than leave a working camera without a recorder.
    tracing::info!("installing pinned ffmpeg ({})…", pin.id);
    match install_pinned(data_dir, &pin).await {
        Ok(()) => {
            tracing::info!("ffmpeg {} installed (SHA verified)", pin.id);
            Ok(bin)
        }
        Err(e) if runs(&bin).await => {
            tracing::warn!("could not install the pinned ffmpeg ({e}); keeping the existing one");
            Ok(bin)
        }
        Err(e) => Err(e),
    }
}

/// Both tools present, full-size, and from the pin this build expects.
fn installed(data_dir: &Path, pin: &Pin) -> bool {
    let full = crate::provision::Requirement::MinSize(MIN_FFMPEG_BYTES);
    full.met(&data_dir.join(FFMPEG))
        && full.met(&data_dir.join(FFPROBE))
        && std::fs::read_to_string(data_dir.join(PIN_MARKER)).is_ok_and(|s| s.trim() == pin.id)
}

async fn runs(bin: &Path) -> bool {
    crate::proc::tokio_cmd(bin).arg("-version").output().await
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Where each build comes from, pinned by content.
///
/// ffmpeg is fetched on the user's behalf rather than shipped: the builds are
/// GPL, and distributing them would put this Apache-2.0 app under those terms
/// (see THIRD-PARTY-NOTICES.md). Every user takes this path once.
///
/// The sources are versioned releases that do not move: GyanD's builds of the
/// current ffmpeg for Windows, and eugeneware/ffmpeg-static for Linux and Apple
/// Silicon, which is the only versioned source with a macOS build at all.
/// BtbN's `latest` was used before and was never verified — a moving target,
/// and its dated tags are pruned, so it cannot be pinned.
///
/// Every digest below is the one GitHub publishes for that release asset, and
/// each was re-hashed locally from a fresh download before being written here.
struct Pin {
    /// Written to [`PIN_MARKER`] once installed.
    id: &'static str,
    sources: &'static [Source],
}

struct Source {
    url: &'static str,
    sha256: &'static str,
    packing: Packing,
}

#[derive(Clone, Copy)]
enum Packing {
    /// A zip that nests `…/bin/ffmpeg.exe` and `…/bin/ffprobe.exe`.
    ZipWithBoth,
    /// One gzipped executable, installed under this name.
    Gzip(&'static str),
}

fn pin() -> Option<Pin> {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("windows", "x86_64") => Some(Pin {
            id: "gyan-9.0.2-essentials",
            sources: &[Source {
                url: "https://github.com/GyanD/codexffmpeg/releases/download/9.0.2/ffmpeg-9.0.2-essentials_build.zip",
                sha256: "60f467265b1e312373dbcd92200c2618a74850f98d3d078e94296bb3fa2047ba",
                packing: Packing::ZipWithBoth,
            }],
        }),
        ("linux", "x86_64") => Some(Pin {
            id: "ffmpeg-static-b6.1.1",
            sources: &[
                Source {
                    url: "https://github.com/eugeneware/ffmpeg-static/releases/download/b6.1.1/ffmpeg-linux-x64.gz",
                    sha256: "bfe8a8fc511530457b528c48d77b5737527b504a3797a9bc4866aeca69c2dffa",
                    packing: Packing::Gzip("ffmpeg"),
                },
                Source {
                    url: "https://github.com/eugeneware/ffmpeg-static/releases/download/b6.1.1/ffprobe-linux-x64.gz",
                    sha256: "25d9b6ccb05e3d9de9e04e31e2506d8dd7f9f0418981965ac6df12e8d3afd067",
                    packing: Packing::Gzip("ffprobe"),
                },
            ],
        }),
        ("macos", "aarch64") => Some(Pin {
            id: "ffmpeg-static-b6.1.1",
            sources: &[
                Source {
                    url: "https://github.com/eugeneware/ffmpeg-static/releases/download/b6.1.1/ffmpeg-darwin-arm64.gz",
                    sha256: "8923876afa8db5585022d7860ec7e589af192f441c56793971276d450ed3bbfa",
                    packing: Packing::Gzip("ffmpeg"),
                },
                Source {
                    url: "https://github.com/eugeneware/ffmpeg-static/releases/download/b6.1.1/ffprobe-darwin-arm64.gz",
                    sha256: "d986a8ec7b030899fe66a8a288ed809a3543338705a3ce178cfb85869c5d80be",
                    packing: Packing::Gzip("ffprobe"),
                },
            ],
        }),
        _ => None,
    }
}

/// Download, verify, stage, prove, then swap in — in that order.
///
/// Nothing touches the binaries in use until both new ones have been verified
/// against their pin and have run: a download that fails half-way, or a build
/// that will not start on this machine, leaves the current install exactly as
/// it was.
async fn install_pinned(dir: &Path, pin: &Pin) -> anyhow::Result<()> {
    let staging = dir.join("ffmpeg-staging");
    let _ = tokio::fs::remove_dir_all(&staging).await;
    tokio::fs::create_dir_all(&staging).await?;

    let client = crate::provision::client();
    for src in pin.sources {
        tracing::info!("Downloading {}", src.url);
        let bytes = crate::provision::fetch(&client, src.url, "ffmpeg", |_| {}).await?;
        // The integrity pin: a moved, re-uploaded or tampered file stops here.
        crate::provision::verify_sha256(&bytes, src.sha256, src.url)?;
        let (into, packing) = (staging.clone(), src.packing);
        tokio::task::spawn_blocking(move || unpack(&bytes, packing, &into)).await??;
    }

    for name in [FFMPEG, FFPROBE] {
        let staged = staging.join(name);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&staged, std::fs::Permissions::from_mode(0o755))
                .map_err(|e| anyhow::anyhow!("{name}: could not make it executable: {e}"))?;
        }
        anyhow::ensure!(runs(&staged).await, "{name} from {} does not run on this machine", pin.id);
    }

    for name in [FFMPEG, FFPROBE] {
        let target = dir.join(name);
        // Windows will not rename onto an existing file.
        let _ = tokio::fs::remove_file(&target).await;
        tokio::fs::rename(staging.join(name), &target).await?;
        #[cfg(windows)]
        {
            // A downloaded file carries a Zone.Identifier stream that can make
            // Windows hold up its execution.
            let _ = std::fs::remove_file(format!("{}:Zone.Identifier", target.display()));
        }
    }

    // Last, so a failure anywhere above means the next launch tries again.
    tokio::fs::write(dir.join(PIN_MARKER), pin.id).await?;
    let _ = tokio::fs::remove_dir_all(&staging).await;
    Ok(())
}

fn unpack(bytes: &[u8], packing: Packing, into: &Path) -> anyhow::Result<()> {
    match packing {
        Packing::ZipWithBoth => {
            for exe in [FFMPEG, FFPROBE] {
                let suffix = format!("bin/{exe}");
                let n = crate::provision::unpack_zip(bytes, into, |e| e.ends_with(&suffix))?;
                anyhow::ensure!(n == 1, "{exe} not found in the archive");
            }
        }
        Packing::Gzip(name) => {
            use std::io::Read;
            let mut exe = Vec::new();
            flate2::read::GzDecoder::new(bytes).read_to_end(&mut exe)?;
            std::fs::write(into.join(name), exe)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CI runs this on Windows, macOS and Linux, so each shipped target checks
    /// its own pin: present, versioned, and carrying a real digest.
    #[test]
    fn every_pin_is_a_versioned_github_release_with_a_real_digest() {
        let shipped = matches!(
            (std::env::consts::OS, std::env::consts::ARCH),
            ("windows", "x86_64") | ("linux", "x86_64") | ("macos", "aarch64")
        );
        assert_eq!(pin().is_some(), shipped, "a shipped target must have a pinned build");
        if let Some(pin) = pin() {
            assert!(!pin.sources.is_empty());
            for src in pin.sources {
                assert!(src.url.starts_with("https://github.com/"), "{}", src.url);
                assert!(src.url.contains("/releases/download/"), "{}", src.url);
                assert!(!src.url.contains("/latest/"), "a moving target cannot be pinned: {}", src.url);
                assert_eq!(src.sha256.len(), 64, "{}", src.url);
                assert!(src.sha256.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
            }
        }
    }

    #[test]
    fn gzip_packing_installs_the_executable_under_its_name() {
        use std::io::Write;
        let dir = std::env::temp_dir().join(format!("ffmpeg_unpack_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let payload = b"\x7fELF not really a binary";
        let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
        gz.write_all(payload).unwrap();
        let bytes = gz.finish().unwrap();

        unpack(&bytes, Packing::Gzip("ffprobe"), &dir).unwrap();
        assert_eq!(std::fs::read(dir.join("ffprobe")).unwrap(), payload);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_tampered_download_is_refused() {
        // The same check go2rtc relies on, exercised with a pin from this file:
        // anything but the exact bytes GitHub published must not be installed.
        let Some(pin) = pin() else { return };
        let err = crate::provision::verify_sha256(b"not the release", pin.sources[0].sha256, "ffmpeg")
            .unwrap_err().to_string();
        assert!(err.contains("mismatch"), "{err}");
    }

    #[test]
    fn a_stale_pin_is_not_an_install() {
        let dir = std::env::temp_dir().join(format!("ffmpeg_pin_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let big = vec![0u8; (MIN_FFMPEG_BYTES + 1) as usize];
        std::fs::write(dir.join(FFMPEG), &big).unwrap();
        std::fs::write(dir.join(FFPROBE), &big).unwrap();
        let pin = Pin { id: "current", sources: &[] };

        // An ffmpeg left by an older version, with no record of where it came
        // from, is replaced: that is how existing installs get the verified
        // build, and get ffprobe at all.
        assert!(!installed(&dir, &pin));
        std::fs::write(dir.join(PIN_MARKER), "older").unwrap();
        assert!(!installed(&dir, &pin));
        std::fs::write(dir.join(PIN_MARKER), "current").unwrap();
        assert!(installed(&dir, &pin));

        // And ffmpeg alone is not enough: Add Camera's test runs ffprobe.
        std::fs::remove_file(dir.join(FFPROBE)).unwrap();
        assert!(!installed(&dir, &pin));
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[cfg(test)]
mod cold_path {
    /// The real cold install — fetch → verify → stage → run → publish — into a
    /// scratch dir, so the path every new user takes is proven, not assumed.
    ///
    /// `#[ignore]`d: it downloads the pinned build (~110 MB on Windows). Run
    /// deliberately with `cargo test --lib cold_path -- --ignored --nocapture`.
    #[tokio::test]
    #[ignore]
    async fn installs_and_runs_the_pinned_build() {
        let dir = std::env::temp_dir().join(format!("sc_ffmpeg_cold_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let pin = super::pin().expect("this platform has a pinned build");
        super::install_pinned(&dir, &pin).await.expect("cold install should succeed");

        assert!(super::installed(&dir, &pin), "both tools and the marker must be in place");
        assert!(!dir.join("ffmpeg-staging").exists(), "staging must be cleaned up");
        for tool in [super::FFMPEG, super::FFPROBE] {
            let out = crate::proc::std_cmd(dir.join(tool)).arg("-version").output().unwrap();
            assert!(out.status.success(), "{tool} must execute");
            eprintln!("{tool}: {}", String::from_utf8_lossy(&out.stdout).lines().next().unwrap_or(""));
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
