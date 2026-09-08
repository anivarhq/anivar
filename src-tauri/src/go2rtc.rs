//! go2rtc WebRTC restreamer — sub-second live view for RTSP cameras.
//!
//! Mature NVRs' model: go2rtc (single MIT-licensed Go binary) consumes each
//! camera's RTSP stream and restreams it over WebRTC with ZERO re-encoding;
//! the WebView plays it via WHEP through a `<video>` element. Latency drops
//! from ~3 s (HLS) to sub-second. Everything here is BEST-EFFORT: if the
//! binary is missing, the download fails, or the process dies, WebRTC simply
//! never answers and the frontend's ladder (WebRTC → HLS → MJPEG) keeps the
//! live view working.
//!
//! Windows-only for now (this appliance); other platforms no-op → HLS.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use tokio::sync::Mutex;

const GO2RTC_VERSION: &str = "v1.9.14";

/// One release asset, pinned by hash.
///
/// `zipped` matters: Windows and macOS ship a `.zip`, but **Linux ships the raw
/// executable with no extension** — so on Linux the download IS the binary and
/// there is nothing to unpack. Getting that wrong is the obvious way to write a
/// Linux path that never works.
struct Asset {
    name:   &'static str,
    sha256: &'static str,
    zipped: bool,
}

/// The pinned asset for this build target, or `None` where upstream publishes
/// nothing we support — in which case live view degrades to HLS exactly as it
/// did before, which is why this returns an Option rather than failing loudly.
///
/// Every hash below was taken from the v1.9.14 release and independently
/// re-hashed locally. The win64 value reproduces the constant this file has
/// carried since the original pin, which is the check that the method is sound
/// and the release has not been re-uploaded underneath us.
fn asset() -> Option<Asset> {
    let (name, sha256, zipped) = match (std::env::consts::OS, std::env::consts::ARCH) {
        ("windows", "x86_64")  => ("go2rtc_win64.zip",
            "dd4167d75cb04abe618855b7c71f8658bd009f60c1a71835d134d2c11c939907", true),
        ("windows", "aarch64") => ("go2rtc_win_arm64.zip",
            "814be0f6d8669025c7bccdd1f026ffaf613abae5352239f4ec84de543b94594a", true),
        ("macos",   "aarch64") => ("go2rtc_mac_arm64.zip",
            "919b78adc759d6b3883d1e1b2ac915ac0985bb903ff1897b4d228527bd64690c", true),
        ("macos",   "x86_64")  => ("go2rtc_mac_amd64.zip",
            "9b0b9a27a4dc3a5b8b93376e7e8fc2787c6af624a512842622be84aec0171c7a", true),
        ("linux",   "x86_64")  => ("go2rtc_linux_amd64",
            "32d616af226bd731678ffde328b94cfb94e30339bfefc469cfb76323144615a6", false),
        ("linux",   "aarch64") => ("go2rtc_linux_arm64",
            "359fabade8a7a51e81a55fe6df6b0ef81764a5e1d63179577534eaaa71904b50", false),
        _ => return None,
    };
    Some(Asset { name, sha256, zipped })
}

/// go2rtc API — loopback only (the WHEP exchange is proxied through our
/// token-authed axum server; nothing on the LAN can reach this port).
const API_ADDR: &str = "127.0.0.1:1984";
/// WebRTC media port. Binds all interfaces (ICE needs a reachable host
/// candidate); media is DTLS-SRTP so exposure is limited to an encrypted feed
/// endpoint that still requires the SDP handshake via the authed API.
const WEBRTC_PORT: u16 = 8555;

fn asset_url(name: &str) -> String {
    format!("https://github.com/AlexxIT/go2rtc/releases/download/{GO2RTC_VERSION}/{name}")
}

/// The spawned go2rtc child, held so kill_on_drop reaps it with the app.
static PROC: OnceLock<Mutex<Option<tokio::process::Child>>> = OnceLock::new();
fn proc_cell() -> &'static Mutex<Option<tokio::process::Child>> {
    PROC.get_or_init(|| Mutex::new(None))
}

fn exe_path(data_dir: &Path) -> PathBuf {
    let name = if cfg!(windows) { "go2rtc.exe" } else { "go2rtc" };
    data_dir.join("go2rtc").join(name)
}

/// Download + SHA-verify + install the pinned go2rtc release. Idempotent.
///
/// Was Windows-only, which is why sub-second live view was a Windows-only
/// feature and mac/Linux silently fell back to ~3 s HLS. All supported targets
/// are handled now; an unsupported target still returns `Err` and the caller
/// still degrades to HLS, so the failure mode is unchanged.
async fn ensure_binary(data_dir: &Path) -> anyhow::Result<PathBuf> {
    let Some(asset) = asset() else {
        anyhow::bail!("no pinned go2rtc build for {}/{} — live view uses HLS",
                      std::env::consts::OS, std::env::consts::ARCH);
    };
    let exe = exe_path(data_dir);
    // A truncated binary (killed mid-install) used to satisfy `exists()` forever
    // and then fail to spawn on every launch.
    if crate::provision::Requirement::MinSize(1_000_000).met(&exe) { return Ok(exe); }
    let dir = exe.parent().unwrap().to_path_buf();
    tokio::fs::create_dir_all(&dir).await?;

    let url = asset_url(asset.name);
    tracing::info!("Downloading go2rtc {GO2RTC_VERSION} from {url}");
    let bytes = crate::provision::fetch(&crate::provision::client(), &url, "go2rtc", |_| {}).await?;
    // Integrity pin — refuse to run a tampered binary.
    crate::provision::verify_sha256(&bytes, asset.sha256, asset.name)?;

    if asset.zipped {
        // Unzip the single binary out of the archive (blocking: zip is sync),
        // publishing by rename so a partial write is never read as an install.
        // The filter matches the platform's own binary name — the macOS archives
        // contain `go2rtc`, not `go2rtc.exe`.
        let dir2 = dir.clone();
        let want = exe.file_name().unwrap().to_string_lossy().to_string();
        tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
            let n = crate::provision::unpack_zip(&bytes, &dir2, |e| e.ends_with(&want))?;
            anyhow::ensure!(n > 0, "{want} not found in release archive");
            Ok(())
        }).await??;
    } else {
        // Linux publishes the bare executable — the download IS the binary.
        crate::provision::install_atomic(&bytes, &exe).await?;
    }

    // A freshly extracted or downloaded file carries no execute bit, and a
    // go2rtc that cannot be spawned looks exactly like one that was never
    // installed. This is the step that is easy to forget and impossible to
    // diagnose from the symptom.
    #[cfg(unix)]
    {
        // Same shape as crypto.rs:30 — one call, no metadata round-trip. NOT
        // `let _ =`: if the execute bit does not stick, go2rtc simply will not
        // spawn, and that must surface here rather than as a mystery later.
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&exe, std::fs::Permissions::from_mode(0o755))
            .map_err(|e| anyhow::anyhow!("go2rtc: could not make {} executable: {e}", exe.display()))?;
    }

    tracing::info!("go2rtc {GO2RTC_VERSION} installed from {} (SHA verified)", asset.name);
    Ok(exe)
}

/// Spawn go2rtc if it isn't already running. Sweeps orphans from previous app
/// sessions first (only processes running OUR data-dir copy — never a user's
/// own go2rtc). Safe to call repeatedly; cheap once running.
pub(crate) async fn ensure_running(data_dir: &Path) -> anyhow::Result<()> {
    let mut guard = proc_cell().lock().await;
    if let Some(child) = guard.as_mut() {
        if child.try_wait().ok().flatten().is_none() { return Ok(()); } // still alive
        *guard = None; // died — respawn below
    }
    let exe = ensure_binary(data_dir).await?;

    // Orphan sweep: a previous session's go2rtc still holds the ports. Both
    // branches match on OUR copy's path, never on a go2rtc the user installed
    // themselves. Best-effort on purpose — a missing pkill is not a reason to
    // refuse to start.
    #[cfg(windows)]
    {
        let sweep = format!(
            "Get-Process go2rtc -ErrorAction SilentlyContinue | Where-Object {{ $_.Path -like '{}*' }} | Stop-Process -Force",
            data_dir.display()
        );
        let _ = crate::proc::tokio_cmd("powershell")
            .args(["-NoProfile", "-NonInteractive", "-Command", &sweep])
            .output().await;
    }
    #[cfg(not(windows))]
    {
        // `pkill -f` matches the whole command line, and our child always runs
        // as "<data_dir>/go2rtc/go2rtc -config …", so the full exe path can only
        // ever match our own instance.
        let _ = crate::proc::tokio_cmd("pkill")
            .args(["-f", &exe.to_string_lossy()])
            .output().await;
    }

    // Minimal config: authed-proxy-only API on loopback, WebRTC media port,
    // go2rtc's own RTSP/SRTP servers disabled (we only need WHEP out).
    let cfg = format!(
        "api:\n  listen: \"{API_ADDR}\"\nrtsp:\n  listen: \"\"\nsrtp:\n  listen: \"\"\nwebrtc:\n  listen: \":{WEBRTC_PORT}\"\nlog:\n  level: warn\n"
    );
    let cfg_path = data_dir.join("go2rtc").join("go2rtc.yaml");
    tokio::fs::write(&cfg_path, cfg).await?;

    let child = crate::proc::tokio_cmd(&exe)
        .args(["-config", &cfg_path.to_string_lossy()])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .kill_on_drop(true)
        .spawn()?;
    *guard = Some(child);
    tracing::info!("go2rtc started (api {API_ADDR}, webrtc :{WEBRTC_PORT})");
    Ok(())
}

/// Register (or refresh) a camera's RTSP source as go2rtc stream `cam<N>`.
/// Best-effort with one retry — go2rtc may still be binding its API port.
pub(crate) async fn register_stream(cam_id: u8, rtsp_url: &str) {
    let client = reqwest::Client::new();
    let api = format!(
        "http://{API_ADDR}/api/streams?name=cam{cam_id}&src={}",
        urlencoding::encode(rtsp_url)
    );
    for attempt in 0..2u8 {
        match client.put(&api).timeout(std::time::Duration::from_secs(5)).send().await {
            Ok(r) if r.status().is_success() => {
                tracing::info!("go2rtc: registered cam{cam_id} for WebRTC live view");
                return;
            }
            Ok(r) => { tracing::warn!("go2rtc: register cam{cam_id} → HTTP {}", r.status()); return; }
            Err(_) if attempt == 0 => tokio::time::sleep(std::time::Duration::from_millis(700)).await,
            Err(e) => tracing::info!("go2rtc unavailable ({e}) — cam{cam_id} stays on HLS live view"),
        }
    }
}

/// WHEP SDP exchange for a camera — the axum route proxies here so the WebView
/// talks only to our token-authed server. Depth-anonymized cameras are refused:
/// go2rtc restreams the RAW camera feed, and anonymized cams must only ever
/// show the depth stream (the HLS fallback, which records depth frames).
pub(crate) async fn whep_exchange(cam_id: u8, offer_sdp: String) -> anyhow::Result<String> {
    if crate::depth::is_anonymized(cam_id) {
        anyhow::bail!("camera is depth-anonymized — WebRTC (raw) live view disabled");
    }
    let resp = reqwest::Client::new()
        .post(format!("http://{API_ADDR}/api/whep?src=cam{cam_id}"))
        .header("Content-Type", "application/sdp")
        .body(offer_sdp)
        .timeout(std::time::Duration::from_secs(10))
        .send().await?;
    if !resp.status().is_success() {
        anyhow::bail!("go2rtc WHEP HTTP {}", resp.status());
    }
    Ok(resp.text().await?)
}
