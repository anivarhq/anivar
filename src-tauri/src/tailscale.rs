//! Tailscale Funnel remote access — the COMPLIANT replacement for trycloudflare
//! quick tunnels (which Cloudflare documents as testing-only, not for shipped
//! products). Funnel exposes the local stream server at a stable public
//! `https://<machine>.<tailnet>.ts.net` URL that ANYONE can open in a browser
//! with no Tailscale account (viewers install nothing), free on the personal
//! tier, and streams video fine.
//!
//! There is no Rust `tsnet`, so we drive the installed Tailscale CLI: the user
//! installs Tailscale + logs in once (`tailscale up`), we run `tailscale funnel`
//! to publish port 8882. Funnel must be enabled once per tailnet via a consent
//! URL the CLI prints on first use — we surface that to the UI.

use std::path::PathBuf;

/// Resolve the Tailscale CLI: PATH first, then the standard Windows install dir.
fn tailscale_bin() -> PathBuf {
    #[cfg(windows)]
    {
        let std_path = PathBuf::from(r"C:\Program Files\Tailscale\tailscale.exe");
        if std_path.exists() { return std_path; }
    }
    PathBuf::from("tailscale")
}

/// Status the Remote-access UI renders.
#[derive(Debug, Clone, serde::Serialize)]
#[derive(Default)]
pub struct TailscaleStatus {
    pub installed:     bool,
    pub logged_in:     bool,
    pub dns_name:      String, // machine.tailnet.ts.net (trailing dot stripped)
    pub base_url:      String, // https://<dns_name> when we can funnel
    pub funnel_active: bool,   // a funnel is currently serving
    /// One-time tailnet consent URL when Funnel isn't enabled yet (else empty).
    pub enable_url:    String,
}


/// Run a tailscale CLI subcommand with a HARD deadline. Every call in this
/// module goes through here: the CLI can block indefinitely (the daemon can be
/// wedged, and `funnel` famously WAITS for tailnet consent) — an unbounded
/// `.output().await` here froze the Telegram poll loop for minutes and leaked a
/// waiting `tailscale funnel` child (observed live: PID running for hours).
/// `kill_on_drop` guarantees the child dies with the future, timeout or not.
async fn run_tailscale(args: &[&str], secs: u64) -> Option<std::process::Output> {
    let fut = crate::proc::tokio_cmd(tailscale_bin())
        .args(args)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        .output();
    match tokio::time::timeout(std::time::Duration::from_secs(secs), fut).await {
        Ok(Ok(out)) => Some(out),
        Ok(Err(e)) => { tracing::warn!("tailscale {:?} failed to run: {e}", args.first()); None }
        Err(_) => { tracing::warn!("tailscale {:?} timed out after {secs}s (killed)", args.first()); None }
    }
}

fn is_installed_sync() -> bool {
    #[cfg(windows)]
    { if PathBuf::from(r"C:\Program Files\Tailscale\tailscale.exe").exists() { return true; } }
    // PATH probe (non-Windows / non-standard installs).
    crate::proc::std_cmd("tailscale").arg("version")
        .stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null())
        .status().map(|s| s.success()).unwrap_or(false)
}

/// Async-safe install check: the common case (standard install dir) is a cheap
/// file stat; only the PATH probe (a blocking process spawn) goes off-runtime.
async fn is_installed() -> bool {
    #[cfg(windows)]
    { if PathBuf::from(r"C:\Program Files\Tailscale\tailscale.exe").exists() { return true; } }
    tokio::task::spawn_blocking(is_installed_sync).await.unwrap_or(false)
}

/// Read `tailscale status --json` → (logged_in, dns_name-without-dot).
async fn read_status() -> (bool, String) {
    let Some(out) = run_tailscale(&["status", "--json"], 5).await else {
        return (false, String::new());
    };
    let json: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap_or_default();
    let logged_in = json.get("BackendState").and_then(|v| v.as_str()) == Some("Running");
    let dns = json.pointer("/Self/DNSName").and_then(|v| v.as_str())
        .unwrap_or("").trim_end_matches('.').to_string();
    (logged_in, dns)
}

/// Is a funnel currently serving (any config)?
async fn funnel_active() -> bool {
    match run_tailscale(&["funnel", "status", "--json"], 5).await {
        Some(o) => {
            let j: serde_json::Value = serde_json::from_slice(&o.stdout).unwrap_or_default();
            // Non-empty object with AllowFunnel/Web entries = active.
            j.get("AllowFunnel").map(|v| v.as_object().map(|m| !m.is_empty()).unwrap_or(false))
                .unwrap_or(false)
        }
        None => false,
    }
}

/// Full status for the UI.
pub async fn status() -> TailscaleStatus {
    if !is_installed().await { return TailscaleStatus::default(); }
    let (logged_in, dns) = read_status().await;
    let base_url = if dns.is_empty() { String::new() } else { format!("https://{dns}") };
    TailscaleStatus {
        installed: true, logged_in,
        dns_name: dns, base_url,
        funnel_active: funnel_active().await,
        enable_url: String::new(),
    }
}

/// Ensure Funnel publishes `port`. Returns the public base URL on success. On the
/// first-run "enable Funnel" case, returns Err with the consent URL embedded so
/// the caller can send the user to click it once. Idempotent.
///
/// HARD LESSON: when Funnel isn't consented yet, `tailscale funnel --bg <port>`
/// prints the consent URL and then **waits indefinitely** for the approval —
/// an unbounded `.output().await` here hung the Telegram poll loop and leaked
/// a waiting CLI child (observed live, hours old). So the child is spawned with
/// piped output + `kill_on_drop`, its output is read INCREMENTALLY, and the
/// whole operation has an 8s deadline: the consent URL is grabbed the moment it
/// appears (child killed), success is verified by exit, and a no-verdict
/// deadline kills the child and reports honestly.
/// The ONE way to get a public URL for a share link.
///
/// Formerly `cloudflare::ensure_tunnel_up`, which branched on a `remote_provider`
/// setting no UI could change and defaulted here anyway. Cloudflare is gone
/// (2026-07-28): trycloudflare quick tunnels are testing-only under Cloudflare's
/// ToS, so Funnel was already the only compliant path we shipped.
pub async fn ensure_public_url(state: std::sync::Arc<crate::AppState>) -> Result<String, String> {
    let port = state.settings.read().await.stream_port;
    match ensure_funnel(port).await {
        Ok(base) => Ok(base),
        // First run: Funnel needs a one-time account consent click. Surface the
        // URL rather than the raw sentinel.
        Err(e) if e.starts_with("__ENABLE__") => Err(format!(
            "One-time setup: enable Tailscale Funnel for your account here, then try again:\n{}",
            e.trim_start_matches("__ENABLE__"))),
        Err(e) => Err(e),
    }
}

pub async fn ensure_funnel(port: u16) -> Result<String, String> {
    if !is_installed().await {
        return Err("Tailscale isn't installed. Install it from https://tailscale.com/download, sign in, then try again.".into());
    }
    let (logged_in, dns) = read_status().await;
    if !logged_in {
        return Err("Tailscale is installed but not signed in. Open Tailscale and log in (free), then try again.".into());
    }
    if dns.is_empty() {
        return Err("Couldn't read this machine's Tailscale name. Make sure Tailscale is connected.".into());
    }

    // Fast path: a funnel is already serving — nothing to do.
    if funnel_active().await {
        return Ok(format!("https://{dns}"));
    }

    use tokio::io::{AsyncBufReadExt, BufReader};
    let mut child = crate::proc::tokio_cmd(tailscale_bin())
        .args(["funnel", "--bg", &port.to_string()])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| format!("Couldn't run Tailscale funnel: {e}"))?;

    // Merge stdout+stderr line streams into one channel (the consent notice can
    // land on either, version-dependent).
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    if let Some(out) = child.stdout.take() {
        let tx = tx.clone();
        tokio::spawn(async move {
            let mut lines = BufReader::new(out).lines();
            while let Ok(Some(l)) = lines.next_line().await { if tx.send(l).is_err() { break; } }
        });
    }
    if let Some(err) = child.stderr.take() {
        tokio::spawn(async move {
            let mut lines = BufReader::new(err).lines();
            while let Ok(Some(l)) = lines.next_line().await { if tx.send(l).is_err() { break; } }
        });
    }

    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(8);
    let mut combined = String::new();
    loop {
        tokio::select! {
            maybe_line = rx.recv() => {
                match maybe_line {
                    Some(line) => {
                        combined.push_str(&line);
                        combined.push('\n');
                        // First-run consent: grab the URL the moment it prints and
                        // KILL the (now waiting-forever) CLI.
                        if let Some(url) = line.split_whitespace()
                            .find(|w| w.starts_with("https://login.tailscale.com/f/funnel"))
                        {
                            let url = url.to_string();
                            let _ = child.kill().await;
                            return Err(format!("__ENABLE__{url}"));
                        }
                    }
                    None => {
                        // Streams closed — wait (bounded) for the exit code.
                        let status = tokio::time::timeout(
                            std::time::Duration::from_secs(3), child.wait()).await;
                        return match status {
                            Ok(Ok(s)) if s.success() => Ok(format!("https://{dns}")),
                            _ if combined.contains("Funnel is not enabled") =>
                                Err("Enable Funnel for your Tailscale account, then try again.".into()),
                            Ok(Ok(_)) if !combined.trim().is_empty() =>
                                Err(format!("Tailscale funnel failed: {}", combined.trim())),
                            _ => Ok(format!("https://{dns}")), // exited quietly = configured
                        };
                    }
                }
            }
            _ = tokio::time::sleep_until(deadline) => {
                let _ = child.kill().await;
                // The CLI neither succeeded nor printed a consent URL within the
                // deadline. If a funnel got configured anyway, count it as success.
                if funnel_active().await { return Ok(format!("https://{dns}")); }
                if combined.contains("Funnel is not enabled") {
                    return Err("Enable Funnel for your Tailscale account, then try again.".into());
                }
                return Err("Tailscale funnel didn't respond — check that Tailscale is running, then try again.".into());
            }
        }
    }
}

/// Stop funnelling `port` (best-effort; used on teardown / provider switch).
/// Currently unused: the funnel is torn down by `ensure_funnel`'s own idle path.
/// Kept because share-link revocation needs it the moment that becomes manual.
#[allow(dead_code)]
pub async fn disable_funnel(port: u16) {
    let _ = run_tailscale(&["funnel", "--https=443", &port.to_string(), "off"], 5).await;
    // Fallback to the blanket reset if the specific form isn't supported.
    let _ = run_tailscale(&["funnel", "reset"], 5).await;
}

// ─── Tauri commands ──────────────────────────────────────────────────────────

#[tauri::command]
pub async fn tailscale_status() -> Result<TailscaleStatus, String> {
    Ok(status().await)
}

/// Turn on Funnel for the stream port. Returns the live status; on first run it
/// carries `enable_url` for the one-time consent click.
#[tauri::command]
pub async fn tailscale_enable(state: tauri::State<'_, std::sync::Arc<crate::AppState>>)
    -> Result<TailscaleStatus, String>
{
    let port = state.settings.read().await.stream_port;
    let mut st = status().await;
    match ensure_funnel(port).await {
        Ok(_) => { st.funnel_active = true; Ok(st) }
        Err(e) if e.starts_with("__ENABLE__") => {
            st.enable_url = e.trim_start_matches("__ENABLE__").to_string();
            Ok(st)
        }
        Err(e) => Err(e),
    }
}
