//! GitHub-release update check. (The old global Cmd+K search was removed —
//! search now lives in the Review section, see nvr_recording::search_events.)

/// Check for app updates on GitHub releases.
#[tauri::command]
pub async fn check_for_update(repo: String) -> Result<serde_json::Value, String> {
    let current = env!("CARGO_PKG_VERSION");
    if repo.is_empty() { return Ok(serde_json::json!({ "available": false })); }

    let url = format!("https://api.github.com/repos/{}/releases/latest", repo);
    let resp = reqwest::Client::new()
        .get(&url)
        .header("User-Agent", "Anivar-Updater/1.0")
        .timeout(std::time::Duration::from_secs(10))
        .send().await.map_err(|e| e.to_string())?;

    if !resp.status().is_success() {
        return Err(format!("GitHub API returned {}", resp.status()));
    }

    let body: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
    let latest_tag = body["tag_name"].as_str().unwrap_or("").trim_start_matches('v');
    let name       = body["name"].as_str().unwrap_or(latest_tag);
    let notes      = body["body"].as_str().unwrap_or("").chars().take(500).collect::<String>();
    let published  = body["published_at"].as_str().unwrap_or("");

    // Find Windows .exe or .msi download URL
    let download_url = body["assets"].as_array()
        .and_then(|a| a.iter().find(|asset| {
            let name = asset["name"].as_str().unwrap_or("");
            name.ends_with(".exe") || name.ends_with(".msi") || name.ends_with("-setup.exe")
        }))
        .and_then(|a| a["browser_download_url"].as_str())
        .unwrap_or("")
        .to_string();

    // Simple semver comparison (major.minor.patch)
    let available = is_newer(latest_tag, current);

    Ok(serde_json::json!({
        "available":     available,
        "current":       current,
        "latest":        latest_tag,
        "name":          name,
        "notes":         notes,
        "published":     published,
        "download_url":  download_url,
    }))
}

fn is_newer(latest: &str, current: &str) -> bool {
    fn parse(v: &str) -> (u32, u32, u32) {
        let p: Vec<u32> = v.split('.').map(|x| x.parse().unwrap_or(0)).collect();
        (p.first().copied().unwrap_or(0), p.get(1).copied().unwrap_or(0), p.get(2).copied().unwrap_or(0))
    }
    parse(latest) > parse(current)
}
