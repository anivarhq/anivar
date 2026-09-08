fn main() {
    // Custom Windows manifest, solely to declare PerMonitorV2 DPI awareness.
    // Tauri's default manifest declares no DPI setting, which leaves the process
    // on per-monitor v1 and makes WebView2 lay the page out devicePixelRatio
    // times too large for its window. See `anivar.manifest` for the measurements.
    //
    // Non-Windows targets ignore `windows_attributes` entirely.
    let attrs = tauri_build::Attributes::new().windows_attributes(
        tauri_build::WindowsAttributes::new().app_manifest(include_str!("anivar.manifest")),
    );
    tauri_build::try_build(attrs).expect("failed to run tauri-build");
}
