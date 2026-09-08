// Prevents additional console window on Windows in release
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

/// Declare PerMonitorV2 DPI awareness before anything else runs.
///
/// WebView2 needs PerMonitorV2. Without it the process ends up
/// PER_MONITOR_AWARE (v1), and WebView2 then treats the window's PHYSICAL client
/// size as its CSS viewport while still rasterising at devicePixelRatio — so on
/// any display above 100% the page is laid out dpr times too large for the window
/// it lives in. Measured on a 125% display before this call existed:
///
/// ```text
///   native client   1280 x 800 device px
///   page laid out   1280 x 800 CSS px  ->  1600 x 1000 device px
///   cut off         320 px right, 200 px bottom
/// ```
///
/// Every screen silently lost its right and bottom fifth. In the Review player
/// that is exactly where the Timeline / Bookmark / Export buttons and the whole
/// scrubbable timeline sit, so they rendered off-window and were unreachable —
/// the UI looked broken for reasons that had nothing to do with the code drawing
/// it, and no amount of fixing the timeline could have helped.
///
/// This is done in code rather than in the application manifest on purpose. A
/// manifest was tried first and does not work here: with both `<dpiAware>true/pm`
/// and `<dpiAwareness>PerMonitorV2` present Windows honoured the older element and
/// left the process on v1, and dropping to only the 2016 element left it UNAWARE
/// (which fits the window but makes Windows bitmap-upscale the whole UI). A
/// process's DPI awareness is locked by the FIRST caller to set it, so doing it
/// here — before Tauri/wry gets a chance to ask for v1 — is deterministic.
///
/// Verify after touching this: `GetAwarenessFromDpiAwarenessContext` on the main
/// window must report PER_MONITOR_AWARE_V2, and in the page
/// `innerWidth * devicePixelRatio` must equal the native client width.
#[cfg(windows)]
fn set_per_monitor_v2_dpi_awareness() {
    // DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2. The context handles are
    // sentinel values, not real pointers.
    const PER_MONITOR_AWARE_V2: isize = -4;
    extern "system" {
        fn SetProcessDpiAwarenessContext(value: isize) -> i32;
    }
    // Failure is not fatal: it only means something already set awareness for
    // this process (a manifest, or an injected shim), and that choice stands.
    unsafe {
        SetProcessDpiAwarenessContext(PER_MONITOR_AWARE_V2);
    }
}

fn main() {
    #[cfg(windows)]
    set_per_monitor_v2_dpi_awareness();

    anivar_lib::run();
}
