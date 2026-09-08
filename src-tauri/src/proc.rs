//! Child-process spawning helpers that make children WINDOW-LESS on Windows.
//!
//! The RELEASE build is `windows_subsystem = "windows"` (no console of its own), so
//! every console child we spawn — ffmpeg, ffprobe, powershell, tar, … —
//! would otherwise get its own console window. The NVR segment post-processor alone
//! spawns a short ffmpeg every ~10 s, so that's a constant storm of flashing windows.
//!
//! **Why `DETACHED_PROCESS` and not `CREATE_NO_WINDOW`:** CREATE_NO_WINDOW creates a
//! console but hides its window — which conhost honors, but on Windows 11 with
//! Windows Terminal as the default console host (the OS default since 22H2/"Let
//! Windows decide"), spawning a console child can still FLASH a Terminal window
//! despite the flag (observed live on this machine: each post-process ffmpeg popped
//! an empty terminal that closed ~1 s later). DETACHED_PROCESS instead gives the
//! child **no console at all** — there is nothing for conhost OR Windows Terminal to
//! host, so nothing can flash, on any Windows version. Safe here because every child
//! we spawn has its stdio explicitly piped or nulled (none of them need a console).
//! Per CreateProcess docs the two flags are mutually exclusive — use DETACHED alone.
//!
//! Use `proc::tokio_cmd(..)` / `proc::std_cmd(..)` EVERYWHERE instead of
//! `tokio::process::Command::new` / `std::process::Command::new`. This module is the
//! ONLY place the raw constructors may appear.

/// Windows `DETACHED_PROCESS` creation flag — the child gets NO console at all.
#[cfg(windows)]
pub const DETACHED_PROCESS: u32 = 0x0000_0008;

/// A `tokio::process::Command` whose child can never show a console window.
pub fn tokio_cmd(program: impl AsRef<std::ffi::OsStr>) -> tokio::process::Command {
    #[allow(unused_mut)]
    let mut c = tokio::process::Command::new(program);
    #[cfg(windows)]
    {
        c.creation_flags(DETACHED_PROCESS);
    }
    c
}

/// Put THIS process into a Job Object with KILL_ON_JOB_CLOSE, so every child we
/// spawn (ffmpeg recorders/detection pipes, ollama, ...) dies with us
/// — including Task Manager force-kills and crashes. Without this, a force-killed
/// run leaves its recorder ffmpeg ALIVE (the dshow mic input keeps it running
/// after the frame pipe dies) encoding forever and locking .tmp.mp4 files; six
/// accumulated zombies measurably crippled the whole machine. The job handle is
/// deliberately leaked: the OS closes it when we die, and THAT close is what
/// tears the job (and all children) down. Best-effort — failure just means the
/// boot-time orphan sweep remains the only safety net.
#[cfg(windows)]
pub fn adopt_kill_on_close_job() {
    use windows::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
        SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };
    use windows::Win32::System::Threading::GetCurrentProcess;
    unsafe {
        let job = match CreateJobObjectW(None, None) {
            Ok(h) => h,
            Err(e) => { tracing::warn!("job object create failed: {e}"); return; }
        };
        let mut info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if let Err(e) = SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            &info as *const _ as *const std::ffi::c_void,
            std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
        ) {
            tracing::warn!("job object limit set failed: {e}");
            return;
        }
        if let Err(e) = AssignProcessToJobObject(job, GetCurrentProcess()) {
            tracing::warn!("job object assign failed: {e}");
        }
    }
}
#[cfg(not(windows))]
pub fn adopt_kill_on_close_job() {}

/// Ask Windows Error Reporting to RELAUNCH the app after an unexpected crash —
/// including fast-fail aborts (0xc0000409 from the GPU stack) that the panic
/// hook can never catch and that killed the NVR silently. The OS only honors
/// the restart when the process has lived ≥ 60 s, so a crash-on-boot loop
/// can't spin. Deliberate quits (tray → Quit → process::exit) are unaffected.
#[cfg(windows)]
pub fn register_crash_restart() {
    use windows::core::PCWSTR;
    use windows::Win32::System::Recovery::{
        RegisterApplicationRestart, REGISTER_APPLICATION_RESTART_FLAGS,
    };
    // RESTART_NO_HANG | RESTART_NO_PATCH | RESTART_NO_REBOOT → crash-only.
    const FLAGS: REGISTER_APPLICATION_RESTART_FLAGS = REGISTER_APPLICATION_RESTART_FLAGS(2 | 4 | 8);
    // Marker arg so the relaunched instance can tell (logged at boot).
    let cmd: Vec<u16> = "--crash-restart\0".encode_utf16().collect();
    let res = unsafe { RegisterApplicationRestart(PCWSTR(cmd.as_ptr()), FLAGS) };
    match res {
        Ok(()) => tracing::info!("crash auto-restart registered — Windows relaunches the NVR if it dies unexpectedly"),
        Err(e) => tracing::warn!("RegisterApplicationRestart failed: {e}"),
    }
}
#[cfg(not(windows))]
pub fn register_crash_restart() {}

/// A `std::process::Command` whose child can never show a console window.
pub fn std_cmd(program: impl AsRef<std::ffi::OsStr>) -> std::process::Command {
    #[allow(unused_mut)]
    let mut c = std::process::Command::new(program);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        c.creation_flags(DETACHED_PROCESS);
    }
    c
}
