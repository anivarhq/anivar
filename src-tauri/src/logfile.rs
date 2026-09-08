//! Persistent log file — so "which accelerator armed?" is answerable after the fact.
//!
//! The app runs detached (no console attached), so `tracing_subscriber::fmt::init()`
//! alone meant every boot line — EP canary results, pack provisioning, camera
//! failures — vanished the moment it was printed. That is precisely why a
//! TensorRT-RTX lane that could never load went unnoticed: the fall-through was
//! logged, to nobody.
//!
//! Deliberately tiny: one active file plus one `.1` backup, size-capped. No
//! dependency added — `tracing-appender` only rotates on time, which isn't the
//! bound that matters for a 24/7 appliance.
//!
//! The sink is inert until [`set_dir`] is called (the app-data dir isn't known
//! until Tauri's `setup`), so the handful of lines emitted before that reach the
//! console only. Everything worth reading later happens after boot.

use std::io::Write;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

/// Rotate once the active file passes this. Two files → bounded at ~2× this.
const MAX_BYTES: u64 = 8 * 1024 * 1024;

/// Active log file; the single rotated backup is this plus `.1`.
const LOG_FILENAME: &str = "anivar.log";

struct Sink {
    dir: Option<PathBuf>,
    file: Option<std::fs::File>,
    written: u64,
}

fn sink() -> &'static Mutex<Sink> {
    static SINK: OnceLock<Mutex<Sink>> = OnceLock::new();
    SINK.get_or_init(|| Mutex::new(Sink { dir: None, file: None, written: 0 }))
}

/// Point the log sink at `<data_dir>/logs/`. Call once, from boot.
pub(crate) fn set_dir(data_dir: &std::path::Path) {
    let dir = data_dir.join("logs");
    if std::fs::create_dir_all(&dir).is_err() { return; }
    let mut s = sink().lock().unwrap_or_else(|p| p.into_inner());
    s.dir = Some(dir);
    s.file = None; // opened lazily on the next write
}

impl Sink {
    fn path(&self) -> Option<PathBuf> { self.dir.as_ref().map(|d| d.join(LOG_FILENAME)) }

    /// Open (or reopen after rotation) the active file, seeding `written` from
    /// its current length so restarts don't reset the size budget.
    fn ensure_open(&mut self) -> Option<&mut std::fs::File> {
        if self.file.is_none() {
            let path = self.path()?;
            let f = std::fs::OpenOptions::new().create(true).append(true).open(&path).ok()?;
            self.written = f.metadata().map(|m| m.len()).unwrap_or(0);
            self.file = Some(f);
        }
        self.file.as_mut()
    }

    fn rotate_if_needed(&mut self) {
        if self.written < MAX_BYTES { return; }
        let Some(path) = self.path() else { return };
        self.file = None; // close before renaming (Windows)
        let _ = std::fs::rename(&path, path.with_extension("log.1"));
        self.written = 0;
    }
}

impl Write for Sink {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if self.dir.is_none() { return Ok(buf.len()); } // inert before set_dir
        self.rotate_if_needed();
        let n = match self.ensure_open() {
            Some(f) => f.write(buf).unwrap_or(0),
            None => 0,
        };
        self.written += n as u64;
        Ok(buf.len()) // never fail a log write — a full disk must not kill tracing
    }
    fn flush(&mut self) -> std::io::Result<()> {
        if let Some(f) = self.file.as_mut() { let _ = f.flush(); }
        Ok(())
    }
}

/// `MakeWriter` handing out the shared sink. Cheap: a lock per log line, which is
/// the same cost `fmt::init()`'s stdout lock already pays.
#[derive(Clone, Copy)]
pub(crate) struct FileWriter;

pub(crate) struct Guard(std::sync::MutexGuard<'static, Sink>);

impl Write for Guard {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> { self.0.write(buf) }
    fn flush(&mut self) -> std::io::Result<()> { self.0.flush() }
}

impl tracing_subscriber::fmt::MakeWriter<'_> for FileWriter {
    type Writer = Guard;
    fn make_writer(&self) -> Self::Writer {
        Guard(sink().lock().unwrap_or_else(|p| p.into_inner()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bound that matters: a 24/7 appliance must not fill the disk with logs.
    #[test]
    fn rotates_and_stays_bounded() {
        let dir = std::env::temp_dir().join(format!("sc_log_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        set_dir(&dir);
        let logs = dir.join("logs");

        let line = vec![b'x'; 64 * 1024];
        // Enough to cross MAX_BYTES twice over.
        for _ in 0..(MAX_BYTES / line.len() as u64 * 2 + 8) {
            let mut s = sink().lock().unwrap();
            s.write_all(&line).unwrap();
        }
        sink().lock().unwrap().flush().unwrap();

        let active = std::fs::metadata(logs.join(LOG_FILENAME)).map(|m| m.len()).unwrap_or(0);
        assert!(active > 0, "active log should exist");
        assert!(active <= MAX_BYTES + line.len() as u64, "active log exceeded the cap: {active}");
        assert!(logs.join(format!("{LOG_FILENAME}.1")).exists(), "rotation should leave one backup");

        // Reset the global sink so other tests aren't writing into a temp dir.
        { let mut s = sink().lock().unwrap(); s.dir = None; s.file = None; s.written = 0; }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
