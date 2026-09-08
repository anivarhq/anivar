/**
 * Custom window titlebar — replaces the native OS frame (decorations:false), so the
 * minimize / maximize / close controls live in the app's own top ribbon.
 *
 * Modelled on Readest's WindowButtons.tsx: the bar itself is the drag region
 * (`startDragging()` on press, `toggleMaximize()` on double-click), with the control
 * buttons excluded from dragging. Window control buttons call the Tauri window API
 * (`@tauri-apps/api/window`). Renders nothing in a non-Tauri (web) build.
 */
import { useEffect, useState } from "react";
import { Fluffy } from "./Fluffy";
import styles from "./TitleBar.module.css";

const isTauri = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

async function appWindow() {
  const { getCurrentWindow } = await import("@tauri-apps/api/window");
  return getCurrentWindow();
}

export function TitleBar() {
  const [maximized, setMaximized] = useState(false);

  useEffect(() => {
    if (!isTauri) return;
    let unlisten: (() => void) | undefined;
    (async () => {
      try {
        const w = await appWindow();
        setMaximized(await w.isMaximized());
        // Keep the maximize/restore glyph in sync when the user resizes/snaps.
        unlisten = await w.onResized(async () => setMaximized(await w.isMaximized()));
      } catch { /* non-fatal */ }
    })();
    return () => unlisten?.();
  }, []);

  if (!isTauri) return null; // web build keeps the browser chrome

  // Press-to-drag, double-click-to-maximize — but never when the press lands on a
  // control button (or anything that opts out via .no-drag).
  const onBarMouseDown = async (e: React.MouseEvent) => {
    if (e.button !== 0) return;
    const t = e.target as HTMLElement;
    if (t.closest("button") || t.closest(".no-drag")) return;
    try { (await appWindow()).startDragging(); } catch { /* ignore */ }
  };
  const onBarDoubleClick = async (e: React.MouseEvent) => {
    const t = e.target as HTMLElement;
    if (t.closest("button") || t.closest(".no-drag")) return;
    await toggleMaximize();
  };

  const minimize     = async () => { try { (await appWindow()).minimize(); } catch { /* ignore */ } };
  const close        = async () => { try { (await appWindow()).close(); } catch { /* ignore */ } };
  const toggleMaximize = async () => {
    try {
      const w = await appWindow();
      await w.toggleMaximize();
      setMaximized(await w.isMaximized());
    } catch { /* ignore */ }
  };

  return (
    <div className={styles.bar} onMouseDown={onBarMouseDown} onDoubleClick={onBarDoubleClick}>
      <div className={styles.brand}>
        <Fluffy size={17} className={styles.brandIcon} />
        <span className={styles.brandText}>Anivar</span>
      </div>

      <div className={styles.spacer} />

      <div className={styles.controls}>
        <button className={styles.ctl} onClick={minimize} aria-label="Minimize" title="Minimize">
          <svg width="11" height="11" viewBox="0 0 24 24"><path fill="currentColor" d="M20 14H4v-2h16" /></svg>
        </button>
        <button className={styles.ctl} onClick={toggleMaximize} aria-label={maximized ? "Restore" : "Maximize"} title={maximized ? "Restore" : "Maximize"}>
          {maximized ? (
            <svg width="11" height="11" viewBox="0 0 24 24"><path fill="currentColor" d="M8 4h12v12h-4v4H4V8h4zm0 2H6v10h8v-2H8zm2 8h6V6h-6z" /></svg>
          ) : (
            <svg width="11" height="11" viewBox="0 0 24 24"><path fill="currentColor" d="M4 4h16v16H4zm2 4v10h12V8z" /></svg>
          )}
        </button>
        <button className={`${styles.ctl} ${styles.close}`} onClick={close} aria-label="Close" title="Close">
          <svg width="11" height="11" viewBox="0 0 24 24"><path fill="currentColor" d="M19 6.41L17.59 5L12 10.59L6.41 5L5 6.41L10.59 12L5 17.59L6.41 19L12 13.41L17.59 19L19 17.59L13.41 12z" /></svg>
        </button>
      </div>
    </div>
  );
}
