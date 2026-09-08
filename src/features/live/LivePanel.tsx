/**
 * LivePanel — the primary monitoring screen.
 *
 * Grid mode:  All enabled cameras shown in a configurable grid.
 *             Click any tile → Focus mode.
 *
 * Focus mode: Single camera at full height with:
 *             - Full CameraView (motion detection active, AI overlay, face re-ID)
 *             - 24h mini-timeline below (NVR blocks + event markers)
 *             - Recent events strip
 *
 * No modals. No page transitions. One surface.
 */
import { useEffect, useRef, useState, useCallback, useMemo } from "react";
import { useStore } from "../../store";
import { useShallow } from "zustand/react/shallow";
import { api, MotionEvent } from "../../api";
import { CameraView } from "../camera/CameraView";
import { FocusHeader, KebabMenu, type PanelState, PANEL_CLOSED, PANEL_OPEN } from "./FocusHeader";
import { BottomPanel } from "./BottomPanel";
import { BrowseDrawer } from "./BrowseDrawer";
import { ClipOverlay, type ClipOverlayHandle } from "./ClipOverlay";
import { CameraSettingsModal } from "./CameraSettingsModal";
import type { Segment } from "../nvr/NVRPanel";
import { usePlayback } from "./usePlayback";
import styles from "./LivePanel.module.css";
import { fetchClipStartMs } from "../../lib/clipStart";
import focusStyles from "./FocusHeader.module.css";
import { listen } from "@tauri-apps/api/event";
import {
  AlertTriangle, ArrowLeft, Clock, LayoutGrid, Link, Loader, Monitor, Plus, RefreshCw, Wifi,
} from "lucide-react";
// `startOfLocalDay`/`endOfLocalDay` used to be redefined at the top of this
// file. lib/time.ts already had them, is unit-tested, and every sibling in
// this folder imports from there.
import { dayBoundsUtc, dayStartMs, dayEndMs } from "../../lib/time";
import { writeCamSource, clearCamSource } from "../../lib/camSource";

// ── Local day-window helpers (single source for the timeline's day math) ────────
// Day boundaries are computed in LOCAL time; nvr_segments are stored UTC and the
// API query converts. endOfLocalDay = next local midnight − 1ms, so it's DST-safe
// (a transition day is really 23h/25h; `+24h` would be off by an hour).
import { AddCameraModal, type ModalTab } from "./AddCameraModal";
export { AddCameraModal };  // re-export: Onboarding historically imports it from here


// ── Layout options ────────────────────────────────────────────────────────────

const LAYOUTS = [
  { cols: 1, label: "1" },
  { cols: 2, label: "4" },
  { cols: 3, label: "9" },
  { cols: 4, label: "16" },
] as const;

// ── Camera grid tile — runs real CameraView so the feed is always active ────────

function CameraTile({
  camId, name, onDoubleClick, onRemove,
}: {
  camId: number;
  name: string;
  onDoubleClick: () => void;
  onRemove: () => void;
}) {
  return (
    <div className={styles.tile}>
      <div className={styles.tileInner}>
        {/* `onRemove` used to be `() => {}` here. It is truthy, so CameraView's
            red "Remove Slot" button rendered in a broken tile's error state,
            wiped cam_source_* from localStorage, and did nothing — loadCams
            rewrote the key on the very next poll. Wired to the real removal. */}
        <CameraView camId={camId} onRemove={onRemove} />
      </div>
      {/* Name pill — always visible, stacked ABOVE CameraView's z-indexed
          layers (depth canvas z:5, buttons z:6); it used to render BEHIND
          them. The red pulsing "live" dot is gone — decoration, not signal. */}
      <div className={styles.tileOverlay} onDoubleClick={onDoubleClick}>
        {name && <span className={styles.tileName}>{name}</span>}
      </div>
    </div>
  );
}

// ── Main ──────────────────────────────────────────────────────────────────────

// `onOpenSettings` was declared here and passed from App.tsx, and never called
// once in this file. Dropped rather than left as a prop that looks load-bearing.
export function LivePanel() {
  // focusedCam + liveView live in the store so the side-nav can swap the Live icon
  // → CCTV and highlight Review while in the player.
  const { streamInfo, settings, focusedCam, setFocusedCam, liveView, setLiveView } = useStore(useShallow(s => ({ streamInfo: s.streamInfo, settings: s.settings, focusedCam: s.focusedCam, setFocusedCam: s.setFocusedCam, liveView: s.liveView, setLiveView: s.setLiveView })));

  const [configs, setConfigs]     = useState<Array<{ cam_id: number; name: string; enabled: boolean }>>([]);
  /** Has the first camera fetch finished, and did it fail? Without these,
   *  `configs === []` meant three different things at once. */
  const [loaded, setLoaded]       = useState(false);
  /** Which path the user picked on the empty grid, so the modal opens there. */
  const [addTab, setAddTab]       = useState<ModalTab | undefined>(undefined);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [activeCams, setActiveCams] = useState<number[]>([]);
  const [events, setEvents]       = useState<MotionEvent[]>([]);
  // v14: focus-mode bottom panel — null = no panel, "events" = events list,
  // "history" = NVR timeline. Replaces the v13 floating CameraHistoryModal.
  const [panels,        setPanels]        = useState<PanelState>(PANEL_CLOSED);
  const [selectedDate,  setSelectedDate]  = useState<Date>(() => {
    const d = new Date(); d.setHours(0, 0, 0, 0); return d;
  });
  const togglePanel = (which: "events" | "timeline") =>
    setPanels(p => ({ ...p, [which]: !p[which] }));
  // v21: timeline view-window state lifted so the FocusHeader date-nav
  // (prev / date / next / Now / [− 24h +]) can drive the zoom even when
  // the Timeline drawer isn't mounted. HistoryDrawer consumes these as
  // controlled props.
  const [viewStart, setViewStart] = useState<number>(() => {
    const d = new Date(); d.setHours(0, 0, 0, 0);
    return d.getTime();
  });
  const [viewEnd, setViewEnd] = useState<number>(() => Date.now());
  const [selectedEvent, setSelectedEvent] = useState<MotionEvent | null>(null);
  // v26: NVR segment list for the focused camera. Feeds the controller's
  // coverage bands (timeline gaps, click snapping, "camera was off").
  const [focusedCamSegments, setFocusedCamSegments] = useState<Segment[]>([]);

  // ── Playback ──────────────────────────────────────────────────────────────
  // ONE controller, shared with the Review player (see usePlayback.ts). This
  // surface used to carry its own copy of seek / skip / ended / coverage /
  // anchor, and the two had already drifted into opposite bugs: Review quit at
  // the end of a chunk, this one re-anchored every few seconds at the live edge
  // and rebuilt the whole pipeline each time.
  const focusDayBounds = useMemo(() => dayBoundsUtc(selectedDate), [selectedDate]);
  const {
    clipSource, setClipSource, playheadMs, setPlayheadMs,
    isOffGap, historyUrl, overlayRef,
    seekTo, skip, handleEnded, onSourceReady,
  } = usePlayback({
    camId: focusedCam ?? 0,
    segments: focusedCamSegments,
    streamInfo,
    dayFromUtc: focusDayBounds.fromUtc,
    dayToUtc: focusDayBounds.toUtc,
  });

  // TRUE playback start for EVENT CLIPS only (server keyframe/coverage snap). A
  // recorded chunk needs no such probe: its playlist carries PROGRAM-DATE-TIME,
  // which is exact by construction and cannot disagree with the media.
  const [trueAnchorMs, setTrueAnchorMs] = useState<number | null>(null);
  useEffect(() => {
    setTrueAnchorMs(null);
    if (clipSource.kind !== "event" || !streamInfo) return;
    let alive = true;
    fetchClipStartMs(streamInfo, { eventId: clipSource.eventId })
      .then(ms => { if (alive && ms != null) setTrueAnchorMs(ms); });
    return () => { alive = false; };
  }, [clipSource, streamInfo]);

  // v24: live mirrors of clipSource / selectedEvent / playheadMs read by the
  // keyboard listener at fire time. Without these refs the listener would
  // need to be in the deps array and would be rebound every ~250 ms while
  // a clip plays (playheadMs ticks at the timeupdate rate).
  const clipSourceRef    = useRef(clipSource);
  const selectedEventRef = useRef(selectedEvent);
  useEffect(() => { clipSourceRef.current    = clipSource;    }, [clipSource]);
  useEffect(() => { selectedEventRef.current = selectedEvent; }, [selectedEvent]);
  // Grid layout choice persists across restarts — picking 1/4/9/16 is a
  // preference, not a session whim. Until the user picks one, the layout
  // AUTO-FITS the camera count (1 cam ⇒ 1-up, 4 ⇒ 4-up, …) so a fresh install
  // never shows one camera stranded in a 4-up grid of empty slots, and adding
  // cameras never leaves them hidden behind a too-small default.
  const userPickedLayout = useRef(localStorage.getItem("sc.gridCols") !== null);
  const [cols, setCols] = useState(() => {
    const n = Number(localStorage.getItem("sc.gridCols"));
    return n >= 1 && n <= 4 ? n : 1;
  });
  const pickLayout = useCallback((c: number) => {
    userPickedLayout.current = true;
    localStorage.setItem("sc.gridCols", String(c));
    setCols(c);
  }, []);
  const [showAddModal, setShowAddModal] = useState(false);
  // Floating per-camera settings window (opened from the focus kebab).
  const [settingsCamId, setSettingsCamId] = useState<number | null>(null);
  // 12h/24h time format for the focus timeline + clocks. Shares the NVR key so
  // the preference is consistent across the app.
  const [use12h, setUse12h] = useState(() => localStorage.getItem("nvr_12h") === "1");
  const toggle12h = useCallback(() => setUse12h(v => {
    const n = !v; localStorage.setItem("nvr_12h", n ? "1" : "0"); return n;
  }), []);

  // ── Ambient color-bleed glow ───────────────────────────────────────────────
  // Samples whatever <video> is currently rendering inside the focus card —
  // live CameraView OR the event ClipOverlay — into a small canvas, then CSS
  // blurs that canvas so the on-screen scene's colors refract into the padding.
  // One mechanism for both modes; no cross-component refs and no pixel readback
  // (drawImage of a cross-origin clip video taints the canvas, which is fine for
  // DISPLAY — we never call getImageData/toDataURL on it).
  const focusInnerRef  = useRef<HTMLDivElement>(null);
  const ambientCanvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    if (focusedCam === null) return;
    let raf = 0; let last = 0;
    const tick = (ts: number) => {
      raf = requestAnimationFrame(tick);
      if (ts - last < 330) return;       // ~3 fps — cheap; it's only an ambient glow
      last = ts;
      const cv = ambientCanvasRef.current;
      const host = focusInnerRef.current;
      if (!cv || !host) return;
      // Prefer a playing/ready video; fall back to an <img> (MJPEG) frame.
      // Search in REVERSE DOM order so the event ClipOverlay's <video> (mounted
      // AFTER the live CameraView) wins while a clip is playing — that's what
      // makes the glow refract the EVENT, not the live feed underneath.
      const vids = (Array.from(host.querySelectorAll("video")) as HTMLVideoElement[]).reverse();
      const vid = vids.find(v => v.readyState >= 2 && v.videoWidth > 0 && !v.paused)
        ?? vids.find(v => v.readyState >= 2 && v.videoWidth > 0);
      const img = host.querySelector("img") as HTMLImageElement | null;
      const src: CanvasImageSource | null =
        vid ?? (img && img.naturalWidth > 0 ? img : null);
      if (!src) return;
      const ctx = cv.getContext("2d");
      if (!ctx) return;
      // drawImage of a cross-origin video can THROW (not just taint) in the
      // webview; the clip <video> sets crossOrigin="anonymous" (footage routes
      // send CORS) so this stays clean. try/catch guards the not-ready window.
      try { ctx.drawImage(src, 0, 0, cv.width, cv.height); } catch { /* not ready */ }
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [focusedCam]);

  // Load camera configs + active cams, and sync DB → localStorage so CameraView auto-starts
  const loadCams = useCallback(async () => {
    // A REJECTED load is not an empty one.
    //
    // This had no catch and no loaded flag, so a `Promise.all` rejection left
    // `configs` at [] permanently and the panel rendered "No cameras
    // configured — add your first camera" over a working install whose backend
    // had merely hiccuped. A DB lock read as onboarding.
    try {
      const [cfgs, active] = await Promise.all([
        api.getCameraConfigs(),
        api.getActiveCameras(),
      ]);
      // Sync every enabled camera's source to localStorage so CameraView auto-starts.
      // Only write if not already there (don't clobber CameraView's resolved source).
      for (const cfg of cfgs) {
        if (cfg.enabled && !localStorage.getItem(`cam_source_${cfg.cam_id}`)) {
          writeCamSource(cfg.cam_id, cfg);
        }
      }
      setConfigs(cfgs.filter(c => c.enabled));
      setActiveCams(active);
      setLoadError(null);
    } catch (e: any) {
      setLoadError(e?.message ?? String(e));
    } finally {
      setLoaded(true);
    }
  }, []);

  /** Skip, bound for the keyboard listener so it never rebinds.
   *
   *  The skip RULE lives in the controller; this is only the ref that lets the
   *  once-bound keydown handler reach the current one. Event bounds come from
   *  the selected event, which is this surface's own state.
   */
  const skipRef = useRef<(d: number) => void>(() => {});
  skipRef.current = (deltaSec: number) => {
    const sel = selectedEventRef.current;
    const bounds = sel ? (() => {
      const startMs = new Date(sel.started_at).getTime();
      return {
        startMs,
        endMs: sel.ended_at ? new Date(sel.ended_at).getTime()
                            : startMs + (sel.duration_secs ?? 10) * 1000,
      };
    })() : undefined;
    skip(deltaSec, bounds);
  };

  // Disable + forget a camera. Shared by the focus kebab and the floating
  // camera-settings window. Exits focus mode immediately so the user isn't left
  // staring at a frozen frame while the API call completes.
  const removeCamera = useCallback(async (id: number) => {
    setFocusedCam(null);
    setLiveView("grid");
    setClipSource({ kind: "none" });
    setSettingsCamId(null);
    try {
      await api.setCameraConfig({
        cam_id: id, name: "",
        source_type: "native", source_url: "", device_id: "", enabled: false,
      });
      clearCamSource(id);
    } catch (e) {
      console.error("Failed to remove camera", e);
    } finally {
      loadCams();
    }
  }, [loadCams]);

  useEffect(() => { loadCams(); }, [loadCams]);

  // Refresh grid immediately when a camera is added/changed in Settings
  useEffect(() => {
    const u = listen("cameras:updated", () => loadCams());
    return () => { u.then(f => f()); };
  }, [loadCams]);

  // Poll active cameras every 5s (skipped while the window is hidden).
  useEffect(() => {
    const id = setInterval(() => {
      if (document.hidden) return;
      api.getActiveCameras().then(setActiveCams).catch(() => {});
    }, 5000);
    return () => clearInterval(id);
  }, []);

  // Load events for the currently-selected date. Refetches whenever the user
  // picks a new day in the FocusHeader's date picker; refreshes on a 30s
  // tick but only when the selected day is today (past days don't change).
  useEffect(() => {
    // TZ-CORRECT local-day → UTC bounds via the time SSOT (started_at is stored UTC).
    const { fromUtc: rangeStart, toUtc: rangeEnd } = dayBoundsUtc(selectedDate);
    const load = () => api.getEventsInRange(rangeStart, rangeEnd).then(setEvents).catch(() => {});
    load();
    const today = new Date(); today.setHours(0, 0, 0, 0);
    const isToday = today.getTime() === selectedDate.getTime();
    if (!isToday) return;
    // Instant refresh the moment a motion event opens (backend emits this), plus
    // when the AI finishes analysing one — so the timeline updates immediately
    // instead of waiting for the 30s safety poll.
    const u1 = listen("event:opened",   () => { if (!document.hidden) load(); });
    const u2 = listen("agent:analyzed", () => { if (!document.hidden) load(); });
    const id = setInterval(() => { if (!document.hidden) load(); }, 30_000);
    // Returning after hidden/minimized → one immediate refresh.
    const onVis = () => { if (!document.hidden) load(); };
    document.addEventListener("visibilitychange", onVis);
    return () => {
      clearInterval(id);
      document.removeEventListener("visibilitychange", onVis);
      u1.then(f => f()); u2.then(f => f());
    };
  }, [selectedDate]);

  const enabledCams = configs;
  /** Smallest grid that fits `n` cameras. Shared by the auto-fit effect and the
   *  "+N more" pill, so both agree on what "big enough" means. */
  const fitCols = (n: number) => (n <= 1 ? 1 : n <= 4 ? 2 : n <= 9 ? 3 : 4);
  // Auto-fit the grid to the camera count until the user picks a layout.
  useEffect(() => {
    if (userPickedLayout.current) return;
    setCols(fitCols(enabledCams.length));
  }, [enabledCams.length]);
  /** Cameras the chosen layout cannot show. They used to just disappear: pick
   *  "4", add a fifth camera, and it was invisible with no badge, no overflow
   *  and no page 2 — `userPickedLayout` persists forever, so the auto-fit that
   *  would have rescued it never runs again. */
  const hiddenCams = Math.max(0, enabledCams.length - cols * cols);
  const focusedConfig = focusedCam !== null ? enabledCams.find(c => c.cam_id === focusedCam) : null;

  // v26: keep NVR segment list fresh for the focused camera. Used to
  // detect camera-off gaps and to render the "no video for this period"
  // overlay during history scrub. Refreshes every 30s while focused.
  useEffect(() => {
    if (focusedCam === null) { setFocusedCamSegments([]); return; }
    let alive = true;
    // Scope the segment fetch to the selected LOCAL day so we don't transfer
    // the whole archive (40k+ rows) just to detect gaps for one day.
    const { fromUtc, toUtc } = dayBoundsUtc(selectedDate);
    const load = async () => {
      try {
        const recs = await api.listNvrRecordings(focusedCam, fromUtc, toUtc);
        if (alive) setFocusedCamSegments(recs as Segment[]);
      } catch { /* ignore */ }
    };
    void load();
    const id = window.setInterval(() => { if (!document.hidden) void load(); }, 30_000);
    return () => { alive = false; window.clearInterval(id); };
  }, [focusedCam, selectedDate]);

  // v28: reset the timeline view window whenever the focused camera or the
  // selected day changes. HistoryDrawer resets its own window, but ONLY while
  // the timeline drawer is mounted — with the drawer closed the FocusHeader
  // date-nav would otherwise leave a stale window from a different day. Today
  // frames the last 6h (most recordings visible without zooming); a past day
  // frames the full local day. User zoom/pan persists until the day changes.
  useEffect(() => {
    if (focusedCam === null) return;
    const dayStart = dayStartMs(selectedDate);
    const isToday  = dayStart === dayStartMs(new Date());
    const dayEnd   = isToday ? Date.now() : dayEndMs(selectedDate);
    const span     = isToday ? 6 * 60 * 60 * 1000 : 24 * 60 * 60 * 1000;
    setViewStart(Math.max(dayStart, dayEnd - span));
    setViewEnd(dayEnd);
  }, [focusedCam, selectedDate]);

  // v28: midnight roll-over. When focused on TODAY and watching the live feed
  // (not scrubbing history/events), advance selectedDate to the new day the
  // moment the local clock crosses midnight, so 24/7 live keeps flowing and the
  // new day's segments/events load. Never rolls while a clip is open.
  useEffect(() => {
    if (focusedCam === null) return;
    if (dayStartMs(selectedDate) !== dayStartMs(new Date())) return;
    const id = window.setInterval(() => {
      if (clipSourceRef.current.kind !== "none") return; // don't yank an open scrub
      if (dayStartMs(selectedDate) !== dayStartMs(new Date())) {
        const d = new Date(); d.setHours(0, 0, 0, 0);
        setSelectedDate(d);
      }
    }, 30_000);
    return () => window.clearInterval(id);
  }, [focusedCam, selectedDate]);

  // Reset the live playhead only when the source KIND changes (history →
  // event, event → history, either → none). Within a kind, moving through
  // chunks is normal and the needle should keep tracking. Resetting on every
  // change made it blink off on every ±10s skip.
  const prevKindRef = useRef(clipSource.kind);
  useEffect(() => {
    if (clipSource.kind !== prevKindRef.current) {
      setPlayheadMs(null);
      prevKindRef.current = clipSource.kind;
    }
  }, [clipSource.kind]);

  // v23: keyboard shortcuts while a clip is overlaid on the focus viewport.
  // Mature NVRs set:
  //   Space            play / pause
  //   ←  / →           ±5s
  //   Shift+← Shift+→  ±10s
  //   m                mute toggle
  //   f                fullscreen toggle
  //   Esc              dismiss clip overlay (back to live)
  // Ignored when the user is typing in an input/textarea.
  // v24: bind exactly once when a camera is focused. All live values come
  // from refs so the listener never rebinds during normal playback.
  useEffect(() => {
    if (focusedCam === null) return;
    const onKey = (e: KeyboardEvent) => {
      const cs = clipSourceRef.current;
      if (cs.kind === "none") return;
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;

      switch (e.key) {
        case " ":          e.preventDefault(); overlayRef.current?.togglePlay(); break;
        case "ArrowLeft":  e.preventDefault(); skipRef.current(e.shiftKey ? -10 : -5); break;
        case "ArrowRight": e.preventDefault(); skipRef.current(e.shiftKey ?  10 :  5); break;
        case "m": case "M": overlayRef.current?.toggleMute(); break;
        case "f": case "F": overlayRef.current?.toggleFullscreen(); break;
        case "Escape":
          // v24: fullscreen-first. The browser handles Esc when full-screen
          // is active; a subsequent Esc closes the overlay. Without this
          // guard the overlay closes AND the user falls out of fullscreen
          // at the same time.
          if (document.fullscreenElement) return;
          e.preventDefault();
          setClipSource({ kind: "none" });
          break;
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [focusedCam]);

  return (
    <div className={styles.root}>
      {/* Grid view: no header tile — just floating glass buttons in the top-right
       * corner overlaying the camera grid. Focus mode stays chromeless. */}
      {focusedCam === null && enabledCams.length > 0 && (
        <div className={styles.floatingActions}>
          <div className={styles.layoutPicker}>
            {LAYOUTS.map(l => (
              <button key={l.cols}
                className={`${styles.layoutBtn} ${cols === l.cols ? styles.layoutBtnActive : ""}`}
                onClick={() => pickLayout(l.cols)}
                title={`${l.label}-camera grid`}
              >{l.label}</button>
            ))}
          </div>
          <button className={styles.addCamHeaderBtn} onClick={() => setShowAddModal(true)}>
            <Plus size={13} /> Add Camera
          </button>
        </div>
      )}

      {/* Say so when the chosen layout is hiding cameras, and offer the one
          click that fixes it. Silently dropping them was the old behaviour. */}
      {focusedCam === null && hiddenCams > 0 && (
        <button
          className={styles.hiddenCamsPill}
          onClick={() => pickLayout(fitCols(enabledCams.length))}
          title={`${hiddenCams} camera${hiddenCams === 1 ? "" : "s"} will not fit in this grid`}
        >
          +{hiddenCams} hidden - show all
        </button>
      )}

      {/* ── Grid / onboarding ───────────────────────────────────────────── */}
      {focusedCam === null && (
        !loaded ? (
          /* Loading is not the same as empty. Rendering onboarding here for the
             one frame before the first fetch resolves made every cold start
             flash "No cameras configured" at someone who has cameras. */
          <div className={styles.onboarding}>
            <Loader size={28} className="spin" style={{ color: "var(--text-muted)", opacity: 0.5 }} />
          </div>
        ) : loadError ? (
          /* ...and a FAILED load is not empty either. */
          <div className={styles.onboarding}>
            <AlertTriangle size={40} style={{ color: "var(--danger, #C9605C)", opacity: 0.8, marginBottom: 14 }} />
            <h2 className={styles.onboardingTitle}>Couldn't load your cameras</h2>
            <p className={styles.onboardingDesc}>{loadError}</p>
            <button className={styles.onboardingBtn} onClick={() => { setLoaded(false); loadCams(); }}>
              <RefreshCw size={14} /> Try again
            </button>
          </div>
        ) : enabledCams.length === 0 ? (
          /* Nothing connected yet.
           *
           * This used to be a grey grid glyph, "No cameras configured", and
           * "Click below to add your first camera" — a caption explaining what
           * the button under it plainly does. The three ways in ARE the
           * explanation, so they are the interface: pick one and land on that
           * step, rather than reading a sentence and then choosing anyway. */
          <div className={styles.empty}>
            <div className={styles.emptyStage} aria-hidden="true">
              {[0, 1, 2, 3].map(i => (
                <div key={i} className={styles.emptyTile} style={{ animationDelay: `${i * 90}ms` }} />
              ))}
            </div>
            <h2 className={styles.emptyTitle}>Nothing to watch yet</h2>
            {/* No second line. It was "Click below to add your first camera"
                (instructions for a button), then briefly a feature tagline
                (marketing, aimed at someone who already owns the app). Both
                were filler. The title and the three cards say everything. */}
            <div className={styles.emptyPaths}>
              {([
                { id: "discover", icon: <Wifi size={17} />,    label: "Find on network", sub: "ONVIF discovery" },
                { id: "ip",       icon: <Link size={17} />,    label: "IP camera",       sub: "RTSP or MJPEG URL" },
                { id: "usb",      icon: <Monitor size={17} />, label: "This device",     sub: "Built-in or USB" },
              ] as const).map(o => (
                <button key={o.id} className={styles.emptyPath}
                  onClick={() => { setAddTab(o.id); setShowAddModal(true); }}>
                  <span className={styles.emptyPathIcon}>{o.icon}</span>
                  <span className={styles.emptyPathLabel}>{o.label}</span>
                  <span className={styles.emptyPathSub}>{o.sub}</span>
                </button>
              ))}
            </div>
          </div>
        ) : (
          /* Strict N×N grid — exactly cols² uniform cells */
          <div className={styles.grid}
            style={{
              gridTemplateColumns: `repeat(${cols}, 1fr)`,
              gridTemplateRows:    `repeat(${cols}, 1fr)`,
            }}>
            {(() => {
              const maxSlots = cols * cols;
              const gridCams = enabledCams.slice(0, maxSlots);
              const empty    = Math.max(0, maxSlots - gridCams.length);
              // Cameras past the chosen layout used to just disappear: pick "4",
              // add a fifth camera, and it was invisible with no badge, no
              // overflow and no page 2 — `userPickedLayout` is remembered
              // forever, so auto-fit never runs again to rescue it.
              return (
                <>
                  {gridCams.map(cam => (
                    <CameraTile
                      key={cam.cam_id}
                      onRemove={() => removeCamera(cam.cam_id)}
                      camId={cam.cam_id}
                      name={cam.name || `Camera ${cam.cam_id + 1}`}
                      onDoubleClick={() => {
                        setClipSource({ kind: "none" });
                        setSelectedEvent(null);
                        setFocusedCam(cam.cam_id);
                        setLiveView("camera");
                      }}
                    />
                  ))}
                  {Array.from({ length: empty }).map((_, i) => (
                    <div key={`empty-${i}`} className={styles.emptySlot}
                      onClick={() => setShowAddModal(true)}>
                      <Plus size={16} style={{ color: "var(--text-muted)", opacity: 0.4 }} />
                    </div>
                  ))}
                </>
              );
            })()}
          </div>
        )
      )}

      {/* ── Camera view (drill-down L2): full live feed + floating glass controls ── */}
      {focusedCam !== null && liveView === "camera" && (
        <div className={styles.focusView}>
          <div className={styles.focusCam}>
            {/* Back — top-left floating glass (same `.headerBtn` material as the player header) */}
            <button className={focusStyles.headerBtn}
              style={{ position: "absolute", top: 12, left: 12, zIndex: 10 }}
              onClick={() => { setFocusedCam(null); setLiveView("grid"); }}
              title="Back to all cameras" aria-label="Back to all cameras">
              <ArrowLeft size={13} /> Back
            </button>
            {/* History + ⋮ menu — top-right floating glass (same material as the player header) */}
            <div style={{ position: "absolute", top: 12, right: 12, zIndex: 10, display: "flex", gap: 8 }}>
              <button className={focusStyles.headerBtn}
                onClick={() => { setLiveView("player"); setPanels(p => ({ ...p, timeline: true })); }}
                title="Recordings & timeline" aria-label="History">
                <Clock size={13} /> History
              </button>
              <KebabMenu
                onOpenSettings={() => setSettingsCamId(focusedCam)}
                onRemoveCamera={() => removeCamera(focusedCam)}
              />
            </div>
            <div className={styles.focusCamInner}>
              <CameraView camId={focusedCam} cornered />
            </div>
          </div>
        </div>
      )}

      {/* ── Player (drill-down L3): full timeline playback (clip overlay + history) ── */}
      {focusedCam !== null && liveView === "player" && (() => {
        const camName = focusedConfig?.name ?? `Camera ${focusedCam + 1}`;
        const isLive = activeCams.includes(focusedCam);
        const focusedCamEvents = events.filter(ev => (ev.cam_id ?? 0) === focusedCam);

        // v25: events sorted in chronological order so prev/next-event
        // navigation matches the timeline reading direction.
        const chronoEvents = [...focusedCamEvents].sort((a, b) =>
          new Date(a.started_at).getTime() - new Date(b.started_at).getTime());
        // Where the user IS. The playhead is the only honest answer: the chunk
        // index says which half hour is loaded, not where in it.
        const referenceMs = selectedEvent
          ? new Date(selectedEvent.started_at).getTime()
          : (playheadMs ?? Date.now());
        // "Previous event" = strictly earlier than the current reference.
        // "Next event" = strictly later. Strict so pressing Next/Prev never
        // re-selects the same event.
        const prevEv = [...chronoEvents].reverse().find(ev =>
          new Date(ev.started_at).getTime() < referenceMs - 500) ?? null;
        const nextEv = chronoEvents.find(ev =>
          new Date(ev.started_at).getTime() > referenceMs + 500) ?? null;
        const jumpToEvent = (ev: MotionEvent) => {
          setSelectedEvent(ev);
          setClipSource({ kind: "event", eventId: ev.id });
        };

        const clipUrl = (() => {
          if (!streamInfo) return null;
          // Recorded chunks are the controller's business — one stable URL per
          // half hour. Event clips are plain cached files and stay ours.
          if (clipSource.kind === "history") return historyUrl;
          if (clipSource.kind === "event") {
            return `http://localhost:${streamInfo.port}/footage/${clipSource.eventId}/clip?token=${streamInfo.auth_token}`;
          }
          return null;
        })();

        // Anchor for the wall-clock math inside ClipOverlay, for EVENT CLIPS.
        // The server starts the clip at (first_object_at ?? started_at) −
        // pre_buffer to trim the empty motion-only lead-in, so the anchor must
        // subtract the same pre-buffer for the needle to line up.
        const preBufferMs = (settings?.record_pre_buffer_secs ?? 3) * 1000;
        // The server FRONT-CLAMPS the clip start to the first recorded segment (so the
        // pre-buffer never reaches unrecorded time), and the clip is now re-encoded to
        // begin EXACTLY at that start. Mirror the same clamp here so the timeline
        // needle (anchor + currentTime) lines up precisely with the footage.
        const firstSegMs = focusedCamSegments.length
          ? Math.min(...focusedCamSegments.map(sg => new Date(sg.started_at).getTime()))
          : -Infinity;
        // Event clips only. A plain file has no PROGRAM-DATE-TIME, so the anchor
        // is its only clock; a recorded chunk carries its own in the playlist.
        const clipAnchorMs =
          clipSource.kind === "event" && selectedEvent
            ? Math.max(
                new Date(selectedEvent.first_object_at ?? selectedEvent.started_at).getTime() - preBufferMs,
                firstSegMs,
              )
            : undefined;
        const exitClip = () => setClipSource({ kind: "none" });

        // "Camera was off": the loaded chunk holds no footage at all. The
        // ClipOverlay is rendered only when `!isOffGap`, which is the guard that
        // stops the player from ever streaming another day's video.
        //
        // Seeking, skipping and rolling forward are the controller's job now.

        const handleSkip = (deltaSec: number) => skipRef.current(deltaSec);

        // v21: date-nav + zoom handlers shared by FocusHeader and HistoryDrawer.
        const DAY_MS_ = 24 * 60 * 60 * 1000;
        const todayStart = dayStartMs(new Date());
        const selDayStart = dayStartMs(selectedDate);
        // Today ends at "now"; a past day ends at its real local midnight (DST-safe).
        const selDayEnd   = selDayStart === todayStart ? Date.now() : dayEndMs(selectedDate);
        const viewDurationMs = Math.max(60_000, viewEnd - viewStart);

        const shiftDate = (deltaDays: number) => {
          const d = new Date(selectedDate);
          d.setDate(d.getDate() + deltaDays);
          if (d.getTime() > todayStart) return;
          d.setHours(0, 0, 0, 0);
          setSelectedDate(d);
          exitClip();
        };
        const jumpToNow = () => {
          const d = new Date(); d.setHours(0, 0, 0, 0);
          setSelectedDate(d);
          exitClip();
        };
        const zoomBy = (dir: "in" | "out") => {
          const factor = dir === "in" ? 0.6 : 1.66;
          const pivot = playheadMs ?? (viewStart + viewEnd) / 2;
          const newDur = Math.max(60_000, Math.min(DAY_MS_, viewDurationMs * factor));
          const rel = (pivot - viewStart) / viewDurationMs;
          let ns = pivot - rel * newDur;
          let ne = ns + newDur;
          if (ns < selDayStart) { ns = selDayStart; ne = ns + newDur; }
          if (ne > selDayEnd)   { ne = selDayEnd;   ns = ne - newDur; }
          setViewStart(ns);
          setViewEnd(ne);
        };

        return (
          <div className={styles.focusView}>
            <div className={styles.focusCam}>
              {/* `viewportRange` is declared on FocusHeader and forwarded to
                  ExportDropdown, but was never passed from here — so "Custom
                  range…" always prefilled the last hour instead of the window
                  the user is actually looking at. */}
              <FocusHeader
                camId={focusedCam}
                cameraName={camName}
                isLive={isLive}
                historyMode={clipSource.kind === "history"}
                eventClipMode={clipSource.kind === "event"}
                panels={panels}
                onTogglePanel={togglePanel}
                selectedDate={selectedDate}
                onSelectDate={setSelectedDate}
                onBack={() => setLiveView("camera")}
                onExitClip={exitClip}
                viewDurationMs={viewDurationMs}
                viewportRange={{ start: viewStart, end: viewEnd }}
                onShiftDate={shiftDate}
                onJumpToNow={jumpToNow}
                onZoom={zoomBy}
                selectedEvent={selectedEvent}
                use12h={use12h}
                onToggle12h={toggle12h}
              />
              {/* Ambient color-bleed glow — a downscaled copy of whatever video
                  is on screen (live OR event clip), CSS-blurred so the scene's
                  colors refract into the surrounding padding (Apple-TV backlight).
                  Fed by the rAF sampler that reads the <video> inside the inner. */}
              <canvas ref={ambientCanvasRef} className={styles.ambientGlow}
                width={64} height={36} aria-hidden />
              {/* Inner card clips the media to the rounded edge; the glow above
                  sits behind it and escapes the rounding. */}
              <div ref={focusInnerRef} className={styles.focusCamInner}>
              <CameraView camId={focusedCam} cornered />
              {/* v26: explicit "camera was off" overlay for history scrubs
                that land in a recording gap. Without this the player would
                either freeze on the previous frame or silently jump to
                whatever recording exists nearby — both are confusing. */}
              {isOffGap && (
                <div className={styles.cameraOffOverlay}>
                  <div className={styles.cameraOffBox}>
                    <strong>Camera was off</strong>
                    <span>No recording for this time. Skip forward / back or click another point on the timeline.</span>
                  </div>
                </div>
              )}
              {clipUrl && !isOffGap && (
                <ClipOverlay
                  ref={overlayRef}
                  src={clipUrl}
                  anchorMs={trueAnchorMs ?? clipAnchorMs}
                  onWallClock={setPlayheadMs}
                  onSourceReady={onSourceReady}
                  onSkip={handleSkip}
                  onPrevEvent={prevEv ? () => jumpToEvent(prevEv) : undefined}
                  onNextEvent={nextEv ? () => jumpToEvent(nextEv) : undefined}
                  onEnded={() => {
                    // An event clip that runs out rolls into the continuous
                    // recording at its end, so "play this event" flows into
                    // "keep watching" without a click. Everything else is the
                    // controller's one chunk-advance rule.
                    if (clipSource.kind === "event" && selectedEvent) {
                      const endIso = selectedEvent.ended_at
                        ?? new Date(
                          new Date(selectedEvent.started_at).getTime()
                            + (selectedEvent.duration_secs ?? 10) * 1000
                        ).toISOString();
                      seekTo(new Date(endIso).getTime());
                      return;
                    }
                    handleEnded();
                  }}
                />
              )}
              </div>{/* end focusCamInner */}
            </div>
            <BottomPanel
              kind={PANEL_OPEN(panels) ? "browse" : null}
              compact
            >
              {PANEL_OPEN(panels) && (
                <BrowseDrawer
                  camId={focusedCam}
                  panels={panels}
                  selectedDate={selectedDate}
                  selectedEventId={selectedEvent?.id ?? null}
                  positionMs={clipSource.kind === "none" ? null : (playheadMs ?? trueAnchorMs ?? clipAnchorMs ?? null)}
                  events={focusedCamEvents}
                  onSelectEvent={(ev) => {
                    setSelectedEvent(ev);
                    setClipSource({ kind: "event", eventId: ev.id });
                  }}
                  onSeek={seekTo}
                  viewStart={viewStart}
                  viewEnd={viewEnd}
                  onSetView={(s, e) => { setViewStart(s); setViewEnd(e); }}
                  use12h={use12h}
                />
              )}
            </BottomPanel>
          </div>
        );
      })()}

      {/* ── Floating per-camera settings window ──────────────────────────── */}
      {settingsCamId !== null && (
        <CameraSettingsModal
          camId={settingsCamId}
          onClose={() => setSettingsCamId(null)}
          onRemoveCamera={() => removeCamera(settingsCamId)}
        />
      )}

      {/* ── Add Camera modal ────────────────────────────────────────────── */}
      {showAddModal && (() => {
        const nextSlot = Array.from({ length: 16 }, (_, i) => i)
          .find(i => !configs.some(c => c.cam_id === i)) ?? 0;
        return (
          <AddCameraModal
            nextSlotId={nextSlot}
            initialTab={addTab}
            onClose={() => { setShowAddModal(false); setAddTab(undefined); }}
            onAdded={(newCfg) => {
              // Instant optimistic update — don't wait for DB round-trip
              setConfigs(prev => {
                const without = prev.filter(c => c.cam_id !== newCfg.cam_id);
                return [...without, newCfg].sort((a, b) => a.cam_id - b.cam_id);
              });
              setShowAddModal(false);
              // Also reload from DB in background to stay in sync
              loadCams();
            }}
          />
        );
      })()}
    </div>
  );
}
