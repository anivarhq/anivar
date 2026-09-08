// v16: Header strip overlaid on the focus-mode viewport. standard:
// back + camera name + LIVE / HISTORY / EVENT pill on the left, action
// toggles (Events / History / Export ▾) on the right.
//
// When a clip is overlaying the live viewport (history scrub or event clip),
// the pill becomes a clickable button — click it to dismiss the clip and
// return to the live feed.

import { useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowLeft, Calendar, ChevronLeft, ChevronRight,
  Clock, Film, MoreVertical, Settings as SettingsIcon, Trash2,
  ZoomIn, ZoomOut,
} from "lucide-react";

import type { MotionEvent } from "../../types";

import { GlassCalendar } from "../../components/ui/GlassCalendar";
import { useRecordedDays } from "../../lib/useRecordedDays";
import { ExportDropdown } from "./ExportDropdown";
import { nearestZoomIdx, zoomLabel, ZOOM_LEVELS } from "../nvr/NVRPanel";

import styles from "./FocusHeader.module.css";

/** v18: which sections of the bottom drawer are currently visible. The user
 *  can show Events alone, Timeline alone, both together, or close it. */
export interface PanelState {
  events: boolean;
  timeline: boolean;
}
export const PANEL_CLOSED: PanelState = { events: false, timeline: false };
export const PANEL_OPEN = (s: PanelState) => s.events || s.timeline;


interface Props {
  camId: number;
  cameraName: string;
  isLive: boolean;
  historyMode: boolean;
  eventClipMode: boolean;
  panels: PanelState;
  onTogglePanel: (which: "events" | "timeline") => void;
  selectedDate: Date;
  onSelectDate: (d: Date) => void;
  onBack: () => void;
  /** Dismiss the history-scrub or event-clip overlay and return to live. */
  onExitClip: () => void;
  /** v21: timeline view-window controls (driven by the date-nav cluster). */
  viewDurationMs: number;
  onShiftDate: (deltaDays: number) => void;
  onJumpToNow: () => void;
  onZoom: (dir: "in" | "out") => void;
  selectedEvent: MotionEvent | null;
  viewportRange?: { start: number; end: number };
  /** 12h/24h time format toggle for the focus timeline + clocks. */
  use12h: boolean;
  onToggle12h: () => void;
}

export function KebabMenu({ onOpenSettings, onRemoveCamera, className }: {
  onOpenSettings: () => void;
  onRemoveCamera: () => void;
  /** Override the trigger button style — e.g. the camera view passes the
   *  floating glass `.addCamHeaderBtn`. Falls back to the in-header `.headerBtn`. */
  className?: string;
}) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
    window.addEventListener("mousedown", onDown);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("mousedown", onDown);
      window.removeEventListener("keydown", onKey);
    };
  }, [open]);
  return (
    <div ref={rootRef} className={styles.kebabRoot}>
      <button
        className={className ?? `${styles.headerBtn} ${open ? styles.headerBtnActive : ""}`}
        onClick={() => setOpen(v => !v)}
        title="More options"
        aria-label="More options"
      >
        <MoreVertical size={14} />
      </button>
      {open && (
        <div className={styles.kebabMenu} role="menu">
          <button className={styles.kebabItem}
            onClick={() => { setOpen(false); onOpenSettings(); }}>
            <SettingsIcon size={13} />
            <span>Camera settings</span>
          </button>
          <div className={styles.kebabSep} />
          <button className={`${styles.kebabItem} ${styles.kebabItemDanger}`}
            onClick={() => {
              setOpen(false);
              const ok = window.confirm("Remove this camera? You can re-add it later in Settings.");
              if (ok) onRemoveCamera();
            }}>
            <Trash2 size={13} />
            <span>Remove camera</span>
          </button>
        </div>
      )}
    </div>
  );
}

/** v21: Apple-style segmented date-nav cluster.
 *  Layout:  [‹]  Today/Jun 1  [›]  •  [Now]  •  [− 24h +]
 *  All glass-tinted, no harsh borders, monospaced numerals. */
function DateNavCluster({
  selectedDate, onSelectDate,
  viewDurationMs, onShiftDate, onJumpToNow, onZoom, recordedDays,
}: {
  selectedDate: Date;
  onSelectDate: (d: Date) => void;
  viewDurationMs: number;
  onShiftDate: (deltaDays: number) => void;
  onJumpToNow: () => void;
  onZoom: (dir: "in" | "out") => void;
  /** Days this camera has footage for. Passed down rather than fetched here:
   *  the set is camera-scoped and only FocusHeader knows which camera. */
  recordedDays?: Set<string>;
}) {
  const today = useMemo(() => {
    const d = new Date(); d.setHours(0, 0, 0, 0); return d;
  }, []);
  const isToday = selectedDate.toDateString() === today.toDateString();
  const yesterday = useMemo(() => {
    const d = new Date(today); d.setDate(d.getDate() - 1); return d;
  }, [today]);
  const isYesterday = selectedDate.toDateString() === yesterday.toDateString();
  const label =
    isToday     ? "Today" :
    isYesterday ? "Yesterday" :
    selectedDate.toLocaleDateString(undefined, { month: "short", day: "numeric" });

  const [calOpen, setCalOpen] = useState(false);
  // Anchor for GlassCalendar's popover — it positions itself directly
  // beneath this button.
  const labelBtnRef = useRef<HTMLButtonElement>(null);

  const dateToIso = (d: Date) =>
    `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
  const parseIso = (s: string) => {
    // Parse at noon to avoid DST edge cases shifting the day.
    const [y, m, day] = s.split("-").map(Number);
    return new Date(y, m - 1, day, 12, 0, 0, 0);
  };
  const todayIso = dateToIso(today);

  const zoomIdx     = nearestZoomIdx(viewDurationMs);
  const canZoomIn   = zoomIdx > 0;
  const canZoomOut  = zoomIdx < ZOOM_LEVELS.length - 1;

  const canShiftFwd = !isToday;

  return (
    <div className={styles.dateNav}>
      {/* Date segment: prev / label / next */}
      <div className={styles.segGroup}>
        <button className={styles.segBtn} onClick={() => onShiftDate(-1)} title="Previous day" aria-label="Previous day">
          <ChevronLeft size={13} />
        </button>
        <div className={styles.segCenter}>
          <button ref={labelBtnRef}
            className={styles.segLabel}
            onClick={() => setCalOpen(o => !o)}
            title="Pick a date">
            <Calendar size={11} />
            <span>{label}</span>
          </button>
          <GlassCalendar
            value={dateToIso(selectedDate)}
            onChange={(iso) => { onSelectDate(parseIso(iso)); setCalOpen(false); }}
            max={todayIso}
            open={calOpen}
            onClose={() => setCalOpen(false)}
            anchorRef={labelBtnRef}
            recordedDays={recordedDays}
          />
        </div>
        <button className={styles.segBtn}
          onClick={() => onShiftDate(1)}
          disabled={!canShiftFwd}
          title="Next day" aria-label="Next day">
          <ChevronRight size={13} />
        </button>
      </div>

      {/* Now — only meaningful when not on today */}
      {!isToday && (
        <button className={styles.nowBtn} onClick={onJumpToNow} title="Jump to live">
          Now
        </button>
      )}

      {/* Zoom segmented control */}
      <div className={styles.segGroup}>
        <button className={styles.segBtn}
          onClick={() => onZoom("in")}
          disabled={!canZoomIn}
          title="Zoom in" aria-label="Zoom in">
          <ZoomIn size={11} />
        </button>
        <span className={styles.zoomLabel}>{zoomLabel(viewDurationMs)}</span>
        <button className={styles.segBtn}
          onClick={() => onZoom("out")}
          disabled={!canZoomOut}
          title="Zoom out" aria-label="Zoom out">
          <ZoomOut size={11} />
        </button>
      </div>
    </div>
  );
}

export function FocusHeader({
  camId, cameraName, isLive, historyMode, eventClipMode,
  panels, onTogglePanel, selectedDate, onSelectDate,
  onBack, onExitClip,
  viewDurationMs, onShiftDate, onJumpToNow, onZoom,
  selectedEvent, viewportRange, use12h, onToggle12h,
}: Props) {
  // Which days this camera actually has footage for — marks the picker so
  // hunting for footage isn't guesswork.
  const recordedDays = useRecordedDays(camId);
  let pillClass: string;
  let pillText: string;
  let pillTitle: string;
  if (eventClipMode)        { pillClass = styles.pillEvent;   pillText = "● EVENT";    pillTitle = "Return to live feed"; }
  else if (historyMode)     { pillClass = styles.pillHistory; pillText = "● TIMELINE"; pillTitle = "Return to live feed"; }
  else if (isLive)          { pillClass = styles.pillLive;    pillText = "";           pillTitle = ""; }
  else                      { pillClass = styles.pillOff;     pillText = "OFFLINE";    pillTitle = ""; }

  const inClip = eventClipMode || historyMode;
  const pillCommon = `${styles.pill} ${pillClass} ${inClip ? styles.pillClickable : ""}`;

  return (
    <div className={styles.root}>
      <div className={styles.left}>
        <button className={styles.backBtn} onClick={onBack} title="Back to all cameras" aria-label="Back to all cameras">
          <ArrowLeft size={14} />
        </button>
        <span className={styles.name}>{cameraName}</span>
        {inClip ? (
          <button className={pillCommon} onClick={onExitClip} title={pillTitle} aria-label={pillTitle}>
            {pillText}
          </button>
        ) : isLive ? (
          /* Live = just a pulsing red dot — no text. */
          <span className={styles.liveDot} title="Live" aria-label="Live" />
        ) : (
          <span className={pillCommon}>{pillText}</span>
        )}
      </div>
      <div className={styles.right}>
        <DateNavCluster
          selectedDate={selectedDate}
          onSelectDate={onSelectDate}
          viewDurationMs={viewDurationMs}
          onShiftDate={onShiftDate}
          onJumpToNow={onJumpToNow}
          onZoom={onZoom}
          recordedDays={recordedDays}
        />
        {/* 12h / 24h time-format toggle — same glass as the date/zoom cluster. */}
        <button className={styles.headerBtn} onClick={onToggle12h}
          title="Toggle 12-hour / 24-hour time">
          {use12h ? "12h" : "24h"}
        </button>
        <button
          className={`${styles.headerBtn} ${panels.events ? styles.headerBtnActive : ""}`}
          onClick={() => onTogglePanel("events")}
          title="Show / hide recent events"
        >
          <Film size={13} />
          <span>Events</span>
        </button>
        <button
          className={`${styles.headerBtn} ${panels.timeline ? styles.headerBtnActive : ""}`}
          onClick={() => onTogglePanel("timeline")}
          title="Show / hide wall-clock timeline"
        >
          <Clock size={13} />
          <span>Timeline</span>
        </button>
        <ExportDropdown
          camId={camId}
          cameraName={cameraName}
          selectedEvent={selectedEvent}
          viewportRange={viewportRange}
        />
      </div>
    </div>
  );
}
