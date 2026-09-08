// v16: Timeline-only History drawer. Nothing else — no transport row, no
// thumbnail rail. Click anywhere on the timeline to seek the main viewport's
// ClipOverlay. Click an event spike to play that event's clip.
//
// "Back to live" lives on the FocusHeader pill, not in this drawer.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { api } from "../../api";
import type { MotionEvent } from "../../types";
import { HorizontalTimeline, type Segment } from "../nvr/NVRPanel";
import { dayStartMs, dayEndMs } from "../../lib/time";

import styles from "./HistoryDrawer.module.css";
import { SEVERITY_FG, tint } from "../../lib/palette";

interface Props {
  camId: number;
  /** Day (00:00–23:59) the timeline covers. Defaults to today if absent. */
  selectedDate?: Date;
  /** Wall-clock anchor of the clip currently overlaid in the main viewport,
   *  or null when the live feed is showing. Drives the needle position. */
  positionMs: number | null;
  events: MotionEvent[];
  /** v29: event opened from Review — auto-center the view on it + mark it on the
   *  timeline. LivePanel focus mode passes nothing → unchanged. */
  selectedEventId?: string;
  onSeek: (ms: number) => void;
  onSelectEvent: (ev: MotionEvent) => void;
  /** v21: controlled view window. If provided, HistoryDrawer ignores its
   *  internal state and reads / writes the parent's via onSetView. */
  viewStart?: number;
  viewEnd?: number;
  onSetView?: (start: number, end: number) => void;
  use12h?: boolean;
  /** Optional review-item severity bands — only the Review history view passes
   *  these; live focus / NVRPanel leave them undefined. */
  reviewBands?: Array<{ id: string; start: number; end: number; severity: "alert" | "detection" }>;
}

/** v19: severity filter — same thresholds NVRPanel uses on its filter bar.
 *  Alert (>0.5), Detection (0.3-0.5), Motion (<=0.3), All. */
type Severity = "all" | "alert" | "detection" | "motion";
function passes(ev: MotionEvent, sev: Severity): boolean {
  const s = ev.peak_score ?? 0;
  if (sev === "all")       return true;
  if (sev === "alert")     return s > 0.5;
  if (sev === "detection") return s > 0.3 && s <= 0.5;
  return s <= 0.3;
}

const DAY_MS = 24 * 60 * 60 * 1000;

export function HistoryDrawer({
  camId, selectedDate, positionMs, events, selectedEventId,
  onSeek, onSelectEvent,
  viewStart: viewStartProp, viewEnd: viewEndProp, onSetView, use12h = false,
  reviewBands,
}: Props) {
  const [segments, setSegments] = useState<Segment[]>([]);
  const [severity, setSeverity] = useState<Severity>("all");
  const filteredEvents = useMemo(
    () => severity === "all" ? events : events.filter(ev => passes(ev, severity)),
    [events, severity]);

  const controlled = viewStartProp !== undefined && viewEndProp !== undefined && !!onSetView;

  // Day window comes from the user-selected date. If we're on "today",
  // dayEnd ticks with the wall clock; if on a past day, dayEnd is fixed
  // at 23:59:59.999 so the timeline shows the full day.
  const dayStart = useMemo(() => dayStartMs(selectedDate ?? new Date()), [selectedDate]);

  const isToday = useMemo(() => {
    const t = new Date(); t.setHours(0, 0, 0, 0);
    return t.getTime() === dayStart;
  }, [dayStart]);

  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!isToday) return;
    const id = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, [isToday]);

  // Past day ends at the next LOCAL midnight − 1ms (DST-safe: a transition day
  // is really 23h/25h, so `dayStart + 24h` would be off by an hour).
  const localDayEnd = useMemo(() => dayEndMs(selectedDate ?? new Date()), [selectedDate]);
  const dayEnd = isToday ? now : localDayEnd;

  const [viewStartLocal, setViewStartLocal] = useState(() => dayStart);
  const [viewEndLocal,   setViewEndLocal]   = useState(() => dayEnd);

  const viewStart = controlled ? viewStartProp! : viewStartLocal;
  const viewEnd   = controlled ? viewEndProp!   : viewEndLocal;
  const setView = useCallback((s: number, e: number) => {
    if (controlled) onSetView!(s, e);
    else { setViewStartLocal(s); setViewEndLocal(e); }
  }, [controlled, onSetView]);

  // Reset the view window when the date changes so we're not stuck on a stale
  // range from a different day. Past days show the full 24h; today defaults
  // to the last 6 hours so most recordings are visible without zooming out.
  useEffect(() => {
    const span = isToday ? 6 * 60 * 60 * 1000 : 24 * 60 * 60 * 1000;
    setView(Math.max(dayStart, dayEnd - span), dayEnd);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dayStart]);

  // v29: when an event is opened (from Review), center the timeline on it so the
  // marker is on-screen the moment the view opens — and again on prev/next jump.
  const selectedEventMs = useMemo(() => {
    if (!selectedEventId) return null;
    const ev = events.find(e => e.id === selectedEventId);
    return ev ? new Date(ev.started_at).getTime() : null;
  }, [selectedEventId, events]);
  useEffect(() => {
    if (selectedEventMs === null) return;
    const half = 10 * 60 * 1000; // ~20-min window centered on the event
    let ns = selectedEventMs - half;
    let ne = selectedEventMs + half;
    if (ns < dayStart) { ns = dayStart; ne = Math.min(dayEnd, ns + half * 2); }
    if (ne > dayEnd)   { ne = dayEnd;   ns = Math.max(dayStart, ne - half * 2); }
    setView(ns, ne);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedEventMs]);

  // Keep the viewport glued to "now" on today's date when no clip is playing —
  // but NOT while a selected event is centering the view (avoids a fight/flicker).
  useEffect(() => {
    if (!isToday) return;
    if (positionMs === null && !selectedEventId) {
      const dur = viewEnd - viewStart;
      setView(Math.max(dayStart, now - dur), now);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [now, positionMs, isToday]);

  useEffect(() => {
    let alive = true;
    // Scope to the visible day so we fetch only that day's segments, not the
    // whole archive (40k+ rows). Use the full day end (not the ticking `now`)
    // so the effect doesn't re-run every second on today; the 30s interval
    // keeps today's in-progress segments fresh.
    const fromUtc = new Date(dayStart).toISOString();
    const toUtc   = new Date(localDayEnd).toISOString(); // DST-safe local day end
    const load = async () => {
      try {
        const recs = await api.listNvrRecordings(camId, fromUtc, toUtc);
        if (!alive) return;
        setSegments(recs as Segment[]);
      } catch { /* ignore */ }
    };
    void load();
    const id = window.setInterval(load, 30_000);
    return () => { alive = false; window.clearInterval(id); };
  }, [camId, dayStart, localDayEnd]);

  const playingMsRef = useRef<number | null>(positionMs);
  playingMsRef.current = positionMs;

  const handlePan = useCallback((newStart: number) => {
    const dur = viewEnd - viewStart;
    const clamped = Math.max(dayStart, Math.min(newStart, dayEnd - dur));
    setView(clamped, clamped + dur);
  }, [viewStart, viewEnd, dayStart, dayEnd, setView]);

  const handleZoom = useCallback((dir: "in" | "out", pivot: number) => {
    const dur = viewEnd - viewStart;
    const factor = dir === "in" ? 0.6 : 1.66;
    const newDur = Math.max(60_000, Math.min(DAY_MS, dur * factor));
    const rel = (pivot - viewStart) / dur;
    let ns = pivot - rel * newDur;
    let ne = ns + newDur;
    if (ns < dayStart) { ns = dayStart; ne = ns + newDur; }
    if (ne > dayEnd)   { ne = dayEnd;   ns = ne - newDur; }
    setView(ns, ne);
  }, [viewStart, viewEnd, dayStart, dayEnd, setView]);

  const handleWheelZoom = useCallback((deltaY: number, pivot: number) => {
    const dur = viewEnd - viewStart;
    const factor = deltaY > 0 ? 1.12 : 0.88;
    const newDur = Math.max(60_000, Math.min(DAY_MS, dur * factor));
    const rel = (pivot - viewStart) / dur;
    let ns = pivot - rel * newDur;
    let ne = ns + newDur;
    if (ns < dayStart) { ns = dayStart; ne = ns + newDur; }
    if (ne > dayEnd)   { ne = dayEnd;   ns = ne - newDur; }
    setView(ns, ne);
  }, [viewStart, viewEnd, dayStart, dayEnd, setView]);

  const hasDataForDay = useMemo(() => {
    const segHit = segments.some(s => {
      const t = new Date(s.started_at).getTime();
      return t >= dayStart && t <= dayEnd;
    });
    if (segHit) return true;
    return filteredEvents.some(ev => {
      const t = new Date(ev.started_at).getTime();
      return t >= dayStart && t <= dayEnd;
    });
  }, [segments, filteredEvents, dayStart, dayEnd]);

  const sevOptions: Array<{ kind: Severity; label: string; color?: string }> = [
    { kind: "all",       label: "All" },
    { kind: "alert",     label: "Alert",     color: SEVERITY_FG.alert },
    { kind: "detection", label: "Detection", color: SEVERITY_FG.warn  },
    { kind: "motion",    label: "Motion",    color: SEVERITY_FG.idle  },
  ];

  return (
    <div className={styles.drawer}>
      <div className={styles.filterBar}>
        {sevOptions.map(opt => {
          const active = severity === opt.kind;
          return (
            <button key={opt.kind}
              className={`${styles.filterChip} ${active ? styles.filterChipActive : ""}`}
              onClick={() => setSeverity(opt.kind)}
              style={active && opt.color ? {
                borderColor: tint(opt.color, 60),
                color: opt.color,
              } : undefined}>
              {opt.color && <span className={styles.filterDot} style={{ background: opt.color }} />}
              {opt.label}
            </button>
          );
        })}
        {/* Was rendered INSIDE the track as a static element after eleven
            absolutely-positioned siblings, so it painted over the top-left of
            the timeline. It is a hint; it belongs in the chrome. */}
        <span className={styles.hint}>scroll to zoom · drag to pan · click to seek</span>
      </div>
      <HorizontalTimeline
        segments={segments}
        events={filteredEvents}
        selectedEventId={selectedEventId}
        playingMs={positionMs}
        playingMsRef={playingMsRef as React.RefObject<number | null>}
        dayStart={dayStart}
        dayEnd={dayEnd}
        viewStart={viewStart}
        viewEnd={viewEnd}
        onSeekMs={onSeek}
        onSelectEvent={onSelectEvent}
        onPan={handlePan}
        onZoom={handleZoom}
        onWheelZoom={handleWheelZoom}
        use12h={use12h}
        isLive={positionMs === null}
        reviewBands={reviewBands}
      />
      {!hasDataForDay && (
        <div className={styles.empty}>
          No recordings or events on this date.
        </div>
      )}
    </div>
  );
}
