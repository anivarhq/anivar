/**
 * NVR Panel — Apple-style unified camera view.
 *
 * Single layout:  Live stream → Apple timeline (motion spikes) → Thumbnail strip → Event list.
 * Clicking any event or timeline position switches the video to the recorded clip at that moment.
 * A "LIVE" pill button snaps back to the live feed.
 * No explicit mode toggle — state is implicit (live stream vs recorded clip).
 */
import { eventThumbSrc } from "../../lib/eventThumb";
import { useEffect, useRef, useState, useCallback, useMemo } from "react";
import { listen } from "@tauri-apps/api/event";
import { useStore } from "../../store";
import { useShallow } from "zustand/react/shallow";
import { api, MotionEvent } from "../../api";
import styles from "./NVRPanel.module.css";
import { GlassCalendar } from "../../components/ui/GlassCalendar";
import {
  RotateCcw, Play, Pause, SkipBack, SkipForward,
  Calendar, ChevronRight, Loader, ZoomIn, ZoomOut,
  Download, ChevronLeft, Film, AlertTriangle,
  Volume2, VolumeX,
} from "lucide-react";
import { attachSource, type MediaHandle } from "../../lib/hlsAttach";
import { MAX_RETRIES, retryDelayMs, bustUrl } from "../live/clipRetry";
import { loadSavedVolume, saveVolume } from "../../lib/volume";
import { localDateStr, dayBoundsUtc, dayStartMs, dayEndMs, shiftDay } from "../../lib/time";
import { SEVERITY_FG, severityOfRisk, tint } from "../../lib/palette";
import { CATEGORY_ICON, CATEGORY_CHIP } from "../../lib/palette";

// ── Types ─────────────────────────────────────────────────────────────────────

export interface Segment {
  filename: string;
  cam_id: number;
  started_at: string;
  size_bytes: number;
  duration_secs: number;
}


// ── Helpers ───────────────────────────────────────────────────────────────────

const pad = (n: number) => String(n).padStart(2, "0");
// localDateStr + all day-bounds come from the shared time SSOT (../../lib/time).

// ── Zoom levels ───────────────────────────────────────────────────────────────
// Snap-to levels in milliseconds. Each step is a meaningful time window.
export const ZOOM_LEVELS = [
  5  * 60_000,       // 5 min  — frame-level scrubbing
  15 * 60_000,       // 15 min
  30 * 60_000,       // 30 min
  60 * 60_000,       // 1 h    — default playback view
  2  * 3_600_000,    // 2 h
  4  * 3_600_000,    // 4 h
  8  * 3_600_000,    // 8 h
  12 * 3_600_000,    // 12 h
  24 * 3_600_000,    // 24 h   — full day
] as const;

export function zoomLabel(ms: number): string {
  if (ms <  60_000) return `${Math.round(ms / 60_000)}m`;
  if (ms <  3_600_000) return `${Math.round(ms / 60_000)}m`;
  if (ms === 3_600_000) return "1h";
  return `${Math.round(ms / 3_600_000)}h`;
}

// Given current duration, return the index into ZOOM_LEVELS closest to it.
export function nearestZoomIdx(durMs: number): number {
  let best = 0;
  let bestDiff = Infinity;
  ZOOM_LEVELS.forEach((l, i) => {
    const d = Math.abs(l - durMs);
    if (d < bestDiff) { bestDiff = d; best = i; }
  });
  return best;
}

export function fmtAbsTime(ms: number, use12h = false): string {
  const d = new Date(ms);
  if (use12h) {
    const h = d.getHours(); const ampm = h >= 12 ? "PM" : "AM";
    return `${h % 12 || 12}:${pad(d.getMinutes())}:${pad(d.getSeconds())} ${ampm}`;
  }
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}



// Event-display formatters live in the shared module (Review uses them too);
// re-exported here so existing NVRPanel importers keep working.
export { fmtShortTime, riskColor, riskLabel, aiText, aiTitle } from "../../lib/eventFormat";
import { fmtShortTime, riskColor, riskLabel, aiText, aiTitle } from "../../lib/eventFormat";


// Merge segments into continuous recording bands — the single source of truth
// for "what was actually recorded". Each segment contributes its OWN real end
// (start + measured duration, floored at the 10s nominal stride so a 0/under-
// reported duration still covers the segment). A new segment extends the current
// band only if it starts within GAP_MS of that real end; otherwise it opens a
// new band, leaving a true gap that renders as a "camera off" off-band. This is
// Mature NVRs' approach: gaps come from timestamp deltas, not bitrate estimates.
//
// NOTE: do NOT use the next segment's start as a segment's end — that collapses
// last.end onto the next start every iteration, making the gap check always 0 so
// real mid-day outages are silently swallowed into one continuous band.
export function mergeSegmentBands(segs: Segment[]): Array<{ start: number; end: number; isLive: boolean }> {
  // Tolerate ffmpeg rotation jitter (~10-20s on 10s segments) but render a REAL
  // outage as a gap. 30s == computeOffBands' default minGapMs, so a span is
  // either coverage or an off-band, never both.
  const GAP_MS = 30_000;
  // If the newest segment started within this window of "now" it's the
  // actively-recording tail — extend coverage to the live edge. On a past day
  // the newest segment is hours/days old, so this never fires and the last band
  // ends at the real recorded duration (no phantom coverage).
  const LIVE_TAIL_GRACE = 90_000;
  const now = Date.now();
  const sorted = [...segs]
    .filter(s => !s.filename.startsWith("__event__"))
    .sort((a, b) => new Date(a.started_at).getTime() - new Date(b.started_at).getTime());

  const bands: Array<{ start: number; end: number; isLive: boolean }> = [];
  for (let i = 0; i < sorted.length; i++) {
    const seg  = sorted[i];
    const s    = new Date(seg.started_at).getTime();
    const live = seg.filename === "__live__";
    // Real end of THIS segment from its measured duration.
    const realEnd = s + Math.max(seg.duration_secs ?? 0, 10) * 1000;
    const last = bands[bands.length - 1];
    if (last && s - last.end <= GAP_MS) {
      last.end = Math.max(last.end, realEnd);
      if (live) last.isLive = true;
    } else {
      bands.push({ start: s, end: realEnd, isLive: live });
    }
  }

  // Live tail: extend ONLY the final band to "now" when the newest segment is
  // the in-progress recording, so the live edge has no phantom-free sliver gap.
  const lastSeg = sorted[sorted.length - 1];
  const lastBand = bands[bands.length - 1];
  if (lastSeg && lastBand) {
    const lastStart = new Date(lastSeg.started_at).getTime();
    if (lastSeg.filename === "__live__" || now - lastStart <= LIVE_TAIL_GRACE) {
      lastBand.end = Math.max(lastBand.end, now);
    }
  }
  return bands;
}

// Compute the "camera off / no footage" gaps — the inverse of the recording
// bands, clipped to a [from, to] window. Only windows that have already elapsed
// count as "off" (we never mark the future as off). This is the single source
// of truth for gap rendering AND gap-aware seeking, so the timeline stripes and
// the seek/skip snapping always agree.
export function computeOffBands(
  bands: Array<{ start: number; end: number }>,
  from: number,
  to: number,
  minGapMs = 30_000,
): Array<{ start: number; end: number }> {
  const out: Array<{ start: number; end: number }> = [];
  const horizon = Math.min(to, Date.now());
  if (horizon <= from) return out;
  const sorted = [...bands].sort((a, b) => a.start - b.start);
  let cursor = from;
  for (const b of sorted) {
    const s = Math.max(from, b.start);
    const e = Math.min(horizon, b.end);
    if (s > cursor) out.push({ start: cursor, end: Math.min(s, horizon) });
    cursor = Math.max(cursor, e);
    if (cursor >= horizon) break;
  }
  if (cursor < horizon) out.push({ start: cursor, end: horizon });
  return out.filter(g => g.end - g.start >= minGapMs);
}

// Is `ms` inside a recording band (camera was on)? Mature NVRs use this to decide
// whether a seek target is playable or must snap to the nearest footage.
export function msInCoverage(
  bands: Array<{ start: number; end: number }>,
  ms: number,
): boolean {
  return bands.some(b => ms >= b.start && ms < b.end);
}

// True if ANY recorded footage overlaps the wall-clock window [startMs, endMs]
// — used to tell whether an event has playable video BEFORE the user clicks it
// (older events from before continuous recording started, or events in a
// pruned gap, have none). Overlap = band.start < endMs AND band.end > startMs.
export function rangeInCoverage(
  bands: Array<{ start: number; end: number }>,
  startMs: number,
  endMs: number,
): boolean {
  return bands.some(b => b.start < endMs && b.end > startMs);
}

// Given a seek target inside a gap, return the nearest playable wall-clock ms in
// any recording band — mature NVRs' "snap to nearest footage" rule. `dir` biases
// the search: "fwd" prefers the next band's start, "back" prefers the previous
// band's end, "any" picks whichever edge is closer. Returns null if there are
// no bands at all.
export function snapToCoverage(
  bands: Array<{ start: number; end: number }>,
  ms: number,
  dir: "fwd" | "back" | "any" = "any",
): number | null {
  if (bands.length === 0) return null;
  if (msInCoverage(bands, ms)) return ms;
  const sorted = [...bands].sort((a, b) => a.start - b.start);
  // First band that starts at/after ms (the "next" footage going forward).
  const next = sorted.find(b => b.start >= ms);
  // Last band that ends at/before ms (the "previous" footage going back).
  const prev = [...sorted].reverse().find(b => b.end <= ms);
  // Land 1ms inside the band edge so msInCoverage() agrees with the snap.
  const fwdMs  = next ? next.start : null;
  const backMs = prev ? prev.end - 1 : null;
  if (dir === "fwd")  return fwdMs ?? backMs;
  if (dir === "back") return backMs ?? fwdMs;
  if (fwdMs === null) return backMs;
  if (backMs === null) return fwdMs;
  return Math.abs(fwdMs - ms) <= Math.abs(ms - backMs) ? fwdMs : backMs;
}


// ── standard Recording Map ───────────────────────────────────────────────
//
// ── Horizontal precision timeline ────────────────────────────────────────────
//
// Two-tier design:
//   TOP (overview):  Full 24h compressed view — click anywhere to jump the
//                    detail view to that point. Viewport box shows current window.
//   BOTTOM (detail): Zoomed, scrollable. Hover shows exact time under cursor.
//                    Recording bands + event spikes. Playhead needle. Drag to pan.

export interface HorizontalTimelineProps {
  segments:     Segment[];
  events:       MotionEvent[];
  /** v29: the event the user opened (e.g. from Review) — drawn as a distinct
   *  cyan pin + highlighted spike so it's obvious which event is loaded. */
  selectedEventId?: string;
  playingMs:    number | null;
  playingMsRef?: React.RefObject<number | null>;
  dayStart:     number;
  dayEnd:       number;
  viewStart:    number;
  viewEnd:      number;
  onSeekMs:       (ms: number) => void;          // seek to NVR band — never plays event clips
  onSelectEvent?: (ev: MotionEvent) => void;   // spike click — plays event clip if available
  onPan:          (newStart: number) => void;
  onZoom:       (dir: "in" | "out", pivot: number) => void;
  onWheelZoom:  (deltaY: number, pivot: number) => void;
  use12h:       boolean;
  isLive:       boolean;
  /** Optional: server-side review-item bands (severity-colored) drawn as a thin
   *  strip at the top of the detail tier. Only the Review history view passes
   *  these — NVRPanel / live focus leave them undefined (nothing rendered). */
  reviewBands?: TimelineItem[];
}

/** One thing that happened, reduced to what the timeline lane draws. */
export interface TimelineItem {
  id: string;
  start: number;
  end: number;
  severity: "alert" | "detection";
  reviewed?: boolean;
}

/** A contiguous run of occupied buckets — one drawn block. */
export interface TimelineBlock {
  start: number;
  end: number;
  severity: "alert" | "detection";
  /** True only when EVERY item in the run is reviewed. One unreviewed item in a
   *  block must keep it at full strength, or activity hides inside a dimmed run. */
  reviewed: boolean;
  count: number;
}

/**
 * Quantise items into fixed buckets, then merge adjacent occupied buckets into
 * blocks. This is how mature NVRs draw a review timeline, and the reason is
 * arithmetic: drawn at true width, a 35 s event on a 6 h view of a 1200 px track
 * is 2 px and a 10 s event is half a pixel. Marker width therefore cannot encode
 * duration — the bucket is the unit, so a block is never sub-pixel at any zoom,
 * and dense activity reads as one solid run instead of a picket fence.
 *
 * Severity is the max over the bucket: one alert in a run of detections colours
 * the whole run, because the point of the lane is "look here".
 */
export function eventBlocks(
  items: TimelineItem[],
  viewStart: number,
  viewEnd: number,
  trackPx: number,
  minBlockPx = 6,
): TimelineBlock[] {
  const dur = viewEnd - viewStart;
  if (dur <= 0 || trackPx <= 0) return [];
  const bucketMs = Math.max((minBlockPx / trackPx) * dur, 1000);
  const nBuckets = Math.ceil(dur / bucketMs);

  const sev = new Array<0 | 1 | 2>(nBuckets).fill(0); // 0 empty, 1 detection, 2 alert
  const unreviewed = new Array<boolean>(nBuckets).fill(false);
  const counts = new Array<number>(nBuckets).fill(0);

  for (const it of items) {
    const lo = Math.max(0, Math.floor((it.start - viewStart) / bucketMs));
    const hi = Math.min(nBuckets - 1, Math.floor((Math.max(it.end, it.start) - viewStart) / bucketMs));
    if (hi < 0 || lo > nBuckets - 1) continue;
    const rank = it.severity === "alert" ? 2 : 1;
    for (let b = lo; b <= hi; b++) {
      if (rank > sev[b]) sev[b] = rank;
      if (!it.reviewed) unreviewed[b] = true;
      if (b === lo) counts[b] += 1;
    }
  }

  const blocks: TimelineBlock[] = [];
  let b = 0;
  while (b < nBuckets) {
    if (sev[b] === 0) { b++; continue; }
    const runStart = b;
    let rank: 0 | 1 | 2 = 0;
    let anyUnreviewed = false;
    let count = 0;
    while (b < nBuckets && sev[b] !== 0) {
      if (sev[b] > rank) rank = sev[b];
      if (unreviewed[b]) anyUnreviewed = true;
      count += counts[b];
      b++;
    }
    blocks.push({
      start: viewStart + runStart * bucketMs,
      end: viewStart + b * bucketMs,
      severity: rank === 2 ? "alert" : "detection",
      reviewed: !anyUnreviewed,
      count,
    });
  }
  return blocks;
}

/** Pointer movement (px) above which a gesture is a drag, not a click. Shared by
 *  the click test and the pan so they can never disagree about what happened. */
const DRAG_PX = 4;
/** Click tolerance (px) around an event's start marker. */
const HIT_PX = 6;

/**
 * The event whose START sits within `tolMs` of `ms`, else null.
 *
 * START-anchored on purpose. Hit-testing an event's whole span sounds right and
 * is unusable in practice: real events run three to five minutes and arrive
 * back-to-back, so their spans blanket the visible timeline and EVERY click
 * opened an event — bare-track seeking became unreachable. The track seeks; the
 * spike's start opens. Ties go to the nearest start.
 *
 * `tolMs` must stay pixel-derived. A tolerance proportional to the view duration
 * grows as you zoom out and brings the blanketing straight back.
 */
export function eventAtStart(events: MotionEvent[], ms: number, tolMs: number): MotionEvent | null {
  let best: MotionEvent | null = null;
  let bestDist = Infinity;
  for (const ev of events) {
    const dist = Math.abs(new Date(ev.started_at).getTime() - ms);
    if (dist <= tolMs && dist < bestDist) { best = ev; bestDist = dist; }
  }
  return best;
}

export function HorizontalTimeline({
  segments, events, selectedEventId, playingMs, playingMsRef,
  dayStart, dayEnd, viewStart, viewEnd,
  onSeekMs, onSelectEvent, onPan, onZoom, onWheelZoom, use12h, isLive,
  reviewBands,
}: HorizontalTimelineProps) {
  // Timelines visualize VIDEO events only — audio events cluttered the lane
  // and confused the read (they still live in the Review feed + Audio tab).
  events = useMemo(() => events.filter(e => e.event_category !== "audio"), [events]);
  // Event thumbnails are URL-served ('@thumb' marker) — needs the stream port/token.
  const streamInfo = useStore(s => s.streamInfo);

  // ── Overview tier ──────────────────────────────────────────────────────────
  const ovRef = useRef<HTMLDivElement>(null);
  const dayDur  = dayEnd - dayStart;
  const pctDay  = (ms: number) => Math.max(0, Math.min(100, ((ms - dayStart) / dayDur) * 100));
  const vpLeft  = pctDay(viewStart);
  const vpWidth = Math.max(0.5, pctDay(viewEnd) - vpLeft);
  const bands   = useMemo(() => mergeSegmentBands(segments), [segments]);

  // Real track width — the bucket size is in PIXELS, so it has to be measured.
  const [trackPx, setTrackPx] = useState(1200);
  useEffect(() => {
    const el = detailRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setTrackPx(e.contentRect.width || 1200));
    ro.observe(el);
    setTrackPx(el.getBoundingClientRect().width || 1200);
    return () => ro.disconnect();
  }, []);

  // What the event lane draws. Review items are the canonical grouping (they
  // carry severity and a reviewed flag from the server), so prefer them; the NVR
  // tab and live focus don't have them, and fall back to raw events so no surface
  // loses its indication.
  const laneItems = useMemo<TimelineItem[]>(() => {
    if (reviewBands?.length) return reviewBands;
    return events
      // Sounds are not VIDEO activity; they have their own tab and would
      // otherwise paint the lane red at YAMNet confidence.
      .filter(ev => ev.event_category !== "audio")
      .map(ev => {
        const st = new Date(ev.started_at).getTime();
        return {
          id: ev.id,
          start: st,
          end: ev.ended_at ? new Date(ev.ended_at).getTime()
                           : st + Math.max((ev.duration_secs ?? 0) * 1000, 10_000),
          // Severity, not raw confidence: a person detected at 0.23 is still a
          // person, and height/colour must say so.
          severity: (ev.peak_score ?? 0) > 0.5 || ev.event_category === "person"
            || ev.dominant_label === "person" ? "alert" as const : "detection" as const,
        };
      });
  }, [reviewBands, events]);

  const blocks = useMemo(
    () => eventBlocks(laneItems, viewStart, viewEnd, trackPx),
    [laneItems, viewStart, viewEnd, trackPx]);
  // Semantic zoom: once buckets stop merging distinct items, mark each item's
  // exact start inside its block.
  const showNotches = (viewEnd - viewStart) / trackPx * 6 <= 5000;

  // v26: "camera off" gap bands — the inverse of `bands`, clipped to the
  // day window's PAST half (we don't mark "future" portions of today as
  // off, only times that have already happened with no recording). These
  // sit underneath the green recording bands so the user can see at a
  // glance which slices of the day had no footage.
  const offBands = useMemo(
    () => computeOffBands(bands, dayStart, dayEnd),
    [bands, dayStart, dayEnd],
  );

  const onOverviewClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const r = ovRef.current?.getBoundingClientRect();
    if (!r) return;
    const ms  = dayStart + ((e.clientX - r.left) / r.width) * dayDur;
    // Seek to the time under the cursor. This used to pass `ms - half` — the
    // PAN math (a viewport *start*) pasted into the seek call, which is why the
    // comment below says it doesn't seek. At a 6 h zoom that landed the playhead
    // three hours before the click; at the 24 h default, twelve.
    onSeekMs(ms);
    // Pan the detail viewport to centre on the clicked overview point.
    const range   = viewEnd - viewStart;
    const newStart = Math.max(dayStart, Math.min(dayEnd - range, ms - range / 2));
    onPan(newStart);
  };

  // ── Detail tier ───────────────────────────────────────────────────────────
  const detailRef  = useRef<HTMLDivElement>(null);
  const needleRef     = useRef<HTMLDivElement>(null);
  const needleTimeRef = useRef<HTMLDivElement>(null);
  const [hoverMs, setHoverMs]   = useState<number | null>(null);
  const [hoverX,  setHoverX]    = useState(0);
  const [hovEvId, setHovEvId]   = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const dragStartX = useRef(0);
  const dragStartMs = useRef(0);

  const dur = viewEnd - viewStart;
  const pct = (ms: number) => Math.max(0, Math.min(100, ((ms - viewStart) / dur) * 100));

  // 60fps smooth needle
  useEffect(() => {
    if (!playingMsRef) return;
    let raf: number;
    const tick = () => {
      const ms = playingMsRef.current;
      if (needleRef.current) {
        if (ms !== null && ms >= viewStart && ms <= viewEnd) {
          needleRef.current.style.left    = `${pct(ms)}%`;
          needleRef.current.style.display = "block";
          // v24: keep the time pill on the needle head in sync with the
          // playhead. fmtAbsTime is the same helper the crosshair / event
          // popup time labels use.
          if (needleTimeRef.current) {
            needleTimeRef.current.textContent = fmtAbsTime(ms, use12h);
          }
        } else {
          needleRef.current.style.display = "none";
        }
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playingMsRef, viewStart, viewEnd, dur, use12h]);

  // Adaptive time labels based on zoom level
  const labels = useMemo(() => {
    const durMin = dur / 60_000;
    const interval =
      durMin <=   5 ?       30_000 :   // 30-sec ticks
      durMin <=  15 ?      60_000  :   // 1-min ticks
      durMin <=  60 ?   5*60_000   :   // 5-min ticks
      durMin <= 120 ?  15*60_000   :   // 15-min ticks
      durMin <= 360 ?  30*60_000   :   // 30-min ticks
                       60*60_000;      // 1-hour ticks
    const out: { ms: number; label: string; major: boolean }[] = [];
    const first = Math.ceil(viewStart / interval) * interval;
    for (let ms = first; ms <= viewEnd; ms += interval) {
      const d    = new Date(ms);
      const secs = d.getSeconds();
      const mins = d.getMinutes();
      const major = secs === 0 && mins % (interval >= 3_600_000 ? 60 : interval >= 1_800_000 ? 30 : interval >= 300_000 ? 15 : 5) === 0;
      let label: string;
      if (use12h) {
        const h = d.getHours();
        label = secs > 0
          ? `${h % 12 || 12}:${pad(mins)}:${pad(secs)}`
          : mins > 0 ? `${h % 12 || 12}:${pad(mins)}` : `${h % 12 || 12}${h >= 12 ? "p" : "a"}`;
      } else {
        label = secs > 0
          ? `${pad(d.getHours())}:${pad(mins)}:${pad(secs)}`
          : mins > 0 ? `${pad(d.getHours())}:${pad(mins)}` : `${pad(d.getHours())}:00`;
      }
      out.push({ ms, label, major: major || secs === 0 });
    }
    return out;
  }, [viewStart, viewEnd, dur, use12h]);

  const getMsFromEvent = (e: React.MouseEvent<HTMLDivElement>) => {
    const r = detailRef.current?.getBoundingClientRect();
    if (!r) return null;
    return viewStart + ((e.clientX - r.left) / r.width) * dur;
  };

  const onDetailMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const r = detailRef.current?.getBoundingClientRect();
    if (!r) return;
    const hms = viewStart + ((e.clientX - r.left) / r.width) * dur;
    setHoverX(e.clientX - r.left);
    setHoverMs(hms);
    const tol = Math.min(dur * 0.02, 60_000);
    const close = events.find(ev => Math.abs(new Date(ev.started_at).getTime() - hms) < tol);
    setHovEvId(close?.id ?? null);

    // Same threshold the click uses. Without it, 1-3 px of pointer jitter panned
    // `viewStart` mid-click; `getMsFromEvent` then converted the OLD clientX
    // against the NEW bounds, so the seek landed up to a few minutes off at wide
    // zooms. The click still counts as a click; it must resolve against the same
    // viewport the user aimed at.
    if (dragging && Math.abs(e.clientX - dragStartX.current) > DRAG_PX) {
      // Drag pans the viewport (does NOT seek video — mature NVRs' rule)
      const deltaPx  = dragStartX.current - e.clientX;
      const deltaMs  = (deltaPx / r.width) * dur;
      const newStart = Math.max(dayStart, Math.min(dayEnd - dur, dragStartMs.current + deltaMs));
      onPan(newStart);
    }
  };

  const onDetailClick = (e: React.MouseEvent<HTMLDivElement>) => {
    // Only seek if not a drag
    const moved = Math.abs(e.clientX - dragStartX.current) > DRAG_PX;
    if (moved) return;
    const ms = getMsFromEvent(e);
    if (ms === null) return;
    // The TRACK ALWAYS SEEKS. Only a click on an event's start marker opens it.
    const r = detailRef.current?.getBoundingClientRect();
    const tolMs = r && r.width > 0 ? (HIT_PX / r.width) * dur : 0;
    const hit = onSelectEvent ? eventAtStart(events, ms, tolMs) : null;
    if (hit && onSelectEvent) { onSelectEvent(hit); return; }
    // Seek video to clicked time. The VIEWPORT STAYS FIXED — only playhead moves.
    onSeekMs(ms);
  };

  const onDetailWheel = (e: React.WheelEvent<HTMLDivElement>) => {
    e.preventDefault();
    const r = detailRef.current?.getBoundingClientRect();
    if (!r) return;
    const pivot = viewStart + ((e.clientX - r.left) / r.width) * dur;
    onWheelZoom(e.deltaY, pivot);
  };

  // Drag lifecycle on the DOCUMENT, not the element (the same thing Frigate's
  // ReviewTimeline does). With the listeners on the track, releasing the mouse
  // anywhere else never ended the drag, and `onMouseLeave` was the only escape —
  // which had the opposite failure too, killing a legitimate pan the moment the
  // cursor crossed the 72px track's edge.
  const onMouseDown = (e: React.MouseEvent) => {
    setDragging(true);
    dragStartX.current  = e.clientX;
    dragStartMs.current = viewStart; // record viewStart at drag start
  };

  useEffect(() => {
    if (!dragging) return;
    const end = () => setDragging(false);
    document.addEventListener("mouseup", end);
    document.addEventListener("mouseleave", end);
    window.addEventListener("blur", end);
    return () => {
      document.removeEventListener("mouseup", end);
      document.removeEventListener("mouseleave", end);
      window.removeEventListener("blur", end);
    };
  }, [dragging]);

  // Find hovered event for popup
  const hovEv = hovEvId ? events.find(e => e.id === hovEvId) ?? null : null;

  return (
    <div className={styles.htRoot}>

      {/* ── Overview ─────────────────────────────────────────────────────── */}
      <div ref={ovRef} className={styles.htOverview} onClick={onOverviewClick} title="Click to jump">
        {/* v26: "camera off" gap bands — dark amber stripes where the
            camera produced no footage during a window that has already
            elapsed. Drawn BEFORE the green recording bands so they sit
            visually behind. */}
        {offBands.map((g, i) => (
          <div key={`ovgap-${i}`} className={styles.htOvBandOff}
            style={{ left: `${pctDay(g.start)}%`, width: `${Math.max(0.2, pctDay(g.end) - pctDay(g.start))}%` }}
            title="Camera was off"
          />
        ))}
        {/* Recording bands */}
        {bands.map((b, i) => (
          <div key={i}
            className={b.isLive ? styles.htOvBandLive : styles.htOvBand}
            style={{ left: `${pctDay(b.start)}%`, width: `${Math.max(0.2, pctDay(b.end) - pctDay(b.start))}%` }}
          />
        ))}
        {/* Event dots */}
        {events.map(ev => (
          <div key={ev.id} className={styles.htOvEvent}
            style={{ left: `${pctDay(new Date(ev.started_at).getTime())}%`, background: riskColor(ev.peak_score ?? 0) }}
          />
        ))}
        {/* Viewport box */}
        <div className={styles.htOvViewport} style={{ left: `${vpLeft}%`, width: `${vpWidth}%` }} />
        {/* Playhead in overview */}
        {playingMs !== null && (
          <div className={styles.htOvPlayhead} style={{ left: `${pctDay(playingMs)}%` }} />
        )}
      </div>

      {/* ── Detail ───────────────────────────────────────────────────────── */}
      <div ref={detailRef} className={styles.htDetail}
        style={{ cursor: dragging ? "grabbing" : "crosshair" }}
        onMouseMove={onDetailMouseMove}
        onMouseLeave={() => { setHoverMs(null); setHovEvId(null); }}
        onMouseDown={onMouseDown}
        onClick={onDetailClick}
        onWheel={onDetailWheel}
      >
        {/* Background */}
        <div className={styles.htTrackBg} />

        {/* v26: "camera off" gap stripes — same colour family as the
            overview gap bands. Span the full detail-tier height so the
            user sees the off-window clearly even when zoomed in. */}
        {offBands.map((g, i) => {
          const l = pct(g.start); const w = Math.max(0.1, pct(g.end) - l);
          if (l > 100 || l + w < 0) return null;
          return (
            <div key={`gap-${i}`} className={styles.htDetailOff}
              style={{ left: `${l}%`, width: `${w}%` }}
              title="Camera was off"
            />
          );
        })}

        {/* Recording coverage band (bottom 4px) */}
        {bands.map((b, i) => {
          const l = pct(b.start); const w = Math.max(0.1, pct(b.end) - l);
          if (l > 100 || l + w < 0) return null;
          return (
            <div key={i} className={b.isLive ? styles.htRecLive : styles.htRec}
              style={{ left: `${l}%`, width: `${w}%` }}
            />
          );
        })}

        {/* ── EVENT LANE ────────────────────────────────────────────────
            One indicator, not two. This used to draw raw event spikes AND
            server review bands in the same 54 px, both at z-index 3 — and the
            spikes encoded DETECTOR CONFIDENCE as height, so the real person
            events in this archive (0.23-0.38) rendered as stubs while a
            0.99-confidence sound rendered full height and clipped out of the
            box. Now: bucketed blocks, coloured and sized by SEVERITY. */}
        {blocks.map((b, i) => {
          const l = pct(b.start); const w = Math.max(pct(b.end) - l, 0.15);
          if (l > 100 || l + w < 0) return null;
          const c = b.severity === "alert" ? "var(--status-alert)" : "var(--status-warn)";
          return (
            <div key={`blk-${i}`} className={styles.htBlock}
              title={`${b.count} ${b.count === 1 ? "event" : "events"} · ${b.severity}`}
              style={{
                left: `${l}%`, width: `${w}%`,
                // Pixels, not %: a percentage resolves against the whole 72px
                // track, not the 18px lane.
                height: b.severity === "alert" ? 18 : 11,
                background: tint(c, 85),
                border: `1px solid ${tint(c, 55)}`,
                // Reviewed fades, it does not vanish — you still need to see that
                // something happened there.
                opacity: b.reviewed ? 0.45 : 1,
              }} />
          );
        })}

        {/* Exact item starts, once the zoom can separate them. */}
        {showNotches && laneItems.map(it => {
          const l = pct(it.start);
          if (l < 0 || l > 100) return null;
          return <div key={`nt-${it.id}`} className={styles.htNotch} style={{ left: `${l}%` }} />;
        })}

        {/* v29: selected-event marker — the event opened from Review. A bright
            cyan pin + line, distinct from the white playhead needle, so it's
            obvious which event is loaded even before playback emits a wall-clock. */}
        {selectedEventId && (() => {
          const sel = events.find(e => e.id === selectedEventId);
          if (!sel) return null;
          const x = pct(new Date(sel.started_at).getTime());
          if (x < 0 || x > 100) return null;
          return (
            <div className={styles.htSelMarker} style={{ left: `${x}%` }}>
              <div className={styles.htSelPin} />
              <div className={styles.htSelLine} />
            </div>
          );
        })()}

        {/* Hover crosshair. Its TIME LABEL and the thumbnail popup are rendered
            outside this box — both used to live in here at `bottom: calc(100% +
            N)`, i.e. entirely above a container with `overflow: hidden`, so
            neither had ever been visible. */}
        {hoverMs !== null && <div className={styles.htCrosshair} style={{ left: hoverX }} />}

        {/* Playhead needle — 60fps via rAF. v24: a glass time pill rides
            on the needle head and shows the absolute wall-clock at the
            playhead position. */}
        {playingMsRef ? (
          <div ref={needleRef} className={styles.htNeedle} style={{ display: "none", left: `${pct(playingMs ?? viewStart)}%` }}>
            <div ref={needleTimeRef} className={styles.htNeedleTime} />
            <div className={styles.htNeedleHead} /><div className={styles.htNeedleLine} />
          </div>
        ) : (
          playingMs !== null && playingMs >= viewStart && playingMs <= viewEnd && (
            <div className={styles.htNeedle} style={{ left: `${pct(playingMs)}%` }}>
              <div className={styles.htNeedleTime}>{fmtAbsTime(playingMs, use12h)}</div>
              <div className={styles.htNeedleHead} /><div className={styles.htNeedleLine} />
            </div>
          )
        )}

        {/* Live now-edge */}
        {isLive && <div className={styles.htNowEdge} />}

        {/* Time axis labels */}
        <div className={styles.htAxis}>
          {labels.map(l => (
            <div key={l.ms} className={l.major ? styles.htAxisLabelMajor : styles.htAxisLabel}
              style={{ left: `${pct(l.ms)}%` }}>
              {l.label}
            </div>
          ))}
          {/* Grid tick lines */}
          {labels.filter(l => l.major).map(l => (
            <div key={`g${l.ms}`} className={styles.htGrid} style={{ left: `${pct(l.ms)}%` }} />
          ))}
        </div>

      </div>

    {/* Hover readouts — siblings of the track, so they are NOT clipped. */}
      {hovEv?.thumbnail && (() => {
        const x = pct(new Date(hovEv.started_at).getTime());
        return (
          <div className={styles.htPopup} style={{ left: `${Math.min(80, Math.max(5, x))}%` }}>
            <img src={eventThumbSrc(hovEv.thumbnail, hovEv.id, streamInfo) ?? ""} className={styles.htPopupImg} alt="" />
            <div className={styles.htPopupMeta}>
              <span>{fmtShortTime(new Date(hovEv.started_at).getTime(), use12h)}</span>
              <span style={{ color: riskColor(hovEv.peak_score ?? 0), fontWeight: 700, fontSize: 9 }}>
                {riskLabel(hovEv.peak_score ?? 0)} {((hovEv.peak_score ?? 0) * 100).toFixed(0)}%
              </span>
            </div>
            {aiText(hovEv.ai_summary) && <div className={styles.htPopupSummary}>{aiText(hovEv.ai_summary)}</div>}
          </div>
        );
      })()}

      {hoverMs !== null && (
        <div className={styles.htHoverTime} style={{ left: hoverX }}>
          {fmtAbsTime(hoverMs, use12h)}
        </div>
      )}
    </div>
  );
}

// ── Event thumbnail strip ─────────────────────────────────────────────────────

export function ThumbnailStrip({
  events, activeEvId, onSelect, use12h, layout = "row", noFootageIds,
}: {
  events: MotionEvent[];
  activeEvId: string | null;
  onSelect: (ev: MotionEvent) => void;
  use12h: boolean;
  /** "row" = horizontal scroll strip (NVR tab); "grid" = 3-per-row vertical
   *  grid (the player's Events side panel). */
  layout?: "row" | "grid";
  /** Event ids that have NO recorded video covering them (before continuous
   *  recording started, or in a pruned gap). Rendered dimmed with a "No video"
   *  badge so the user knows they won't play BEFORE clicking. Optional — callers
   *  without coverage data (main NVR tab) pass nothing and nothing is dimmed. */
  noFootageIds?: Set<string>;
}) {
  // Event thumbnails are URL-served ('@thumb' marker) — needs the stream port/token.
  const streamInfo = useStore(s => s.streamInfo);
  const stripRef = useRef<HTMLDivElement>(null);
  const activeRef = useRef<HTMLButtonElement>(null);

  // Auto-scroll to active event
  useEffect(() => {
    if (activeRef.current && stripRef.current) {
      activeRef.current.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "center" });
    }
  }, [activeEvId]);

  if (events.length === 0) return null;

  return (
    <div className={layout === "grid" ? styles.thumbGridWrap : styles.thumbStrip}>
      <div ref={stripRef} className={layout === "grid" ? styles.thumbGrid : styles.thumbRow}>
        {[...events].sort((a, b) => b.started_at.localeCompare(a.started_at)).map(ev => {
          const isActive = ev.id === activeEvId;
          const score = ev.peak_score ?? 0;
          const color = riskColor(score);
          const noFootage = noFootageIds?.has(ev.id) ?? false;
          return (
            <button
              key={ev.id}
              ref={isActive ? activeRef : undefined}
              className={`${styles.thumbCell} ${isActive ? styles.thumbCellActive : ""}`}
              style={isActive ? { borderColor: color, boxShadow: `0 0 0 1px ${tint(color, 25)}` } : {}}
              onClick={() => onSelect(ev)}
              title={noFootage ? "No recorded video for this event — continuous recording started later, or its footage aged out of retention." : undefined}
            >
              {/* Thumbnail — a failed load degrades to the dark card background
                  instead of the browser's broken-image glyph. */}
              <div className={styles.thumbImg} style={noFootage ? { opacity: 0.5 } : undefined}>
                {ev.thumbnail
                  ? <img src={eventThumbSrc(ev.thumbnail, ev.id, streamInfo) ?? ""} alt="" loading="lazy"
                      onError={e => { e.currentTarget.style.display = "none"; }} />
                  : <div className={styles.thumbImgBlank}><Film size={14} /></div>
                }
                {/* Risk pill overlay */}
                <div className={styles.thumbRiskPill} style={{ background: tint(color, 87) }}>
                  {riskLabel(score)}
                </div>
                {/* "No video" badge — this event has no playable footage. */}
                {noFootage && (
                  <div style={{ position: "absolute", left: 4, bottom: 4, zIndex: 2,
                    display: "inline-flex", alignItems: "center", gap: 3,
                    padding: "2px 6px", borderRadius: 6, fontSize: 9, fontWeight: 800,
                    letterSpacing: 0.02, background: "rgba(0,0,0,0.72)", color: "var(--text-primary)",
                    border: "1px solid rgb(var(--ink) / 0.18)" }}>
                    <Film size={9} /> No video
                  </div>
                )}
              </div>
              {/* Time label */}
              <div className={styles.thumbTime}>{fmtShortTime(new Date(ev.started_at).getTime(), use12h)}</div>
              {/* AI summary snippet */}
              {aiText(ev.ai_summary) && (
                <div className={styles.thumbSummary}>{aiText(ev.ai_summary)}</div>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}


