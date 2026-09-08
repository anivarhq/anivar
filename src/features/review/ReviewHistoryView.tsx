// v28: standard "History" view for the Review section — ONLY the recording
// player + a scrubbable timeline (no live feed, no camera grid, no settings).
// Reached via the "History" button on the Review player; Back returns to the
// feed. It reuses the SAME leaf components as live focus mode — ClipOverlay (the
// polished player) and HistoryDrawer (the timeline) — plus the shared coverage
// band helpers, so playback/seeking behave identically without touching
// LivePanel's focus mode.

import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import { ArrowLeft, Film, Clock, Bookmark as BookmarkIcon, Download } from "lucide-react";

import { useStore } from "../../store";
import { useShallow } from "zustand/react/shallow";
import { api, type EventMarker, type ReviewSegment } from "../../api";
import type { MotionEvent, TimelineEntry } from "../../types";
import { localDateStr, dayBoundsUtc, isLocalToday, isLocalYesterday } from "../../lib/time";
import { exportSegment, saveIncidentReport, safeName } from "../live/cameraExport";

/** Widen a lightweight timeline marker into a MotionEvent (null heavy fields)
 *  so the timeline + clip-anchor logic stay unchanged. */
function markerToEvent(m: EventMarker): MotionEvent {
  return {
    id: m.id, started_at: m.started_at, ended_at: m.ended_at,
    duration_secs: m.duration_secs, first_object_at: m.first_object_at,
    cam_id: m.cam_id, event_category: (m.event_category ?? null) as MotionEvent["event_category"],
    peak_score: m.peak_score,
    // '@clip' = a standalone cached clip exists — such an event stays playable
    // even after its raw NVR footage prunes (keep_event_clips), so the
    // no-footage dimming must skip it. The real path stays server-side.
    clip_path: m.has_clip ? "@clip" : null,
    // '@thumb' = URL-served preview (the fix for the player's Events panel
    // rendering every card blank: markers used to hardcode thumbnail: null).
    thumbnail: m.has_thumb ? "@thumb" : null,
    detections: null, ai_summary: null,
    dominant_label: null, sub_label: null, recognized_plate: null, top_speed_kmh: null,
  };
}
import { ClipOverlay } from "../live/ClipOverlay";
import { HistoryDrawer } from "../live/HistoryDrawer";
import { EventsListPanel } from "../live/EventsListPanel";
import { rangeInCoverage, type Segment } from "../nvr/NVRPanel";

import styles from "./ReviewHistoryView.module.css";
import { fetchClipStartMs } from "../../lib/clipStart";
import { usePlayback } from "../live/usePlayback";

interface Props {
  camId: number;
  /** Selected day as a "YYYY-MM-DD" string (stable — avoids re-loading every render). */
  dayStr: string;
  initialEventId: string | null;
  use12h: boolean;
  onBack: () => void;
}

const pad = (n: number) => String(n).padStart(2, "0");

// ── Event lifecycle ("what happened") — mature NVRs Timeline parity ─────────────
const LC_ICON: Record<string, string> = {
  appeared: "👁️", entered_zone: "📍", recognized: "👤", lpr: "🚗", attribute: "🏷️",
  speed: "🏎️", crossing: "🚷", audio: "🔊", fall: "⚠️", gone: "🚪",
};
const LC_LABEL: Record<string, string> = {
  appeared: "Appeared", entered_zone: "Entered", recognized: "Recognized", lpr: "Plate",
  attribute: "Object", speed: "Speed", crossing: "Crossed", audio: "Audio", fall: "Fall", gone: "Left",
};

/** Compact horizontal "what happened" strip for the selected event — fetches the
 *  ordered lifecycle (appeared → entered zone → recognized → … → left) from the
 *  backend. Renders nothing until entries arrive, so old events stay clean. */
function EventLifecycle({ eventId, use12h }: { eventId: string; use12h: boolean }) {
  const [entries, setEntries] = useState<TimelineEntry[]>([]);
  useEffect(() => {
    let alive = true;
    setEntries([]);
    api.getEventTimeline(eventId).then(e => { if (alive) setEntries(e); }).catch(() => {});
    return () => { alive = false; };
  }, [eventId]);
  if (entries.length === 0) return null;
  const fmt = (ts: string) => {
    const d = new Date(ts);
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: use12h });
  };
  return (
    <div style={{
      display: "flex", gap: 8, padding: "8px 12px", overflowX: "auto", alignItems: "center",
      background: "rgb(var(--ink) / 0.03)", borderTop: "1px solid rgb(var(--ink) / 0.06)",
    }}>
      <span style={{ fontSize: 10, letterSpacing: 0.6, opacity: 0.55, fontWeight: 700, whiteSpace: "nowrap" }}>
        WHAT HAPPENED
      </span>
      {entries.map((e, i) => (
        <div key={i} title={fmt(e.ts)} style={{
          display: "flex", alignItems: "center", gap: 5, padding: "4px 9px", borderRadius: 999,
          background: "rgb(var(--ink) / 0.06)", whiteSpace: "nowrap", fontSize: 12,
        }}>
          <span>{LC_ICON[e.class_type] ?? "•"}</span>
          <span style={{ opacity: 0.7 }}>{LC_LABEL[e.class_type] ?? e.class_type}</span>
          {e.value && <span style={{ fontWeight: 600 }}>{e.value}</span>}
          {e.score != null && (
            <span style={{ opacity: 0.55, fontVariantNumeric: "tabular-nums" }}>{Math.round(e.score * 100)}%</span>
          )}
        </div>
      ))}
    </div>
  );
}

// ── Inline styles for the player toolbar buttons (no CSS-module additions) ──
const HEADER_BTN: CSSProperties = {
  display: "inline-flex", alignItems: "center", gap: 5,
  padding: "5px 10px", borderRadius: 8, cursor: "pointer",
  fontSize: 12, fontWeight: 600, color: "var(--text-primary)",
  background: "rgb(var(--ink) / 0.06)", border: "1px solid rgb(var(--ink) / 0.12)",
};
const HEADER_BTN_ACTIVE: CSSProperties = {
  background: "color-mix(in srgb, var(--status-idle) 18%, transparent)", borderColor: "color-mix(in srgb, var(--status-idle) 50%, transparent)", color: "var(--status-idle)",
};

export function ReviewHistoryView({ camId, dayStr, initialEventId, use12h, onBack }: Props) {
  const { streamInfo, settings, showToast } = useStore(useShallow(s => ({ streamInfo: s.streamInfo, settings: s.settings, showToast: s.showToast })));

  const day = useMemo(() => new Date(`${dayStr}T12:00:00`), [dayStr]);
  const ymdKey = useMemo(() => localDateStr(day), [day]);
  const dayBounds = useMemo(() => dayBoundsUtc(ymdKey), [ymdKey]);

  const [events, setEvents]     = useState<MotionEvent[]>([]);
  const [segments, setSegments] = useState<Segment[]>([]);
  // Distinguishes "recordings not fetched yet" from "fetched, none exist" — the
  // latter (a day entirely before recording started) must flag EVERY event as
  // no-footage, so we can't infer it from an empty `segments` array alone.
  const [segmentsLoaded, setSegmentsLoaded] = useState(false);
  const [selectedEvent, setSelectedEvent] = useState<MotionEvent | null>(null);
  // Seek / scrub / skip / ended / coverage all live in the shared controller now,
  // so this surface and Live focus mode cannot drift apart again.
  const {
    clipSource, setClipSource, playheadMs, setPlayheadMs,
    coverageBands, isOffGap, historyUrl, overlayRef,
    seekTo, skip, onSourceReady, handleEnded: rollForward,
  } = usePlayback({ camId, segments, streamInfo, dayFromUtc: dayBounds.fromUtc, dayToUtc: dayBounds.toUtc });
  // Server-side review items → severity bands on the timeline.
  const [reviewSegs, setReviewSegs] = useState<ReviewSegment[]>([]);
  // Saved-event ids — drives the Bookmark toggle's active state for the loaded event.
  const [bookmarkedIds, setBookmarkedIds] = useState<Set<string>>(new Set());
  // Events / Timeline toggles — same pair of header buttons as the camera
  // focus player, so every player drives the same way.
  const [panels, setPanels] = useState({ events: false, timeline: true });
  const loadBookmarkIds = useCallback(() => {
    api.listBookmarkIds().then(ids => setBookmarkedIds(new Set(ids))).catch(() => {});
  }, []);

  const ymd = ymdKey;
  const seededRef = useRef(false);
  useEffect(() => {
    if (seededRef.current || !initialEventId) return;
    seededRef.current = true;
    setClipSource({ kind: "event", eventId: initialEventId });
  }, [initialEventId, setClipSource]);

  // ── Load this camera's events + segments for the day ──────────────────────
  useEffect(() => {
    let alive = true;
    // TZ-CORRECT day bounds: the DB stores started_at in UTC, so the LOCAL day must
    // be converted to UTC ISO before querying. Passing the naive `${ymd}T00:00:00`
    // local string made string-comparison DROP every event whose UTC date rolled over
    // (anything after ~19:00 local at UTC-5) → the timeline showed footage but NO event
    // markers / severity bands for "new" events. `.toISOString()` fixes all three.
    const { fromUtc, toUtc } = dayBoundsUtc(ymd);
    // COMPLETE event list for the timeline (markers — no thumbnail, no cap), so
    // a busy day never silently drops events from the timeline.
    api.getEventMarkers(fromUtc, toUtc)
      .then(ms => { if (alive) setEvents(ms.filter(m => (m.cam_id ?? 0) === camId).map(markerToEvent)); })
      .catch(() => {});
    setSegmentsLoaded(false);
    api.listNvrRecordings(camId, fromUtc, toUtc)
      .then(recs => { if (alive) { setSegments(recs as Segment[]); setSegmentsLoaded(true); } })
      .catch(() => { if (alive) setSegmentsLoaded(true); }); // failed fetch = treat as known-empty, don't hang undimmed
    // Server-side review items → severity bands on the timeline.
    api.getReviewSegments(fromUtc, toUtc)
      .then(ss => { if (alive) setReviewSegs(ss.filter(s => s.cam_id === camId)); })
      .catch(() => {});
    loadBookmarkIds();
    return () => { alive = false; };
  }, [camId, day, ymd, loadBookmarkIds]);

  // Resolve the selectedEvent object once events arrive (for prev/next + anchor).
  useEffect(() => {
    if (clipSource.kind === "event") {
      const ev = events.find(e => e.id === clipSource.eventId) ?? null;
      setSelectedEvent(ev);
    } else {
      setSelectedEvent(null);
    }
  }, [clipSource, events]);

  const clipUrl = useMemo(() => {
    if (!streamInfo) return null;
    // Recorded windows are HLS VOD, built by the controller. Cached event clips
    // are plain files and stay this surface's business.
    if (clipSource.kind === "history") return historyUrl;
    if (clipSource.kind === "event") {
      return `http://localhost:${streamInfo.port}/footage/${clipSource.eventId}/clip?token=${streamInfo.auth_token}`;
    }
    return null;
  }, [streamInfo, clipSource, historyUrl]);

  // Anchor for ClipOverlay's wall-clock math. Mirrors live focus mode: event
  // clips start at (first_object_at ?? started_at) − pre_buffer (the v28 trim),
  // history at the seek timestamp.
  const preBufferMs = (settings?.record_pre_buffer_secs ?? 3) * 1000;
  // Event clips only: a plain file has no PROGRAM-DATE-TIME, so the anchor is the
  // only clock it has. A history chunk carries its own wall clock in the playlist.
  const clipAnchorMs =
    clipSource.kind === "event" && selectedEvent
      ? new Date(selectedEvent.first_object_at ?? selectedEvent.started_at).getTime() - preBufferMs
      : undefined;
  // TRUE playback start from the server (keyframe/coverage snap) — see clipStart.ts.
  const [trueAnchorMs, setTrueAnchorMs] = useState<number | null>(null);
  useEffect(() => {
    setTrueAnchorMs(null);
    if (!streamInfo) return;
    let alive = true;
    const opts = clipSource.kind === "event" && selectedEvent
      ? { eventId: selectedEvent.id }
      : null;
    if (!opts) return;
    fetchClipStartMs(streamInfo, opts).then(ms => { if (alive && ms != null) setTrueAnchorMs(ms); });
    return () => { alive = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [clipSource, selectedEvent?.id, streamInfo, camId]);

  // Events whose window has NO recorded video (recording started after them, or
  // their footage aged out) — so the Events list can dim them + badge "No video"
  // instead of the user clicking through to a "Clip not available" error. Only
  // computed once segments have loaded; an empty band set means "unknown yet",
  // so we don't prematurely flag everything as missing.
  const noFootageIds = useMemo(() => {
    const s = new Set<string>();
    if (!segmentsLoaded) return s; // unknown until recordings fetched — don't prejudge
    // segmentsLoaded && coverageBands empty ⇒ zero footage this day ⇒ flag ALL.
    for (const ev of events) {
      // A standalone cached clip keeps the event playable after its raw
      // footage prunes (keep_event_clips) — never dim those.
      if (ev.clip_path) continue;
      const startMs = new Date(ev.started_at).getTime();
      const endMs = ev.ended_at
        ? new Date(ev.ended_at).getTime()
        : startMs + (ev.duration_secs ?? 10) * 1000;
      if (!rangeInCoverage(coverageBands, startMs, endMs)) s.add(ev.id);
    }
    return s;
  }, [events, coverageBands, segmentsLoaded]);

  // ── chronological prev/next event navigation ──────────────────────────────
  const chrono = useMemo(() =>
    [...events].sort((a, b) => new Date(a.started_at).getTime() - new Date(b.started_at).getTime()),
    [events]);
  // Prev/next must step from where the user IS. The playhead is the only honest
  // answer: the chunk index says which half hour is loaded, not where in it.
  const referenceMs =
    selectedEvent ? new Date(selectedEvent.started_at).getTime()
    : (playheadMs ?? Date.now());
  const prevEv = [...chrono].reverse().find(ev => new Date(ev.started_at).getTime() < referenceMs - 500) ?? null;
  const nextEv = chrono.find(ev => new Date(ev.started_at).getTime() > referenceMs + 500) ?? null;

  const jumpToEvent = (ev: MotionEvent) => {
    setSelectedEvent(ev);
    setClipSource({ kind: "event", eventId: ev.id });
  };

  // Gap-aware timeline seek (mature NVRs rule): snap into the nearest real footage.
  //
  // The loaded HLS playlist is 30 minutes wide, so the instant the user clicks is
  // almost always already IN it. Try seeking inside it first: that is a single
  // `currentTime` assignment against media hls.js already holds — no playlist
  // request, no segment fetch, no ffmpeg spawn, no teardown of the player. Only
  // when the target is outside the loaded window do we pay for a new playlist.
  // Seeking, skipping and rolling forward are the controller's job. What stays
  // here is the one thing this surface knows and Live focus does not: an EVENT
  // clip is a separate cached file with its own bounds.
  const eventBounds = useMemo(() => {
    if (!selectedEvent) return undefined;
    const startMs = new Date(selectedEvent.started_at).getTime();
    return {
      startMs,
      endMs: selectedEvent.ended_at
        ? new Date(selectedEvent.ended_at).getTime()
        : startMs + (selectedEvent.duration_secs ?? 10) * 1000,
    };
  }, [selectedEvent]);

  const handleSkip = (deltaSec: number) => skip(deltaSec, eventBounds);

  const handleEnded = () => {
    // An event clip that runs out rolls into the continuous recording at its end,
    // so "play this event" flows into "keep watching" without a click.
    if (clipSource.kind === "event" && eventBounds) {
      seekTo(eventBounds.endMs);
      return;
    }
    rollForward();
  };

  const dateLabel = useMemo(() => {
    if (isLocalToday(ymd))     return "Today";
    if (isLocalYesterday(ymd)) return "Yesterday";
    return day.toLocaleDateString([], { month: "short", day: "numeric" });
  }, [ymd, day]);

  // ── Timeline overlays: review-item severity bands + bookmark pins ──────────
  const reviewBands = useMemo(
    () => reviewSegs
      // Audio-only segments never paint the timeline (bands are for VIDEO
      // activity; sounds live in the feed cards + Audio tab). Audio-only by
      // EITHER signal: categories==["audio"] (covers legacy rows whose data
      // lost the audio field) or audio set with no non-audio categories.
      .filter(s => {
        const cats = s.categories ?? [];
        const audioOnly = cats.every(c => c === "audio") && (!!s.audio || cats.length > 0);
        return !audioOnly;
      })
      .map(s => ({
      id: s.id,
      start: new Date(s.start_time).getTime(),
      end: s.end_time ? new Date(s.end_time).getTime() : new Date(s.start_time).getTime() + 30_000,
      severity: s.severity,
      // Drives the lane's dimmed state. Dropping it here is why a fully-triaged
      // day looked identical to an untouched one.
      reviewed: s.reviewed,
    })),
    [reviewSegs]);
  // Event-anchored bookmark toggle for the loaded event (no timeline pins/list).
  const currentEventId = selectedEvent?.id ?? (clipSource.kind === "event" ? clipSource.eventId : null);
  const isCurrentBookmarked = !!currentEventId && bookmarkedIds.has(currentEventId);
  const toggleCurrentBookmark = async () => {
    if (!currentEventId) return;
    const has = bookmarkedIds.has(currentEventId);
    setBookmarkedIds(prev => {
      const n = new Set(prev);
      if (has) n.delete(currentEventId); else n.add(currentEventId);
      return n;
    });
    try { if (has) await api.removeBookmark(currentEventId); else await api.addBookmark(currentEventId); }
    catch { /* ignore */ }
  };

  const camLabel = useMemo(() => safeName(`cam${camId + 1}`), [camId]);

  // Export an "incident" — the MP4 window + a text report — for an event or a
  // [start,end] window. Reuses the existing /nvr-export plumbing.
  const exportIncident = async (startMs: number, endMs: number, ev?: MotionEvent | null) => {
    if (!streamInfo) { showToast("Server not ready", "error"); return; }
    await exportSegment(streamInfo, camId, camLabel, startMs, endMs, showToast);
    // Build + offer a text incident report from data we already hold.
    const lines: string[] = [
      `Anivar incident report`,
      `Camera: CAM ${camId + 1}`,
      `Time: ${new Date(startMs).toLocaleString([], { hour12: use12h })} – ${new Date(endMs).toLocaleTimeString([], { hour12: use12h })}`,
    ];
    if (ev) {
      if (ev.event_category) lines.push(`Category: ${ev.event_category}`);
      if (ev.dominant_label) lines.push(`Object: ${ev.dominant_label}`);
      if (ev.sub_label) lines.push(`Identity: ${ev.sub_label}`);
      if (ev.recognized_plate) lines.push(`Plate: ${ev.recognized_plate}`);
      try {
        const tl = await api.getEventTimeline(ev.id);
        if (tl.length) {
          lines.push("", "What happened:");
          tl.forEach(e => lines.push(`  ${new Date(e.ts).toLocaleTimeString([], { hour12: use12h })}  ${e.class_type}${e.value ? ` — ${e.value}` : ""}`));
        }
      } catch { /* ignore */ }
    }
    await saveIncidentReport(lines.join("\n"), `${camLabel}_incident-${pad(new Date(startMs).getHours())}${pad(new Date(startMs).getMinutes())}.txt`, showToast);
  };

  // Export incident for the selected event, or a ±30s window around the playhead.
  const handleExportIncident = () => {
    if (selectedEvent) {
      const s = new Date(selectedEvent.started_at).getTime();
      const e = selectedEvent.ended_at
        ? new Date(selectedEvent.ended_at).getTime()
        : s + (selectedEvent.duration_secs ?? 30) * 1000;
      void exportIncident(s, e, selectedEvent);
    } else {
      const at = playheadMs ?? referenceMs;
      void exportIncident(at - 30_000, at + 30_000, null);
    }
  };

  return (
    <div className={styles.root}>
      <div className={styles.topbar}>
        <button className={styles.backBtn} onClick={onBack} title="Back to Review">
          <ArrowLeft size={15} /> Back
        </button>
        <span className={styles.title}>CAM {camId + 1} · {dateLabel}</span>
        <span className={styles.spacer} />
        <button onClick={() => setPanels(p => ({ ...p, events: !p.events }))}
          title="Show / hide the day's events" style={{ ...HEADER_BTN, ...(panels.events ? HEADER_BTN_ACTIVE : {}) }}>
          <Film size={13} /> Events
        </button>
        <button onClick={() => setPanels(p => ({ ...p, timeline: !p.timeline }))}
          title="Show / hide the wall-clock timeline" style={{ ...HEADER_BTN, ...(panels.timeline ? HEADER_BTN_ACTIVE : {}) }}>
          <Clock size={13} /> Timeline
        </button>
        <button onClick={toggleCurrentBookmark} disabled={!currentEventId}
          title={currentEventId ? (isCurrentBookmarked ? "Remove bookmark" : "Bookmark this event") : "Load an event to bookmark"}
          style={{ ...HEADER_BTN, ...(isCurrentBookmarked ? HEADER_BTN_ACTIVE : {}),
            opacity: currentEventId ? 1 : 0.5, cursor: currentEventId ? "pointer" : "default" }}>
          <BookmarkIcon size={13} fill={isCurrentBookmarked ? "var(--status-idle)" : "none"} />
          {isCurrentBookmarked ? "Saved" : "Bookmark"}
        </button>
        <button onClick={handleExportIncident} title="Export incident (video + report)" style={HEADER_BTN}>
          <Download size={13} /> Export
        </button>
      </div>

      <div className={styles.playerArea} style={{ display: "flex", minHeight: 0 }}>
        <div style={{ flex: 1, minWidth: 0, position: "relative" }}>
          {isOffGap ? (
            <div className={styles.offOverlay}>
              <strong>Camera was off</strong>
              <span>No recording for this time. Click another point on the timeline.</span>
            </div>
          ) : clipUrl ? (
            <ClipOverlay
              ref={overlayRef}
              src={clipUrl}
              anchorMs={trueAnchorMs ?? clipAnchorMs}
              onWallClock={setPlayheadMs}
              onSourceReady={onSourceReady}
              onSkip={handleSkip}
              onPrevEvent={prevEv ? () => jumpToEvent(prevEv) : undefined}
              onNextEvent={nextEv ? () => jumpToEvent(nextEv) : undefined}
              onEnded={handleEnded}
            />
          ) : (
            <div className={styles.placeholder}>
              <Film size={30} />
              <span>Select an event or click the timeline</span>
            </div>
          )}
        </div>
        {panels.events && (
          <div style={{ width: 320, flexShrink: 0, overflow: "hidden",
            display: "flex", flexDirection: "column",
            borderLeft: "1px solid rgb(var(--ink) / 0.08)", background: "var(--bg-base)" }}>
            <EventsListPanel
              camId={camId}
              events={events}
              selectedDate={day}
              selectedEventId={clipSource.kind === "event" ? clipSource.eventId : null}
              onSelectEvent={jumpToEvent}
              noFootageIds={noFootageIds}
            />
          </div>
        )}
      </div>

      {clipSource.kind === "event" && <EventLifecycle eventId={clipSource.eventId} use12h={use12h} />}

      {panels.timeline && (
      <div className={styles.timelineArea}>
        <HistoryDrawer
          camId={camId}
          selectedDate={day}
          positionMs={playheadMs}
          events={events}
          selectedEventId={clipSource.kind === "event" ? clipSource.eventId : undefined}
          onSeek={seekTo}
          onSelectEvent={jumpToEvent}
          use12h={use12h}
          reviewBands={reviewBands}
        />
      </div>
      )}
    </div>
  );
}
