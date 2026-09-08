/**
 * ReviewFeed — standard Review page.
 *
 * mature NVRs model (docs.frigate.video/configuration/review):
 *   • A "review item" is a TIME PERIOD where tracked objects were active on one
 *     camera. Overlapping activity is bundled into ONE item (not per-object),
 *     and items never overlap for a single camera.
 *   • Two severities: ALERTS (person/car, or objects in required zones) and
 *     DETECTIONS (everything else).
 *   • UI: a reverse-chronological feed of preview thumbnails grouped by day;
 *     hover-to-play preview; click → full recording; mark-as-reviewed; filter by
 *     date / camera / object type; Alerts ⇄ Detections toggle.
 *
 * We synthesize review items from our motion_events: group overlapping events
 * per camera into items, classify alert vs detection, and render the feed.
 * Clicking a card opens the full event recording in a player modal.
 */
import { eventThumbSrc } from "../../lib/eventThumb";
import { useEffect, useMemo, useRef, useState, useCallback, type CSSProperties } from "react";
import { listen } from "@tauri-apps/api/event";
import {
  AlertTriangle, Eye, Calendar, RotateCcw, Film, Play, ChevronDown, Check, Search, X, Sparkles,
  UserRound, Car, AudioLines, Bookmark,
} from "lucide-react";

import { useStore } from "../../store";
import { useShallow } from "zustand/react/shallow";
import { api, MotionEvent, ReviewSegment, type StreamInfo } from "../../api";
import { exportEventById, safeName } from "../live/cameraExport";
import { GlassCalendar } from "../../components/ui/GlassCalendar";
import { useRecordedDays } from "../../lib/useRecordedDays";
import {
  fmtShortTime, aiText, aiTitle, riskColor, riskLabel,
} from "../../lib/eventFormat";
import { ReviewHistoryView } from "./ReviewHistoryView";
import { VehiclesView } from "./VehiclesView";
import { AudioView } from "./AudioView";
import { CardActions, CardInfoButton, CardInfoModal, type CardChip } from "./CardChrome";
import { usePanelCache } from "../../lib/panelCache";
import { localDateStr, dayBoundsUtc } from "../../lib/time";
import { tint } from "../../lib/palette";

import styles from "./ReviewFeed.module.css";

// ── Severity model (NVR parity) ───────────────────────────────────────────

type Severity = "alert" | "detection";

interface ReviewItem {
  id: string;            // synthetic id (first member event id)
  camId: number;
  start: number;         // ms
  end: number;           // ms
  severity: Severity;
  thumbnail: string | null;
  thumbId: string | null;   // event id whose thumbnail to fetch when `thumbnail` is the '@thumb' marker
  title: string | null;     // short VLM headline (standard review title)
  labels: string[];      // specific COCO labels present (dog, car, person…)
  categories: string[];  // broad buckets (person/vehicle/animal…) for severity
  subLabel: string | null; // known face name (or plate fallback) — see `plate`
  plate: string | null;    // recognised licence plate, if any
  audioSound: string | null; // detected sound (event_category 'audio'), if any
  fallLabel: string | null;  // 'fall' event (person down), if any
  crossingLabel: string | null; // line-crossing event description, if any
  speedLabel: string | null;    // top zone speed (e.g. "32 km/h"), if any
  zones: string[];
  peak: number;
  summary: string | null;
  memberIds: string[];
  clipEventId: string | null;  // an event with a clip to play
  reviewed?: boolean;          // server-side reviewed flag (segment mode)
}

// localDateStr comes from the shared time SSOT (../../lib/time).

const ALERT_CATEGORIES = new Set(["person", "vehicle"]);

// Group a camera's chronologically-sorted events into non-overlapping review
// items. Events whose [start,end] overlap (or sit within GAP) merge into one.
const GAP_MS = 30_000;

function buildReviewItems(events: MotionEvent[]): ReviewItem[] {
  const byCam = new Map<number, MotionEvent[]>();
  for (const e of events) {
    const c = e.cam_id ?? 0;
    if (!byCam.has(c)) byCam.set(c, []);
    byCam.get(c)!.push(e);
  }

  const items: ReviewItem[] = [];
  for (const [camId, evs] of byCam) {
    const sorted = [...evs].sort(
      (a, b) => new Date(a.started_at).getTime() - new Date(b.started_at).getTime(),
    );
    let cur: MotionEvent[] = [];
    let curEnd = -Infinity;
    const flush = () => {
      if (cur.length === 0) return;
      items.push(makeItem(camId, cur));
      cur = [];
    };
    for (const e of sorted) {
      const s = new Date(e.started_at).getTime();
      const en = e.ended_at
        ? new Date(e.ended_at).getTime()
        : s + Math.max(e.duration_secs ?? 0, 10) * 1000;
      if (cur.length === 0 || s <= curEnd + GAP_MS) {
        cur.push(e);
        curEnd = Math.max(curEnd, en);
      } else {
        flush();
        cur = [e];
        curEnd = en;
      }
    }
    flush();
  }
  // newest first
  return items.sort((a, b) => b.start - a.start);
}

function makeItem(camId: number, members: MotionEvent[]): ReviewItem {
  const start = Math.min(...members.map(m => new Date(m.started_at).getTime()));
  const end = Math.max(...members.map(m =>
    m.ended_at ? new Date(m.ended_at).getTime()
      : new Date(m.started_at).getTime() + Math.max(m.duration_secs ?? 0, 10) * 1000));
  const peak = Math.max(...members.map(m => m.peak_score ?? 0));
  // Specific objects (mature NVRs per-label): prefer dominant_label, fall back to the
  // bucket so older rows still show something.
  const labels = [...new Set(members
    .map(m => (m.dominant_label || m.event_category || "") as string)
    .filter(c => c && c !== "other"))];
  // Broad buckets drive severity (person/vehicle = alert).
  const categories = [...new Set(members
    .map(m => (m.event_category ?? "") as string)
    .filter(c => c && c !== "other"))];
  const subLabel = members.map(m => m.sub_label).find(s => !!s) ?? null;
  const plate = members.map(m => m.recognized_plate).find(p => !!p) ?? null;
  const zones = [...new Set(members.flatMap(m =>
    (m.zones_entered ?? "").split(",").map(z => z.trim()).filter(Boolean)))];
  // Audio events (event_category 'audio') carry the detected sound in dominant_label.
  const audioMember = members.find(m => m.event_category === "audio");
  const audioSound = audioMember ? (audioMember.dominant_label || "Sound") : null;
  const fallLabel = members.some(m => m.event_category === "fall") ? "Person down" : null;
  const crossingLabel = members.find(m => m.event_category === "crossing")?.dominant_label ?? null;
  const topSpeed = Math.max(0, ...members.map(m => m.top_speed_kmh ?? 0));
  const speedLabel = topSpeed >= 1 ? `${Math.round(topSpeed)} km/h` : null;
  // Alert if any member is a person/vehicle, high-risk, entered a zone, audio, fall, or a line crossing.
  // Audio never escalates: a sound is information, not an alert (and its
  // "peak" is YAMNet confidence, excluded from the risk math upstream).
  const isAlert =
    categories.some(l => ALERT_CATEGORIES.has(l)) || peak > 0.5 || zones.length > 0 || !!fallLabel || !!crossingLabel;
  // Best thumbnail = highest-score member that has one.
  const withThumb = [...members].filter(m => m.thumbnail).sort((a, b) => (b.peak_score ?? 0) - (a.peak_score ?? 0));
  const summaryMember = members.find(m => aiText(m.ai_summary));
  const clipMember = members.find(m => m.clip_path) ?? members[0];
  return {
    id: members[0].id,
    camId, start, end,
    severity: isAlert ? "alert" : "detection",
    thumbnail: withThumb[0]?.thumbnail ?? null,
    thumbId: withThumb[0]?.id ?? null,
    labels, categories, subLabel, plate, audioSound, fallLabel, crossingLabel, speedLabel, zones, peak,
    summary: summaryMember ? aiText(summaryMember.ai_summary) : null,
    title: summaryMember ? aiTitle(summaryMember.ai_summary) : null,
    memberIds: members.map(m => m.id),
    // Every event is playable as a virtual NVR slice via GET /footage/:id/clip
    // (footage_clip slices from NVR regardless of clip_path, which is NULL for
    // all but Telegram-exported events since v12). Gating on clip_path hid the
    // Play badge + the card→history player + the "Find similar" button. The
    // player + clip-retry + "Camera was off" overlay handle no-coverage cleanly.
    clipEventId: clipMember?.id ?? members[0].id,
  };
}

// Map a server-side review segment (the canonical grouping) onto the render
// shape used by the feed. The day view reads these instead of grouping events
// client-side, so Review + the NVR timeline + notifications agree on items.
function segmentToItem(s: ReviewSegment): ReviewItem {
  const start = new Date(s.start_time).getTime();
  const end = s.end_time ? new Date(s.end_time).getTime() : start;
  // Normalize AUDIO-ONLY segments at read time: never "alert", never a red
  // risk pill — covers legacy rows stored as alert before audio stopped
  // escalating (the backend migration handles most; this is the belt).
  const cats = s.categories ?? [];
  const audioOnly = cats.every(c => c === "audio") && (!!s.audio || cats.length > 0);
  return {
    id: s.id,
    camId: s.cam_id,
    start, end,
    severity: audioOnly ? "detection" : s.severity,
    thumbnail: s.thumbnail,
    // Segments carry their own thumbnail copy (real base64) — no event lookup needed;
    // clip_event_id doubles as the URL fallback if a segment ever ships a marker.
    thumbId: s.clip_event_id ?? (s.member_ids?.[0] ?? null),
    labels: s.labels ?? [],
    categories: s.categories ?? [],
    subLabel: s.sub_label,
    plate: s.plate,
    audioSound: s.audio,
    fallLabel: s.fall,
    crossingLabel: s.crossing,
    speedLabel: s.speed,
    zones: s.zones ?? [],
    peak: audioOnly ? 0 : s.peak,
    summary: s.summary ? aiText(s.summary) : null,
    title: s.summary ? aiTitle(s.summary) : null,
    memberIds: s.member_ids ?? [],
    clipEventId: s.clip_event_id ?? (s.member_ids?.[0] ?? null),
    reviewed: s.reviewed,
  };
}


// Identity chip on a Review card — shows a recognised face name (👤) or plate (🚗).

// Common tracked object classes (mature NVRs/COCO) — always offered in the Labels
// filter so person/car/dog/cat/etc. are selectable regardless of the day.
const TRACKED_LABELS = [
  "person", "car", "truck", "bus", "motorcycle", "bicycle", "dog", "cat", "bird", "package",
];

// Toolbar dropdown options for the Vehicles / Sounds tabs (server-side
// whitelists live in list_vehicle_events / list_audio_events — keep in sync).
const VEHICLE_TYPES = ["car", "truck", "bus", "motorcycle", "bicycle"];
const VEHICLE_COLORS = ["white", "black", "silver", "gray", "red", "blue", "green", "yellow", "orange", "purple", "brown"];
const PLATE_OPTS = ["with", "known", "unknown"];
const PLATE_LABEL: Record<string, string> = { with: "With plate", known: "Known owner", unknown: "Unknown plate" };
const SOUND_CATS = ["high_pitch", "human", "alarm", "animal", "vehicle", "impact", "music"];
const SOUND_LABEL: Record<string, string> = {
  high_pitch: "High-pitch", human: "Human", alarm: "Alarms", animal: "Animals",
  vehicle: "Vehicles", impact: "Impact", music: "Music",
};
const cap = (v: string) => v.charAt(0).toUpperCase() + v.slice(1);

// Bookmarks-tab Kind filter — saved items span all three browsers now.
const KIND_OPTS = ["motion", "vehicle", "audio"];
const KIND_LABEL: Record<string, string> = { motion: "Events", vehicle: "Vehicles", audio: "Sounds" };
const itemKind = (i: ReviewItem): string =>
  i.categories.includes("vehicle") ? "vehicle" : i.categories.includes("audio") ? "audio" : "motion";

/** Multi-select dropdown (Labels / Zones) — anchored glass button + checkbox
 *  popover. Mirrors the KebabMenu/GlassCalendar pattern (click-outside + Escape). */
function FilterDropdown({ label, options, selected, onToggle, onSelectAll, onClear, format, emptyText }: {
  label: string;
  options: readonly string[];
  selected: Set<string>;
  onToggle: (value: string) => void;
  onSelectAll: () => void;
  onClear: () => void;
  format?: (v: string) => string;
  emptyText?: string;
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
    return () => { window.removeEventListener("mousedown", onDown); window.removeEventListener("keydown", onKey); };
  }, [open]);

  // Always render the BUTTON (even with no options) so the filter is
  // discoverable — the user sees it exists and understands new cameras/zones
  // will appear here. An empty menu shows a muted hint.
  const count = selected.size;
  return (
    <div ref={rootRef} className={styles.ddRoot}>
      <button className={`lg ${styles.filterBtn} ${count > 0 ? styles.filterBtnActive : ""}`}
        onClick={() => setOpen(o => !o)}>
        {label}{count > 0 ? ` · ${count}` : ""}
        <ChevronDown size={13} />
      </button>
      {open && (
        <div className={`glass ${styles.filterMenu}`} role="menu">
          {options.length === 0 ? (
            <div className={styles.filterEmpty}>{emptyText ?? "None yet"}</div>
          ) : (
            <>
              <div className={styles.filterHead}>
                <button className={styles.filterHeadBtn}
                  onClick={onSelectAll}
                  disabled={count === options.length}>Select all</button>
                <button className={styles.filterHeadBtn}
                  onClick={onClear}
                  disabled={count === 0}>Clear</button>
              </div>
              {options.map(opt => {
                const on = selected.has(opt);
                return (
                  <button key={opt} className={styles.filterOpt} onClick={() => onToggle(opt)}>
                    <span className={`${styles.checkbox} ${on ? styles.checkboxOn : ""}`}>
                      {on && <Check size={11} />}
                    </span>
                    <span className={styles.filterOptLabel}>{format ? format(opt) : opt}</span>
                  </button>
                );
              })}
            </>
          )}
        </div>
      )}
    </div>
  );
}

// ── Component ──────────────────────────────────────────────────────────────────

/**
 * The events a card actually stands for.
 *
 * A card is a review ITEM — a group of overlapping events, not one event. On a
 * busy camera a single card could stand for a dozen, and nothing on it said so:
 * that is what "I'm not seeing all the events" meant. The grouping is capped
 * server-side now, but a card can still legitimately hold several, so it has to
 * be able to show them.
 *
 * Reuses `get_events_in_range` over the SEGMENT's own window (a couple of
 * minutes, not the day), so this needs no backend or DTO change.
 */
function EventMembers({ item, use12h, streamInfo, onPick }: {
  item: ReviewItem;
  use12h: boolean;
  streamInfo: StreamInfo | null;
  onPick: (eventId: string) => void;
}) {
  const [members, setMembers] = useState<MotionEvent[] | null>(null);
  const ids = item.memberIds.join(",");
  useEffect(() => {
    let alive = true;
    setMembers(null);
    const idSet = new Set(item.memberIds);
    api.getEventsInRange(new Date(item.start - 1000).toISOString(),
                         new Date(item.end + 1000).toISOString())
      .then(evs => { if (alive) setMembers(
        evs.filter(e => idSet.has(e.id))
           .sort((a, b) => new Date(a.started_at).getTime() - new Date(b.started_at).getTime())); })
      .catch(() => { if (alive) setMembers([]); });
    return () => { alive = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ids, item.start, item.end]);

  if (members === null) return <div style={{ fontSize: 11, color: "var(--text-muted)" }}>Loading events…</div>;
  if (members.length < 2) return null;
  return (
    <div>
      <div style={{ fontSize: 10, letterSpacing: 0.6, fontWeight: 700, opacity: 0.55, marginBottom: 5 }}>
        {members.length} EVENTS IN THIS ITEM
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 3, maxHeight: 190, overflowY: "auto" }}>
        {members.map(m => (
          <button key={m.id} onClick={() => onPick(m.id)} title="Play this event"
            style={{ display: "flex", alignItems: "center", gap: 7, padding: "3px 6px", borderRadius: 6,
              background: "rgb(var(--ink) / 0.05)", border: "1px solid rgb(var(--ink) / 0.08)",
              cursor: "pointer", color: "var(--text-primary)", textAlign: "left" }}>
            {m.thumbnail
              ? <img src={eventThumbSrc(m.thumbnail, m.id, streamInfo) ?? ""} alt="" loading="lazy"
                  style={{ width: 40, height: 23, objectFit: "cover", borderRadius: 3, flexShrink: 0, background: "#000" }}
                  onError={e => { e.currentTarget.style.visibility = "hidden"; }} />
              : <span style={{ width: 40, height: 23, borderRadius: 3, flexShrink: 0,
                  background: "rgb(var(--ink) / 0.08)", display: "inline-flex", alignItems: "center",
                  justifyContent: "center" }}><Film size={10} /></span>}
            <span style={{ fontSize: 11, fontVariantNumeric: "tabular-nums" }}>
              {fmtShortTime(new Date(m.started_at).getTime(), use12h)}
            </span>
            <span style={{ fontSize: 11, opacity: 0.6, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {m.sub_label || m.dominant_label || m.event_category || "motion"}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}

export function ReviewFeed() {
  const { settings, streamInfo, showToast } = useStore(useShallow(s => ({ settings: s.settings, streamInfo: s.streamInfo, showToast: s.showToast })));

  const [selectedDate, setSelectedDate] = useState(() => localDateStr());
  // Day view reads server-side review segments (canonical grouping); search /
  // find-similar modes return raw events grouped client-side (see usingSegments).
  // Stale-while-revalidate: last-known data survives tab switches, so a revisit
  // paints instantly and load() refreshes in the background (no blank flash).
  const [segments, setSegments] = usePanelCache<ReviewSegment[]>("review.segments", []);
  const [events, setEvents] = usePanelCache<MotionEvent[]>("review.events", []);
  const [loading, setLoading] = useState(false);
  // v30: whole-archive search box. A non-empty (debounced) query searches ALL
  // dates via `search_events`; empty falls back to the selected day.
  const [query, setQuery] = useState("");
  const [debouncedQuery, setDebouncedQuery] = useState("");
  const searching = debouncedQuery.length > 0;
  // Segmented control: severity tabs + bookmarks + the Vehicles/Sounds browsers
  // (moved here from the People panel — Review is the event home; they're
  // self-contained all-time views with their own filters and players).
  const [tab, setTab] = useState<Severity | "bookmark" | "vehicles" | "sounds">("alert");
  // Saved/favorites: the set of bookmarked event ids (card active-state + count)
  // and the bookmarked events rendered as cards for the Bookmarks tab.
  const [bookmarkIds, setBookmarkIds] = useState<Set<string>>(new Set());
  const [bookmarkedItems, setBookmarkedItems] = usePanelCache<ReviewItem[]>("review.bookmarkedItems", []);
  // v28: multi-select camera + label + zone filters (empty = all) + hide-reviewed.
  const [camFilter, setCamFilter] = useState<Set<string>>(new Set());
  const [labelFilter, setLabelFilter] = useState<Set<string>>(new Set());
  const [zoneFilter, setZoneFilter] = useState<Set<string>>(new Set());
  // standard person MODE: selecting people (enrolled OR unknown clusters)
  // switches the feed to THEIR events across the last year — like search mode —
  // instead of intersecting against one day (which read as "filter broken" on
  // any day the person wasn't seen, and silently blanked on errors).
  // Tokens: enrolled person id, or "cluster:<id>" for an unknown stranger.
  const [personFilter, setPersonFilter] = useState<Set<string>>(new Set());
  const [personEvents, setPersonEvents] = useState<MotionEvent[] | null>(null); // null = mode off
  const [personCrops, setPersonCrops] = useState<Map<string, string>>(new Map()); // event id → face crop
  const [knownPersons, setKnownPersons] = useState<Array<{ id: string; name: string }>>([]);
  const [unknownClusters, setUnknownClusters] = useState<Array<{ token: string; label: string; faceIds: string[] }>>([]);
  const [hideReviewed, setHideReviewed] = useState(false);
  // Vehicles/Sounds tab filters — same toolbar dropdowns, server-side applied
  // (cameras + date are SHARED with the event tabs; these are tab-specific).
  const [vtypeFilter, setVtypeFilter] = useState<Set<string>>(new Set());
  const [vcolorFilter, setVcolorFilter] = useState<Set<string>>(new Set());
  const [plateFilter, setPlateFilter] = useState<Set<string>>(new Set());
  // Sounds default = All (empty set); high-pitch etc. are opt-in via the dropdown.
  const [soundFilter, setSoundFilter] = useState<Set<string>>(new Set());
  // Bookmarks tab: filter saved items by kind (events / vehicles / sounds).
  const [bookmarkKindFilter, setBookmarkKindFilter] = useState<Set<string>>(new Set());
  // Toolbar refresh button on the Vehicles/Sounds tabs: bump → views reload.
  const [browserTick, setBrowserTick] = useState(0);
  const [use12h] = useState(() => localStorage.getItem("nvr_12h") === "1");
  const [calOpen, setCalOpen] = useState(false);
  // Days with footage, across all cameras (the feed is not cam-scoped).
  const recordedDays = useRecordedDays();
  const calAnchor = useRef<HTMLButtonElement>(null);
  // Configured cameras (so the Cameras filter lists ALL cameras, not just the
  // ones that happened to record an event today).
  const [camConfigs, setCamConfigs] = useState<Array<{ cam_id: number; name: string; enabled: boolean }>>([]);
  useEffect(() => {
    api.getCameraConfigs().then(c => setCamConfigs(c.filter(x => x.enabled))).catch(() => {});
    // Enrolled people for the People filter (id + name only).
    api.listKnownPersons().then(ps => setKnownPersons(ps.map(p => ({ id: p.id, name: p.name })))).catch(() => {});
    // Unknown recurring people (Train-tab clusters) — their events are filterable too.
    api.listUnknownClusters().then(cs => setUnknownClusters(cs.map((c, i) => ({
      token: `cluster:${c.cluster_id}`,
      label: `Stranger ${i + 1} · ${c.count}× · ${c.time_pattern}`,
      faceIds: c.face_ids,
    })))).catch(() => {});
  }, []);

  // Resolve the selection → THEIR events (id-bound; '@thumb' payloads, so this
  // is KBs not MBs). Errors surface as a toast + the filter clears — never a
  // silently blank feed. An EMPTY result keeps mode on (banner + empty state).
  useEffect(() => {
    if (personFilter.size === 0) { setPersonEvents(null); setPersonCrops(new Map()); return; }
    let cancelled = false;
    (async () => {
      try {
        const results = await Promise.all([...personFilter].map(token => {
          const cluster = token.startsWith("cluster:")
            ? unknownClusters.find(c => c.token === token) : null;
          return api.getPersonEvents(cluster
            ? { faceIds: cluster.faceIds, days: 365, limit: 300 }
            : { personId: token, days: 365, limit: 300 });
        }));
        if (cancelled) return;
        const seen = new Set<string>();
        const evs: MotionEvent[] = [];
        const crops = new Map<string, string>();
        for (const pe of results.flat()) {
          if (!seen.has(pe.event.id)) { seen.add(pe.event.id); evs.push(pe.event); }
          if (pe.person_crop && !crops.has(pe.event.id)) crops.set(pe.event.id, pe.person_crop);
        }
        evs.sort((a, b) => b.started_at.localeCompare(a.started_at));
        setPersonEvents(evs);
        setPersonCrops(crops);
      } catch (e) {
        if (cancelled) return;
        showToast(`Couldn't load that person's events: ${String(e)}`, "error");
        setPersonFilter(new Set());
        setPersonEvents(null);
      }
    })();
    return () => { cancelled = true; };
  }, [personFilter, unknownClusters, showToast]);
  // v28: when set, the Review root is replaced by a focus-style player+timeline
  // view that starts playing the event. Clicking a card sets it; Back clears it.
  // Widened past ReviewItem so vehicle/audio cards open the SAME player.
  const [historyItem, setHistoryItem] = useState<{ camId: number; clipEventId: string | null } | null>(null);
  // "Find similar" mode — image→image semantic results for one source event.
  const [similarTo, setSimilarTo] = useState<{ id: string; label: string } | null>(null);
  const [infoItem, setInfoItem] = useState<ReviewItem | null>(null); // (i) details popover
  // Two-step delete inside that popover — a destructive action never fires on a
  // single click, and the arming resets whenever the popover changes item.

  // Day view → server segments; search / find-similar / person mode / bookmark tab → events.
  const usingSegments = !debouncedQuery && !similarTo && tab !== "bookmark" && personEvents === null;

  // Reviewed set — same localStorage key family as NVRPanel for consistency.
  const [reviewed, setReviewed] = useState<Set<string>>(() => {
    try { return new Set(JSON.parse(localStorage.getItem("review_reviewed") ?? "[]")); }
    catch { return new Set(); }
  });
  const markReviewed = useCallback((ids: string[]) => {
    setReviewed(prev => {
      const next = new Set(prev);
      ids.forEach(id => next.add(id));
      if (usingSegments) {
        // Persist reviewed-state server-side (DB column) — no longer localStorage.
        ids.forEach(id => api.setReviewSegmentReviewed(id, true).catch(() => {}));
      } else {
        // Cap at the newest 1000 ids — this set otherwise grows for the life of
        // the install and is JSON.parsed on every mount.
        localStorage.setItem("review_reviewed", JSON.stringify([...next].slice(-1000)));
      }
      return next;
    });
  }, [usingSegments]);

  // Debounce the search box so we don't query on every keystroke.
  useEffect(() => {
    const t = setTimeout(() => setDebouncedQuery(query.trim()), 250);
    return () => clearTimeout(t);
  }, [query]);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      // On the Vehicles/Sounds tabs the search box filters THOSE views
      // (client-side, passed as a prop) — it must not swap the event feed
      // into search mode underneath, or the day counts blank while typing.
      const searchQ = tab === "vehicles" || tab === "sounds" ? "" : debouncedQuery;
      if (similarTo) {
        setSegments([]);
        setEvents(await api.findSimilarEvents(similarTo.id));
      } else if (searchQ) {
        setSegments([]);
        setEvents(await api.searchEvents(searchQ));
      } else {
        // Day view: the canonical server-side review items.
        setEvents([]);
        { const { fromUtc, toUtc } = dayBoundsUtc(selectedDate);
          setSegments(await api.getReviewSegments(fromUtc, toUtc)); }
      }
    } finally { setLoading(false); }
  }, [selectedDate, debouncedQuery, similarTo, tab]);

  // Typing a search exits "Find similar" AND person mode (mutually exclusive views).
  useEffect(() => { if (debouncedQuery) { setSimilarTo(null); setPersonFilter(new Set()); } }, [debouncedQuery]);

  useEffect(() => { load(); }, [load]);
  // Refresh when motion opens or AI finishes — debounced (500 ms trailing): a
  // busy scene fires these per event, and each load() is a full segment query.
  // Skipped while hidden; the visibilitychange refresh below covers the return.
  useEffect(() => {
    let t: ReturnType<typeof setTimeout> | null = null;
    const bump = () => {
      if (document.hidden) return;
      if (t) clearTimeout(t);
      t = setTimeout(() => { t = null; load(); }, 500);
    };
    const u0 = listen("event:opened", bump);
    const u1 = listen("agent:analyzed", bump);
    return () => { if (t) clearTimeout(t); u0.then(f => f()); u1.then(f => f()); };
  }, [load]);
  // Returning after the window was hidden/minimized → one immediate refresh,
  // so the feed the user comes back to is current (deterministic, not polled).
  useEffect(() => {
    const onVis = () => { if (!document.hidden) load(); };
    document.addEventListener("visibilitychange", onVis);
    return () => document.removeEventListener("visibilitychange", onVis);
  }, [load]);

  // ── Bookmarks (saved/favorites) ───────────────────────────────────────────
  const loadBookmarkIds = useCallback(() => {
    api.listBookmarkIds().then(ids => setBookmarkIds(new Set(ids))).catch(() => {});
  }, []);
  const loadBookmarkedItems = useCallback(async () => {
    try {
      const evs = await api.listBookmarkedEvents();
      setBookmarkedItems(evs.map(ev => makeItem(ev.cam_id ?? 0, [ev])));
    } catch { /* ignore */ }
  }, []);
  // Reload on mount + whenever we return from the player (a bookmark may have
  // been toggled there).
  useEffect(() => { loadBookmarkIds(); }, [loadBookmarkIds, historyItem]);
  // Load the saved-event cards whenever the Bookmarks tab is shown.
  useEffect(() => {
    if (tab === "bookmark" && !debouncedQuery && !similarTo) loadBookmarkedItems();
  }, [tab, debouncedQuery, similarTo, loadBookmarkedItems]);

  // Toggle a card's saved state (optimistic) + persist + refresh the tab list.
  const toggleBookmark = useCallback(async (eventId: string) => {
    const has = bookmarkIds.has(eventId);
    setBookmarkIds(prev => {
      const n = new Set(prev);
      if (has) n.delete(eventId); else n.add(eventId);
      return n;
    });
    try { if (has) await api.removeBookmark(eventId); else await api.addBookmark(eventId); }
    catch { /* ignore */ }
    if (tab === "bookmark") loadBookmarkedItems();
  }, [bookmarkIds, tab, loadBookmarkedItems]);

  // Delete a card for good: its clip file, its event rows, its bookmark. A card
  // is a GROUP, so every member id goes in ONE call (deleting them one by one
  // re-aggregates the group in between and the card flickers back). Optimistic
  // everywhere the card can be rendered from, then a reload for server truth.
  const deleteEventIds = useCallback(async (ids: string[], segmentId?: string) => {
    if (ids.length === 0) return;
    const dead = new Set(ids);
    setSegments(prev => prev.filter(s => s.id !== segmentId));
    setEvents(prev => prev.filter(e => !dead.has(e.id)));
    setBookmarkedItems(prev => prev.filter(i => !i.memberIds.some(m => dead.has(m))));
    setPersonEvents(prev => prev === null ? prev : prev.filter(e => !dead.has(e.id)));
    setInfoItem(null);
    try {
      await api.deleteEvents(ids);
      showToast(ids.length > 1 ? `Deleted ${ids.length} events` : "Event deleted", "success");
    } catch (e) {
      showToast(`Couldn't delete: ${String(e)}`, "error");
    }
    // Reload regardless: on failure this puts the card back rather than leaving
    // the feed lying about what's on disk.
    load();
    loadBookmarkIds();
    setBrowserTick(t => t + 1); // Vehicles/Sounds browsers page their own data
    if (tab === "bookmark") loadBookmarkedItems();
  }, [load, loadBookmarkIds, loadBookmarkedItems, tab, showToast, setSegments, setEvents, setBookmarkedItems]);

  const deleteItem = useCallback((item: ReviewItem) => deleteEventIds(
    item.memberIds.length > 0 ? item.memberIds : (item.clipEventId ? [item.clipEventId] : []),
    item.id,
  ), [deleteEventIds]);

  // The day's review items (drives Alerts/Detections counts + filter options).
  const dayItems = useMemo(() => segments.map(segmentToItem), [segments]);
  // What the feed shows: bookmark tab → saved cards; search/similar → grouped
  // events; otherwise the day's review items.
  const displayItems = useMemo(() => {
    if (debouncedQuery || similarTo) return buildReviewItems(events);
    if (personEvents !== null) return buildReviewItems(personEvents); // person mode
    if (tab === "bookmark") return bookmarkedItems;
    return dayItems;
  }, [debouncedQuery, similarTo, personEvents, tab, events, bookmarkedItems, dayItems]);
  // Counts: alert/detection always reflect the day (stable while on the bookmark tab).
  const countItems = tab === "bookmark" ? dayItems : displayItems;
  // Reviewed = optimistic local set OR the segment's server flag.
  const isReviewed = useCallback(
    (i: ReviewItem) => reviewed.has(i.id) || !!i.reviewed,
    [reviewed],
  );

  // All configured cameras (fallback to the cams that recorded events if configs
  // aren't loaded yet) so the Cameras filter always lists every camera.
  const cams = useMemo(() => {
    const ids = camConfigs.length > 0
      ? camConfigs.map(c => c.cam_id)
      : [...new Set(displayItems.map(i => i.camId))];
    return [...new Set(ids)].sort((a, b) => a - b);
  }, [camConfigs, displayItems]);
  const camName = (id: number) =>
    camConfigs.find(c => c.cam_id === id)?.name || `CAM ${id + 1}`;

  // Every filter EXCEPT the severity tab. Shared with the tab counts below so a
  // badge can never disagree with the grid it labels — "Alerts 42" over five
  // visible cards is what an active camera/label filter used to produce.
  const passesFilters = useCallback((i: ReviewItem) =>
    (camFilter.size === 0 || camFilter.has(String(i.camId)))
    && (labelFilter.size === 0 || i.labels.some(l => labelFilter.has(l)))
    && (zoneFilter.size === 0 || i.zones.some(z => zoneFilter.has(z)))
    && (tab !== "bookmark" || bookmarkKindFilter.size === 0 || bookmarkKindFilter.has(itemKind(i)))
    && (!hideReviewed || !isReviewed(i)),
    [camFilter, labelFilter, zoneFilter, bookmarkKindFilter, hideReviewed, isReviewed, tab]);

  const items = useMemo(
    () => displayItems.filter(i => (tab === "bookmark" || i.severity === tab) && passesFilters(i)),
    [displayItems, tab, passesFilters]);

  // ── Keyboard triage: j/k (or ←/→) select · Enter/Space play · r reviewed ·
  //    b bookmark · i info. Turns the daily check into a seconds-long keyboard
  //    sweep instead of click-per-card. Inactive while typing, or while the
  //    player / info popover is open.
  const [kbIndex, setKbIndex] = useState(-1);
  useEffect(() => { if (kbIndex >= items.length) setKbIndex(items.length - 1); }, [items.length, kbIndex]);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (historyItem || infoItem || items.length === 0) return;
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable)) return;
      // A focused button handles Enter/Space natively — don't double-fire.
      if ((e.key === "Enter" || e.key === " ") && t?.closest("button")) return;
      const move = (d: number) => {
        e.preventDefault();
        setKbIndex(prev => {
          const next = Math.min(items.length - 1, Math.max(0, prev < 0 ? 0 : prev + d));
          document.querySelector(`[data-kbi="${next}"]`)?.scrollIntoView({ block: "nearest" });
          return next;
        });
      };
      const it = items[kbIndex];
      switch (e.key) {
        case "j": case "ArrowRight": move(1); break;
        case "k": case "ArrowLeft":  move(-1); break;
        case "Enter": case " ":
          if (!it) return;
          e.preventDefault();
          if (tab !== "bookmark") markReviewed([it.id]);
          if (it.clipEventId) setHistoryItem(it);
          break;
        case "r":
          if (!it) return;
          e.preventDefault();
          markReviewed([it.id]);
          break;
        case "b":
          if (it?.clipEventId) { e.preventDefault(); toggleBookmark(it.clipEventId); }
          break;
        case "i":
          if (it) { e.preventDefault(); setInfoItem(it); }
          break;
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [items, kbIndex, historyItem, infoItem, tab, markReviewed, toggleBookmark]);

  const alertCount = countItems.filter(i => i.severity === "alert" && passesFilters(i)).length;
  const detCount   = countItems.filter(i => i.severity === "detection" && passesFilters(i)).length;

  // Labels: always offer the common tracked objects (person, car, dog, cat, …)
  // so they're selectable even on days they didn't occur, plus any extra
  // specific labels actually seen that day. Zones only show what's present.
  const availableLabels = useMemo(() => {
    const present = new Set(displayItems.flatMap(i => i.labels));
    const extras = [...present].filter(l => !TRACKED_LABELS.includes(l)).sort();
    return [...TRACKED_LABELS, ...extras];
  }, [displayItems]);
  const availableZones = useMemo(() =>
    [...new Set(displayItems.flatMap(i => i.zones))].sort(), [displayItems]);

  const isToday = selectedDate === localDateStr();
  const dateLabel = isToday ? "Today"
    : selectedDate === localDateStr(new Date(Date.now() - 86400000)) ? "Yesterday"
    : new Date(selectedDate + "T12:00:00").toLocaleDateString([], { month: "short", day: "numeric" });

  // Full-screen "History" takeover — player + scrubbable timeline only (the
  // standard recording view). Clicking a card opens this directly and
  // starts playing the event; Back returns to the feed.
  if (historyItem) {
    return (
      <ReviewHistoryView
        camId={historyItem.camId}
        dayStr={selectedDate}
        initialEventId={historyItem.clipEventId}
        use12h={use12h}
        onBack={() => setHistoryItem(null)}
      />
    );
  }

  return (
    <div className={styles.root}>
      {/* ── Top toolbar: tabs · search · date (all one row) ── */}
      <div className={styles.header}>
        <div className={`lg ${styles.segTabs}`}>
          <button className={`${styles.segTab} ${tab === "alert" ? styles.segTabActive : ""}`}
            onClick={() => setTab("alert")}>
            <AlertTriangle size={13} /> Alerts <span className={styles.count}>{alertCount}</span>
          </button>
          <button className={`${styles.segTab} ${tab === "detection" ? styles.segTabActive : ""}`}
            onClick={() => setTab("detection")}>
            <Eye size={13} /> Detections <span className={styles.count}>{detCount}</span>
          </button>
          <button className={`${styles.segTab} ${tab === "vehicles" ? styles.segTabActive : ""}`}
            onClick={() => setTab("vehicles")}
            title="Vehicles detected by licence plate (ALPR)">
            <Car size={13} /> Vehicles
          </button>
          <button className={`${styles.segTab} ${tab === "sounds" ? styles.segTabActive : ""}`}
            onClick={() => setTab("sounds")}
            title="Sounds detected by the mic (barking, speech, alarms…)">
            <AudioLines size={13} /> Sounds
          </button>
          <button className={`${styles.segTab} ${tab === "bookmark" ? styles.segTabActive : ""}`}
            onClick={() => setTab("bookmark")}>
            <Bookmark size={13} /> Bookmarks <span className={styles.count}>{bookmarkIds.size}</span>
          </button>
        </div>

        {/* ── Search — whole-archive on event tabs, live filter on the
               Vehicles/Sounds tabs (applied client-side in those views) ── */}
        <div className={`lg ${styles.searchRow}`}>
          <Search size={14} className={styles.searchIcon} />
          <input
            className={styles.searchInput}
            placeholder={tab === "vehicles" ? "Filter sightings — plate, owner, type, color…"
              : tab === "sounds" ? "Filter sounds — bark, alarm, speech, glass…"
              : "Search all events — people, objects, plates, zones…"}
            value={query}
            onChange={e => setQuery(e.target.value)}
            spellCheck={false}
          />
          {searching && tab !== "vehicles" && tab !== "sounds" && (
            <span className={styles.searchStatus}>
              {events.length} result{events.length === 1 ? "" : "s"} · all dates
            </span>
          )}
          {query && (
            <button className={styles.searchClear} onClick={() => setQuery("")} title="Clear search" aria-label="Clear search">
              <X size={14} />
            </button>
          )}
        </div>

        <div className={styles.headerRight}>
          {/* Date — scopes every tab, Vehicles/Sounds included */}
          <button ref={calAnchor} className={`lg ${styles.glassBtn}`} onClick={() => setCalOpen(o => !o)}>
            <Calendar size={12} /> {dateLabel}
          </button>
          <GlassCalendar value={selectedDate} onChange={d => { setSelectedDate(d); setCalOpen(false); }}
            max={localDateStr()} open={calOpen} onClose={() => setCalOpen(false)} anchorRef={calAnchor}
            recordedDays={recordedDays} />
          <button className={`lg ${styles.iconBtn}`}
            onClick={() => tab === "vehicles" || tab === "sounds" ? setBrowserTick(t => t + 1) : load()}
            disabled={loading} title="Refresh">
            <RotateCcw size={13} className={loading ? styles.spin : ""} />
          </button>
        </div>
      </div>

      {/* ── Filters — Cameras is shared by every tab; the rest swap per tab ── */}
      <div className={styles.filters}>
        <FilterDropdown
          label="Cameras"
          options={cams.map(String)}
          selected={camFilter}
          onToggle={(c) => setCamFilter(prev => {
            const next = new Set(prev);
            next.has(c) ? next.delete(c) : next.add(c);
            return next;
          })}
          onSelectAll={() => setCamFilter(new Set(cams.map(String)))}
          onClear={() => setCamFilter(new Set())}
          format={(v) => camName(Number(v))}
          emptyText="No cameras"
        />
        {tab === "vehicles" && (<>
          <FilterDropdown
            label="Type"
            options={VEHICLE_TYPES}
            selected={vtypeFilter}
            onToggle={(v) => setVtypeFilter(prev => {
              const next = new Set(prev);
              next.has(v) ? next.delete(v) : next.add(v);
              return next;
            })}
            onSelectAll={() => setVtypeFilter(new Set(VEHICLE_TYPES))}
            onClear={() => setVtypeFilter(new Set())}
            format={cap}
          />
          <FilterDropdown
            label="Color"
            options={VEHICLE_COLORS}
            selected={vcolorFilter}
            onToggle={(v) => setVcolorFilter(prev => {
              const next = new Set(prev);
              next.has(v) ? next.delete(v) : next.add(v);
              return next;
            })}
            onSelectAll={() => setVcolorFilter(new Set(VEHICLE_COLORS))}
            onClear={() => setVcolorFilter(new Set())}
            format={cap}
          />
          <FilterDropdown
            label="Plate"
            options={PLATE_OPTS}
            selected={plateFilter}
            onToggle={(v) => setPlateFilter(prev => {
              const next = new Set(prev);
              next.has(v) ? next.delete(v) : next.add(v);
              return next;
            })}
            onSelectAll={() => setPlateFilter(new Set(PLATE_OPTS))}
            onClear={() => setPlateFilter(new Set())}
            format={(v) => PLATE_LABEL[v] ?? v}
          />
        </>)}
        {tab === "sounds" && (
          <FilterDropdown
            label="Sounds"
            options={SOUND_CATS}
            selected={soundFilter}
            onToggle={(v) => setSoundFilter(prev => {
              const next = new Set(prev);
              next.has(v) ? next.delete(v) : next.add(v);
              return next;
            })}
            onSelectAll={() => setSoundFilter(new Set(SOUND_CATS))}
            onClear={() => setSoundFilter(new Set())}
            format={(v) => SOUND_LABEL[v] ?? v}
          />
        )}
        {tab !== "vehicles" && tab !== "sounds" && (<>
        {tab === "bookmark" && (
          <FilterDropdown
            label="Kind"
            options={KIND_OPTS}
            selected={bookmarkKindFilter}
            onToggle={(k) => setBookmarkKindFilter(prev => {
              const next = new Set(prev);
              next.has(k) ? next.delete(k) : next.add(k);
              return next;
            })}
            onSelectAll={() => setBookmarkKindFilter(new Set(KIND_OPTS))}
            onClear={() => setBookmarkKindFilter(new Set())}
            format={(k) => KIND_LABEL[k] ?? k}
          />
        )}
        <FilterDropdown
          label="Labels"
          options={availableLabels}
          selected={labelFilter}
          onToggle={(l) => setLabelFilter(prev => {
            const next = new Set(prev);
            next.has(l) ? next.delete(l) : next.add(l);
            return next;
          })}
          onSelectAll={() => setLabelFilter(new Set(availableLabels))}
          onClear={() => setLabelFilter(new Set())}
        />
        <FilterDropdown
          label="Zones"
          options={availableZones}
          selected={zoneFilter}
          onToggle={(z) => setZoneFilter(prev => {
            const next = new Set(prev);
            next.has(z) ? next.delete(z) : next.add(z);
            return next;
          })}
          onSelectAll={() => setZoneFilter(new Set(availableZones))}
          onClear={() => setZoneFilter(new Set())}
          emptyText="No zones for this date"
        />
        {/* standard person filter — selecting switches into PERSON MODE
            (their events across the last year). Includes UNKNOWN recurring
            strangers (Train-tab clusters), id-bound — never name matching. */}
        <FilterDropdown
          label="People"
          options={[...knownPersons.map(p => p.id), ...unknownClusters.map(c => c.token)]}
          selected={personFilter}
          onToggle={(t) => setPersonFilter(prev => {
            const next = new Set(prev);
            next.has(t) ? next.delete(t) : next.add(t);
            return next;
          })}
          onSelectAll={() => setPersonFilter(new Set([...knownPersons.map(p => p.id), ...unknownClusters.map(c => c.token)]))}
          onClear={() => setPersonFilter(new Set())}
          format={(t) => t.startsWith("cluster:")
            ? (unknownClusters.find(c => c.token === t)?.label ?? "Stranger")
            : (knownPersons.find(p => p.id === t)?.name ?? t)}
          emptyText="No people recognized yet"
        />
        {/* Hide-reviewed toggle — pushed to the right of the bar. */}
        <button
          className={`lg ${styles.toggle} ${hideReviewed ? styles.toggleOn : ""}`}
          onClick={() => setHideReviewed(v => !v)}
          title="Show only un-reviewed items">
          <span className={styles.toggleTrack}><span className={styles.toggleKnob} /></span>
          Hide reviewed
        </button>
        </>)}
      </div>

      {/* ── Vehicles / Sounds browsers — filtered by the toolbar above ── */}
      {tab === "vehicles" && (
        <div style={{ flex: 1, overflow: "auto", padding: "0 16px 16px" }}>
          <VehiclesView
            cams={[...camFilter].join(",")}
            types={[...vtypeFilter].join(",")}
            colors={[...vcolorFilter].join(",")}
            plates={[...plateFilter].join(",")}
            date={selectedDate}
            query={debouncedQuery}
            refreshTick={browserTick}
            bookmarkIds={bookmarkIds}
            onToggleBookmark={toggleBookmark}
            onDelete={(id) => void deleteEventIds([id])}
            onOpenPlayer={(id, camId) => setHistoryItem({ camId, clipEventId: id })}
          />
        </div>
      )}
      {tab === "sounds" && (
        <div style={{ flex: 1, overflow: "auto", padding: "0 16px 16px" }}>
          <AudioView
            cams={[...camFilter].join(",")}
            cats={[...soundFilter].join(",")}
            date={selectedDate}
            query={debouncedQuery}
            refreshTick={browserTick}
            bookmarkIds={bookmarkIds}
            onToggleBookmark={toggleBookmark}
            onDelete={(id) => void deleteEventIds([id])}
            onOpenPlayer={(id, camId) => setHistoryItem({ camId, clipEventId: id })}
          />
        </div>
      )}

      {/* ── Feed ── */}
      {tab !== "vehicles" && tab !== "sounds" && (
      <div className={styles.feed}>
        {similarTo && (
          <div style={{
            gridColumn: "1 / -1", flexBasis: "100%", width: "100%",
            display: "flex", alignItems: "center", gap: 8,
            padding: "8px 12px", marginBottom: 4, borderRadius: 10,
            background: "color-mix(in srgb, var(--status-idle) 10%, transparent)", border: "1px solid color-mix(in srgb, var(--status-idle) 35%, transparent)",
            fontSize: 12, color: "var(--text-secondary)",
          }}>
            <Sparkles size={13} color="var(--status-idle)" />
            <span style={{ flex: 1 }}>
              Showing events visually similar to <strong style={{ color: "var(--text-primary)" }}>{similarTo.label}</strong>
            </span>
            <button onClick={() => setSimilarTo(null)}
              style={{
                display: "inline-flex", alignItems: "center", gap: 4,
                padding: "3px 10px", borderRadius: 999, cursor: "pointer",
                border: "1px solid var(--border-strong)", background: "transparent",
                color: "var(--text-primary)", fontSize: 11, fontWeight: 700,
              }}>
              <X size={11} /> Clear
            </button>
          </div>
        )}
        {personEvents !== null && !similarTo && !debouncedQuery && (
          <div style={{
            gridColumn: "1 / -1", flexBasis: "100%", width: "100%",
            display: "flex", alignItems: "center", gap: 8,
            padding: "8px 12px", marginBottom: 4, borderRadius: 10,
            background: "var(--accent-glow)", border: "1px solid var(--accent)",
            fontSize: 12, color: "var(--text-secondary)",
          }}>
            <UserRound size={13} style={{ color: "var(--accent)" }} />
            <span style={{ flex: 1 }}>
              Showing events with <strong style={{ color: "var(--text-primary)" }}>
                {[...personFilter].map(t => t.startsWith("cluster:")
                  ? (unknownClusters.find(c => c.token === t)?.label ?? "a stranger")
                  : (knownPersons.find(p => p.id === t)?.name ?? t)).join(", ")}
              </strong> · last 365 days
            </span>
            <button onClick={() => setPersonFilter(new Set())}
              style={{
                display: "inline-flex", alignItems: "center", gap: 4,
                padding: "3px 10px", borderRadius: 999, cursor: "pointer",
                border: "1px solid var(--border-strong)", background: "transparent",
                color: "var(--text-primary)", fontSize: 11, fontWeight: 700,
              }}>
              <X size={11} /> Clear
            </button>
          </div>
        )}
        {!loading && items.length === 0 && (
          <div className={styles.empty}>
            <Film size={32} />
            <span>
              {similarTo
                ? "No similar events — needs the Semantic Search skill"
                : personEvents !== null
                ? `No ${tab === "alert" ? "alerts" : "detections"} in the last year`
                : tab === "bookmark"
                ? "No bookmarks"
                : searching
                ? `No matching ${tab === "alert" ? "alerts" : "detections"} for “${debouncedQuery}”`
                : `No ${tab === "alert" ? "alerts" : "detections"} on ${dateLabel.toLowerCase()}`}
            </span>
          </div>
        )}
        {items.map((item, idx) => {
          const color = riskColor(item.peak);
          // Match the focus-section event card (ThumbnailStrip): 16:9 thumb +
          // risk pill + time + AI-summary snippet. Identity (face name / plate)
          // gets its own chip below, so the snippet falls back to labels only.
          const snippet = item.summary
            ?? (item.labels.length > 0 ? item.labels.join(", ") : null);
          return (
            <button key={item.id} data-kbi={idx}
              className={`${styles.card} ${idx === kbIndex ? styles.cardKb : ""}`}
              onClick={() => { setKbIndex(idx); if (tab !== "bookmark") markReviewed([item.id]); if (item.clipEventId) setHistoryItem(item); }}>
              <div className={styles.cardImg}>
                {item.thumbnail
                  ? <img src={eventThumbSrc(item.thumbnail, item.thumbId, streamInfo) ?? ""} alt="" loading="lazy"
                      onError={e => { e.currentTarget.style.display = "none"; }} />
                  : <div className={styles.cardImgBlank}><Film size={16} /></div>}
                <div className={styles.cardRisk} style={{ background: tint(color, 87) }}>
                  {riskLabel(item.peak)}
                </div>
                {cams.length > 1 && <span className={styles.cardCam}>CAM {item.camId + 1}</span>}
                {item.clipEventId && <span className={styles.cardPlay}><Play size={12} fill="#fff" /></span>}
                {/* Every functional icon in ONE row, top-right, hover-revealed.
                    These were four separate absolutely-positioned buttons whose
                    corners disagreed with the Sounds and Vehicles cards. */}
                <CardActions
                  saved={!!item.clipEventId && bookmarkIds.has(item.clipEventId)}
                  bookmarkTitle={item.clipEventId && bookmarkIds.has(item.clipEventId) ? "Remove bookmark" : "Bookmark this event"}
                  onBookmark={item.clipEventId ? () => toggleBookmark(item.clipEventId!) : undefined}
                  onSimilar={item.clipEventId ? () => {
                    setQuery("");
                    setSimilarTo({ id: item.clipEventId!, label: snippet ?? "this event" });
                  } : undefined}
                  onDownload={tab === "bookmark" && item.clipEventId && streamInfo
                    ? () => void exportEventById(streamInfo, safeName(camName(item.camId)), item.clipEventId!, showToast)
                    : undefined}
                  onDelete={() => void deleteItem(item)}
                  deleteTitle={item.memberIds.length > 1
                    ? `Delete these ${item.memberIds.length} events and their clip`
                    : "Delete this event and its clip"}
                />
              </div>
              {/* Clean card: time + a details (i). Name / description / labels live
                  in the floating info popover so the feed scans cleanly. */}
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "0 8px 4px", gap: 6 }}>
                {/* Person-mode crop — WHO is in this event. Moved off the
                    thumbnail: the action column now runs down the left edge and
                    a 34px avatar at bottom-left sat directly under it. */}
                {(() => {
                  const crop = personEvents !== null
                    ? item.memberIds.map(id => personCrops.get(id)).find(Boolean) : null;
                  return crop ? (
                    <img src={`data:image/jpeg;base64,${crop}`} alt="" title="This person, in this event"
                      style={{ width: 18, height: 18, borderRadius: 5, objectFit: "cover",
                        border: "1px solid var(--border-strong)", background: "#000", flexShrink: 0 }} />
                  ) : null;
                })()}
                <span className={styles.cardTime} style={{ padding: 0, flex: 1 }}>{fmtShortTime(item.start, use12h)}</span>
                {/* A card is a GROUP. Without this it looked identical whether it
                    held one event or ten, so the other nine were simply invisible. */}
                {item.memberIds.length > 1 && (
                  <span role="button" tabIndex={0}
                    title={`${item.memberIds.length} events in this item — click to list them`}
                    onClick={e => { e.stopPropagation(); setInfoItem(item); }}
                    onKeyDown={e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); e.stopPropagation(); setInfoItem(item); } }}
                    style={{ fontSize: 10, fontWeight: 700, padding: "1px 6px", borderRadius: 999,
                      background: "rgb(var(--ink) / 0.10)", border: "1px solid rgb(var(--ink) / 0.16)",
                      color: "var(--text-primary)", cursor: "pointer", flexShrink: 0,
                      fontVariantNumeric: "tabular-nums" }}>
                    {item.memberIds.length}
                  </span>
                )}
                <CardInfoButton onOpen={() => setInfoItem(item)} />
              </div>
            </button>
          );
        })}
      </div>
      )}

      {!settings?.nvr_enabled && tab !== "vehicles" && tab !== "sounds" && (
        <div className={styles.nvrHint}>Enable NVR in Settings to record continuous footage for review.</div>
      )}

      {/* Floating details popover — the name / description / labels that used to
          clutter every card, shown on demand from the (i) button. */}
      {infoItem && (() => {
        const it = infoItem;
        const chips: CardChip[] = [];
        if (it.subLabel)      chips.push({ icon: "👤", text: it.subLabel });
        if (it.plate)         chips.push({ icon: "🚗", text: it.plate });
        if (it.crossingLabel) chips.push({ icon: "🚷", text: it.crossingLabel });
        if (it.fallLabel)     chips.push({ icon: "⚠️", text: it.fallLabel });
        if (it.audioSound)    chips.push({ icon: "🔊", text: it.audioSound });
        if (it.speedLabel)    chips.push({ icon: "🏎️", text: it.speedLabel });
        return (
          <CardInfoModal
            onClose={() => setInfoItem(null)}
            thumb={it.thumbnail ? eventThumbSrc(it.thumbnail, it.thumbId, streamInfo) : null}
            title={it.title ?? `${riskLabel(it.peak)} event`}
            camLabel={camName(it.camId)}
            timeLabel={fmtShortTime(it.start, use12h)}
            chips={chips}
            summary={it.summary}
            detail={
              <>
                {it.labels.length > 0 && (
                  <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
                    Detected: {it.labels.join(", ")}{it.zones.length ? ` · zones: ${it.zones.join(", ")}` : ""}
                  </div>
                )}
                {!it.summary && it.labels.length === 0 && chips.length === 0 && (
                  <div style={{ fontSize: 12, color: "var(--text-muted)" }}>Motion event — no AI description.</div>
                )}
                {it.memberIds.length > 1 && (
                  <EventMembers item={it} use12h={use12h} streamInfo={streamInfo}
                    onPick={id => { setInfoItem(null); setHistoryItem({ camId: it.camId, clipEventId: id }); }} />
                )}
              </>
            }
          />
        );
      })()}
    </div>
  );
}
