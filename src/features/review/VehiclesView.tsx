// Vehicles view — ALPR sightings browser (moved from the People panel into
// Review, where events live). Filters live in the Review toolbar (shared
// Cameras/date/search + Type/Color/Plate dropdowns) and arrive as props;
// this view owns keyset paging, the plate roster, the recurring-plate nudge
// and the clip player.
import { api, Vehicle, VehicleEvent } from "../../api";
import { useEffect, useRef, useState, useCallback } from "react";
import { listen } from "@tauri-apps/api/event";
import { Car, Play, Sparkles } from "lucide-react";
import { useStore } from "../../store";
import { eventThumbSrc } from "../../lib/eventThumb";
import cardStyles from "./ReviewFeed.module.css";
import { CardActions, CardInfoButton, CardInfoModal, type CardChip } from "./CardChrome";
import { fmtWhen, dayBoundsUtc, isLocalToday } from "../../lib/time";
import { OBJECT_COLOR_SWATCH, tint } from "../../lib/palette";


function Empty({ text }: { text: string }) {
  return <div style={{ fontSize: 12, color: "var(--text-muted)", padding: "14px 0", textAlign: "center" }}>{text}</div>;
}

const VEHICLE_COLOR_HEX = OBJECT_COLOR_SWATCH;

export function VehiclesView({ cams, types, colors, plates, date, query, refreshTick, bookmarkIds, onToggleBookmark, onDelete, onOpenPlayer }: {
  /** CSV multi-select filters from the Review toolbar dropdowns ("" = all). */
  cams: string; types: string; colors: string; plates: string;
  /** Selected day (Review toolbar date) — scopes the sightings server-side. */
  date: string;
  /** Toolbar search box — filters the loaded sightings client-side. */
  query: string;
  /** Toolbar refresh button — bump forces a reset + reload. */
  refreshTick: number;
  /** Saved-event ids + toggle — same bookmark set as the Events feed. */
  bookmarkIds: Set<string>;
  onToggleBookmark: (id: string) => void;
  /** Delete this sighting's event + clip (same path as the Events feed). */
  onDelete: (id: string) => void;
  /** Open the shared full events player (ReviewHistoryView) on this sighting. */
  onOpenPlayer: (eventId: string, camId: number) => void;
}) {
  const streamInfo = useStore(s => s.streamInfo);
  /** Sighting whose details sheet is open. */
  const [info, setInfo] = useState<VehicleEvent | null>(null);
  const [vehicles, setVehicles] = useState<Vehicle[]>([]);
  const [loading, setLoading] = useState(true);
  const [events, setEvents] = useState<VehicleEvent[]>([]);
  const [hasMore, setHasMore] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [refreshErr, setRefreshErr] = useState(false);
  const inFlight = useRef(false);
  const PAGE = 60;
  const CAP = 2000;
  const mergeEvents = useCallback((prev: VehicleEvent[], page: VehicleEvent[]) => {
    const seen = new Map(prev.map(e => [e.id, e] as const));
    for (const e of page) seen.set(e.id, e);
    return [...seen.values()]
      .sort((a, b) => b.started_at.localeCompare(a.started_at))
      .slice(0, CAP);
  }, []);
  // Server-side filter set — each combo pages its own keyset stream, scoped
  // to the toolbar's selected day (same window contract as the Events feed).
  const filterOpts = useCallback(() => {
    const { fromUtc, toUtc } = dayBoundsUtc(date);
    return {
      vtype: types || undefined,
      color: colors || undefined,
      plate: plates || undefined,
      cams:  cams  || undefined,
      from: fromUtc, to: toUtc,
    };
  }, [types, colors, plates, cams, date]);
  const load = useCallback(async () => {
    if (inFlight.current) return;
    inFlight.current = true;
    try {
      const [ev, ros] = await Promise.all([
        api.listVehicleEvents({ limit: PAGE, ...filterOpts() }),
        api.listVehicles(365),
      ]);
      setEvents(prev => mergeEvents(prev, ev));
      if (ev.length >= PAGE) setHasMore(true);
      setVehicles(ros);
      setRefreshErr(false);
    } catch {
      setRefreshErr(true); // keep last-good
    } finally {
      inFlight.current = false;
      setLoading(false);
    }
  }, [filterOpts, mergeEvents]);
  const loadMore = useCallback(async () => {
    if (loadingMore || events.length === 0) return;
    setLoadingMore(true);
    try {
      const before = events[events.length - 1].started_at;
      const page = await api.listVehicleEvents({ limit: PAGE, before, ...filterOpts() });
      setEvents(prev => mergeEvents(prev, page));
      setHasMore(page.length >= PAGE);
      setRefreshErr(false);
    } catch { setRefreshErr(true); }
    finally { setLoadingMore(false); }
  }, [events, filterOpts, loadingMore, mergeEvents]);
  useEffect(() => {
    setEvents([]);
    setHasMore(false);
    setLoading(true);
    load();
    const un = listen("agent:analyzed", () => { if (!document.hidden) load(); });
    const iv = setInterval(() => { if (!document.hidden) load(); }, 60_000);
    return () => { un.then(f => f()); clearInterval(iv); };
  }, [load, refreshTick]);

  // Self-learning proposal: a plate that keeps coming back but has no name is a
  // REGULAR the user probably wants to label (same propose-then-confirm pattern as
  // recurring-stranger faces). Surface it instead of leaving it buried in the grid.
  const recurringUnknown = vehicles.filter(v => !v.name && v.count >= 3);

  if (loading) return <Empty text="Loading…" />;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      {refreshErr && (
        <div className="glass" style={{ padding: "8px 12px", fontSize: 11, color: "var(--status-warn)",
          display: "flex", alignItems: "center", gap: 8 }}>
          Couldn't refresh vehicles — showing the last loaded set.
          <button onClick={() => load()} style={{ fontSize: 11, fontWeight: 700, cursor: "pointer",
            background: "none", border: "none", color: "var(--accent)", textDecoration: "underline" }}>
            Retry
          </button>
        </div>
      )}
      {recurringUnknown.length > 0 && (
        <div className="glass" style={{ padding: "12px 16px", display: "flex",
          alignItems: "center", gap: 12, border: "1px solid var(--accent)" }}>
          <Sparkles size={16} style={{ color: "var(--accent)", flexShrink: 0 }} />
          <div style={{ fontSize: 12, lineHeight: 1.55 }}>
            <b>Recurring vehicle{recurringUnknown.length > 1 ? "s" : ""} spotted:</b>{" "}
            {recurringUnknown.slice(0, 3).map(v =>
              `${v.plate} (${v.count}×${v.cameras.length > 1 ? `, ${v.cameras.length} cams` : ""})`
            ).join(" · ")}
            {" — "}name {recurringUnknown.length > 1 ? "them" : "it"} in Settings → known plates
            (<code style={{ fontFamily: "var(--font-mono)", fontSize: 11 }}>PLATE=Name</code>)
            so alerts say "the mail van", not a plate number.
          </div>
        </div>
      )}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))", gap: 12 }}>
      {vehicles.map(v => (
        <div key={v.plate} className="glass" style={{ padding: 0, overflow: "hidden" }}>
          <div style={{ position: "relative" }}>
            {v.thumbnail
              ? <img src={v.thumbnail.startsWith("data:") ? v.thumbnail : `data:image/jpeg;base64,${v.thumbnail}`} alt=""
                  style={{ width: "100%", aspectRatio: "16/10", objectFit: "cover", display: "block" }} />
              : <div style={{ width: "100%", aspectRatio: "16/10", background: "rgb(var(--ink) / 0.05)",
                  display: "flex", alignItems: "center", justifyContent: "center" }}><Car size={20} /></div>}
            <div style={{ position: "absolute", top: 6, right: 6, padding: "2px 8px", borderRadius: 999,
              fontSize: 10, fontWeight: 800, background: "rgba(0,0,0,0.62)", color: "#fff" }}>×{v.count}</div>
          </div>
          <div style={{ padding: "8px 10px" }}>
            <div style={{ fontSize: 13, fontWeight: 700, fontFamily: "var(--font-mono)", letterSpacing: 0.5 }}>{v.plate}</div>
            {v.name && <div style={{ fontSize: 11, color: "var(--accent)", fontWeight: 600 }}>{v.name}</div>}
            <div style={{ fontSize: 10.5, color: "var(--text-muted)", marginTop: 3, lineHeight: 1.5 }}>
              {v.count} sighting{v.count === 1 ? "" : "s"}
              {v.cameras.length > 1 ? ` · ${v.cameras.length} cams` : ` · camera ${(v.cameras[0] ?? 0) + 1}`}
              <br />last seen {fmtWhen(v.last_seen)}
            </div>
          </div>
        </div>
      ))}
      </div>

      {/* ── Vehicle sightings — Events-style cards; filters come from the
             Review toolbar (Cameras/Type/Color/Plate + date server-side,
             search box client-side over the loaded set) ─────────────────── */}
      {(() => {
      const q = query.trim().toLowerCase();
      const shown = q
        ? events.filter(e => [e.plate, e.owner, e.vtype, e.color]
            .some(v => v && v.toLowerCase().includes(q)))
        : events;
      const dayTxt = isLocalToday(date) ? "today" : `on ${date}`;
      return shown.length === 0 ? (
        // Same empty state as the Alerts / Detections feed: centred 32 px icon over
        // muted text (`.empty` in ReviewFeed.module.css), so every Review tab reads
        // as one surface instead of each inventing its own box.
        <div className={cardStyles.empty}>
          <Car size={32} />
          <span>
            {q ? `No sightings matching “${query.trim()}” ${dayTxt}.`
              : types || colors || plates || cams
              ? `No sightings match these filters ${dayTxt}`
              : `No vehicle sightings ${dayTxt}`}
          </span>
        </div>
      ) : (
        <>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(150px, 1fr))", gap: 10 }}>
          {shown.map(e => {
            const dur = e.duration_secs != null && e.duration_secs > 0
              ? `${e.duration_secs < 60 ? `${Math.round(e.duration_secs)}s` : `${Math.round(e.duration_secs / 60)}m`}` : null;
            const thumb = e.thumbnail ? eventThumbSrc(e.thumbnail, e.id, streamInfo) : null;
            const chex = e.color ? VEHICLE_COLOR_HEX[e.color] : null;
            const pillFg = e.color && ["white", "silver", "yellow"].includes(e.color) ? "#15171c" : "#fff";
            const saved = bookmarkIds.has(e.id);
            return (
              <button key={e.id} className={cardStyles.card} onClick={() => onOpenPlayer(e.id, e.cam_id ?? 0)}
                title="Play this sighting in the full player">
                <div className={cardStyles.cardImg}>
                  {thumb
                    ? <img src={thumb} alt="" loading="lazy"
                        onError={e => { e.currentTarget.style.display = "none"; }} />
                    : <div className={cardStyles.cardImgBlank}><Car size={16} /></div>}
                  <div className={cardStyles.cardRisk}
                    /* Hairline so the pale swatches (white/silver) still have an
                       edge on a light card — the fill alone vanishes there. */
                    style={{ background: chex ? tint(chex, 90) : tint("var(--status-idle)", 85),
                             color: pillFg, border: "1px solid rgb(var(--ink) / 0.18)" }}>
                    {e.color ? `${e.color} ${e.vtype}` : e.vtype}
                  </div>
                  <span className={cardStyles.cardCam}>CAM {(e.cam_id ?? 0) + 1}</span>
                  <span className={cardStyles.cardPlay}><Play size={12} fill="#fff" /></span>
                  {/* Same row, same corner, same order as Events and Sounds. The
                      plate pill that sat bottom-right collided with .cardRisk and
                      now lives in the details sheet. */}
                  <CardActions
                    saved={saved}
                    bookmarkTitle={saved ? "Remove bookmark" : "Bookmark this sighting"}
                    onBookmark={() => onToggleBookmark(e.id)}
                    onDelete={() => onDelete(e.id)}
                    deleteTitle="Delete this sighting"
                  />
                </div>
                {/* Identical footer to Events and Sounds: time left, (i) right. */}
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between",
                  padding: "0 8px 4px", gap: 6 }}>
                  <span className={cardStyles.cardTime} style={{ padding: 0 }}>
                    {fmtWhen(e.started_at)}{dur ? ` · ${dur}` : ""}
                  </span>
                  <CardInfoButton onOpen={() => setInfo(e)} />
                </div>
              </button>
            );
          })}
        </div>
        {info && (() => {
          const v = info;
          const chips: CardChip[] = [];
          if (v.plate) chips.push({ icon: "\u{1F520}", text: v.plate });
          if (v.color) chips.push({ icon: "\u{1F3A8}", text: v.color });
          if (v.owner) chips.push({ icon: "\u{1F464}", text: v.owner });
          if (v.speed_kmh != null && v.speed_kmh >= 1)
            chips.push({ icon: "\u{1F3CE}️", text: `${Math.round(v.speed_kmh)} km/h` });
          return (
            <CardInfoModal
              onClose={() => setInfo(null)}
              thumb={v.thumbnail ? eventThumbSrc(v.thumbnail, v.id, streamInfo) : null}
              thumbFallback={<Car size={16} />}
              title={v.color ? `${v.color} ${v.vtype}` : v.vtype}
              camLabel={`Camera ${(v.cam_id ?? 0) + 1}`}
              timeLabel={fmtWhen(v.started_at)}
              chips={chips}
            />
          );
        })()}
        {hasMore && (
          <div style={{ display: "flex", justifyContent: "center", marginTop: 12 }}>
            <button onClick={loadMore} disabled={loadingMore}
              style={{ fontSize: 11.5, fontWeight: 700, padding: "8px 18px", borderRadius: 999,
                cursor: loadingMore ? "wait" : "pointer", background: "rgb(var(--ink) / 0.05)",
                color: "var(--text-secondary)", border: "1px solid var(--border)" }}>
              {loadingMore ? "Loading…" : "Load older sightings"}
            </button>
          </div>
        )}
        </>
      );
      })()}

    </div>
  );
}
