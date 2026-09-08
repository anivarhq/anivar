// Sounds view — YAMNet audio-event browser (moved from the People panel into
// Review). Filters live in the Review toolbar (shared Cameras/date/search +
// the Sounds category dropdown) and arrive as props; this view owns the 7-day
// stats strip, loudness badges, keyset paging and the clip player with audio.
import { api, AudioEvent } from "../../api";
import { useEffect, useRef, useState, useCallback } from "react";
import { listen } from "@tauri-apps/api/event";
import { Play, AudioLines, Volume2 } from "lucide-react";
import { useStore } from "../../store";
import { eventThumbSrc } from "../../lib/eventThumb";
import cardStyles from "./ReviewFeed.module.css";
import { CardActions, CardInfoButton, CardInfoModal, type CardChip } from "./CardChrome";
import { fmtWhen, dayBoundsUtc, isLocalToday } from "../../lib/time";


function Empty({ text }: { text: string }) {
  return <div style={{ fontSize: 12, color: "var(--text-muted)", padding: "14px 0", textAlign: "center" }}>{text}</div>;
}

// ─── Audio tab — YAMNet sound-detection events (barking, speech, alarms…) ──────
export function AudioView({ cams, cats, date, query, refreshTick, bookmarkIds, onToggleBookmark, onDelete, onOpenPlayer }: {
  /** CSV multi-select filters from the Review toolbar dropdowns ("" = all). */
  cams: string; cats: string;
  /** Selected day (Review toolbar date) — scopes the sounds server-side. */
  date: string;
  /** Toolbar search box — filters the loaded sounds client-side. */
  query: string;
  /** Toolbar refresh button — bump forces a reset + reload. */
  refreshTick: number;
  /** Saved-event ids + toggle — same bookmark set as the Events feed. */
  bookmarkIds: Set<string>;
  onToggleBookmark: (id: string) => void;
  /** Delete this sound event + clip (same path as the Events feed). */
  onDelete: (id: string) => void;
  /** Open the shared full events player (ReviewHistoryView) on this sound. */
  onOpenPlayer: (eventId: string, camId: number) => void;
}) {
  const settings = useStore(s => s.settings);
  const streamInfo = useStore(s => s.streamInfo);
  const [events, setEvents] = useState<AudioEvent[]>([]);
  /** Sound whose details sheet is open — everything descriptive lives here now. */
  const [info, setInfo] = useState<AudioEvent | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshErr, setRefreshErr] = useState(false);
  // Reliable loading on the single source of truth (motion_events): the same
  // event-driven refresh contract every other event surface uses. One in-flight
  // load at a time; failures KEEP the last-good data (never blank-on-error).
  const inFlight = useRef(false);
  const [hasMore, setHasMore] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const PAGE = 60;
  const CAP = 2000; // responsiveness ceiling for the accumulated grid
  /** Merge a page into the accumulated set: dedupe by id, newest first. */
  const mergeEvents = useCallback((prev: AudioEvent[], page: AudioEvent[]) => {
    const seen = new Map(prev.map(e => [e.id, e] as const));
    for (const e of page) seen.set(e.id, e); // pages carry fresher rows (durations fill in)
    return [...seen.values()]
      .sort((a, b) => b.started_at.localeCompare(a.started_at))
      .slice(0, CAP);
  }, []);
  // Server-side filter set from the toolbar props: categories (CSV, union),
  // cameras, and the selected day's bounds (same window contract as Events).
  const filterOpts = useCallback(() => {
    const { fromUtc, toUtc } = dayBoundsUtc(date);
    return { category: cats || undefined, cams: cams || undefined, from: fromUtc, to: toUtc };
  }, [cats, cams, date]);
  // Page 1 + live refresh: MERGES so already-browsed older pages survive.
  const load = useCallback(async () => {
    if (inFlight.current) return;
    inFlight.current = true;
    try {
      const e = await api.listAudioEvents({ limit: PAGE, ...filterOpts() });
      setEvents(prev => mergeEvents(prev, e));
      if (e.length >= PAGE) setHasMore(true);
      setRefreshErr(false);
    } catch {
      setRefreshErr(true); // keep last-good events visible
    } finally {
      inFlight.current = false;
      setLoading(false);
    }
  }, [filterOpts, mergeEvents]);
  // Older history: keyset cursor from the oldest loaded row.
  const loadMore = useCallback(async () => {
    if (loadingMore || events.length === 0) return;
    setLoadingMore(true);
    try {
      const before = events[events.length - 1].started_at;
      const page = await api.listAudioEvents({ limit: PAGE, before, ...filterOpts() });
      setEvents(prev => mergeEvents(prev, page));
      setHasMore(page.length >= PAGE);
      setRefreshErr(false);
    } catch {
      setRefreshErr(true);
    } finally { setLoadingMore(false); }
  }, [events, filterOpts, loadingMore, mergeEvents]);
  // Filter change = a different server-filtered stream: reset + refetch page 1.
  useEffect(() => {
    setEvents([]);
    setHasMore(false);
    setLoading(true);
    load();
    // Backend emits agent:analyzed on audio event OPEN and CLOSE — refresh live.
    const un = listen("agent:analyzed", () => { if (!document.hidden) load(); });
    const iv = setInterval(() => { if (!document.hidden) load(); }, 60_000); // fallback heartbeat
    return () => { un.then(f => f()); clearInterval(iv); };
  }, [load, refreshTick]);

  // Honest empty state: "no events" when detection is OFF means the feature is
  // off, not that nothing happened — say so, and point at the exact switch.
  const audioOff = settings ? !settings.audio_detection : false;
  if (loading) return <Empty text="Loading…" />;

  // Confidence → color: strong hits pop accent, weak ones stay muted.
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      {audioOff && (
        <div className="glass" style={{ padding: "10px 14px", margin: "0 2px 2px", fontSize: 11.5,
          color: "var(--text-secondary)", display: "flex", alignItems: "center", gap: 10 }}>
          <AudioLines size={14} style={{ opacity: 0.6, flexShrink: 0 }} />
          <span><b>Audio detection is OFF</b> — turn it on in Settings</span>
        </div>
      )}
      {refreshErr && (
        <div className="glass" style={{ padding: "8px 12px", margin: "0 2px 10px", fontSize: 11,
          color: "var(--status-warn)", display: "flex", alignItems: "center", gap: 8 }}>
          Couldn't refresh audio events — showing the last loaded set.
          <button onClick={() => load()} style={{ fontSize: 11, fontWeight: 700, cursor: "pointer",
            background: "none", border: "none", color: "var(--accent)", textDecoration: "underline" }}>
            Retry
          </button>
        </div>
      )}

      {(() => {
        // Toolbar search box filters the loaded (server-filtered) set client-side.
        const q = query.trim().toLowerCase();
        const shown = q
          ? events.filter(e => e.sound.toLowerCase().includes(q)
              || (e.classes ?? []).some(c => c.l.toLowerCase().includes(q)))
          : events;
        const dayTxt = isLocalToday(date) ? "today" : `on ${date}`;
        if (shown.length === 0) {
          return (
            // Matches the Alerts / Detections feed empty state (`.empty` in
            // ReviewFeed.module.css): centred 32 px icon over muted text.
            <div className={cardStyles.empty}>
              <AudioLines size={32} />
              <span>
                {q ? `No sounds matching “${query.trim()}” ${dayTxt}.`
                  : cats === "high_pitch"
                  ? `No high-pitched sounds ${dayTxt}`
                  : cats || cams
                  ? `No sounds match these filters ${dayTxt}`
                  : `No sound events ${dayTxt}`}
              </span>
            </div>
          );
        }
        return (
          <>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(150px, 1fr))", gap: 10 }}>
            {shown.map(e => {
              const dur = e.duration_secs != null && e.duration_secs > 0
                ? `${e.duration_secs < 60 ? `${Math.round(e.duration_secs)}s` : `${Math.round(e.duration_secs / 60)}m`}` : null;
              const thumb = e.thumbnail ? eventThumbSrc(e.thumbnail, e.id, streamInfo) : null;
              // Same skeleton + CSS as the Events feed card — one visual language.
              const saved = bookmarkIds.has(e.id);
              return (
                <button key={e.id} className={cardStyles.card} onClick={() => onOpenPlayer(e.id, e.cam_id ?? 0)}
                  title="Play this sound in the full player (with audio)">
                  <div className={cardStyles.cardImg}>
                    {thumb
                      ? <img src={thumb} alt="" loading="lazy"
                          onError={e => { e.currentTarget.style.display = "none"; }} />
                      : <div className={cardStyles.cardImgBlank}><Volume2 size={16} /></div>}
                    <div className={cardStyles.cardRisk}
                      style={{ background: e.high_pitch ? "color-mix(in srgb, var(--status-warn) 87%, transparent)" : "color-mix(in srgb, var(--status-idle) 85%, transparent)" }}>
                      {e.high_pitch ? "HIGH-PITCH" : e.sound}
                    </div>
                    <span className={cardStyles.cardCam}>CAM {(e.cam_id ?? 0) + 1}</span>
                    <span className={cardStyles.cardPlay}><Play size={12} fill="#fff" /></span>
                    {/* Same row, same corner, same order as the Events cards.
                        The loudness pill that used to sit bottom-right collided
                        with .cardRisk and is now in the details sheet. */}
                    <CardActions
                      saved={saved}
                      bookmarkTitle={saved ? "Remove bookmark" : "Bookmark this sound"}
                      onBookmark={() => onToggleBookmark(e.id)}
                      onDelete={() => onDelete(e.id)}
                      deleteTitle="Delete this sound event"
                    />
                  </div>
                  {/* Identical footer to the Events card: time left, (i) right.
                      The sound name, confidence and YAMNet class chips used to
                      stack here and made these tiles a different height and a
                      different shape from every other card. */}
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
            const s = info;
            const chips: CardChip[] = [{ icon: "🔊", text: s.sound }];
            if (s.high_pitch)          chips.push({ icon: "📈", text: "High-pitch" });
            if (s.loudness_db != null) chips.push({ icon: "🔉", text: `${Math.round(s.loudness_db)} dB` });
            chips.push({ icon: "🎯", text: `${Math.round(s.score * 100)}% confidence` });
            return (
              <CardInfoModal
                onClose={() => setInfo(null)}
                thumb={s.thumbnail ? eventThumbSrc(s.thumbnail, s.id, streamInfo) : null}
                thumbFallback={<Volume2 size={16} />}
                title={s.sound.charAt(0).toUpperCase() + s.sound.slice(1)}
                camLabel={`Camera ${(s.cam_id ?? 0) + 1}`}
                timeLabel={fmtWhen(s.started_at)}
                chips={chips}
                detail={s.classes && s.classes.length > 0 ? (
                  <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
                    Also heard: {s.classes.map(c => `${c.l} ${Math.round(c.s * 100)}%`).join(" · ")}
                  </div>
                ) : undefined}
              />
            );
          })()}
          {hasMore && (
            <div style={{ display: "flex", justifyContent: "center", marginTop: 12 }}>
              <button onClick={loadMore} disabled={loadingMore}
                style={{ fontSize: 11.5, fontWeight: 700, padding: "8px 18px", borderRadius: 999,
                  cursor: loadingMore ? "wait" : "pointer", background: "rgb(var(--ink) / 0.05)",
                  color: "var(--text-secondary)", border: "1px solid var(--border)" }}>
                {loadingMore ? "Loading…" : "Load older sounds"}
              </button>
            </div>
          )}
          {events.length >= CAP && (
            <div style={{ textAlign: "center", fontSize: 10.5, color: "var(--text-muted)", marginTop: 8 }}>
              Showing the {CAP} most recent — older events remain in the archive.
            </div>
          )}
        </>
        );
      })()}

    </div>
  );
}
