/**
 * Search results — describe someone and get PEOPLE back, grouped by who they
 * were.
 *
 * "blue top with a backpack", "Ravi at the front door", "unfamiliar, no hat".
 * The chips say what was understood; words the search couldn't use are named,
 * so an empty result is never a silent mystery. Gender and age work as search
 * words only — they are never printed on a person.
 *
 * The query, the camera filter and the clear button live in the panel's toolbar
 * (this was a whole tab with its own search box and its own range control), so
 * what is left here is the results wall and the two things only it can say:
 * what the parser understood, and whether it is answering "looks like this one"
 * instead of a description.
 */
import { useEffect, useState } from "react";
import { Users } from "lucide-react";
import { api, KnownPerson, PeopleSearchResult, PersonGroup, Visit } from "../../api";
import { useStore } from "../../store";
import { useShallow } from "zustand/react/shallow";
import { trackCropSrc } from "../../lib/eventThumb";
import { fmtWhen } from "../../lib/time";
import { whoLabel, cameraPath, OUTFIT_DOT } from "./shared";
import {
  CardGrid, Card, CardMedia, CardBadge, CardPill, CardFooter, CardTime, CardEmpty, CARD_MIN,
} from "../review/Card";
import { VisitSheet } from "./VisitSheet";
import styles from "../review/ReviewFeed.module.css";

/** Appearance matching is only meaningful while the clothes last — a body match
 *  three months old is noise, not a lead. Text search has no such limit. */
const SIMILAR_DAYS = 30;

export function PeopleSearch({ query, cams, cameraName, persons, similarTo, onClearSimilar, onFindSimilar, onCount, onChanged, showToast }: {
  /** Debounced, trimmed query from the panel's header box. */
  query: string;
  /** Camera ids from the toolbar dropdown; empty = every camera. */
  cams: number[];
  cameraName: (id: number) => string;
  persons: KnownPerson[];
  /** Track id when showing "looks like this person". */
  similarTo: string | null;
  onClearSimilar: () => void;
  onFindSimilar: (trackId: string) => void;
  /** Result count for the header status line; null while in flight. */
  onCount: (n: number | null) => void;
  onChanged: () => void;
  showToast: (msg: string, type?: "success" | "error" | "info") => void;
}) {
  const [res, setRes] = useState<PeopleSearchResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [open, setOpen] = useState<Visit | null>(null);
  const { streamInfo } = useStore(useShallow(s => ({ streamInfo: s.streamInfo })));

  // Typing a description leaves "looks like" mode.
  useEffect(() => { if (query && similarTo) onClearSimilar(); }, [query]); // eslint-disable-line react-hooks/exhaustive-deps

  const camsKey = cams.join(",");
  useEffect(() => {
    if (!similarTo && !query) { setRes(null); onCount(null); return; }
    let cancelled = false;
    setLoading(true);
    onCount(null);
    (similarTo
      ? api.findSimilarPerson(similarTo, SIMILAR_DAYS)
      : api.searchPeople(query, cams.length ? { cams } : undefined))
      .then(r => { if (!cancelled) { setRes(r); onCount(r.groups.length); } })
      .catch(e => {
        if (!cancelled) { showToast(`Search failed: ${String(e)}`, "error"); onCount(0); }
      })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [query, similarTo, camsKey]); // eslint-disable-line react-hooks/exhaustive-deps

  const openGroup = (g: PersonGroup) => setOpen({
    key: g.key, person_id: g.person_id, name: g.name, maybe_name: g.maybe_name,
    start: g.first_seen, end: g.last_seen, cameras: g.cameras,
    behaviours: Array.from(new Set(g.tracks.flatMap(t => t.behaviours))),
    tracks: [...g.tracks].sort((a, b) => a.started_at.localeCompare(b.started_at)),
  });

  const chips = res && (res.understood.length > 0 || res.ignored.length > 0);

  return (
    <>
      {(similarTo || chips) && (
        <div style={{
          flexShrink: 0, padding: "0 16px 8px", display: "flex",
          alignItems: "center", gap: 6, flexWrap: "wrap",
        }}>
          {similarTo && (<>
            <span style={{ fontSize: 12, color: "var(--text-secondary)" }}>
              People who look like the one you picked, by appearance · last {SIMILAR_DAYS} days
            </span>
            <button className={styles.chip} onClick={onClearSimilar}>Clear</button>
          </>)}
          {res?.understood.map(u => (
            <span key={u} className={`${styles.chip} ${styles.chipActive}`}>{u}</span>
          ))}
          {res && res.ignored.length > 0 && (
            <span style={{ fontSize: 11, color: "var(--text-tertiary)" }}
              title="These words need a semantic search model (Arsenal → Search).">
              couldn't use: {res.ignored.join(", ")}
            </span>
          )}
        </div>
      )}

      <CardGrid min={CARD_MIN}>
        {loading && !res ? (
          <CardEmpty icon={<Users size={32} />}>Searching…</CardEmpty>
        ) : !res || res.groups.length === 0 ? (
          <CardEmpty icon={<Users size={32} />}>
            No one matched. Try fewer words, or drop a camera filter.
          </CardEmpty>
        ) : res.groups.map(g => {
          const best = g.tracks[0];
          const crop = trackCropSrc(best?.id, streamInfo);
          const colours = [best?.top_color, best?.bottom_color].filter(Boolean) as string[];
          const camLabel = g.cameras.length > 1
            ? `${cameraName(g.cameras[0])} +${g.cameras.length - 1}`
            : cameraName(g.cameras[0] ?? 0);
          return (
            <Card key={g.key} onClick={() => openGroup(g)}
              title={`${whoLabel(g)} · ${g.tracks.length} sighting${g.tracks.length === 1 ? "" : "s"} · ${cameraPath(g.cameras, cameraName)}`}>
              <CardMedia src={crop} aspect="3 / 4" fallback={<Users size={20} />}>
                <CardBadge>{camLabel}</CardBadge>
                {g.tracks.length > 1 && (
                  <CardPill color="var(--status-idle)">×{g.tracks.length}</CardPill>
                )}
              </CardMedia>
              <div style={{
                padding: "0 3px", fontSize: 12,
                fontWeight: g.name ? 700 : 600,
                color: g.name ? "var(--text-primary)" : "var(--text-secondary)",
                whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
              }}>{whoLabel(g)}</div>
              {(colours.length > 0 || best?.evidence.length) && (
                <div style={{
                  padding: "0 3px", display: "flex", alignItems: "center", gap: 4,
                  fontSize: 9.5, color: "var(--text-tertiary)", overflow: "hidden", whiteSpace: "nowrap",
                }}>
                  {colours.map(c => (
                    <span key={c} title={c} style={{
                      width: 8, height: 8, borderRadius: 999, flexShrink: 0,
                      background: OUTFIT_DOT[c as keyof typeof OUTFIT_DOT] ?? "var(--text-muted)",
                      border: "1px solid rgb(var(--ink) / 0.25)",
                    }} />
                  ))}
                  <span style={{ overflow: "hidden", textOverflow: "ellipsis" }}>
                    {best?.evidence.slice(0, 2).join(" · ")}
                  </span>
                </div>
              )}
              <CardFooter left={<CardTime>{fmtWhen(g.last_seen)}</CardTime>} />
            </Card>
          );
        })}
      </CardGrid>

      {open && (
        <VisitSheet visit={open} cameraName={cameraName} persons={persons}
          onClose={() => setOpen(null)}
          onChanged={onChanged}
          onFindSimilar={id => { setOpen(null); onFindSimilar(id); }}
          showToast={showToast} />
      )}
    </>
  );
}
