/**
 * Today — everyone who was here, as VISITS (one continuous stay across
 * cameras), newest first, under a one-line summary.
 *
 * Visits, not detections: someone who walked Door → Hall → Garden is one tile,
 * not three events and forty frames. Identity reads in words ("Maybe Priya?",
 * "Unfamiliar"); the evidence behind it is one tap away in the visit sheet.
 *
 * The day and the camera filter belong to the panel's toolbar, so this view has
 * no chrome of its own — it is the wall of cards under it, exactly as Review's
 * feed sits under Review's toolbar.
 */
import { useCallback, useEffect, useState } from "react";
import { Users } from "lucide-react";
import { api, KnownPerson, PeopleDay, Visit } from "../../api";
import { useStore } from "../../store";
import { useShallow } from "zustand/react/shallow";
import { trackCropSrc } from "../../lib/eventThumb";
import { dayBoundsUtc, isLocalToday } from "../../lib/time";
import { whoLabel, timeSpan, cameraPath, behaviourPhrase } from "./shared";
import {
  CardGrid, Card, CardMedia, CardBadge, CardPill, CardFooter, CardTime, CardEmpty, CARD_MIN,
} from "../review/Card";
import { VisitSheet } from "./VisitSheet";

export function TodayView({ day, cams, cameraName, persons, onChanged, onFindSimilar, showToast }: {
  /** Local `YYYY-MM-DD`, owned by the panel's date button. */
  day: string;
  /** Camera ids (as strings) from the toolbar dropdown; empty = every camera. */
  cams: Set<string>;
  cameraName: (id: number) => string;
  persons: KnownPerson[];
  onChanged: () => void;
  onFindSimilar: (trackId: string) => void;
  showToast: (msg: string, type?: "success" | "error" | "info") => void;
}) {
  const [data, setData] = useState<PeopleDay | null>(null);
  const [loading, setLoading] = useState(false);
  const [open, setOpen] = useState<Visit | null>(null);
  const { streamInfo } = useStore(useShallow(s => ({ streamInfo: s.streamInfo })));

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const { fromUtc, toUtc } = dayBoundsUtc(day);
      setData(await api.getPeopleDay(fromUtc, toUtc));
    } catch (e) {
      showToast(`Couldn't load visits: ${String(e)}`, "error");
    } finally { setLoading(false); }
  }, [day, showToast]);
  useEffect(() => { load(); }, [load]);
  // A track is written a few seconds after someone leaves view — keep today current.
  useEffect(() => {
    if (!isLocalToday(day)) return;
    const t = setInterval(() => { if (!document.hidden) load(); }, 60_000);
    return () => clearInterval(t);
  }, [day, load]);

  const today = isLocalToday(day);
  const visits = (data?.visits ?? []).filter(v =>
    cams.size === 0 || v.cameras.some(c => cams.has(String(c))));

  const summary = (() => {
    if (!data) return loading ? "Loading…" : "";
    const { known_visits: k, unfamiliar_visits: u, unfamiliar_repeat: r } = data;
    if (k + u === 0) return "Nobody seen";
    const parts: string[] = [];
    if (k) parts.push(`${k} visit${k === 1 ? "" : "s"} by people you know`);
    if (u) parts.push(`${u} unfamiliar${r ? ` — ${r} came back more than once` : ""}`);
    return parts.join(" · ");
  })();

  return (
    <>
      {summary && (
        <div style={{ flexShrink: 0, padding: "0 16px 8px", fontSize: 12, color: "var(--text-secondary)" }}>
          {summary}
          {cams.size > 0 && visits.length !== (data?.visits.length ?? 0) && (
            <span style={{ color: "var(--text-tertiary)" }}> · {visits.length} on the selected cameras</span>
          )}
        </div>
      )}

      <CardGrid min={CARD_MIN}>
        {visits.length === 0 ? (
          <CardEmpty icon={<Users size={32} />}>
            {today
              ? "Nobody yet today. A visit appears a few seconds after someone leaves a camera's view."
              : "Nobody seen that day."}
          </CardEmpty>
        ) : visits.map(v => {
          const crop = trackCropSrc(v.tracks[0]?.id, streamInfo);
          const known = !!v.name;
          const path = cameraPath(v.cameras, cameraName);
          // One badge, not two: a camera badge and a hop count both live
          // top-right, so the second would sit on the first.
          const camLabel = v.cameras.length > 1
            ? `${cameraName(v.cameras[0])} +${v.cameras.length - 1}`
            : cameraName(v.cameras[0] ?? 0);
          return (
            <Card key={`${v.key}-${v.start}`} onClick={() => setOpen(v)}
              title={`${whoLabel(v)} · ${timeSpan(v.start, v.end)} · ${path}`}>
              <CardMedia src={crop} aspect="3 / 4" fallback={<Users size={20} />}>
                <CardBadge>{camLabel}</CardBadge>
                {v.behaviours.length > 0 && (
                  <CardPill color="var(--status-warn)">
                    {behaviourPhrase(v.behaviours[0])}
                    {v.behaviours.length > 1 ? ` +${v.behaviours.length - 1}` : ""}
                  </CardPill>
                )}
              </CardMedia>
              <div style={{
                padding: "0 3px", fontSize: 12,
                fontWeight: known ? 700 : 600,
                color: known ? "var(--text-primary)" : "var(--text-secondary)",
                fontStyle: v.maybe_name && !known ? "italic" : "normal",
                whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
              }}>{whoLabel(v)}</div>
              <CardFooter left={<CardTime>{timeSpan(v.start, v.end)}</CardTime>} />
            </Card>
          );
        })}
      </CardGrid>

      {open && (
        <VisitSheet visit={open} cameraName={cameraName} persons={persons}
          onClose={() => setOpen(null)}
          onChanged={() => { onChanged(); load(); }}
          onFindSimilar={id => { setOpen(null); onFindSimilar(id); }}
          showToast={showToast} />
      )}
    </>
  );
}
