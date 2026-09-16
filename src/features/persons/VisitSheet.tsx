/**
 * One visit — or one searched person — with the evidence behind every claim:
 * the cameras in order with a crop from each, what they wore and carried, what
 * they did, and why they have (or don't have) a name. Then the three things you
 * can do about it: play it, find them again, or say who they are.
 */
import { useState, type ReactNode } from "react";
import { Play, Search as SearchIcon, UserCheck, X } from "lucide-react";
import { api, KnownPerson, TrackHit, Visit } from "../../api";
import { useStore } from "../../store";
import { useShallow } from "zustand/react/shallow";
import { trackCropSrc } from "../../lib/eventThumb";
import { fmtWhen } from "../../lib/time";
import { useDismiss } from "../../components/ui/Modal";
import { PersonPickList, whoLabel, timeSpan, cameraPath, behaviourPhrase, OUTFIT_DOT } from "./shared";

export function VisitSheet({ visit, cameraName, persons, onClose, onChanged, onFindSimilar, showToast }: {
  visit: Visit;
  cameraName: (id: number) => string;
  persons: KnownPerson[];
  onClose: () => void;
  onChanged: () => void;
  onFindSimilar: (trackId: string) => void;
  showToast: (msg: string, type?: "success" | "error" | "info") => void;
}) {
  const [playing, setPlaying] = useState<TrackHit | null>(null);
  const [naming, setNaming] = useState(false);
  const [busy, setBusy] = useState(false);
  const { streamInfo } = useStore(useShallow(s => ({ streamInfo: s.streamInfo })));
  useDismiss(() => (playing ? setPlaying(null) : onClose()));

  const top = visit.tracks.find(t => t.top_color)?.top_color ?? null;
  const bottom = visit.tracks.find(t => t.bottom_color)?.bottom_color ?? null;
  const evidence = Array.from(new Set(visit.tracks.flatMap(t => t.evidence)));
  const playable = visit.tracks.find(t => t.event_id) ?? null;
  // Anonymous body identities in this visit — what "This is…" binds to a person,
  // through the same command the body-match queue uses.
  const bodyIds = Array.from(new Set(visit.tracks
    .map(t => t.body_person_id).filter((b): b is string => !!b && b.startsWith("body_"))));
  const claimsName = !!(visit.name || visit.maybe_name);
  const why = visit.name
    ? "Recognised by face."
    : visit.maybe_name
      ? `Looks like ${visit.maybe_name} by body appearance — not confirmed. A face, or you, confirms it.`
      : "Not recognised.";

  const assign = async (p: KnownPerson) => {
    setBusy(true);
    try {
      await api.assignTrackedToKnown(bodyIds, p.id);
      showToast(`Marked as ${p.name}`, "success");
      onChanged();
      onClose();
    } catch (e) {
      showToast(String(e), "error");
    } finally { setBusy(false); }
  };

  return (
    <div onClick={onClose} style={{
      position: "fixed", inset: 0, zIndex: 1100, background: "rgba(5,4,4,0.65)",
      display: "flex", alignItems: "center", justifyContent: "center", padding: 20,
    }}>
      <div className="glass-strong" onClick={e => e.stopPropagation()} style={{
        width: "100%", maxWidth: 620, maxHeight: "88vh", overflow: "auto",
        padding: 20, display: "flex", flexDirection: "column", gap: 14,
      }}>
        <div style={{ display: "flex", alignItems: "flex-start", gap: 12 }}>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ fontSize: 16, fontWeight: 800, fontStyle: visit.maybe_name && !visit.name ? "italic" : "normal" }}>
              {whoLabel(visit)}
            </div>
            <div style={{ fontSize: 12, color: "var(--text-tertiary)", marginTop: 2 }}>
              {fmtWhen(visit.start)} · {timeSpan(visit.start, visit.end)} · {cameraPath(visit.cameras, cameraName)}
            </div>
          </div>
          <button type="button" onClick={onClose} title="Close"
            style={{ background: "none", border: "none", cursor: "pointer", color: "var(--text-tertiary)", padding: 4 }}>
            <X size={18} />
          </button>
        </div>

        {/* The hops, in order. Dashed = linked to the name by appearance only. */}
        <div style={{ display: "flex", gap: 10, overflowX: "auto", paddingBottom: 4 }}>
          {visit.tracks.map(t => {
            const crop = trackCropSrc(t.id, streamInfo);
            const dashed = claimsName && t.identity_method !== "face";
            return (
              <button key={t.id} type="button" disabled={!t.event_id} onClick={() => setPlaying(t)}
                title={t.event_id ? "Play this moment" : "No recording linked"}
                style={{ flexShrink: 0, width: 92, padding: 0, background: "none", border: "none",
                  textAlign: "left", cursor: t.event_id ? "pointer" : "default", color: "inherit" }}>
                <div style={{ width: 92, height: 124, borderRadius: 12, overflow: "hidden", position: "relative",
                  background: "#000", border: `1.5px ${dashed ? "dashed" : "solid"} var(--border-strong)` }}>
                  {crop && <img src={crop} alt="" onError={e => { e.currentTarget.style.visibility = "hidden"; }}
                    style={{ width: "100%", height: "100%", objectFit: "cover" }} />}
                  {t.event_id && <Play size={13} style={{ position: "absolute", right: 6, bottom: 6, color: "#fff", opacity: 0.85 }} />}
                </div>
                <div style={{ fontSize: 11, fontWeight: 600, marginTop: 4, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {cameraName(t.cam_id)}
                </div>
                <div style={{ fontSize: 10, color: "var(--text-tertiary)" }}>{timeSpan(t.started_at, t.ended_at)}</div>
              </button>
            );
          })}
        </div>

        {(top || bottom || evidence.length > 0 || visit.behaviours.length > 0) && (
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
            {top && <Chip dot={top}>{top} top</Chip>}
            {bottom && <Chip dot={bottom}>{bottom} bottom</Chip>}
            {evidence.map(e => <Chip key={e}>{e}</Chip>)}
            {visit.behaviours.map(b => <Chip key={b} warn>{behaviourPhrase(b)}</Chip>)}
          </div>
        )}
        <div style={{ fontSize: 11, color: "var(--text-tertiary)" }}>{why}</div>

        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          <button type="button" className="btn-primary" disabled={!playable}
            onClick={() => playable && setPlaying(playable)}
            style={{ padding: "8px 16px", display: "inline-flex", alignItems: "center", gap: 6, opacity: playable ? 1 : 0.5 }}>
            <Play size={13} /> Play
          </button>
          {visit.tracks[0] && (
            <button type="button" onClick={() => onFindSimilar(visit.tracks[0].id)} style={secondaryBtn}>
              <SearchIcon size={13} /> Find similar
            </button>
          )}
          {!visit.name && bodyIds.length > 0 && (
            <button type="button" onClick={() => setNaming(n => !n)} style={secondaryBtn}>
              <UserCheck size={13} /> This is…
            </button>
          )}
        </div>
        {naming && <PersonPickList persons={persons} onPick={assign} disabled={busy} />}
      </div>

      {playing?.event_id && streamInfo && (
        <div onClick={e => { e.stopPropagation(); setPlaying(null); }} style={{
          position: "fixed", inset: 0, zIndex: 1300, background: "rgba(5,4,4,0.78)",
          display: "flex", alignItems: "center", justifyContent: "center", padding: 24,
        }}>
          <div onClick={e => e.stopPropagation()} style={{ width: "100%", maxWidth: 860 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
              <span style={{ fontWeight: 700, fontSize: 13, color: "#fff" }}>
                {whoLabel(visit)} · {cameraName(playing.cam_id)} · {fmtWhen(playing.started_at)}
              </span>
              <div style={{ flex: 1 }} />
              <button type="button" onClick={() => setPlaying(null)}
                style={{ background: "none", border: "none", color: "#fff", cursor: "pointer", padding: 4 }}>
                <X size={18} />
              </button>
            </div>
            <video src={`http://localhost:${streamInfo.port}/footage/${playing.event_id}/clip?token=${streamInfo.auth_token}`}
              controls autoPlay style={{ width: "100%", borderRadius: 14, background: "#000" }} />
          </div>
        </div>
      )}
    </div>
  );
}

function Chip({ children, dot, warn }: { children: ReactNode; dot?: string; warn?: boolean }) {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 5, fontSize: 11, padding: "3px 9px",
      borderRadius: 999, border: "1px solid var(--border)",
      color: warn ? "var(--accent-amber)" : "var(--text-secondary)" }}>
      {dot && <span style={{ width: 8, height: 8, borderRadius: 999, border: "1px solid rgb(var(--ink) / 0.25)",
        background: OUTFIT_DOT[dot as keyof typeof OUTFIT_DOT] ?? "var(--text-muted)" }} />}
      {children}
    </span>
  );
}

const secondaryBtn = {
  padding: "8px 14px", borderRadius: 999, border: "1px solid var(--border-strong)", background: "transparent",
  color: "var(--text-primary)", cursor: "pointer", fontSize: 13, display: "inline-flex", alignItems: "center", gap: 6,
} as const;
