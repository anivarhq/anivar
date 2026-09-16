/**
 * One person, everything known about them: when they tend to be here (the hour
 * histogram), their visits over 30 days (continuous stays across cameras, each
 * playable), and the face gallery recognition matches against.
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import { Camera, Clock, MapPin, Play, Trash2, Pencil, Check, X, UserPlus, Activity } from "lucide-react";
import { api, KnownPerson, FaceShot, PersonStats, TrackHit, Visit } from "../../api";
import { useStore } from "../../store";
import { useShallow } from "zustand/react/shallow";
import { trackCropSrc } from "../../lib/eventThumb";
import { fmtWhen, localDateStr } from "../../lib/time";
import { useDismiss } from "../../components/ui/Modal";
import { RemovePersonDialog, Stat, Section, Empty, ZoomableImg, formatRelative, timeSpan, cameraPath, behaviourPhrase, fmtLocalDay } from "./shared";
import { ActivityPattern } from "./PersonInsights";

export function PersonDetail({ person, stats, cameraName, onClose, onDeleted, onAddAngles, showToast }: {
  person: KnownPerson;
  /** This person's 30-day pattern. The panel already fetches every person's
   *  stats for the roster cards, so re-fetching one here would be a second
   *  round-trip for data sitting in memory. */
  stats?: PersonStats;
  cameraName: (id: number) => string;
  onClose: () => void;
  onDeleted: () => void;
  onAddAngles: () => void;
  showToast: (msg: string, type?: "success" | "error" | "info") => void;
}) {
  const [faces, setFaces] = useState<FaceShot[]>([]);
  // Visits — continuous stays across cameras, bound by person id (a rename can't
  // lose or mix them up). Replaced the per-event list + name-keyed movement trail.
  const [visits, setVisits] = useState<Visit[]>([]);
  const [playing, setPlaying] = useState<TrackHit | null>(null);
  const [confirmRemove, setConfirmRemove] = useState(false);
  const { streamInfo } = useStore(useShallow(s => ({ streamInfo: s.streamInfo })));
  const [loading, setLoading] = useState(true);
  // Inline rename (pencil → input): fixing a typo'd name no longer requires
  // delete + re-enroll. Sightings/history follow the rename server-side.
  const [editing, setEditing] = useState(false);
  const [editName, setEditName] = useState(person.name);
  const [displayName, setDisplayName] = useState(person.name);
  const [savingName, setSavingName] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const [f, v] = await Promise.all([
        api.listPersonFaces(person.id).catch(() => [] as FaceShot[]),
        api.getPeopleDay(new Date(Date.now() - 30 * 86_400_000).toISOString(), new Date().toISOString(), person.id)
          .then(d => d.visits).catch(() => [] as Visit[]),
      ]);
      setFaces(f);
      setVisits(v);
    } finally { setLoading(false); }
  }, [person.id]);
  useEffect(() => { load(); }, [load]);

  const saveName = async () => {
    const next = editName.trim();
    if (!next || next === displayName) { setEditing(false); setEditName(displayName); return; }
    setSavingName(true);
    try {
      await api.renamePerson(person.id, next);
      setDisplayName(next);
      setEditing(false);
      showToast(`Renamed to ${next}`, "success");
      load(); // id-bound history — a rename can't lose or mix up the events

    } catch (e: any) {
      showToast(String(e?.message ?? e ?? "Rename failed"), "error");
    } finally { setSavingName(false); }
  };

  // Server-computed — listings no longer ship the embeddings JSON.
  const angles = person.embedding_count;
  const cameras = useMemo(
    () => Array.from(new Set(visits.flatMap(v => v.cameras))).sort((a, b) => a - b), [visits]);

  const removeShot = async (id: string) => {
    setFaces(fs => fs.filter(f => f.id !== id));
    try { await api.deleteFaceEmbedding(id); }
    catch { showToast("Couldn't remove that shot", "error"); load(); }
  };
  // "Not this person" — the only place a wrong recognition is fixed now that the
  // roster's confirm queues are gone. Unlike Remove, it TEACHES: a hard negative
  // for this person, the sample dropped from their gallery, a retrain.
  const notThem = async (id: string) => {
    setFaces(fs => fs.filter(f => f.id !== id));
    try {
      await api.correctFace(id, null);
      showToast(`Marked as not ${displayName} — recognition learns from it`, "success");
    } catch { showToast("Couldn't correct that shot", "error"); load(); }
  };

  const roleColor: Record<string, string> = {
    resident: "var(--accent)", family: "var(--accent)",
    employee: "var(--accent-amber)", visitor: "var(--status-idle)",
  };

  useDismiss(onClose);

  return (
    <div onClick={onClose} style={{
      position: "fixed", inset: 0, zIndex: 1000, background: "rgba(5,4,4,0.65)",
      display: "flex", alignItems: "center", justifyContent: "center", padding: 20,
    }}>
      <div className="glass-strong" onClick={e => e.stopPropagation()} style={{
        width: "100%", maxWidth: 640, maxHeight: "88vh", overflow: "auto",
        padding: 22, display: "flex", flexDirection: "column", gap: 18,
      }}>
        {/* Header */}
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          {person.thumbnail ? (
            <ZoomableImg src={person.thumbnail.startsWith("data:") ? person.thumbnail : `data:image/jpeg;base64,${person.thumbnail}`}
              alt={person.name} caption={person.name}
              style={{ width: 64, height: 64, borderRadius: 16, objectFit: "cover" }} />
          ) : (
            <div style={{ width: 64, height: 64, borderRadius: 16, background: "rgb(var(--ink) / 0.05)",
              display: "flex", alignItems: "center", justifyContent: "center", fontSize: 30 }}>👤</div>
          )}
          <div style={{ flex: 1, minWidth: 0 }}>
            {editing ? (
              <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <input
                  autoFocus
                  value={editName}
                  onChange={e => setEditName(e.target.value)}
                  onKeyDown={e => {
                    if (e.key === "Enter") saveName();
                    if (e.key === "Escape") { setEditing(false); setEditName(displayName); }
                  }}
                  disabled={savingName}
                  style={{
                    fontWeight: 800, fontSize: 16, letterSpacing: -0.02,
                    background: "rgb(var(--ink) / 0.06)", color: "inherit",
                    border: "1px solid rgb(var(--ink) / 0.18)", borderRadius: 8,
                    padding: "4px 8px", width: "100%", maxWidth: 220,
                  }}
                />
                <button onClick={saveName} disabled={savingName}
                  style={{ fontSize: 11, fontWeight: 700, padding: "5px 10px", borderRadius: 8,
                    border: "1px solid var(--accent)", background: "var(--accent-glow)",
                    color: "var(--accent)", cursor: "pointer" }}>
                  {savingName ? "…" : "Save"}
                </button>
              </div>
            ) : (
              <div style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 0 }}>
                <div style={{ fontWeight: 800, fontSize: 18, letterSpacing: -0.02,
                  overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{displayName}</div>
                <button onClick={() => { setEditName(displayName); setEditing(true); }}
                  title="Rename"
                  style={{ background: "none", border: "none", cursor: "pointer",
                    color: "var(--text-tertiary)", padding: 2, display: "inline-flex" }}>
                  <Pencil size={13} />
                </button>
              </div>
            )}
            <div style={{ fontSize: 11, fontWeight: 700, textTransform: "uppercase", letterSpacing: 0.04,
              color: roleColor[person.role] ?? "var(--text-tertiary)" }}>
              {person.role}
              {angles === 0 && (
                <span title="Named from body appearance only — add face angles to enable face recognition"
                  style={{ marginLeft: 8, padding: "2px 8px", borderRadius: 999, fontSize: 9.5,
                    background: "color-mix(in srgb, var(--status-idle) 15%, transparent)", color: "var(--status-idle)",
                    border: "1px solid color-mix(in srgb, var(--status-idle) 40%, transparent)", textTransform: "none", letterSpacing: 0 }}>
                  body-only
                </span>
              )}
            </div>
          </div>
          <button onClick={onClose} style={{ background: "none", border: "none", cursor: "pointer", color: "var(--text-tertiary)", padding: 4 }}>
            <X size={18} />
          </button>
        </div>

        {/* Stats */}
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
          <Stat label="Angles" value={String(angles)} />
          <Stat label="Visits · 30 days" value={String(visits.length)} />
          <Stat label="Cameras" value={cameras.length ? cameras.map(c => c + 1).join(", ") : "—"} />
          <Stat label="Last seen" value={person.last_seen_at ? formatRelative(new Date(person.last_seen_at)) : "never"} />
        </div>

        {/* WHEN — the hour histogram the backend always computed and never shipped. */}
        <Section icon={<Activity size={12} />} title="Activity"
          more="Sightings by hour of day over the last 30 days. The shape is the point: a tall single bar is one visit, a broad evening block is a routine.">
          <ActivityPattern stats={stats} />
        </Section>

        {/* WHERE and WHEN, as visits: one row per continuous stay, with the
            cameras passed in order. Replaced the name-keyed movement trail and
            the per-event list, which showed one person forty times. */}
        <Section icon={<MapPin size={12} />} title={`Visits (${visits.length})`}
          more="Each visit is one continuous stay, across cameras, in the last 30 days. Tap one to play its first recorded moment.">
          {visits.length === 0 ? (
            <Empty text={loading ? "Loading…" : "No visits in the last 30 days"} />
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              {Array.from(
                visits.reduce((days, v) => {
                  const day = localDateStr(new Date(v.start));
                  (days.get(day) ?? days.set(day, []).get(day)!).push(v);
                  return days;
                }, new Map<string, Visit[]>()),
              ).map(([day, vs]) => (
                <div key={day}>
                  <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: 0.05, textTransform: "uppercase",
                    color: "var(--text-tertiary)", margin: "0 2px 6px" }}>
                    {fmtLocalDay(day)} · {vs.length} visit{vs.length === 1 ? "" : "s"}
                  </div>
                  <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                    {vs.map(v => {
                      const playable = v.tracks.find(t => t.event_id) ?? null;
                      const crop = trackCropSrc(v.tracks[0]?.id, streamInfo);
                      return (
                        <button key={`${v.key}-${v.start}`} type="button" disabled={!playable}
                          onClick={() => playable && setPlaying(playable)}
                          title={playable ? "Play this visit" : "No recording linked to this visit"}
                          style={{ display: "flex", gap: 10, alignItems: "center", padding: 8, textAlign: "left",
                            borderRadius: 12, background: "rgb(var(--ink) / 0.02)", border: "1px solid var(--border)",
                            cursor: playable ? "pointer" : "default", width: "100%", color: "inherit" }}>
                          {crop ? (
                            <img src={crop} alt="" onError={e => { e.currentTarget.style.visibility = "hidden"; }}
                              style={{ width: 30, height: 40, borderRadius: 7, objectFit: "cover", flexShrink: 0, background: "#000" }} />
                          ) : (
                            <div style={{ width: 30, height: 40, borderRadius: 7, flexShrink: 0, background: "rgb(var(--ink) / 0.05)" }} />
                          )}
                          <div style={{ minWidth: 0, flex: 1 }}>
                            <div style={{ fontSize: 12, color: "var(--text-secondary)", overflow: "hidden",
                              textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                              {timeSpan(v.start, v.end)} · {cameraPath(v.cameras, cameraName)}
                            </div>
                            {v.behaviours.length > 0 && (
                              <div style={{ fontSize: 10.5, color: "var(--accent-amber)", marginTop: 2 }}>
                                {v.behaviours.map(behaviourPhrase).join(" · ")}
                              </div>
                            )}
                          </div>
                          {playable && <Play size={13} style={{ opacity: 0.5, flexShrink: 0 }} />}
                        </button>
                      );
                    })}
                  </div>
                </div>
              ))}
            </div>
          )}
        </Section>

        {/* Face gallery */}
        <Section icon={<Camera size={12} />} title={`Face gallery (${faces.length})`}
          hint={`Not ${displayName}? Mark it and recognition learns. Blurry? Remove it.`}>
          {faces.length === 0 ? (
            <Empty text={loading ? "Loading…" : "No sightings yet"} />
          ) : (
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(84px, 1fr))", gap: 8 }}>
              {faces.map(f => (
                <div key={f.id} style={{ position: "relative", borderRadius: 12, overflow: "hidden", aspectRatio: "1/1" }}>
                  <ZoomableImg src={`data:image/jpeg;base64,${f.thumbnail_b64}`}
                    caption={`${cameraName(f.cam_id)} · ${fmtWhen(f.seen_at)}`}
                    style={{ width: "100%", height: "100%", objectFit: "cover" }} />
                  <button onClick={() => removeShot(f.id)} title="Remove this shot (blurry or unusable)" style={{
                    position: "absolute", top: 4, right: 4, width: 20, height: 20, borderRadius: 999,
                    border: "none", cursor: "pointer", background: "rgba(0,0,0,0.6)", color: "#fff",
                    display: "flex", alignItems: "center", justifyContent: "center" }}>
                    <X size={11} />
                  </button>
                  <button onClick={() => notThem(f.id)} title={`This isn't ${displayName}`} style={{
                    position: "absolute", bottom: 4, left: 4, right: 4, padding: "2px 0", borderRadius: 999,
                    border: "none", cursor: "pointer", background: "rgba(0,0,0,0.6)", color: "#fff",
                    fontSize: 9, fontWeight: 700 }}>
                    Not them
                  </button>
                </div>
              ))}
            </div>
          )}
        </Section>

        {/* Actions */}
        <div style={{ display: "flex", gap: 10 }}>
          <button onClick={onAddAngles} className="btn-primary" style={{ flex: 1, padding: "10px 0" }}>
            <UserPlus size={13} /> Add more angles
          </button>
          <button onClick={() => setConfirmRemove(true)} style={{
            padding: "10px 16px", borderRadius: 999, border: "1px solid color-mix(in srgb, var(--status-alert) 25%, transparent)",
            background: "transparent", color: "var(--accent-red)", cursor: "pointer", fontSize: 13, fontWeight: 600,
            display: "inline-flex", alignItems: "center", gap: 6 }}>
            <Trash2 size={13} /> Remove
          </button>
        </div>
      </div>

      {confirmRemove && (
        <RemovePersonDialog
          person={{ id: person.id, name: person.name }}
          showToast={showToast}
          onCancel={() => setConfirmRemove(false)}
          onDone={() => { setConfirmRemove(false); onDeleted(); }} />
      )}

      {/* Event clip player — server NVR slice via /footage/:id/clip (same pattern
          as the Vehicles/Audio players; never gated on clip_path). */}
      {playing?.event_id && streamInfo && (
        <div onClick={e => { e.stopPropagation(); setPlaying(null); }} style={{
          position: "fixed", inset: 0, zIndex: 1300,
          background: "rgba(5,4,4,0.78)", display: "flex", alignItems: "center", justifyContent: "center", padding: 24,
        }}>
          <div onClick={e => e.stopPropagation()} style={{ width: "100%", maxWidth: 860 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
              <span style={{ fontWeight: 700, fontSize: 13, color: "#fff" }}>
                {displayName} · {cameraName(playing.cam_id)} · {fmtWhen(playing.started_at)}
              </span>
              <div style={{ flex: 1 }} />
              <button onClick={() => setPlaying(null)}
                style={{ background: "none", border: "none", color: "#fff", cursor: "pointer", padding: 4 }}>
                <X size={18} />
              </button>
            </div>
            <video
              src={`http://localhost:${streamInfo.port}/footage/${playing.event_id}/clip?token=${streamInfo.auth_token}`}
              controls autoPlay
              style={{ width: "100%", borderRadius: 14, background: "#000" }} />
          </div>
        </div>
      )}
    </div>
  );
}
