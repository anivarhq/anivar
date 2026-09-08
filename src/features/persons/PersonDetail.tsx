/**
 * One person, everything known about them.
 *
 * The gallery and the 30-day event history were always here. What was not:
 * where they move (`get_camera_correlations`, fully wired and called by
 * nothing), and WHEN they show up — the backend built a 24-bucket hour
 * histogram and shipped only its argmax, so the UI could say "most often
 * around 18:00" but never tell "home every evening" from "here once".
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import { Camera, Clock, MapPin, Play, Trash2, Pencil, Check, X, UserPlus, Activity } from "lucide-react";
import { api, KnownPerson, FaceShot, PersonEvent, PersonStats } from "../../api";
import { useStore } from "../../store";
import { useShallow } from "zustand/react/shallow";
import { faceCropSrc, eventThumbSrc } from "../../lib/eventThumb";
import { fmtWhen } from "../../lib/time";
import { Modal, useDismiss } from "../../components/ui/Modal";
import { CardGrid, Card, CardMedia, CardFooter, CardTime, CardEmpty } from "../review/Card";
import { RemovePersonDialog, Stat, Section, Empty, ZoomableImg, summaryText, formatRelative, fmtDay } from "./shared";
import { ActivityPattern, MovementTrail } from "./PersonInsights";

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
  // standard person-events: ID-BOUND event history (face_sightings /
  // face_embeddings by person_id) — replaced the old fuzzy name-LIKE search
  // that could show someone else's events or miss renamed people entirely.
  const [history, setHistory] = useState<PersonEvent[]>([]);
  const [playing, setPlaying] = useState<PersonEvent | null>(null);
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
      const [f, h] = await Promise.all([
        api.listPersonFaces(person.id).catch(() => [] as FaceShot[]),
        api.getPersonEvents({ personId: person.id, days: 30, limit: 100 }).catch(() => [] as PersonEvent[]),
      ]);
      setFaces(f);
      setHistory(h);
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
    () => Array.from(new Set(faces.map(f => f.cam_id))).sort((a, b) => a - b), [faces]);

  const removeShot = async (id: string) => {
    setFaces(fs => fs.filter(f => f.id !== id));
    try { await api.deleteFaceEmbedding(id); }
    catch { showToast("Couldn't remove that shot", "error"); load(); }
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
          <Stat label="Sightings" value={String(faces.length)} />
          <Stat label="Cameras" value={cameras.length ? cameras.map(c => c + 1).join(", ") : "—"} />
          <Stat label="Last seen" value={person.last_seen_at ? formatRelative(new Date(person.last_seen_at)) : "never"} />
        </div>

        {/* WHEN — the hour histogram the backend always computed and never shipped. */}
        <Section icon={<Activity size={12} />} title="Activity"
          more="Sightings by hour of day over the last 30 days. The shape is the point: a tall single bar is one visit, a broad evening block is a routine.">
          <ActivityPattern stats={stats} />
        </Section>

        {/* WHERE — get_camera_correlations, wired since it was written and never called. */}
        <Section icon={<MapPin size={12} />} title="Movement"
          more="Which cameras saw this person, in order. Repeated sightings on the same camera collapse into one hop, so the path is the path and not a list of frames.">
          <MovementTrail
            personName={displayName}
            cameraName={cameraName}
            onOpenEvent={(id) => {
              const ev = history.find(h => h.event.id === id);
              if (ev) setPlaying(ev);
              else showToast("That event is outside this person's 30-day history", "info");
            }} />
        </Section>

        {/* Face gallery */}
        <Section icon={<Camera size={12} />} title={`Face gallery (${faces.length})`}
          more="What the cameras matched to this person. Remove blurry shots to keep recognition sharp.">
          {faces.length === 0 ? (
            <Empty text={loading ? "Loading…" : "No sightings yet"} />
          ) : (
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(84px, 1fr))", gap: 8 }}>
              {faces.map(f => (
                <div key={f.id} style={{ position: "relative", borderRadius: 12, overflow: "hidden", aspectRatio: "1/1" }}>
                  <ZoomableImg src={`data:image/jpeg;base64,${f.thumbnail_b64}`}
                    caption={`CAM ${f.cam_id + 1} · ${fmtWhen(f.seen_at)} · quality ${Math.round(f.quality * 100)}`}
                    style={{ width: "100%", height: "100%", objectFit: "cover" }} />
                  <button onClick={() => removeShot(f.id)} title="Remove this shot" style={{
                    position: "absolute", top: 4, right: 4, width: 20, height: 20, borderRadius: 999,
                    border: "none", cursor: "pointer", background: "rgba(0,0,0,0.6)", color: "#fff",
                    display: "flex", alignItems: "center", justifyContent: "center" }}>
                    <X size={11} />
                  </button>
                  <div style={{ position: "absolute", bottom: 3, left: 4, fontSize: 8,
                    color: "rgb(var(--ink) / 0.85)", textShadow: "0 1px 2px rgba(0,0,0,0.8)" }}>
                    CAM {f.cam_id + 1}
                  </div>
                </div>
              ))}
            </div>
          )}
        </Section>

        {/* Events timeline (mature NVRs person-events): id-bound, grouped by day,
            each row playable. */}
        <Section icon={<Clock size={12} />} title={`Events (${history.length})`}
          more="Video events this person appears in.">
          {history.length === 0 ? (
            <Empty text={loading ? "Loading…" : "No events in the last 30 days"} />
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              {Array.from(
                history.reduce((days, h) => {
                  const day = h.event.started_at.slice(0, 10);
                  (days.get(day) ?? days.set(day, []).get(day)!).push(h);
                  return days;
                }, new Map<string, PersonEvent[]>()),
              ).map(([day, evs]) => (
                <div key={day}>
                  <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: 0.05, textTransform: "uppercase",
                    color: "var(--text-tertiary)", margin: "0 2px 6px" }}>
                    {fmtDay(day)} · {evs.length} event{evs.length === 1 ? "" : "s"}
                  </div>
                  <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                    {evs.map(h => {
                      const thumb = h.event.thumbnail ? eventThumbSrc(h.event.thumbnail, h.event.id, streamInfo) : null;
                      return (
                      <button key={h.event.id} type="button" onClick={() => setPlaying(h)}
                        title="Play this event's clip"
                        style={{ display: "flex", gap: 10, alignItems: "center", padding: 8, textAlign: "left",
                          borderRadius: 12, background: "rgb(var(--ink) / 0.02)", border: "1px solid var(--border)",
                          cursor: "pointer", width: "100%" }}>
                        <div style={{ position: "relative", flexShrink: 0 }}>
                          {thumb ? (
                            <img src={thumb} alt=""
                              style={{ width: 52, height: 38, borderRadius: 8, objectFit: "cover", display: "block" }} />
                          ) : (
                            <div style={{ width: 52, height: 38, borderRadius: 8, background: "rgb(var(--ink) / 0.05)",
                              display: "flex", alignItems: "center", justifyContent: "center" }}>
                              <Play size={13} style={{ opacity: 0.5 }} />
                            </div>
                          )}
                          {/* mature NVRs object-crop: THIS person, in THIS event. */}
                          {h.person_crop && (
                            <img src={`data:image/jpeg;base64,${h.person_crop}`} alt=""
                              title="This person, in this event"
                              style={{ position: "absolute", right: -5, bottom: -5, width: 22, height: 22,
                                borderRadius: 7, objectFit: "cover", border: "1.5px solid var(--accent)",
                                background: "#000" }} />
                          )}
                        </div>
                        <div style={{ minWidth: 0, flex: 1 }}>
                          <div style={{ fontSize: 11, color: "var(--text-secondary)", overflow: "hidden",
                            textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{summaryText(h.event.ai_summary)}</div>
                          <div style={{ fontSize: 10, color: "var(--text-tertiary)", marginTop: 2, display: "flex", gap: 6, alignItems: "center" }}>
                            <span>CAM {(h.event.cam_id ?? 0) + 1}</span>
                            <span>·</span>
                            <span>{fmtWhen(h.event.started_at)}</span>
                            {h.event.event_category && h.event.event_category !== "other" && (<><span>·</span><span>{h.event.event_category}</span></>)}
                          </div>
                        </div>
                      </button>
                    );})}
                  </div>
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
      {playing && streamInfo && (
        <div onClick={e => { e.stopPropagation(); setPlaying(null); }} style={{
          position: "fixed", inset: 0, zIndex: 1300,
          background: "rgba(5,4,4,0.78)", display: "flex", alignItems: "center", justifyContent: "center", padding: 24,
        }}>
          <div onClick={e => e.stopPropagation()} style={{ width: "100%", maxWidth: 860 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
              <span style={{ fontWeight: 700, fontSize: 13, color: "#fff" }}>
                {displayName} · CAM {(playing.event.cam_id ?? 0) + 1} · {fmtWhen(playing.event.started_at)}
              </span>
              <div style={{ flex: 1 }} />
              <button onClick={() => setPlaying(null)}
                style={{ background: "none", border: "none", color: "#fff", cursor: "pointer", padding: 4 }}>
                <X size={18} />
              </button>
            </div>
            <video
              src={`http://localhost:${streamInfo.port}/footage/${playing.event.id}/clip?token=${streamInfo.auth_token}`}
              controls autoPlay
              style={{ width: "100%", borderRadius: 14, background: "#000" }} />
          </div>
        </div>
      )}
    </div>
  );
}
