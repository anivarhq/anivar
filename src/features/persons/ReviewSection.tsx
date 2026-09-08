/**
 * Needs your review — faces the cameras saw but could not identify.
 *
 * Two shapes: a CLUSTER (the same stranger seen repeatedly, grouped by
 * whole-gallery clustering) and a single unmatched face. A cluster is the more
 * useful object — it carries how often, over how many days, at what time of
 * day — so it leads.
 */
import { useCallback, useEffect, useMemo, useState, type CSSProperties } from "react";
import { Sparkles, Users, Check, X, Clock, MapPin, Play, Trash2, UserPlus } from "lucide-react";
import { api, KnownPerson, UnknownFace, UnknownCluster, PersonSighting } from "../../api";
import { useStore } from "../../store";
import { createPortal } from "react-dom";
import { faceCropSrc, eventThumbSrc } from "../../lib/eventThumb";
import { fmtWhen } from "../../lib/time";
import { Modal, useDismiss } from "../../components/ui/Modal";
import { CardGrid, Card, CardMedia, ProfileCard, ProfileMedia, CardCount, CardEmpty, CardFooter, CardTime } from "../review/Card";
import { PersonPickList, SectionHeader, FaceContextZoom, formatRelative, ZoomableImg, summaryText } from "./shared";

export function ReviewSection({ unknowns, clusters, persons, onTagged, showToast }: {
  unknowns: UnknownFace[];
  clusters: UnknownCluster[];
  persons:  KnownPerson[];
  onTagged: () => void;
  showToast: (msg: string, type?: "success" | "error" | "info") => void;
}) {
  const [active, setActive] = useState<UnknownFace | null>(null);
  const [activeCluster, setActiveCluster] = useState<UnknownCluster | null>(null);
  const streamInfo = useStore(s => s.streamInfo); // '@crop' markers → URL-served crops

  // Faces already grouped into a cluster shouldn't also show as singletons.
  const clusteredIds = useMemo(() => new Set(clusters.flatMap(c => c.face_ids)), [clusters]);
  const singletons = useMemo(() => unknowns.filter(u => !clusteredIds.has(u.id)), [unknowns, clusteredIds]);

  // Someone who keeps coming back is a finding; someone seen once is a chore.
  //
  // The backend has always returned `days_active`, `time_pattern` and
  // `first_seen` per cluster and this section rendered two of them, unsorted, as
  // a flat tagging queue. Ordering by distinct days puts the stranger who has
  // been here nine evenings in a row above the one who walked past on Tuesday.
  const [recurring, oneOff] = useMemo(() => {
    const sorted = clusters.slice().sort((a, b) =>
      b.days_active - a.days_active || b.count - a.count);
    return [sorted.filter(c => c.days_active >= 2), sorted.filter(c => c.days_active < 2)];
  }, [clusters]);

  // One-tap confirm of the agent's "looks like {name}" guess (mature NVRs confirm loop).
  // Tagging is a user action — a failure must be visible, not swallowed.
  const confirmCluster = useCallback(async (c: UnknownCluster, personId: string) => {
    try { await api.assignFacesToPerson(c.face_ids, personId); onTagged(); }
    catch (e) { showToast(`Couldn't tag: ${String(e).replace(/^.*Error:\s*/, "")}`, "error"); }
  }, [onTagged, showToast]);
  const confirmFace = useCallback(async (faceId: string, personId: string) => {
    try { await api.assignFaceToPerson(faceId, personId); onTagged(); }
    catch (e) { showToast(`Couldn't tag: ${String(e).replace(/^.*Error:\s*/, "")}`, "error"); }
  }, [onTagged, showToast]);
  // Wipe the stranger backlog (e.g. old low-quality crops) — cameras re-capture clean ones.
  const handleClear = async () => {
    if (!confirm("Remove ALL un-tagged stranger faces from Train? Cameras will re-capture clean ones, and enrolled people are kept.")) return;
    try { const n = await api.clearUnknownFaces(); onTagged(); showToast(`Cleared ${n} face${n === 1 ? "" : "s"}`, "info"); }
    catch (e) { showToast(`Clear failed: ${String(e).replace(/^.*Error:\s*/, "")}`, "error"); }
  };

  if (clusters.length === 0 && unknowns.length === 0) {
    return (
      <div className="glass" style={{
        padding: "40px 28px", textAlign: "center",
        display: "flex", flexDirection: "column", alignItems: "center", gap: 14,
      }}>
        <Sparkles size={36} style={{ opacity: 0.35 }} />
        <div>
          <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 6 }}>No unidentified faces in the last 30 days</div>
        </div>
      </div>
    );
  }

  const sectionLabel: CSSProperties = { fontSize: 11, fontWeight: 700, letterSpacing: 0.04, textTransform: "uppercase", color: "var(--text-tertiary)", margin: "0 2px 8px" };

  return (
    <div style={{ display: "grid", gap: 18 }}>
      {/* Hint banner */}
      <div className="glass-accent" style={{
        padding: "12px 16px", display: "flex", alignItems: "center", gap: 12,
        fontSize: 12, color: "var(--text-secondary)",
      }}>
        <Sparkles size={14} style={{ color: "var(--accent)" }} />
        <span style={{ flex: 1 }}>
          {/* Lead with the finding when there IS one. "3 people keep coming back"
              is worth reading; "you have 47 faces to tag" is a chore list. */}
          {recurring.length > 0
            ? <><strong style={{ color: "var(--accent)" }}>{recurring.length}</strong> unidentified {recurring.length === 1 ? "person has" : "people have"} been seen on more than one day. Naming one names every sighting of them at once.</>
            : clusters.length > 0
            ? <>The agent grouped <strong style={{ color: "var(--accent)" }}>{clusters.length}</strong> distinct {clusters.length === 1 ? "person" : "people"} it couldn't identify, none of them more than once.</>
            : <>The agent saw <strong style={{ color: "var(--accent)" }}>{singletons.length}</strong> unidentified face{singletons.length === 1 ? "" : "s"}. Tap one to tag it.</>}
        </span>
        <button onClick={handleClear} title="Remove all un-tagged stranger faces (cameras re-capture clean ones)"
          style={{ flexShrink: 0, display: "inline-flex", alignItems: "center", gap: 5,
            padding: "5px 11px", borderRadius: 999, cursor: "pointer", fontSize: 11, fontWeight: 600,
            border: "1px solid color-mix(in srgb, var(--status-alert) 30%, transparent)", background: "transparent", color: "var(--accent-red)" }}>
          <Trash2 size={12} /> Clear
        </button>
      </div>

      {/* Recurring strangers — the security finding, not a tagging queue. */}
      {recurring.length > 0 && (
        <div>
          <div style={sectionLabel}>Keeps coming back ({recurring.length})</div>
          <CardGrid scroll={false} min={168}>
            {recurring.map(c => (
              <ClusterProfile key={c.cluster_id} c={c} persons={persons}
                streamInfo={streamInfo} onOpen={() => setActiveCluster(c)}
                onConfirm={confirmCluster} />
            ))}
          </CardGrid>
        </div>
      )}

      {/* Seen once. Same card, lower billing. */}
      {oneOff.length > 0 && (
        <div>
          <div style={sectionLabel}>Seen once ({oneOff.length})</div>
          <CardGrid scroll={false} min={168}>
            {oneOff.map(c => (
              <ClusterProfile key={c.cluster_id} c={c} persons={persons}
                streamInfo={streamInfo} onOpen={() => setActiveCluster(c)}
                onConfirm={confirmCluster} />
            ))}
          </CardGrid>
        </div>
      )}

      {/* Single unmatched faces — not (yet) grouped into a recurring person. */}
      {singletons.length > 0 && (
        <div>
          <div style={sectionLabel}>Other recent faces ({singletons.length})</div>
          <CardGrid scroll={false} min={124}>
            {singletons.map(u => {
              const sp = (u.suggested_person_id ? persons.find(p => p.id === u.suggested_person_id) : null)
                ?? (u.suggested_name ? persons.find(p => p.name === u.suggested_name) : null);
              return (
                <ProfileCard key={u.id} onClick={() => setActive(u)}
                  title={`Camera ${u.cam_id + 1} \u00b7 ${formatRelative(new Date(u.seen_at))}`}
                  media={
                    <ProfileMedia aspect="1 / 1" fallback={<Users size={18} />}
                      src={faceCropSrc(u.thumbnail_b64, u.id, streamInfo) ?? undefined}>
                      {/* Crop quality drives whether this face is usable for
                          training, so it stays visible rather than living in a
                          tooltip \u2014 but it is a passive mark, top-right, like the
                          camera badge on an event card. */}
                      <span title={`Crop quality ${Math.round(u.quality * 100)}%`}
                        style={{
                          position: "absolute", top: 6, right: 6, padding: "2px 7px",
                          borderRadius: 999, fontSize: 9, fontWeight: 700,
                          background: "rgba(0,0,0,0.6)",
                          color: u.quality > 0.6 ? "var(--accent)"
                            : u.quality > 0.35 ? "var(--accent-amber)" : "#fff",
                        }}>{Math.round(u.quality * 100)}</span>
                    </ProfileMedia>
                  }>
                  {sp ? (
                    <div onClick={e => e.stopPropagation()}
                      style={{ display: "flex", flexDirection: "column", gap: 5 }}>
                      <div style={{ fontSize: 10, color: "var(--text-secondary)",
                        whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                        Looks like <strong style={{ color: "var(--accent)" }}>{u.suggested_name}</strong>
                      </div>
                      <button onClick={() => confirmFace(u.id, sp.id)} className="btn-primary"
                        style={{ padding: "3px 8px", fontSize: 10, borderRadius: 999,
                          display: "inline-flex", alignItems: "center", justifyContent: "center", gap: 4 }}>
                        <Check size={10} /> Confirm
                      </button>
                    </div>
                  ) : (
                    <div style={{ fontSize: 10, color: "var(--text-tertiary)" }}>
                      camera {u.cam_id + 1}<br />{formatRelative(new Date(u.seen_at))}
                    </div>
                  )}
                </ProfileCard>
              );
            })}
          </CardGrid>
        </div>
      )}

      {active && (
        <TagModal face={active} persons={persons} showToast={showToast}
          onClose={() => setActive(null)}
          onDone={() => { setActive(null); onTagged(); }} />
      )}
      {activeCluster && (
        <ClusterTagModal cluster={activeCluster} persons={persons} showToast={showToast}
          onClose={() => setActiveCluster(null)}
          onDone={() => { setActiveCluster(null); onTagged(); }} />
      )}
    </div>
  );
}

/** Name a whole cluster of unrecognised faces at once — assign to an existing
 *  person or create a new one, then batch-tag every face in the cluster. */
/** Display colors for the HSV-voted vehicle body colors. */

/**
 * One recurring stranger.
 *
 * Leads with recurrence — how many sightings, over how many distinct days, at
 * what time of day, and since when — because that is what makes an unidentified
 * face worth looking at. `first_seen` was fetched on every call and never
 * rendered; "first seen 3 weeks ago" is the difference between a delivery driver
 * and someone who has been watching the house.
 */
function ClusterProfile({ c, persons, streamInfo, onOpen, onConfirm }: {
  c: UnknownCluster;
  persons: KnownPerson[];
  streamInfo: ReturnType<typeof useStore.getState>["streamInfo"];
  onOpen: () => void;
  onConfirm: (c: UnknownCluster, personId: string) => void;
}) {
  const suggested = (c.suggested_person_id ? persons.find(p => p.id === c.suggested_person_id) : null)
    ?? (c.suggested_name ? persons.find(p => p.name === c.suggested_name) : null);

  return (
    <ProfileCard
      onClick={onOpen}
      title={`${c.count} sightings across ${c.days_active} day${c.days_active === 1 ? "" : "s"}`}
      media={
        <ProfileMedia aspect="1 / 1" fallback={<Users size={20} />}
          src={faceCropSrc(c.rep_thumbnail, c.rep_id, streamInfo) ?? undefined}>
          <CardCount>\u00d7{c.count}</CardCount>
        </ProfileMedia>
      }>
      {/* The recurrence line, most-significant first. */}
      <div style={{ fontSize: 12, fontWeight: 700, color: "var(--text-primary)" }}>
        {c.days_active >= 2
          ? <>{c.days_active} separate days</>
          : <>{c.count} sighting{c.count === 1 ? "" : "s"}</>}
        {c.time_pattern && c.time_pattern !== "Any time" && (
          <span style={{ color: "var(--accent)", fontWeight: 600 }}> \u00b7 {c.time_pattern}</span>
        )}
      </div>
      <div style={{ fontSize: 10.5, color: "var(--text-secondary)", lineHeight: 1.5, marginTop: 2 }}>
        {c.days_active >= 2 && <>{c.count} sighting{c.count === 1 ? "" : "s"} \u00b7 </>}
        {c.cameras.length > 1 ? `${c.cameras.length} cameras` : `camera ${(c.cameras[0] ?? 0) + 1}`}
        <br />
        first seen {formatRelative(new Date(c.first_seen))} \u00b7 last {fmtWhen(c.last_seen)}
      </div>

      {suggested ? (
        <div onClick={e => e.stopPropagation()} style={{ marginTop: 8 }}>
          <div style={{ fontSize: 11, marginBottom: 5 }}>
            Looks like <strong style={{ color: "var(--accent)" }}>{c.suggested_name}</strong>
            {c.suggested_score != null && (
              <span style={{ color: "var(--text-secondary)" }}> \u00b7 {Math.round(c.suggested_score * 100)}%</span>
            )}
          </div>
          <div style={{ display: "flex", gap: 6 }}>
            <button onClick={() => onConfirm(c, suggested.id)} className="btn-primary"
              style={{ padding: "4px 10px", fontSize: 11, borderRadius: 999,
                display: "inline-flex", alignItems: "center", gap: 4 }}>
              <Check size={11} /> Confirm
            </button>
            <button onClick={onOpen}
              style={{ padding: "4px 10px", fontSize: 11, borderRadius: 999, cursor: "pointer",
                border: "1px solid var(--border-strong)", background: "transparent",
                color: "var(--text-secondary)" }}>
              Not them
            </button>
          </div>
        </div>
      ) : (
        <div style={{ fontSize: 11, fontWeight: 700, color: "var(--accent)", marginTop: 6 }}>
          Name this person \u2192
        </div>
      )}
    </ProfileCard>
  );
}

function ClusterTagModal({ cluster, persons, onClose, onDone, showToast }: {
  cluster: UnknownCluster;
  persons: KnownPerson[];
  onClose: () => void;
  onDone:  () => void;
  showToast: (msg: string, type?: "success" | "error" | "info") => void;
}) {
  const [busy, setBusy] = useState(false);
  const [mode, setMode] = useState<"existing" | "new">(persons.length > 0 ? "existing" : "new");
  const [newName, setNewName] = useState("");
  const [newRole, setNewRole] = useState("resident");
  // Clip-linked sighting history: where (camera) + when this stranger was seen.
  const [sightings, setSightings] = useState<PersonSighting[]>([]);
  useEffect(() => {
    let alive = true;
    api.getPersonSightings(cluster.face_ids).then(s => { if (alive) setSightings(s); }).catch(() => {});
    return () => { alive = false; };
  }, [cluster.face_ids]);

  const toExisting = async (personId: string) => {
    setBusy(true);
    try { await api.assignFacesToPerson(cluster.face_ids, personId); onDone(); }
    catch (e) { showToast(`Couldn't tag: ${String(e).replace(/^.*Error:\s*/, "")}`, "error"); }
    finally { setBusy(false); }
  };
  const toNew = async () => {
    if (!newName.trim()) return;
    setBusy(true);
    try {
      // Seed a person from the first face, then batch-tag the rest into them.
      const person = await api.createPersonFromFace(cluster.face_ids[0], newName.trim(), newRole);
      if (cluster.face_ids.length > 1) await api.assignFacesToPerson(cluster.face_ids.slice(1), person.id);
      onDone();
    }
    catch (e) { showToast(`Couldn't create person: ${String(e).replace(/^.*Error:\s*/, "")}`, "error"); }
    finally { setBusy(false); }
  };

  useDismiss(onClose);

  return createPortal(
    <div style={{ position: "fixed", inset: 0, zIndex: 1000, background: "rgba(5,4,4,0.65)",
      display: "flex", alignItems: "center", justifyContent: "center", padding: 20 }} onClick={onClose}>
      <div className="glass-strong" onClick={e => e.stopPropagation()} style={{
        width: "100%", maxWidth: 460, padding: 22, display: "flex", flexDirection: "column", gap: 18 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <img src={faceCropSrc(cluster.rep_thumbnail, cluster.rep_id, useStore.getState().streamInfo) ?? undefined} alt="person"
            style={{ width: 84, height: 84, borderRadius: 18, objectFit: "cover" }} />
          <div style={{ flex: 1 }}>
            <div style={{ fontWeight: 700, fontSize: 16, letterSpacing: -0.02 }}>Who is this?</div>
            <div style={{ fontSize: 12, color: "var(--text-secondary)", marginTop: 4 }}>
              {cluster.count} sightings · {cluster.cameras.length} camera{cluster.cameras.length === 1 ? "" : "s"} ·
              naming applies to all {cluster.face_ids.length}
            </div>
          </div>
          <button onClick={onClose} style={{ background: "none", border: "none", cursor: "pointer", color: "var(--text-tertiary)", padding: 4 }}>
            <X size={18} />
          </button>
        </div>

        {/* The actual faces grouped here — so a high-count cluster isn't one mystery pic. */}
        {cluster.samples && cluster.samples.length > 1 && (
          <div>
            <div style={{ fontSize: 11, color: "var(--text-tertiary)", marginBottom: 6 }}>
              {cluster.count} captures grouped here — tap a face to see the full scene:
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(56px, 1fr))", gap: 6 }}>
              {cluster.samples.map(s => (
                <FaceContextZoom key={s.id} faceId={s.id} thumb={s.thumbnail} />
              ))}
            </div>
          </div>
        )}

        {/* Where & when: the clip-linked sighting history — location (camera) + time
            for every event this stranger appeared in; tap a scene to expand it. */}
        {sightings.length > 0 && (
          <div>
            <div style={{ fontSize: 11, color: "var(--text-tertiary)", marginBottom: 6, display: "flex", alignItems: "center", gap: 5 }}>
              <MapPin size={12} /> Seen in {sightings.length} clip{sightings.length === 1 ? "" : "s"} — where &amp; when:
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 6, maxHeight: 208, overflowY: "auto" }}>
              {sightings.map(s => (
                <div key={s.event_id} style={{ display: "flex", gap: 10, alignItems: "center", padding: 7,
                  borderRadius: 10, background: "rgb(var(--ink) / 0.02)", border: "1px solid var(--border)" }}>
                  <ZoomableImg
                    src={s.thumbnail.startsWith("data:") ? s.thumbnail : `data:image/jpeg;base64,${s.thumbnail}`}
                    caption={`Camera ${s.cam_id + 1} · ${fmtWhen(s.seen_at)}${s.ai_summary ? " · " + summaryText(s.ai_summary) : ""}`}
                    style={{ width: 58, height: 42, borderRadius: 8, objectFit: "cover", flexShrink: 0 }} />
                  <div style={{ minWidth: 0, flex: 1 }}>
                    <div style={{ fontSize: 11, fontWeight: 600 }}>Camera {s.cam_id + 1}</div>
                    <div style={{ fontSize: 10, color: "var(--text-tertiary)" }}>{fmtWhen(s.seen_at)}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* mature NVRs confirm-the-guess: lead with the agent's closest match. */}
        {(() => {
          const sp = (cluster.suggested_person_id ? persons.find(p => p.id === cluster.suggested_person_id) : null)
            ?? (cluster.suggested_name ? persons.find(p => p.name === cluster.suggested_name) : null);
          if (!sp) return null;
          return (
            <button type="button" disabled={busy} onClick={() => toExisting(sp.id)}
              style={{ display: "flex", alignItems: "center", gap: 10, padding: "11px 13px", borderRadius: 14,
                border: "1px solid var(--accent)", background: "var(--accent-glow)",
                cursor: busy ? "wait" : "pointer", textAlign: "left" }}>
              <Sparkles size={15} style={{ color: "var(--accent)" }} />
              <span style={{ flex: 1, fontSize: 13 }}>
                Looks like <strong style={{ color: "var(--accent)" }}>{sp.name}</strong>
                {cluster.suggested_score != null && (
                  <span style={{ color: "var(--text-tertiary)" }}> · {Math.round(cluster.suggested_score * 100)}% match</span>
                )}
              </span>
              <span style={{ display: "inline-flex", alignItems: "center", gap: 4, fontWeight: 700, fontSize: 12, color: "var(--accent)" }}>
                <Check size={14} /> Confirm
              </span>
            </button>
          );
        })()}

        <div style={{ display: "inline-flex", gap: 4, padding: 4, borderRadius: 999, background: "rgb(var(--ink) / 0.04)", alignSelf: "flex-start" }}>
          {([{ id: "existing" as const, label: "Existing person", disabled: persons.length === 0 },
             { id: "new" as const, label: "New person", disabled: false }]).map(m => (
            <button key={m.id} type="button" disabled={m.disabled} onClick={() => setMode(m.id)}
              style={{ padding: "6px 14px", borderRadius: 999, border: "none",
                background: mode === m.id ? "var(--accent)" : "transparent",
                color: mode === m.id ? "var(--on-accent)" : m.disabled ? "var(--text-tertiary)" : "var(--text-secondary)",
                fontSize: 12, fontWeight: 600, cursor: m.disabled ? "not-allowed" : "pointer" }}>
              {m.label}
            </button>
          ))}
        </div>

        {mode === "existing" && persons.length > 0 && (
          <PersonPickList persons={persons} disabled={busy} onPick={k => toExisting(k.id)} />
        )}

        {mode === "new" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <input value={newName} onChange={e => setNewName(e.target.value)} placeholder="Name (e.g. John Smith)" autoFocus
              style={{ padding: "10px 14px", borderRadius: 12, fontSize: 14, border: "1px solid var(--border-strong)",
                background: "rgb(var(--ink) / 0.04)", color: "var(--text-primary)", outline: "none" }} />
            <select value={newRole} onChange={e => setNewRole(e.target.value)}
              style={{ padding: "10px 14px", borderRadius: 12, fontSize: 14, border: "1px solid var(--border-strong)",
                background: "rgb(var(--ink) / 0.04)", color: "var(--text-primary)", outline: "none" }}>
              <option value="resident">Resident / Family</option>
              <option value="employee">Employee / Staff</option>
              <option value="visitor">Trusted Visitor</option>
            </select>
            <button onClick={toNew} disabled={busy || !newName.trim()} className="btn-primary"
              style={{ padding: "10px 16px", opacity: !newName.trim() ? 0.5 : 1 }}>
              <UserPlus size={13} /> Create person from {cluster.face_ids.length} faces
            </button>
          </div>
        )}
      </div>
    </div>,
    document.body,
  );
}

function TagModal({ face, persons, onClose, onDone, showToast }: {
  face: UnknownFace;
  persons: KnownPerson[];
  onClose: () => void;
  onDone:  () => void;
  showToast: (msg: string, type?: "success" | "error" | "info") => void;
}) {
  const [busy, setBusy] = useState(false);
  const [mode, setMode] = useState<"existing" | "new">(persons.length > 0 ? "existing" : "new");
  const [newName, setNewName] = useState("");
  const [newRole, setNewRole] = useState("resident");

  const tag = async (personId: string) => {
    setBusy(true);
    try { await api.assignFaceToPerson(face.id, personId); onDone(); }
    catch (e) { showToast(`Couldn't tag: ${String(e).replace(/^.*Error:\s*/, "")}`, "error"); }
    finally { setBusy(false); }
  };
  const create = async () => {
    if (!newName.trim()) return;
    setBusy(true);
    try { await api.createPersonFromFace(face.id, newName.trim(), newRole); onDone(); }
    catch (e) { showToast(`Couldn't create person: ${String(e).replace(/^.*Error:\s*/, "")}`, "error"); }
    finally { setBusy(false); }
  };

  useDismiss(onClose);

  return createPortal(
    <div style={{
      position: "fixed", inset: 0, zIndex: 1000,
      background: "rgba(5,4,4,0.65)",
      display: "flex", alignItems: "center", justifyContent: "center",
      padding: 20,
    }} onClick={onClose}>
      <div className="glass-strong" onClick={e => e.stopPropagation()} style={{
        width: "100%", maxWidth: 460, padding: 22,
        display: "flex", flexDirection: "column", gap: 18,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <img src={faceCropSrc(face.thumbnail_b64, face.id, useStore.getState().streamInfo) ?? undefined} alt="face"
            style={{ width: 84, height: 84, borderRadius: 18, objectFit: "cover" }} />
          <div style={{ flex: 1 }}>
            <div style={{ fontWeight: 700, fontSize: 16, letterSpacing: -0.02 }}>Who is this?</div>
            <div style={{ fontSize: 12, color: "var(--text-secondary)", marginTop: 4 }}>
              Camera {face.cam_id + 1} · {formatRelative(new Date(face.seen_at))} · quality {Math.round(face.quality * 100)}
            </div>
          </div>
          <button onClick={onClose}
            style={{ background: "none", border: "none", cursor: "pointer", color: "var(--text-tertiary)", padding: 4 }}>
            <X size={18} />
          </button>
        </div>

        {/* Mode pill */}
        <div style={{
          display: "inline-flex", gap: 4, padding: 4, borderRadius: 999,
          background: "rgb(var(--ink) / 0.04)", alignSelf: "flex-start",
        }}>
          {([
            { id: "existing" as const, label: "Existing person", disabled: persons.length === 0 },
            { id: "new"      as const, label: "New person",      disabled: false },
          ]).map(m => (
            <button key={m.id} type="button" disabled={m.disabled} onClick={() => setMode(m.id)}
              style={{
                padding: "6px 14px", borderRadius: 999, border: "none",
                background: mode === m.id ? "var(--accent)" : "transparent",
                color:      mode === m.id ? "var(--on-accent)"     : m.disabled ? "var(--text-tertiary)" : "var(--text-secondary)",
                fontSize: 12, fontWeight: 600,
                cursor: m.disabled ? "not-allowed" : "pointer",
              }}>
              {m.label}
            </button>
          ))}
        </div>

        {mode === "existing" && persons.length > 0 && (
          <PersonPickList persons={persons} disabled={busy} onPick={k => tag(k.id)} />
        )}

        {mode === "new" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <input value={newName} onChange={e => setNewName(e.target.value)}
              placeholder="Name (e.g. John Smith)" autoFocus
              style={{
                padding: "10px 14px", borderRadius: 12, fontSize: 14,
                border: "1px solid var(--border-strong)",
                background: "rgb(var(--ink) / 0.04)",
                color: "var(--text-primary)", outline: "none",
              }} />
            <select value={newRole} onChange={e => setNewRole(e.target.value)}
              style={{
                padding: "10px 14px", borderRadius: 12, fontSize: 14,
                border: "1px solid var(--border-strong)",
                background: "rgb(var(--ink) / 0.04)",
                color: "var(--text-primary)", outline: "none",
              }}>
              <option value="resident">Resident / Family</option>
              <option value="employee">Employee / Staff</option>
              <option value="visitor">Trusted Visitor</option>
            </select>
            <button onClick={create} disabled={busy || !newName.trim()} className="btn-primary"
              style={{ padding: "10px 16px", opacity: !newName.trim() ? 0.5 : 1 }}>
              <UserPlus size={13} /> Create person from this face
            </button>
          </div>
        )}
      </div>
    </div>,
    document.body,
  );
}

// ── Enroll (manual, from a live frame) — kept from the original flow ────────
