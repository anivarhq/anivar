/**
 * Review — "Who is this?", and nothing else.
 *
 * One card per PERSON the cameras keep seeing but can't name: a group of
 * unknown faces (`list_unknown_clusters`, already noise-filtered). Name them
 * once and every sighting is theirs; Remove them if nobody will ever name them
 * (a face on a TV, a passer-by). When the matcher thinks it knows who it is, the
 * same card asks "Is this Ravi?" instead.
 *
 * This used to show every internal row the pipeline produced: up to 60 single
 * face crops, the same groups split into "keeps coming back" and "seen once",
 * anonymous body fragments with merge proposals, and quality scores. On real
 * footage all of that was ONE person — 341 captures that clustered into a single
 * group. Real systems (UniFi Protect, the Frigate face library, Nest familiar
 * faces) ask one question per person, and ask it once.
 */
import { useEffect, useMemo, useState } from "react";
import { Sparkles, Users, Check, X, MapPin, UserPlus } from "lucide-react";
import { api, KnownPerson, UnknownCluster, PersonSighting } from "../../api";
import { useStore } from "../../store";
import { createPortal } from "react-dom";
import { faceCropSrc } from "../../lib/eventThumb";
import { fmtWhen } from "../../lib/time";
import { Confirm, useDismiss } from "../../components/ui/Modal";
import { CardGrid, ProfileCard, ProfileMedia, CardEmpty, CARD_MIN } from "../review/Card";
import { PersonPickList, FaceContextZoom, formatRelative, ZoomableImg, summaryText } from "./shared";
import styles from "../review/ReviewFeed.module.css";

const errText = (e: unknown) => String(e).replace(/^.*Error:\s*/, "");

export function ReviewSection({ clusters, persons, onChanged, showToast }: {
  clusters: UnknownCluster[];
  persons:  KnownPerson[];
  onChanged: () => void;
  showToast: (msg: string, type?: "success" | "error" | "info") => void;
}) {
  const streamInfo = useStore(s => s.streamInfo); // '@crop' markers → URL-served crops
  /** The group being named; `skipSuggestion` when the user just said "No" to it. */
  const [naming, setNaming] = useState<{ c: UnknownCluster; skipSuggestion: boolean } | null>(null);
  const [removing, setRemoving] = useState<UnknownCluster | null>(null);

  // Someone seen on many days is the person you'll meet again: name them first.
  const sorted = useMemo(() => clusters.slice().sort((a, b) =>
    b.days_active - a.days_active || b.count - a.count), [clusters]);

  const confirm = async (c: UnknownCluster, person: KnownPerson) => {
    try {
      await api.assignFacesToPerson(c.face_ids, person.id);
      showToast(`Named ${person.name}`, "success");
      onChanged();
    } catch (e) { showToast(`Couldn't name them: ${errText(e)}`, "error"); }
  };
  const remove = async (c: UnknownCluster) => {
    setRemoving(null);
    try {
      await api.deleteUnknownFaces(c.face_ids);
      showToast("Removed", "info");
      onChanged();
    } catch (e) { showToast(`Couldn't remove: ${errText(e)}`, "error"); }
  };

  return (
    <>
      {sorted.length > 0 && (
        <div style={{ flexShrink: 0, padding: "0 16px 8px", fontSize: 12, color: "var(--text-secondary)" }}>
          {sorted.length} {sorted.length === 1 ? "person" : "people"} to name — naming someone once names every sighting of them.
        </div>
      )}

      <CardGrid min={CARD_MIN}>
        {sorted.length === 0 ? (
          <CardEmpty icon={<Sparkles size={32} />}>
            Nothing to review. People your cameras keep seeing will appear here to be named.
          </CardEmpty>
        ) : sorted.map(c => (
          <WhoIsThis key={c.cluster_id} c={c} persons={persons} streamInfo={streamInfo}
            onName={skipSuggestion => setNaming({ c, skipSuggestion })}
            onConfirm={p => confirm(c, p)}
            onRemove={() => setRemoving(c)} />
        ))}
      </CardGrid>

      {naming && (
        <ClusterTagModal cluster={naming.c} persons={persons} skipSuggestion={naming.skipSuggestion}
          showToast={showToast}
          onClose={() => setNaming(null)}
          onDone={() => { setNaming(null); onChanged(); }} />
      )}
      {removing && (
        <Confirm title="Remove this person?"
          body={<>Their {removing.face_ids.length} face picture{removing.face_ids.length === 1 ? "" : "s"} go. If
            the cameras see them again, they'll be asked about again. Nobody you've named is affected.</>}
          confirmLabel="Remove" onConfirm={() => remove(removing)} onCancel={() => setRemoving(null)} />
      )}
    </>
  );
}

/** "Seen 341 times · last 5m ago", or "On 3 days · usually evenings". */
function seenLine(c: UnknownCluster): string {
  if (c.days_active >= 2) {
    const when = c.time_pattern && c.time_pattern !== "Any time" ? ` · ${c.time_pattern.toLowerCase()}` : "";
    return `On ${c.days_active} days${when}`;
  }
  return `Seen ${c.count} time${c.count === 1 ? "" : "s"} · last ${formatRelative(new Date(c.last_seen))}`;
}

/** One person to name. A div-based card (`ProfileCard`), because it carries buttons. */
function WhoIsThis({ c, persons, streamInfo, onName, onConfirm, onRemove }: {
  c: UnknownCluster;
  persons: KnownPerson[];
  streamInfo: ReturnType<typeof useStore.getState>["streamInfo"];
  onName: (skipSuggestion: boolean) => void;
  onConfirm: (p: KnownPerson) => void;
  onRemove: () => void;
}) {
  const suggested = (c.suggested_person_id ? persons.find(p => p.id === c.suggested_person_id) : undefined)
    ?? (c.suggested_name ? persons.find(p => p.name === c.suggested_name) : undefined);
  const btn = { flex: 1, justifyContent: "center" } as const;

  return (
    <ProfileCard onClick={() => onName(false)} title={seenLine(c)}
      media={
        <ProfileMedia aspect="1 / 1" fallback={<Users size={20} />}
          src={faceCropSrc(c.rep_thumbnail, c.rep_id, streamInfo) ?? undefined} />
      }>
      <div style={{ fontSize: 12, fontWeight: 700, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
        {suggested ? `Is this ${suggested.name}?` : "Who is this?"}
      </div>
      <div style={{ fontSize: 10.5, color: "var(--text-secondary)", marginTop: 2,
        whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
        {seenLine(c)}
      </div>
      {/* Buttons stop the click reaching the card, which would open naming. */}
      <div onClick={e => e.stopPropagation()} style={{ display: "flex", gap: 6, marginTop: 8 }}>
        {suggested ? (<>
          <button type="button" className={`lg ${styles.glassBtn}`} style={btn} onClick={() => onConfirm(suggested)}>
            <Check size={12} /> Yes
          </button>
          <button type="button" className={`lg ${styles.glassBtn}`} style={btn} onClick={() => onName(true)}>
            No
          </button>
        </>) : (<>
          <button type="button" className={`lg ${styles.glassBtn}`} style={btn} onClick={() => onName(false)}>
            <UserPlus size={12} /> Name
          </button>
          <button type="button" className={`lg ${styles.glassBtn}`} style={btn} onClick={onRemove}>
            Remove
          </button>
        </>)}
      </div>
    </ProfileCard>
  );
}

/** Name a whole group of unrecognised faces at once — assign to an existing
 *  person or create a new one, then tag every face in the group. */
function ClusterTagModal({ cluster, persons, skipSuggestion, onClose, onDone, showToast }: {
  cluster: UnknownCluster;
  persons: KnownPerson[];
  /** Opened from "No" on "Is this X?" — don't offer X again. */
  skipSuggestion: boolean;
  onClose: () => void;
  onDone:  () => void;
  showToast: (msg: string, type?: "success" | "error" | "info") => void;
}) {
  const [busy, setBusy] = useState(false);
  const [mode, setMode] = useState<"existing" | "new">(persons.length > 0 ? "existing" : "new");
  const [newName, setNewName] = useState("");
  const [newRole, setNewRole] = useState("resident");
  // Clip-linked sighting history: where (camera) + when this person was seen.
  const [sightings, setSightings] = useState<PersonSighting[]>([]);
  useEffect(() => {
    let alive = true;
    api.getPersonSightings(cluster.face_ids).then(s => { if (alive) setSightings(s); }).catch(() => {});
    return () => { alive = false; };
  }, [cluster.face_ids]);

  const suggested = skipSuggestion ? undefined
    : (cluster.suggested_person_id ? persons.find(p => p.id === cluster.suggested_person_id) : undefined)
      ?? (cluster.suggested_name ? persons.find(p => p.name === cluster.suggested_name) : undefined);
  // "No, it isn't X" leaves X out of the pick list too.
  const pickable = skipSuggestion && cluster.suggested_person_id
    ? persons.filter(p => p.id !== cluster.suggested_person_id) : persons;

  const toExisting = async (personId: string) => {
    setBusy(true);
    try { await api.assignFacesToPerson(cluster.face_ids, personId); onDone(); }
    catch (e) { showToast(`Couldn't name them: ${errText(e)}`, "error"); }
    finally { setBusy(false); }
  };
  const toNew = async () => {
    if (!newName.trim()) return;
    setBusy(true);
    try {
      // Seed a person from the first face, then tag the rest into them.
      const person = await api.createPersonFromFace(cluster.face_ids[0], newName.trim(), newRole);
      if (cluster.face_ids.length > 1) await api.assignFacesToPerson(cluster.face_ids.slice(1), person.id);
      onDone();
    }
    catch (e) { showToast(`Couldn't create person: ${errText(e)}`, "error"); }
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
              {seenLine(cluster)} · the name applies to every sighting
            </div>
          </div>
          <button onClick={onClose} style={{ background: "none", border: "none", cursor: "pointer", color: "var(--text-tertiary)", padding: 4 }}>
            <X size={18} />
          </button>
        </div>

        {/* The actual faces grouped here — so a large group isn't one mystery picture. */}
        {cluster.samples && cluster.samples.length > 1 && (
          <div>
            <div style={{ fontSize: 11, color: "var(--text-tertiary)", marginBottom: 6 }}>
              Tap a face to see the full scene:
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(56px, 1fr))", gap: 6 }}>
              {cluster.samples.map(s => (
                <FaceContextZoom key={s.id} faceId={s.id} thumb={s.thumbnail} />
              ))}
            </div>
          </div>
        )}

        {/* Where & when: every clip this person appeared in; tap a scene to expand it. */}
        {sightings.length > 0 && (
          <div>
            <div style={{ fontSize: 11, color: "var(--text-tertiary)", marginBottom: 6, display: "flex", alignItems: "center", gap: 5 }}>
              <MapPin size={12} /> Seen in {sightings.length} clip{sightings.length === 1 ? "" : "s"}:
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

        {/* Lead with the matcher's guess when it has one — in words, not a percentage. */}
        {suggested && (
          <button type="button" disabled={busy} onClick={() => toExisting(suggested.id)}
            style={{ display: "flex", alignItems: "center", gap: 10, padding: "11px 13px", borderRadius: 14,
              border: "1px solid var(--accent)", background: "var(--accent-glow)",
              cursor: busy ? "wait" : "pointer", textAlign: "left" }}>
            <Sparkles size={15} style={{ color: "var(--accent)" }} />
            <span style={{ flex: 1, fontSize: 13 }}>
              Looks like <strong style={{ color: "var(--accent)" }}>{suggested.name}</strong>
            </span>
            <span style={{ display: "inline-flex", alignItems: "center", gap: 4, fontWeight: 700, fontSize: 12, color: "var(--accent)" }}>
              <Check size={14} /> Confirm
            </span>
          </button>
        )}

        <div style={{ display: "inline-flex", gap: 4, padding: 4, borderRadius: 999, background: "rgb(var(--ink) / 0.04)", alignSelf: "flex-start" }}>
          {([{ id: "existing" as const, label: "Existing person", disabled: pickable.length === 0 },
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

        {mode === "existing" && pickable.length > 0 && (
          <PersonPickList persons={pickable} disabled={busy} onPick={k => toExisting(k.id)} />
        )}

        {(mode === "new" || pickable.length === 0) && (
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <input value={newName} onChange={e => setNewName(e.target.value)} placeholder="Name (e.g. John Smith)" autoFocus
              onKeyDown={e => { if (e.key === "Enter") toNew(); }}
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
              <UserPlus size={13} /> Save name
            </button>
          </div>
        )}
      </div>
    </div>,
    document.body,
  );
}
