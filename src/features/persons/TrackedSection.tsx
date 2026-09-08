/**
 * Tracked across cameras — body Re-ID.
 *
 * A body match only ever PROPOSES a name; a face, or the user, confirms it.
 * That asymmetry is the whole design: clothing is a weak identifier that
 * changes daily, so nothing here writes a durable identity on its own.
 */
import { useCallback, useEffect, useState, type CSSProperties, type ReactNode } from "react";
import { Layers, Users, Check, X, Sparkles, UserPlus, Activity } from "lucide-react";
import { api, KnownPerson, TrackedPerson, TrackedCluster } from "../../api";
import { useStore } from "../../store";
import { createPortal } from "react-dom";
import { fmtWhen } from "../../lib/time";
import { bodyCropSrc } from "../../lib/eventThumb";
import { Modal, useDismiss } from "../../components/ui/Modal";
import { CardGrid, ProfileCard, ProfileMedia, CardCount, CardEmpty } from "../review/Card";
import { PersonPickList, SectionHeader, ZoomableImg, OutfitLine, fmtDay, formatRelative } from "./shared";

export function TrackedSection({ tracked, backend, persons, onChanged, showToast }: {
  tracked: TrackedPerson[];
  backend: string | null;
  persons: KnownPerson[];
  onChanged: () => void;
  showToast: (msg: string, type?: "success" | "error" | "info") => void;
}) {
  const named   = tracked.filter(p => p.known_name);
  const unknown = tracked.filter(p => !p.known_name);
  const [showUnknown, setShowUnknown] = useState(false);
  const [naming, setNaming] = useState<TrackedPerson | null>(null);
  // A track being CORRECTED: who it was wrongly shown as (known_name or suggestion).
  const [correcting, setCorrecting] = useState<{ p: TrackedPerson; wrongName: string } | null>(null);

  // Self-grouping: proposed clusters of anonymous tracks that look like one person.
  const [clusters, setClusters] = useState<TrackedCluster[]>([]);
  // The group being named, with the user's CURATED member ids (after deselecting outliers).
  const [grouping, setGrouping] = useState<{ cluster: TrackedCluster; ids: string[] } | null>(null);
  const [dismissed, setDismissed] = useState<Set<string>>(new Set());
  const loadClusters = useCallback(() => {
    api.listTrackedClusters().then(setClusters).catch(() => setClusters([]));
  }, []);
  // Mount-only: re-fetching on every `tracked` change re-ran the whole
  // clustering pass once per parent refresh; mutations below call it directly.
  useEffect(() => { loadClusters(); }, [loadClusters]);

  const assign = async (bodyId: string, knownId: string, name: string) => {
    try {
      await api.assignTrackedToKnown([bodyId], knownId);
      showToast(`Tagged as ${name} — their appearance history was merged`, "success");
      setNaming(null);
      onChanged();
      loadClusters();
    } catch (e) { showToast(String(e), "error"); }
  };

  // DURABLE correction: record a hard negative for the wrongly-shown person, then
  // reassign to the right one (correctId) or detach (null). Stops the mistake repeating.
  const doCorrect = async (p: TrackedPerson, wrongName: string, correctId: string | null) => {
    // Bind by ID (authoritative); the name lookup is only a fallback for stale
    // rows — matching by name mis-resolved after renames/duplicate names.
    const wrongId = p.known_person_id ?? persons.find(k => k.name === wrongName)?.id ?? null;
    try {
      await api.correctTrack(p.person_id, wrongId, correctId);
      const correctName = correctId ? persons.find(k => k.id === correctId)?.name : null;
      showToast(correctName ? `Corrected to ${correctName} — won't repeat` : `Marked: not ${wrongName}`, "success");
      setCorrecting(null); loadClusters(); onChanged();
    } catch (e) { showToast(String(e), "error"); }
  };

  // Batch-train a CURATED set of tracks → an existing person.
  const assignGroup = async (ids: string[], knownId: string, name: string) => {
    if (ids.length === 0) { showToast("Select at least one photo", "info"); return; }
    try {
      await api.assignTrackedToKnown(ids, knownId);
      showToast(`${ids.length} track${ids.length === 1 ? "" : "s"} merged into ${name}`, "success");
      setGrouping(null); loadClusters(); onChanged();
    } catch (e) { showToast(String(e), "error"); }
  };
  // Batch-train a curated set → a brand-new body-only ("no face yet") person.
  const nameGroup = async (ids: string[], name: string) => {
    if (ids.length === 0) { showToast("Select at least one photo", "info"); return; }
    try {
      await api.nameTrackedGroup(ids, name, "unknown");
      showToast(`Created ${name} from ${ids.length} track${ids.length === 1 ? "" : "s"} (no face yet)`, "success");
      setGrouping(null); loadClusters(); onChanged();
    } catch (e) { showToast(String(e), "error"); }
  };
  const visibleClusters = clusters.filter(c => !dismissed.has(c.cluster_id));

  if (tracked.length === 0) {
    return (
      <div className="glass" style={{
        padding: "40px 28px", textAlign: "center",
        display: "flex", flexDirection: "column", alignItems: "center", gap: 14,
      }}>
        <Layers size={42} style={{ opacity: 0.35 }} />
        <div>
          <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 6 }}>No tracked people yet</div>
        </div>
      </div>
    );
  }

  /** Grid for the merge PROPOSALS only. Those cards expand to `gridColumn: 1/-1`
   *  to show sample photos, which CardGrid's tighter columns would fight; the
   *  identity cards below use CardGrid like every other identity list. */
  const grid: CSSProperties = { display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(230px, 1fr))", gap: 12 };
  return (
    <>
      <div style={{ fontSize: 11, color: "var(--text-tertiary)", margin: "0 2px 10px", lineHeight: 1.5 }}>
        Recognised people are matched even at a distance (appearance, anchored by their face). Unknowns are
        appearance-only and expire in ~2 days — tag one as a known person to teach the system.
        {backend && (
          <span style={{ marginLeft: 6 }}>
            Engine: <strong style={{ color: backend.startsWith("Deep") ? "var(--accent)" : "var(--text-secondary)" }}>{backend}</strong>
          </span>
        )}
      </div>

      {visibleClusters.length > 0 && (
        <div style={{ marginBottom: 18 }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: "var(--accent)", margin: "0 2px 4px",
            display: "inline-flex", alignItems: "center", gap: 6 }}>
            <Sparkles size={12} /> Suggested groups
          </div>
          <div style={grid}>
            {visibleClusters.map(c => (
              <ClusterCard key={c.cluster_id} c={c} persons={persons}
                onAssign={(ids, kid, n) => assignGroup(ids, kid, n)}
                onOpen={(ids) => setGrouping({ cluster: c, ids })}
                onDismiss={() => setDismissed(prev => new Set(prev).add(c.cluster_id))} />
            ))}
          </div>
        </div>
      )}

      {named.length > 0 && (
        <div style={{ marginBottom: 16 }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: "var(--accent)", margin: "0 2px 8px" }}>
            Recognised people
          </div>
          <CardGrid scroll={false} min={190}>
            {named.map(p => <TrackedCard key={p.person_id} p={p} persons={persons} onName={setNaming} onConfirm={assign} onCorrect={(pp, w) => setCorrecting({ p: pp, wrongName: w })} />)}
          </CardGrid>
        </div>
      )}

      {unknown.length > 0 && (
        <div>
          <button type="button" onClick={() => setShowUnknown(v => !v)}
            style={{ display: "inline-flex", alignItems: "center", gap: 7, padding: "7px 12px", borderRadius: 10,
              fontSize: 12, fontWeight: 700, cursor: "pointer", marginBottom: 8,
              border: "1px solid var(--border-strong)", background: "transparent", color: "var(--text-secondary)" }}>
            {showUnknown ? "▾" : "▸"} Recent unknowns ({unknown.length}) — appearance only, expire in ~2 days
          </button>
          {showUnknown && (
            <CardGrid scroll={false} min={190}>
              {unknown.map(p => <TrackedCard key={p.person_id} p={p} persons={persons} onName={setNaming} onConfirm={assign} onCorrect={(pp, w) => setCorrecting({ p: pp, wrongName: w })} />)}
            </CardGrid>
          )}
        </div>
      )}

      {naming && (
        <NameTrackedModal person={naming} persons={persons}
          onPick={(id, name) => assign(naming.person_id, id, name)}
          onClose={() => setNaming(null)} />
      )}

      {correcting && (
        <NameTrackedModal person={correcting.p} persons={persons}
          title={`Not ${correcting.wrongName} — who is it?`}
          note={<>Shown as <strong>{correcting.wrongName}</strong>. Pick who it really is, or “No one”.</>}
          onNone={() => doCorrect(correcting.p, correcting.wrongName, null)}
          onPick={(id) => doCorrect(correcting.p, correcting.wrongName, id)}
          onClose={() => setCorrecting(null)} />
      )}

      {grouping && (
        <GroupNameModal cluster={grouping.cluster} count={grouping.ids.length} persons={persons}
          onPickExisting={(id, name) => assignGroup(grouping.ids, id, name)}
          onNameNew={(name) => nameGroup(grouping.ids, name)}
          onClose={() => setGrouping(null)} />
      )}
    </>
  );
}

// A proposed group of anonymous tracks (self-grouping). Expand to REVIEW every
// photo in the group, deselect any that don't belong, then confirm to the "looks
// like X" suggestion or name it (curated). Dismiss hides the whole group.
function ClusterCard({ c, persons, onAssign, onOpen, onDismiss }: {
  c: TrackedCluster;
  persons: KnownPerson[];
  onAssign: (ids: string[], knownId: string, name: string) => void;
  onOpen: (ids: string[]) => void;
  onDismiss: () => void;
}) {
  const suggestion = (c.suggested_person_id ? persons.find(k => k.id === c.suggested_person_id) : undefined)
    ?? (c.suggested_name ? persons.find(k => k.name === c.suggested_name) : undefined);
  const [expanded, setExpanded] = useState(false);
  const streamInfo = useStore(s => s.streamInfo); // '@crop' markers → URL-served crops
  // Tracks the user removed from the group before naming (by their person_id).
  const [excluded, setExcluded] = useState<Set<string>>(new Set());
  const activeIds = c.member_ids.filter(id => !excluded.has(id));
  const toggle = (pid: string) =>
    setExcluded(prev => { const n = new Set(prev); n.has(pid) ? n.delete(pid) : n.add(pid); return n; });

  return (
    <div className="glass" style={{ padding: 12, gridColumn: expanded ? "1 / -1" : undefined }}>
      <div style={{ display: "flex", gap: 12, alignItems: "flex-start" }}>
        <button type="button" onClick={() => setExpanded(v => !v)} title={expanded ? "Collapse" : "Expand to review photos"}
          style={{ width: 52, height: 52, borderRadius: 12, overflow: "hidden", flexShrink: 0, padding: 0, cursor: "pointer",
            border: "1px solid var(--border)", background: "rgb(var(--ink) / 0.05)",
            display: "flex", alignItems: "center", justifyContent: "center" }}>
          {c.rep_thumbnail
            ? <img src={bodyCropSrc(c.rep_thumbnail, c.rep_track_id, streamInfo) ?? undefined} alt="" loading="lazy"
                style={{ width: "100%", height: "100%", objectFit: "cover" }} />
            : <Layers size={20} style={{ opacity: 0.4 }} />}
        </button>
        <div style={{ minWidth: 0, flex: 1 }}>
          <div style={{ fontWeight: 700, fontSize: 13 }}>
            {activeIds.length}{excluded.size > 0 ? ` of ${c.track_count}` : ""} tracks · same person?
          </div>
          <div style={{ fontSize: 11, color: "var(--text-secondary)" }}>
            {c.sighting_count} sighting{c.sighting_count === 1 ? "" : "s"} · {c.cameras.length} camera{c.cameras.length === 1 ? "" : "s"}
          </div>
          <button type="button" onClick={() => setExpanded(v => !v)}
            style={{ marginTop: 4, padding: 0, border: "none", background: "transparent", cursor: "pointer",
              fontSize: 10.5, fontWeight: 700, color: "var(--accent)", display: "inline-flex", alignItems: "center", gap: 3 }}>
            {expanded ? "▾ Hide photos" : `▸ Review ${c.samples.length} photo${c.samples.length === 1 ? "" : "s"}`}
          </button>
        </div>
      </div>

      {expanded && (
        <div style={{ marginTop: 10 }}>
          <div style={{ fontSize: 10, color: "var(--text-tertiary)", marginBottom: 6 }}>
            Click a photo to enlarge it. Use the corner toggle to remove one that's a different person.
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(72px, 1fr))", gap: 8 }}>
            {c.samples.map(s => {
              const off = excluded.has(s.person_id);
              return (
                <div key={s.person_id}
                  style={{ position: "relative", aspectRatio: "1", borderRadius: 10, overflow: "hidden",
                    border: off ? "1px solid var(--border)" : "2px solid var(--accent)", background: "rgb(var(--ink) / 0.04)" }}>
                  <ZoomableImg src={bodyCropSrc(s.thumbnail, s.person_id, streamInfo) ?? ""}
                    caption={`${s.sightings} sighting${s.sightings === 1 ? "" : "s"} · last seen ${fmtWhen(s.last_seen)}`}
                    style={{ width: "100%", height: "100%", objectFit: "cover", opacity: off ? 0.3 : 1, filter: off ? "grayscale(1)" : "none" }} />
                  <button type="button" onClick={(e) => { e.stopPropagation(); toggle(s.person_id); }}
                    title={off ? "Removed — add back to group" : "Remove from group (different person)"}
                    style={{ position: "absolute", top: 3, right: 3, width: 18, height: 18, borderRadius: 999, padding: 0,
                      cursor: "pointer", border: "none", display: "flex", alignItems: "center", justifyContent: "center",
                      background: off ? "rgba(0,0,0,0.66)" : "var(--accent)", color: off ? "#fff" : "var(--on-accent)" }}>
                    {off ? <X size={11} /> : <Check size={11} />}
                  </button>
                </div>
              );
            })}
          </div>
        </div>
      )}

      <div style={{ marginTop: 10, display: "flex", gap: 6, flexWrap: "wrap", alignItems: "center" }}>
        {suggestion && (
          <button type="button" disabled={activeIds.length === 0}
            onClick={() => onAssign(activeIds, suggestion.id, suggestion.name)}
            title="Confirm the selected photos are this person"
            style={{ display: "inline-flex", alignItems: "center", gap: 4, padding: "4px 9px", borderRadius: 999,
              fontSize: 10.5, fontWeight: 700, cursor: activeIds.length ? "pointer" : "default", opacity: activeIds.length ? 1 : 0.5,
              border: "1px solid var(--accent)", background: "var(--accent-glow)", color: "var(--accent)" }}>
            <Check size={11} /> {excluded.size > 0 ? "Selected" : "All"} are {suggestion.name}
          </button>
        )}
        <button type="button" disabled={activeIds.length === 0} onClick={() => onOpen(activeIds)}
          style={{ display: "inline-flex", alignItems: "center", gap: 4, padding: "4px 9px", borderRadius: 999,
            fontSize: 10.5, fontWeight: 600, cursor: activeIds.length ? "pointer" : "default", opacity: activeIds.length ? 1 : 0.5,
            border: "1px solid var(--border-strong)", background: "transparent", color: "var(--text-primary)" }}>
          <UserPlus size={11} /> Name group
        </button>
        <button type="button" onClick={onDismiss} title="Not the same person"
          style={{ display: "inline-flex", alignItems: "center", gap: 4, padding: "4px 8px", borderRadius: 999,
            fontSize: 10.5, fontWeight: 600, cursor: "pointer",
            border: "1px solid var(--border)", background: "transparent", color: "var(--text-tertiary)" }}>
          <X size={11} /> Not a group
        </button>
      </div>
    </div>
  );
}

function GroupNameModal({ cluster, count, persons, onPickExisting, onNameNew, onClose }: {
  cluster: TrackedCluster;
  count: number;   // curated track count (after deselecting outliers)
  persons: KnownPerson[];
  onPickExisting: (knownId: string, name: string) => void;
  onNameNew: (name: string) => void;
  onClose: () => void;
}) {
  const [newName, setNewName] = useState("");
  useDismiss(onClose);
  return createPortal(
    <div onClick={onClose} style={{
      position: "fixed", inset: 0, background: "rgba(0,0,0,0.6)", backdropFilter: "blur(4px)",
      display: "flex", alignItems: "center", justifyContent: "center", zIndex: 1000, padding: 20,
    }}>
      <div onClick={e => e.stopPropagation()} className="glass" style={{ width: 440, maxWidth: "100%", padding: 20 }}>
        <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 4 }}>Name this group</div>
        <div style={{ fontSize: 11.5, color: "var(--text-secondary)", marginBottom: 14, lineHeight: 1.5 }}>
          Merge these <strong>{count} track{count === 1 ? "" : "s"}</strong> into one identity. Pick an enrolled person, or
          create a new <strong>body-only</strong> identity (no face yet — it upgrades automatically once a face links).
        </div>

        {/* New body-only identity */}
        <div style={{ display: "flex", gap: 8, marginBottom: 14 }}>
          <input value={newName} onChange={e => setNewName(e.target.value)} placeholder="New person name…"
            onKeyDown={e => { if (e.key === "Enter" && newName.trim()) onNameNew(newName.trim()); }}
            style={{ flex: 1, padding: "9px 12px", borderRadius: 10, fontSize: 13, fontFamily: "inherit",
              border: "1px solid var(--border-strong)", background: "rgb(var(--ink) / 0.04)", color: "var(--text-primary)", outline: "none" }} />
          <button type="button" disabled={!newName.trim()} onClick={() => newName.trim() && onNameNew(newName.trim())}
            className="btn-primary" style={{ padding: "8px 14px", opacity: newName.trim() ? 1 : 0.5 }}>
            <UserPlus size={12} /> Create
          </button>
        </div>

        {persons.length > 0 && (
          <>
            <div style={{ fontSize: 10, fontWeight: 700, color: "var(--text-tertiary)", textTransform: "uppercase",
              letterSpacing: 0.05, margin: "0 0 8px" }}>Or assign to enrolled person</div>
            <PersonPickList persons={persons} onPick={k => onPickExisting(k.id, k.name)} />
          </>
        )}
        <button type="button" onClick={onClose}
          style={{ marginTop: 14, width: "100%", padding: "8px 0", borderRadius: 10, fontSize: 12, fontWeight: 600,
            cursor: "pointer", border: "1px solid var(--border-strong)", background: "transparent", color: "var(--text-secondary)" }}>
          Cancel
        </button>
      </div>
    </div>,
    document.body,
  );
}

function TrackedCard({ p, persons, onName, onConfirm, onCorrect }: {
  p: TrackedPerson;
  persons: KnownPerson[];
  onName: (p: TrackedPerson) => void;
  onConfirm: (bodyId: string, knownId: string, name: string) => void;
  onCorrect: (p: TrackedPerson, wrongName: string) => void;
}) {
  const suggestion = (p.suggested_person_id ? persons.find(k => k.id === p.suggested_person_id) : undefined)
    ?? (p.suggested_name ? persons.find(k => k.name === p.suggested_name) : undefined);
  const streamInfo = useStore(s => s.streamInfo); // '@crop' markers → URL-served crops
  // A tracked body is an IDENTITY, so it gets the identity card the roster and
  // the stranger clusters use — it was a 52px horizontal row, the only one of
  // the three shaped that way.
  return (
    <ProfileCard
      title={`${p.known_name ?? p.label} · last seen ${fmtWhen(p.last_seen)}`}
      media={
        <ProfileMedia aspect="3 / 4" fallback={<Users size={22} />}
          src={p.thumbnail ? (bodyCropSrc(p.thumbnail, p.person_id, streamInfo) ?? undefined) : undefined}>
          <CardCount>×{p.sighting_count}</CardCount>
        </ProfileMedia>
      }>
      <div style={{ fontWeight: 700, fontSize: 13, display: "flex", alignItems: "center",
        gap: 6, flexWrap: "wrap", marginBottom: 2 }}>
        {p.known_name ?? p.label}
        {p.known_name && (
          <span title="Recognised by body appearance (anchored by this person's face). Reliable same-day."
            style={{ fontSize: 9, fontWeight: 700, padding: "2px 7px", borderRadius: 999,
              background: "var(--hl)", color: "var(--text-secondary)", border: "1px solid var(--hl-edge)",
              display: "inline-flex", alignItems: "center", gap: 3, whiteSpace: "nowrap" }}>
            <Activity size={9} /> by appearance
          </span>
        )}
        {p.known_name && (
          <button type="button" onClick={() => onCorrect(p, p.known_name!)}
            title="Wrong person? Teach the system it's not them (this correction sticks)."
            style={{ fontSize: 9, fontWeight: 700, padding: "2px 7px", borderRadius: 999, cursor: "pointer",
              background: "transparent", color: "var(--text-tertiary)", border: "1px solid var(--border-strong)" }}>
            Wrong?
          </button>
        )}
      </div>
        <div style={{ fontSize: 11, color: "var(--text-secondary)" }}>
          {p.sighting_count} sighting{p.sighting_count === 1 ? "" : "s"} · {p.cameras.length} camera{p.cameras.length === 1 ? "" : "s"}
        </div>
        {p.outfit && <OutfitLine outfit={p.outfit} />}
        <div style={{ fontSize: 10, color: "var(--text-tertiary)", marginTop: 3 }}>
          last seen {fmtWhen(p.last_seen)}
        </div>
        {!p.known_name && (
          <div style={{ marginTop: 8, display: "flex", gap: 6, flexWrap: "wrap", alignItems: "center" }}>
            {suggestion && (
              <button type="button" onClick={() => onConfirm(p.person_id, suggestion.id, suggestion.name)}
                title="Confirm this is the suggested person"
                style={{ display: "inline-flex", alignItems: "center", gap: 4, padding: "4px 9px", borderRadius: 999,
                  fontSize: 10.5, fontWeight: 700, cursor: "pointer",
                  border: "1px solid var(--accent)", background: "var(--accent-glow)", color: "var(--accent)" }}>
                <Check size={11} /> Looks like {suggestion.name}
              </button>
            )}
            {suggestion && (
              <button type="button" onClick={() => onCorrect(p, suggestion.name)}
                title={`Not ${suggestion.name} — stops suggesting them for this appearance`}
                style={{ display: "inline-flex", alignItems: "center", gap: 4, padding: "4px 9px", borderRadius: 999,
                  fontSize: 10.5, fontWeight: 600, cursor: "pointer",
                  border: "1px solid var(--border-strong)", background: "transparent", color: "var(--text-tertiary)" }}>
                <X size={11} /> Not {suggestion.name}
              </button>
            )}
            <button type="button" onClick={() => onName(p)}
              style={{ display: "inline-flex", alignItems: "center", gap: 4, padding: "4px 9px", borderRadius: 999,
                fontSize: 10.5, fontWeight: 600, cursor: "pointer",
                border: "1px solid var(--border-strong)", background: "transparent", color: "var(--text-primary)" }}>
              <UserPlus size={11} /> This is…
            </button>
          </div>
        )}
    </ProfileCard>
  );
}

// Pick an enrolled person to bind an anonymous tracked body to (the "train" action),
// OR — in correction mode (title/onNone given) — to re-attribute a mislabeled track.
function NameTrackedModal({ person, persons, onPick, onClose, title, note, onNone }: {
  person: TrackedPerson;
  persons: KnownPerson[];
  onPick: (knownId: string, name: string) => void;
  onClose: () => void;
  title?: string;
  note?: ReactNode;
  onNone?: () => void;
}) {
  useDismiss(onClose);
  return createPortal(
    <div onClick={onClose} style={{
      position: "fixed", inset: 0, background: "rgba(0,0,0,0.6)", backdropFilter: "blur(4px)",
      display: "flex", alignItems: "center", justifyContent: "center", zIndex: 1000, padding: 20,
    }}>
      <div onClick={e => e.stopPropagation()} className="glass" style={{ width: 420, maxWidth: "100%", padding: 20 }}>
        <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 4 }}>{title ?? "Who is this?"}</div>
        {/* Evidence strip: crop + outfit so the naming decision is grounded in
            what this track actually LOOKS like, not just an abstract label. */}
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
          {person.thumbnail && (
            <img src={bodyCropSrc(person.thumbnail, person.person_id, useStore.getState().streamInfo) ?? undefined} alt={person.label}
              style={{ width: 44, height: 44, borderRadius: 10, objectFit: "cover",
                border: "1px solid var(--border)" }} />
          )}
          <div style={{ minWidth: 0 }}>
            <div style={{ fontSize: 12, fontWeight: 700 }}>{person.label}</div>
            {person.outfit && <OutfitLine outfit={person.outfit} />}
          </div>
        </div>
        <div style={{ fontSize: 11.5, color: "var(--text-secondary)", marginBottom: 14, lineHeight: 1.5 }}>
          {note ?? (<>Tag <strong>{person.label}</strong> as an enrolled person.</>)}
        </div>
        {onNone && (
          <button type="button" onClick={onNone}
            style={{ width: "100%", marginBottom: 10, padding: "8px 0", borderRadius: 10, fontSize: 12, fontWeight: 700,
              cursor: "pointer", border: "1px solid var(--border-strong)", background: "transparent", color: "var(--text-secondary)" }}>
            No one / not in my people
          </button>
        )}
        {persons.length === 0 ? (
          <div style={{ fontSize: 12, color: "var(--text-tertiary)", padding: "10px 0" }}>
            No enrolled people yet — enroll a face from the People header first, then come back to tag this body.
          </div>
        ) : (
          <PersonPickList persons={persons} onPick={k => onPick(k.id, k.name)} />
        )}
        <button type="button" onClick={onClose}
          style={{ marginTop: 14, width: "100%", padding: "8px 0", borderRadius: 10, fontSize: 12, fontWeight: 600,
            cursor: "pointer", border: "1px solid var(--border-strong)", background: "transparent", color: "var(--text-secondary)" }}>
          Cancel
        </button>
      </div>
    </div>,
    document.body,
  );
}

// ── Roster ──────────────────────────────────────────────────────────────────
