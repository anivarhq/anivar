/**
 * Shared pieces of the People section.
 *
 * These were defined inline in a 2,900-line PersonsPanel and used from three or
 * four places each, so every split of that file had to either duplicate them or
 * import backwards out of the panel. They live here instead.
 *
 * Nothing here holds state that outlives a render except the dialogs' own local
 * UI state — the People sections all funnel mutations back through the panel's
 * single `refresh()`.
 */
import { useEffect, useMemo, useRef, useState, type ReactNode, type CSSProperties } from "react";
import { createPortal } from "react-dom";
import { Check, X, Users } from "lucide-react";
import { api, KnownPerson } from "../../api";
import { useStore } from "../../store";
import { faceCropSrc } from "../../lib/eventThumb";
import { Modal, useDismiss } from "../../components/ui/Modal";
import { CardEmpty } from "../review/Card";
import { OBJECT_COLOR_SWATCH } from "../../lib/palette";

/**
 * Removing a person is two different operations that used to be one button.
 *
 * "I mislabelled this face" wants the crops KEPT so they can be re-tagged —
 * that is `deletePerson`, and it stays the default because it is the common
 * case and the recoverable one.
 *
 * "This person asked to be deleted" wants the biometrics DESTROYED. Face
 * descriptors are special-category data under GDPR/UK-DPA and unlinking a name
 * does not satisfy an erasure request — the descriptor still identifies them.
 * That is `forgetPerson`, and it is deliberately the second, explicit answer.
 *
 * This was two chained native `confirm()` calls, where OK on the SECOND was the
 * destructive branch — a browser dialog whose default action erased biometrics.
 * Both choices are now buttons that say what they do.
 */
export function RemovePersonDialog({ person, onDone, onCancel, showToast }: {
  person: { id: string; name: string };
  onDone: () => void;
  onCancel: () => void;
  showToast: (msg: string, type?: "success" | "error" | "info") => void;
}) {
  const [busy, setBusy] = useState(false);
  const run = async (erase: boolean) => {
    setBusy(true);
    try {
      if (!erase) {
        await api.deletePerson(person.id);
        showToast(`${person.name} removed`, "info");
      } else {
        const n = await api.forgetPerson(person.id);
        showToast(
          `${person.name} erased — ${n} biometric record${n === 1 ? "" : "s"} destroyed`,
          "info");
      }
      onDone();
    } catch (e) {
      showToast(`Couldn't remove ${person.name}: ${String(e)}`, "error");
      setBusy(false);
    }
  };

  return (
    <Modal onClose={onCancel} layer="top" closeOnBackdrop={!busy}>
      <div className="glass" style={{
        padding: 20, borderRadius: 14, width: 440, maxWidth: "100%",
        display: "flex", flexDirection: "column", gap: 12,
      }}>
        <div style={{ fontWeight: 700, fontSize: 14 }}>Remove {person.name}?</div>
        <div style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.6 }}>
          Recorded footage is not affected either way — it expires on the
          retention schedule.
        </div>
        <button disabled={busy} onClick={() => run(false)} style={{
          padding: "10px 14px", borderRadius: 12, textAlign: "left", cursor: busy ? "wait" : "pointer",
          border: "1px solid var(--border)", background: "rgb(var(--ink) / 0.03)",
          color: "var(--text-primary)",
        }}>
          <div style={{ fontWeight: 700, fontSize: 13 }}>Remove the name</div>
          <div style={{ fontSize: 11, color: "var(--text-tertiary)", marginTop: 2 }}>
            Keeps the face crops so you can re-label them later.
          </div>
        </button>
        <button disabled={busy} onClick={() => run(true)} style={{
          padding: "10px 14px", borderRadius: 12, textAlign: "left", cursor: busy ? "wait" : "pointer",
          border: "1px solid color-mix(in srgb, var(--status-alert) 35%, transparent)",
          background: "transparent", color: "var(--accent-red)",
        }}>
          <div style={{ fontWeight: 700, fontSize: 13 }}>Erase biometric data</div>
          <div style={{ fontSize: 11, opacity: 0.85, marginTop: 2 }}>
            Permanently destroys every face descriptor, crop, body-appearance
            vector and sighting. Cannot be undone. Use this for a data-deletion
            request.
          </div>
        </button>
        <button disabled={busy} onClick={onCancel} style={{
          alignSelf: "flex-end", padding: "6px 14px", borderRadius: 999,
          fontSize: 12, fontWeight: 700, border: "1px solid var(--border)",
          background: "var(--bg-elevated)", color: "var(--text-primary)",
          cursor: busy ? "wait" : "pointer",
        }}>Cancel</button>
      </div>
    </Modal>
  );
}

// Assign a proposed group to an existing person, OR name a brand-new body-only
// ("no face yet") identity from it.
/**
 * PersonPickList — "which of your people is this?", once.
 *
 * There were four hand-built copies of this list across the tagging modals, and
 * NONE of them had a search box — while the roster grid, which needs it least,
 * did. At thirty enrolled people every tag became a scroll hunt through an
 * unfiltered 320px box.
 *
 * The search input only appears once the list is long enough to need one;
 * showing a filter over four names is clutter.
 */
export function PersonPickList({ persons, onPick, disabled = false, emptyHint }: {
  persons: KnownPerson[];
  onPick: (person: KnownPerson) => void;
  disabled?: boolean;
  emptyHint?: React.ReactNode;
}) {
  const [q, setQ] = useState("");
  const shown = useMemo(() => {
    const t = q.trim().toLowerCase();
    return t ? persons.filter(k => k.name.toLowerCase().includes(t)) : persons;
  }, [persons, q]);

  if (persons.length === 0) {
    return (
      <div style={{ fontSize: 12, color: "var(--text-tertiary)", padding: "10px 0" }}>
        {emptyHint ?? "No enrolled people yet — enroll a face first, then come back."}
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      {persons.length > 6 && (
        <input
          value={q}
          onChange={e => setQ(e.target.value)}
          placeholder={`Search ${persons.length} people…`}
          autoFocus
          style={{
            padding: "7px 10px", borderRadius: 10, fontSize: 12,
            border: "1px solid var(--border)", background: "var(--bg-surface)",
            color: "var(--text-primary)", outline: "none",
          }} />
      )}
      <div style={{ display: "flex", flexDirection: "column", gap: 6, maxHeight: 320, overflowY: "auto" }}>
        {shown.map(k => (
          <button key={k.id} type="button" disabled={disabled} onClick={() => onPick(k)}
            style={{
              display: "flex", alignItems: "center", gap: 10, padding: "8px 10px", borderRadius: 10,
              cursor: disabled ? "wait" : "pointer", textAlign: "left",
              border: "1px solid var(--border)", background: "rgb(var(--ink) / 0.03)",
              color: "var(--text-primary)",
            }}>
            {k.thumbnail
              ? <img src={k.thumbnail.startsWith("data:") ? k.thumbnail : `data:image/jpeg;base64,${k.thumbnail}`}
                  alt={k.name} style={{ width: 30, height: 30, borderRadius: 8, objectFit: "cover" }} />
              : <span style={{ fontSize: 20 }}>👤</span>}
            <span style={{ fontWeight: 700, fontSize: 13, flex: 1 }}>{k.name}</span>
            <span style={{ fontSize: 10, color: "var(--text-tertiary)", textTransform: "capitalize" }}>{k.role}</span>
          </button>
        ))}
        {shown.length === 0 && (
          <div style={{ fontSize: 12, color: "var(--text-tertiary)", padding: "10px 2px" }}>
            Nobody matches “{q}”.
          </div>
        )}
      </div>
    </div>
  );
}

/** A thumbnail that expands into a full-screen lightbox on click. The stored
 *  face/body crops are small, so the cards render them tiny — this lets the user
 *  click to see the person at full size. Portaled to <body> so the overlay is
 *  never clipped by a card's overflow/stacking context. */
export function ZoomableImg({ src, alt, caption, style }: {
  src: string; alt?: string; caption?: string; style?: CSSProperties;
}) {
  const [open, setOpen] = useState(false);
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);
  return (
    <>
      <img src={src} alt={alt ?? ""} loading="lazy" title="Click to enlarge"
        onClick={e => { e.stopPropagation(); setOpen(true); }}
        style={{ cursor: "zoom-in", ...style }} />
      {open && createPortal(
        <div onClick={e => { e.stopPropagation(); setOpen(false); }}
          style={{
            position: "fixed", inset: 0, zIndex: 3000, background: "rgba(5,4,4,0.9)",
            backdropFilter: "blur(4px)",
            display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
            gap: 14, padding: 28, cursor: "zoom-out",
          }}>
          <img src={src} alt={alt ?? ""} style={{
            maxWidth: "min(92vw, 520px)", maxHeight: "80vh", objectFit: "contain",
            borderRadius: 16, boxShadow: "0 24px 70px rgba(0,0,0,0.6)",
            border: "1px solid rgb(var(--ink) / 0.1)",
          }} />
          {caption && <div style={{ fontSize: 13, fontWeight: 600, color: "rgb(var(--ink) / 0.9)" }}>{caption}</div>}
          <div style={{ fontSize: 11, color: "rgb(var(--ink) / 0.5)" }}>click anywhere or press Esc to close</div>
        </div>,
        document.body,
      )}
    </>
  );
}

/** A face crop that expands into the FULL frame it was captured in (lazy-fetched
 *  via get_face_context — the whole scene, not just the tight face). Mirrors
 *  ZoomableImg's portal/Escape pattern. */
export function FaceContextZoom({ faceId, thumb }: { faceId: string; thumb: string }) {
  const [open, setOpen] = useState(false);
  const [full, setFull] = useState<string | null>(null);
  const streamInfo = useStore(s => s.streamInfo);
  // '@crop' marker → URL-served crop (legacy inline base64 still renders).
  const src = faceCropSrc(thumb, faceId, streamInfo) ?? "";
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
    window.addEventListener("keydown", onKey);
    if (full === null) api.getFaceContext(faceId).then(c => setFull(c ?? "")).catch(() => setFull(""));
    return () => window.removeEventListener("keydown", onKey);
  }, [open, faceId, full]);
  return (
    <>
      <img src={src} alt="" loading="lazy" title="Click to see the full scene"
        onClick={e => { e.stopPropagation(); setOpen(true); }}
        style={{ width: "100%", aspectRatio: "1/1", objectFit: "cover", borderRadius: 8,
          border: "1px solid var(--border)", cursor: "zoom-in" }} />
      {open && createPortal(
        <div onClick={e => { e.stopPropagation(); setOpen(false); }}
          style={{ position: "fixed", inset: 0, zIndex: 3000, background: "rgba(5,4,4,0.9)",
            backdropFilter: "blur(4px)", display: "flex", flexDirection: "column",
            alignItems: "center", justifyContent: "center", gap: 14, padding: 28, cursor: "zoom-out" }}>
          <img src={full ? `data:image/jpeg;base64,${full}` : src} alt="" style={{
            maxWidth: "92vw", maxHeight: "82vh", objectFit: "contain", borderRadius: 14,
            boxShadow: "0 24px 70px rgba(0,0,0,0.6)", border: "1px solid rgb(var(--ink) / 0.1)" }} />
          <div style={{ fontSize: 11, color: "rgb(var(--ink) / 0.5)" }}>
            {full === null ? "loading full frame…" : "the full scene · click anywhere or press Esc to close"}
          </div>
        </div>,
        document.body,
      )}
    </>
  );
}

export function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ flex: 1, minWidth: 78, padding: "8px 12px", borderRadius: 12,
      background: "rgb(var(--ink) / 0.03)", border: "1px solid var(--border)" }}>
      <div style={{ fontSize: 16, fontWeight: 800, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{value}</div>
      <div style={{ fontSize: 9, fontWeight: 700, textTransform: "uppercase", letterSpacing: 0.05, color: "var(--text-tertiary)" }}>{label}</div>
    </div>
  );
}

/** `hint` is the actionable line; `more` is the explanation, on hover only. */
export function Section({ icon, title, hint, more, children }: {
  icon: ReactNode; title: string; hint?: string; more?: string; children: ReactNode;
}) {
  return (
    <div title={more} style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 7, fontSize: 11, fontWeight: 700,
        letterSpacing: 0.04, textTransform: "uppercase", color: "var(--text-tertiary)" }}>{icon} {title}</div>
      {hint && <div style={{ fontSize: 10, color: "var(--text-tertiary)", marginTop: -4, lineHeight: 1.45 }}>{hint}</div>}
      {children}
    </div>
  );
}

/** One empty state for the whole app — the same icon-over-sentence block Review
 *  shows. People had three unrelated shapes for "there is nothing here". */
export function Empty({ text }: { text: string }) {
  return <CardEmpty>{text}</CardEmpty>;
}

export function summaryText(ai: string | null): string {
  if (!ai) return "Motion event";
  try { const o = JSON.parse(ai); return o.text || o.description || "Event"; } catch { return ai; }
}

export function formatRelative(d: Date): string {
  const diff = (Date.now() - d.getTime()) / 1000;
  if (diff < 60)        return "just now";
  if (diff < 3600)      return `${Math.round(diff / 60)}m ago`;
  if (diff < 86400)     return `${Math.round(diff / 3600)}h ago`;
  if (diff < 86400 * 7) return `${Math.round(diff / 86400)}d ago`;
  return d.toLocaleDateString();
}

/** "Ravi" · "Maybe Priya?" · "Unfamiliar" — identity in words, never a
 *  percentage. A body match only ever reads as a question. */
export function whoLabel(v: { name: string | null; maybe_name: string | null }): string {
  return v.name ?? (v.maybe_name ? `Maybe ${v.maybe_name}?` : "Unfamiliar");
}

/** Local "08:02–08:15" (one time when both ends fall in the same minute). */
export function timeSpan(start: string, end: string): string {
  const f = (iso: string) => new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  const a = f(start), b = f(end);
  return a === b ? a : `${a}–${b}`;
}

/** "Front Door → Hall" — cameras in the order the person passed them. */
export function cameraPath(cams: number[], cameraName: (id: number) => string): string {
  return cams.map(cameraName).join(" → ");
}

const BEHAVIOUR_PHRASE: Record<string, string> = {
  loitering: "loitered", intrusion: "entered a zone", running: "ran",
  person_down: "was down on the ground", climbing: "climbed", crowd: "in a crowd",
};
export function behaviourPhrase(b: string): string {
  return BEHAVIOUR_PHRASE[b] ?? b.replace(/_/g, " ");
}

/** "Today" / "Yesterday" / "Sep 12" for a LOCAL YYYY-MM-DD key. (The `fmtDay` this
 *  replaced compared against UTC dates, which mislabels local evenings.) */
export function fmtLocalDay(day: string): string {
  const key = (d: Date) =>
    `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
  const now = new Date();
  if (day === key(now)) return "Today";
  if (day === key(new Date(now.getTime() - 86_400_000))) return "Yesterday";
  const d = new Date(`${day}T12:00:00`);
  return isNaN(d.getTime()) ? day : d.toLocaleDateString([], { month: "short", day: "numeric" });
}

/// CSS swatch for each classifier color word (11 bins from the HSV vote).
export const OUTFIT_DOT = OBJECT_COLOR_SWATCH;
