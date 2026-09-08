/**
 * Shared chrome for every review card — Events, Sounds and Vehicles.
 *
 * Why this exists: all three browsers render the SAME card skeleton
 * (`ReviewFeed.module.css` `.card`) but each hand-rolled its own absolutely
 * positioned overlay buttons, and the positions had drifted apart:
 *
 *   - the bookmark sat at top/left 6 on all three, directly on top of the
 *     `.cardCam` badge at top/left 3;
 *   - delete was at `right: 34` on Events but the default `left: 34` on Sounds
 *     and Vehicles — opposite corners for the same action;
 *   - Sounds and Vehicles each put a stat pill at bottom/right 6, on top of the
 *     `.cardRisk` pill at bottom/right 3;
 *   - only Events had an (i) button at all.
 *
 * The rule now: the thumbnail carries only **passive** marks — the camera badge
 * (top-RIGHT), the risk/label pill (bottom-right) and the hover play glyph. Every
 * **functional** icon lives in ONE vertical column down the top-LEFT edge,
 * uniform size and spacing, revealed on hover so a wall of cards reads as
 * pictures rather than as a grid of buttons. The one exception is a *saved*
 * bookmark, which stays lit because that is state, not an action — hiding it
 * would lose the signal.
 *
 * The column is why the camera badge moved to the right and why the person-mode
 * crop moved into the footer: both used to occupy the left edge.
 *
 * Everything descriptive goes behind the (i) in the footer, which is identical
 * on all three card types.
 */
import { useState, type ReactNode } from "react";
import { Trash2, Bookmark, Sparkles, Download, Info, X, MapPin, Clock, Film } from "lucide-react";
import styles from "./ReviewFeed.module.css";

/** One circular action button. Uniform across every card type. */
function ActionBtn({ title, onClick, active, danger, pinned, children }: {
  title: string;
  onClick: (e: React.MouseEvent) => void;
  active?: boolean;
  danger?: boolean;
  /** Stay visible without hover (used for a saved bookmark — that's state). */
  pinned?: boolean;
  children: ReactNode;
}) {
  return (
    <span
      role="button"
      tabIndex={0}
      title={title}
      className={pinned ? styles.cardActionPinned : undefined}
      onClick={(e) => { e.stopPropagation(); e.preventDefault(); onClick(e); }}
      // role="button" + tabIndex made this focusable but NOT activatable: there
      // was no key handler at all, so tabbing to a card action and pressing
      // Enter did nothing. Both keys, because that is what a button promises.
      onKeyDown={(e) => {
        if (e.key !== "Enter" && e.key !== " ") return;
        e.stopPropagation(); e.preventDefault();
        onClick(e as unknown as React.MouseEvent);
      }}
      style={{
        display: "inline-flex", alignItems: "center", justifyContent: "center",
        width: 18, height: 18, borderRadius: 999, cursor: "pointer", flexShrink: 0,
        background: danger ? "var(--status-alert)" : "rgba(0,0,0,0.55)",
        color: active ? "var(--status-idle)" : "#fff",
        border: `1px solid ${
          danger ? "var(--status-alert)"
          : active ? "color-mix(in srgb, var(--status-idle) 70%, transparent)"
          : "rgb(var(--ink) / 0.25)"}`,
        backdropFilter: "blur(4px)",
      }}>
      {children}
    </span>
  );
}

/**
 * The single action row. Renders only the actions a surface actually supports —
 * omitted handlers simply don't produce a button, so the row stays as short as
 * the card's real capabilities.
 */
export function CardActions({
  saved, onBookmark, onSimilar, onDownload, onDelete, deleteTitle, bookmarkTitle,
}: {
  saved?: boolean;
  onBookmark?: () => void;
  onSimilar?: () => void;
  onDownload?: () => void;
  onDelete?: () => void;
  deleteTitle?: string;
  bookmarkTitle?: string;
}) {
  // Arm-then-confirm lives here now (was DeleteTileButton): one stray click can
  // never delete, and leaving the card disarms it.
  const [armed, setArmed] = useState(false);
  return (
    <span className={styles.cardActions} onMouseLeave={() => setArmed(false)}>
      {onBookmark && (
        <ActionBtn
          title={bookmarkTitle ?? (saved ? "Remove bookmark" : "Bookmark")}
          onClick={onBookmark} active={saved} pinned={saved}>
          <Bookmark size={10} fill={saved ? "var(--status-idle)" : "none"} />
        </ActionBtn>
      )}
      {onSimilar && (
        <ActionBtn title="Find visually similar events" onClick={onSimilar}>
          <Sparkles size={10} />
        </ActionBtn>
      )}
      {onDownload && (
        <ActionBtn title="Download clip" onClick={onDownload}>
          <Download size={10} />
        </ActionBtn>
      )}
      {onDelete && (
        <ActionBtn
          title={armed ? "Click again to delete — this can't be undone" : (deleteTitle ?? "Delete")}
          danger={armed}
          onClick={() => { if (armed) onDelete(); else setArmed(true); }}>
          <Trash2 size={10} />
        </ActionBtn>
      )}
    </span>
  );
}

/** The (i) that opens the details sheet. Sits in the card footer, right side. */
export function CardInfoButton({ onOpen }: { onOpen: () => void }) {
  return (
    <span
      role="button" tabIndex={0} title="Details"
      onClick={(e) => { e.stopPropagation(); e.preventDefault(); onOpen(); }}
      // Space too — a control announced as a button answers both.
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") { e.stopPropagation(); e.preventDefault(); onOpen(); }
      }}
      style={{
        display: "inline-flex", alignItems: "center", justifyContent: "center",
        width: 22, height: 22, cursor: "pointer", background: "transparent",
        border: "none", color: "rgb(var(--ink) / 0.92)", flexShrink: 0,
      }}>
      <Info size={15} strokeWidth={2.2} />
    </span>
  );
}

export interface CardChip { icon: string; text: string }

const CHIP: React.CSSProperties = {
  display: "inline-flex", alignItems: "center", gap: 5,
  fontSize: 11, fontWeight: 700, padding: "3px 9px", borderRadius: 999,
  background: "color-mix(in srgb, var(--text-primary) 8%, transparent)",
  color: "var(--text-secondary)", border: "1px solid var(--border)",
};

/**
 * The details sheet behind the (i) — one implementation for all three browsers.
 * Everything that used to be crammed onto the thumbnail or the footer (sound
 * class chips, loudness, confidence, plate, detected labels) belongs here.
 */
export function CardInfoModal({
  onClose, thumb, thumbFallback, title, camLabel, timeLabel, chips, summary, detail,
}: {
  onClose: () => void;
  thumb?: string | null;
  thumbFallback?: ReactNode;
  title: string;
  camLabel: string;
  timeLabel: string;
  chips?: CardChip[];
  summary?: string | null;
  detail?: ReactNode;
}) {
  return (
    <div onClick={onClose} style={{
      position: "fixed", inset: 0, zIndex: 1200, background: "var(--bg-overlay)",
      display: "flex", alignItems: "center", justifyContent: "center", padding: 20,
    }}>
      <div className="glass-strong" onClick={e => e.stopPropagation()}
        style={{ width: "100%", maxWidth: 420, padding: 18, display: "flex", flexDirection: "column", gap: 14 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          {thumb
            ? <img src={thumb} alt="" loading="lazy"
                style={{ width: 84, height: 60, borderRadius: 10, objectFit: "cover" }} />
            : <div style={{ width: 84, height: 60, borderRadius: 10, background: "rgb(var(--ink) / 0.05)",
                display: "flex", alignItems: "center", justifyContent: "center" }}>
                {thumbFallback ?? <Film size={16} />}
              </div>}
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ fontWeight: 700, fontSize: 14 }}>{title}</div>
            <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 3,
              display: "flex", alignItems: "center", gap: 5, flexWrap: "wrap" }}>
              <MapPin size={11} /> {camLabel} · <Clock size={11} /> {timeLabel}
            </div>
          </div>
          <button onClick={onClose} style={{ background: "none", border: "none", cursor: "pointer",
            color: "var(--text-muted)", padding: 4 }}><X size={18} /></button>
        </div>
        {chips && chips.length > 0 && (
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
            {chips.map((c, i) => <span key={i} style={CHIP}>{c.icon} {c.text}</span>)}
          </div>
        )}
        {summary && (
          <div style={{ fontSize: 12.5, lineHeight: 1.55, color: "var(--text-secondary)" }}>{summary}</div>
        )}
        {detail}
      </div>
    </div>
  );
}
