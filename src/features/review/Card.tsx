/**
 * The card shell every browser shares — Events, Sounds, Vehicles and People.
 *
 * `CardChrome` next door already unified the *overlay* widgets (the action
 * column, the (i) button, the info modal). What it deliberately did not own was
 * the card itself, so each surface hand-wrote the shell — and the pieces drifted
 * anyway, just one level down:
 *
 *   - the footer row (`time` left, `(i)` right) is byte-identical in ReviewFeed,
 *     VehiclesView and AudioView;
 *   - the grid is `.feed` in ReviewFeed but re-declared inline as
 *     `minmax(150px,1fr)` in Vehicles and Audio;
 *   - People hand-rolled six more card shapes with none of the above, in ~379
 *     inline style blocks, which is why it never looked like the rest of the app.
 *
 * Styling still comes from `ReviewFeed.module.css` rather than a new stylesheet.
 * That module is already the shared card sheet — Vehicles and Audio import it as
 * `cardStyles` — and moving ~60 class references for zero visual change is churn
 * with a real chance of breaking a working feed.
 *
 * The layout rule from `CardChrome`'s header still governs and is enforced by the
 * slots below: the media carries only PASSIVE marks — camera badge top-right,
 * risk pill bottom-right, count pill top-right, hover play glyph — while every
 * FUNCTIONAL icon goes in `actions`, the one hover-revealed column down the
 * top-left.
 */
import type { ReactNode } from "react";
import styles from "./ReviewFeed.module.css";
import { tint } from "../../lib/palette";

/* ── Container ─────────────────────────────────────────────────────────────── */

/** The dense auto-fill grid. `min` widens for roster cards, which carry text. */
export function CardGrid({ children, min = 150, scroll = true }: {
  children: ReactNode;
  min?: number;
  /** Feeds own their scroll; a section inside a scrolling page must not. */
  scroll?: boolean;
}) {
  // The grid template is always inline so `min` actually applies; `.feed`
  // contributes only the scroll container (flex/overflow/padding) it owns.
  return (
    <div
      className={scroll ? styles.feed : undefined}
      style={{
        display: "grid",
        gridTemplateColumns: `repeat(auto-fill, minmax(${min}px, 1fr))`,
        gap: 10,
        alignContent: "start",
      }}
    >
      {children}
    </div>
  );
}

/** The shared empty state — one icon over one sentence, centred in the grid. */
export function CardEmpty({ icon, children }: { icon?: ReactNode; children: ReactNode }) {
  return <div className={styles.empty}>{icon}<span>{children}</span></div>;
}

/* ── Card ──────────────────────────────────────────────────────────────────── */

/**
 * One card. A `<button>`, so it is focusable and Enter/Space work for free —
 * several of People's hand-rolled cards were click-only `<div>`s, and one had no
 * keyboard path at all.
 */
export function Card({ onClick, selected, title, children }: {
  onClick?: () => void;
  /** Keyboard/selection ring (`j`/`k` triage in Review, current item elsewhere). */
  selected?: boolean;
  title?: string;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      title={title}
      className={`${styles.card} ${selected ? styles.cardKb : ""}`}
      onClick={onClick}
    >
      {children}
    </button>
  );
}

/**
 * The media block. `aspect` is the one real variable between surfaces: events and
 * vehicles are 16:9, face crops are square.
 */
export function CardMedia({ src, fallback, aspect = "16 / 9", dim, children }: {
  src?: string | null;
  /** Shown when there is no image — a lucide icon, sized ~16. */
  fallback?: ReactNode;
  aspect?: string;
  /** Half-opacity, for "this event has no recorded video". */
  dim?: boolean;
  children?: ReactNode;
}) {
  return (
    <div className={styles.cardImg} style={{ aspectRatio: aspect, opacity: dim ? 0.5 : undefined }}>
      {src
        // A broken crop hides itself and leaves the black frame, which reads as
        // "no image" rather than as a broken-image glyph.
        ? <img src={src} alt="" loading="lazy" onError={e => { e.currentTarget.style.display = "none"; }} />
        : <div className={styles.cardImgBlank}>{fallback}</div>}
      {children}
    </div>
  );
}

/** Bottom-right pill. Risk level on events; anything short and categorical here. */
export function CardPill({ color, children }: { color?: string; children: ReactNode }) {
  return (
    <div className={styles.cardRisk} style={color ? { background: tint(color, 87) } : undefined}>
      {children}
    </div>
  );
}

/** Top-right badge. The camera label on events. */
export function CardBadge({ children }: { children: ReactNode }) {
  return <span className={styles.cardCam}>{children}</span>;
}

/**
 * Top-right count pill — "×14 sightings".
 *
 * Vehicles and People's unknown-face cards had independently converged on the
 * exact same declaration; this is that, named.
 */
export function CardCount({ children }: { children: ReactNode }) {
  return (
    <span style={{
      position: "absolute", top: 6, right: 6, padding: "2px 8px", borderRadius: 999,
      fontSize: 10, fontWeight: 800, background: "rgba(0,0,0,0.62)", color: "#fff",
    }}>{children}</span>
  );
}

/** Centred play glyph, revealed on card hover. */
export function CardPlay({ children }: { children: ReactNode }) {
  return <span className={styles.cardPlay}>{children}</span>;
}

/**
 * The footer row: something on the left (usually a timestamp), something on the
 * right (usually the (i) button). Written out identically in three files before
 * this existed.
 */
export function CardFooter({ left, right, children }: {
  left?: ReactNode;
  right?: ReactNode;
  children?: ReactNode;
}) {
  return (
    <div style={{
      display: "flex", alignItems: "center", justifyContent: "space-between",
      padding: "0 8px 4px", gap: 6,
    }}>
      {children ?? <>{left}<span style={{ flex: 1 }} />{right}</>}
    </div>
  );
}

/** Monospace timestamp, the standard left-hand footer content. */
export function CardTime({ children }: { children: ReactNode }) {
  return <span className={styles.cardTime} style={{ padding: 0 }}>{children}</span>;
}

/* ── Profile card ──────────────────────────────────────────────────────────── */

/**
 * The other shell the app already has: a persistent IDENTITY rather than a
 * sighting. Flush media on a `.glass` surface with a meta block beneath, exactly
 * as the vehicle roster renders a plate.
 *
 * People's roster, unknown clusters and tracked bodies are all this shape; each
 * had its own hand-written version.
 */
export function ProfileCard({ media, onClick, selected, title, children }: {
  media: ReactNode;
  onClick?: () => void;
  selected?: boolean;
  title?: string;
  /** The meta block: name, then one or two lines of detail. */
  children: ReactNode;
}) {
  // A div with role="button", NOT a <button>: identity cards carry their own
  // actions ("Confirm", "Not them") and a button inside a button is invalid HTML
  // that browsers resolve by breaking one of them. Enter and Space are handled
  // explicitly, which is what the element would have given us for free.
  return (
    <div
      role={onClick ? "button" : undefined}
      tabIndex={onClick ? 0 : undefined}
      title={title}
      className="glass"
      onClick={onClick}
      onKeyDown={onClick ? (e) => {
        if (e.target !== e.currentTarget) return;   // let nested controls answer first
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onClick(); }
      } : undefined}
      style={{
        padding: 0, overflow: "hidden", textAlign: "left", cursor: onClick ? "pointer" : "default",
        border: selected ? "1.5px solid var(--accent)" : undefined,
        boxShadow: selected ? "0 0 0 1.5px var(--accent)" : undefined,
        transition: "transform 160ms var(--spring, ease), border-color 140ms ease",
      }}
      onMouseEnter={onClick ? (e) => { e.currentTarget.style.transform = "translateY(-2px)"; } : undefined}
      onMouseLeave={onClick ? (e) => { e.currentTarget.style.transform = "translateY(0)"; } : undefined}
    >
      <div style={{ position: "relative" }}>{media}</div>
      <div style={{ padding: "8px 10px" }}>{children}</div>
    </div>
  );
}

/** Flush media for `ProfileCard` — no inner radius, fills the card's top edge. */
export function ProfileMedia({ src, fallback, aspect = "16 / 10", children }: {
  src?: string | null;
  fallback?: ReactNode;
  aspect?: string;
  children?: ReactNode;
}) {
  return (
    <>
      {src
        ? <img src={src} alt="" loading="lazy"
            onError={e => { e.currentTarget.style.display = "none"; }}
            style={{ width: "100%", aspectRatio: aspect, objectFit: "cover", display: "block" }} />
        : <div style={{
            width: "100%", aspectRatio: aspect, background: "rgb(var(--ink) / 0.05)",
            display: "flex", alignItems: "center", justifyContent: "center",
            color: "rgb(var(--ink) / 0.25)",
          }}>{fallback}</div>}
      {children}
    </>
  );
}
