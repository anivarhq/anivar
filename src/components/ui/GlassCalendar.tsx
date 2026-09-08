/**
 * GlassCalendar — themed date picker popover.
 *
 * Drop-in replacement for `<input type="date">` + `.showPicker()`, which
 * opens Chromium's native widget that we can't dark-theme deeply.
 *
 * Behaviour:
 *   • Pops below its anchor button on open.
 *   • Click outside or Escape closes.
 *   • Future dates can be locked (NVR records the past, not the future).
 *   • Returns `YYYY-MM-DD` in local time — not UTC — matching the rest of
 *     the codebase (see `localDateStr` in NVRPanel).
 *
 * Visual layer: uses the global `.glass` class for the popover — the SAME
 * frosted surface as the toolbar filter dropdowns — plus the existing accent
 * tokens for the cells. No new design tokens.
 */

import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { localDateStr } from "../../lib/time";

interface Props {
  /** Currently-selected date in `YYYY-MM-DD` (local time). */
  value: string;
  /** Called with the new date when the user picks one. */
  onChange: (value: string) => void;
  /** Optional max date (`YYYY-MM-DD`) — days after this render disabled. */
  max?: string;
  /** Optional min date. */
  min?: string;
  /** True while the popover should be visible. */
  open: boolean;
  /** Fired when user clicks outside or hits Escape. */
  onClose: () => void;
  /** The trigger element — used to anchor the popover. */
  anchorRef: React.RefObject<HTMLElement | null>;
  /** Local days (`YYYY-MM-DD`) that hold recorded footage. Those cells LIFT; the
   *  rest stay flat, so "where is there anything to watch" is answerable at a
   *  glance instead of by clicking days at random. Omit to mark nothing. */
  recordedDays?: Set<string>;
}

const DOW = ["S", "M", "T", "W", "T", "F", "S"];

// localDateStr comes from the shared time SSOT (../../lib/time).
function parseLocal(s: string): Date {
  // T12:00:00 avoids DST-driven day shifts at the local-midnight boundary.
  return new Date(s + "T12:00:00");
}

export function GlassCalendar({ value, onChange, max, min, open, onClose, anchorRef, recordedDays }: Props) {
  const popRef = useRef<HTMLDivElement>(null);
  const [viewMonth, setViewMonth] = useState(() => parseLocal(value));

  // Keep view in sync when the selected date changes from outside.
  useEffect(() => { setViewMonth(parseLocal(value)); }, [value]);

  // Click-outside + Escape close.
  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      const pop = popRef.current; const anchor = anchorRef.current;
      if (!pop) return;
      if (pop.contains(e.target as Node)) return;
      if (anchor && anchor.contains(e.target as Node)) return;
      onClose();
    };
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    document.addEventListener("mousedown", onDoc);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDoc);
      document.removeEventListener("keydown", onKey);
    };
  }, [open, onClose, anchorRef]);

  if (!open) return null;

  // Compute popover position relative to the viewport.
  // Note: `position: fixed` is normally viewport-relative, but ANY ancestor
  // with `backdrop-filter`, `transform`, `filter`, or `perspective` creates a
  // new containing block (CSS spec) and reroots fixed coords to it. The NVR
  // header is one such ancestor, so we portal to document.body to escape.
  const anchorRect = anchorRef.current?.getBoundingClientRect();
  const popWidth   = 256;
  const top        = anchorRect ? anchorRect.bottom + 6 : 0;
  // Keep left edge aligned with the anchor; clamp so the popover never
  // overflows the right side of the viewport (8px gutter).
  const rawLeft    = anchorRect ? anchorRect.left : 0;
  const left       = Math.min(Math.max(8, rawLeft), window.innerWidth - popWidth - 8);

  // Month grid — start on Sunday of the week containing the 1st.
  const first      = new Date(viewMonth.getFullYear(), viewMonth.getMonth(), 1);
  const startOfGrid = new Date(first);
  startOfGrid.setDate(first.getDate() - first.getDay());
  const cells: Date[] = [];
  for (let i = 0; i < 42; i++) {
    const d = new Date(startOfGrid);
    d.setDate(startOfGrid.getDate() + i);
    cells.push(d);
  }

  const today = localDateStr(new Date());
  const sel   = value;
  const monthLabel = viewMonth.toLocaleDateString([], { month: "long", year: "numeric" });

  return createPortal(
    // Frosted surface = the SAME global `.glass` class the section's filter
    // dropdowns use (tint/blur/border/shadow all come from it), so the
    // calendar reads as one family with them everywhere it pops — Review
    // toolbar, the player's date button, the focus header. Only layout is
    // inline; radius matches the dropdowns' 14px.
    <div ref={popRef} className="glass" style={{
      position: "fixed", top, left, zIndex: 1000,
      padding: 14, width: popWidth,
      borderRadius: 14,
      display: "flex", flexDirection: "column", gap: 10,
      animation: "fade-in 160ms cubic-bezier(0.16,1,0.3,1)",
    }}>
      {/* Month header */}
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <button type="button" onClick={() => setViewMonth(new Date(viewMonth.getFullYear(), viewMonth.getMonth() - 1, 1))}
          style={navBtn}><ChevronLeft size={14} /></button>
        <div style={{
          flex: 1, textAlign: "center",
          fontSize: 13, fontWeight: 700, letterSpacing: -0.01,
          color: "var(--text-primary)",
        }}>{monthLabel}</div>
        <button type="button" onClick={() => setViewMonth(new Date(viewMonth.getFullYear(), viewMonth.getMonth() + 1, 1))}
          style={navBtn}><ChevronRight size={14} /></button>
      </div>

      {/* DoW row */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(7, 1fr)", gap: 2 }}>
        {DOW.map((d, i) => (
          <div key={i} style={{
            textAlign: "center", fontSize: 9, fontWeight: 700,
            color: "var(--text-muted)", letterSpacing: 0.04,
            padding: "2px 0",
          }}>{d}</div>
        ))}
      </div>

      {/* Day grid */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(7, 1fr)", gap: 2 }}>
        {cells.map((d, i) => {
          const ds         = localDateStr(d);
          const inMonth    = d.getMonth() === viewMonth.getMonth();
          const isSelected = ds === sel;
          const isToday    = ds === today;
          const isFuture   = max && ds > max;
          const isPast     = min && ds < min;
          const disabled   = !!isFuture || !!isPast;
          // White that lifts: a day holding footage reads as a raised tile. A
          // second dot would fight the today marker on the same cell.
          const hasFootage = !!recordedDays?.has(ds);
          const restBg     = hasFootage && !disabled ? "rgb(var(--ink) / 0.09)" : "transparent";
          return (
            <button key={i} type="button" disabled={disabled}
              onClick={() => { onChange(ds); onClose(); }}
              title={d.toLocaleDateString([], { weekday: "long", month: "long", day: "numeric", year: "numeric" })
                + (hasFootage ? " — has recordings" : "")}
              style={{
                aspectRatio: "1 / 1",
                display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
                fontSize: 12, fontWeight: 600,
                border: "1px solid transparent",
                borderRadius: 10,
                background: isSelected ? "var(--accent)" : restBg,
                color:
                  isSelected ? "var(--on-accent)"
                  : disabled  ? "var(--text-muted)"
                  : inMonth   ? "var(--text-primary)"
                              : "var(--text-muted)",
                cursor: disabled ? "not-allowed" : "pointer",
                position: "relative",
                transition: "background 140ms cubic-bezier(0.16,1,0.3,1), color 140ms",
                opacity: disabled ? 0.35 : 1,
              }}
              onMouseEnter={e => { if (!disabled && !isSelected) e.currentTarget.style.background = "var(--bg-hover)"; }}
              onMouseLeave={e => { if (!disabled && !isSelected) e.currentTarget.style.background = restBg; }}
            >
              {d.getDate()}
              {isToday && !isSelected && (
                <div style={{
                  position: "absolute", bottom: 3,
                  width: 4, height: 4, borderRadius: 999,
                  background: "var(--accent-fill)",
                }} />
              )}
            </button>
          );
        })}
      </div>

      {/* Quick today */}
      <button type="button" onClick={() => { onChange(today); onClose(); }}
        style={{
          marginTop: 4,
          padding: "6px 0",
          borderRadius: 10,
          border: "1px solid rgb(var(--ink) / 0.14)",
          background: "rgb(var(--ink) / 0.08)",
          color: "var(--text-secondary)",
          fontSize: 11, fontWeight: 700, cursor: "pointer",
        }}>
        Jump to today
      </button>
    </div>,
    document.body,
  );
}

const navBtn: React.CSSProperties = {
  display: "inline-flex", alignItems: "center", justifyContent: "center",
  width: 26, height: 26,
  borderRadius: 8,
  border: "1px solid rgb(var(--ink) / 0.14)",
  background: "rgb(var(--ink) / 0.08)",
  color: "var(--text-secondary)",
  cursor: "pointer",
};
