/**
 * Modal — a portalled dialog that behaves like one.
 *
 * Ten hand-rolled backdrops across People and Live shared the same three bugs:
 * Escape did nothing, focus was never restored to whatever opened the dialog,
 * and only some of them were portalled. The un-portalled ones happened to work
 * because no ancestor currently sets `transform` or `filter` — adding one
 * anywhere up the tree would have silently broken those and not the others.
 *
 * The z-index is a token, not a number picked per call site. The old ladder had
 * a three-way tie at 3000 and six separate 1000s (one of them in GlassCalendar),
 * so which dialog won was decided by DOM order.
 */

import { useEffect, useRef, type ReactNode } from "react";
import { createPortal } from "react-dom";

/** Stacking order, lowest first. Nested dialogs (an image zoom opened from
 *  inside a detail modal) take the next tier up. */
const Z = { base: 1000, nested: 1100, top: 1200 } as const;

/**
 * Escape closes, and focus goes back where it came from.
 *
 * For dialogs that already hand-roll their own backdrop and are not worth
 * rewriting onto `Modal`. Eight dialogs in People handled neither, so the only
 * way out of a mis-clicked one was to find its Cancel button with the mouse.
 */
export function useDismiss(onClose: () => void) {
  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") { e.stopPropagation(); onClose(); }
    };
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("keydown", onKey);
      opener?.focus?.();
    };
  }, [onClose]);
}

export function Modal({
  onClose,
  children,
  layer = "base",
  labelledBy,
  /** Clicking the backdrop closes. Off for dialogs mid-edit, where a stray
   *  click would silently discard what the user typed. */
  closeOnBackdrop = true,
}: {
  onClose: () => void;
  children: ReactNode;
  layer?: keyof typeof Z;
  labelledBy?: string;
  closeOnBackdrop?: boolean;
}) {
  const panelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Whatever had focus when this opened gets it back on close — otherwise
    // focus falls to <body> and keyboard users restart from the top of the page.
    const opener = document.activeElement as HTMLElement | null;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") { e.stopPropagation(); onClose(); }
    };
    document.addEventListener("keydown", onKey);
    // Focus the panel so Escape reaches us even if nothing inside is focusable.
    panelRef.current?.focus();
    return () => {
      document.removeEventListener("keydown", onKey);
      opener?.focus?.();
    };
  }, [onClose]);

  return createPortal(
    <div
      onMouseDown={e => { if (closeOnBackdrop && e.target === e.currentTarget) onClose(); }}
      style={{
        position: "fixed", inset: 0, zIndex: Z[layer],
        display: "flex", alignItems: "center", justifyContent: "center",
        padding: 20,
        background: "rgba(0,0,0,0.55)",
        backdropFilter: "blur(3px)",
      }}
    >
      <div
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={labelledBy}
        tabIndex={-1}
        style={{ outline: "none", maxHeight: "100%", display: "flex" }}
      >
        {children}
      </div>
    </div>,
    document.body,
  );
}

/**
 * Confirm — a dialog for destructive actions, replacing native `confirm()`.
 *
 * Five `confirm()` calls sat behind actions that erase biometric data. In a
 * frameless Tauri window the native dialog is both visually foreign and, worse,
 * inverted from the copy's intent: the person-removal flow chained two of them,
 * so "OK" on the second was the button that destroyed every descriptor.
 *
 * Here the destructive choice is always the one you have to reach for.
 */
export function Confirm({
  title, body, confirmLabel = "Delete", onConfirm, onCancel, layer = "top",
}: {
  title: string;
  body?: ReactNode;
  confirmLabel?: string;
  onConfirm: () => void;
  onCancel: () => void;
  layer?: "base" | "nested" | "top";
}) {
  return (
    <Modal onClose={onCancel} layer={layer} closeOnBackdrop={false}>
      <div className="glass" style={{
        padding: 20, borderRadius: 14, width: 380, maxWidth: "100%",
        display: "flex", flexDirection: "column", gap: 10,
      }}>
        <div style={{ fontWeight: 700, fontSize: 14 }}>{title}</div>
        {body && (
          <div style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.55 }}>
            {body}
          </div>
        )}
        <div style={{ display: "flex", gap: 8, justifyContent: "flex-end", marginTop: 6 }}>
          <button onClick={onCancel} autoFocus style={{
            padding: "6px 14px", borderRadius: 999, fontSize: 12, fontWeight: 700,
            border: "1px solid var(--border)", background: "var(--bg-elevated)",
            color: "var(--text-primary)", cursor: "pointer",
          }}>Cancel</button>
          <button onClick={onConfirm} style={{
            padding: "6px 14px", borderRadius: 999, fontSize: 12, fontWeight: 700,
            border: "1px solid color-mix(in srgb, var(--status-alert) 45%, transparent)",
            background: "transparent", color: "var(--accent-red)", cursor: "pointer",
          }}>{confirmLabel}</button>
        </div>
      </div>
    </Modal>
  );
}
