/**
 * Multi-select toolbar dropdown — anchored glass button + checkbox popover.
 * Mirrors the KebabMenu/GlassCalendar pattern (click-outside + Escape).
 *
 * Lived inside ReviewFeed until People needed the same filter row; it is the
 * toolbar's only stateful control and nothing about it is Review-specific.
 */
import { useEffect, useRef, useState } from "react";
import { Check, ChevronDown } from "lucide-react";
import styles from "./ReviewFeed.module.css";

export function FilterDropdown({ label, options, selected, onToggle, onSelectAll, onClear, format, emptyText }: {
  label: string;
  options: readonly string[];
  selected: Set<string>;
  onToggle: (value: string) => void;
  onSelectAll: () => void;
  onClear: () => void;
  format?: (v: string) => string;
  emptyText?: string;
}) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
    window.addEventListener("mousedown", onDown);
    window.addEventListener("keydown", onKey);
    return () => { window.removeEventListener("mousedown", onDown); window.removeEventListener("keydown", onKey); };
  }, [open]);

  // Always render the BUTTON (even with no options) so the filter is
  // discoverable — the user sees it exists and understands new cameras/zones
  // will appear here. An empty menu shows a muted hint.
  const count = selected.size;
  return (
    <div ref={rootRef} className={styles.ddRoot}>
      <button className={`lg ${styles.filterBtn} ${count > 0 ? styles.filterBtnActive : ""}`}
        onClick={() => setOpen(o => !o)}>
        {label}{count > 0 ? ` · ${count}` : ""}
        <ChevronDown size={13} />
      </button>
      {open && (
        <div className={`glass ${styles.filterMenu}`} role="menu">
          {options.length === 0 ? (
            <div className={styles.filterEmpty}>{emptyText ?? "None yet"}</div>
          ) : (
            <>
              <div className={styles.filterHead}>
                <button className={styles.filterHeadBtn}
                  onClick={onSelectAll}
                  disabled={count === options.length}>Select all</button>
                <button className={styles.filterHeadBtn}
                  onClick={onClear}
                  disabled={count === 0}>Clear</button>
              </div>
              {options.map(opt => {
                const on = selected.has(opt);
                return (
                  <button key={opt} className={styles.filterOpt} onClick={() => onToggle(opt)}>
                    <span className={`${styles.checkbox} ${on ? styles.checkboxOn : ""}`}>
                      {on && <Check size={11} />}
                    </span>
                    <span className={styles.filterOptLabel}>{format ? format(opt) : opt}</span>
                  </button>
                );
              })}
            </>
          )}
        </div>
      )}
    </div>
  );
}
