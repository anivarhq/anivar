// v14: Slide-up wrapper for the focus-mode Events / History panels. Mounts when
// `kind` becomes non-null. On unmount request (kind → null) we keep the panel
// in the DOM long enough to play the slide-out animation, then drop it.
//
// Tab swap (events ↔ history) is a quick crossfade — keyed by `kind` so React
// remounts the inner content while the wrapper stays put.

import { useEffect, useRef, useState } from "react";

import styles from "./BottomPanel.module.css";

/** Any truthy value mounts the drawer; null unmounts it (after exit animation). */
export type BottomPanelKind = string | null;

interface Props {
  kind: BottomPanelKind;
  children: React.ReactNode;
  /** v15: shrink the drawer to its content's intrinsic height instead of the
   *  default 46vh. Used by HistoryDrawer which is a fixed-height 220px design. */
  compact?: boolean;
}

const EXIT_MS = 280;

export function BottomPanel({ kind, children, compact = false }: Props) {
  // What's actually rendered. Lags `kind` by EXIT_MS on the way down so the
  // out animation plays before the node leaves the tree.
  const [renderKind, setRenderKind] = useState<BottomPanelKind>(kind);
  const [phase, setPhase] = useState<"enter" | "exit">("enter");
  const exitTimer = useRef<number | null>(null);

  useEffect(() => {
    if (kind) {
      // entering or switching tabs
      if (exitTimer.current) { clearTimeout(exitTimer.current); exitTimer.current = null; }
      setRenderKind(kind);
      setPhase("enter");
    } else if (renderKind) {
      // closing — play exit, then unmount
      setPhase("exit");
      exitTimer.current = window.setTimeout(() => {
        setRenderKind(null);
        exitTimer.current = null;
      }, EXIT_MS);
    }
    return () => {
      if (exitTimer.current) { clearTimeout(exitTimer.current); exitTimer.current = null; }
    };
  }, [kind, renderKind]);

  if (!renderKind) return null;

  return (
    <div className={`${styles.panel} ${compact ? styles.compact : ""} ${phase === "exit" ? styles.exit : styles.enter}`}>
      <div key={renderKind} className={styles.body}>
        {children}
      </div>
    </div>
  );
}
