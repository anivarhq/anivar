// Miniature stale-while-revalidate for panel data.
//
// App.tsx mounts ONLY the active tab (deliberate: unmounting stops each panel's
// polling and frees its DOM). The cost was that every revisit refetched into a
// BLANK surface. This module keeps the last-known data at module scope so a
// remounting panel renders it instantly, then refreshes in the background —
// the standard SWR pattern, minus the library.
//
// Scope: UI display data only (event lists, rosters, chat history). Nothing
// here is a source of truth — the backend is; the cache just kills the flash.

import { useCallback, useState } from "react";

const cache = new Map<string, unknown>();

/** Drop-in useState replacement whose value survives unmount/remount. */
export function usePanelCache<T>(key: string, initial: T): [T, (v: T | ((prev: T) => T)) => void] {
  const [value, setValue] = useState<T>(() => (cache.has(key) ? (cache.get(key) as T) : initial));
  const set = useCallback((v: T | ((prev: T) => T)) => {
    setValue(prev => {
      const next = typeof v === "function" ? (v as (prev: T) => T)(prev) : v;
      cache.set(key, next);
      return next;
    });
  }, [key]);
  return [value, set];
}
