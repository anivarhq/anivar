import { useEffect, useState } from "react";
import { api } from "../api";

/**
 * Local days (`YYYY-MM-DD`) that hold recorded footage, for marking the date picker.
 *
 * Fetches the whole archive rather than the visible month: the answer is one row per
 * DAY, so it is bounded by retention (a week here) no matter how many segments exist,
 * and a single call spares every month-flip a round trip.
 *
 * Absence is not an error — a failure just leaves the picker unmarked, exactly as it
 * was before. Never let a decoration break a date picker.
 */
export function useRecordedDays(camId?: number, refreshKey?: unknown): Set<string> {
  const [days, setDays] = useState<Set<string>>(() => new Set());
  useEffect(() => {
    let alive = true;
    api.listRecordedDays(camId)
      .then(d => { if (alive) setDays(new Set(d)); })
      .catch(() => { /* unmarked picker is the old behaviour, not a failure */ });
    return () => { alive = false; };
  }, [camId, refreshKey]);
  return days;
}
