/**
 * Single source of truth for all date / time / "day" handling in the UI.
 *
 * Contract (keep the whole app on the same clock):
 *  - The BACKEND stores every timestamp in UTC (rfc3339, e.g. motion_events.started_at,
 *    nvr_segments.started_at). The UI ALWAYS displays + filters in the user's LOCAL
 *    timezone.
 *  - A "day" means a LOCAL calendar day. Any range query that selects "a day" must convert
 *    that local day's midnight→midnight boundaries to UTC ISO strings — use `dayBoundsUtc`.
 *  - NEVER write `new Date("YYYY-MM-DD")`: a date-ONLY string is parsed as UTC midnight,
 *    which is the wrong instant for any timezone behind/ahead of UTC (off-by-one day). Use
 *    `parseLocalDay` / the helpers below, which pin to LOCAL midnight via an explicit time.
 *
 * Every day-bounds / day-string computation in the app routes through this module so the
 * timezone rule can never drift between components again.
 */

const pad = (n: number) => String(n).padStart(2, "0");

/** A `Date` (defaults to now) → its LOCAL calendar date as "YYYY-MM-DD" (never UTC). */
export function localDateStr(d: Date = new Date()): string {
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

/** "YYYY-MM-DD" string (or a `Date`) → a `Date` pinned to LOCAL midnight of that day. */
export function parseLocalDay(day: string | Date): Date {
  if (day instanceof Date) { const d = new Date(day); d.setHours(0, 0, 0, 0); return d; }
  // Explicit "T00:00:00" (no trailing Z/offset) forces LOCAL-time parsing.
  return new Date(`${day}T00:00:00`);
}

/** Local-day START as epoch ms — for timeline axes that work in local-ms. */
export function dayStartMs(day: string | Date): number {
  return parseLocalDay(day).getTime();
}

/** Local-day END (23:59:59.999) as epoch ms. */
export function dayEndMs(day: string | Date): number {
  const d = parseLocalDay(day);
  d.setHours(23, 59, 59, 999);
  return d.getTime();
}

/**
 * Local-day bounds as UTC ISO strings — the ONLY correct way to ask the backend for
 * "events/segments on this local day" (the DB stores UTC).
 */
export function dayBoundsUtc(day: string | Date): { fromUtc: string; toUtc: string } {
  return {
    fromUtc: new Date(dayStartMs(day)).toISOString(),
    toUtc:   new Date(dayEndMs(day)).toISOString(),
  };
}

/** Shift a "YYYY-MM-DD" by N days, DST-safe (anchored at local noon so a 23/25-h day can't slip). */
export function shiftDay(day: string, deltaDays: number): string {
  const d = new Date(`${day}T12:00:00`);
  d.setDate(d.getDate() + deltaDays);
  return localDateStr(d);
}

/** Is this "YYYY-MM-DD" the local today? */
export function isLocalToday(day: string): boolean {
  return day === localDateStr();
}

/** Is this "YYYY-MM-DD" the local yesterday? */
export function isLocalYesterday(day: string): boolean {
  return day === localDateStr(new Date(Date.now() - 86_400_000));
}

/** Friendly local "Jul 14, 09:32" from a UTC timestamp. Accepts both RFC3339
 *  ("...T...Z/+00:00") and SQLite datetime('now') ("YYYY-MM-DD HH:MM:SS", UTC,
 *  no T/Z — normalized here so it isn't misread as local time). */
export function fmtWhen(iso: string): string {
  const d = new Date(iso.includes("T") ? iso : iso.replace(" ", "T") + "Z");
  if (isNaN(d.getTime())) return iso;
  return d.toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}
