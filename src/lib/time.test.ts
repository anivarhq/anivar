import { describe, it, expect } from "vitest";
import {
  localDateStr, parseLocalDay, dayStartMs, dayEndMs,
  dayBoundsUtc, shiftDay, isLocalToday, isLocalYesterday,
} from "./time";

// These tests are TIMEZONE-INDEPENDENT: they construct dates with the LOCAL `Date`
// constructor and assert against LOCAL getters, so they hold in any TZ the CI/dev box
// runs in. That's the whole point — the SSOT's contract is "a day is a LOCAL calendar
// day", and the bug we're guarding against is treating a date-only string as UTC.

const DAY_MS = 86_400_000;

describe("localDateStr", () => {
  it("formats a local date as YYYY-MM-DD (zero-padded)", () => {
    // Month is 0-indexed in the Date constructor: 5 = June, 2 = March.
    expect(localDateStr(new Date(2026, 5, 9, 13, 30))).toBe("2026-06-09");
    expect(localDateStr(new Date(2026, 2, 1, 0, 0))).toBe("2026-03-01");
    expect(localDateStr(new Date(2026, 11, 31, 23, 59))).toBe("2026-12-31");
  });
});

describe("parseLocalDay", () => {
  it("returns LOCAL midnight of the day (NOT UTC midnight)", () => {
    const d = parseLocalDay("2026-06-29");
    expect(d.getFullYear()).toBe(2026);
    expect(d.getMonth()).toBe(5);   // June
    expect(d.getDate()).toBe(29);
    expect(d.getHours()).toBe(0);
    expect(d.getMinutes()).toBe(0);
    expect(d.getSeconds()).toBe(0);
  });

  it("accepts a Date and pins it to local midnight of that day", () => {
    const d = parseLocalDay(new Date(2026, 5, 29, 18, 45));
    expect(localDateStr(d)).toBe("2026-06-29");
    expect(d.getHours()).toBe(0);
  });
});

describe("dayStartMs / dayEndMs", () => {
  it("start is local midnight; end is the same local day at 23:59:59.999", () => {
    const start = new Date(dayStartMs("2026-06-29"));
    const end = new Date(dayEndMs("2026-06-29"));
    expect(localDateStr(start)).toBe("2026-06-29");
    expect(localDateStr(end)).toBe("2026-06-29");
    expect(start.getHours()).toBe(0);
    expect(end.getHours()).toBe(23);
    expect(end.getMinutes()).toBe(59);
  });

  it("spans ~24h (allowing for DST transition days)", () => {
    const span = dayEndMs("2026-06-29") - dayStartMs("2026-06-29");
    // 23h, 24h, or 25h day minus 1ms — accept the DST band.
    expect(span).toBeGreaterThan(23 * 3600_000 - 2);
    expect(span).toBeLessThan(25 * 3600_000);
  });
});

describe("dayBoundsUtc (the regression that bit us)", () => {
  it("fromUtc parses back to LOCAL midnight of the selected day — not the UTC date", () => {
    // The bug: `new Date('2026-06-29')` is UTC midnight, which is the WRONG instant
    // (off-by-one day) for any non-UTC timezone. dayBoundsUtc must encode LOCAL midnight.
    const { fromUtc, toUtc } = dayBoundsUtc("2026-06-29");
    const from = new Date(fromUtc);
    const to = new Date(toUtc);
    expect(localDateStr(from)).toBe("2026-06-29");
    expect(from.getHours()).toBe(0);
    expect(localDateStr(to)).toBe("2026-06-29");
    expect(fromUtc.endsWith("Z")).toBe(true); // UTC ISO for the DB query
  });

  it("a 'now'-stamped event always falls inside today's bounds (events don't vanish)", () => {
    const today = localDateStr();
    const { fromUtc, toUtc } = dayBoundsUtc(today);
    const nowIso = new Date().toISOString();
    expect(nowIso >= fromUtc).toBe(true);
    expect(nowIso <= toUtc).toBe(true);
  });

  it("bounds match the local-ms day window", () => {
    const { fromUtc, toUtc } = dayBoundsUtc("2026-06-29");
    expect(new Date(fromUtc).getTime()).toBe(dayStartMs("2026-06-29"));
    expect(new Date(toUtc).getTime()).toBe(dayEndMs("2026-06-29"));
  });
});

describe("shiftDay (DST-safe, noon-anchored)", () => {
  it("shifts forward and backward across month/year boundaries", () => {
    expect(shiftDay("2026-06-29", 1)).toBe("2026-06-30");
    expect(shiftDay("2026-06-30", 1)).toBe("2026-07-01");
    expect(shiftDay("2026-03-01", -1)).toBe("2026-02-28");
    expect(shiftDay("2026-01-01", -1)).toBe("2025-12-31");
    expect(shiftDay("2026-12-31", 1)).toBe("2027-01-01");
  });

  it("handles leap-year February", () => {
    expect(shiftDay("2028-02-28", 1)).toBe("2028-02-29"); // 2028 is a leap year
    expect(shiftDay("2028-03-01", -1)).toBe("2028-02-29");
  });

  it("is the inverse of itself", () => {
    expect(shiftDay(shiftDay("2026-06-15", 5), -5)).toBe("2026-06-15");
  });
});

describe("isLocalToday / isLocalYesterday", () => {
  it("recognises today and yesterday in local time", () => {
    const today = localDateStr();
    const yesterday = localDateStr(new Date(Date.now() - DAY_MS));
    expect(isLocalToday(today)).toBe(true);
    expect(isLocalToday(yesterday)).toBe(false);
    expect(isLocalYesterday(yesterday)).toBe(true);
    expect(isLocalYesterday(today)).toBe(false);
  });
});
