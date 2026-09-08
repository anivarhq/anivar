import { describe, it, expect } from "vitest";
import { mediaTimeForDate, dateForMediaTime, type FragTime } from "./hlsAttach";

// A playlist as the server emits it: 10 s segments, PROGRAM-DATE-TIME per
// fragment, media time accumulating from 0.
const T0 = Date.parse("2026-09-04T04:00:00Z");
const frags: FragTime[] = [
  { start: 0,  duration: 10, programDateTime: T0 },
  { start: 10, duration: 10, programDateTime: T0 + 10_000 },
  // A recording gap: wall clock jumps 5 minutes, media time does not.
  { start: 20, duration: 10, programDateTime: T0 + 320_000 },
];

describe("mediaTimeForDate", () => {
  it("maps a wall-clock instant to media time inside a fragment", () => {
    expect(mediaTimeForDate(frags, T0)).toBe(0);
    expect(mediaTimeForDate(frags, T0 + 4_000)).toBe(4);
    expect(mediaTimeForDate(frags, T0 + 15_000)).toBe(15);
  });

  it("stays correct across a recording gap", () => {
    // Wall clock is 320s in, but only 20s of media precede it. Getting this
    // wrong is how a seek lands minutes away from the click.
    expect(mediaTimeForDate(frags, T0 + 323_000)).toBe(23);
  });

  it("returns null inside the gap rather than clamping to an edge", () => {
    // THE guard: silently snapping to the nearest fragment would seek the user
    // somewhere they did not click. null tells the caller to load a playlist.
    expect(mediaTimeForDate(frags, T0 + 100_000)).toBeNull();
  });

  it("returns null outside the loaded playlist", () => {
    expect(mediaTimeForDate(frags, T0 - 1)).toBeNull();
    expect(mediaTimeForDate(frags, T0 + 330_001)).toBeNull();
    expect(mediaTimeForDate([], T0)).toBeNull();
  });

  it("ignores fragments with no program-date-time", () => {
    expect(mediaTimeForDate([{ start: 0, duration: 10, programDateTime: null }], T0)).toBeNull();
  });
});

describe("dateForMediaTime", () => {
  it("is the exact inverse of mediaTimeForDate", () => {
    for (const t of [0, 4, 9.999, 10, 15, 20, 23, 29.5]) {
      const ms = dateForMediaTime(frags, t);
      expect(ms, `media ${t}`).not.toBeNull();
      expect(mediaTimeForDate(frags, ms!), `round-trip ${t}`).toBeCloseTo(t, 6);
    }
  });

  it("stays correct across a recording gap", () => {
    // THE case hls.js gets wrong. `hls.playingDate` extrapolates from whichever
    // fragment was last PLAYED — after a seek that is the one you left — so at
    // media 23 it reports T0+23s, missing the 5-minute outage entirely. Locating
    // the CONTAINING fragment gives the real wall clock.
    expect(dateForMediaTime(frags, 23)).toBe(T0 + 323_000);
    expect(dateForMediaTime(frags, 23)).not.toBe(T0 + 23_000);
  });

  it("returns null outside the loaded range rather than inventing a time", () => {
    expect(dateForMediaTime(frags, -1)).toBeNull();
    expect(dateForMediaTime(frags, 30)).toBeNull();
    expect(dateForMediaTime([], 0)).toBeNull();
    expect(dateForMediaTime([{ start: 0, duration: 10, programDateTime: null }], 1)).toBeNull();
  });

  it("never runs faster than real time", () => {
    // The runaway: the old fallback used `Date.now()` as a base, so the reported
    // clock advanced once for the base and once for currentTime — 2x speed. A
    // second of media must be a second of wall clock.
    const a = dateForMediaTime(frags, 5)!;
    const b = dateForMediaTime(frags, 6)!;
    expect(b - a).toBe(1000);
  });
});
