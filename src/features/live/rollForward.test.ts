import { describe, it, expect } from "vitest";
import { chunkDay, findChunk, chunkPlayableEnd, shouldAdvance, CHUNK_MS } from "./usePlayback";
import { mergeSegmentBands, type Segment } from "../nvr/NVRPanel";
import { dayBoundsUtc } from "../../lib/time";

const DAY = Date.parse("2026-09-04T00:00:00Z");
const iso = (ms: number) => new Date(ms).toISOString();
/** `Array.prototype.at` is past this project's TS lib target. */
const last = <T,>(a: T[]): T => a[a.length - 1];
/** A finished day: midnight to midnight, entirely in the past. */
const past = { from: iso(DAY), to: iso(DAY + 86_400_000), now: DAY + 5 * 86_400_000 };
/** Today, 14:37. The day bound still says midnight; only `now` says otherwise. */
const TODAY_NOW = DAY + 14 * 3600_000 + 37 * 60_000;

describe("chunkDay", () => {
  it("covers a finished day in whole aligned hours, gapless", () => {
    const c = chunkDay(past.from, past.to, past.now);
    expect(c).toHaveLength(24);
    expect(c[0].start % CHUNK_MS).toBe(0);
    expect(c[0].start).toBe(DAY);
    expect(last(c).end).toBe(DAY + 86_400_000);
    for (let i = 1; i < c.length; i++) expect(c[i].start).toBe(c[i - 1].end);
  });

  it("NEVER produces a chunk that reaches into the future", () => {
    // THE structural guarantee. A chunk covering unrecorded time can only 404,
    // and a player that advances into one comes straight back for more — that
    // was the live-edge reload storm. Frigate gets this from
    // endOfHourOrCurrentTime; here it is the Math.min in chunkDay.
    const c = chunkDay(past.from, past.to, TODAY_NOW);
    expect(c.every(x => x.end <= TODAY_NOW)).toBe(true);
    expect(last(c).end).toBe(TODAY_NOW);
  });

  it("ends today with a ragged tail chunk for the hour we are living in", () => {
    const c = chunkDay(past.from, past.to, TODAY_NOW);
    expect(c).toHaveLength(15);                       // 14 whole hours + the tail
    expect(last(c).start).toBe(DAY + 14 * 3600_000);
    expect(last(c).end - last(c).start).toBeLessThan(CHUNK_MS);
  });

  it("gives the same chunk for a given hour however it is asked for", () => {
    // THE property the URL depends on: a chunk's identity is the wall clock it
    // covers, not where playback happened to be when it was requested.
    const a = chunkDay(past.from, past.to, past.now);
    const b = chunkDay(iso(DAY + 7 * 60_000), past.to, past.now);
    expect(b[0]).toEqual(a[findChunk(a, DAY + 7 * 60_000)]);
  });

  it("survives a degenerate range", () => {
    expect(chunkDay(past.to, past.from, past.now)).toEqual([]);
    expect(chunkDay("nonsense", past.to, past.now)).toEqual([]);
    expect(chunkDay(past.from, past.to, DAY - 1)).toEqual([]); // day not started
  });
});

describe("findChunk", () => {
  const c = chunkDay(past.from, past.to, past.now);

  it("locates the containing hour", () => {
    expect(findChunk(c, DAY)).toBe(0);
    expect(findChunk(c, DAY + CHUNK_MS)).toBe(1);          // boundary belongs forward
    expect(findChunk(c, DAY + 5 * CHUNK_MS + 60_000)).toBe(5);
    expect(findChunk(c, DAY + 86_400_000)).toBe(23);       // last chunk owns its end
  });

  it("returns -1 outside the day instead of clamping", () => {
    // Clamping would answer a click past the live edge by seeking somewhere the
    // user did not ask for. Frigate's updateSelectedSegment ignores -1.
    expect(findChunk(c, DAY - 1)).toBe(-1);
    expect(findChunk(c, DAY + 86_400_001)).toBe(-1);
    expect(findChunk([], DAY)).toBe(-1);
  });
});

describe("chunkPlayableEnd", () => {
  const c = { start: DAY, end: DAY + CHUNK_MS };
  it("is the chunk end when coverage runs through it", () => {
    expect(chunkPlayableEnd([{ start: DAY, end: DAY + 4 * CHUNK_MS }], c)).toBe(c.end);
  });
  it("is where footage stops when the camera switched off mid-chunk", () => {
    expect(chunkPlayableEnd([{ start: DAY, end: DAY + 400_000 }], c)).toBe(DAY + 400_000);
  });
  it("is the chunk start when the chunk holds nothing", () => {
    expect(chunkPlayableEnd([{ start: DAY + 5 * CHUNK_MS, end: DAY + 6 * CHUNK_MS }], c)).toBe(c.start);
  });
});

describe("shouldAdvance", () => {
  const c = { start: DAY, end: DAY + CHUNK_MS };
  const through = [{ start: DAY, end: DAY + 4 * CHUNK_MS }];

  it("advances when the chunk played through to its end", () => {
    expect(shouldAdvance(through, c, c.end - 100)).toBe(true);
  });

  it("advances when the camera switched off mid-chunk and we reached that point", () => {
    expect(shouldAdvance([{ start: DAY, end: DAY + 400_000 }], c, DAY + 399_000)).toBe(true);
  });

  it("REFUSES an `ended` that fired short of the footage — a hole is not an end", () => {
    // Frigate's onValidateClipEnd. Taking a premature `ended` at face value walks
    // the player forward over video it never showed.
    expect(shouldAdvance(through, c, DAY + 60_000)).toBe(false);
    expect(shouldAdvance(through, c, null)).toBe(false);
  });

  it("advances through an empty chunk instead of stalling on it", () => {
    const empty = { start: DAY + 5 * CHUNK_MS, end: DAY + 6 * CHUNK_MS };
    expect(shouldAdvance([{ start: DAY, end: DAY + CHUNK_MS }], empty, null)).toBe(true);
  });
});

describe("a whole day plays through, continuously", () => {
  it("walks every hour of a finished day without revisiting one", () => {
    // THE thing the player is for. The old rule looked for a coverage band
    // starting AFTER the playhead; on a day of continuous footage there is none,
    // so the player quit at the end of the first window.
    const chunks = chunkDay(past.from, past.to, past.now);
    const bands = [{ start: DAY, end: DAY + 86_400_000 }]; // one unbroken day
    const seen = new Set<number>();
    let idx = 0;
    for (let i = 0; i < 500; i++) {
      expect(seen.has(idx), "must never revisit a chunk").toBe(false);
      seen.add(idx);
      const chunk = chunks[idx];
      // Playback reaches the end of each chunk, so every `ended` is a real one.
      if (!shouldAdvance(bands, chunk, chunk.end)) break;
      if (idx >= chunks.length - 1) break;
      idx += 1;
    }
    expect(seen.size).toBe(24);
    expect(idx).toBe(chunks.length - 1);
  });

  it("stops at the live edge rather than asking for footage that does not exist", () => {
    const chunks = chunkDay(past.from, past.to, TODAY_NOW);
    const last = chunks.length - 1;
    expect(shouldAdvance([{ start: DAY, end: TODAY_NOW }], chunks[last], TODAY_NOW)).toBe(true);
    // ...but there is no chunk to advance INTO, which is what stops the storm.
    expect(chunks[last + 1]).toBeUndefined();
  });
});

describe("everything the timeline draws must be seekable", () => {
  // THE invariant the dead-timeline bug violated. `mergeSegmentBands` turns the
  // day's segments into the bands the timeline paints; `findChunk` decides where a
  // click on one of them lands. If a painted band resolves to -1, `seekTo` returns
  // and the click does nothing at all — no error, no movement. That is what a
  // whole day of footage looked like when the backend handed us the PREVIOUS
  // day's segments (a 1 h query pad that was never narrowed back).
  const seg = (startMs: number): Segment => ({
    filename: `cam0_${startMs}.mp4`, cam_id: 0,
    started_at: new Date(startMs).toISOString(),
    size_bytes: 4_500_000, duration_secs: 10,
  });
  /** 21 minutes of 10 s segments, like the real archive. */
  const runFrom = (startMs: number) =>
    Array.from({ length: 128 }, (_, i) => seg(startMs + i * 10_000));

  // Built from the day's own bounds, so this holds in any timezone the CI or the
  // developer happens to be in — the original bug was a UTC-offset interaction.
  const { fromUtc, toUtc } = dayBoundsUtc("2026-09-03");
  const dayFrom = Date.parse(fromUtc);
  const dayTo = Date.parse(toUtc);
  const chunks = chunkDay(fromUtc, toUtc, dayTo + 86_400_000);

  const placements: Array<[string, number]> = [
    ["first minutes of the day", dayFrom + 60_000],
    ["mid-morning", dayFrom + 9 * 3600_000],
    ["last hour (the ragged tail chunk)", dayTo - 40 * 60_000],
    ["straddling an hour boundary", dayFrom + 3600_000 - 600_000],
  ];

  for (const [where, at] of placements) {
    it(`resolves every band placed ${where}`, () => {
      const bands = mergeSegmentBands(runFrom(at));
      expect(bands.length).toBeGreaterThan(0);
      for (const b of bands) {
        for (const t of [b.start, Math.floor((b.start + b.end) / 2), b.end - 1]) {
          expect(findChunk(chunks, t), `${new Date(t).toISOString()} in no chunk`).not.toBe(-1);
        }
      }
    });
  }

  it("covers the whole day with no unreachable instant between chunks", () => {
    for (let t = dayFrom; t < dayTo; t += 7 * 60_000) {
      expect(findChunk(chunks, t), `${new Date(t).toISOString()} unreachable`).not.toBe(-1);
    }
    expect(findChunk(chunks, dayTo)).not.toBe(-1);
  });
});
