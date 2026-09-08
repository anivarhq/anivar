import { describe, it, expect } from "vitest";
import { eventBlocks, type TimelineItem } from "./NVRPanel";

const T0 = Date.parse("2026-08-29T00:00:00Z");
const min = (m: number) => T0 + m * 60_000;
const item = (id: string, startMin: number, secs: number,
              severity: "alert" | "detection" = "detection",
              reviewed = false): TimelineItem =>
  ({ id, start: min(startMin), end: min(startMin) + secs * 1000, severity, reviewed });

const PX = 1200;
const H6 = 6 * 60 * 60 * 1000;   // a 6-hour view — where markers used to vanish

describe("eventBlocks", () => {
  it("keeps a short event visible at a wide zoom", () => {
    // Drawn at true width this is 2px at 6h and 0.5px at 24h. The bucket is the
    // unit precisely so that cannot happen.
    const b = eventBlocks([item("a", 60, 35)], T0, T0 + H6, PX);
    expect(b).toHaveLength(1);
    const widthPx = ((b[0].end - b[0].start) / H6) * PX;
    expect(widthPx).toBeGreaterThanOrEqual(6);
  });

  it("merges neighbours into one run instead of a picket fence", () => {
    // Three events inside a couple of minutes, far below one bucket at 6h zoom.
    const b = eventBlocks(
      [item("a", 60, 30), item("b", 61, 30), item("c", 62, 30)], T0, T0 + H6, PX);
    expect(b).toHaveLength(1);
    expect(b[0].count).toBe(3);
  });

  it("separates events that are genuinely far apart", () => {
    const b = eventBlocks([item("a", 10, 30), item("b", 200, 30)], T0, T0 + H6, PX);
    expect(b).toHaveLength(2);
  });

  it("lets the highest severity in a run colour it", () => {
    const b = eventBlocks(
      [item("a", 60, 30, "detection"), item("b", 61, 30, "alert")], T0, T0 + H6, PX);
    expect(b).toHaveLength(1);
    expect(b[0].severity).toBe("alert");
  });

  it("dims a run only when every item in it is reviewed", () => {
    const allSeen = eventBlocks(
      [item("a", 60, 30, "detection", true), item("b", 61, 30, "detection", true)],
      T0, T0 + H6, PX);
    expect(allSeen[0].reviewed).toBe(true);

    // One unreviewed item must keep the whole run at full strength, or new
    // activity hides inside a dimmed block.
    const mixed = eventBlocks(
      [item("a", 60, 30, "detection", true), item("b", 61, 30, "detection", false)],
      T0, T0 + H6, PX);
    expect(mixed[0].reviewed).toBe(false);
  });

  it("ignores items outside the view and degenerate inputs", () => {
    expect(eventBlocks([item("a", 600, 30)], T0, T0 + H6, PX)).toHaveLength(0);
    expect(eventBlocks([item("a", 60, 30)], T0, T0, PX)).toHaveLength(0);
    expect(eventBlocks([item("a", 60, 30)], T0, T0 + H6, 0)).toHaveLength(0);
  });

  it("spans a long event across the buckets it covers", () => {
    // A 30-minute event at a 6h zoom is a real block, not a minimum-width stub.
    const b = eventBlocks([item("a", 60, 1800)], T0, T0 + H6, PX);
    expect(b).toHaveLength(1);
    expect(b[0].end - b[0].start).toBeGreaterThanOrEqual(1800 * 1000);
  });
});
