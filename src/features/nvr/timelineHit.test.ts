import { describe, it, expect } from "vitest";
import { eventAtStart } from "./NVRPanel";
import type { MotionEvent } from "../../types";

/** Minimal event: only started_at/ended_at matter to the hit test. */
const ev = (id: string, start: string, end: string): MotionEvent => ({
  id, started_at: start, ended_at: end,
  duration_secs: null, first_object_at: null, cam_id: 0, event_category: null,
  peak_score: 0.5, clip_path: null, thumbnail: null, detections: null, ai_summary: null,
  dominant_label: null, sub_label: null, recognized_plate: null, top_speed_kmh: null,
});

// The real shape that broke seeking: multi-minute events, back-to-back, so their
// spans blanket the whole visible timeline.
const events = [
  ev("a", "2026-08-29T00:24:32Z", "2026-08-29T00:29:32Z"),
  ev("b", "2026-08-29T00:29:33Z", "2026-08-29T00:34:33Z"),
];
const at = (iso: string) => new Date(iso).getTime();
const TOL = 3_000; // ~6px worth of time at a typical zoom

describe("eventAtStart", () => {
  it("selects an event when the click is on its start marker", () => {
    expect(eventAtStart(events, at("2026-08-29T00:24:33Z"), TOL)?.id).toBe("a");
    expect(eventAtStart(events, at("2026-08-29T00:29:33Z"), TOL)?.id).toBe("b");
  });

  it("returns null mid-span so the track can seek", () => {
    // THE regression guard: this instant is deep inside event "a". A whole-span
    // hit test returns "a" here, which is what made every click open an event
    // instead of seeking.
    expect(eventAtStart(events, at("2026-08-29T00:27:00Z"), TOL)).toBeNull();
  });

  it("picks the nearest start when two are within tolerance", () => {
    // 00:29:32.6 — 0.6s after a's end, 0.4s before b's start.
    expect(eventAtStart(events, at("2026-08-29T00:29:32Z") + 600, TOL)?.id).toBe("b");
  });

  it("returns null when nothing is in range", () => {
    expect(eventAtStart(events, at("2026-08-29T02:00:00Z"), TOL)).toBeNull();
    expect(eventAtStart([], at("2026-08-29T00:24:32Z"), TOL)).toBeNull();
  });
});
