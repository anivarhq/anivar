/**
 * The playback controller both recorded-footage surfaces drive.
 *
 * This is the one place that knows how to move through a day of recording. It
 * exists because the Review player and the Live focus player each grew their own
 * copy of seek / skip / ended / coverage / anchor logic — about fifty lines
 * apiece, subtly diverged — so every fix landed in one surface and not the other,
 * and every bug had to be found twice.
 *
 * The shape is Frigate's `RecordingView` / `DynamicVideoController`, and the one
 * decision worth stating is the one an earlier version got wrong:
 *
 *   * **The day is a FIXED LIST OF HOURLY CHUNKS and playback state is an INDEX.**
 *     Not a free-floating "anchorMs". An anchor mints a new URL from whatever
 *     instant playback happened to reach, so the same half hour is a different
 *     source every time you enter it: uncacheable, and re-entrant — which is
 *     exactly how `ended` turned into a reload every few seconds at the live edge.
 *     A chunk's URL is the same string forever, so re-entering one is free and
 *     advancing is `idx + 1` — monotonic over a finite list, so it cannot loop and
 *     cannot stop halfway through a day.
 *   * A seek INSIDE the current chunk is one `currentTime` assignment against
 *     media the player already holds: no request, no ffmpeg, no teardown. Changing
 *     chunk is the fallback, and the test is the chunk's own bounds — not whether
 *     hls.js happens to hold a fragment.
 *   * `ended` is VALIDATED before it is believed. An `ended` that fires short of
 *     the footage this chunk actually contains is a buffer hole or a decode
 *     failure, not an end, and advancing on it is how a player walks off a cliff.
 *   * Scrubbing points a low-resolution PREVIEW file at a timestamp and does not
 *     touch the recording at all. Only `seekTo` moves the real player.
 *   * Nothing here reads the player's reported position back into the timeline.
 *     The player publishes wall clock upward through `onWallClock`; this module
 *     only ever pushes down. Two-way binding between a video element and a
 *     timeline is what makes playheads run away.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { type StreamInfo } from "../../api";
import type { ClipOverlayHandle } from "./ClipOverlay";
import {
  mergeSegmentBands, rangeInCoverage, snapToCoverage, type Segment,
} from "../nvr/NVRPanel";

/** What the player is currently showing. History's POSITION is `chunkIdx`. */
export type ClipSource =
  | { kind: "none" }
  | { kind: "history" }
  | { kind: "event"; eventId: string };

/** One playback chunk: a fixed, hour-aligned slice of the day. */
export interface Chunk { start: number; end: number }

/**
 * Chunk width. Frigate's `getChunkedTimeDay` walks the day an hour at a time and
 * that is the number this matches. An hour is also exactly the server's maximum
 * VOD window (`window` is clamped to 30..3600 in `nvr_vod.rs`), so a chunk is the
 * largest playlist the backend will build — which is what keeps chunk changes,
 * and therefore source reloads, rare.
 */
export const CHUNK_MS = 60 * 60_000;

/**
 * The day as a fixed list of chunks: whole aligned hours, then a ragged tail.
 *
 * **The list never extends past the present.** That is not a detail — it is the
 * structural reason playback cannot storm at the live edge. A chunk covering time
 * that has not been recorded yet can only 404, and a player that advances into
 * one comes straight back for more. Frigate gets this from
 * `endOfHourOrCurrentTime` clamping every chunk end to now and from breaking the
 * loop at the current hour; an earlier version of this file instead let the list
 * run to midnight and bolted a "don't advance into the future" guard onto the
 * advance rule. Same intent, wrong place: the guard has to be re-remembered at
 * every call site, whereas a list that cannot contain a future chunk is simply
 * true everywhere.
 *
 * Aligned to absolute epoch hours, so a chunk's identity — and therefore its URL
 * — depends only on the wall clock it covers, never on when or from which surface
 * it was asked for. `nowMs` is a parameter so this stays pure and testable.
 */
export function chunkDay(fromUtc: string, toUtc: string, nowMs: number = Date.now()): Chunk[] {
  const from = new Date(fromUtc).getTime();
  const rawTo = new Date(toUtc).getTime();
  if (!isFinite(from) || !isFinite(rawTo)) return [];
  const to = Math.min(rawTo, nowMs);
  if (to <= from) return [];
  const out: Chunk[] = [];
  let start = Math.floor(from / CHUNK_MS) * CHUNK_MS;
  // 25, not 24: an aligned start before `from` can put a ragged hour at each end.
  for (let i = 0; i < 25; i++) {
    const next = start + CHUNK_MS;
    if (next > to) break;
    out.push({ start, end: next });
    start = next;
  }
  // The tail — the part-hour we are living in, or the ragged end of a past day.
  if (to > start) out.push({ start, end: to });
  return out;
}

/**
 * Index of the chunk containing `ms`, or -1.
 *
 * -1, never a clamp. Frigate's `findChunkIndex` reports "not in this day" and
 * `updateSelectedSegment` does nothing with it; clamping instead would answer a
 * click outside the recorded day by seeking somewhere the user did not ask for.
 * The last chunk owns its own end so the final instant of the day is reachable.
 */
export function findChunk(chunks: Chunk[], ms: number): number {
  return chunks.findIndex((c, i) =>
    c.start <= ms && (i === chunks.length - 1 ? c.end >= ms : c.end > ms));
}

/**
 * The last instant this chunk actually holds footage for.
 *
 * Frigate validates `ended` against `recordings.at(-1).start_time`; this is the
 * same idea against our coverage bands. Validating against the chunk's own END
 * would be wrong: a chunk whose camera switched off halfway through legitimately
 * ends early, and refusing to advance there would strand playback.
 */
export function chunkPlayableEnd(
  bands: Array<{ start: number; end: number }>, chunk: Chunk,
): number {
  let end = chunk.start;
  for (const b of bands) {
    if (b.start < chunk.end && b.end > chunk.start) end = Math.max(end, Math.min(b.end, chunk.end));
  }
  return end;
}

/**
 * Should an `ended` at `playheadMs` advance to the next chunk?
 *
 * This is Frigate's `onValidateClipEnd`: an `ended` that fires short of the
 * footage the chunk holds is buffering, a hole, or a decode failure — not an end
 * — and advancing on it walks the player forward over video it never showed.
 *
 * There is deliberately no "is this chunk in the future" test here. `chunkDay`
 * cannot produce one.
 */
export function shouldAdvance(
  bands: Array<{ start: number; end: number }>,
  chunk: Chunk,
  playheadMs: number | null,
): boolean {
  const playable = chunkPlayableEnd(bands, chunk);
  // Nothing was playable here at all — an empty chunk cannot end "early".
  if (playable <= chunk.start) return true;
  if (playheadMs == null) return false;
  // 2 s of slack: EXTINF is whole-second and the last fragment is rarely exact.
  return playheadMs >= playable - 2000;
}

interface Args {
  camId: number;
  /** The day's recorded segments — the source of truth for coverage. */
  segments: Segment[];
  streamInfo: StreamInfo | null;
  /** Day bounds (UTC ISO) used to chunk the day and load its scrub previews. */
  dayFromUtc: string;
  dayToUtc: string;
}

export function usePlayback({ camId, segments, streamInfo, dayFromUtc, dayToUtc }: Args) {
  const [clipSource, setClipSource] = useState<ClipSource>({ kind: "none" });
  const [playheadMs, setPlayheadMs] = useState<number | null>(null);
  const overlayRef = useRef<ClipOverlayHandle | null>(null);
  /** Mirror of `clipSource`, so callbacks can read it without a state updater. */
  const clipSourceRef = useRef<ClipSource>(clipSource);
  clipSourceRef.current = clipSource;

  // ── Coverage ────────────────────────────────────────────────────────────
  // One derivation, shared. Both surfaces used to compute this themselves.
  //
  // Coverage draws the timeline, snaps a CLICK into real footage, and validates
  // an `ended`. It is deliberately NOT what decides where playback goes next —
  // that is the chunk list, and conflating the two is what broke both surfaces.
  const coverageBands = useMemo(() => mergeSegmentBands(segments), [segments]);

  // ── Position ────────────────────────────────────────────────────────────
  const chunks = useMemo(() => chunkDay(dayFromUtc, dayToUtc), [dayFromUtc, dayToUtc]);
  const [chunkIdx, setChunkIdx] = useState(0);
  const chunk: Chunk | null = chunks[chunkIdx] ?? null;
  /** Where to land once the new source is ready — Frigate's `startTimestamp`.
   *
   *  A chunk's URL says which hour to load, never where in it to start, so the
   *  instant someone asked for is carried here and applied on `canplay`. STATE,
   *  not a ref: a ref cannot re-trigger the effect that consumes it, and a seek
   *  that silently evaporates because nothing re-rendered is the worst kind. */
  const [playbackStart, setPlaybackStart] = useState<number | null>(null);

  // A new day is a new chunk list; an index into the old one means nothing.
  useEffect(() => { setChunkIdx(0); setPlaybackStart(null); }, [chunks]);

  // No scrub previews. They existed to make DRAGGING cheap, and the timeline's
  // drag pans the viewport here rather than seeking (a deliberate rule), so
  // there is no drag for them to serve. Deleting the in-video scrubber — the
  // second playhead — took the last thing that drove them. `nvr_preview.rs`
  // still knows how to build the files if the timeline ever grows a handlebar.

  // ── Seeking ─────────────────────────────────────────────────────────────
  /**
   * Move the real player to a wall-clock instant.
   *
   * Snaps into recorded footage first (a click in dead air should land on video,
   * not on a black frame), then seeks inside the current chunk, and only changes
   * chunk when the target is in a different one. Frigate's rule exactly: the test
   * is the chunk's bounds, not the player's buffer.
   */
  const seekTo = useCallback((ms: number) => {
    const snapped = snapToCoverage(coverageBands, ms, "any") ?? ms;
    setPlayheadMs(snapped);
    const idx = findChunk(chunks, snapped);
    // Outside the recorded day entirely. Frigate's `updateSelectedSegment` does
    // nothing on -1 and neither do we: answering a click past the live edge by
    // seeking somewhere else is worse than not moving.
    if (idx === -1) {
      // ...but if `snapToCoverage` MOVED the target, coverage exists at `snapped`
      // and the timeline is drawing it — so a chunk should have contained it.
      // Reaching here means the segments we were handed lie outside the day we
      // chunked, which is exactly how a whole timeline went dead-on-click while
      // looking perfectly normal. Never let that be silent again.
      if (snapped !== ms) {
        console.warn("[playback] covered instant", new Date(snapped).toISOString(),
          "is in no chunk of", chunks.length, "— segments outside the chunked day?");
      }
      return;
    }
    // Read the current source from a ref, not from a `setClipSource` updater.
    // Seeking the video and calling setState are side effects, and a state updater
    // must be pure — React is free to run it twice, which would fire two seeks.
    const sameSource = clipSourceRef.current.kind === "history" && idx === chunkIdx;
    if (sameSource && overlayRef.current?.seekToWallClock(snapped)) {
      // A click on the timeline is a request to WATCH from there, not merely to
      // move the needle. Frigate seeks with `play = true` for the same reason.
      overlayRef.current.play();
      // Drop any instant still queued from an earlier click. Without this, a
      // second click that lands before the first chunk's `canplay` is undone by
      // it: the player jumps back to where the user no longer is.
      setPlaybackStart(null);
      return;
    }
    // Either a different chunk, or this one's manifest has not parsed yet. Carry
    // the instant; `onSourceReady` applies it the moment the media can play.
    setPlaybackStart(snapped);
    setChunkIdx(idx);
    // Only mint a new source object when the KIND actually changes — a fresh
    // object on every seek re-runs every effect keyed on `clipSource`.
    if (clipSourceRef.current.kind !== "history") setClipSource({ kind: "history" });
  }, [coverageBands, chunks, chunkIdx]);

  /** The player can play. Apply the instant the user actually asked for —
   *  Frigate's `onPlayerLoaded` -> `seekToTimestamp(startTimestamp, true)`. */
  const onSourceReady = useCallback(() => {
    if (playbackStart == null) return;
    // One attempt, then drop it: by `canplay` the VOD manifest is parsed, so a
    // failure here means the instant genuinely is not in this chunk's media, and
    // retrying forever would fight the user's next click.
    overlayRef.current?.seekToWallClock(playbackStart);
    overlayRef.current?.play();
    setPlaybackStart(null);
  }, [playbackStart]);

  /** ±N seconds from where the playhead actually is. */
  const skip = useCallback((deltaSec: number, eventBounds?: { startMs: number; endMs: number }) => {
    const cs = clipSourceRef.current;
    if (cs.kind === "event" && eventBounds) {
      const wantMs = (playheadMs ?? eventBounds.startMs) + deltaSec * 1000;
      // Inside the event, nudge the element; outside it, roll into the recording.
      if (wantMs < eventBounds.startMs || wantMs > eventBounds.endMs) seekTo(wantMs);
      else overlayRef.current?.seekRelative(deltaSec);
      return;
    }
    if (cs.kind !== "history" || !chunk) return;
    seekTo((playheadMs ?? chunk.start) + deltaSec * 1000);
  }, [playheadMs, chunk, seekTo]);

  /**
   * The loaded chunk ran out. Advance ONE chunk, or stay put.
   *
   * `idx + 1` over a finite list is the whole rule: it cannot revisit an anchor
   * (so no reload loop) and it cannot fail to find one (so playback does not quit
   * in the middle of a day the way "find the next coverage band" did — on
   * continuous footage there is no next band, and the player unmounted).
   *
   * `shouldAdvance` is the guard that makes it safe: an `ended` short of the
   * footage this chunk holds is a buffer hole, and one at the live edge is the
   * recorder being caught up with, not the day being over.
   */
  const handleEnded = useCallback(() => {
    if (clipSourceRef.current.kind !== "history" || !chunk) return;
    if (!shouldAdvance(coverageBands, chunk, playheadMs)) return;
    // The last chunk ends at the live edge (or at the end of a past day). Stopping
    // there is right: there is no next hour to roll into yet.
    if (chunkIdx >= chunks.length - 1) return;
    setPlaybackStart(null); // play the next chunk from its start
    setChunkIdx(chunkIdx + 1);
  }, [chunk, chunkIdx, chunks.length, coverageBands, playheadMs]);

  /**
   * Cross an empty chunk instead of stopping on it.
   *
   * A gap reached by ROLLING FORWARD is something to drive through — the old
   * coverage-snapping machinery existed to do exactly this. A gap reached by a
   * CLICK is where the user asked to be, and deserves the "camera was off"
   * overlay rather than being silently moved somewhere else. `playbackStart`
   * already tells the two apart: it is set only when someone asked for an
   * instant. Bounded and monotonic, so this cannot spin.
   */
  useEffect(() => {
    if (clipSource.kind !== "history" || !chunk) return;
    if (playbackStart != null) return;                   // a click; let it stand
    if (chunkIdx >= chunks.length - 1) return;           // end of the day
    if (rangeInCoverage(coverageBands, chunk.start, chunk.end)) return;
    setChunkIdx(chunkIdx + 1);
  }, [clipSource.kind, chunk, chunkIdx, chunks.length, coverageBands, playbackStart]);

  /** True when the loaded CHUNK holds no footage at all — "camera was off". */
  const isOffGap = clipSource.kind === "history" && !!chunk
    && !rangeInCoverage(coverageBands, chunk.start, chunk.end);

  const historyUrl = useMemo(() => {
    if (!streamInfo || clipSource.kind !== "history" || !chunk) return null;
    const p = new URLSearchParams({
      start: String(Math.floor(chunk.start / 1000)),
      window: String(Math.round((chunk.end - chunk.start) / 1000)),
      token: streamInfo.auth_token,
    });
    return `http://localhost:${streamInfo.port}/nvr-vod/${camId}/playlist.m3u8?${p}`;
  }, [streamInfo, clipSource.kind, chunk, camId]);

  return {
    clipSource, setClipSource,
    playheadMs, setPlayheadMs,
    coverageBands, isOffGap, historyUrl,
    overlayRef,
    seekTo, skip, handleEnded, onSourceReady,
  };
}
