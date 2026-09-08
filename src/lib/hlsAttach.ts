import Hls from "hls.js";
import { useEffect, useRef } from "react";

/**
 * One source-attach for every recorded-footage player.
 *
 * Plain URLs (cached event clips, exports) set `video.src` directly. `.m3u8`
 * URLs attach hls.js (WebView2 has no native HLS) — recorded windows are now
 * served as HLS VOD (`/nvr-vod`, mature NVRs' model) because gluing segments into
 * one stream left AAC seam holes that stuttered playback. hls.js stitches the
 * per-segment media itself and its gap-nudging glides over seams.
 */
export interface MediaHandle {
  isHls: boolean;
  /** Wall-clock ms of the CURRENT playhead (from HLS program-date-time), or
   *  null for plain sources / before the first fragment loads. Exact even
   *  across recording-gap discontinuities. */
  playingDateMs: () => number | null;
  /** Wall clock at an ARBITRARY media time, for previewing a drag target before
   *  committing to it. Same map as `playingDateMs`, different input. */
  dateAtMs: (mediaSecs: number) => number | null;
  /** Seek to a wall-clock instant WITHIN the already-loaded playlist.
   *
   *  Returns false when the instant is outside it, so the caller can fall back to
   *  requesting a new playlist. See `mediaTimeForDate`. */
  seekToDate: (ms: number) => boolean;
  destroy: () => void;
}

/** One loaded fragment, reduced to what the wall-clock mapping needs. */
export interface FragTime {
  /** Media time (seconds into the playlist) where this fragment starts. */
  start: number;
  duration: number;
  /** Wall clock of that start, from EXT-X-PROGRAM-DATE-TIME. */
  programDateTime: number | null;
}

/**
 * Wall clock -> media time, using the playlist's own PROGRAM-DATE-TIME anchors.
 *
 * hls.js exposes `playingDate` as a getter only, so there is no built-in "seek to
 * a date". But every fragment carries `start` (media time), `duration` and
 * `programDateTime`, which is all the mapping needs — and reading it off the
 * ACTUAL loaded playlist makes it exact by construction, rather than recomputing
 * the server's EXTINF sums on the client and hoping they agree.
 *
 * Returns null when `ms` falls outside the loaded fragments; the caller must then
 * load a playlist that covers it. Never clamp instead — silently seeking to the
 * nearest edge is how a scrub ends up somewhere the user did not click.
 */
export function mediaTimeForDate(frags: FragTime[], ms: number): number | null {
  for (const f of frags) {
    const pdt = f.programDateTime;
    if (pdt == null) continue;
    if (ms >= pdt && ms < pdt + f.duration * 1000) {
      return f.start + (ms - pdt) / 1000;
    }
  }
  return null;
}

/**
 * Media time -> wall clock. The exact inverse of `mediaTimeForDate`.
 *
 * This replaces `hls.playingDate`, which is wrong in the two situations that
 * matter most here. It computes `frag.programDateTime + (currentTime -
 * frag.start)` and PREFERS `this.currentFrag` (hls.js `streamController`), which
 * right after a seek is still the fragment you *left*. Extrapolating that far
 * outside a fragment is only correct while media time and wall clock run in
 * lockstep — so it overshoots by the whole outage across an EXT-X-DISCONTINUITY,
 * and reports the old position during the window between a seek and the new
 * fragment being appended. The needle jumped, then snapped back.
 *
 * Finding the CONTAINING fragment instead makes it right by construction, and
 * makes the forward and reverse maps agree because they read the same list.
 *
 * Null outside the loaded range: better no timestamp than an invented one.
 */
export function dateForMediaTime(frags: FragTime[], t: number): number | null {
  for (const f of frags) {
    const pdt = f.programDateTime;
    if (pdt == null) continue;
    if (t >= f.start && t < f.start + f.duration) {
      return pdt + (t - f.start) * 1000;
    }
  }
  return null;
}

export function attachSource(
  video: HTMLVideoElement,
  url: string,
  opts?: { onFatal?: (details: string) => void },
): MediaHandle {
  const isHls = url.includes(".m3u8");
  if (!isHls || !Hls.isSupported()) {
    video.src = url;
    return {
      isHls: false, playingDateMs: () => null, dateAtMs: () => null,
      seekToDate: () => false, // a plain file has no wall-clock anchors
      destroy: () => { /* src players need no teardown */ },
    };
  }
  // VOD tuning (the live view keeps its own config in HlsFeed).
  //
  // What was here before was mostly no-ops that shrank buffers: `startPosition:
  // -1` and `maxBufferLength: 30` are already the hls.js 1.6 defaults, while
  // `maxMaxBufferLength: 60` (default 600) and `backBufferLength: 30` (default
  // Infinity) only REDUCED what is retained — so a small backward seek threw away
  // media it still had and re-fetched it, which on this server means re-spawning
  // one ffmpeg per segment.
  const hls = new Hls({
    maxBufferLength: 30,
    // Keep a real back buffer so backward scrubs are instant. Not Infinity: a long
    // session scrubbing across a day would grow without bound.
    backBufferLength: 90,
    // Step over the AAC-priming holes at segment seams (nvr_vod.rs documents "up
    // to 72 ms" at nearly every boundary) without stalling — a stall that exhausts
    // the nudge budget becomes a fatal, and the recovery ladder below calls
    // recoverMediaError(), a full MSE teardown that re-downloads everything.
    //
    // 0.2, not 0.5. hls.js escapes a hole by HARD-SEEKING the element forward
    // (GapController._trySkipBufferHole), so this value is also the size of jump
    // it will make unasked. 0.5 was chosen while the server was over-declaring
    // EXTINF and manufacturing multi-second holes; with that fixed, the only real
    // holes are the ~72 ms seams, and a tighter bound keeps an unexplained
    // forward jump small if one ever appears again.
    maxBufferHole: 0.2,
    // PROGRAM-DATE-TIME is whole-second (segment filenames are -strftime), so the
    // 0.25 default is tighter than the playlist's own precision.
    maxFragLookUpTolerance: 0.5,
    // Fetch the first fragment while MSE is still attaching — free latency on
    // every source swap.
    startFragPrefetch: true,
    lowLatencyMode: false, // VOD playlists have no EXT-X-PART
  });
  let destroyed = false;
  // Self-healing (hls.js-documented ladder, same as mature NVRs' player): a fatal
  // error first gets in-place recovery — `startLoad()` for network fatals,
  // `recoverMediaError()` (then `swapAudioCodec()+recoverMediaError()`) for
  // media fatals — and only repeated failure destroys + reports. Without this,
  // ONE transient segment fetch/decode hiccup silently killed playback for good.
  let netRetries = 0;
  let mediaRetries = 0;
  const giveUp = (details: string) => {
    if (destroyed) return;
    destroyed = true;
    hls.destroy();
    opts?.onFatal?.(details);
  };
  hls.on(Hls.Events.ERROR, (_evt, data) => {
    if (!data.fatal || destroyed) return;
    const details = data.details ?? "hls fatal error";
    // Log every recovery. These used to be swallowed entirely, which made a slow
    // ffmpeg remux and an hls.js self-restart look identical from the outside —
    // both just "playback hiccuped".
    console.warn("[hls] fatal", data.type, details, "— recovering");
    if (data.type === Hls.ErrorTypes.NETWORK_ERROR) {
      if (netRetries < 2) {
        netRetries += 1;
        // Backoff 250ms then 1s: rides out a segment still finalizing, a
        // server hiccup, or a token race without dropping the session.
        window.setTimeout(() => { if (!destroyed) hls.startLoad(); }, netRetries === 1 ? 250 : 1000);
      } else giveUp(details);
    } else if (data.type === Hls.ErrorTypes.MEDIA_ERROR) {
      if (mediaRetries === 0) {
        mediaRetries = 1;
        hls.recoverMediaError();
      } else if (mediaRetries === 1) {
        mediaRetries = 2;
        // Documented second step: codec swap + recover.
        hls.swapAudioCodec();
        hls.recoverMediaError();
      } else giveUp(details);
    } else giveUp(details);
  });
  // Forward progress = healthy again; a later, unrelated stall gets a fresh
  // recovery budget instead of inheriting exhausted counters.
  hls.on(Hls.Events.FRAG_BUFFERED, () => { netRetries = 0; mediaRetries = 0; });
  hls.loadSource(url);
  hls.attachMedia(video);
  /** The loaded playlist's fragments.
   *
   *  NOT `hls.levels[hls.currentLevel]` — `currentLevel` delegates to the stream
   *  controller and is -1 until a fragment has actually been appended, so for the
   *  first second after every source swap that read came back empty and
   *  `seekToDate` refused a seek it could have served. */
  const loadedFrags = (): FragTime[] =>
    hls.levels[hls.loadLevel]?.details?.fragments
    ?? hls.levels[0]?.details?.fragments
    ?? [];

  return {
    isHls: true,
    playingDateMs: () => dateForMediaTime(loadedFrags(), video.currentTime),
    dateAtMs: (mediaSecs: number) => dateForMediaTime(loadedFrags(), mediaSecs),
    seekToDate: (ms: number) => {
      if (destroyed) return false;
      const t = mediaTimeForDate(loadedFrags(), ms);
      if (t === null) return false;
      video.currentTime = t;
      return true;
    },
    destroy: () => {
      if (!destroyed) { destroyed = true; hls.destroy(); }
    },
  };
}

/**
 * Declarative wrapper for players whose `<video>` previously used a `src`
 * prop (NVRPanel). Re-attaches whenever `url` changes; cleans up on unmount.
 * Returns a ref holding the live MediaHandle (for `playingDateMs`).
 */
export function useHlsVideo(
  videoRef: React.RefObject<HTMLVideoElement | null>,
  url: string | null,
  onFatal?: (details: string) => void,
) {
  const handleRef = useRef<MediaHandle | null>(null);
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !url) return;
    const handle = attachSource(video, url, { onFatal });
    handleRef.current = handle;
    return () => { handle.destroy(); handleRef.current = null; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url]);
  return handleRef;
}
