// v23: Generic <video> overlay stacked over CameraView with a standard
// auto-hiding glass transport bar. Three things this version adds beyond
// v16/17:
//
//  • Frozen-frame canvas painted underneath the <video> just before any src
//    reload. The previous frame stays visible until the new stream emits
//    `canplay`, then fades out. No more solid-black flash during seek.
//  • 500ms-deferred loading spinner — appears only if the new stream takes
//    long enough that the held frame becomes unconvincing.
//  • Imperative handle (forwardRef + useImperativeHandle) exposing
//    seekRelative / togglePlay / toggleMute / toggleFullscreen so the
//    parent can drive the overlay from keyboard shortcuts or skip-overflow
//    handlers without re-implementing the underlying behaviour.
//
// CameraView stays mounted underneath so closing the overlay returns to the
// live feed instantly — no RTSP / MJPEG re-handshake.

import {
  forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState,
} from "react";
import {
  Play, Pause, Rewind, FastForward, SkipBack, SkipForward, Volume2, VolumeX, Maximize2, Loader2,
} from "lucide-react";

import { MAX_RETRIES, retryDelayMs, bustUrl } from "./clipRetry";
import { loadSavedVolume, saveVolume, loadSavedMuted, saveMuted } from "../../lib/volume";
import { attachSource, type MediaHandle } from "../../lib/hlsAttach";
import styles from "./ClipOverlay.module.css";

export interface ClipOverlayHandle {
  /** Bump video.currentTime by deltaSec, clamped to [0, duration]. */
  seekRelative: (deltaSec: number) => void;
  /** Seek to a wall-clock instant WITHIN the loaded HLS playlist.
   *
   *  Returns false when the instant is outside it, so the caller can fall back to
   *  loading a playlist that covers it. The playlist is 30 minutes wide, so the
   *  overwhelming majority of timeline clicks land inside and cost nothing. */
  seekToWallClock: (ms: number) => boolean;
  /** Read current playhead in seconds (video.currentTime). */
  getPosition: () => number;
  /** Read clip duration in seconds (NaN until loadedmetadata). */
  getDuration: () => number;
  togglePlay: () => void;
  /** Start playing. A timeline click is a request to WATCH from there, so the
   *  controller resumes after a seek — Frigate seeks with `play = true`. */
  play: () => void;
  toggleMute: () => void;
  toggleFullscreen: () => void;
}

interface Props {
  /** Full URL to load into <video>. Changing this reloads + seeks. */
  src: string;
  /** Wall-clock anchor (ms). When present, onWallClock fires with the
   *  absolute timestamp; caller drives the HorizontalTimeline needle from it. */
  anchorMs?: number;
  onWallClock?: (ms: number) => void;
  onEnded?: () => void;
  /** The media can play. A chunk's URL says which half hour to load, never where
   *  in it to start, so the controller applies its pending seek here. */
  onSourceReady?: () => void;
  /** When provided, the ±10s buttons call this with the delta in seconds
   *  and the caller decides how to seek (e.g. for history mode the caller
   *  reloads `src` with a new `start=` param). When absent, the overlay
   *  falls back to the imperative `seekRelative` (client-side currentTime). */
  onSkip?: (deltaSec: number) => void;
  /** v25: jump to the previous event in the camera's timeline. When undefined
   *  the button is hidden (e.g. live mode, first event reached). */
  onPrevEvent?: () => void;
  /** Symmetric to onPrevEvent. */
  onNextEvent?: () => void;

}

const SPINNER_DELAY_MS = 500;

export const ClipOverlay = forwardRef<ClipOverlayHandle, Props>(function ClipOverlay(
  { src, anchorMs, onWallClock, onEnded, onSourceReady, onSkip, onPrevEvent, onNextEvent },
  fwdRef,
) {
  const videoRef         = useRef<HTMLVideoElement>(null);
  const rootRef          = useRef<HTMLDivElement>(null);
  const frozenCanvasRef  = useRef<HTMLCanvasElement>(null);
  const anchorRef        = useRef<number | undefined>(anchorMs);
  const hideTimerRef     = useRef<number | null>(null);
  const spinnerTimerRef  = useRef<number | null>(null);
  const lastSrcRef       = useRef<string>(src);
  // Auto-retry state for transient clip-load failures (see clipRetry.ts).
  const retryTimerRef    = useRef<number | null>(null);
  const attemptRef       = useRef<number>(0);
  const playedRef        = useRef<boolean>(false);
  // v24: remembered volume from just before mute. Restored when the user
  // unmutes via the speaker, so a "mute → unmute" round-trip lands at the
  // same level they were listening at. Seeded from the persisted level so
  // the very first unmute after a restart lands where the user last set it.
  const preMuteVolumeRef = useRef<number>(loadSavedVolume());
  // HLS-or-plain source handle. Recorded windows are HLS VOD (.m3u8); cached
  // event clips stay plain files. All source swaps MUST go through attachTo — a
  // bare `v.src=` on an m3u8 can't play.
  //
  // There used to be a SECOND handle here, feeding a blurred full-resolution copy
  // of the same stream into the letterbox. It doubled every playlist fetch, every
  // segment fetch, every ffmpeg spawn on the server and every H.264 decode, then
  // re-blurred a 116%-scaled full-res surface on the GPU every frame — for
  // ambient glow. Removed; the letterbox is flat --bg-base.
  const mediaRef     = useRef<MediaHandle | null>(null);
  // handleVideoError is declared further down; hls.js fatal errors route into
  // it through this ref so attachTo stays stable and dependency-free.
  const hlsFatalRef  = useRef<() => void>(() => {});
  const attachTo = useCallback((v: HTMLVideoElement, url: string) => {
    mediaRef.current?.destroy();
    mediaRef.current = attachSource(v, url, { onFatal: () => hlsFatalRef.current() });
    if (!mediaRef.current.isHls) v.load();
  }, []);
  // Unmount: tear down the live hls.js instance.
  useEffect(() => () => { mediaRef.current?.destroy(); }, []);

  const [playing,   setPlaying]   = useState(true);
  // Muted + volume are BOTH persisted: set the sound once in any player and
  // every player everywhere (events, vehicles, sounds, live focus) opens the
  // same way. First-ever run defaults to muted (no autoplay blast).
  const [muted,     setMuted]     = useState<boolean>(loadSavedMuted);
  const [volume,    setVolume]    = useState<number>(loadSavedVolume); // 0..1, persisted
  // Playback speed — cycles via the ×-button; applied on every (re)load since
  // a src swap resets the element's rate.
  const [rate,      setRate]      = useState(1);

  // Any deliberate level change is remembered for every future player.
  useEffect(() => { saveVolume(volume); }, [volume]);
  useEffect(() => { saveMuted(muted); }, [muted]);
  const [showCtrls, setShowCtrls] = useState(true);
  /** True while a held-frame canvas is painted over the video (during src
   *  reload). Faded out on `canplay`. */
  const [showFrozen, setShowFrozen] = useState(false);
  /** True after SPINNER_DELAY_MS of no canplay. */
  const [showSpinner, setShowSpinner] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);

  // NO in-video scrubber, and no `pos`/`dur`/`buffered` sampling to feed one.
  //
  // The timeline below the player is the ONLY scrub surface — one bar, the way
  // Frigate does it: its `VideoControls` renders buttons and a volume slider and
  // nothing else, because the review timeline already owns position. We had both,
  // so the player showed two playheads that could disagree, and the sampler that
  // fed the in-video one re-rendered this whole component eight times a second
  // for pixels the timeline was already drawing.
  //
  // The low-res scrub PREVIEW went with it: it existed to make DRAGGING cheap,
  // and dragging the timeline pans the viewport here by design rather than
  // seeking, so there is no drag left for it to serve. `nvr_preview.rs` still
  // builds the files; wire them back up if the timeline ever grows a handlebar.

  // ── (re)load when src changes, with a frozen-frame held underneath ──
  useEffect(() => {
    const v = videoRef.current;
    const c = frozenCanvasRef.current;
    if (!v) return;
    const srcChanged = lastSrcRef.current !== src;

    // Paint the last decoded frame to the canvas BEFORE swapping src.
    // v24: also require v.error === null so we don't paint a stale/junk
    // frame from a video that just 404'd onto the new load.
    if (srcChanged && c && v.videoWidth > 0 && v.readyState >= 2 && v.error === null) {
      const ctx = c.getContext("2d");
      if (ctx) {
        c.width = v.videoWidth;
        c.height = v.videoHeight;
        try { ctx.drawImage(v, 0, 0); } catch { /* tainted or not ready */ }
        setShowFrozen(true);
      }
    }
    attachTo(v, src);
    lastSrcRef.current = src;
    setLoadError(null);
    setShowSpinner(false);
    // Fresh source → reset the retry counter and cancel any pending retry from
    // the previous src so a new clip/seek starts with a clean slate.
    attemptRef.current = 0;
    playedRef.current = false;
    if (retryTimerRef.current) { window.clearTimeout(retryTimerRef.current); retryTimerRef.current = null; }

    // Show a subtle spinner if loading takes longer than 500ms.
    if (spinnerTimerRef.current) window.clearTimeout(spinnerTimerRef.current);
    spinnerTimerRef.current = window.setTimeout(
      () => setShowSpinner(true),
      SPINNER_DELAY_MS,
    );

    return () => {
      if (spinnerTimerRef.current) window.clearTimeout(spinnerTimerRef.current);
      if (retryTimerRef.current)   window.clearTimeout(retryTimerRef.current);
    };
    // `anchorMs` is deliberately NOT a dependency — see the effect below it.
  }, [src, attachTo]);

  // The anchor is read only by onTimeUpdate's wall-clock fallback; it has no
  // business tearing down the media source. It was in the reload effect's deps,
  // and parents pass `trueAnchorMs ?? clipAnchorMs` where `trueAnchorMs` mutates
  // twice per seek (null, then the server's answer) — so ONE timeline click
  // destroyed and rebuilt the hls.js instance up to three times on an unchanged
  // URL, aborting each in-flight fragment. That is what made every seek stutter.
  useEffect(() => { anchorRef.current = anchorMs; }, [anchorMs]);

  // ── apply play/pause ─────────────────────────────────────────────────
  // v24: when v.play() rejects (autoplay blocked, src not yet loaded, etc.)
  // we resync the state to playing=false so the toolbar shows the PLAY icon.
  // The user's next click of play is a real user gesture which browsers
  // always allow, so the second attempt succeeds.
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    if (playing) { v.play().catch(() => setPlaying(false)); } else { v.pause(); }
  }, [playing]);

  // ── apply mute + volume ──────────────────────────────────────────────
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    v.muted = muted;
    v.volume = volume;
  }, [muted, volume]);

  // ── apply playback speed (also re-applied on loadedmetadata: a src swap
  //    resets the element's rate to 1) ───────────────────────────────────
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    v.playbackRate = rate;
    v.defaultPlaybackRate = rate;
  }, [rate, src]);

  // ── auto-hide controls after 2.2s mouse idle ─────────────────────────
  const armHide = useCallback(() => {
    if (hideTimerRef.current) window.clearTimeout(hideTimerRef.current);
    hideTimerRef.current = window.setTimeout(() => setShowCtrls(false), 2200);
  }, []);
  const bumpVisibility = useCallback(() => {
    setShowCtrls(true);
    armHide();
  }, [armHide]);
  // Pin the controls open while the pointer is OVER the transport bar. The
  // 2.2s idle timer only re-arms on mousemove, so steadily hovering or clicking
  // a button (no movement) used to let the bar vanish under the cursor. Cancel
  // the hide timer on enter, re-arm it on leave.
  const holdVisible = useCallback(() => {
    if (hideTimerRef.current) window.clearTimeout(hideTimerRef.current);
    setShowCtrls(true);
  }, []);
  useEffect(() => {
    armHide();
    return () => {
      if (hideTimerRef.current) window.clearTimeout(hideTimerRef.current);
    };
  }, [armHide]);

  // ── transport actions ────────────────────────────────────────────────
  const togglePlay = useCallback(() => setPlaying(p => !p), []);
  // v24: pre-mute round-trip. Muting captures the current non-zero volume so
  // the next unmute restores it. If the user dragged the slider down to 0
  // before muting, unmuting bumps volume to the remembered level — never
  // silent-but-unmuted.
  const toggleMute = useCallback(() => {
    setMuted(m => {
      if (m) {
        // unmuting — restore prior volume if the slider was dragged to 0
        if (volume === 0) setVolume(preMuteVolumeRef.current || 0.8);
        return false;
      }
      // muting — remember the volume we're about to silence
      preMuteVolumeRef.current = volume || 0.8;
      return true;
    });
  }, [volume]);
  // v25: buffer-aware client-side seek. fMP4 streams from /footage/:id/clip
  // and /nvr-concat are produced on the fly — at any moment only a leading
  // portion of the clip is buffered. Setting `currentTime` past the buffered
  // end silently aborts (no event fires, the video keeps playing where it
  // was). That's why "+10s sometimes does nothing until you go back first."
  //
  // Fix: if the target is past `buffered.end(N-1)`, wait on the `progress`
  // event for the buffer to grow, then assign currentTime. Retry up to a
  // short timeout; if the buffer never catches up, clamp to the buffered
  // end so the user sees the player jump as far as it can.
  const seekRelative = useCallback((deltaSec: number) => {
    const v = videoRef.current;
    if (!v) return;
    const max = isFinite(v.duration) ? v.duration : Infinity;
    const target = Math.max(0, Math.min(max, v.currentTime + deltaSec));

    // HLS VOD: hls.js fetches whatever segments a seek needs — plain
    // currentTime assignment is fully correct, and the buffer-wait dance
    // below would deadlock a paused forward-seek (no progress events fire).
    if (mediaRef.current?.isHls) {
      v.currentTime = target;
      return;
    }

    const bufferedEnd = () => {
      const b = v.buffered;
      return b.length > 0 ? b.end(b.length - 1) : 0;
    };

    if (target <= bufferedEnd() + 0.25) {
      // already buffered — straight assignment works
      v.currentTime = target;
      return;
    }

    // Past the buffered edge. Wait for more data; retry on each `progress`.
    let done = false;
    const tryApply = () => {
      if (done) return;
      if (target <= bufferedEnd() + 0.25) {
        done = true;
        v.removeEventListener("progress", tryApply);
        window.clearTimeout(timeoutId);
        v.currentTime = target;
      }
    };
    v.addEventListener("progress", tryApply);
    // Safety: after 4 s, take whatever's buffered and seek there so the
    // user gets SOME visible motion instead of a dead button.
    const timeoutId = window.setTimeout(() => {
      if (done) return;
      done = true;
      v.removeEventListener("progress", tryApply);
      const end = bufferedEnd();
      if (end > v.currentTime) {
        v.currentTime = end;
      }
    }, 4000);
  }, []);
  const skip = useCallback((deltaSec: number) => {
    if (onSkip) { onSkip(deltaSec); return; }
    seekRelative(deltaSec);
  }, [onSkip, seekRelative]);
  // WebView2 quirk: element-fullscreen silently drops the native MAXIMIZED
  // state of the (frameless) window, so leaving fullscreen "shrank" the app
  // to its restored size. Snapshot the state before entering fullscreen and
  // re-maximize on the way out.
  const wasMaximizedRef = useRef(false);
  const goFullscreen = useCallback(() => {
    const el = rootRef.current;
    if (!el) return;
    if (document.fullscreenElement) {
      void document.exitFullscreen();
    } else {
      void (async () => {
        try {
          if ("__TAURI_INTERNALS__" in window) {
            const { getCurrentWindow } = await import("@tauri-apps/api/window");
            wasMaximizedRef.current = await getCurrentWindow().isMaximized();
          }
        } catch { /* non-fatal */ }
        void el.requestFullscreen?.();
      })();
    }
  }, []);
  useEffect(() => {
    const restore = async () => {
      if (document.fullscreenElement || !wasMaximizedRef.current) return;
      wasMaximizedRef.current = false;
      try {
        const { getCurrentWindow } = await import("@tauri-apps/api/window");
        await getCurrentWindow().maximize();
      } catch { /* non-fatal */ }
    };
    document.addEventListener("fullscreenchange", restore);
    return () => {
      document.removeEventListener("fullscreenchange", restore);
      // Player unmounted while still fullscreen (e.g. Back) — the browser exits
      // fullscreen a beat later; heal the window state right after.
      if (wasMaximizedRef.current) setTimeout(() => { void restore(); }, 250);
    };
  }, []);

  // ── expose imperative handle ─────────────────────────────────────────
  useImperativeHandle(fwdRef, () => ({
    seekRelative,
    seekToWallClock: (ms: number) => mediaRef.current?.seekToDate(ms) ?? false,
    getPosition: () => videoRef.current?.currentTime ?? 0,
    getDuration: () => videoRef.current?.duration ?? NaN,
    togglePlay,
    play: () => setPlaying(true),
    toggleMute,
    toggleFullscreen: goFullscreen,
  }), [seekRelative, togglePlay, toggleMute, goFullscreen]);
  /* eslint-disable-next-line react-hooks/exhaustive-deps -- mediaRef is a ref */

  const onCanPlay = useCallback(() => {
    if (spinnerTimerRef.current) {
      window.clearTimeout(spinnerTimerRef.current);
      spinnerTimerRef.current = null;
    }
    // Success: cancel any pending retry and mark this source as having played so
    // a later mid-stream blip isn't mistaken for an initial-load failure.
    if (retryTimerRef.current) { window.clearTimeout(retryTimerRef.current); retryTimerRef.current = null; }
    playedRef.current = true;
    setShowSpinner(false);
    setShowFrozen(false);
    onSourceReady?.();
  }, [onSourceReady]);

  // Transient clip-load failures (segment still finalizing, ffmpeg cold-start,
  // brief file lock) arrive as MediaError.code 4. Instead of immediately showing
  // "clip not available" (the old behaviour — which the user worked around by
  // re-clicking), auto-retry a few cache-busted reloads first. Only a persistent
  // failure surfaces the warning. See clipRetry.ts.
  const handleVideoError = useCallback(() => {
    const v = videoRef.current;
    const err = v?.error;
    if (!playedRef.current && attemptRef.current < MAX_RETRIES) {
      const next = attemptRef.current + 1;
      attemptRef.current = next;
      if (retryTimerRef.current) window.clearTimeout(retryTimerRef.current);
      // Keep the spinner up; do NOT show the error yet.
      if (spinnerTimerRef.current) window.clearTimeout(spinnerTimerRef.current);
      spinnerTimerRef.current = window.setTimeout(() => setShowSpinner(true), SPINNER_DELAY_MS);
      retryTimerRef.current = window.setTimeout(() => {
        const vid = videoRef.current;
        if (!vid) return;
        attachTo(vid, bustUrl(src, next));
        vid.play().catch(() => {});
      }, retryDelayMs(next));
      return;
    }
    // Retries exhausted (or an error after successful playback): surface it.
    let msg = "This clip can't be played";
    if (err) {
      if (err.code === 4) msg = "Clip not available — no recording in range";
      else if (err.code === 3) msg = "Clip is corrupt or still being written";
      else if (err.code === 2) msg = "Network error fetching clip";
    }
    console.warn("[ClipOverlay] video error", err?.code, "after", attemptRef.current, "retries for", src);
    setLoadError(msg);
    setShowFrozen(false);
    setShowSpinner(false);
  }, [src, attachTo]);

  // Route hls.js fatal errors into the same retry/error machinery as <video>
  // element errors (see hlsFatalRef declaration above).
  useEffect(() => { hlsFatalRef.current = handleVideoError; }, [handleVideoError]);

  // Manual "Try again" — reset attempts and force a fresh (uniquely-busted)
  // request so a cached 404/empty body from the failed run isn't reused.
  const retryNow = useCallback(() => {
    if (retryTimerRef.current) { window.clearTimeout(retryTimerRef.current); retryTimerRef.current = null; }
    attemptRef.current = 0;
    playedRef.current = false;
    setLoadError(null);
    const v = videoRef.current;
    if (!v) return;
    attachTo(v, bustUrl(src, Date.now()));
    v.play().catch(() => {});
    if (spinnerTimerRef.current) window.clearTimeout(spinnerTimerRef.current);
    spinnerTimerRef.current = window.setTimeout(() => setShowSpinner(true), SPINNER_DELAY_MS);
  }, [src, attachTo]);

  return (
    <div
      ref={rootRef}
      className={styles.overlay}
      onMouseMove={bumpVisibility}
      onMouseEnter={bumpVisibility}
      onMouseLeave={armHide}
      onClick={togglePlay}
    >
      {/* Frozen-frame canvas — painted with the last decoded video frame
       *  before src reload. Stays visible until the new <video> emits
       *  canplay, then crossfades out. Eliminates the solid-black flash
       *  that happens by default during a source swap. */}
      <canvas
        ref={frozenCanvasRef}
        className={`${styles.frozenFrame} ${showFrozen ? styles.frozenVisible : ""}`}
        aria-hidden
      />

      <video
        ref={videoRef}
        className={styles.video}
        autoPlay
        playsInline
        muted={muted}
        onTimeUpdate={() => {
          const v = videoRef.current;
          if (!v || !onWallClock) return;
          // Mid-seek the element reports where it WAS. Say nothing until it lands.
          //
          // `v.seeking` is the browser's own flag and it is set SYNCHRONOUSLY by a
          // `currentTime` assignment, so it already covers every seek we start —
          // no hand-rolled ref, which could only add a way to get stuck gagged if
          // a `seeked` event never arrived. Frigate guards the same way with its
          // `isScrubbing` check.
          if (v.seeking) return;
          // A freshly attached or recovered element reports 0 before it has
          // positioned itself; publishing that snaps the playhead to the start of
          // the window. (Frigate guards the same `time == 0`.)
          if (v.currentTime === 0) return;

          // HLS: wall clock comes from the playlist's own PROGRAM-DATE-TIME, via
          // the fragment CONTAINING the playhead — the same map `seekToDate` uses
          // in reverse, so the needle and the seek cannot disagree.
          const media = mediaRef.current;
          const pdt = media?.playingDateMs();
          if (pdt != null) { onWallClock(pdt); return; }

          // An HLS position with no fragment is a position with no wall clock, and
          // the anchor CANNOT stand in for one. Media time 0 is the first fragment,
          // which the server may start up to 120 s before the requested anchor, so
          // `anchor + currentTime` double-counts that lead-in. Once playback ran
          // past the last fragment this reported a time 84 s beyond the end of all
          // recorded footage — a playhead in the future, climbing. Say nothing.
          if (media?.isHls) return;

          // Plain files (cached event clips) genuinely have no PDT; the anchor is
          // the only clock they have.
          //
          // This used to be `anchorRef.current ?? Date.now()`. A missing anchor
          // produced a base that MOVED — re-evaluated on every timeupdate while
          // `currentTime` also advanced — so the reported clock ran at 2x real
          // time from today's date. With no anchor there is no honest answer.
          const base = anchorRef.current;
          if (base == null) return;
          onWallClock(base + v.currentTime * 1000);
        }}
        onLoadedMetadata={() => {
          const v = videoRef.current;
          if (!v) return;
          v.playbackRate = rate; // src swaps reset the rate — re-apply
        }}
        onDurationChange={() => {
          const v = videoRef.current;
          if (!v) return;
        }}
        onCanPlay={onCanPlay}
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={onEnded}
        onError={handleVideoError}
      />

      {/* Loading spinner — fades in only after 500ms of no canplay. */}
      {showSpinner && !loadError && (
        <div className={styles.spinnerWrap} aria-hidden>
          <Loader2 className={styles.spinner} size={28} />
        </div>
      )}

      {loadError && (
        <div className={styles.errorOverlay}>
          <div className={styles.errorBox}>
            <strong>{loadError}</strong>
            <button
              className={styles.errorRetry}
              onClick={(e) => { e.stopPropagation(); retryNow(); }}
            >
              Try again
            </button>
          </div>
        </div>
      )}

      {/* Glass transport bar — bottom-mounted, auto-hides. Stop click
       *  propagation so clicking a control doesn't also toggle play. */}
      <div
        className={`${styles.controls} ${showCtrls ? styles.controlsVisible : ""}`}
        onClick={e => { e.stopPropagation(); holdVisible(); }}
        onMouseEnter={holdVisible}
        onMouseMove={holdVisible}
        onMouseLeave={armHide}
      >
        {onPrevEvent && (
          <button className={styles.btn} onClick={onPrevEvent}
            title="Previous event" aria-label="Previous event">
            <SkipBack size={14} />
          </button>
        )}
        <button className={styles.btnSkip} onClick={() => skip(-10)} title="Back 10 seconds" aria-label="Back 10 seconds">
          <Rewind size={14} />
          <span className={styles.skipLabel}>10</span>
        </button>
        <button className={styles.btnPlay} onClick={togglePlay}
          title={showSpinner ? "Loading…" : playing ? "Pause" : "Play"}
          aria-label={showSpinner ? "Loading" : playing ? "Pause" : "Play"}>
          {/* While the clip is still loading/retrying (on-demand event clips that
              aren't cached yet), the <video> reloads fire rapid play/pause events
              that flip `playing` → the button used to FLICKER Play↔Pause. Show a
              stable spinner during that window instead. Purely visual — the retry
              logic and `playing` state are untouched. */}
          {showSpinner
            ? <Loader2 className={styles.spinner} size={15} />
            : playing ? <Pause size={15} /> : <Play size={15} />}
        </button>
        <button className={styles.btnSkip} onClick={() => skip(10)} title="Forward 10 seconds" aria-label="Forward 10 seconds">
          <span className={styles.skipLabel}>10</span>
          <FastForward size={14} />
        </button>
        {onNextEvent && (
          <button className={styles.btn} onClick={onNextEvent}
            title="Next event" aria-label="Next event">
            <SkipForward size={14} />
          </button>
        )}

        {/* Speaker + hover-revealed volume slider. Clicking the speaker
         *  toggles mute; hovering it slides out a horizontal slider that
         *  lets the user set 0..1 volume. Like macOS audio menus. */}
        <div className={styles.volGroup}>
          <button className={styles.btn}
            onClick={toggleMute}
            title={muted ? "Unmute" : "Mute"}
            aria-label={muted ? "Unmute" : "Mute"}>
            {muted || volume === 0 ? <VolumeX size={14} /> : <Volume2 size={14} />}
          </button>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={muted ? 0 : volume}
            /* White fill up to the current level; the remainder stays the
             * faint track color. Inline because it tracks `volume` live. */
            style={{
              background: `linear-gradient(to right, #fff ${(muted ? 0 : volume) * 100}%, rgba(255,255,255,0.22) ${(muted ? 0 : volume) * 100}%)`,
            }}
            onChange={e => {
              const v = Number(e.target.value);
              if (v === 0) {
                // dragged down to 0 → mute and remember where we came from
                preMuteVolumeRef.current = volume || 0.8;
                setVolume(0);
                setMuted(true);
              } else {
                if (muted) setMuted(false);
                setVolume(v);
              }
            }}
            className={styles.volSlider}
            aria-label="Volume"
          />
        </div>
        {/* Playback speed — cycles 1 → 1.5 → 2 → 4 → 0.5 → 1. */}
        <button className={styles.btn}
          onClick={() => {
            const SPEEDS = [1, 1.5, 2, 4, 0.5];
            setRate(r => SPEEDS[(SPEEDS.indexOf(r) + 1) % SPEEDS.length]);
          }}
          title="Playback speed" aria-label="Playback speed"
          style={{ fontSize: 11, fontWeight: 800, minWidth: 34, fontVariantNumeric: "tabular-nums" }}>
          {rate}×
        </button>
        <button className={styles.btn} onClick={goFullscreen} title="Fullscreen" aria-label="Fullscreen">
          <Maximize2 size={14} />
        </button>
      </div>
    </div>
  );
});

