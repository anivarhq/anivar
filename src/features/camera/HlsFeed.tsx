import { useEffect, useRef, useState } from "react";
import Hls from "hls.js";

/**
 * Live camera feed via HLS (`<video>` + hardware decode) instead of the old
 * MJPEG-in-`<img>`. This is mature NVRs' model: the backend already produces an H.264
 * HLS stream (`spawn_hls_pipe` → `/hls/camN.m3u8`); a `<video>` element plays it
 * through the GPU's video pipeline, so the compositor shows a cheap video texture
 * instead of decoding + re-compositing a full JPEG every frame (which, stacked under
 * the 188 liquid-glass backdrop-filters, was the source of the UI lag).
 *
 * WebView2/Chromium has no native HLS, so we use hls.js (MSE) — the same library
 * mature NVRs' UI uses. If HLS fails or the pipe is still warming up, we fall back to
 * the proven MJPEG `<img>` so the feed is NEVER blank, and try HLS again every
 * `RETRY_MS`: the fallback used to be for the rest of the session, and MJPEG
 * costs far more CPU and bandwidth.
 */
const RETRY_MS = 30_000;

export function HlsFeed({ port, token, camId, className }: {
  port: number;
  token: string;
  camId: number;
  className?: string;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [fallback, setFallback] = useState(false);
  // Bumped to try HLS again after a fallback.
  const [attempt, setAttempt] = useState(0);
  const base = `http://localhost:${port}`;

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    if (!Hls.isSupported()) { setFallback(true); return; }

    const hls = new Hls({
      // SMOOTHNESS over raw latency (~3s behind live): lowLatencyMode is meant
      // for LL-HLS partial segments we don't produce — with plain 1s segments
      // it just made hls.js ride the live edge with ~2s of buffer, so any
      // hiccup underran and stuttered. Sync 3 segments back, buffer ~10s,
      // and cap drift so latency can't grow unbounded.
      lowLatencyMode: false,
      liveSyncDurationCount: 3,
      liveMaxLatencyDurationCount: 8,
      backBufferLength: 6,
      maxBufferLength: 10,
      // Every request (playlist + .ts segments) carries the desktop auth token.
      xhrSetup: (xhr) => xhr.setRequestHeader("Authorization", `Bearer ${token}`),
    });

    let dead = false;
    let retry: number | undefined;
    // Destroy before falling back: the `<video>` stays mounted (hidden) under the
    // MJPEG `<img>`, and a live hls.js would keep downloading the playlist.
    const fail = () => {
      if (dead) return;
      dead = true; hls.destroy(); setFallback(true);
      retry = window.setTimeout(() => setAttempt(a => a + 1), RETRY_MS);
    };
    // Back on HLS only once it actually has a picture, so a retry against a
    // camera that's still down never blanks the MJPEG view.
    const onPicture = () => { if (!dead) setFallback(false); };
    video.addEventListener("loadeddata", onPicture);
    hls.on(Hls.Events.ERROR, (_evt, data) => {
      // Only a FATAL error (or repeated stalls) drops us to MJPEG — transient network
      // blips are recovered by hls.js itself.
      if (data.fatal) fail();
    });
    hls.loadSource(`${base}/hls/cam${camId}.m3u8`);
    hls.attachMedia(video);
    video.play().catch(() => { /* autoplay policies — muted so this rarely fires */ });

    // If the HLS pipe hasn't produced a playable frame within a few seconds (e.g. the
    // camera capture is still starting up), show MJPEG meanwhile rather than a black box.
    const warmup = window.setTimeout(() => {
      if (video.readyState < 2) fail();
    }, 6000);

    return () => {
      window.clearTimeout(warmup); window.clearTimeout(retry);
      video.removeEventListener("loadeddata", onPicture);
      hls.destroy();
    };
  }, [port, token, camId, attempt]);

  return (
    <>
      <video ref={videoRef} className={className} autoPlay playsInline muted
        style={fallback ? { display: "none" } : undefined} />
      {fallback && (
        <img
          className={className}
          src={`${base}/stream?cam=${camId}&token=${token}`}
          alt=""
        />
      )}
    </>
  );
}
