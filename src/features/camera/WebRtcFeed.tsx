import { useEffect, useRef, useState } from "react";
import { HlsFeed } from "./HlsFeed";

/**
 * Sub-second live feed via WebRTC (WHEP → go2rtc restream, mature NVRs' model).
 * The SDP handshake goes through our token-authed server (`POST /webrtc/:cam`);
 * media then flows peer-to-peer from go2rtc's zero-re-encode restream of the
 * camera's own H.264. Ladder: WebRTC → HLS (~3 s) → MJPEG — any failure
 * (go2rtc missing, depth-anonymized cam, codec the WebView can't take)
 * silently drops one rung, so the tile is never blank.
 */
export function WebRtcFeed({ port, token, camId, className }: {
  port: number;
  token: string;
  camId: number;
  className?: string;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [fallback, setFallback] = useState(false);

  useEffect(() => {
    setFallback(false);
    const video = videoRef.current;
    if (!video) return;
    let dead = false;
    const pc = new RTCPeerConnection();
    pc.ontrack = (e) => {
      if (video.srcObject !== e.streams[0]) video.srcObject = e.streams[0];
      video.play().catch(() => {});
    };
    pc.addTransceiver("video", { direction: "recvonly" });
    pc.addTransceiver("audio", { direction: "recvonly" });
    pc.onconnectionstatechange = () => {
      if (!dead && ["failed", "closed"].includes(pc.connectionState)) setFallback(true);
    };

    (async () => {
      try {
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);
        // Non-trickle WHEP: wait for ICE gathering so the offer carries all
        // candidates in one shot (bounded — localhost gathering is instant).
        await new Promise<void>((res) => {
          if (pc.iceGatheringState === "complete") return res();
          const t = window.setTimeout(res, 2000);
          pc.onicegatheringstatechange = () => {
            if (pc.iceGatheringState === "complete") { window.clearTimeout(t); res(); }
          };
        });
        const resp = await fetch(`http://localhost:${port}/webrtc/${camId}`, {
          method: "POST",
          headers: { "Content-Type": "application/sdp", Authorization: `Bearer ${token}` },
          body: pc.localDescription!.sdp,
        });
        if (!resp.ok) throw new Error(`WHEP ${resp.status}`);
        const answer = await resp.text();
        if (dead) return;
        await pc.setRemoteDescription({ type: "answer", sdp: answer });
      } catch {
        if (!dead) setFallback(true);
      }
    })();

    // No playable frame within a few seconds → drop to the HLS rung.
    const warmup = window.setTimeout(() => {
      if (!dead && video.readyState < 2) setFallback(true);
    }, 6000);

    return () => { dead = true; window.clearTimeout(warmup); pc.close(); };
  }, [port, token, camId]);

  if (fallback) {
    return <HlsFeed port={port} token={token} camId={camId} className={className} />;
  }
  return <video ref={videoRef} className={className} autoPlay playsInline muted />;
}
