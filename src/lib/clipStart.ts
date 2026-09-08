import type { StreamInfo } from "../api";

/**
 * Ask the server where playback of a clip/seek will ACTUALLY start.
 *
 * Copy-concat playback (in-progress event clips, history seeking) starts at the
 * keyframe at/below the requested time — up to 1 s earlier with our 1 s GOP —
 * and a request before coverage starts at the first available segment. The UI
 * used to assume playback starts exactly at the REQUESTED time, so the needle,
 * scrubber and duration read ahead of the real video. `/clip-start` computes
 * the deterministic truth (no ffmpeg spawned); anchoring the player to it makes
 * the visuals match the video frame-for-frame.
 */
export async function fetchClipStartMs(
  streamInfo: Pick<StreamInfo, "port" | "auth_token"> | null,
  opts: { eventId?: string; cam?: number; startMs?: number },
): Promise<number | null> {
  if (!streamInfo) return null;
  const p = new URLSearchParams({ token: streamInfo.auth_token });
  if (opts.eventId) p.set("event", opts.eventId);
  else if (opts.startMs != null) {
    p.set("cam", String(opts.cam ?? 0));
    p.set("start", String(opts.startMs / 1000));
  } else return null;
  try {
    const r = await fetch(`http://localhost:${streamInfo.port}/clip-start?${p}`);
    if (!r.ok) return null;
    const j = await r.json();
    return typeof j.start_ms === "number" ? j.start_ms : null;
  } catch {
    return null;
  }
}
