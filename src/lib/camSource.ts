/**
 * Camera-source SSOT (shape).
 *
 * `camera_configs` (DB) is the authority for what each camera is. CameraView
 * auto-starts a camera from a `cam_source_<id>` localStorage entry. This module
 * is the ONE place that maps a saved config → that localStorage shape, so the
 * handful of writers can't drift (they used to keep separate copies of this
 * mapping — including one that hard-coded `nativeIndex: 0`, which would start the
 * wrong device for a second USB camera).
 */
import type { CameraConfig } from "../types";

type SourceCfg = Pick<CameraConfig, "source_type" | "source_url" | "device_id" | "name">;

/** Map a saved config to the `cam_source_<id>` object CameraView consumes. */
export function camSourceFromConfig(cfg: SourceCfg): Record<string, unknown> {
  if (cfg.source_type === "browser" || cfg.source_type === "native") {
    // USB/integrated — captured server-side. `device_id` holds the native index.
    return {
      kind: "native",
      deviceId: cfg.device_id,
      nativeIndex: Number(cfg.device_id) || 0,
      label: cfg.name || cfg.device_id || "Camera",
    };
  }
  if (cfg.source_type === "rtsp") {
    return { kind: "rtsp", url: cfg.source_url, label: cfg.name };
  }
  return { kind: "mjpeg", url: cfg.source_url, label: cfg.name };
}

/** Save a camera slot's source WITHOUT its URL or login: a network camera's URL
 *  carries its password, and the DB keeps it encrypted. CameraView fills the URL
 *  in from the camera's config when it starts the camera. */
export function storeCamSource(camId: number, src: Record<string, unknown>) {
  const kept = { ...src };
  delete kept.url; delete kept.authUser; delete kept.authPass;
  try { localStorage.setItem(`cam_source_${camId}`, JSON.stringify(kept)); } catch { /* quota */ }
}

/** Write the derived source for a camera slot. */
export function writeCamSource(camId: number, cfg: SourceCfg) {
  storeCamSource(camId, camSourceFromConfig(cfg));
}

/** Put a typed login into a camera URL (`rtsp://user:pass@host/…`), where ffmpeg
 *  and the HTTP client both read it. A URL that already has one, or isn't a URL,
 *  is returned unchanged. */
export function withUrlCredentials(url: string, user: string, pass: string): string {
  const m = url.trim().match(/^([a-z][a-z\d+.-]*:\/\/)([^/]*@)?(.*)$/i);
  if (!m || m[2] || (!user && !pass)) return url;
  return `${m[1]}${encodeURIComponent(user)}${pass ? ":" + encodeURIComponent(pass) : ""}@${m[3]}`;
}

/** A camera URL with its login removed, for showing in messages. */
export function hideUrlLogin(url: string): string {
  return url.replace(/^([a-z][a-z\d+.-]*:\/\/)[^/]*@/i, "$1");
}

/** Forget a camera slot's cached source (on delete / disable). */
export function clearCamSource(camId: number) {
  try { localStorage.removeItem(`cam_source_${camId}`); } catch { /* ignore */ }
}
