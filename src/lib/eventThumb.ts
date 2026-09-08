import type { StreamInfo } from "../api";

/**
 * Resolve an event's `thumbnail` field to an <img> src.
 *
 * Event LIST commands no longer ship inline base64 thumbnails (megabytes of JSON
 * over the WebView2 IPC bridge on every detection refetch — a major jank source).
 * They ship the tiny presence marker `"@thumb"` instead, and the image itself is
 * served by `GET /footage/:id/thumbnail` (Cache-Control: immutable → Chromium
 * caches and EVICTS it like a normal image, instead of pinning megabytes of
 * data-URIs in the DOM).
 *
 * Legacy inline base64 (e.g. review_segments' own thumbnail copy) still renders
 * as a data-URI, so every caller can use this one helper.
 */
export function eventThumbSrc(
  thumbnail: string | null | undefined,
  eventId: string | null | undefined,
  streamInfo: Pick<StreamInfo, "port" | "auth_token"> | null,
): string | null {
  if (!thumbnail) return null;
  if (thumbnail === "@thumb") {
    if (!eventId || !streamInfo) return null;
    return `http://localhost:${streamInfo.port}/footage/${eventId}/thumbnail?token=${streamInfo.auth_token}`;
  }
  return thumbnail.startsWith("data:") ? thumbnail : `data:image/jpeg;base64,${thumbnail}`;
}

/** Face crop ('@crop' marker → GET /face/:id/crop). Legacy inline base64 (e.g.
 *  stale panel caches from before the migration) still renders as a data-URI. */
export function faceCropSrc(
  thumbnail: string | null | undefined,
  faceId: string | null | undefined,
  streamInfo: Pick<StreamInfo, "port" | "auth_token"> | null,
): string | null {
  if (!thumbnail) return null;
  if (thumbnail === "@crop") {
    if (!faceId || !streamInfo) return null;
    return `http://localhost:${streamInfo.port}/face/${faceId}/crop?token=${streamInfo.auth_token}`;
  }
  return thumbnail.startsWith("data:") ? thumbnail : `data:image/jpeg;base64,${thumbnail}`;
}

/** Body-track crop ('@crop' marker → GET /body/:trackId/crop). Same legacy
 *  data-URI fallback contract as faceCropSrc. */
export function bodyCropSrc(
  thumbnail: string | null | undefined,
  trackId: string | null | undefined,
  streamInfo: Pick<StreamInfo, "port" | "auth_token"> | null,
): string | null {
  if (!thumbnail) return null;
  if (thumbnail === "@crop") {
    if (!trackId || !streamInfo) return null;
    return `http://localhost:${streamInfo.port}/body/${trackId}/crop?token=${streamInfo.auth_token}`;
  }
  return thumbnail.startsWith("data:") ? thumbnail : `data:image/jpeg;base64,${thumbnail}`;
}
