// Shared retry policy for recorded-clip <video> loads.
//
// Recorded clips (/footage/:id/clip and /nvr-concat) are served by an ffmpeg
// concat of NVR segments. There are brief transient windows where the first
// request fails — a covering segment still finalizing (postprocess runs a
// faststart pass and indexes the row ~10s+ after a segment rotates), an ffmpeg
// concat cold-start, or a momentary Windows file lock. All surface to the
// browser as MediaError.code 4 (SRC_NOT_SUPPORTED). The historical "fix" was the
// user manually re-clicking the clip — which simply retried once the transient
// cleared. These helpers automate that: a few cache-busted reloads before any
// "clip not available" warning. Only a persistent failure (a genuinely empty /
// missing range, which the backend 404s) reaches the warning.

/** Maximum automatic reloads before surfacing the error to the user. */
export const MAX_RETRIES = 3;

/** Backoff before the Nth retry (attempt is 1-based). ~4.3s worst case. */
export function retryDelayMs(attempt: number): number {
  return [600, 1300, 2400][attempt - 1] ?? 2400;
}

/** Force a fresh request by appending a throwaway query param, so the browser
 *  can't serve a cached 404/empty body from the failed first attempt. */
export function bustUrl(url: string, attempt: number): string {
  if (attempt <= 0) return url;
  return url + (url.includes("?") ? "&" : "?") + "_retry=" + attempt;
}
