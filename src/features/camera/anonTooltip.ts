import { depthModelSkill } from "../agent/skillDownload";

/** Tooltips for the camera's server-side privacy toggle.
 *
 *  The Depth model size is read from the `depth_anything` catalog entry rather
 *  than a literal, so this button and the Arsenal card cannot disagree about how
 *  big the download is. */
export function anonymizeTooltip(anonOn: boolean): string {
  if (anonOn) {
    return "Depth Anonymization ON (server): recordings, streams, snapshots and alerts contain ONLY the depth map. Local AI still detects on raw frames in memory. Click to restore raw video.";
  }
  return `Anonymize this camera at the source: everything stored or sent becomes a colorized depth map — identities never persist. Requires the Depth model (${depthModelSkill().sizeLabel}, downloads on first use).`;
}
