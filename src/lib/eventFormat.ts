// Shared event-display helpers: time chips, risk banding, and v2 ai_summary
// JSON parsing. Extracted from NVRPanel so Review (and anything else showing
// event cards) doesn't have to import pure formatters from a 2,400-line panel.

import { SEVERITY_FG, severityOfScore } from "./palette";

const pad = (n: number) => String(n).padStart(2, "0");

export function fmtShortTime(ms: number, use12h = false): string {
  const d = new Date(ms);
  if (use12h) {
    const h = d.getHours(); const ampm = h >= 12 ? "PM" : "AM";
    return `${h % 12 || 12}:${pad(d.getMinutes())} ${ampm}`;
  }
  return `${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

export function riskColor(score: number) {
  return SEVERITY_FG[severityOfScore(score)];
}

export function riskLabel(score: number) {
  if (score > 0.5) return "HIGH";
  if (score > 0.3) return "MED";
  return "LOW";
}

/** Parse v2 ai_summary JSON and return just the human-readable text. */
export function aiText(raw: string | null | undefined): string | null {
  if (!raw) return null;
  const trimmed = raw.trim();
  if (!trimmed.startsWith("{")) return trimmed; // already plain text
  try {
    const d = JSON.parse(trimmed);
    const text = d.text ?? d.description ?? d.summary ?? null;
    if (text && typeof text === "string" && text.trim().length > 0) return text.trim();
  } catch {}
  return null; // don't show raw JSON
}

/** Extract the short VLM headline ("title") from v2+ ai_summary JSON. */
export function aiTitle(raw: string | null | undefined): string | null {
  if (!raw) return null;
  const trimmed = raw.trim();
  if (!trimmed.startsWith("{")) return null;
  try {
    const d = JSON.parse(trimmed);
    if (d.title && typeof d.title === "string" && d.title.trim().length > 1) return d.title.trim();
  } catch {}
  return null;
}
