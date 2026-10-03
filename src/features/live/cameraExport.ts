// v14: Pure export helpers extracted from the (deleted) CameraHistoryModal.
// Used by the focus-mode ExportDropdown. Tauri save dialog + writeFile so we
// hit the actual filesystem path the user picks, not the browser sandbox.

import { save } from "@tauri-apps/plugin-dialog";
import { open, writeFile } from "@tauri-apps/plugin-fs";

import type { MotionEvent } from "../../types";

interface StreamInfo {
  port: number;
  auth_token: string;
}

const SAFE_NAME_RE = /[^A-Za-z0-9_-]+/g;
export function safeName(s: string): string {
  return s.replace(SAFE_NAME_RE, "_").slice(0, 32) || "camera";
}

export function ymdhm(ms: number): string {
  const d = new Date(ms);
  const pad = (n: number) => n.toString().padStart(2, "0");
  return `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}`;
}

export function toLocalInput(ms: number): string {
  const d = new Date(ms);
  const pad = (n: number) => n.toString().padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

type Toast = (msg: string, kind: "success" | "error" | "info") => void;

export async function downloadVideo(
  srcUrl: string,
  suggested: string,
  toast: Toast,
): Promise<void> {
  try {
    const dest = await save({
      defaultPath: suggested,
      filters: [{ name: "MP4 video", extensions: ["mp4"] }],
    });
    if (!dest) return;
    const resp = await fetch(srcUrl);
    if (!resp.ok || !resp.body) throw new Error(`HTTP ${resp.status}`);
    await streamToFile(resp.body, dest);
    toast(`Saved ${suggested}`, "success");
  } catch (e: any) {
    toast(`Export failed: ${e?.message ?? String(e)}`, "error");
  }
}

/** Write a download to disk as it arrives. This used to `arrayBuffer()` the
 *  whole response first, so an hour of footage sat in the WebView's memory —
 *  gigabytes — before a byte reached the file. Chunks are gathered to ~4 MB per
 *  write so a long export isn't thousands of tiny IPC calls. */
async function streamToFile(body: ReadableStream<Uint8Array>, dest: string): Promise<void> {
  const file = await open(dest, { write: true, create: true, truncate: true });
  const BATCH = 4 * 1024 * 1024;
  let parts: Uint8Array[] = [];
  let size = 0;
  const flush = async () => {
    if (size === 0) return;
    const buf = new Uint8Array(size);
    let off = 0;
    for (const p of parts) { buf.set(p, off); off += p.length; }
    await file.write(buf);
    parts = []; size = 0;
  };
  try {
    const reader = body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      parts.push(value); size += value.length;
      if (size >= BATCH) await flush();
    }
    await flush();
  } finally {
    await file.close();
  }
}

export function exportRelative(
  info: StreamInfo,
  camId: number,
  camLabel: string,
  secs: number,
  toast: Toast,
): Promise<void> {
  const end = Math.floor(Date.now() / 1000);
  const start = end - secs;
  const suggested = `${camLabel}_${ymdhm(start * 1000)}-${ymdhm(end * 1000)}.mp4`;
  const u = `http://localhost:${info.port}/nvr-export?cam=${camId}&start=${start}&end=${end}&download=1&filename=${encodeURIComponent(suggested)}&token=${info.auth_token}`;
  return downloadVideo(u, suggested, toast);
}

export function exportEvent(
  info: StreamInfo,
  camLabel: string,
  event: MotionEvent,
  toast: Toast,
): Promise<void> {
  return exportEventById(info, camLabel, event.id, toast);
}

/** Download an event's bounded clip by id (the Review bookmark-card download). */
export function exportEventById(
  info: StreamInfo,
  camLabel: string,
  eventId: string,
  toast: Toast,
): Promise<void> {
  const suggested = `${camLabel}_event-${eventId.slice(0, 8)}.mp4`;
  const u = `http://localhost:${info.port}/footage/${eventId}/clip?download=1&token=${info.auth_token}`;
  return downloadVideo(u, suggested, toast);
}

/** Export an arbitrary [startMs,endMs] window as an "incident" MP4 — used by the
 *  Review history view for review segments + bookmarks. Reuses the /nvr-export
 *  stitch route (no new backend). */
export function exportSegment(
  info: StreamInfo,
  camId: number,
  camLabel: string,
  startMs: number,
  endMs: number,
  toast: Toast,
): Promise<void> {
  const start = Math.floor(startMs / 1000);
  const end = Math.max(start + 1, Math.floor(endMs / 1000));
  const suggested = `${camLabel}_incident-${ymdhm(startMs)}.mp4`;
  const u = `http://localhost:${info.port}/nvr-export?cam=${camId}&start=${start}&end=${end}&download=1&filename=${encodeURIComponent(suggested)}&token=${info.auth_token}`;
  return downloadVideo(u, suggested, toast);
}

/** Save a plain-text incident report alongside the exported MP4 (Nx/UniFi-style),
 *  via the same Tauri save-dialog + writeFile path. */
export async function saveIncidentReport(
  text: string,
  suggested: string,
  toast: Toast,
): Promise<void> {
  try {
    const dest = await save({
      defaultPath: suggested,
      filters: [{ name: "Text report", extensions: ["txt"] }],
    });
    if (!dest) return;
    await writeFile(dest, new TextEncoder().encode(text));
    toast(`Saved ${suggested}`, "success");
  } catch (e: any) {
    toast(`Report failed: ${e?.message ?? String(e)}`, "error");
  }
}

export function exportCustom(
  info: StreamInfo,
  camId: number,
  camLabel: string,
  startLocal: string,
  endLocal: string,
  toast: Toast,
): Promise<void> | void {
  if (!startLocal || !endLocal) {
    toast("Pick a start and end", "error");
    return;
  }
  const s = new Date(startLocal).getTime();
  const e = new Date(endLocal).getTime();
  if (!isFinite(s) || !isFinite(e) || e <= s) {
    toast("End must be after start", "error");
    return;
  }
  const start = Math.floor(s / 1000);
  const end = Math.floor(e / 1000);
  const suggested = `${camLabel}_${ymdhm(s)}-${ymdhm(e)}.mp4`;
  const u = `http://localhost:${info.port}/nvr-export?cam=${camId}&start=${start}&end=${end}&download=1&filename=${encodeURIComponent(suggested)}&token=${info.auth_token}`;
  return downloadVideo(u, suggested, toast);
}
