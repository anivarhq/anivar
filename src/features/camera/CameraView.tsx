import { useRef, useEffect, useCallback, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { useStore } from "../../store";
import { findSkill, downloadSkill } from "../agent/skillDownload";
import { useShallow } from "zustand/react/shallow";
import { api, DiscoveredCamera, StreamInfo, Detection, BrowserCameraInfo } from "../../api";
import { HlsFeed } from "./HlsFeed";
import { WebRtcFeed } from "./WebRtcFeed";
import { BOX_COLOR } from "../../lib/palette";

import styles from "./CameraView.module.css";
import {
  Camera, CameraOff, RefreshCw,
  Wifi, WifiOff, Search, Link, Plus, ChevronRight, Eye, EyeOff,
} from "lucide-react";

const STUN_SERVERS = [
  { urls: "stun:stun.l.google.com:19302" },
  { urls: "stun:stun1.l.google.com:19302" },
  { urls: "stun:stun.cloudflare.com:3478" },
];

const MOTION_COOLDOWN_MS = 1500;

/**
 * Resolve a `var(--token)` string to a concrete colour for canvas drawing.
 *
 * `ctx.strokeStyle` cannot take a CSS variable, so the detection overlay has to
 * read the computed value. Cached: this runs per box per frame, and
 * getComputedStyle forces a style recalc.
 */
const cssVarCache = new Map<string, string>();
function cssVar(token: string): string {
  // A literal colour is returned untouched. Without this guard the lookup below
  // asks getPropertyValue("#6FA97C"), gets "", and falls through to the default
  // — so every pinned BOX_COLOR would render as the same off-white.
  if (!token.startsWith("var(")) return token;
  const hit = cssVarCache.get(token);
  if (hit) return hit;
  const name = token.replace(/^var\(\s*|\s*\)$/g, "");
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const out = v || "#F5F2EF"; // never leave the canvas with an empty strokeStyle
  cssVarCache.set(token, out);
  return out;
}

/** `#rrggbb` + alpha -> `rgba(...)`, for canvas gradients that need transparency. */
function withAlpha(hex: string, a: number): string {
  const h = hex.replace("#", "");
  if (h.length < 6) return hex;
  const [r, g, b] = [0, 2, 4].map(i => parseInt(h.slice(i, i + 2), 16));
  return `rgba(${r},${g},${b},${a})`;
}

// Refresh the store's event feed at most once per 2 s across ALL camera tiles.
// Both the detection and motion paths used to refetch the full 500-row event
// list on EVERY frame while anything moved — a continuous multi-KB IPC storm
// that made the whole UI feel laggy during activity. Trailing-coalesced: the
// feed still updates promptly after motion, once, for all tiles together.
let eventsRefreshTimer: ReturnType<typeof setTimeout> | null = null;
function scheduleEventsRefresh() {
  if (eventsRefreshTimer) return;
  eventsRefreshTimer = setTimeout(() => {
    eventsRefreshTimer = null;
    api.getMotionEvents(500).then(useStore.getState().setEvents).catch(() => {});
  }, 2000);
}

// ── Fix WebM duration so the browser knows total length and can seek ──────────
// MediaRecorder leaves Duration=0xFFFFFFFFFFFFFFFF (unknown) in the header.
// Writing the real duration allows the browser to seek by percentage.
// Reference: https://www.matroska.org/technical/specs/index.html
async function fixWebmDuration(blobs: Blob[], durationMs: number, mimeType: string): Promise<Blob[]> {
  try {
    const blob = new Blob(blobs, { type: mimeType });
    const buf  = await blob.arrayBuffer();
    const view = new DataView(buf);
    const bytes = new Uint8Array(buf);

    // Find the Duration element (0x4489) in the first 2KB (Segment Info)
    const search = Math.min(buf.byteLength, 2048);
    for (let i = 0; i < search - 4; i++) {
      if (bytes[i] === 0x44 && bytes[i + 1] === 0x89) {
        // Next byte is the size of the float field
        const size = bytes[i + 2];
        if (size === 0x84) { // 4-byte float
          const seconds = durationMs / 1000;
          view.setFloat32(i + 3, seconds, false);
          return [new Blob([buf], { type: mimeType })];
        } else if (size === 0x88) { // 8-byte double
          const seconds = durationMs / 1000;
          view.setFloat64(i + 3, seconds, false);
          return [new Blob([buf], { type: mimeType })];
        }
      }
    }
  } catch { /* keep original if patch fails */ }
  return blobs;
}
// Mature NVRs processes motion detection on every frame at 30fps.
// We use 20fps (50ms) — a safe balance between latency and IPC load.
const CAPTURE_INTERVAL_MS  = 50;  // 20 FPS → ~50ms motion detection latency

// ── Person tracker (SORT-lite) ─────────────────────────────────────────────
// Carries an identity (name) across frames via bounding-box IoU matching so
// the person only needs their face verified once, even at distance afterwards.
interface PersonTrack {
  id: number;
  name: string | null;          // null = unidentified
  box: Detection["box"];
  vx: number; vy: number;       // velocity in image-space pixels (updated each match)
  lastSeenMs: number;
  missedFrames: number;
  firstSeenMs: number;          // when this track was created
  posHistory: { cx: number; cy: number }[];  // last 10 centre positions
  behaviorFlags: Set<string>;   // "loitering" | "running" | "erratic"
}

function boxIoU(a: Detection["box"], b: Detection["box"]): number {
  const ix1 = Math.max(a.xmin, b.xmin), iy1 = Math.max(a.ymin, b.ymin);
  const ix2 = Math.min(a.xmax, b.xmax), iy2 = Math.min(a.ymax, b.ymax);
  const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
  if (inter === 0) return 0;
  const aA = (a.xmax - a.xmin) * (a.ymax - a.ymin);
  const bA = (b.xmax - b.xmin) * (b.ymax - b.ymin);
  return inter / (aA + bA - inter);
}

// Predict where a track's box will be in the next frame using its velocity
function predictBox(t: PersonTrack): Detection["box"] {
  return {
    xmin: t.box.xmin + t.vx, ymin: t.box.ymin + t.vy,
    xmax: t.box.xmax + t.vx, ymax: t.box.ymax + t.vy,
  };
}

// Labels that represent a living presence — keep tracking alive when detected
const PRESENCE_LABELS = new Set([
  "person",
  "cat", "dog", "bird",
  "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
]);

type CameraSource =
  | { kind: "native"; nativeIndex: number; deviceId?: string; label: string }
  | { kind: "mjpeg";  url: string; label: string; authUser?: string; authPass?: string }
  | { kind: "rtsp";   url: string; label: string; authUser?: string; authPass?: string };

/**
 * Probe a camera base URL server-side to find the actual MJPEG stream endpoint.
 * Uses the Rust probe_mjpeg_url command which bypasses browser CSP.
 */
async function resolveMjpegUrl(url: string, authUser?: string, authPass?: string): Promise<string> {
  if (!url) return url;
  try {
    const resolved = await api.probeMjpegUrl(url, authUser, authPass);
    return resolved || url;
  } catch { return url; }
}

/**
 * Build a proxied URL for an MJPEG camera stream.
 * Routes through localhost:PORT/cam-proxy to bypass Tauri CSP restrictions.
 * Includes optional Basic Auth credentials so the proxy can authenticate with the camera.
 */
function buildProxyUrl(
  streamUrl: string,
  streamInfo: { port: number; auth_token: string } | null,
  authUser?: string,
  authPass?: string,
): string {
  if (!streamInfo || !streamUrl) return streamUrl;
  let url = `http://localhost:${streamInfo.port}/cam-proxy?url=${encodeURIComponent(streamUrl)}&token=${streamInfo.auth_token}`;
  if (authUser) url += `&user=${encodeURIComponent(authUser)}`;
  if (authPass) url += `&pass=${encodeURIComponent(authPass)}`;
  return url;
}

interface BrowserCamera { deviceId: string; label: string }

export function CameraView({ camId = 0, onRemove, cornered }: {
  camId?: number;
  onRemove?: () => void;
  /** v14: applies rounded-corner + soft-shadow styling to the viewport.
   *  Used by the focus mode so the camera reads as a discrete card against
   *  the focus background. Grid tiles stay edge-to-edge. */
  cornered?: boolean;
}) {
  const captureCanvasRef  = useRef<HTMLCanvasElement>(null);
  const overlayRef        = useRef<HTMLCanvasElement>(null);
  const mjpegImgRef       = useRef<HTMLImageElement>(null);
  const motionCoolRef     = useRef(0);
  const fullRef           = useRef<HTMLDivElement>(null);
  const captureIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const captureRafRef     = useRef<number>(0);          // requestAnimationFrame id
  const prevGrayRef       = useRef<Uint8Array | null>(null); // for JS pre-screen
  const jsMotionRef       = useRef(false);              // instant JS motion state
  const tinyCanvasRef     = useRef<OffscreenCanvas | null>(null);
  const isActiveRef       = useRef(false);
  const fpsFramesRef      = useRef(0);
  // Mature NVRs' "skip if busy" pattern: never queue frames, just drop them.
  // Without this, 50ms intervals with 150ms IPC builds up a backlog causing
  // detections to arrive seconds after the actual motion.
  const frameInFlightRef  = useRef(false);
  const fpsLastRef        = useRef(Date.now());
  const activeEventIdRef   = useRef<string | null>(null); // latest event_id from Rust
  const liveDetectionsRef  = useRef<Detection[]>([]);     // latest AI detections (from Rust YOLO26)
  const lastFrameStoreRef  = useRef<number>(0);           // throttle latest-frame store writes
  const personNumberMapRef = useRef<Map<string, number>>(new Map()); // pos-key → person number
  const personNumberSeqRef = useRef(0);                              // counter reset per event
  const lastEventIdForNumRef = useRef<string | null>(null);          // detect event change
  // IoU-based person tracker — carries identity across frames without needing a visible face
  const personTracksRef = useRef<PersonTrack[]>([]);
  const trackIdSeqRef   = useRef(0);

  // ── NVR is now fully Rust-side (ffmpeg pipe in run_capture_loop) ─────────────
  // No browser MediaRecorder for NVR. The Rust inference loop also pushes
  // detections via "detections:update" events — no frame IPC for AI.

  // The YOLO26 status badge this drove was removed in an earlier declutter, but
  // its state, ref and FOUR Tauri subscriptions (inference:status / :tick /
  // :ready + a getInferenceStatus call) stayed behind writing to nothing. The
  // information itself mattered — a detector that fails to load was otherwise
  // silent — so it now renders once, in the sidebar Activity popover, instead of
  // per camera tile. See TelemetryPopover in App.tsx.

  // Receive YOLO26 bounding boxes from Rust native inference — no frame data crosses IPC.
  // This is the only detection path. The JS web worker is no longer used.
  useEffect(() => {
    const u = listen<{ cam_id: number; detections: Detection[] }>("detections:update", ({ payload }) => {
      if (payload.cam_id !== camId) return;
      const dets = payload.detections;
      liveDetectionsRef.current = dets;
      setLiveDetections(dets);

      if (dets.length > 0) {
        const eventId = activeEventIdRef.current;
        drawDetectionBoxes(dets, eventId);

        const hasPresence = dets.some(d => PRESENCE_LABELS.has(d.label.toLowerCase()));
        if (hasPresence && eventId) api.keepAliveEvent(camId).catch(() => {});

        updateTracks(dets);

        const hasPerson = dets.some(d => d.label.toLowerCase() === "person");
        if (hasPerson && captureCanvasRef.current) runFaceRecognition(captureCanvasRef.current, dets);

        // Persist detections to DB so agent clip analysis can use them
        if (eventId) {
          api.storeDetections(eventId, JSON.stringify(dets)).catch(() => {});
        }

        // Crowd counting
        const personCount = dets.filter(d => d.label.toLowerCase() === "person").length;
        if (personCount > 1) api.reportCrowdCount(camId, personCount, eventId).catch(() => {});

        if (camId === 0) useStore.getState().setLatestDetections(dets.map(d => ({ label: d.label, score: d.score })));

        // Refresh event list so UI shows updated detection data (coalesced).
        scheduleEventsRefresh();
      } else {
        clearOverlay();
      }
    });
    return () => { u.then(f => f()); };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [camId]);

  // ── Depth privacy ─────────────────────────────────────────────────────────────
  // NOTE: there is no client-side "depth view" overlay any more.
  //
  // It polled a snapshot every 700 ms and composited it over the live feed,
  // which produced the ghosting and stutter this replaced — a ~1.4 fps image
  // over 30 fps of video can never line up. It was also showing the RAW frame:
  // get_camera_snapshot only renders depth when SERVER anonymisation is on, and
  // this toggle was independent of it.
  //
  // Redundant anyway. When anonymisation IS on, the capture pipeline converts
  // frames to depth before any consumer sees them (dshow.rs / go2rtc.rs), so
  // the live feed is already a depth map — smoothly, at full frame rate, with
  // no client work at all. The 🛡 button below toggles that real setting.
  const [anonBusy, setAnonBusy]           = useState(false);
  const svSettings = useStore(st => st.settings);
  const anonOn = (() => {
    try { return JSON.parse(svSettings?.depth_anonymize || "{}")[String(camId)] === true; }
    catch { return false; }
  })();
  const toggleAnonymize = async () => {
    if (anonBusy || !svSettings) return;
    setAnonBusy(true);
    try {
      if (!anonOn) {
        // The server model must exist BEFORE we promise anonymization
        // (backend fails closed: no model = blank feed, never raw leaks).
        const installed = await api.listInstalledSkills();
        if (!installed.some(sk => sk.id === "depth_anything" && sk.installed)) {
          const skill = findSkill("depth_anything");
          if (skill) {
            useStore.getState().showToast("Downloading depth model (~50 MB, FP16)…", "info");
            await downloadSkill(skill, () => {});
          }
        }
      }
      const map = (() => { try { return JSON.parse(svSettings.depth_anonymize || "{}"); } catch { return {}; } })();
      map[String(camId)] = !anonOn;
      await api.saveSettings({ ...svSettings, depth_anonymize: JSON.stringify(map) });
      useStore.getState().setSettings({ ...svSettings, depth_anonymize: JSON.stringify(map) });
      useStore.getState().showToast(
        !anonOn ? "Depth Anonymization ON — recordings & streams are now depth-only"
                : "Depth Anonymization OFF — raw video restored", "success");
    } catch (e) {
      useStore.getState().showToast(`Anonymization toggle failed: ${e}`, "error");
    } finally { setAnonBusy(false); }
  };
  // The server-rendered anonymized frame, shown in place of the live image.

  // Per-event clips are recorded SERVER-SIDE, sliced from the continuous NVR
  // recording — never a browser MediaRecorder (hard rule; see memory).

  // Subscribe with a SHALLOW SELECTOR — a naked useStore() re-rendered this whole
  // 1400-line component (feed + canvases) on EVERY store change, including the
  // per-tick motionScore/fps churn IT writes itself (self-inflicted re-render loop).
  const {
    setCameraActive,
    setFps,
    showToast,
    settings,
  } = useStore(useShallow(s => ({
    setCameraActive: s.setCameraActive,
    setFps: s.setFps,
    showToast: s.showToast,
    settings: s.settings,
  })));

  const streamPort  = settings?.stream_port ?? 8880;

  const [isActive, setIsActive]           = useState(false);
  const [error, setError]                 = useState<string | null>(null);
  const [source, setSource]               = useState<CameraSource | null>(null);
  const [setupTab, setSetupTab]           = useState<"local" | "network" | "manual">("local");
  const [browserCameras, setBrowserCameras] = useState<BrowserCamera[]>([]);
  const [discovered, setDiscovered]       = useState<DiscoveredCamera[]>([]);
  const [scanning, setScanning]           = useState(false);
  const [manualUrl, setManualUrl]         = useState("");
  const [streamInfo, setStreamInfo]       = useState<StreamInfo | null>(null);
  const [liveDetections, setLiveDetections] = useState<Detection[]>([]);
  const [, setRecognizedPersons] = useState<string[]>([]);
  const lastFaceRecogRef = useRef(0);
  // face recognition results: name + face bounding box in image pixel space
  const faceResultsRef = useRef<{ name: string; fx: number; fy: number; fw: number; fh: number }[]>([]);
  // Persistent identity cache: once verified, remember by spatial bucket + appearance color
  const verifiedIdentitiesRef = useRef<Map<string, { name: string; hue: number; sat: number; lum: number; lastSeen: number }>>(new Map());

  useEffect(() => { isActiveRef.current = isActive; }, [isActive]);

  // NVR auto-start: Rust-side ffmpeg pipe is already running from startup.
  // When camera goes live, Rust's process_frame_inner automatically pipes frames.
  // No browser action needed — NVR is headless and survives UI crashes.

  // Report current detections to Rust every 5 s so Guardian can track objects
  useEffect(() => {
    if (!isActive) return;
    const id = setInterval(() => {
      const objs = liveDetectionsRef.current.map(d => ({ label: d.label, score: d.score }));
      api.updateSceneObjects(camId, objs).catch(() => {});
    }, 5000);
    return () => clearInterval(id);
  }, [isActive, camId]);

  // Report behavior flags to Rust every 10 s
  useEffect(() => {
    if (!isActive) return;
    const id = setInterval(() => {
      const behaviors = personTracksRef.current
        .filter(t => t.behaviorFlags.size > 0)
        .map(t => ({
          track_id: t.id,
          name: t.name,
          flags: [...t.behaviorFlags],
          duration_secs: Math.round((Date.now() - t.firstSeenMs) / 1000),
        }));
      if (behaviors.length > 0) {
        api.reportBehaviorEvents(camId, behaviors).catch(() => {});
      }
    }, 10_000);
    return () => clearInterval(id);
  }, [isActive, camId]);

  useEffect(() => {
    api.getStreamInfo().then(setStreamInfo).catch(() => {});
  }, []);

  // Sample dominant hue/sat/lum from the torso region of a bounding box
  function sampleAppearance(canvas: HTMLCanvasElement, box: { xmin: number; ymin: number; xmax: number; ymax: number }): { hue: number; sat: number; lum: number } | null {
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    const margin = 0.25;
    const sx = Math.floor(box.xmin + (box.xmax - box.xmin) * margin);
    // Sample upper-body / torso: skip top 20% (face area) and bottom 20% (legs)
    const sy = Math.floor(box.ymin + (box.ymax - box.ymin) * 0.2);
    const sw = Math.floor((box.xmax - box.xmin) * (1 - 2 * margin));
    const sh = Math.floor((box.ymax - box.ymin) * 0.6);
    if (sw <= 2 || sh <= 2) return null;
    const data = ctx.getImageData(sx, sy, sw, sh).data;
    let r = 0, g = 0, b = 0;
    const n = data.length / 4;
    for (let i = 0; i < data.length; i += 4) { r += data[i]; g += data[i + 1]; b += data[i + 2]; }
    r /= n; g /= n; b /= n;
    const max = Math.max(r, g, b) / 255, min = Math.min(r, g, b) / 255;
    const lum = (max + min) / 2;
    const d = max - min;
    const sat = d < 0.001 ? 0 : d / (1 - Math.abs(2 * lum - 1));
    let hue = 0;
    if (d > 0.01) {
      const nr = r / 255, ng = g / 255, nb = b / 255;
      const mx = Math.max(nr, ng, nb), mn = Math.min(nr, ng, nb), df = mx - mn;
      if (mx === nr) hue = 60 * (((ng - nb) / df) % 6);
      else if (mx === ng) hue = 60 * ((nb - nr) / df + 2);
      else hue = 60 * ((nr - ng) / df + 4);
      if (hue < 0) hue += 360;
    }
    return { hue, sat, lum };
  }

  // Spatial bucket key — 80px grid for position tolerance
  function personBucketKey(cx: number, cy: number) {
    return `${Math.round(cx / 80)}_${Math.round(cy / 80)}`;
  }

  // Look up a verified identity by spatial bucket only (color matching removed — too many false positives)
  function lookupVerifiedIdentity(cx: number, cy: number): string | null {
    const bucketKey = personBucketKey(cx, cy);
    const direct = verifiedIdentitiesRef.current.get(bucketKey);
    if (direct) { direct.lastSeen = Date.now(); return direct.name; }
    return null;
  }

  // Match new detections to existing tracks via IoU, update velocities, create/drop tracks
  const updateTracks = useCallback((detections: Detection[]) => {
    const now = Date.now();
    const persons = detections.filter(d => d.label.toLowerCase() === "person");
    const tracks = personTracksRef.current;

    // Mark all tracks as unmatched
    const matchedTrack = new Set<number>();
    const matchedDet   = new Set<number>();

    // Greedily match detections to tracks by highest IoU of predicted positions
    const scored: { ti: number; di: number; iou: number }[] = [];
    for (let ti = 0; ti < tracks.length; ti++) {
      const pred = predictBox(tracks[ti]);
      for (let di = 0; di < persons.length; di++) {
        const iou = boxIoU(pred, persons[di].box);
        if (iou > 0.3) scored.push({ ti, di, iou });
      }
    }
    scored.sort((a, b) => b.iou - a.iou);
    for (const { ti, di } of scored) {
      if (matchedTrack.has(ti) || matchedDet.has(di)) continue;
      matchedTrack.add(ti); matchedDet.add(di);
      const t = tracks[ti];
      const nb = persons[di].box;
      const cx1 = (t.box.xmin + t.box.xmax) / 2, cy1 = (t.box.ymin + t.box.ymax) / 2;
      const cx2 = (nb.xmin + nb.xmax) / 2,       cy2 = (nb.ymin + nb.ymax) / 2;
      // Smooth velocity with 0.4 weight on new delta
      t.vx = t.vx * 0.6 + (cx2 - cx1) * 0.4;
      t.vy = t.vy * 0.6 + (cy2 - cy1) * 0.4;
      t.box = nb;
      t.lastSeenMs = now;
      t.missedFrames = 0;

      // Update position history (keep last 10)
      t.posHistory.push({ cx: cx2, cy: cy2 });
      if (t.posHistory.length > 10) t.posHistory.shift();

      // ── Enhanced behavior analysis ──────────────────────────────────────────
      const speed = Math.hypot(t.vx, t.vy);
      const boxH  = nb.ymax - nb.ymin;

      // RUNNING: high velocity
      if (speed > 15) {
        t.behaviorFlags.add("running");
        t.behaviorFlags.delete("loitering");
      } else {
        t.behaviorFlags.delete("running");
      }

      // LOITERING: present > 90s, low spatial spread
      if ((now - t.firstSeenMs) > 90_000 && t.posHistory.length >= 5) {
        let maxDist = 0;
        for (let i = 0; i < t.posHistory.length; i++) {
          for (let j = i + 1; j < t.posHistory.length; j++) {
            const dx = t.posHistory[i].cx - t.posHistory[j].cx;
            const dy = t.posHistory[i].cy - t.posHistory[j].cy;
            const d = Math.hypot(dx, dy);
            if (d > maxDist) maxDist = d;
          }
        }
        if (maxDist < 60) t.behaviorFlags.add("loitering");
        else t.behaviorFlags.delete("loitering");
      }

      // PACING: person moves back and forth (direction reversal count)
      if (t.posHistory.length >= 8) {
        let reversals = 0;
        for (let i = 2; i < t.posHistory.length; i++) {
          const dx1 = t.posHistory[i-1].cx - t.posHistory[i-2].cx;
          const dx2 = t.posHistory[i].cx   - t.posHistory[i-1].cx;
          if (dx1 * dx2 < -5) reversals++; // sign change = reversal
        }
        if (reversals >= 3) t.behaviorFlags.add("pacing");
        else t.behaviorFlags.delete("pacing");
      }

      // APPROACHING CAMERA: person bounding box growing (person getting closer)
      if (t.posHistory.length >= 5) {
        const prevBox = persons[di]?.box; // rough proxy — use height growth instead
        if (boxH > 200 && t.vy < 0) { // large and moving up = approaching
          t.behaviorFlags.add("approaching");
        } else {
          t.behaviorFlags.delete("approaching");
        }
      }
    }

    // Unmatched detections → new tracks; reset face-recog cooldown so new arrivals get identified fast
    let newTrackCreated = false;
    for (let di = 0; di < persons.length; di++) {
      if (matchedDet.has(di)) continue;
      trackIdSeqRef.current += 1;
      tracks.push({
        id: trackIdSeqRef.current,
        name: null,
        box: persons[di].box,
        vx: 0, vy: 0,
        lastSeenMs: now,
        missedFrames: 0,
        firstSeenMs: now,
        posHistory: [],
        behaviorFlags: new Set(),
      });
      newTrackCreated = true;
    }
    if (newTrackCreated) lastFaceRecogRef.current = 0; // force face recog on next frame

    // Unmatched tracks → increment missed counter
    // Named tracks stay alive 30 s (person may briefly leave frame); unnamed tracks drop after ~2.4 s
    for (let ti = 0; ti < tracks.length; ti++) {
      if (!matchedTrack.has(ti)) tracks[ti].missedFrames += 1;
    }
    const NAMED_KEEP_MS = 30_000;
    personTracksRef.current = tracks.filter(t =>
      t.missedFrames < 4 || (t.name !== null && now - t.lastSeenMs < NAMED_KEEP_MS)
    );
  }, []);

  // Find the track that best overlaps a given detection box
  function trackForBox(box: Detection["box"]): PersonTrack | null {
    let best: PersonTrack | null = null;
    let bestIou = 0.25; // minimum threshold to avoid false associations
    for (const t of personTracksRef.current) {
      const iou = boxIoU(t.box, box);
      if (iou > bestIou) { best = t; bestIou = iou; }
    }
    return best;
  }

  const runFaceRecognition = useCallback(async (canvas: HTMLCanvasElement, detections: Detection[]) => {
    // Skip if every tracked person is already named — no need to run recognition.
    const persons = detections.filter(d => d.label.toLowerCase() === "person");
    const allNamed = persons.every(d => trackForBox(d.box)?.name != null);
    if (allNamed) return;

    const now = Date.now();
    if (now - lastFaceRecogRef.current < 4000) return; // 4s cooldown
    lastFaceRecogRef.current = now;
    try {
      // Unified ArcFace recognition (backend) — the SAME 512-d model as enrollment
      // + events, so the live overlay and event sub_labels always agree. Returns
      // boxes in this frame's pixel space (we send the canvas as the JPEG).
      const b64 = canvas.toDataURL("image/jpeg", 0.8).split(",")[1];
      const recs = await api.recognizeFrame(b64);
      const results: { name: string; fx: number; fy: number; fw: number; fh: number }[] = [];
      for (const rec of recs) {
        if (!rec.name || rec.name === "unknown") continue;
        const [bx1, by1, bx2, by2] = rec.bbox;
        const x = bx1, y = by1, width = bx2 - bx1, height = by2 - by1;
        const matchId: string | null = rec.person_id || null;
        if (matchId) api.markPersonSeen(matchId).catch(() => {});
        const match = { name: rec.name, personId: matchId, distance: rec.score };
        if (match) {
          results.push({ name: match.name, fx: x, fy: y, fw: width, fh: height });
          if (matchId) api.recordFaceSighting(match.name, camId, activeEventIdRef.current, 0.9).catch(() => {});

          // Assign name to the IoU-matching track — this persists identity across all future frames
          const faceCx = x + width / 2;
          const faceCy = y + height / 2;
          const personDet = detections.find(d => {
            const dcx = (d.box.xmin + d.box.xmax) / 2;
            const dcy = (d.box.ymin + d.box.ymax) / 2;
            return Math.abs(dcx - faceCx) < 80 && Math.abs(dcy - faceCy) < 100;
          });
          if (personDet) {
            const track = trackForBox(personDet.box);
            if (track) track.name = match.name;
          }

          // Also update appearance cache for cross-session fallback
          const bucketKey = personBucketKey(faceCx, faceCy);
          const app = personDet ? sampleAppearance(canvas, personDet.box) : null;
          verifiedIdentitiesRef.current.set(bucketKey, {
            name: match.name,
            hue: app?.hue ?? 0,
            sat: app?.sat ?? 0,
            lum: app?.lum ?? 0,
            lastSeen: Date.now(),
          });
        }
      }
      faceResultsRef.current = results;

      // Named persons = those with a named track OR in the appearance cache
      const namedSet = new Set<string>();
      for (const t of personTracksRef.current) if (t.name) namedSet.add(t.name);
      for (const [, v] of verifiedIdentitiesRef.current) namedSet.add(v.name);
      setRecognizedPersons([...namedSet]);
    } catch { /* face models not ready */ }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);


  // Restore saved source — clear any nokhwa-era entries (had deviceIndex, not deviceId).
  // The DB is the AUTHORITY: a slot the user REMOVED must never auto-start from a
  // leftover localStorage entry (that stale restore is what re-opened the webcam —
  // light back on — for a camera that no longer existed). The backend refuses such
  // starts too (`camera_start_allowed`); this just avoids the pointless round-trip
  // and cleans up the stale key.
  useEffect(() => {
    const key = `cam_source_${camId}`;
    let cancelled = false;
    (async () => {
      let raw: any;
      try {
        const saved = localStorage.getItem(key);
        if (!saved) return;
        raw = JSON.parse(saved);
      } catch { localStorage.removeItem(key); return; }

      // Slot removed/disabled in the DB ⇒ drop the stale entry, start nothing.
      try {
        const cfgs = await api.getCameraConfigs();
        const cfg = cfgs.find(c => c.cam_id === camId);
        if (cfg && !cfg.enabled) { localStorage.removeItem(key); return; }
      } catch { /* config unreadable — fall through; backend still guards */ }
      if (cancelled) return;

      // MIGRATION: the browser getUserMedia camera (kind "usb") is gone — USB cameras
      // are now captured SERVER-SIDE (cross-platform ffmpeg). Convert any persisted
      // "usb" source to "native" so it runs through the server path + /stream display.
      if (raw?.kind === "usb") {
        raw = { kind: "native", nativeIndex: 0, label: raw.label || "USB Camera", deviceId: raw.deviceId };
        try { localStorage.setItem(key, JSON.stringify(raw)); } catch { /* quota */ }
      }
      const src: CameraSource = raw;
      setSource(src);
      setTimeout(() => { if (!cancelled) startCamera(src); }, 150);
    })();
    return () => { cancelled = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [camId]);

  const refreshBrowserCameras = useCallback(async () => {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const cams = devices
        .filter(d => d.kind === "videoinput")
        .map((d, i) => ({ deviceId: d.deviceId, label: d.label || `Camera ${i + 1}` }));
      setBrowserCameras(cams);
      // Report to Rust so Guardian agent knows what local cameras are available
      const infos: BrowserCameraInfo[] = cams.map(c => ({ device_id: c.deviceId, label: c.label }));
      api.reportBrowserCameras(infos).catch(() => {});
    } catch { setBrowserCameras([]); }
  }, []);

  // Enumerate on mount (so Guardian knows cameras immediately) + when setup tab opens
  useEffect(() => { refreshBrowserCameras(); }, [refreshBrowserCameras]);
  useEffect(() => {
    if (setupTab === "local") refreshBrowserCameras();
  }, [setupTab, refreshBrowserCameras]);


  // ── JavaScript motion pre-screener ────────────────────────────────────────────
  // Sub-1ms pixel diff on a 80×45 thumbnail — instant visual feedback before
  // Rust IPC returns. Mature NVRs' philosophy: fast motion screen first, detailed
  // object detection second. This gives immediate UI response.
  const jsPreScreen = useCallback((sourceCanvas: HTMLCanvasElement): boolean => {
    const W = 80; const H = 45;
    if (!tinyCanvasRef.current) tinyCanvasRef.current = new OffscreenCanvas(W, H);
    const tc  = tinyCanvasRef.current;
    const ctx = tc.getContext("2d") as OffscreenCanvasRenderingContext2D | null;
    if (!ctx) return false;
    ctx.drawImage(sourceCanvas, 0, 0, W, H);
    const px = ctx.getImageData(0, 0, W, H).data;
    // Convert to grayscale
    const gray = new Uint8Array(W * H);
    for (let i = 0; i < W * H; i++) gray[i] = (px[i*4]*77 + px[i*4+1]*150 + px[i*4+2]*29) >> 8;
    const prev = prevGrayRef.current;
    prevGrayRef.current = gray;
    if (!prev) return false;
    let changed = 0;
    for (let i = 0; i < gray.length; i++) { if (Math.abs(gray[i] - prev[i]) > 20) changed++; }
    return changed / gray.length > 0.004; // 0.4% pixels changed
  }, []);

  // (The browser getUserMedia capture loop is gone — USB cameras are captured
  // server-side; the only in-WebView ingestion left is the MJPEG loop below.)

  const stopCaptureLoop = useCallback(() => {
    if (captureRafRef.current) {
      cancelAnimationFrame(captureRafRef.current);
      captureRafRef.current = 0;
    }
    if (captureIntervalRef.current) {
      clearInterval(captureIntervalRef.current);
      captureIntervalRef.current = null;
    }
  }, []);

  // ── MJPEG capture loop ────────────────────────────────────────────────────────
  // Draws the MJPEG <img> to captureCanvasRef at CAPTURE_INTERVAL_MS and feeds
  // frames through the same process_frame + stream_frame pipeline as USB cameras.
  // This gives MJPEG cameras full motion detection, NVR recording, and AI analysis.
  const startMjpegCaptureLoop = useCallback(() => {
    stopCaptureLoop();
    prevGrayRef.current = null;
    let lastSentMs = 0;
    const SEND_INTERVAL = 50;

    const tick = async () => {
      captureRafRef.current = requestAnimationFrame(tick);
      // Hidden window → skip work, keep the loop armed (see the video loop note).
      if (document.hidden) return;
      const img    = mjpegImgRef.current;
      const canvas = captureCanvasRef.current;
      if (!img || !canvas || !isActiveRef.current) return;
      // Only capture if the image has loaded and has valid dimensions
      // MJPEG streams never "complete" — don't check img.complete.
      // Use clientWidth/clientHeight fallback if naturalWidth isn't set yet.
      const w = img.naturalWidth  || img.clientWidth  || 0;
      const h = img.naturalHeight || img.clientHeight || 0;
      if (!w || !h) return;

      canvas.width  = w;
      canvas.height = h;
      const ctx = canvas.getContext("2d", { willReadFrequently: true });
      if (!ctx) return;
      try {
        ctx.drawImage(img, 0, 0, w, h);
      } catch (drawErr) {
        // drawImage fails if img hasn't loaded yet — skip this tick
        return;
      }
      let b64: string;
      try {
        b64 = canvas.toDataURL("image/jpeg", 0.55).split(",")[1];
      } catch (toDataErr) {
        // toDataURL throws SecurityError if canvas is tainted (cross-origin without CORS).
        // This happens when crossOrigin="anonymous" is missing from the <img> or
        // the proxy doesn't send Access-Control-Allow-Origin.
        console.error("[MJPEG capture] Canvas tainted — ensure crossOrigin=anonymous on img:", toDataErr);
        return;
      }
      if (!b64) return;

      // JS pre-screen at full RAF rate for instant motion feedback
      const hasMotionNow = jsPreScreen(canvas);
      if (hasMotionNow !== jsMotionRef.current) {
        jsMotionRef.current = hasMotionNow;
        if (hasMotionNow) syncOverlaySize();
      }

      const now = Date.now();
      fpsFramesRef.current++;
      if (now - fpsLastRef.current >= 1000) {
        setFps(fpsFramesRef.current); fpsFramesRef.current = 0; fpsLastRef.current = now;
      }

      // Rate-limit Rust calls to 20fps
      if (now - lastSentMs < SEND_INTERVAL) return;
      lastSentMs = now;

      if (frameInFlightRef.current) return;
      frameInFlightRef.current = true;
      try {
        const result = await api.processFrame(b64, camId, now);
        if (result.event_id) activeEventIdRef.current = result.event_id;
        else if (!result.recording) activeEventIdRef.current = null;

        // (v14: motion clips are recorded server-side from the NVR — no browser recorder here.)

        if (result.motion_detected && now > motionCoolRef.current) {
          motionCoolRef.current = now + MOTION_COOLDOWN_MS;
          syncOverlaySize();
        }

        api.streamFrame(b64, camId).catch(() => {});
      } catch { /* backend busy, skip frame */ }
      finally { frameInFlightRef.current = false; }
    }; // end tick

    captureRafRef.current = requestAnimationFrame(tick);
  }, [camId, jsPreScreen, setFps, stopCaptureLoop]);

  // ── Anonymized preview ────────────────────────────────────────────────────
  //
  // The server now returns an already-depth-mapped frame for an anonymized
  // camera (`get_camera_snapshot` enforces it, fail-closed), so the browser just
  // displays what it is given.
  //

  // ── Camera start / stop ──────────────────────────────────────────────────────
  const startCamera = useCallback(async (src?: CameraSource) => {
    const chosen = src ?? source;
    if (!chosen) return;
    setError(null);

    if (chosen.kind === "mjpeg") {
      const authUser = (chosen as any).authUser as string | undefined;
      const authPass = (chosen as any).authPass as string | undefined;
      // Resolve the actual MJPEG stream URL if only a base URL was saved
      const resolvedUrl = await resolveMjpegUrl((chosen as any).url ?? "", authUser, authPass);
      if (resolvedUrl !== (chosen as any).url) {
        const updated = { ...chosen, url: resolvedUrl, authUser, authPass };
        setSource(updated);
        try { localStorage.setItem(`cam_source_${camId}`, JSON.stringify(updated)); } catch { }
        selectSource(updated as any);
      }
      setIsActive(true);
      if (camId === 0) { setCameraActive(true); api.notifyCameraState(true).catch(() => {}); }
      // SERVER-SIDE capture (mature NVRs model): the relay's ffmpeg pulls the stream and
      // handles recording + detection in Rust, so the camera records 24/7 even when
      // you leave the Live view. The browser <img> below only DISPLAYS the feed.
      // (The old browser canvas capture stopped recording off-view — that's why the
      // IP camera had no NVR footage.) Fall back to browser capture if the relay can't
      // start (e.g. ffmpeg missing).
      try {
        await api.startRtspRelay(camId, resolvedUrl);
      } catch {
        startMjpegCaptureLoop();
      }
      return;
    }
    if (chosen.kind === "rtsp") {
      const url = (chosen as any).url ?? "";

      // http:// or https:// URLs are MJPEG, not RTSP — never need ffmpeg relay
      if (url.startsWith("http://") || url.startsWith("https://")) {
        const resolvedUrl = await resolveMjpegUrl(url);
        const authUser2 = (chosen as any).authUser as string | undefined;
        const authPass2 = (chosen as any).authPass as string | undefined;
        const resolvedUrl2 = await resolveMjpegUrl(url, authUser2, authPass2);
        const mjpegSrc = { kind: "mjpeg" as const, url: resolvedUrl2, label: (chosen as any).label ?? url, authUser: authUser2, authPass: authPass2 };
        // Re-route through the mjpeg branch via selectSource — it starts the
        // SERVER relay (detection + recording) and only falls back to the
        // browser capture loop if the relay can't start. The old code here
        // ALSO started the browser loop unconditionally, so both pipelines
        // pushed frames through process_frame for the same camera (double
        // motion/IPC work per frame).
        setSource(mjpegSrc);
        try { localStorage.setItem(`cam_source_${camId}`, JSON.stringify(mjpegSrc)); } catch { }
        selectSource(mjpegSrc);
        return;
      }

      // True RTSP — relay via ffmpeg
      try {
        await api.startRtspRelay(camId, url);
        setIsActive(true);
        if (camId === 0) { setCameraActive(true); api.notifyCameraState(true).catch(() => {}); }
      } catch (e: any) {
        setError(e?.message ?? `RTSP relay failed. Make sure ffmpeg is installed.\nFallback: Open in VLC: ${url}`);
      }
      return;
    }

    // NATIVE — server-side Rust (nokhwa) capture. Records 24/7 independent of the
    // UI (frames are grabbed in a Rust OS thread, fed straight to the NVR pipe +
    // inference), so it does NOT stop when you leave the Live view. The live picture
    // is the server's MJPEG /stream (no browser getUserMedia, no rAF capture loop).
    if (chosen.kind === "native") {
      // Stop any in-WebView loop (MJPEG) before switching this slot to server capture.
      stopCaptureLoop();
      // PREFER ffmpeg DirectShow (reliable Windows backend — real fps/HD + records
      // 24/7 server-side). The device name is the camera label ("Integrated Camera").
      // Fall back to the nokhwa native path only if dshow can't start.
      try {
        await api.startDshowCamera(camId, chosen.label);
        setIsActive(true);
        if (camId === 0) { setCameraActive(true); api.notifyCameraState(true).catch(() => {}); }
      } catch {
        try {
          await api.startNativeCamera(camId, chosen.nativeIndex);
          setIsActive(true);
          if (camId === 0) { setCameraActive(true); api.notifyCameraState(true).catch(() => {}); }
        } catch (e: any) {
          setError(e?.message ?? "Camera failed to open. It may be in use by another app.");
        }
      }
      return;
    }
  }, [source, camId, setCameraActive]);

  const stopCamera = useCallback(async () => {
    stopCaptureLoop();
    api.stopRtspRelay(camId).catch(() => {}); // stop RTSP relay if running
    api.stopNativeCamera(camId).catch(() => {}); // stop native (nokhwa) capture if running
    // NVR is Rust-side — no browser action needed
    setIsActive(false);
    if (camId === 0) {
      setCameraActive(false); setFps(0);
      api.notifyCameraState(false).catch(() => {});
    }
    clearOverlay();
  }, [camId, stopCaptureLoop, setCameraActive, setFps]);

  const selectSource = (src: CameraSource) => {
    setSource(src);
    try { localStorage.setItem(`cam_source_${camId}`, JSON.stringify(src)); } catch { }
    startCamera(src);
  };

  // Remote control + Guardian camera/settings events
  useEffect(() => {
    if (camId !== 0) return;
    const u1 = listen("remote:start_camera", () => { if (!isActive) startCamera(); });
    const u2 = listen("remote:stop_camera",  () => { if (isActive)  stopCamera(); });

    const u3 = listen<import("../../api").Settings>("guardian:settings-updated", (e) => {
      useStore.getState().setSettings(e.payload);
    });

    // NOTE: the Guardian agent has NO camera control. The previous
    // "guardian:camera-start"/"guardian:camera-stop" listeners were removed so
    // the monitoring agent (which ingests untrusted chat) can never start/stop a
    // camera. The "remote:*" events above are the user's own remote control.

    return () => { u1.then(f => f()); u2.then(f => f()); u3.then(f => f()); };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isActive, source]);

  // Audio-event detection is now SERVER-SIDE for every camera kind (USB via the OS
  // ffmpeg capture, IP via the RTSP relay → shared YAMNet path). The old in-browser
  // getUserMedia({audio}) tap was removed with the browser camera.

  useEffect(() => {
    const h = () => stopCamera();
    window.addEventListener("beforeunload", h);
    return () => window.removeEventListener("beforeunload", h);
  }, [stopCamera]);

  // ── Motion overlay — sync pixel dims to CSS display size, then CLEAR ─────────
  // The old behaviour painted a red full-frame border + red tint + corner
  // brackets whenever motion fired, which made the live feed look "alarmed".
  // The separate MOTION chip already signals motion, so we keep this as a plain
  // canvas reset — per-object detection boxes still draw on top via
  // drawDetectionBoxes(). No red frame on the live view.
  /** Size the overlay canvas to its CSS box. Assigning `width` also clears it,
   *  which is the whole reason this is called before a repaint — the explicit
   *  `clearRect` that used to follow was redundant. Kept as a function rather
   *  than inlined: three call sites. */
  const syncOverlaySize = () => {
    const ol = overlayRef.current;
    if (!ol) return;
    ol.width  = ol.offsetWidth  || ol.clientWidth  || 640;
    ol.height = ol.offsetHeight || ol.clientHeight || 360;
  };

  const clearOverlay = () => {
    const ol = overlayRef.current;
    if (ol) ol.getContext("2d")!.clearRect(0, 0, ol.width, ol.height);
  };

  const drawDetectionBoxes = (detections: Detection[], eventId?: string | null) => {
    const ol = overlayRef.current;
    if (!ol || detections.length === 0) return;
    ol.width  = ol.offsetWidth  || ol.clientWidth  || 640;
    ol.height = ol.offsetHeight || ol.clientHeight || 360;
    const ctx = ol.getContext("2d")!;

    // Draw motion overlay first, then detection boxes on top
    syncOverlaySize();

    const imgW = captureCanvasRef.current?.width  || ol.width;
    const imgH = captureCanvasRef.current?.height || ol.height;
    const scaleX = ol.width  / imgW;
    const scaleY = ol.height / imgH;

    // A pulsing radial glow, a pulsing ring and a crosshair used to be drawn on
    // the weighted centroid of all people here — four marks for one subject whose
    // bounding box is already stroked in the alert colour below and already
    // carries a text label. Removed; the box is the signal. `people` survives
    // because the peak-person test below still needs it.
    const people = detections.filter(d => d.label.toLowerCase() === "person");

    for (const det of detections) {
      const { xmin, ymin, xmax, ymax } = det.box;
      const x = xmin * scaleX;
      const y = ymin * scaleY;
      const w = (xmax - xmin) * scaleX;
      const h = (ymax - ymin) * scaleY;

      const isPerson = det.label.toLowerCase() === "person";
      const cx = (xmin + xmax) / 2;
      const cy = (ymin + ymax) / 2;

      // Identity lookup: IoU track (primary) → appearance cache → current-frame face result
      let recognizedName: string | null = null;
      if (isPerson) {
        recognizedName = trackForBox(det.box)?.name ?? null;
      }

      // Assign stable number to unknown persons
      const personNum = (isPerson && !recognizedName && eventId)
        ? getPersonNumber(cx * scaleX, cy * scaleY, eventId)
        : null;

      // Four fixed, muted colours from the shared palette — see lib/palette.ts.
      // The object case used to be `hsl(stringToHue(label), 90%, 60%)`: a hash of
      // the class name, so any YOLO label could land on any hue including ones
      // that collided with "recognised" or "alert", and 90% saturation vibrated
      // over real footage. Resolved through the canvas since ctx.strokeStyle
      // cannot take a CSS variable.
      const isPeakPerson = isPerson && !recognizedName &&
        people.length > 0 && det === people.reduce((a, b) => a.score > b.score ? a : b);
      let color: string;
      if (recognizedName) color = cssVar(BOX_COLOR.recognised);
      else if (isPeakPerson) color = cssVar(BOX_COLOR.alert);
      else if (isPerson) color = cssVar(BOX_COLOR.person);
      else color = cssVar(BOX_COLOR.object);

      // Box — outline only
      ctx.strokeStyle = color;
      ctx.lineWidth = recognizedName ? 2.5 : 2;
      ctx.strokeRect(x, y, w, h);

      // Label
      const labelText = recognizedName
        ? `✓ ${recognizedName} ${(det.score * 100).toFixed(0)}%`
        : personNum !== null
          ? `#${personNum} ${(det.score * 100).toFixed(0)}%`
          : `${det.label} ${(det.score * 100).toFixed(0)}%`;

      ctx.font = "bold 11px monospace";
      const tw = ctx.measureText(labelText).width;
      const ph = 17;
      const px = 6;
      ctx.fillStyle = recognizedName ? withAlpha(cssVar(BOX_COLOR.recognised), 0.92) : "rgba(0,0,0,0.75)";
      ctx.fillRect(x, y - ph - 2, tw + px * 2, ph + 2);
      ctx.fillStyle = recognizedName ? "#000" : color;
      ctx.fillText(labelText, x + px, y - 4);
    }
  };

  // Assign a stable number to an unknown person by bucketing their center position
  function getPersonNumber(cx: number, cy: number, eventId: string): number {
    if (lastEventIdForNumRef.current !== eventId) {
      personNumberMapRef.current.clear();
      personNumberSeqRef.current = 0;
      lastEventIdForNumRef.current = eventId;
      // New event = clear transient caches (tracks re-associate, appearance cache resets)
      personTracksRef.current = [];
      verifiedIdentitiesRef.current.clear();
    }
    // Round to nearest 60px bucket for stability across frames
    const key = `${Math.round(cx / 60)}_${Math.round(cy / 60)}`;
    if (!personNumberMapRef.current.has(key)) {
      personNumberSeqRef.current += 1;
      personNumberMapRef.current.set(key, personNumberSeqRef.current);
    }
    return personNumberMapRef.current.get(key)!;
  }


  // `stringToHue` lived here and hashed a label into a hue for detection boxes.
  // Removed with that behaviour — see the BOX_COLOR block in drawDetectionBoxes.

  const scanNetwork = async () => {
    setScanning(true); setDiscovered([]);
    try {
      const cams = await api.discoverCameras();
      setDiscovered(cams);
      if (cams.length === 0) showToast("No cameras found on the network", "info");
    } catch (e: any) {
      showToast(e.message ?? "Scan failed", "error");
    } finally { setScanning(false); }
  };

  const addManualUrl = () => {
    const url = manualUrl.trim();
    if (!url) return;
    selectSource({ kind: url.startsWith("rtsp://") ? "rtsp" : "mjpeg", url, label: url });
    setManualUrl("");
  };

  // ── Render ───────────────────────────────────────────────────────────────────
  const showNativeFeed = isActive && source?.kind === "native";
  const showRTSPFeed   = isActive && source?.kind === "rtsp";
  const showMJPEGFeed  = isActive && source?.kind === "mjpeg";

  return (
    <div className={`${styles.root} ${cornered ? styles.cornered : ""}`}>
      {/* A live tile is fully chromeless — no status badges or controls, just the
       * video (the grid tile overlay shows the camera name). The old "Start" button
       * that appeared here for a not-yet-active camera was removed: cameras
       * AUTO-START from their saved source on mount, so the button only ever
       * flashed during the startup gap and read as UI noise. */}

      {/* Feed area */}
      <div className={styles.feedWrap} ref={fullRef}>

        {cornered && isActive && (
          <div style={{ position: "absolute", right: 12, bottom: 12, zIndex: 6, display: "flex", gap: 6 }}>
            {/* SERVER anonymization — the real privacy switch. */}
            <button
              type="button"
              disabled={anonBusy}
              title={anonOn
                ? "Depth Anonymization ON (server): recordings, streams, snapshots and alerts contain ONLY the depth map. Local AI still detects on raw frames in memory. Click to restore raw video."
                : "Anonymize this camera at the source: everything stored or sent becomes a colorized depth map — identities never persist. Requires the Depth model (~95 MB, downloads on first use)."}
              onClick={toggleAnonymize}
              style={{
                display: "inline-flex", alignItems: "center", gap: 6,
                padding: "6px 12px", borderRadius: 999, cursor: anonBusy ? "wait" : "pointer",
                fontSize: 11, fontWeight: 700,
                background: anonOn ? "var(--status-idle)" : "rgba(0,0,0,0.55)",
                color: anonOn ? "#fff" : "rgba(255,255,255,0.9)",
                border: "1px solid " + (anonOn ? "var(--status-idle)" : "rgba(255,255,255,0.18)"),
                backdropFilter: "blur(8px)",
              }}>
              {anonOn ? <Eye size={12} /> : <EyeOff size={12} />}
              {anonBusy ? "…" : anonOn ? "Anonymized" : "Anonymize"}
            </button>
            {/* CLIENT view overlay — cosmetic, hidden when the feed IS depth. */}
          </div>
        )}

        {/* NATIVE — server-side Rust (nokhwa) capture, shown via the local MJPEG
            /stream endpoint. This camera records 24/7 server-side regardless of
            which page is open (unlike the browser USB path). */}
        <div className={styles.videoContainer} style={{ display: showNativeFeed ? "flex" : "none" }}>
          {showNativeFeed && streamInfo && (
            <HlsFeed
              port={streamInfo.port}
              token={streamInfo.auth_token}
              camId={camId}
              className={styles.video}
            />
          )}
        </div>

        {/* RTSP (IP camera) — WebRTC sub-second live via go2rtc, falling back to
            the relay's copy-HLS then MJPEG. (Previously RTSP cams had NO display
            element here at all — only native/MJPEG containers existed.) */}
        <div className={styles.videoContainer} style={{ display: showRTSPFeed ? "flex" : "none" }}>
          {showRTSPFeed && streamInfo && (
            <WebRtcFeed
              port={streamInfo.port}
              token={streamInfo.auth_token}
              camId={camId}
              className={styles.video}
            />
          )}
        </div>

        {/* External MJPEG camera — proxied through localhost to bypass CSP */}
        <div className={styles.videoContainer} style={{ display: showMJPEGFeed ? "flex" : "none" }}>
          {source?.kind === "mjpeg" && (
            <img
              aria-hidden
              crossOrigin="anonymous"
              src={buildProxyUrl((source as any).url, streamInfo, (source as any).authUser, (source as any).authPass)}
              className={styles.videoBlurBg}
              alt=""
            />
          )}
          {source?.kind === "mjpeg" && (
            <img
              ref={mjpegImgRef}
              crossOrigin="anonymous"
              src={buildProxyUrl((source as any).url, streamInfo, (source as any).authUser, (source as any).authPass)}
              className={styles.video}
              alt="MJPEG stream"
              onError={() => setError(
                `Cannot connect to camera at ${(source as any).url}\n\nMake sure the camera is on and reachable at that address.`
              )}
            />
          )}
          {/* Live detection overlay — sized by syncOverlaySize, painted by drawDetectionBoxes. */}
          <canvas
            ref={overlayRef}
            style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" }}
          />
        </div>

        {/* Error */}
        {error && (
          <div className={styles.errorMsg}>
            <CameraOff size={36} style={{ color: "var(--text-muted)", marginBottom: 12 }} />
            <p style={{ color: "var(--accent-red)", marginBottom: 8, whiteSpace: "pre-wrap", maxWidth: 320, textAlign: "center" }}>{error}</p>
            <div style={{ display: "flex", gap: 8 }}>
              <button className={styles.retryBtn} onClick={() => { setError(null); setSource(null); }}>
                <RefreshCw size={13} /> Try Again
              </button>
              {onRemove && (
                <button
                  className={styles.retryBtn}
                  onClick={() => { setError(null); localStorage.removeItem(`cam_source_${camId}`); onRemove(); }}
                  style={{ color: "var(--accent-red)", borderColor: "color-mix(in srgb, var(--status-alert) 30%, transparent)" }}
                >
                  Remove Slot
                </button>
              )}
            </div>
          </div>
        )}

        {/* Setup panel removed — cameras are added via the Live tab modal */}
      </div>

      {/* Hidden canvas used only for extracting frames to send to Rust */}
      <canvas ref={captureCanvasRef} style={{ display: "none" }} />
    </div>
  );
}

