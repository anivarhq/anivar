import { create } from "zustand";
import type { Settings, MotionEvent, StreamInfo, AgentAlert, AgentStatus } from "../api";

type Tab = "live" | "review" | "people" | "guardian" | "arsenal" | "settings";
// Sub-view within the Live section: the all-cameras grid, a single camera (live
// feed + its settings), or the full playback player. Lifted here so the side-nav
// can swap the Live icon → CCTV and highlight Review while in the player.
type LiveView = "grid" | "camera" | "player";

interface AppStore {
  tab: Tab;
  setTab: (t: Tab) => void;

  // Live drill-down: grid → camera → player
  liveView: LiveView;
  setLiveView: (v: LiveView) => void;
  focusedCam: number | null;
  setFocusedCam: (v: number | null) => void;

  // Camera state
  cameraActive: boolean;
  setCameraActive: (v: boolean) => void;
  // motionDetected / motionScore lived here and were written on every frame but
  // READ by nothing — the MOTION chip and red tint they drove were removed in an
  // earlier declutter and the plumbing outlived them.
  fps: number;
  setFps: (v: number) => void;

  // Stream info
  streamInfo: StreamInfo | null;
  setStreamInfo: (v: StreamInfo | null) => void;

  // Tunnel

  // Settings
  settings: Settings | null;
  setSettings: (v: Settings) => void;

  // Events
  events: MotionEvent[];
  setEvents: (v: MotionEvent[]) => void;
  addEvent: (e: MotionEvent) => void;

  // Latest camera frame + detections for Guardian snapshot
  latestFrame: string | null;
  setLatestFrame: (v: string) => void;
  latestDetections: { label: string; score: number }[];
  setLatestDetections: (v: { label: string; score: number }[]) => void;

  // Guardian agent
  agentAlerts: AgentAlert[];
  setAgentAlerts: (v: AgentAlert[]) => void;
  agentStatus: AgentStatus | null;
  setAgentStatus: (v: AgentStatus) => void;

  // Model downloads, held here rather than in Arsenal.
  //
  // App.tsx mounts panels ACTIVE-ONLY, so switching tabs unmounts Arsenal. The
  // Rust download keeps streaming regardless, but the component-local progress
  // went with it: coming back showed "Install" again, and clicking it hit
  // `DownloadGuard::acquire`'s "that download is already running" as a red
  // failure toast. On the 1.3 GB Vision tier that was the default experience.
  skillProgress: Record<string, { pct: number; downloaded?: number; total?: number | null }>;
  setSkillProgress: (id: string, p: { pct: number; downloaded?: number; total?: number | null }) => void;
  skillDownloading: Record<string, boolean>;
  setSkillDownloading: (id: string, v: boolean) => void;

  // Toast
  toast: { msg: string; type: "success" | "error" | "info" } | null;
  showToast: (msg: string, type?: "success" | "error" | "info") => void;
  clearToast: () => void;
}

/** Single pending toast-dismissal timer — see showToast. */
let toastTimer: ReturnType<typeof setTimeout> | null = null;

export const useStore = create<AppStore>((set, get) => ({
  tab: "live",
  setTab: (tab) => set({ tab }),

  liveView: "grid",
  setLiveView: (liveView) => set({ liveView }),
  focusedCam: null,
  setFocusedCam: (focusedCam) => set({ focusedCam }),

  skillProgress: {},
  setSkillProgress: (id, p) => set(s => ({ skillProgress: { ...s.skillProgress, [id]: p } })),
  skillDownloading: {},
  setSkillDownloading: (id, v) => set(s => ({ skillDownloading: { ...s.skillDownloading, [id]: v } })),

  cameraActive: false,
  setCameraActive: (cameraActive) => set({ cameraActive }),
  fps: 0,
  setFps: (fps) => set({ fps }),

  streamInfo: null,
  setStreamInfo: (streamInfo) => set({ streamInfo }),


  settings: null,
  setSettings: (settings) => set({ settings }),

  events: [],
  setEvents: (events) => set({ events }),
  addEvent: (e) => set({ events: [e, ...get().events].slice(0, 200) }),

  latestFrame: null,
  setLatestFrame: (latestFrame) => set({ latestFrame }),
  latestDetections: [],
  setLatestDetections: (latestDetections) => set({ latestDetections }),

  agentAlerts: [],
  setAgentAlerts: (agentAlerts) => set({ agentAlerts }),
  agentStatus: null,
  setAgentStatus: (agentStatus) => set({ agentStatus }),

  toast: null,
  showToast: (msg, type = "info") => {
    // Cancel the previous dismissal before arming a new one. Without this the
    // FIRST toast's timer would fire while the SECOND was on screen and clear
    // it early: two toasts 2 s apart meant the second was visible for 1.5 s
    // instead of 3.5 s. There is only one toast slot, so a later message
    // replaces an earlier one — it should still get its full dwell time.
    if (toastTimer !== null) clearTimeout(toastTimer);
    set({ toast: { msg, type } });
    toastTimer = setTimeout(() => { toastTimer = null; set({ toast: null }); }, 3500);
  },
  clearToast: () => {
    if (toastTimer !== null) { clearTimeout(toastTimer); toastTimer = null; }
    set({ toast: null });
  },
}));
