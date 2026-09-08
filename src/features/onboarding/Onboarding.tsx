/**
 * First-run onboarding wizard.
 *
 * Shown once, on a genuinely fresh install (no cameras configured and the
 * `sc-onboarded` flag unset — see the gate in App.tsx). Walks a new user through
 * the four things that otherwise require a scavenger hunt through the tabs:
 *
 *   1. Welcome        — what this is + the detected hardware.
 *   2. Performance    — if an NVIDIA GPU is present, OFFER the TensorRT pack
 *                       (~1.85 GB, one-time) with one tap + a live progress bar,
 *                       then activate. Non-NVIDIA machines are told they're
 *                       already running on DirectML/CPU (no download). This is the
 *                       fix for "TensorRT only installs when you dig up a button".
 *   3. Add a camera   — reuses the real AddCameraModal (USB / RTSP / ONVIF).
 *   4. All set        — asks the user to CHOOSE a detector. Nothing is bundled:
 *                       shipping a model means distributing it, and the YOLO
 *                       weights are AGPL-3.0. The pick is the user's, made with
 *                       the licence in view (see THIRD-PARTY-NOTICES.md).
 *
 * Completion is remembered in localStorage (`sc-onboarded`), so a fresh device
 * shows it again (correct) and an existing install never does.
 */

import { useEffect, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import {
  Shield, Cpu, Zap, Camera, Check, ChevronRight, ChevronLeft, Download,
  Sparkles, Rocket, MonitorPlay,
} from "lucide-react";
import { api } from "../../api";
import type { TrtxStatus, SystemMetrics, CameraConfig, Settings } from "../../types";
import { AddCameraModal } from "../live/LivePanel";

const ONBOARDED_KEY = "sc-onboarded";

/** Mark onboarding done so it never shows again on this install. */
export function markOnboarded() { try { localStorage.setItem(ONBOARDED_KEY, "1"); } catch { /* ignore */ } }
export function isOnboarded(): boolean { try { return !!localStorage.getItem(ONBOARDED_KEY); } catch { return true; } }

type AccelProgress = { percent: number; downloaded: number; total: number; label: string; step: number; steps: number };

const EP_LABEL: Record<string, string> = {
  nvrtx: "TensorRT-RTX", tensorrt: "TensorRT", cuda: "CUDA",
};

export function Onboarding({ onDone }: { onDone: () => void }) {
  const [step, setStep] = useState(0);
  const total = 4;

  // Shared state fetched once up front.
  const [trtx, setTrtx] = useState<TrtxStatus | null>(null);
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null);
  const [settings, setSettings] = useState<Settings | null>(null);
  const [cams, setCams] = useState<CameraConfig[]>([]);

  const refreshCams = () => api.getCameraConfigs().then(setCams).catch(() => {});

  useEffect(() => {
    api.trtxStatus().then(setTrtx).catch(() => setTrtx(null));
    api.getSystemMetrics().then(setMetrics).catch(() => {});
    api.getSettings().then(setSettings).catch(() => {});
    refreshCams();
  }, []);

  const finish = () => { markOnboarded(); onDone(); };

  return (
    <div style={{
      position: "fixed", inset: 0, zIndex: 5000,
      background: "radial-gradient(1200px 600px at 50% -10%, color-mix(in srgb, var(--accent) 10%, transparent), transparent 60%), var(--bg, var(--bg-base))",
      display: "flex", alignItems: "center", justifyContent: "center", padding: 24,
    }}>
      <div className="glass" style={{
        width: "min(600px, 96vw)", maxHeight: "92vh", overflow: "auto",
        borderRadius: 20, padding: "28px 30px 22px",
        display: "flex", flexDirection: "column", gap: 18,
        boxShadow: "0 30px 90px rgba(0,0,0,0.55)",
      }}>
        {/* Progress dots */}
        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
          {Array.from({ length: total }).map((_, i) => (
            <div key={i} style={{
              height: 4, flex: 1, borderRadius: 999,
              background: i <= step ? "var(--accent)" : "var(--border)",
              transition: "background 240ms ease",
            }} />
          ))}
        </div>

        {step === 0 && <WelcomeStep metrics={metrics} onNext={() => setStep(1)} onSkip={finish} />}
        {step === 1 && (
          <PerformanceStep
            trtx={trtx} metrics={metrics} settings={settings}
            onStatus={setTrtx}
            onBack={() => setStep(0)} onNext={() => setStep(2)}
          />
        )}
        {step === 2 && (
          <CameraStep
            cams={cams} onRefresh={refreshCams}
            onBack={() => setStep(1)} onNext={() => setStep(3)}
          />
        )}
        {step === 3 && (
          <FinishStep cams={cams} onBack={() => setStep(2)} onDone={finish} />
        )}
      </div>
    </div>
  );
}

// ── Step 0: Welcome ──────────────────────────────────────────────────────────

function WelcomeStep({ metrics, onNext, onSkip }: {
  metrics: SystemMetrics | null; onNext: () => void; onSkip: () => void;
}) {
  const gpu = metrics?.gpus?.find(g => g.is_discrete) ?? metrics?.gpus?.[0];
  return (
    <>
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 12, textAlign: "center", padding: "10px 0 4px" }}>
        <div style={{
          width: 60, height: 60, borderRadius: 16, display: "flex", alignItems: "center", justifyContent: "center",
          background: "var(--accent-glow)", color: "var(--accent)", border: "1px solid var(--accent)",
        }}>
          <Shield size={30} />
        </div>
        <div style={{ fontWeight: 800, fontSize: 22, letterSpacing: -0.02 }}>Welcome to Anivar</div>
        <div style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.6, maxWidth: 440 }}>
          A private, local-first AI security NVR — continuous recording, on-device
          person / face / plate detection, and an agentic guardian. Nothing leaves
          this machine unless you choose to share it. Let's get you set up in a
          couple of steps.
        </div>
      </div>

      {metrics && (
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", justifyContent: "center" }}>
          <HwPill icon={<Cpu size={12} />} label={`${metrics.per_core?.length ?? "?"}-core CPU`} />
          {gpu && <HwPill icon={<MonitorPlay size={12} />} label={gpu.name} />}
          {metrics.accelerator && <HwPill icon={<Zap size={12} />} label={metrics.accelerator} accent />}
        </div>
      )}

      <div style={{ display: "flex", gap: 10, marginTop: 4 }}>
        <button onClick={onSkip} style={ghostBtn}>Skip setup</button>
        <div style={{ flex: 1 }} />
        <button onClick={onNext} className="btn-primary" style={{ padding: "10px 20px" }}>
          Get started <ChevronRight size={15} />
        </button>
      </div>
    </>
  );
}

// ── Step 1: Performance / acceleration ───────────────────────────────────────

function PerformanceStep({ trtx, metrics, settings, onStatus, onBack, onNext }: {
  trtx: TrtxStatus | null; metrics: SystemMetrics | null; settings: Settings | null;
  onStatus: (s: TrtxStatus) => void; onBack: () => void; onNext: () => void;
}) {
  const [installing, setInstalling] = useState(false);
  const [prog, setProg] = useState<AccelProgress | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [done, setDone] = useState(false);

  // Live progress from the backend pack download.
  useEffect(() => {
    const un = listen<AccelProgress>("accel:progress", ({ payload }) => setProg(payload));
    return () => { un.then(f => f()); };
  }, []);

  const nvidia = !!trtx?.supported;                 // Windows + NVIDIA adapter
  const alreadyOn = !!trtx?.active;                 // an NVIDIA EP already passed its canary
  const epLabel = trtx ? (EP_LABEL[trtx.active_ep] ?? trtx.active_ep) : "";
  // Non-NVIDIA: what are we running on? DirectML on any DX12 GPU, else CPU.
  const baseline = settings?.inference_device === "cpu"
    ? "your CPU"
    : (metrics?.accelerator && metrics.accelerator !== "CPU" ? metrics.accelerator : "your GPU (DirectML)");

  const enable = async () => {
    setInstalling(true); setErr(null); setProg({ percent: 0, downloaded: 0, total: 0, label: "Starting…", step: 0, steps: 5 });
    try {
      const st = await api.installTrtxPack();
      onStatus(st);
      setDone(true);
    } catch (e: any) {
      // Backend returns Err when the pack downloaded but no NVIDIA EP passed its
      // canary on this GPU — detection still works (DirectML), so this is non-fatal.
      setErr(String(e));
    } finally {
      setInstalling(false);
    }
  };

  return (
    <>
      <StepHeader
        icon={<Rocket size={22} />}
        title="Performance"
        subtitle="Make on-device detection as fast as your hardware allows."
      />

      {alreadyOn || done ? (
        <InfoCard tone="good" icon={<Check size={18} />}
          title={`Max performance is on${epLabel ? ` — ${epLabel}` : ""}`}
          body="NVIDIA acceleration is active. Detection runs on the fastest path available on this GPU." />
      ) : nvidia ? (
        installing ? (
          <div className="glass" style={{ padding: 16, borderRadius: 14, display: "flex", flexDirection: "column", gap: 10 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8, fontWeight: 700, fontSize: 13 }}>
              <Download size={16} style={{ color: "var(--accent)" }} />
              Downloading NVIDIA Performance Pack…
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "var(--text-secondary)" }}>
              <span>{prog?.label ?? "Starting…"}{prog?.steps ? ` · step ${prog.step}/${prog.steps}` : ""}</span>
              <span>{prog?.total ? `${prog.percent}% · ${fmtGB(prog.downloaded)} / ${fmtGB(prog.total)}` : "…"}</span>
            </div>
            <div style={{ height: 6, borderRadius: 999, background: "var(--bg-hover)", overflow: "hidden" }}>
              <div style={{
                height: "100%", width: `${prog?.percent ?? 0}%`, background: "var(--accent)",
                transition: "width 240ms cubic-bezier(0.16,1,0.3,1)",
              }} />
            </div>
            <div style={{ fontSize: 10.5, color: "var(--text-muted)" }}>
              This is a one-time ~1.85 GB download and can take several minutes. You can keep using the app; it activates automatically when done.
            </div>
          </div>
        ) : (
          <div className="glass" style={{ padding: 16, borderRadius: 14, display: "flex", flexDirection: "column", gap: 12 }}>
            <div style={{ display: "flex", gap: 12 }}>
              <div style={{
                width: 40, height: 40, borderRadius: 10, flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "center",
                background: "rgba(118,185,0,0.12)", color: "#76b900", border: "1px solid rgba(118,185,0,0.4)",
              }}>
                <Zap size={20} />
              </div>
              <div>
                <div style={{ fontWeight: 700, fontSize: 13 }}>NVIDIA GPU detected</div>
                <div style={{ fontSize: 11.5, color: "var(--text-secondary)", lineHeight: 1.55, marginTop: 3 }}>
                  Install the TensorRT Performance Pack for substantially faster detection
                  (NVIDIA's optimized engine, ~50% higher throughput than the default).
                  One-time <b>~1.85 GB</b> download; runs entirely on-device afterward.
                </div>
              </div>
            </div>
            {err && (
              <div style={{ fontSize: 11, color: "var(--accent-amber)", lineHeight: 1.5 }}>
                {err} — detection still works on {baseline}; you can retry later in Arsenal.
              </div>
            )}
            <button onClick={enable} className="btn-primary" style={{ padding: "10px 16px", justifyContent: "center" }}>
              <Download size={14} /> Enable max performance (~1.85 GB)
            </button>
            <button onClick={onNext} style={{ ...ghostBtn, alignSelf: "center", padding: "4px 8px" }}>
              Not now — I'll decide later
            </button>
          </div>
        )
      ) : (
        <InfoCard tone="neutral" icon={<Sparkles size={18} />}
          title={`Running on ${baseline}`}
          body="Detection is ready with no extra download. Acceleration packs are NVIDIA-only; your hardware already runs models on the best available path." />
      )}

      <StepFooter onBack={onBack} onNext={onNext}
        nextLabel={installing ? "Continue (runs in background)" : "Continue"} />
    </>
  );
}

// ── Step 2: Add a camera ─────────────────────────────────────────────────────

function CameraStep({ cams, onRefresh, onBack, onNext }: {
  cams: CameraConfig[]; onRefresh: () => void; onBack: () => void; onNext: () => void;
}) {
  const [showModal, setShowModal] = useState(false);
  // get_camera_configs ALWAYS returns 16 rows (real cameras + placeholder slots).
  // A real camera is `enabled` — the same signal LivePanel uses. Everything else
  // is an empty slot and must NOT be shown as a configured camera.
  const configured = cams.filter(c => c.enabled);
  const nextSlot = Array.from({ length: 16 }, (_, i) => i).find(i => !configured.some(c => c.cam_id === i)) ?? 0;

  return (
    <>
      <StepHeader
        icon={<Camera size={22} />}
        title="Add your first camera"
        subtitle="A USB webcam, or any RTSP / ONVIF IP camera on your network."
      />

      {configured.length === 0 ? (
        <div className="glass" style={{ padding: 20, borderRadius: 14, textAlign: "center", display: "flex", flexDirection: "column", gap: 12, alignItems: "center" }}>
          <div style={{ width: 46, height: 46, borderRadius: 12, display: "flex", alignItems: "center", justifyContent: "center", background: "var(--accent-glow)", color: "var(--accent)" }}>
            <Camera size={22} />
          </div>
          <div style={{ fontSize: 12.5, color: "var(--text-secondary)", maxWidth: 400, lineHeight: 1.55 }}>
            No cameras yet. Add one now to start recording and detecting — Anivar
            auto-discovers ONVIF cameras on your LAN, or you can pick a USB device.
          </div>
          <button onClick={() => setShowModal(true)} className="btn-primary" style={{ padding: "10px 18px" }}>
            <Camera size={14} /> Add a camera
          </button>
        </div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {configured.map(c => (
            <div key={c.cam_id} className="glass" style={{ padding: "10px 14px", borderRadius: 12, display: "flex", alignItems: "center", gap: 10 }}>
              <span style={{ width: 8, height: 8, borderRadius: 999, background: "var(--accent)" }} />
              <span style={{ fontWeight: 700, fontSize: 13 }}>{c.name || `Camera ${c.cam_id + 1}`}</span>
              <span style={{ fontSize: 10.5, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: 0.04 }}>{c.source_type}</span>
              <div style={{ flex: 1 }} />
              <Check size={15} style={{ color: "var(--accent)" }} />
            </div>
          ))}
          <button onClick={() => setShowModal(true)} style={{ ...ghostBtn, alignSelf: "flex-start" }}>
            <Camera size={13} /> Add another
          </button>
        </div>
      )}

      <StepFooter onBack={onBack} onNext={onNext}
        nextLabel={configured.length === 0 ? "Skip for now" : "Continue"} />

      {showModal && (
        <AddCameraModal
          nextSlotId={nextSlot}
          onClose={() => setShowModal(false)}
          onAdded={() => { setShowModal(false); onRefresh(); }}
        />
      )}
    </>
  );
}

// ── Step 3: Finish ───────────────────────────────────────────────────────────

function FinishStep({ cams, onBack, onDone }: { cams: CameraConfig[]; onBack: () => void; onDone: () => void }) {
  const configuredCount = cams.filter(c => c.enabled).length;
  return (
    <>
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 12, textAlign: "center", padding: "6px 0" }}>
        <div style={{ width: 56, height: 56, borderRadius: 16, display: "flex", alignItems: "center", justifyContent: "center", background: "var(--accent-glow)", color: "var(--accent)", border: "1px solid var(--accent)" }}>
          <Check size={30} />
        </div>
        <div style={{ fontWeight: 800, fontSize: 20 }}>You're all set</div>
        <div style={{ fontSize: 12.5, color: "var(--text-secondary)", lineHeight: 1.6, maxWidth: 440 }}>
          Recording and motion detection work now.{" "}
          {configuredCount > 0 ? `${configuredCount} camera${configuredCount === 1 ? "" : "s"} recording.` : "Add a camera anytime from the Live tab."}
          {" "}To recognise <em>what</em> moved — people, vehicles, animals — pick an
          object-detection model in Arsenal. None is preinstalled, so the choice is yours.
        </div>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        <NextThing icon={<Download size={15} />} title="Choose a detector in Arsenal"
          body="Object detection needs a model, and none ships with the app — models carry their own licences, so the pick is yours. Each one shows its licence next to it." />
        <NextThing icon={<Sparkles size={15} />} title="More models in Arsenal"
          body="Face recognition, license plates, semantic search, audio events — install any of them with one click." />
        <NextThing icon={<Shield size={15} />} title="Remote access & alerts in Settings"
          body="Set up a Tailscale share link or connect Telegram to watch live and get alerts on your phone." />
      </div>

      <div style={{ display: "flex", gap: 10, marginTop: 4 }}>
        <button onClick={onBack} style={ghostBtn}><ChevronLeft size={15} /> Back</button>
        <div style={{ flex: 1 }} />
        <button onClick={onDone} className="btn-primary" style={{ padding: "10px 22px" }}>
          <Rocket size={15} /> Start using Anivar
        </button>
      </div>
    </>
  );
}

// ── Shared bits ──────────────────────────────────────────────────────────────

function StepHeader({ icon, title, subtitle }: { icon: React.ReactNode; title: string; subtitle: string }) {
  return (
    <div style={{ display: "flex", gap: 12, alignItems: "flex-start", paddingTop: 2 }}>
      <div style={{ width: 42, height: 42, borderRadius: 11, flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "center", background: "var(--accent-glow)", color: "var(--accent)" }}>
        {icon}
      </div>
      <div>
        <div style={{ fontWeight: 800, fontSize: 17, letterSpacing: -0.01 }}>{title}</div>
        <div style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.5, marginTop: 3 }}>{subtitle}</div>
      </div>
    </div>
  );
}

function StepFooter({ onBack, onNext, nextLabel }: { onBack: () => void; onNext: () => void; nextLabel: string }) {
  return (
    <div style={{ display: "flex", gap: 10, marginTop: 2 }}>
      <button onClick={onBack} style={ghostBtn}><ChevronLeft size={15} /> Back</button>
      <div style={{ flex: 1 }} />
      <button onClick={onNext} className="btn-primary" style={{ padding: "10px 18px" }}>
        {nextLabel} <ChevronRight size={15} />
      </button>
    </div>
  );
}

function InfoCard({ tone, icon, title, body }: { tone: "good" | "neutral"; icon: React.ReactNode; title: string; body: string }) {
  const color = tone === "good" ? "var(--accent)" : "var(--text-secondary)";
  return (
    <div className="glass" style={{ padding: 16, borderRadius: 14, display: "flex", gap: 12, alignItems: "flex-start" }}>
      <div style={{ color, marginTop: 1 }}>{icon}</div>
      <div>
        <div style={{ fontWeight: 700, fontSize: 13, color: tone === "good" ? "var(--accent)" : "var(--text-primary)" }}>{title}</div>
        <div style={{ fontSize: 11.5, color: "var(--text-secondary)", lineHeight: 1.55, marginTop: 3 }}>{body}</div>
      </div>
    </div>
  );
}

function NextThing({ icon, title, body }: { icon: React.ReactNode; title: string; body: string }) {
  return (
    <div style={{ display: "flex", gap: 10, alignItems: "flex-start", padding: "10px 12px", borderRadius: 10, border: "1px solid var(--border)", background: "rgb(var(--ink) / 0.02)" }}>
      <div style={{ color: "var(--accent)", marginTop: 1 }}>{icon}</div>
      <div>
        <div style={{ fontWeight: 700, fontSize: 12.5 }}>{title}</div>
        <div style={{ fontSize: 11, color: "var(--text-muted)", lineHeight: 1.5, marginTop: 2 }}>{body}</div>
      </div>
    </div>
  );
}

function HwPill({ icon, label, accent }: { icon: React.ReactNode; label: string; accent?: boolean }) {
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 6, padding: "5px 11px", borderRadius: 999,
      fontSize: 11, fontWeight: 700,
      background: accent ? "var(--accent-glow)" : "rgb(var(--ink) / 0.04)",
      color: accent ? "var(--accent)" : "var(--text-secondary)",
      border: `1px solid ${accent ? "var(--accent)" : "var(--border)"}`,
    }}>
      {icon}{label}
    </span>
  );
}

const ghostBtn: React.CSSProperties = {
  display: "inline-flex", alignItems: "center", gap: 5, padding: "9px 14px", borderRadius: 10,
  border: "1px solid var(--border-strong)", background: "transparent", color: "var(--text-secondary)",
  fontSize: 12, fontWeight: 700, cursor: "pointer",
};

function fmtGB(n: number): string {
  if (!n) return "0 MB";
  const gb = n / 1_073_741_824;
  return gb >= 1 ? `${gb.toFixed(2)} GB` : `${(n / 1_048_576).toFixed(0)} MB`;
}
