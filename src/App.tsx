import { useEffect, useState, useRef } from "react";
import { listen } from "@tauri-apps/api/event";
import { useStore } from "./store";
import { useShallow } from "zustand/react/shallow";
import { api } from "./api";
import type { Settings as AppSettings, SystemMetrics, InferenceStatus } from "./types";
import { LivePanel }    from "./features/live/LivePanel";
import { ReviewFeed }   from "./features/review/ReviewFeed";
import { PersonsPanel } from "./features/persons/PersonsPanel";
import { AgentPanel }   from "./features/agent/AgentPanel";
import { Arsenal }      from "./features/agent/Arsenal";
import { SettingsPanel } from "./features/settings/SettingsPanel";
import { Onboarding, isOnboarded, markOnboarded } from "./features/onboarding/Onboarding";
import { Toast }        from "./components/ui/Toast";
import { LoginGate }    from "./features/auth/LoginGate";
import { TitleBar }      from "./components/ui/TitleBar";
import styles           from "./App.module.css";
import { LayoutGrid, Film, Users, Bot, Brain, Settings, Shield, Cctv,
  User, Lock, LogOut, ShieldCheck, ChevronRight, Car, AudioLines, Activity, Cpu } from "lucide-react";

// ── Theme persistence ──────────────────────────────────────────────────────
// Dark only, twice over now. A light theme has been built and removed from this
// codebase two separate times; the second attempt was structurally sound (one
// `--ink` token flipped, no duplicated palette) and was still not wanted. Do not
// build a third without asking.
//
// What survives from that work is `--ink` in index.css — the white that lifts a
// surface — and the `--hl` highlight derived from it, which is what selection is
// painted with now.
/** Surface material applied to all panels/chrome:
 *  frosted = classic translucent frosted glass (blur + tint)
 *  solid   = opaque, no blur (max performance / minimal style)
 *  ("liquid" retired — SVG refraction was the app's biggest GPU cost) */
export type AppSurface = "frosted" | "solid";

/**
 * Apply the saved surface to <html>.
 *
 * The accent is no longer a choice: there is one, it lives in index.css, and
 * every accent-dependent token derives from it. The old five-accent picker
 * re-declared seven tokens per theme while six other sites hardcoded the green
 * regardless, so "red" rendered a red-and-green app. Removing the axis removes
 * the whole bug class.
 *
 * Stale `accent-*` classes are still stripped so an existing install sheds the
 * old class on first run.
 */
export function applyTheme(surface: AppSurface = loadSurface()) {
  const html = document.documentElement;
  html.classList.remove("light"); // clear the class anyone still has from an old build
  html.classList.remove("accent-blue", "accent-purple", "accent-red", "accent-amber");
  html.classList.remove("surface-liquid", "surface-frosted", "surface-solid");
  html.classList.add(`surface-${surface}`);
  // Both deleted light themes' keys. Cleared so an install that had either one
  // saved does not carry a dead preference forever.
  localStorage.removeItem("sc-theme");
  localStorage.removeItem("sc-theme-mode");
  document.documentElement.classList.remove("theme-light");
  localStorage.removeItem("sc-accent");
  localStorage.setItem("sc-surface", surface);
}

export function loadSurface(): AppSurface {
  const s = localStorage.getItem("sc-surface") ?? "frosted";
  // "liquid" was retired (its SVG-refraction filter was the single biggest GPU
  // cost in the app) — anyone saved on it migrates to frosted, the closest look.
  return (s === "liquid" ? "frosted" : s) as AppSurface;
}

const TABS = [
  { id: "live",     label: "Live",     Icon: LayoutGrid },
  { id: "review",   label: "Review",   Icon: Film       },
  { id: "people",   label: "People",   Icon: Users      },
  { id: "guardian", label: "Guardian", Icon: Bot        },
  { id: "arsenal",  label: "Arsenal",  Icon: Brain      },
  { id: "settings", label: "Settings", Icon: Settings   },
] as const;

export default function App() {
  // Shallow selector: App re-renders ONLY when tab/liveView/toast change — not on the
  // per-tick camera churn (motionScore/fps/latestFrame) that lives in the same store.
  const { tab, setTab, liveView, setLiveView, setFocusedCam, toast, clearToast, setSettings, setStreamInfo, setEvents } = useStore(useShallow(s => ({
    tab: s.tab, setTab: s.setTab, liveView: s.liveView, setLiveView: s.setLiveView,
    setFocusedCam: s.setFocusedCam, toast: s.toast, clearToast: s.clearToast,
    setSettings: s.setSettings, setStreamInfo: s.setStreamInfo, setEvents: s.setEvents,
  })));

  // The Live drill-down (grid → camera → player) drives two side-nav behaviours:
  // the Live icon becomes a CCTV/hidden-camera icon once a camera is focused, and
  // the player level highlights Review instead of Live.
  const navActive = (id: typeof TABS[number]["id"]) =>
    id === "review" ? (tab === "review" || (tab === "live" && liveView === "player"))
    : id === "live" ? (tab === "live" && liveView !== "player")
    : tab === id;
  // Any top-level nav click resets the Live flow to the grid (Back unwinds the
  // drill-down; the nav is a top-level jump).
  const goTab = (id: typeof TABS[number]["id"]) => {
    setTab(id); setLiveView("grid"); setFocusedCam(null);
  };

  // ACTIVE-ONLY MOUNTING: render ONLY the current tab's panel; unmount the rest.
  // Keeping every visited panel alive made the app progressively slower — each panel's
  // polling loops (LivePanel ×4, People ×2) and its loaded data (Review's event feed,
  // 150 face thumbnails, agent history) stacked up and never went away. Unmounting an
  // inactive panel runs its cleanup (clearInterval + frees its DOM/data), so exactly
  // ONE panel ever costs anything → constant, snappy load however much you navigate.
  // Cost: revisiting a tab re-fetches its data (brief) instead of it silently growing.

  // Camera-health watchdog toasts — the backend emits when an enabled camera
  // stops producing frames (>90 s) and when it recovers. Telegram gets the
  // alert through the normal chokepoint; this is the in-app visibility.
  useEffect(() => {
    const un = listen<{ cam_id: number; online: boolean }>("camera:health", (e) => {
      const { cam_id, online } = e.payload;
      useStore.getState().showToast(
        online ? `Camera ${cam_id + 1} is back online`
               : `Camera ${cam_id + 1} is OFFLINE — no frames arriving`,
        online ? "success" : "error",
      );
    });
    return () => { un.then(f => f()); };
  }, []);

  // ── Desktop login gate ──────────────────────────────────────────────────
  // `checked` gates the first render; `locked` shows the LoginGate. On boot we
  // ask the backend whether login is required + already unlocked, and try to
  // resume from a remembered-device token before falling back to the gate.
  const [authChecked, setAuthChecked] = useState(false);
  const [locked, setLocked] = useState(false);
  const [rememberEnabled, setRememberEnabled] = useState(false);
  // First-run onboarding gate. Shows ONLY on a genuinely fresh install: the
  // `sc-onboarded` flag unset AND no cameras configured. An existing install
  // (has cameras) auto-marks itself onboarded so the wizard never interrupts it.
  const [showOnboarding, setShowOnboarding] = useState(false);
  useEffect(() => {
    (async () => {
      try {
        const st = await api.authStatus();
        setRememberEnabled(st.remember_enabled);
        if (!st.login_required || st.unlocked) { setLocked(false); setAuthChecked(true); return; }
        const token = localStorage.getItem("sc-remember-token");
        if (token && await api.authResume(token).catch(() => false)) { setLocked(false); setAuthChecked(true); return; }
        setLocked(true); setAuthChecked(true);
      } catch { setLocked(false); setAuthChecked(true); } // fail open if auth backend errors
    })();
  }, []);

  // Decide whether to show first-run onboarding once auth resolves.
  // NOTE: get_camera_configs always returns 16 rows (real cameras + placeholder
  // slots). A REAL, configured camera is the one that's `enabled` — the same
  // signal LivePanel uses (`cfgs.filter(c => c.enabled)`). So gate on that, not
  // on array length (which is always 16).
  useEffect(() => {
    if (!authChecked || locked || isOnboarded()) return;
    (async () => {
      try {
        const cams = await api.getCameraConfigs();
        const configured = cams.filter(c => c.enabled);
        if (configured.length > 0) { markOnboarded(); return; } // existing install — never interrupt
        setShowOnboarding(true);                                // fresh device — guide setup
      } catch { /* can't tell → don't block the app */ }
    })();
  }, [authChecked, locked]);

  // Apply saved appearance on mount. The ramp is already on <html> from import
  // time; this only re-asserts it and subscribes to OS changes for `system`.
  useEffect(() => { applyTheme(); }, []);

  useEffect(() => {
    api.getSettings().then(setSettings).catch(console.error);
    api.getStreamInfo().then(setStreamInfo).catch(console.error);
    api.getMotionEvents(500).then(setEvents).catch(console.error);
  }, []);

  // Live settings sync: the backend emits `settings:updated` whenever settings
  // change from ANY source (the app's Save, or a Telegram /menu toggle), so the
  // Settings panel stays in lockstep with the Telegram bot without a restart.
  useEffect(() => {
    const u = listen<AppSettings>("settings:updated", ({ payload }) => setSettings(payload));
    return () => { u.then(f => f()); };
  }, [setSettings]);

  // Blank holding screen until the auth check resolves (avoids a flash of the app).
  if (!authChecked) return <div style={{ position: "fixed", inset: 0, background: "var(--bg, var(--bg-base))" }} />;
  if (locked) return <LoginGate rememberEnabled={rememberEnabled} onUnlocked={() => setLocked(false)} />;
  if (showOnboarding) return <Onboarding onDone={() => setShowOnboarding(false)} />;

  return (
    <div className={styles.root}>
      {/* Liquid-glass SVG refraction filter (defined once, used via .lg-* classes) */}
      {/* ── Custom window titlebar (frameless window — min/max/close live here) ── */}
      <TitleBar />
      {/* ── Body: sidebar + main ── */}
      <div className={styles.body}>
      {/* ── Sidebar — icon-only, no text ── */}
      <aside className={styles.sidebar}>
        {/* Icon nav — no labels, tooltip on hover */}
        <nav className={styles.nav}>
          {TABS.map(({ id, label, Icon }) => {
            const active = navActive(id);
            // Live icon swaps to a CCTV/hidden-camera icon once a camera is focused.
            const inCamera = id === "live" && liveView !== "grid";
            const NavIcon  = inCamera ? Cctv : Icon;
            const navLabel = inCamera ? "Camera" : label;
            return (
              <button
                key={id}
                className={`${styles.navBtn} ${active ? styles.navBtnActive : ""}`}
                onClick={() => goTab(id)}
                title={navLabel}
                aria-label={navLabel}
              >
                <NavIcon size={19} strokeWidth={active ? 2.2 : 1.8} />
              </button>
            );
          })}
        </nav>

        <div className={styles.sidebarFooter} style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 10 }}>
          <TelemetryPopover />
          <UserMenu
            onOpenSecurity={() => { goTab("settings"); setTimeout(() => document.getElementById("s-security")?.scrollIntoView({ behavior: "smooth", block: "start" }), 120); }}
            onLocked={() => setLocked(true)}
          />
        </div>
      </aside>

      {/* ── Main ── */}
      <main className={styles.main}>
        <div style={{ display: tab === "live"     ? "flex" : "none", flex: 1, flexDirection: "column", overflow: "hidden", minHeight: 0 }}>
          {tab === "live"     && <LivePanel />}
        </div>
        <div style={{ display: tab === "review"   ? "flex" : "none", flex: 1, flexDirection: "column", overflow: "hidden", minHeight: 0 }}>
          {tab === "review"   && <ReviewFeed />}
        </div>
        <div style={{ display: tab === "people"   ? "flex" : "none", flex: 1, flexDirection: "column", overflow: "hidden", minHeight: 0 }}>
          {tab === "people"   && <PersonsPanel />}
        </div>
        <div style={{ display: tab === "guardian" ? "flex" : "none", flex: 1, flexDirection: "column", overflow: "hidden", minHeight: 0 }}>
          {tab === "guardian" && <AgentPanel />}
        </div>
        <div style={{ display: tab === "arsenal"  ? "flex" : "none", flex: 1, flexDirection: "column", overflow: "hidden", minHeight: 0 }}>
          {tab === "arsenal"  && <Arsenal />}
        </div>
        <div style={{ display: tab === "settings" ? "flex" : "none", flex: 1, flexDirection: "column", overflow: "hidden", minHeight: 0 }}>
          {tab === "settings" && <SettingsPanel />}
        </div>
      </main>
      </div>

      {toast && <Toast msg={toast.msg} type={toast.type} onClose={clearToast} />}
    </div>
  );
}

// ── System telemetry popover (sidebar bottom, above the account button) ──────
// Edge-AI-style compact card: CPU / RAM / GPU bars + the active accelerator
// + per-model inference latency. Polls only while open.
function TelemetryPopover() {
  const [open, setOpen] = useState(false);
  const [m, setM] = useState<SystemMetrics | null>(null);
  // Detector state. `infer_stats` only lists models that ARE running, so a
  // detector that failed to load showed up nowhere at all — the app knew
  // (CameraView held this in state) but never rendered it, which made a broken
  // detector completely silent. This is the one place a user can now see it.
  const [det, setDet] = useState<InferenceStatus | null>(null);
  const wrapRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    let alive = true;
    const load = () => {
      api.getSystemMetrics().then(r => { if (alive) setM(r); }).catch(() => {});
      api.getInferenceStatus().then(r => { if (alive) setDet(r); }).catch(() => {});
    };
    load();
    const iv = setInterval(load, 2000);
    const onDoc = (e: MouseEvent) => { if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) setOpen(false); };
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
    document.addEventListener("mousedown", onDoc);
    document.addEventListener("keydown", onKey);
    return () => {
      alive = false; clearInterval(iv);
      document.removeEventListener("mousedown", onDoc);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const Metric = ({ label, value, pct }: { label: string; value: string; pct: number }) => (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <div style={{ display: "flex", justifyContent: "space-between", gap: 8, fontSize: 11 }}>
        <span style={{ color: "var(--text-muted)", fontWeight: 700, overflow: "hidden",
          textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{label}</span>
        <span style={{ fontWeight: 700, fontVariantNumeric: "tabular-nums", flexShrink: 0 }}>{value}</span>
      </div>
      <div style={{ height: 4, borderRadius: 999, background: "rgb(var(--ink) / 0.08)", overflow: "hidden" }}>
        <div style={{ width: `${Math.min(100, Math.max(0, pct))}%`, height: "100%", borderRadius: 999,
          background: pct > 85 ? "var(--accent-red)" : pct > 60 ? "var(--status-warn)" : "var(--accent)",
          transition: "width 300ms ease" }} />
      </div>
    </div>
  );

  return (
    <div ref={wrapRef} style={{ position: "relative", display: "flex", justifyContent: "center" }}>
      <button title="System telemetry" aria-label="System telemetry" onClick={() => setOpen(o => !o)}
        className={`${styles.iconBtn} ${open ? styles.iconBtnActive : ""}`}>
        <Cpu size={17} />
      </button>

      {open && (
        <div style={{
          position: "absolute", left: "calc(100% + 12px)", bottom: 0, width: 264, padding: 14, zIndex: 4000,
          display: "flex", flexDirection: "column", gap: 11,
          background: "var(--bg-elevated)", border: "1px solid var(--border-strong)",
          boxShadow: "0 18px 50px rgba(0,0,0,0.5)", borderRadius: 14,
        }}>
          {(() => {
            // The backend accelerator string is an engineering line like
            // "DirectML → NVIDIA GeForce RTX 5060 Laptop GPU (adapter 1)" —
            // the card shows just the ENGINE; the full line lives in the tooltip.
            const engine = (m?.accelerator ?? "").split("→")[0].replace(/\(.*\)/, "").trim();
            const gpuName = (n: string) => n
              .replace(/\((R|TM)\)/gi, "")
              .replace(/^(NVIDIA GeForce|NVIDIA|AMD Radeon|AMD|Intel)\s*/i, "")
              .replace(/\s+Graphics$/i, "")
              .trim() || n;
            const modelName = (s: string) =>
              s.replace(/_/g, " ").replace(/^./, c => c.toUpperCase());
            return (
              <>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <Activity size={13} style={{ color: "var(--accent)" }} />
                  <span style={{ fontWeight: 800, fontSize: 12.5 }}>System</span>
                  {engine && (
                    <span title={m!.accelerator} style={{ marginLeft: "auto", fontSize: 9.5, fontWeight: 800, padding: "2px 8px",
                      borderRadius: 999, background: "var(--accent-glow)", color: "var(--accent)",
                      textTransform: "uppercase", letterSpacing: 0.04, cursor: "default" }}>{engine}</span>
                  )}
                </div>
                {m ? (
                  <>
                    <Metric label="CPU" value={`${m.cpu_total.toFixed(0)}%`} pct={m.cpu_total} />
                    <Metric label="RAM"
                      value={`${(m.mem_used_mb / 1024).toFixed(1)} / ${(m.mem_total_mb / 1024).toFixed(0)} GB`}
                      pct={(m.mem_used_mb / Math.max(1, m.mem_total_mb)) * 100} />
                    {/* EVERY adapter — the iGPU runs the UI compositor, the dGPU
                        runs the models; both loads are worth a glance. */}
                    {m.gpus.map(g => (
                      <Metric key={g.name}
                        label={`${gpuName(g.name)}${g.is_discrete ? "" : " · iGPU"}`}
                        value={`${g.util.toFixed(0)}%${g.mem_mb ? ` · ${(g.mem_mb / 1024).toFixed(1)} GB` : ""}`}
                        pct={g.util} />
                    ))}
                    {(m.infer_stats.length > 0 || (det && det.state !== "ready")) && (
                      <div style={{ borderTop: "1px solid var(--border)", paddingTop: 9,
                        display: "flex", flexDirection: "column", gap: 5 }}>
                        <span style={{ fontSize: 9.5, fontWeight: 800, color: "var(--text-muted)",
                          textTransform: "uppercase", letterSpacing: 0.05 }}>Inference</span>
                        {det && det.state !== "ready" && (
                          <div style={{ display: "flex", justifyContent: "space-between", gap: 8, fontSize: 11 }}>
                            <span style={{ color: "var(--text-secondary)" }}>Detector</span>
                            <span style={{ fontWeight: 700, flexShrink: 0,
                              color: det.state === "error" ? "var(--status-alert)"
                                   : det.state === "loading" ? "var(--status-warn)"
                                   : "var(--text-muted)" }}>
                              {det.state === "error" ? "Failed to load"
                               : det.state === "loading" ? "Loading…"
                               : "Not installed"}
                            </span>
                          </div>
                        )}
                        {m.infer_stats.slice(0, 5).map(r => (
                          <div key={r.model} style={{ display: "flex", justifyContent: "space-between", gap: 8, fontSize: 11 }}>
                            <span style={{ color: "var(--text-secondary)", overflow: "hidden",
                              textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{modelName(r.model)}</span>
                            <span style={{ fontWeight: 700, fontVariantNumeric: "tabular-nums", flexShrink: 0 }}>
                              {r.avg_ms.toFixed(0)} ms
                            </span>
                          </div>
                        ))}
                      </div>
                    )}
                  </>
                ) : (
                  <span style={{ fontSize: 11.5, color: "var(--text-muted)" }}>Reading metrics…</span>
                )}
              </>
            );
          })()}
        </div>
      )}
    </div>
  );
}

// ── Account / user menu (sidebar bottom) ────────────────────────────────────
// Avatar button anchored at the bottom of the rail; click opens a popover with the
// account/session actions (lock, log out). Security & login CONFIG lives in
// Settings → Security; this menu only exposes quick session actions.
function UserMenu({ onOpenSecurity, onLocked }: {
  onOpenSecurity: () => void;
  onLocked: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [st, setSt] = useState<{ login_required: boolean; has_password: boolean; twofa_enabled: boolean } | null>(null);
  const wrapRef = useRef<HTMLDivElement>(null);

  // Refresh status whenever the menu opens.
  useEffect(() => {
    if (!open) return;
    api.authStatus().then(s => setSt({ login_required: s.login_required, has_password: s.has_password, twofa_enabled: s.twofa_enabled })).catch(() => setSt(null));
  }, [open]);

  // Close on outside click / Escape.
  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => { if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) setOpen(false); };
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
    document.addEventListener("mousedown", onDoc);
    document.addEventListener("keydown", onKey);
    return () => { document.removeEventListener("mousedown", onDoc); document.removeEventListener("keydown", onKey); };
  }, [open]);

  const protect = !!st?.login_required;
  const close = () => setOpen(false);

  const lockNow = async () => { try { await api.lockApp(); } catch {} close(); onLocked(); };
  const logout = async () => {
    const t = localStorage.getItem("sc-remember-token") ?? undefined;
    try { await api.logout(t); } catch {}
    localStorage.removeItem("sc-remember-token");
    close(); onLocked();
  };

  const row: React.CSSProperties = {
    display: "flex", alignItems: "center", gap: 10, width: "100%", padding: "9px 11px",
    borderRadius: 9, fontSize: 12.5, fontWeight: 600, color: "var(--text-primary)",
    background: "transparent", border: "none", cursor: "pointer", textAlign: "left",
  };
  const Row = ({ icon, label, onClick, danger, accent }: { icon: React.ReactNode; label: string; onClick: () => void; danger?: boolean; accent?: boolean }) => (
    <button style={{ ...row, color: danger ? "var(--accent-red)" : accent ? "var(--accent)" : "var(--text-primary)" }}
      onMouseEnter={e => (e.currentTarget.style.background = "rgb(var(--ink) / 0.06)")}
      onMouseLeave={e => (e.currentTarget.style.background = "transparent")}
      onClick={onClick}>
      <span style={{ display: "inline-flex", width: 16, justifyContent: "center", opacity: 0.9 }}>{icon}</span>
      <span style={{ flex: 1 }}>{label}</span>
      <ChevronRight size={13} style={{ opacity: 0.35 }} />
    </button>
  );

  return (
    <div ref={wrapRef} style={{ position: "relative", display: "flex", justifyContent: "center" }}>
      <button title="Account" aria-label="Account" onClick={() => setOpen(o => !o)}
        className={`${styles.iconBtn} ${open ? styles.iconBtnActive : ""}`}>
        <User size={18} />
        {protect && (
          <span title="Login protected" style={{
            position: "absolute", right: -2, bottom: -2, width: 15, height: 15, borderRadius: "50%",
            background: "var(--accent-fill)", color: "var(--on-accent)", display: "flex", alignItems: "center", justifyContent: "center",
            border: "2px solid var(--bg, var(--bg-base))",
          }}><Lock size={8} strokeWidth={3} /></span>
        )}
      </button>

      {open && (
        <div style={{
          position: "absolute", left: "calc(100% + 12px)", bottom: 0, width: 232, padding: 8, zIndex: 4000,
          // Near-opaque elevated surface so the menu is easy to read (not glassy).
          background: "var(--bg-elevated)", border: "1px solid var(--border-strong)",
          boxShadow: "0 18px 50px rgba(0,0,0,0.5)", borderRadius: 14,
        }}>
          {/* Header */}
          <div style={{ display: "flex", alignItems: "center", gap: 11, padding: "8px 9px 12px" }}>
            <div style={{ width: 38, height: 38, borderRadius: "50%", flexShrink: 0, display: "flex",
              alignItems: "center", justifyContent: "center", background: "var(--accent-glow)", color: "var(--accent)" }}>
              <User size={19} />
            </div>
            <div style={{ minWidth: 0 }}>
              <div style={{ fontWeight: 800, fontSize: 13 }}>This device</div>
              <div style={{ fontSize: 10.5, color: protect ? "var(--accent)" : "var(--text-muted)", display: "flex", alignItems: "center", gap: 4 }}>
                {protect
                  ? <><ShieldCheck size={11} /> Login protected{st?.twofa_enabled ? " · 2FA" : ""}</>
                  : <>Login not set up</>}
              </div>
            </div>
          </div>
          <div style={{ height: 1, background: "var(--border)", margin: "0 4px 6px" }} />

          {/* Session actions only — security/login CONFIG lives in Settings → Security. */}
          {protect && <Row icon={<Lock size={15} />} label="Lock now" onClick={lockNow} />}
          {protect && <Row icon={<LogOut size={15} />} label="Log out" danger onClick={logout} />}
          {!protect && (
            <Row icon={<Shield size={15} />} label="Set up login" accent onClick={() => { close(); onOpenSecurity(); }} />
          )}
        </div>
      )}
    </div>
  );
}

