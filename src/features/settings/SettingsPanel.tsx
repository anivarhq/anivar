import { useEffect, useLayoutEffect, useState, useCallback, useRef, useMemo } from "react";
import { createPortal } from "react-dom";
import { applyTheme, loadSurface, type AppSurface } from "../../App";
import { useStore } from "../../store";
import { useShallow } from "zustand/react/shallow";
import { api, Settings, StorageInfo, FaceRecommendation } from "../../api";
import { SENSITIVITY_PRESETS, matchPreset, SensitivityPresetPicker, type SensPreset } from "./sensitivityPresets";
import styles from "./SettingsPanel.module.css";
import {
  Save, HardDrive, RefreshCw, Trash2, Activity, Video, Radio, ExternalLink, Send,
  Zap, Film, User, Bell, ChevronDown,
  Bot, SlidersHorizontal, Shield, Lock, KeyRound,
  Info, Check, X,
} from "lucide-react";
import type { AuthStatus } from "../../api";

/// The repo in-app updates are checked against.
///
/// This used to be a text field the user had to fill in, and "Check for Updates"
/// was disabled until they did — asking them to configure the vendor's own
/// identity, with no way to know the answer. It is the same for every install.
const APP_REPO = "anivarhq/anivar";

/* ── Remote-access first-time setup guide ────────────────────────────────────── */
// Reactive 4-step walkthrough for Tailscale Funnel, opened from the info icon on
// the Remote-access card (and auto-opened once while setup is incomplete). Each
// step reflects the LIVE status — done / current / pending — so the user always
// sees exactly which action is theirs. Rendered via createPortal: the settings
// glass window's backdrop-filter turns position:fixed into a positioning trap
// (documented in CameraSettingsModal), so the modal must mount on <body>.
function TailscaleSetupGuide({ status, busy, onEnable, onRefresh, onClose }: {
  status: import("../../types").TailscaleStatus | null;
  busy: boolean;
  onEnable: () => void;
  onRefresh: () => void;
  onClose: () => void;
}) {
  const openExternal = async (url: string) => {
    try { const { open } = await import("@tauri-apps/plugin-shell"); await open(url); }
    catch { window.open(url, "_blank"); }
  };
  const installed = !!status?.installed;
  const loggedIn  = !!status?.logged_in;
  const active    = !!status?.funnel_active;
  const enableUrl = status?.enable_url ?? "";
  const allDone   = installed && loggedIn && active;

  type StepState = "done" | "current" | "pending";
  const steps: { title: string; desc: string; state: StepState; action?: React.ReactNode }[] = [
    {
      title: "Install Tailscale (free)",
      desc: "A small app that gives this PC a secure public address. Viewers of your links never need it — they just open a browser.",
      state: installed ? "done" : "current",
      action: !installed ? (
        <button onClick={() => openExternal("https://tailscale.com/download")} style={btnStyle(true)}>
          Download Tailscale →
        </button>
      ) : undefined,
    },
    {
      title: "Sign in",
      desc: "Open the Tailscale app and log in with any Google/Microsoft/GitHub account (~1 minute).",
      state: loggedIn ? "done" : installed ? "current" : "pending",
      action: installed && !loggedIn ? (
        <button onClick={onRefresh} style={btnStyle(false)}>I signed in — re-check</button>
      ) : undefined,
    },
    {
      title: "Enable remote access",
      desc: "Publishes your camera links at a stable https://…ts.net address.",
      state: active ? "done" : (installed && loggedIn && !enableUrl) ? "current" : "pending",
      action: installed && loggedIn && !active && !enableUrl ? (
        <button onClick={onEnable} disabled={busy} style={btnStyle(true)}>
          {busy ? "Enabling…" : "Enable remote access"}
        </button>
      ) : undefined,
    },
    {
      title: "Approve sharing (one-time)",
      desc: "Tailscale asks once per account before anything is shared over the internet. Approve it, then re-check.",
      state: active ? "done" : enableUrl ? "current" : "pending",
      action: !active && enableUrl ? (
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          <button onClick={() => openExternal(enableUrl)} style={btnStyle(true)}>Approve on Tailscale →</button>
          <button onClick={onRefresh} style={btnStyle(false)}>I approved — re-check</button>
        </div>
      ) : undefined,
    },
  ];

  return createPortal(
    <div onClick={e => { if (e.target === e.currentTarget) onClose(); }}
      style={{ position: "fixed", inset: 0, zIndex: 1200, background: "rgba(5,4,4,0.65)",
        display: "flex", alignItems: "center", justifyContent: "center", padding: 20 }}>
      <div className="glass-strong" style={{ width: "100%", maxWidth: 520, maxHeight: "86vh",
        overflow: "auto", padding: 22, display: "flex", flexDirection: "column", gap: 14 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ fontSize: 18 }}>📡</span>
          <div style={{ flex: 1 }}>
            <div style={{ fontWeight: 800, fontSize: 15 }}>Set up remote access</div>
            <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
              One-time, ~2 minutes. After this, 📡 Live and 🔗 Share links in Telegram just work.
            </div>
          </div>
          <button onClick={onClose} aria-label="Close"
            style={{ background: "none", border: "none", cursor: "pointer", color: "var(--text-muted)", padding: 4 }}>
            <X size={16} />
          </button>
        </div>

        {allDone ? (
          <div style={{ padding: "14px 16px", borderRadius: 12, fontSize: 12.5, fontWeight: 600,
            background: "color-mix(in srgb, var(--accent) 10%, transparent)", border: "1px solid var(--accent)", color: "var(--accent)" }}>
            ✓ Remote access is set up — links use {status?.base_url}
          </div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {steps.map((s, i) => (
              <div key={s.title} style={{ display: "flex", gap: 12, alignItems: "flex-start",
                opacity: s.state === "pending" ? 0.45 : 1,
                padding: "10px 12px", borderRadius: 12,
                background: s.state === "current" ? "rgb(var(--ink) / 0.045)" : "transparent",
                border: `1px solid ${s.state === "current" ? "var(--border)" : "transparent"}` }}>
                <div style={{ width: 22, height: 22, borderRadius: 999, flexShrink: 0, marginTop: 1,
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: 11, fontWeight: 800,
                  background: s.state === "done" ? "var(--accent)" : "rgb(var(--ink) / 0.08)",
                  color: s.state === "done" ? "#08130b" : "var(--text-secondary)",
                  border: s.state === "current" ? "1px solid var(--accent)" : "1px solid transparent" }}>
                  {s.state === "done" ? <Check size={12} strokeWidth={3} /> : i + 1}
                </div>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: 12.5, fontWeight: 700 }}>{s.title}</div>
                  <div style={{ fontSize: 11, color: "var(--text-secondary)", lineHeight: 1.55, marginTop: 2 }}>{s.desc}</div>
                  {s.action && <div style={{ marginTop: 8 }}>{s.action}</div>}
                </div>
              </div>
            ))}
          </div>
        )}

        <div style={{ fontSize: 10.5, color: "var(--text-muted)", lineHeight: 1.6,
          borderTop: "1px solid var(--border)", paddingTop: 10 }}>
          Whoever you send a link to just opens it in their browser — <strong>they don't need Tailscale</strong>.
          Free for personal use.
        </div>
      </div>
    </div>,
    document.body,
  );
}

function btnStyle(primary: boolean): React.CSSProperties {
  return primary
    ? { fontSize: 12, fontWeight: 700, padding: "6px 14px", borderRadius: 8, cursor: "pointer",
        background: "color-mix(in srgb, var(--accent) 12%, transparent)", color: "var(--accent)", border: "1px solid var(--accent)" }
    : { fontSize: 12, fontWeight: 600, padding: "6px 14px", borderRadius: 8, cursor: "pointer",
        background: "rgb(var(--ink) / 0.06)", color: "var(--text-secondary)", border: "1px solid var(--border)" };
}

/* ── Appearance section ──────────────────────────────────────────────────────── */
// Settings section nav with a liquid-glass pill that slides to the active section.
// Active = last clicked OR the section currently scrolled under the glass header
// (scroll-spy). The pill animates with a spring; buttons accent when active.
function AnchorNav({ items, contentRef }: {
  items: { id: string; label: string }[];
  contentRef: React.RefObject<HTMLDivElement | null>;
}) {
  const navRef = useRef<HTMLDivElement>(null);
  const btnRefs = useRef<Record<string, HTMLButtonElement | null>>({});
  const [activeId, setActiveId] = useState(items[0]?.id ?? "");
  const activeIdRef = useRef(activeId);
  activeIdRef.current = activeId;
  const [pill, setPill] = useState<{ left: number; width: number }>({ left: 0, width: 0 });
  // While a chip click is smooth-scrolling, suppress the scroll-spy so the pill
  // glides STRAIGHT to the target instead of sweeping through every section.
  const lockUntil = useRef(0);

  // FLIP-style measurement: read the ACTIVE chip's live geometry on demand (never
  // cached) — robust to the panel being `display:none` at mount (we re-measure the
  // moment it gains size via the ResizeObserver below) and to font/resize reflow.
  const measure = useCallback(() => {
    const nav = navRef.current;
    const btn = btnRefs.current[activeIdRef.current];
    if (!nav || !btn || nav.offsetWidth === 0) return; // zero-size → keep pill hidden
    setPill({ left: btn.offsetLeft, width: btn.offsetWidth });
  }, []);

  useLayoutEffect(() => { measure(); }, [measure, activeId, items.length]);
  useEffect(() => {
    const nav = navRef.current;
    if (!nav) return;
    const ro = new ResizeObserver(() => measure());   // fires on display:none→visible + reflow
    ro.observe(nav);
    const onWinResize = () => measure();
    window.addEventListener("resize", onWinResize);
    document.fonts?.ready.then(() => measure()).catch(() => {});
    return () => { ro.disconnect(); window.removeEventListener("resize", onWinResize); };
  }, [measure]);

  // Keep the active chip visible within the horizontal scroller.
  useEffect(() => { btnRefs.current[activeId]?.scrollIntoView({ block: "nearest", inline: "nearest" }); }, [activeId]);

  // Scroll-spy (rAF-throttled, suppressed during a click's programmatic scroll).
  useEffect(() => {
    const container = contentRef.current;
    if (!container) return;
    let raf = 0;
    const compute = () => {
      raf = 0;
      if (Date.now() < lockUntil.current) return;
      let current = items[0]?.id ?? "";
      // Edge-case fix: the LAST section can never scroll high enough to cross the
      // detection line (nothing below it pushes it up), so a plain "last section
      // past the line" scan falls back to the previous one — the pill bounces off
      // the right-most chip. Anchor to the last item when scrolled to the bottom
      // (and the first when at the very top); otherwise use the line scan.
      const maxScroll = container.scrollHeight - container.clientHeight;
      if (maxScroll > 0 && container.scrollTop >= maxScroll - 2) {
        current = items[items.length - 1]?.id ?? current;
      } else if (container.scrollTop <= 2) {
        current = items[0]?.id ?? current;
      } else {
        const overlay = (container.previousElementSibling as HTMLElement | null)?.offsetHeight || 120;
        const line = container.getBoundingClientRect().top + overlay + 14;
        for (const { id } of items) {
          const el = container.querySelector(`#${id}`) as HTMLElement | null;
          if (el && el.getBoundingClientRect().top <= line) current = id;
        }
      }
      setActiveId(prev => (prev === current ? prev : current));
    };
    const onScroll = () => { if (!raf) raf = requestAnimationFrame(compute); };
    compute();
    container.addEventListener("scroll", onScroll, { passive: true });
    return () => { container.removeEventListener("scroll", onScroll); if (raf) cancelAnimationFrame(raf); };
  }, [contentRef, items]);

  const scrollTo = (id: string) => {
    const container = contentRef.current;
    const target = container?.querySelector(`#${id}`) as HTMLElement | null;
    if (!container || !target) return;
    const overlay = (container.previousElementSibling as HTMLElement | null)?.offsetHeight || 120;
    const top = container.scrollTop + (target.getBoundingClientRect().top - container.getBoundingClientRect().top) - overlay - 8;
    lockUntil.current = Date.now() + 650;  // glide straight; ignore spy until settled
    setActiveId(id);                        // pill springs to the clicked chip now
    container.scrollTo({ top: Math.max(0, top), behavior: "smooth" });
  };

  return (
    <div ref={navRef} className={styles.anchorNav} style={{ position: "relative" }}>
      {/* Sliding liquid-glass capsule (composited; specular rim, no heavy refraction). */}
      <span aria-hidden className={styles.anchorPill}
        style={{ width: pill.width, transform: `translateX(${pill.left}px)`, opacity: pill.width ? 1 : 0 }} />
      {items.map(({ id, label }) => {
        const active = activeId === id;
        return (
          <button key={id} ref={el => { btnRefs.current[id] = el; }}
            className={styles.anchorBtn}
            style={{ position: "relative", zIndex: 1,
              color: active ? "var(--accent)" : undefined,
              fontWeight: active ? 700 : undefined,
              // Active chip is transparent so the capsule shows through; inactive
              // chips keep their normal hover background.
              background: active ? "transparent" : undefined }}
            onClick={() => scrollTo(id)}>
            {label}
          </button>
        );
      })}
    </div>
  );
}

function AppearanceSection() {
  const [surface, setSurfaceState] = useState<AppSurface>(() => loadSurface());

  const setSurface = (s: AppSurface) => {
    applyTheme(s);
    setSurfaceState(s);
  };

  // The accent swatch row lived here. It is gone deliberately: there is one
  // accent, defined once in index.css, and everything else derives from it.
  // The five swatches also duplicated the hexes from index.css, so changing a
  // theme in one place silently desynced the other.

  const SURFACES: { id: AppSurface; label: string }[] = [
    { id: "frosted", label: "Frosted" },
    { id: "solid",   label: "Solid"   },
  ];

  // Shared segmented control — one visual language for every appearance choice
  // (Apple Settings style: quiet rows, no preview cards, no decoration).
  const Segmented = <T extends string>({ options, value, onPick }: {
    options: { id: T; label: string }[]; value: T; onPick: (v: T) => void;
  }) => (
    <div style={{ display: "flex", background: "var(--bg-base)", border: "1px solid var(--border)", borderRadius: 8, overflow: "hidden" }}>
      {options.map((o, i) => (
        <button key={o.id} onClick={() => onPick(o.id)}
          style={{ padding: "5px 14px", fontSize: 12, fontWeight: 600, border: "none", cursor: "pointer",
            background: value === o.id ? "var(--hl)" : "transparent",
            color: value === o.id ? "var(--accent)" : "var(--text-muted)",
            borderRight: i < options.length - 1 ? "1px solid var(--border)" : "none" }}>
          {o.label}
        </button>
      ))}
    </div>
  );

  return (
    <div id="s-themes" className={styles.section}>
      <div className={styles.sectionHeader}>
        <span className={styles.sectionTitle}>
          <Zap size={12} /> Appearance
        </span>
      </div>

      <div style={{ padding: "10px 16px", display: "flex", alignItems: "center", gap: 8 }}>
        <span style={{ fontSize: 12, color: "var(--text-secondary)", flex: 1 }}>Surface</span>
        <Segmented options={SURFACES} value={surface} onPick={setSurface} />
      </div>
    </div>
  );
}

/* ── Telegram ───────────────────────────────────────────────────────────────
   There is ONE notification channel. A `ChannelDef` interface, a CHANNEL_DEFS
   array with a single entry, an accordion `ChannelRow` with expand/collapse
   state, a `{n} active` counter over a one-element list, and a
   Record<string, ...> of per-channel test callbacks all used to wrap it — a
   generic multi-channel framework around one form. The others (ntfy, pairing,
   webhooks) were deleted long ago and are not coming back; Home Assistant and
   MQTT live in their own sections. Flattened.
   ─────────────────────────────────────────────────────────────────────── */

function TestBtn({ disabled, loading, onClick, color = "var(--accent)" }: { disabled: boolean; loading: boolean; onClick: () => void; color?: string }) {
  return (
    <button disabled={disabled} onClick={onClick}
      style={{ alignSelf: "flex-start", padding: "6px 16px", fontSize: 12,
        fontWeight: 600, borderRadius: 20,
        border: `1px solid ${disabled ? "var(--border)" : color + "60"}`,
        background: disabled ? "var(--bg-surface)" : color + "18",
        color: disabled ? "var(--text-muted)" : color,
        cursor: disabled ? "not-allowed" : "pointer",
        display: "flex", alignItems: "center", gap: 6,
      }}>
      <Send size={11} />
      {loading ? "Sending…" : "Send Test"}
    </button>
  );
}

const DEFAULT: Settings = {
  // Detection knobs below mirror the Rust backend Default (state.rs) == the
  // "Balanced" preset. Keep them in sync — `reset()` writes these verbatim.
  sensitivity: 0.04,
  motion_threshold: 8,
  motion_min_frames: 3,
  motion_open_score_mult: 1.0,
  require_object_to_open_event: false,
  detect_max_disappeared_frames: 15,
  re_analysis_interval_secs: 60,
  record_on_motion: true,
  record_pre_buffer_secs: 3,
  record_post_buffer_secs: 10,
  record_max_event_secs: 300,
  stream_port: 8882,
  stream_quality: 60,
  retention_days: 7,
  ai_model: "Xenova/yolov9-c",
  agent_enabled: false,
  vision_model: "",
  agent_poll_secs: 30,
  alert_min_risk: "suspicious",
  camera_name: "",
  telegram_bot_token: "",
  telegram_chat_id: "",
  auto_reid: true,
  reid_threshold: 0.50,
  face_model: "off",
  // The four face thresholds are RAW ArcFace cosine, matching state.rs.
  // They used to sit on the legacy probability scale (0.7 / 0.9 / 0.8 / 0.20),
  // which is the band this file's own hint calls non-functional — and since
  // this object seeds `form` before settings hydrate, hitting Save early wrote
  // those values for real and silently killed recognition.
  face_detection_threshold: 0.5,
  face_recognition_threshold: 0.5,
  face_unknown_score: 0.4,
  face_class_confidence: 0.6,
  face_liveness: false,
  face_quality_floor: 0.10,
  strobe_profile: "balanced",
  yolo_confidence_threshold: 0.40,
  yolo_class_filter: "",
  alpr_region: "global",
  github_repo: "",
  device_name: "",
  auth_username: "",
  auth_password_hash: "",
  nvr_enabled: true,
  nvr_segment_mins: 1,
  nvr_max_gb: 50,
  nvr_record_mode: "always" as const,
  nvr_retain_days: 7,
  nvr_retain_event_days: 30,
  keep_event_clips: true,   // no longer a choice — see `save`
  ai_provider: "local",
  openai_api_key: "",
  anthropic_api_key: "",
  groq_api_key: "",
  xai_api_key: "",
  gemini_api_key: "",
  openai_compatible_url: "",
  openai_compatible_key: "",
};

function fmtBytes(b: number) {
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
  if (b < 1024 * 1024 * 1024) return `${(b / 1024 / 1024).toFixed(1)} MB`;
  return `${(b / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

// ── Channel configuration panel ─────────────────────────────────────────────
function TelegramConfig({ form, patch, showToast }: {
  form: Settings;
  patch: (k: keyof Settings, v: any) => void;
  showToast: (msg: string, type: "success"|"error"|"info") => void;
}) {
  const [connectingTg, setConnectingTg] = useState(false);
  const [testing, setTesting] = useState(false);

  return (
    <div style={{ padding: "4px 14px 14px" }}>
      <>
        <ChannelField label="Bot Token" more="From @BotFather — t.me/botfather">
          <input type="password" placeholder="1234567890:AAF…" value={form.telegram_bot_token}
            onChange={e => patch("telegram_bot_token", e.target.value)}
            style={inputStyle} />
        </ChannelField>
        {/* One-tap connect: validates the token + auto-fills the Chat ID so the
            user never has to hunt for a numeric ID via a third-party bot. */}
        <div style={{ display: "flex", alignItems: "center", gap: 8, margin: "2px 0 8px" }}>
          <button
            disabled={connectingTg || !form.telegram_bot_token}
            onClick={async () => {
              setConnectingTg(true);
              try {
                const r = await api.telegramConnect(form.telegram_bot_token);
                if (r.needs_message) {
                  showToast(`Bot @${r.bot_username} is valid! Now open Telegram, send /start to @${r.bot_username}, then tap Connect again.`, "info");
                } else {
                  patch("telegram_chat_id", r.chat_id);
                  showToast(`Connected to @${r.bot_username} — chatting with ${r.chat_name}. Save to finish.`, "success");
                }
              } catch (e: any) {
                showToast(e.message ?? String(e), "error");
              } finally { setConnectingTg(false); }
            }}
            style={{
              padding: "6px 14px", borderRadius: 8, fontSize: 12, fontWeight: 700,
              cursor: connectingTg || !form.telegram_bot_token ? "not-allowed" : "pointer",
              background: "rgba(0,136,204,0.14)", color: "#0088cc",
              border: "1px solid rgba(0,136,204,0.4)", opacity: !form.telegram_bot_token ? 0.5 : 1,
            }}>
            {connectingTg ? "Connecting…" : "⚡ Connect & auto-fill Chat ID"}
          </button>
          <span style={{ fontSize: 10, color: "var(--text-muted)" }}>
            Paste the token, message your bot once, then tap this.
          </span>
        </div>
        <ChannelField label="Chat ID" more="Auto-filled by Connect above — or paste it manually">
          <input type="text" placeholder="123456789" value={form.telegram_chat_id}
            onChange={e => patch("telegram_chat_id", e.target.value)}
            style={inputStyle} />
        </ChannelField>
        <TestBtn color={"#0088cc"} disabled={testing || !form.telegram_bot_token || !form.telegram_chat_id} loading={testing}
          onClick={async () => { setTesting(true); try { showToast(await api.sendTelegramTest(form.telegram_bot_token, form.telegram_chat_id), "success"); } catch (e: any) { showToast(e.message ?? String(e), "error"); } finally { setTesting(false); } }} />
      </>
    </div>
  );
}

const inputStyle: React.CSSProperties = {
  width: "100%", maxWidth: 360, boxSizing: "border-box",
  padding: "7px 10px", borderRadius: 7,
  border: "1px solid var(--border)", background: "var(--bg-surface)",
  color: "var(--text-primary)", fontSize: 12, fontFamily: "var(--font-mono)",
  outline: "none",
};

/** Same hint/more split as `Field` — short line visible, detail on hover. */
function ChannelField({ label, hint, more, children }: {
  label: string; hint?: string; more?: string; children: React.ReactNode;
}) {
  return (
    <div title={more} style={{ display: "flex", flexDirection: "column", gap: 4, marginBottom: 10 }}>
      <div style={{ fontSize: 11, fontWeight: 600, color: "var(--text-secondary)" }}>{label}</div>
      {children}
      {hint && <div style={{ fontSize: 10, color: "var(--text-muted)", lineHeight: 1.4 }}>{hint}</div>}
    </div>
  );
}

/* ── Face Recognition (standard) + llmfit-core Auto-pick ────────────── */
function FaceRecognitionSection({
  form, patch, showAdvanced,
}: {
  form: Settings;
  patch: (k: keyof Settings, v: any) => void;
  showAdvanced: boolean;
}) {
  /* Every control here is conditional, so the section could render as a title
     with nothing under it — which reads as a broken feature rather than a
     hidden one. Bail before the heading instead of after it. Note this needs
     BOTH conditions: with a model set to "off", Advanced alone still left an
     empty box. */
  if (!showAdvanced || !form.face_model || form.face_model === "off") return null;

  return (
    <div id="s-face" className={styles.section}>
      <div className={styles.sectionHeader}>
        <span className={styles.sectionTitle}><User size={12} /> Face Recognition</span>
      </div>

      {/* The model picker (Off / Small / Large) and the llmfit "Auto" button lived
          here. Both moved out rather than being duplicated:

          • Arsenal's picker knows which tiers are actually INSTALLED; this one
            let you select "Large" with no weights on disk, and recognition then
            silently did nothing.
          • "Auto" called `recommendFaceModel`, the last consumer of the llmfit
            scoring layer that Arsenal removed for advertising an engine that no
            longer exists.

          What stays here is the part Arsenal got wrong: the raw thresholds, on
          the real ArcFace cosine scale, behind Advanced. */}

      {/* Raw threshold knobs — ADVANCED only. These are RAW ArcFace cosine
          values (confident match ≈ 0.5, unknown floor ≈ 0.4) — NOT mature NVRs'
          0.9/0.8 scale; typing mature NVRs' documented numbers here silently
          breaks recognition, which is exactly why they hide behind Advanced.
          The old hints cited mature NVRs' scale and the ??/|| fallbacks reset
          cleared fields to 0.9/0.8 — both actively harmful. */}
      <>
        <Field label="Detection threshold" more="Confidence required to consider a region a face. Default 0.5 (liberal — recognition stays strict).">
          <input type="number" min="0.3" max="0.99" step="0.05"
            value={form.face_detection_threshold ?? 0.5}
            onChange={e => patch("face_detection_threshold", parseFloat(e.target.value) || 0.5)}
            className={styles.numInput} style={{ width: 70 }} />
        </Field>
        {/* max is the documented ceiling, not 0.99. The hint has always said
            "above ~0.65 stops recognition"; leaving the input open to 0.99 made
            that advice the user's problem to remember, and 0.9 is exactly the
            number the other NVR docs on the internet tell them to type. */}
        <Field label="Recognition threshold" hint="Above ~0.65 stops recognition" more="RAW ArcFace cosine required to apply a name. Default 0.5. This is NOT mature NVRs' 0.9 scale — values above ~0.65 stop all recognition.">
          <input type="number" min="0.3" max="0.65" step="0.05"
            value={form.face_recognition_threshold ?? 0.5}
            onChange={e => patch("face_recognition_threshold", parseFloat(e.target.value) || 0.5)}
            className={styles.numInput} style={{ width: 70 }} />
        </Field>
        <Field label="Unknown score" more="Below this raw cosine the match is labelled 'unknown' instead of a name. Default 0.4.">
          <input type="number" min="0.2" max="0.99" step="0.05"
            value={form.face_unknown_score ?? 0.4}
            onChange={e => patch("face_unknown_score", parseFloat(e.target.value) || 0.4)}
            className={styles.numInput} style={{ width: 70 }} />
        </Field>
        <Field label="Quality floor" more="Laplacian blur floor (÷300 scale). Lower keeps more (potentially blurry) faces. Default 0.10.">
          <input type="number" min="0" max="0.95" step="0.05"
            value={form.face_quality_floor ?? 0.10}
            onChange={e => patch("face_quality_floor", parseFloat(e.target.value) || 0.10)}
            className={styles.numInput} style={{ width: 70 }} />
        </Field>
      </>
    </div>
  );
}

export function SettingsPanel() {
  const { settings: stored, setSettings: setStoreSettings, showToast } = useStore(useShallow(s => ({ settings: s.settings, setSettings: s.setSettings, showToast: s.showToast })));
  const setSettings = setStoreSettings;
  const contentRef = useRef<HTMLDivElement>(null);
  const [form, setForm] = useState<Settings>(stored ?? DEFAULT);
  const [saving, setSaving] = useState(false);
  const [storageInfo, setStorageInfo] = useState<StorageInfo | null>(null);
  const [loadingStorage, setLoadingStorage] = useState(false);
  const [clearingAll, setClearingAll] = useState(false);
  const [clearingNvr, setClearingNvr] = useState(false);
  const [purgingOrphans, setPurgingOrphans] = useState(false);
  const [updateInfo,       setUpdateInfo]       = useState<any>(null);
  const [checkingUpdate,   setCheckingUpdate]   = useState(false);
  // Progressive disclosure — raw threshold knobs hide behind this persisted switch.
  const [showAdvanced, setShowAdvanced] = useState<boolean>(() => {
    try { return localStorage.getItem("sc-settings-advanced") === "1"; } catch { return false; }
  });
  const toggleAdvanced = (v: boolean) => {
    setShowAdvanced(v);
    try { localStorage.setItem("sc-settings-advanced", v ? "1" : "0"); } catch {}
  };
  const activePreset = matchPreset(form);
  const applyPreset = (p: SensPreset) => setForm(f => ({ ...f, ...SENSITIVITY_PRESETS[p] }));

  useEffect(() => { if (stored) setForm(stored); }, [stored]);

  const loadStorage = useCallback(async () => {
    setLoadingStorage(true);
    try { setStorageInfo(await api.getStorageInfo()); }
    catch (e: any) { showToast(e.message ?? "Failed to load storage info", "error"); }
    finally { setLoadingStorage(false); }
  }, [showToast]);

  // Auto-load storage once on mount (now cheap — backend walks the FS off-thread).
  useEffect(() => { loadStorage(); }, [loadStorage]);

  // Measured recording rate → projected REAL retention at the disk cap. This is
  // what actually bounds retention when many cameras record (the cap prunes
  // oldest-first long before retention_days at higher camera counts).
  const [diskProj, setDiskProj] = useState<import("../../types").DiskProjection | null>(null);
  useEffect(() => {
    api.nvrDiskProjection().then(setDiskProj).catch(() => setDiskProj(null));
  }, []);

  // Remote access (Tailscale Funnel — compliant live/clip sharing).
  const [tsStatus, setTsStatus] = useState<import("../../types").TailscaleStatus | null>(null);
  const [tsBusy, setTsBusy] = useState(false);
  const refreshTs = useCallback(() => { api.tailscaleStatus().then(setTsStatus).catch(() => setTsStatus(null)); }, []);
  useEffect(() => { refreshTs(); }, [refreshTs]);
  const enableTs = async () => {
    setTsBusy(true);
    try {
      const st = await api.tailscaleEnable();
      setTsStatus(st);
      if (st.enable_url) {
        showToast("One-time: enable Funnel for your Tailscale account in the opened page, then click again.", "info");
        try { const { open } = await import("@tauri-apps/plugin-shell"); await open(st.enable_url); }
        catch { window.open(st.enable_url, "_blank"); }
      } else if (st.funnel_active) {
        showToast(`Remote access ON — live links now use ${st.base_url}`, "success");
      }
    } catch (e: any) { showToast(e.message ?? String(e), "error"); }
    finally { setTsBusy(false); }
  };
  // First-time setup guide (info icon + reactive stepper). Auto-opens ONCE when
  // remote access isn't fully set up (until dismissed) — this is how the app
  // TELLS the user what to do instead of failing silently when they ask
  // Telegram for a live link before Tailscale is configured.
  const [tsGuideOpen, setTsGuideOpen] = useState(false);
  const tsGuideAutoShown = useRef(false);
  useEffect(() => {
    if (tsGuideAutoShown.current || tsStatus === null) return;
    const ready = tsStatus.installed && tsStatus.logged_in && tsStatus.funnel_active;
    if (!ready && !localStorage.getItem("sc-tailscale-guide-dismissed")) {
      tsGuideAutoShown.current = true;
      setTsGuideOpen(true);
    }
  }, [tsStatus]);
  const closeTsGuide = useCallback(() => {
    setTsGuideOpen(false);
    localStorage.setItem("sc-tailscale-guide-dismissed", "1");
  }, []);

  const clearAll = async () => {
    const used = storageInfo ? ` (${fmtBytes(storageInfo.total_bytes)})` : "";
    if (!confirm(`Delete ALL footage — events, clips, and NVR recordings${used}? This cannot be undone.`)) return;
    setClearingAll(true);
    try {
      await api.clearAllEvents();
      showToast("All footage cleared — events, clips and NVR recordings deleted", "success");
      setStorageInfo(null);
    } catch (e: any) {
      showToast(e.message ?? "Failed to clear footage", "error");
    } finally { setClearingAll(false); }
  };

  // Enterprise ranged delete: remove all footage (NVR + events) in [fromMs, toMs].
  const [deletingRange, setDeletingRange] = useState(false);
  const [rangeFrom, setRangeFrom] = useState("");
  const [rangeTo,   setRangeTo]   = useState("");
  const deleteRange = async (fromMs: number, toMs: number, label: string) => {
    if (!confirm(`Delete all footage ${label}? This deletes the NVR recordings and motion events in that window and cannot be undone.`)) return;
    setDeletingRange(true);
    try {
      const r = await api.deleteFootageInRange(
        new Date(fromMs).toISOString(), new Date(toMs).toISOString());
      showToast(`Deleted ${r.segments_deleted} recording${r.segments_deleted !== 1 ? "s" : ""} and ${r.events_deleted} event${r.events_deleted !== 1 ? "s" : ""}`, "success");
      setStorageInfo(null);
    } catch (e: any) {
      showToast(e.message ?? "Failed to delete footage", "error");
    } finally { setDeletingRange(false); }
  };
  const deleteOlderThan = (days: number) =>
    deleteRange(0, Date.now() - days * 86400000, `older than ${days} days`);
  const deleteCustomRange = () => {
    if (!rangeFrom || !rangeTo) { showToast("Pick both a start and end date", "error"); return; }
    const from = new Date(rangeFrom + "T00:00:00").getTime();
    const to   = new Date(rangeTo   + "T23:59:59").getTime();
    if (from > to) { showToast("Start date must be before end date", "error"); return; }
    deleteRange(from, to, `from ${rangeFrom} to ${rangeTo}`);
  };

  const clearNvr = async () => {
    if (!confirm("Delete all NVR continuous recordings? This cannot be undone.")) return;
    setClearingNvr(true);
    try {
      const count = await api.clearNvrRecordings();
      showToast(`Deleted ${count} NVR recording${count !== 1 ? "s" : ""}`, "success");
      setStorageInfo(null);
    } catch (e: any) {
      showToast(e.message ?? "Failed to clear NVR recordings", "error");
    } finally { setClearingNvr(false); }
  };

  const purgeOrphans = async () => {
    setPurgingOrphans(true);
    try {
      const n = await api.purgeOrphanedClips();
      showToast(`Removed ${n} orphaned clip${n !== 1 ? "s" : ""}`, "success");
      setStorageInfo(null);
    } catch (e: any) {
      showToast(e.message ?? "Failed to purge orphans", "error");
    } finally { setPurgingOrphans(false); }
  };

  const save = async () => {
    setSaving(true);
    try {
      // `keep_event_clips` no longer has a control. Forcing it true on every save
      // migrates anyone who had it off, so a stored `false` from an older build
      // cannot keep quietly discarding clips after the toggle is gone.
      const next: Settings = { ...form, keep_event_clips: true };
      await api.saveSettings(next);
      setSettings(next);
      showToast("Settings saved", "success");
    } catch (e: any) {
      showToast(e.message ?? "Save failed", "error");
    } finally { setSaving(false); }
  };

  const patch = (k: keyof Settings, v: any) => setForm((f) => ({ ...f, [k]: v }));

  return (
    <div className={styles.root}>
      {/* Floating Liquid Glass stack — header + tabs + anchor nav.
       * Position: absolute on top so the .content scroller can pass under it
       * and the backdrop-filter has live content to refract. */}
      <div className={styles.glassStack}>
        {/* Header */}
        <div className="panelHeader">
          <span className="panelTitle">Settings</span>
          <div className="panelActions">
            <button className={styles.saveBtn} onClick={save} disabled={saving}>
              <Save size={13} /> {saving ? "Saving…" : "Save"}
            </button>
          </div>
        </div>

        {/* Section nav row — the sliding-pill anchor nav on the left, and the
         * Advanced slider pinned on the right (sits directly below Save). */}
        <div className={styles.navRow}>
          {/* ORDER MUST MATCH THE DOM. The scroll-spy highlights whichever section
              is in view, so a nav that disagrees with the render order makes the
              pill jump backwards as you scroll. Recording / NVR / Storage now sit
              together (they were separated by six unrelated sections), and each
              label matches its section heading — the nav used to say "Themes" over
              a heading reading "Appearance". */}
          <AnchorNav contentRef={contentRef} items={[
            { id: "s-themes",        label: "Appearance"  },
            // ── Cameras & recording ──
            { id: "s-motion",        label: "Motion"      },
            { id: "s-recording",     label: "Event clips" },
            { id: "s-nvr",           label: "24/7 video"  },
            // These two used to render with NO id and NO nav chip, so the
            // scroll-spy pill stayed stuck on "24/7 video" the whole way past
            // them - the very invariant the comment above is about.
            { id: "s-remote",        label: "Remote"      },
            { id: "s-updates",       label: "Updates"     },
            { id: "s-storage",       label: "Storage"     },
            // ── Alerts & sharing ──
            // The ids are crossed for historical reasons (#s-telegram is the
            // Telegram section, #s-channels is Sharing); the LABELS are what the
            // user reads, so they match their headings.
            { id: "s-telegram",      label: "Telegram"    },
            { id: "s-notifications", label: "Alerts"      },
            { id: "s-channels",      label: "Sharing"     },
            // ── AI ──
            { id: "s-detection",     label: "Detection"   },
            // Both sections are Advanced-only, so their chips are too. A chip
            // pointing at a section that is not rendered scrolls nowhere and
            // wedges the scroll-spy pill — the invariant noted above.
            ...(showAdvanced ? [
            { id: "s-face",          label: "Faces"       },
            { id: "s-reid",          label: "People"      },
            ] : []),
            // ── App ──
            { id: "s-security",      label: "Security"    },
          ]} />
          <div className={styles.advToggle}
            title={showAdvanced ? "Hide raw threshold knobs" : "Show every raw threshold knob"}>
            <SlidersHorizontal size={12} />
            <span>Advanced</span>
            <Toggle checked={showAdvanced} onChange={toggleAdvanced} />
          </div>
        </div>
      </div>{/* /glassStack */}

      <div ref={contentRef} className={styles.content}>

        {/* ── Appearance ────────────────────────────────────────────────── */}
        <AppearanceSection />

        {/* Motion Detection */}
        <div id="s-motion" className={styles.section}>
          <div className={styles.sectionHeader}>
            <span className={styles.sectionTitle}>
              <Activity size={12} /> Motion Detection
            </span>
          </div>

          {/* Detection-sensitivity preset — one control writes the six "open an
              event" knobs (sensitivity, pixel threshold, frames-to-open, open
              multiplier, YOLO confidence, require-object). Simple users stop here;
              Advanced reveals the raw physics below. */}
          <div style={{ padding: "12px 16px", borderBottom: "1px solid var(--border)" }}>
            <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", marginBottom: 8 }}>
              <span style={{ fontSize: 12, fontWeight: 600, color: "var(--text-secondary)" }}>Detection sensitivity</span>
              {activePreset === "custom" && (
                <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.05em", color: "var(--accent)" }}>CUSTOM</span>
              )}
            </div>
            <SensitivityPresetPicker active={activePreset} onPick={applyPreset} />
            {!showAdvanced && (
              <p style={{ fontSize: 10.5, color: "var(--text-muted)", margin: "8px 2px 0", lineHeight: 1.5 }}>
                {activePreset === "custom"
                  ? "Hand-tuned values are active."
                  : "Covers most setups."}
              </p>
            )}
          </div>

          {showAdvanced && (<>
            <Field label="Sensitivity" more="Fraction of pixels that must change to trigger motion. Raised by a preset above; editing it here switches the preset to Custom.">
              <input
                type="range" min="0.005" max="0.3" step="0.005"
                value={form.sensitivity}
                onChange={(e) => patch("sensitivity", parseFloat(e.target.value))}
                className={styles.range}
              />
              <span className={styles.rangeVal}>{(form.sensitivity * 100).toFixed(1)}%</span>
            </Field>

            <Field label="Pixel threshold (0–255)" more="Brightness change per pixel to count as motion (0–255)">
              <input
                type="range" min="5" max="100" step="1"
                value={form.motion_threshold}
                onChange={(e) => patch("motion_threshold", parseInt(e.target.value))}
                className={styles.range}
              />
              <span className={styles.rangeVal}>{form.motion_threshold}</span>
            </Field>

            {/* v8 hysteresis knobs — fix for the "event stuck on" bug. The
                streak-based open/close prevents single noisy frames from
                tripping events. Defaults work well; power users can lower
                the streak for very twitchy cameras or raise the multiplier. */}
            <Field label="Consecutive frames to open" more="Require this many over-threshold frames in a row before opening an event. 3 = stops single-frame noise. Lower if your camera runs <10fps.">
              <input type="number" min="1" max="10" step="1"
                value={form.motion_min_frames ?? 3}
                onChange={e => patch("motion_min_frames", parseInt(e.target.value, 10) || 3)}
                className={styles.numInput} style={{ width: 70 }} />
            </Field>

            <Field label="Open-event score multiplier" more="Multiplier on Sensitivity for OPENING a new event. Sustaining uses the raw sensitivity. Higher = harder to false-trigger events. Backend clamps to ≥ 1.0.">
              <input type="number" min="1.0" max="5.0" step="0.1"
                value={form.motion_open_score_mult ?? 1.0}
                onChange={e => patch("motion_open_score_mult", parseFloat(e.target.value) || 1.0)}
                className={styles.numInput} style={{ width: 70 }} />
            </Field>

            {/* "Always require object confirmation" lived here. motion_lifecycle.rs
                reads `require_object_to_open_event`, but state.rs forces it true
                whenever YOLO is active — which is every install that has a detector
                — so the toggle changed nothing for anyone who could see it. */}

            {/* v9 mature NVRs-model close + mid-event re-alerting */}
            <Field label="Max object disappearance (frames)" more="Frames YOLO can lose the tracked bbox before the event is considered ended. Mature NVRs' `detect.max_disappeared`. Default 15 ≈ 5s at the 3fps inference rate. Raise for cameras with frequent occlusion.">
              <input type="number" min="1" max="120" step="1"
                value={form.detect_max_disappeared_frames ?? 15}
                onChange={e => patch("detect_max_disappeared_frames", parseInt(e.target.value, 10) || 15)}
                className={styles.numInput} style={{ width: 70 }} />
            </Field>
            <Field label="Mid-event re-analysis (seconds)" more="During a long open event, re-run the agent's clip analysis every N seconds so fresh activity actually alerts you instead of waiting for event close. 0 disables. Default 60.">
              <input type="number" min="0" max="600" step="10"
                value={form.re_analysis_interval_secs ?? 60}
                onChange={e => patch("re_analysis_interval_secs", parseInt(e.target.value, 10) || 0)}
                className={styles.numInput} style={{ width: 70 }} />
            </Field>
          </>)}
        </div>

        {/* Recording */}
        <div id="s-recording" className={styles.section}>
          <div className={styles.sectionHeader}>
            <span className={styles.sectionTitle}>
              <Video size={12} /> Recording
            </span>
          </div>

          {/* "Record on Motion" toggle removed — the backend has ignored it since
              v12 (events are virtual slices of the continuous NVR recording). */}
          {/* Behind Advanced: three interacting numbers with correct defaults.
              Nobody opens Settings wanting to change "pre-buffer" without already
              knowing what it is, and the motion sensitivity preset above is the
              control this section actually needs. */}
          {showAdvanced && (<>
          <Field label="Pre-buffer (sec)" more="Seconds to include before motion starts">
            <input
              type="number" min="0" max="30"
              value={form.record_pre_buffer_secs}
              onChange={(e) => patch("record_pre_buffer_secs", parseInt(e.target.value) || 0)}
              className={styles.numInput}
            />
            <span className={styles.rangeVal} style={{ minWidth: 24 }}>s</span>
          </Field>

          <Field label="Post-buffer (sec)" more="Seconds to keep recording after motion stops">
            <input
              type="number" min="0" max="60"
              value={form.record_post_buffer_secs}
              onChange={(e) => patch("record_post_buffer_secs", parseInt(e.target.value) || 0)}
              className={styles.numInput}
            />
            <span className={styles.rangeVal} style={{ minWidth: 24 }}>s</span>
          </Field>

          <Field label="Max event length (sec)" more="Hard cap on a single event. Force-closes runaways so nothing records for minutes with nobody acting (0 = unlimited). Long activity splits into chunks the Review view re-merges.">
            <input
              type="number" min="0" max="3600"
              value={form.record_max_event_secs}
              onChange={(e) => patch("record_max_event_secs", parseInt(e.target.value) || 0)}
              className={styles.numInput}
            />
            <span className={styles.rangeVal} style={{ minWidth: 24 }}>s</span>
          </Field>
          </>)}

          <Field label="Delete everything older than" hint="Overrides the two 24/7 video retention numbers"
                 more="An absolute ceiling: nvr_recording.rs applies this OVER the continuous-footage and event-clip retention set in 24/7 video, so nothing survives past it whatever those say. 0 = keep forever.">
            <input
              type="number" min="0" max="365"
              value={form.retention_days}
              onChange={(e) => patch("retention_days", parseInt(e.target.value) || 0)}
              className={styles.numInput}
            />
            <span className={styles.rangeVal} style={{ minWidth: 32 }}>days</span>
          </Field>
        </div>

        <div id="s-nvr" className={styles.section}>
          <div className={styles.sectionHeader}>
            <span className={styles.sectionTitle}><Film size={12} /> 24/7 Video Recording</span>
          </div>

          {/* Master toggle */}
          <Field label="Enable NVR" more="Continuously records all camera footage using ffmpeg. Independent of motion detection.">
            <Toggle checked={form.nvr_enabled} onChange={v => patch("nvr_enabled", v)} />
          </Field>

          {form.nvr_enabled && (<>
            {/* Recording mode */}
            <div style={{ padding: "10px 16px", borderBottom: "1px solid var(--border)" }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: "var(--text-secondary)", marginBottom: 8 }}>Recording Mode</div>
              <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                {([
                  { id: "always",       label: "Always",       desc: "Record everything 24/7 — maximum coverage" },
                  { id: "motion_only",  label: "Motion Only",  desc: "Only record when motion is detected — saves ~80% storage" },
                  { id: "events_only",  label: "Events Only",  desc: "Save event clips only — no continuous recording" },
                ] as const).map(m => {
                  const active = (form.nvr_record_mode ?? "always") === m.id;
                  return (
                    <button key={m.id} type="button" onClick={() => patch("nvr_record_mode", m.id)}
                      title={m.desc}
                      style={{
                        padding: "5px 14px", borderRadius: 8, fontSize: 11, fontWeight: 600, cursor: "pointer",
                        border: `1px solid ${active ? "var(--accent)" : "var(--border)"}`,
                        background: active ? "color-mix(in srgb, var(--accent) 10%, transparent)" : "var(--bg-elevated)",
                        color: active ? "var(--accent)" : "var(--text-secondary)",
                      }}>
                      {m.label}
                    </button>
                  );
                })}
              </div>
              <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 6 }}>
                {(form.nvr_record_mode ?? "always") === "always"      && "Recording everything — highest storage usage, best forensic coverage."}
                {(form.nvr_record_mode ?? "always") === "motion_only" && "Only records during detected motion — reduces storage by ~80% vs always-on."}
                {(form.nvr_record_mode ?? "always") === "events_only" && "Saves only short motion event clips — minimal storage, no browsable timeline."}
              </div>
            </div>

            {/* Retention — two tiers like mature NVRs */}
            <div style={{ padding: "8px 16px 12px", borderBottom: "1px solid var(--border)" }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: "var(--text-secondary)", marginBottom: 8 }}>Retention</div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                <div>
                  <div style={{ fontSize: 10, color: "var(--text-muted)", marginBottom: 4 }}>Continuous footage</div>
                  <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                    <input type="number" min="0" max="365"
                      value={form.nvr_retain_days ?? 7}
                      onChange={e => patch("nvr_retain_days", parseInt(e.target.value) || 0)}
                      className={styles.numInput} />
                    <span className={styles.rangeVal}>days</span>
                  </div>
                  <div style={{ fontSize: 9, color: "var(--text-muted)", marginTop: 3 }}>0 = keep forever</div>
                </div>
                <div>
                  <div style={{ fontSize: 10, color: "var(--text-muted)", marginBottom: 4 }}>Event clips</div>
                  <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                    <input type="number" min="0" max="365"
                      value={form.nvr_retain_event_days ?? 30}
                      onChange={e => patch("nvr_retain_event_days", parseInt(e.target.value) || 0)}
                      className={styles.numInput} />
                    <span className={styles.rangeVal}>days</span>
                  </div>
                  <div style={{ fontSize: 9, color: "var(--text-muted)", marginTop: 3 }}>0 = keep forever</div>
                </div>
              </div>
            </div>

            {/* "Keep event clips" was a toggle here; it is now always on (forced in
                the save handler). Pre-exporting the clip before the bulk tape is
                pruned is never the wrong answer — the only thing the choice bought
                the user was a way to lose footage they would later come looking for. */}

            {/* Segment length. A global key whose ONLY control used to be a slider
                inside the per-camera "Camera N settings" dialog, so the one place
                it belonged never had it. Advanced: 1 minute is right for almost
                everyone (shorter segments seek faster, longer ones cost less
                filesystem overhead). */}
            {showAdvanced && (
              <Field label="Segment length (min)" more="How long each recorded chunk is on disk. Shorter = faster seeking and finer pruning; longer = fewer files. 1 is a good default.">
                <input type="number" min="1" max="5"
                  value={form.nvr_segment_mins ?? 1}
                  onChange={e => patch("nvr_segment_mins", parseInt(e.target.value) || 1)}
                  className={styles.numInput} />
                <span className={styles.rangeVal} style={{ minWidth: 24 }}>min</span>
              </Field>
            )}

            {/* Storage cap */}
            <Field label="Storage cap (GB)" more="Hard limit — oldest segments deleted first when exceeded. 0 = no limit.">
              <input type="number" min="0" max="10000"
                value={form.nvr_max_gb}
                onChange={e => patch("nvr_max_gb", parseInt(e.target.value) || 0)}
                className={styles.numInput} />
              <span className={styles.rangeVal} style={{ minWidth: 24 }}>GB</span>
            </Field>

            {/* Storage projection — MEASURED rate, not an estimate */}
            {diskProj && diskProj.gb_per_day > 0.01 ? (
              <div style={{
                padding: "6px 16px 10px", fontSize: 10, lineHeight: 1.6,
                color: (form.nvr_max_gb > 0 && diskProj.projected_days_at_cap < (form.retention_days || 7))
                  ? "var(--danger, var(--status-alert))" : "var(--text-muted)",
              }}>
                Measured: {diskProj.gb_per_day.toFixed(1)} GB/day across {diskProj.recording_cams} camera{diskProj.recording_cams === 1 ? "" : "s"} (last 24 h).
                {form.nvr_max_gb > 0 && diskProj.projected_days_at_cap < 9999 &&
                  ` At this rate the ${form.nvr_max_gb} GB cap holds ~${diskProj.projected_days_at_cap < 10
                    ? diskProj.projected_days_at_cap.toFixed(1) : Math.round(diskProj.projected_days_at_cap)} days` +
                  (diskProj.projected_days_at_cap < (form.retention_days || 7)
                    ? ` — LESS than your ${form.retention_days || 7}-day retention. Raise the cap or reduce cameras/quality.` : ".")}
              </div>
            ) : (form.nvr_record_mode ?? "always") !== "events_only" && (
              <div style={{ padding: "6px 16px 10px", fontSize: 10, color: "var(--text-muted)", lineHeight: 1.6 }}>
                Estimate at 1080p (2 Mbps): {Math.round(2 * 3600 * (form.nvr_retain_days ?? 7) / 8 / 1024)} GB / camera for {form.nvr_retain_days ?? 7} days.
                {form.nvr_max_gb > 0 && ` Capped at ${form.nvr_max_gb} GB.`}
              </div>
            )}
          </>)}
        </div>

        {/* ── Remote access (Tailscale Funnel) ─────────────────────────── */}
        <div id="s-remote" className={styles.section}>
          <div className={styles.sectionHeader}>
            <span className={styles.sectionTitle}><Radio size={12} /> Remote access (live &amp; clip links)</span>
            <button onClick={() => setTsGuideOpen(true)} title="First-time setup guide"
              style={{ background: "none", border: "none", cursor: "pointer", padding: 3,
                color: "var(--text-muted)", display: "inline-flex", alignItems: "center" }}>
              <Info size={14} />
            </button>
          </div>
          {tsGuideOpen && (
            <TailscaleSetupGuide
              status={tsStatus}
              busy={tsBusy}
              onEnable={enableTs}
              onRefresh={refreshTs}
              onClose={closeTsGuide}
            />
          )}
          <div style={{ padding: "8px 16px 12px", fontSize: 11, color: "var(--text-secondary)", lineHeight: 1.6 }}>
            Shared links are served over <strong>Tailscale Funnel</strong>. Whoever you send one to just opens it in a browser.
          </div>
          <div style={{ padding: "0 16px 14px", display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
            {!tsStatus?.installed ? (
              <a href="https://tailscale.com/download" target="_blank" rel="noreferrer"
                style={{ fontSize: 12, fontWeight: 700, padding: "6px 14px", borderRadius: 8, textDecoration: "none",
                  background: "color-mix(in srgb, var(--status-idle) 14%, transparent)", color: "var(--status-idle)", border: "1px solid color-mix(in srgb, var(--status-idle) 40%, transparent)" }}>
                Install Tailscale (free) →
              </a>
            ) : !tsStatus?.logged_in ? (
              <span style={{ fontSize: 11.5, color: "var(--danger, var(--status-alert))" }}>
                Tailscale installed but not signed in — open the Tailscale app and log in, then refresh.
              </span>
            ) : tsStatus?.funnel_active ? (
              <span style={{ fontSize: 11.5, color: "var(--accent)", fontWeight: 700 }}>
                ✓ Active — links use {tsStatus.base_url}
              </span>
            ) : (
              <button onClick={enableTs} disabled={tsBusy}
                style={{ fontSize: 12, fontWeight: 700, padding: "6px 14px", borderRadius: 8, cursor: tsBusy ? "wait" : "pointer",
                  background: "color-mix(in srgb, var(--accent) 12%, transparent)", color: "var(--accent)", border: "1px solid var(--accent)" }}>
                {tsBusy ? "Enabling…" : "Enable remote access"}
              </button>
            )}
            <button onClick={refreshTs} className={styles.ghostBtn} style={{ padding: "5px 12px", fontSize: 11 }}>Refresh</button>
          </div>
        </div>

        {/* ── App Updates ──────────────────────────────────────────────── */}
        <div id="s-updates" className={styles.section}>
          <div className={styles.sectionHeader}>
            <span className={styles.sectionTitle}><RefreshCw size={12} /> App Updates</span>
          </div>
          <Field label="Relaunch after crash" hint="Comes back after you quit, too">
            <Toggle checked={form.relaunch_after_crash ?? false}
              onChange={v => patch("relaunch_after_crash", v)} />
          </Field>
            {/* The "GitHub Repository" text field lived here. `github_repo` has no
                Rust reader at all — checkForUpdate takes the repo as a call
                argument, and the persisted key was write-only. It is also the
                app's OWN repo, so the panel was asking the user to configure the
                vendor's identity, and then disabling "Check for Updates" until
                they guessed it. */}
          <div style={{ padding: "8px 16px", display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
            <button className={styles.ghostBtn}
              disabled={checkingUpdate}
              style={{ padding: "5px 14px", fontSize: 12 }}
              onClick={async () => {
                setCheckingUpdate(true);
                try {
                  const info = await api.checkForUpdate(APP_REPO);
                  setUpdateInfo(info);
                  if (!info.available) showToast(`You're on the latest version (${info.current})`, "success");
                } catch (e: any) {
                  showToast(e.message ?? "Update check failed", "error");
                } finally { setCheckingUpdate(false); }
              }}>
              <RefreshCw size={12} style={{ animation: checkingUpdate ? "spin 1s linear infinite" : "none" }} />
              {checkingUpdate ? "Checking…" : "Check for Updates"}
            </button>
            {updateInfo?.available && (
              <div style={{ flex: 1, background: "color-mix(in srgb, var(--accent) 8%, transparent)", border: "1px solid var(--border-accent)",
                borderRadius: 10, padding: "8px 14px", display: "flex", flexDirection: "column", gap: 4 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span style={{ fontSize: 13, fontWeight: 700, color: "var(--accent)" }}>
                    v{updateInfo.latest} available
                  </span>
                  <span style={{ fontSize: 10, color: "var(--text-muted)" }}>
                    (current: v{updateInfo.current})
                  </span>
                </div>
                {updateInfo.notes && (
                  <div style={{ fontSize: 11, color: "var(--text-secondary)", lineHeight: 1.5,
                    maxHeight: 80, overflow: "auto", whiteSpace: "pre-wrap" }}>
                    {updateInfo.notes}
                  </div>
                )}
                {updateInfo.download_url && (
                  <a href={updateInfo.download_url} target="_blank" rel="noopener noreferrer"
                    style={{ alignSelf: "flex-start", marginTop: 4, padding: "5px 14px",
                      fontSize: 12, fontWeight: 700, borderRadius: 20,
                      background: "var(--accent-fill)", color: "var(--on-accent)",
                      textDecoration: "none", display: "flex", alignItems: "center", gap: 5 }}>
                    <ExternalLink size={11} /> Download v{updateInfo.latest}
                  </a>
                )}
              </div>
            )}
            {updateInfo && !updateInfo.available && (
              <span style={{ fontSize: 12, color: "var(--accent)" }}>
                ✓ v{updateInfo.current} is up to date
              </span>
            )}
          </div>
        </div>

        {/* Storage */}
        <div id="s-storage" className={styles.section}>
          <div className={styles.sectionHeader}>
            <span className={styles.sectionTitle}>
              <HardDrive size={12} /> Storage
            </span>
          </div>

          {storageInfo ? (
            <>
              <div className={styles.storageGrid}>
                <div className={styles.statCell}>
                  <span className={styles.statLabel}>Total Used</span>
                  <span className={styles.statValue}>{fmtBytes(storageInfo.total_bytes)}</span>
                </div>
                <div className={styles.statCell}>
                  <span className={styles.statLabel}>Events</span>
                  <span className={styles.statValue}>{storageInfo.event_count}</span>
                  <span className={styles.statSub}>{storageInfo.clip_count} with clips</span>
                </div>
                {storageInfo.orphaned_clips > 0 && (
                  <div className={styles.statCell} style={{ gridColumn: "1 / -1" }}>
                    <span className={styles.statLabel} style={{ color: "var(--warn, var(--status-warn))" }}>
                      Orphaned Clips
                    </span>
                    <span className={styles.statValue} style={{ color: "var(--warn, var(--status-warn))" }}>
                      {storageInfo.orphaned_clips}
                    </span>
                    <span className={styles.statSub}>Video files on disk not linked to any event</span>
                  </div>
                )}
                {storageInfo.oldest_event && (
                  <div className={styles.statCell} style={{ gridColumn: "1 / -1" }}>
                    <span className={styles.statLabel}>Date Range</span>
                    <span className={styles.statSub} style={{ fontSize: 12, marginTop: 2 }}>
                      {storageInfo.oldest_event.slice(0, 10)} → {storageInfo.newest_event?.slice(0, 10) ?? "now"}
                    </span>
                  </div>
                )}
              </div>
              {(storageInfo.nvr_count ?? 0) > 0 && (
                <div className={styles.statCell} style={{ gridColumn: "1 / -1" }}>
                  <span className={styles.statLabel}>NVR Recordings</span>
                  <span className={styles.statValue}>{fmtBytes(storageInfo.nvr_bytes ?? 0)}</span>
                  <span className={styles.statSub}>{storageInfo.nvr_count} segment{storageInfo.nvr_count !== 1 ? "s" : ""} of continuous footage</span>
                </div>
              )}
              {/* ── Manage footage — delete by duration (enterprise NVR control) ── */}
              <div style={{ padding: "12px 16px 4px", borderTop: "1px solid var(--border)", marginTop: 8 }}>
                <div style={{ fontSize: 11, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--text-muted)", marginBottom: 8 }}>
                  Delete footage older than
                </div>
                <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 14 }}>
                  {[7, 14, 30, 90].map(d => (
                    <button key={d} className={styles.ghostBtn} disabled={deletingRange}
                      style={{ padding: "5px 14px", fontSize: 12 }}
                      onClick={() => deleteOlderThan(d)}>
                      {d} days
                    </button>
                  ))}
                </div>

                <div style={{ fontSize: 11, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--text-muted)", marginBottom: 8 }}>
                  Delete a date range
                </div>
                <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                  <input type="date" max={new Date().toISOString().slice(0, 10)}
                    value={rangeFrom} onChange={e => setRangeFrom(e.target.value)}
                    className={styles.numInput} style={{ width: 150, textAlign: "left", colorScheme: "dark" }} />
                  <span style={{ color: "var(--text-muted)", fontSize: 12 }}>to</span>
                  <input type="date" max={new Date().toISOString().slice(0, 10)}
                    value={rangeTo} onChange={e => setRangeTo(e.target.value)}
                    className={styles.numInput} style={{ width: 150, textAlign: "left", colorScheme: "dark" }} />
                  <button className={styles.dangerBtn} disabled={deletingRange || !rangeFrom || !rangeTo}
                    style={{ padding: "5px 14px", fontSize: 12 }}
                    onClick={deleteCustomRange}>
                    <Trash2 size={12} /> {deletingRange ? "Deleting…" : "Delete range"}
                  </button>
                </div>
                <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 6 }}>
                  Tip: set both dates to the same day to delete a single day's footage.
                </div>
              </div>

              <div className={styles.storageActions}>
                {storageInfo.orphaned_clips > 0 && (
                  <button className={styles.ghostBtn} onClick={purgeOrphans} disabled={purgingOrphans}
                    style={{ padding: "4px 12px", fontSize: 12, color: "var(--warn, var(--status-warn))", borderColor: "var(--warn, var(--status-warn))" }}>
                    <Trash2 size={13} /> {purgingOrphans ? "Purging…" : `Purge ${storageInfo.orphaned_clips} orphaned`}
                  </button>
                )}
                <button className={styles.dangerBtn} onClick={clearAll} disabled={clearingAll}>
                  <Trash2 size={13} /> {clearingAll ? "Clearing…" : "Clear All Footage"}
                </button>
              </div>
            </>
          ) : (
            <div className={styles.storageEmpty}>
              {loadingStorage ? "Reading storage…" : "Storage usage unavailable"}
            </div>
          )}
        </div>

        {/* ── Channels ──────────────────────────────────────────────────── */}
        <div id="s-telegram" className={styles.section}>
          <div className={styles.sectionHeader}>
            <span className={styles.sectionTitle}>
              <Send size={12} /> Telegram
            </span>
            {!!(form.telegram_bot_token && form.telegram_chat_id) && (
              <span style={{ fontSize: 10, fontWeight: 700, color: "#0088cc",
                background: "rgba(0,136,204,0.14)", borderRadius: 20, padding: "2px 8px",
                border: "1px solid rgba(0,136,204,0.4)" }}>
                Chat {form.telegram_chat_id}
              </span>
            )}
          </div>
          <TelegramConfig form={form} patch={patch} showToast={showToast} />
        </div>

        {/* Alerts — what the agent sends and when. Mirrors the Telegram /menu
         * (both write the same settings) so the two stay in lockstep. Per-camera
         * muting lives in the Telegram /menu (🎥 Per-camera). */}
        <div id="s-notifications" className={styles.section}>
          <div className={styles.sectionHeader}>
            <span className={styles.sectionTitle}><Bell size={12} /> Alerts</span>
            <span style={{ fontSize: 10, color: "var(--text-muted)" }}>Also in Telegram /menu</span>
          </div>
          <Field label="Alert level" more="Lowest risk that triggers a push. Off silences everything; Critical only the highest; All sends every confirmed event.">
            <select value={form.alert_min_risk ?? "suspicious"}
              onChange={e => patch("alert_min_risk", e.target.value)}
              className={styles.numInput} style={{ width: 170 }}>
              <option value="off">Off — silence alerts</option>
              <option value="critical">Critical only</option>
              <option value="suspicious">Suspicious + Critical</option>
              <option value="normal">All events</option>
            </select>
          </Field>
          <Field label="Attach snapshot" more="Carry the event thumbnail inline in the alert.">
            <Toggle checked={form.attach_snapshot_to_alerts ?? true}
              onChange={v => patch("attach_snapshot_to_alerts", v)} />
          </Field>
          <Field label="Attach clip" more="Carry the recorded MP4 inline (Telegram ≤50 MB, capped to ~2 min).">
            <Toggle checked={form.attach_clip_to_alerts ?? true}
              onChange={v => patch("attach_clip_to_alerts", v)} />
          </Field>
          <Field label="Alert me about" hint="Muted categories still record" more="Muted categories still record and analyze — you just aren't messaged. Same toggles as Telegram's 👁 Alert filter.">
            <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
              {([
                ["person",  "👤 People"],
                ["vehicle", "🚗 Vehicles"],
                ["animal",  "🐾 Animals"],
                ["audio",   "🔊 Sounds"],
                ["other",   "📦 Other"],
              ] as const).map(([key, label]) => {
                const muted = (form.alert_muted_categories ?? []).includes(key);
                return (
                  <button key={key} type="button"
                    onClick={() => {
                      const cur = form.alert_muted_categories ?? [];
                      patch("alert_muted_categories",
                        muted ? cur.filter(c => c !== key) : [...cur, key]);
                    }}
                    title={muted ? "Muted — click to alert again" : "Alerting — click to mute"}
                    style={{
                      fontSize: 11, fontWeight: 700, padding: "5px 12px", borderRadius: 999,
                      cursor: "pointer",
                      border: `1px solid ${muted ? "var(--border)" : "var(--accent)"}`,
                      background: muted ? "transparent" : "var(--hl)",
                      color: muted ? "var(--text-muted)" : "var(--accent)",
                      textDecoration: muted ? "line-through" : "none",
                    }}>
                    {label}
                  </button>
                );
              })}
            </div>
          </Field>
          <Field label="Quiet hours" more="When enabled, only critical alerts are pushed during the window below.">
            <Toggle checked={!!form.quiet_hours_enabled} onChange={v => patch("quiet_hours_enabled", v)} />
          </Field>
          {form.quiet_hours_enabled && (
            <>
              <Field label="From">
                <input type="time" value={form.quiet_hours_start ?? "22:00"}
                  onChange={e => patch("quiet_hours_start", e.target.value)}
                  className={styles.numInput} style={{ width: 100 }} />
              </Field>
              <Field label="To" hint="Earlier than From wraps midnight" more="If 'To' is earlier than 'From' the window wraps midnight.">
                <input type="time" value={form.quiet_hours_end ?? "07:00"}
                  onChange={e => patch("quiet_hours_end", e.target.value)}
                  className={styles.numInput} style={{ width: 100 }} />
              </Field>
            </>
          )}
          {showAdvanced && (<>
            <Field label="Min frames per clip" more="Floor for the adaptive frame picker. Longer clips get more frames automatically; this guarantees a minimum.">
              <input type="number" min="1" max="36" step="1"
                value={form.strobe_frames ?? 4}
                onChange={e => patch("strobe_frames", parseInt(e.target.value, 10) || 4)}
                className={styles.numInput} style={{ width: 70 }} />
            </Field>
            <Field label="Strobe profile" more="Picks the duration→frame-count table. Aggressive feeds the VLM more frames on long clips (best for cloud models); Conservative keeps it cheap (best for small local VLMs).">
              <select value={form.strobe_profile ?? "balanced"}
                onChange={e => patch("strobe_profile", e.target.value as any)}
                className={styles.numInput} style={{ width: 140 }}>
                <option value="conservative">Conservative</option>
                <option value="balanced">Balanced (recommended)</option>
                <option value="aggressive">Aggressive</option>
              </select>
            </Field>
          </>)}
        </div>

        {/* Sharing & tunnel — the on-demand Tailscale Funnel + share-link
            controls. Alert behavior (level, quiet hours, snapshot/clip) lives in
            the Alerts section above. */}
        <div id="s-channels" className={styles.section}>
          <div className={styles.sectionHeader}>
            <span className={styles.sectionTitle}><Send size={12} /> Sharing & tunnel</span>
          </div>

          <Field label="Default share-link expiry" more="How long a 'Share Live View' / 'Share Clip' URL stays valid. Shorter = safer; longer = more convenient.">
            <select value={form.live_share_default_minutes ?? 30}
              onChange={e => patch("live_share_default_minutes", parseInt(e.target.value, 10) || 30)}
              className={styles.numInput} style={{ width: 180 }}>
              <option value={15}>15 minutes</option>
              <option value={30}>30 minutes (default)</option>
              <option value={60}>1 hour</option>
              <option value={1440}>24 hours</option>
              <option value={0}>Until app restart</option>
            </select>
          </Field>

          {/* "Auto-stop tunnel when idle" lived here. `tunnel_auto_stop` has ZERO
              read sites in the Rust tree — state.rs declares it and nothing else
              mentions it, while the auto-stop loop in share_cmds.rs runs
              unconditionally and never consults the flag. A security control that
              promised a "minimal remote-exposure window" and controlled nothing. */}

          <div style={{
            marginTop: 12, padding: "10px 12px", borderRadius: 8,
            background: "rgba(255,193,7,0.07)",
            border: "1px solid rgba(255,193,7,0.20)",
            fontSize: 11, color: "var(--text-muted)", lineHeight: 1.55,
          }}>
            <strong style={{ color: "var(--text-secondary)" }}>Privacy note.</strong>{" "}
            Alerts sent through Telegram pass through Telegram's servers, and bot
            traffic isn't end-to-end encrypted. Share links are served
            over your own Tailscale Funnel: TLS terminates at Tailscale's relay, so they can see the bytes in
            transit but don't cache or store them.
            Whatever channel you use, you remain the data controller — if you record people who haven't consented,
            that's on you, not the app.
          </div>
        </div>

        {/* Detection — YOLO 2026 + ALPR tuning. Cookbook is the guided picker;
            this section is for power users who want all knobs in one place. */}
        <div id="s-detection" className={styles.section}>
          <div className={styles.sectionHeader}>
            <span className={styles.sectionTitle}><Bot size={12} /> Detection</span>
          </div>
          {showAdvanced && (<>
            {/* YOLO confidence + class filter lived here. Arsenal owns them now:
                its class-group CHIPS ("Person / Vehicle / Animal / Package") beat
                a free-text comma-separated list of COCO class names that the user
                has to know by heart, and it sits beside the detector it configures. */}
          </>)}
            {/* The ALPR region select lived here. Arsenal's version gates on which
                region packs are actually installed; this one happily picked one
                that was not, and the reader silently fell back. */}
          <Field label="Known plates" more="One PLATE=Name per line (e.g. ABC1234=Mom's car). A recognised plate that matches (exact or within 1 character) is labelled with the name on the event. The raw plate stays searchable.">
            <textarea
              value={form.known_plates ?? ""}
              onChange={e => patch("known_plates", e.target.value)}
              placeholder={"ABC1234=Mom's car\nXYZ789=Delivery van"}
              rows={3}
              className={styles.numInput}
              style={{ width: 240, fontFamily: "var(--font-mono)", fontSize: 11, resize: "vertical" }} />
          </Field>
          <Field label="Audio detection" hint="Needs the YAMNet skill and RTSP audio" more="Detect scream / glass / alarm / gunshot / bark from camera audio. Requires the YAMNet audio skill (Arsenal) + an RTSP camera with an audio track (browser/USB cameras don't carry audio yet).">
            <Toggle checked={!!form.audio_detection} onChange={v => patch("audio_detection", v)} />
          </Field>
          {form.audio_detection && (<>
            <Field label="Listen for" more="Comma-separated sound types, matched against the YAMNet/AudioSet class names (e.g. scream, glass, alarm, gunshot, bark).">
              <input type="text"
                value={form.audio_listen ?? ""}
                onChange={e => patch("audio_listen", e.target.value)}
                placeholder="scream,glass,alarm,gunshot,bark"
                className={styles.numInput} style={{ width: 240 }} />
            </Field>
            <Field label="Audio sensitivity" more="Per-class confidence (0–1) required to fire. Lower = more sensitive (more false alarms). 0.30 default.">
              <input type="number" min="0.05" max="0.95" step="0.05"
                value={form.audio_threshold ?? 0.30}
                onChange={e => patch("audio_threshold", parseFloat(e.target.value) || 0.30)}
                className={styles.numInput} style={{ width: 70 }} />
            </Field>
          </>)}
          {/* "Face liveness (anti-spoofing)" lived here, shipped with the hint
              "Model not bundled — no effect yet". It is read by face.rs, but
              liveness.rs fail-opens because no MiniFASNet ONNX ships, so the
              control has never done anything. Bring it back when the weights do. */}
        </div>

        {/* Face recognition — model tier + thresholds (s-face anchor target) */}
        <FaceRecognitionSection form={form} patch={patch} showAdvanced={showAdvanced} />

        {/* People — re-identification (matching the same person across events).
            One knob, and it is Advanced-only, so the whole section is — a lone
            heading with an empty body is worse than no section at all. */}
        {showAdvanced && (
        <div id="s-reid" className={styles.section}>
          <div className={styles.sectionHeader}>
            <span className={styles.sectionTitle}><User size={12} /> People — Re-identification</span>
          </div>
          {/* "Auto-recognize people" lived here. `auto_reid` has ZERO read sites in
              Rust: inference.rs runs body Re-ID unconditionally and reads only
              `reid_threshold`. The toggle's single consumer was an "In use" pill in
              Arsenal, so turning it off changed a badge and nothing else — the app
              carried on creating "Person 1"/"Person 2" either way. A privacy
              promise the code never kept is worse than no toggle. */}
          <Field label="Match threshold">
            <input type="number" min="0.1" max="0.9" step="0.05"
              value={form.reid_threshold}
              onChange={e => patch("reid_threshold", parseFloat(e.target.value) || 0.5)}
              className={styles.numInput} style={{ width: 70 }} />
          </Field>
        </div>
        )}


        {/* Security & Login */}
        <SecuritySection showToast={showToast} />

      </div>
    </div>
  );
}

/**
 * One setting row.
 *
 * `hint` is the OPERATIONAL line — the unit, the default, the one thing you need
 * to set the value. It stays on screen. `more` is the rationale and the caveats;
 * it lives in the tooltip, costing zero pixels until asked for.
 *
 * The split exists because all 47 fields used to render their full explanation
 * permanently, up to 263 characters each, which turned Settings into a wall of
 * prose you had to read past to find the control. Nothing was deleted in making
 * this change — the long text moved to `more`.
 */
function Field({ label, hint, more, children }: {
  label: string; hint?: string; more?: string; children: React.ReactNode;
}) {
  return (
    <div className={styles.field} title={more}>
      <div className={styles.fieldLabel}>
        <span className={styles.fieldName}>{label}</span>
        {hint && <span className={styles.fieldHint}>{hint}</span>}
      </div>
      <div className={styles.fieldControl}>{children}</div>
    </div>
  );
}

// ── Security & Login — self-contained (uses the dedicated auth commands, not the
//    settings form, since password/2FA/recovery have their own secure endpoints).
function SecuritySection({ showToast }: { showToast: (msg: string, type: "success"|"error"|"info") => void }) {
  const [st, setSt] = useState<AuthStatus | null>(null);
  const [curPw, setCurPw] = useState("");
  const [newPw, setNewPw] = useState("");
  const [busy, setBusy] = useState(false);
  const refresh = useCallback(() => { api.authStatus().then(setSt).catch(() => {}); }, []);
  useEffect(() => { refresh(); }, [refresh]);
  if (!st) return null;

  const guard = async (fn: () => Promise<unknown>, ok: string) => {
    setBusy(true);
    try { await fn(); showToast(ok, "success"); refresh(); }
    catch (e) { showToast(String(e).replace(/^.*Error:\s*/, ""), "error"); }
    finally { setBusy(false); }
  };

  const savePassword = () => {
    if (newPw.length < 12) { showToast("Password must be at least 12 characters.", "error"); return; }
    guard(() => api.setLoginPassword(newPw, st.has_password ? curPw : undefined), "Password saved")
      .then(() => { setCurPw(""); setNewPw(""); });
  };

  return (
    <div id="s-security" className={styles.section}>
      <div className={styles.sectionHeader}>
        <span className={styles.sectionTitle}><Shield size={12} /> Security & Login</span>
      </div>

      {!st.telegram_ready && (
        <div style={{ fontSize: 11.5, color: "var(--accent-amber)", background: "color-mix(in srgb, var(--status-warn) 10%, transparent)",
          borderRadius: 10, padding: "9px 12px", margin: "0 0 12px", lineHeight: 1.5 }}>
          Set up <strong>Telegram</strong> in <strong>Channels</strong> first — login needs it for recovery.
        </div>
      )}

      <Field label={st.has_password ? "Change password" : "Set a login password"} more="Argon2id-hashed, encrypted at rest. Minimum 12 characters.">
        <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
          {st.has_password && (
            <input type="password" value={curPw} placeholder="Current password" className={styles.numInput}
              style={{ width: "100%", textAlign: "left" }} onChange={e => setCurPw(e.target.value)} />
          )}
          <input type="password" value={newPw} placeholder="New password (min 12)" className={styles.numInput}
            style={{ width: "100%", textAlign: "left" }} onChange={e => setNewPw(e.target.value)} />
          <button className={styles.ghostBtn} disabled={busy || !st.telegram_ready || !newPw}
            style={{ padding: "6px 14px", fontSize: 12, alignSelf: "flex-start" }} onClick={savePassword}>
            <Lock size={12} /> {st.has_password ? "Update password" : "Set password"}
          </button>
        </div>
      </Field>

      <Field label="Require login on this device" hint="Needs a password + Telegram">
        <Toggle checked={st.login_required}
          onChange={v => guard(() => api.setLoginRequired(v), v ? "Login enabled" : "Login disabled")} />
      </Field>

      <Field label="Two-factor (Telegram code)" more="After the password, require a 6-digit code sent to your Telegram.">
        <Toggle checked={st.twofa_enabled}
          onChange={v => guard(() => api.set2faEnabled(v), v ? "2FA enabled" : "2FA disabled")} />
      </Field>

      <Field label="Remember this device"
        hint={`Stays unlocked for ${st.remember_days} days`}>
        <Toggle checked={st.remember_enabled}
          onChange={v => guard(() => api.setRememberDevice(v, st.remember_days), v ? "Remember enabled" : "Remember disabled")} />
      </Field>

      {st.telegram_ready && st.has_password && (
        <button className={styles.ghostBtn} disabled={busy} style={{ padding: "6px 14px", fontSize: 12, marginTop: 4 }}
          onClick={() => guard(() => api.requestRecovery(), "Recovery code sent to Telegram")}>
          <KeyRound size={12} /> Send a test recovery code
        </button>
      )}

      <div style={{ fontSize: 10.5, color: "var(--text-muted)", marginTop: 12, lineHeight: 1.5 }}>
        Stops casual access — not someone with full control of this computer.
      </div>
    </div>
  );
}

function Toggle({ checked, onChange }: { checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <button
      className={`${styles.toggle} ${checked ? styles.toggleOn : ""}`}
      onClick={() => onChange(!checked)}
      role="switch"
      aria-checked={checked}
    >
      <span className={styles.toggleThumb} />
    </button>
  );
}
