/**
 * Arsenal — the on-device model marketplace.
 *
 * One screen, one scroll. Top-to-bottom:
 *
 *   1. GPU acceleration — which execution provider is armed, and why the others
 *                         are not (see `accel_report`). The one hardware ACTION
 *                         (installing the NVIDIA pack) lives here.
 *   2. Model Station    — provider switcher, API keys, on-device model install.
 *   3. Model cards      — Face / YOLO / ALPR / Search / Depth / Enhancements,
 *                         each with Use / Install / Remove driven by what is
 *                         actually on disk (`list_installed_skills`).
 *
 * The old "hardware-aware picks" layer (llmfit-core scoring, a hardware stat
 * card, and per-card "recommended because…" reasons) was REMOVED: llmfit-core
 * itself had already been dropped from the backend, so the panel was advertising
 * a scoring engine that no longer existed, and the picks were noise next to the
 * plain install/remove actions people actually use.
 */

import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  Zap, Download, Check, RotateCcw, Bot, ChevronRight, Trash2,
} from "lucide-react";
import {
  api, SkillStatus, Settings, TrtxStatus, AccelRow,
} from "../../api";
import { useStore } from "../../store";
import { useShallow } from "zustand/react/shallow";
import {
  SkillDef, SkillId, SKILL_REGISTRY, downloadSkill, findSkill,
  DEPTH_PRESETS, depthModelSkill,
} from "./skillDownload";
import { ModelStation } from "./ModelStation";

type InstallState = "not_installed" | "downloading" | "installed" | "error";

// Per-card remove — the "delete this model" action lives right on each card
// now, not only in the list at the bottom. Context-fed so we don't thread
// onRemove/removing/skills through all six card components.
const RemoveCtx = React.createContext<{
  skills:   SkillStatus[];
  removing: Record<string, boolean>;
  onRemove: (sk: SkillStatus) => void;
} | null>(null);

/// Trash button for a skill row — renders nothing until that skill is
/// installed, so it's safe to drop into every card row unconditionally.
function RemoveButton({ skillId }: { skillId: string }) {
  const ctx = React.useContext(RemoveCtx);
  const sk  = ctx?.skills.find(s => s.id === skillId && s.installed);
  if (!ctx || !sk) return null;
  const busy = !!ctx.removing[skillId];
  return (
    <button onClick={() => ctx.onRemove(sk)} disabled={busy}
      title="Remove this model from disk — reinstall anytime, your data is never touched"
      style={{
        display: "inline-flex", alignItems: "center", justifyContent: "center",
        width: 26, height: 26, borderRadius: 999, flexShrink: 0,
        border: "1px solid var(--border-strong)", background: "transparent",
        color: busy ? "var(--text-muted)" : "var(--accent-red)",
        cursor: busy ? "default" : "pointer", opacity: busy ? 0.6 : 1,
      }}>
      <Trash2 size={12} />
    </button>
  );
}

export function Arsenal() {
  const {
    settings, setSettings, showToast,
    // Download progress lives in the store, NOT here: this panel is unmounted
    // whenever the user leaves the tab (App.tsx mounts panels active-only) while
    // the Rust download keeps streaming. Component-local state meant returning to
    // Arsenal showed "Install" on a job already in flight.
    progress, setSkillProgress, downloading, setSkillDownloading,
  } = useStore(useShallow(s => ({
    settings: s.settings, setSettings: s.setSettings, showToast: s.showToast,
    progress: s.skillProgress, setSkillProgress: s.setSkillProgress,
    downloading: s.skillDownloading, setSkillDownloading: s.setSkillDownloading,
  })));
  const [skills, setSkills] = useState<SkillStatus[]>([]);
  const [fsState, setState] = useState<Record<string, InstallState>>({});
  /** Filesystem scan, overlaid with whatever is downloading right now.
   *  Derived rather than stored so the six cards keep their existing
   *  `state` prop and none of them needs to know where it comes from. */
  const state = useMemo<Record<string, InstallState>>(() => {
    const merged = { ...fsState };
    for (const [id, on] of Object.entries(downloading)) {
      if (on) merged[id] = "downloading";
    }
    return merged;
  }, [fsState, downloading]);
  const [loading, setLoading] = useState(false);
  /** Per-skill uninstall-in-flight flag, so the Remove button can show "Removing…". */
  const [removing, setRemoving] = useState<Record<string, boolean>>({});

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const s = await api.listInstalledSkills().catch(() => [] as SkillStatus[]);
      setSkills(s);
      // Hydrate install state from the filesystem scan.
      const next: Record<string, InstallState> = {};
      for (const sk of s) next[sk.id] = sk.installed ? "installed" : "not_installed";
      setState(prev => ({ ...prev, ...next }));
    } finally {
      setLoading(false);
    }
  }, []);
  useEffect(() => { refresh(); }, [refresh]);

  const handleInstall = async (def: SkillDef) => {
    setSkillDownloading(def.id, true);
    setSkillProgress(def.id, { pct: 0 });
    try {
      await downloadSkill(def, (pct, downloaded, total) => {
        setSkillProgress(def.id, { pct, downloaded, total });
      });
      setSkillDownloading(def.id, false);
      setState(p => ({ ...p, [def.id]: "installed" }));
      showToast(`${def.name} installed`, "success");
      refresh();
    } catch (e: any) {
      setSkillDownloading(def.id, false);
      setState(p => ({ ...p, [def.id]: "error" }));
      showToast(`Install failed: ${e}`, "error");
    }
  };

  // Which installed skills are CURRENTLY selected/active per settings — drives the
  // "In use" tag + a stronger removal warning. A soft signal: the confirm dialog is
  // the real safety net, and `remove_skill` only ever deletes the downloaded model
  // file under `skills/<id>/` — never any DB data (enrolled people, footage, events).
  const activeSkillIds = useMemo(() => {
    const s = new Set<string>();
    if (settings?.face_model === "small") s.add("face_small");
    if (settings?.face_model === "large") s.add("face_large");
    const yMap: Record<string, string> = {
      nano: "yolo26n", small: "yolo26s", medium: "yolo26m", large: "yolo26l", xlarge: "yolo26x",
    };
    s.add(yMap[settings?.yolo_variant ?? "xlarge"] ?? "yolo26x");
    if (settings?.alpr_region) s.add(`alpr_${settings.alpr_region}`);
    if (settings?.search_model && settings.search_model !== "off") s.add(settings.search_model);
    if (settings?.audio_detection) s.add("audio_yamnet");
    // On-device LLM. `""` means on-device too (settings v3/v4 default), so treat a
    // missing provider as local — otherwise the model serving chat right now would
    // show as unused and get the weaker removal warning.
    if ((settings?.ai_provider ?? "local") === "local" || settings?.ai_provider === "") s.add("local_llm");
    if (settings?.auto_reid) s.add("reid_osnet");
    // Depth model is "in use" whenever any camera has anonymization enabled.
    try {
      const anon = JSON.parse(settings?.depth_anonymize || "{}");
      if (Object.values(anon).some(v => v === true)) s.add("depth_anything");
    } catch { /* ignore */ }
    return s;
  }, [settings]);

  const handleRemove = async (sk: SkillStatus) => {
    const inUse = activeSkillIds.has(sk.id);
    const freed = sk.size_on_disk_mb >= 1024
      ? `${(sk.size_on_disk_mb / 1024).toFixed(1)} GB`
      : `${sk.size_on_disk_mb} MB`;
    // Reassure the user: this is reversible (reinstall) and never touches their data.
    const msg = inUse
      ? `${sk.name} is currently in use.\n\nRemoving it frees ${freed}, and that feature stops working until you reinstall it from the cards above.\n\nYour data — enrolled people, footage and events — is NOT deleted.\n\nRemove it?`
      : `Remove ${sk.name}?\n\nFrees ${freed}. Reinstall anytime from the cards above. None of your data is touched.`;
    if (!window.confirm(msg)) return;
    setRemoving(p => ({ ...p, [sk.id]: true }));
    try {
      await api.removeSkill(sk.id);
      showToast(`${sk.name} removed · ${freed} freed`, "info");
      await refresh();
    } catch (e: any) {
      showToast(`Couldn't remove ${sk.name}: ${e}`, "error");
    } finally {
      setRemoving(p => ({ ...p, [sk.id]: false }));
    }
  };

  const handleSwitchFace = async (tier: "off" | "small" | "large") => {
    if (!settings) return;
    const next: Settings = { ...settings, face_model: tier };
    await api.saveSettings(next);
    setSettings(next);
    showToast(`Face tier set to ${tier}`, "success");
  };

  const handleSwitchYolo = async (tier: "nano" | "small" | "medium" | "large" | "xlarge") => {
    if (!settings) return;
    const next: Settings = { ...settings, yolo_variant: tier };
    await api.saveSettings(next);
    setSettings(next);
    showToast(`YOLO tier set to ${tier}`, "success");
  };

  /// Pick which on-device model runs. Commits the tier AND the provider together:
  /// choosing a local model IS the intent to use it, so leaving `ai_provider` on
  /// a cloud endpoint would show "In use" beside a model that answers nothing.
  const handleSwitchLlm = async (tier: "fast" | "balanced" | "vision") => {
    if (!settings) return;
    const next: Settings = {
      ...settings, local_llm_tier: tier,
      ai_provider: "local", vision_model: "", agent_enabled: true,
    };
    await api.saveSettings(next);
    setSettings(next);
    showToast(`On-device AI set to ${tier}`, "success");
  };

  const handleSwitchAlpr = async (region: "global" | "european" | "argentinian") => {
    if (!settings) return;
    const next: Settings = { ...settings, alpr_region: region };
    await api.saveSettings(next);
    setSettings(next);
    showToast(`ALPR region set to ${region}`, "success");
  };

  const handleSwitchSearch = async (model: "off" | "mobileclip_s0" | "clip_b32" | "jina_clip") => {
    if (!settings) return;
    const next: Settings = { ...settings, search_model: model };
    await api.saveSettings(next);
    setSettings(next);
    if (model === "off") { showToast("Semantic search disabled", "info"); return; }
    // Backfill embeddings for the newly-active model so existing events become
    // semantically searchable (runs in the background server-side).
    //
    // ONE toast, reported after the count is known. This used to fire an
    // immediate "indexing history…" and then a second toast with the count —
    // two messages saying the same thing, and if the reindex returned inside
    // 3.5 s the second one inherited the first's dying timer.
    try {
      const n = await api.reindexSemanticSearch();
      showToast(n > 0
        ? `Search model activated — indexing ${n} past events`
        : "Search model activated", "success");
    } catch {
      showToast("Search model activated", "success"); // keyword search works regardless
    }
  };

  // Persist an arbitrary settings patch (used by the Advanced disclosures
  // on the Face / YOLO cards). Debounced via React state — saves on every
  // change because the slider events are cheap (no `await` race).
  const handlePatchSettings = async (patch: Partial<Settings>) => {
    if (!settings) return;
    const next: Settings = { ...settings, ...patch };
    setSettings(next);              // optimistic UI
    await api.saveSettings(next);
  };

  return (
    <div style={{
      flex: 1, overflow: "auto",
      padding: "20px 22px 60px",
      display: "flex", flexDirection: "column", gap: 18,
    }}>
      <Header onRefresh={refresh} spinning={loading} />
      {/* Telemetry lives in the sidebar Activity card now — Arsenal is actions
        * only. The one hardware ACTION (TensorRT acceleration) stays here. */}
      <AcceleratorRow />
      {/* Full-width: provider switcher, API keys, model list, pull progress. */}
      <ModelStation />
      <RemoveCtx.Provider value={{ skills, removing, onRemove: handleRemove }}>
        <RecommendationGrid
          skills={skills}
          state={state}
          progress={progress}
          settings={settings}
          currentFace={settings?.face_model ?? "off"}
          currentYolo={settings?.yolo_variant ?? "xlarge"}
          currentAlprRegion={settings?.alpr_region ?? "global"}
          currentSearchModel={settings?.search_model ?? "off"}
          onInstall={handleInstall}
          onSwitchFace={handleSwitchFace}
          onSwitchYolo={handleSwitchYolo}
          onSwitchAlpr={handleSwitchAlpr}
          onSwitchSearch={handleSwitchSearch}
          onSwitchLlm={handleSwitchLlm}
          onPatchSettings={handlePatchSettings}
          onRefresh={refresh}
        />
      </RemoveCtx.Provider>
      <IdentityFooter
        settings={settings}
        onSave={async (patch) => {
          if (!settings) return;
          const next: Settings = { ...settings, ...patch };
          await api.saveSettings(next);
          setSettings(next);
          showToast("Saved", "success");
        }}
      />
    </div>
  );
}

// ── Accelerator row — the TensorRT ACTION from the old telemetry card. Live
//    metrics/latency moved to the sidebar Activity popover; this row only
//    exists for NVIDIA hosts that haven't activated an accelerated EP yet
//    (or to show the active badge). Renders nothing on unsupported hardware.
function AcceleratorRow() {
  const [trtx, setTrtx] = useState<TrtxStatus | null>(null);
  const [rows, setRows] = useState<AccelRow[]>([]);
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<string | null>(null);
  const refresh = useCallback(() => {
    api.trtxStatus().then(setTrtx).catch(() => {});
    api.accelReport().then(setRows).catch(() => {});
  }, []);
  useEffect(() => { refresh(); }, [refresh]);
  // NOTE: this used to bail out unless an NVIDIA pack was supported, so every
  // AMD / Intel / Apple user saw nothing at all about their own accelerator —
  // the one screen that answers "is my GPU being used?" was NVIDIA-only.
  const armed = rows.find(r => r.state === "armed");

  const installPack = async () => {
    if (busy) return;
    setBusy(true);
    setNote("Downloading TensorRT 10.9 + CUDA 12 + cuDNN 9 runtimes (~1.85 GB, one-time, SHA-verified)…");
    try {
      const st = await api.installTrtxPack();
      setTrtx(st);
      setNote(`Pack active (${st.active_ep}). First model load builds engines (minutes, once, cached). Restart the app to switch every model.`);
    } catch (e) { setNote(`${e}`); } finally { setBusy(false); refresh(); }
  };
  const importSdk = async () => {
    if (busy) return;
    try {
      const { open } = await import("@tauri-apps/plugin-dialog");
      const picked = await open({
        title: `Select the downloaded TensorRT-for-RTX ${trtx?.trtx_required_runtime || ""} SDK (.zip or extracted folder)`.replace("  ", " "),
        filters: [{ name: "SDK", extensions: ["zip"] }],
      });
      if (!picked || typeof picked !== "string") return;
      setBusy(true);
      setNote("Importing SDK + running activation canary…");
      const st = await api.importTrtxSdk(picked);
      setTrtx(st);
      setNote("TensorRT-RTX active — concurrent GPU inference enabled. Restart the app to switch every model.");
    } catch (e) { setNote(`${e}`); } finally { setBusy(false); refresh(); }
  };

  const epBadge: Record<string, string> = {
    nvrtx: "TensorRT-RTX active", tensorrt: "TensorRT active", cuda: "CUDA active",
  };
  return (
    <div style={{ background: "var(--bg-surface)", border: "1px solid var(--border)", borderRadius: "var(--radius-lg)", padding: "12px 16px", display: "flex", flexDirection: "column", gap: 8 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
        <Zap size={13} style={{ color: "#76b900" }} />
        <span style={{ fontSize: 12.5, fontWeight: 800 }}>GPU acceleration</span>
        {trtx?.active ? (
          <span style={{ fontSize: 10, fontWeight: 700, padding: "2px 8px", borderRadius: 20, background: "rgba(118,185,0,0.14)", color: "#76b900", border: "1px solid rgba(118,185,0,0.35)" }}>
            {epBadge[trtx.active_ep] ?? "NVIDIA EP active"}
          </span>
        ) : armed ? (
          <span style={{ fontSize: 10, fontWeight: 700, padding: "2px 8px", borderRadius: 20, background: "var(--hl)", color: "var(--text-secondary)", border: "1px solid var(--hl-edge)" }}>
            {armed.ep} active
          </span>
        ) : null}
        {trtx?.supported && !trtx.active && (
          <>
            <button onClick={installPack} disabled={busy}
              title="Download NVIDIA TensorRT 10.9 + CUDA runtimes (~1.85 GB, one-time, SHA-pinned). Runs YOLO several times faster than DirectML."
              style={{ fontSize: 10, fontWeight: 700, padding: "3px 10px", borderRadius: 20, cursor: busy ? "wait" : "pointer", background: "rgba(118,185,0,0.14)", color: "#76b900", border: "1px solid rgba(118,185,0,0.35)" }}>
              {busy ? "Installing…" : "⚡ Install TensorRT Pack (~1.9 GB)"}
            </button>
            <button onClick={importSdk} disabled={busy}
              title={`TensorRT-for-RTX is lighter (~200 MB), JIT-fast and allows concurrent GPU inference. This build links runtime ${trtx?.trtx_required_runtime || "?"}, which NVIDIA ships only in the SDK (developer.nvidia.com) — the public wheels are a different version.`}
              style={{ fontSize: 10, fontWeight: 700, padding: "3px 10px", borderRadius: 20, cursor: busy ? "wait" : "pointer", background: "transparent", color: "#76b900", border: "1px dashed rgba(118,185,0,0.45)" }}>
              Import TensorRT-RTX SDK…
            </button>
          </>
        )}
      </div>
      {/* Per-provider state + the concrete reason. This is what makes "does my
          device use its GPU?" answerable on any machine, instead of a silent
          fall-through nobody could see. */}
      {rows.length > 0 && (
        <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
          {rows.map(r => {
            const color = r.state === "armed" ? "var(--accent)"
              : r.state === "ready" ? "var(--text-secondary)"
              : r.state === "unavailable" ? "var(--status-warn)" : "var(--text-muted)";
            return (
              <div key={r.ep} style={{ display: "flex", alignItems: "baseline", gap: 8, fontSize: 10.5,
                opacity: r.state === "n/a" ? 0.55 : 1 }}>
                <span style={{ width: 5, height: 5, borderRadius: 999, background: color, flexShrink: 0, transform: "translateY(-1px)" }} />
                <span style={{ fontWeight: 700, minWidth: 92, color: "var(--text-secondary)" }}>{r.ep}</span>
                <span style={{ color, minWidth: 68, fontWeight: 600 }}>{r.state}</span>
                <span style={{ color: "var(--text-muted)", flex: 1 }}>{r.detail}</span>
              </div>
            );
          })}
        </div>
      )}
      {note && <div style={{ fontSize: 10.5, color: "var(--text-muted)" }}>{note}</div>}
    </div>
  );
}

// ── Header ───────────────────────────────────────────────────────────────────

function Header({ onRefresh, spinning }: { onRefresh: () => void; spinning: boolean }) {
  return (
    <div style={{ display: "flex", alignItems: "baseline", gap: 12 }}>
      <span style={{ fontWeight: 800, fontSize: 18, letterSpacing: -0.025 }}>Arsenal</span>
      <div style={{ flex: 1 }} />
      <button onClick={onRefresh}
        title="Re-scan installed models"
        style={{
          display: "inline-flex", alignItems: "center", gap: 6,
          padding: "5px 10px", borderRadius: 999,
          border: "1px solid var(--border)", background: "transparent",
          color: "var(--text-secondary)", cursor: "pointer", fontSize: 11, fontWeight: 600,
        }}>
        <RotateCcw size={12} className={spinning ? "spin" : ""} /> Re-scan
      </button>
    </div>
  );
}

// ── Recommendation grid ──────────────────────────────────────────────────────

// ── Depth Anonymization card — install once (FP16 ~50MB), pick a SPEED ───────
// The model is a single FP16 file; the preset only changes inference RESOLUTION
// (the real efficiency lever — the model input is dynamic). Quantized "smaller"
// variants were rejected: smaller download but slower on GPU. Installs into the
// single `depth_anything` skill (uninstall list + per-cam toggle unchanged).

function DepthCard({
  skills, settings, onPatchSettings, onRefresh,
}: {
  skills:          SkillStatus[];
  settings:        Settings | null;
  onPatchSettings: (patch: Partial<Settings>) => void;
  onRefresh:       () => void;
}) {
  const { showToast } = useStore(useShallow(s => ({ showToast: s.showToast })));
  const installed = skills.some(s => s.id === "depth_anything" && s.installed);
  const current = settings?.depth_model ?? "balanced";
  const [dlPct, setDlPct] = useState<number | null>(null);

  const install = async () => {
    if (dlPct != null) return;
    setDlPct(0);
    try {
      await downloadSkill(depthModelSkill(), pct => setDlPct(pct));
      showToast("Depth model installed (FP16, ~50 MB)", "success");
      onRefresh();
    } catch (e: any) {
      showToast(`Depth model install failed: ${e}`, "error");
    } finally { setDlPct(null); }
  };

  return (
    <Card title="Depth anonymization"
      subtitle="Feed becomes a colorized depth map — motion stays visible, identities don't.">
      {!installed ? (
        <button onClick={install} disabled={dlPct != null}
          style={{ fontSize: 11, fontWeight: 700, padding: "7px 14px", borderRadius: 10, width: "100%",
            cursor: dlPct != null ? "wait" : "pointer", background: "color-mix(in srgb, var(--accent) 14%, transparent)",
            color: "var(--accent)", border: "1px solid color-mix(in srgb, var(--accent) 35%, transparent)" }}>
          {dlPct != null ? `Downloading… ${dlPct}%` : "Install depth model (FP16 · ~50 MB)"}
        </button>
      ) : (
        <>
          <div style={{ display: "flex", alignItems: "center", marginBottom: 6 }}>
            <span style={{ fontSize: 10, color: "var(--text-muted)", fontWeight: 700 }}>SPEED</span>
            <div style={{ flex: 1 }} />
            <RemoveButton skillId="depth_anything" />
          </div>
          <div style={{ display: "flex", gap: 6 }}>
            {DEPTH_PRESETS.map(p => {
              const active = current === p.preset;
              return (
                <button key={p.preset} onClick={() => onPatchSettings({ depth_model: p.preset })}
                  title={`${p.res} inference`}
                  style={{ flex: 1, padding: "7px 6px", borderRadius: 9, cursor: "pointer",
                    display: "flex", flexDirection: "column", alignItems: "center", gap: 2,
                    background: active ? "color-mix(in srgb, var(--accent) 10%, transparent)" : "rgb(var(--ink) / 0.02)",
                    border: `1px solid ${active ? "var(--accent)" : "var(--border)"}` }}>
                  <span style={{ fontSize: 11, fontWeight: 700,
                    color: active ? "var(--accent)" : "var(--text-primary)" }}>{p.label}</span>
                  <span style={{ fontSize: 8.5, color: active ? "var(--accent)" : p.badgeColor }}>{p.note}</span>
                </button>
              );
            })}
          </div>
        </>
      )}
      <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 8 }}>
        Enable per-camera with the Anonymize button on the live view.
      </div>
    </Card>
  );
}

function RecommendationGrid({
  skills, state, progress, settings,
  currentFace, currentYolo, currentAlprRegion, currentSearchModel,
  onInstall, onSwitchFace, onSwitchYolo, onSwitchAlpr, onSwitchSearch, onSwitchLlm, onPatchSettings, onRefresh,
}: {
  skills:            SkillStatus[];
  state:             Record<string, InstallState>;
  progress:          Record<string, { pct: number; downloaded?: number; total?: number | null }>;
  settings:          Settings | null;
  currentFace:       string;
  currentYolo:       string;
  currentAlprRegion: string;
  currentSearchModel: string;
  onInstall:         (def: SkillDef) => void;
  onSwitchFace:      (tier: "off" | "small" | "large") => void;
  onSwitchYolo:      (tier: "nano" | "small" | "medium" | "large" | "xlarge") => void;
  onSwitchAlpr:      (region: "global" | "european" | "argentinian") => void;
  onSwitchSearch:    (model: "off" | "mobileclip_s0" | "clip_b32" | "jina_clip") => void;
  onSwitchLlm:       (tier: "fast" | "balanced" | "vision") => void;
  onPatchSettings:   (patch: Partial<Settings>) => void;
  onRefresh:         () => void;
}) {
  return (
    <div style={{
      display: "grid",
      gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
      gap: 14,
    }}>
      {/* First in the grid: it is the model that answers when you type. */}
      <LocalAiCard
        skills={skills}
        state={state}
        progress={progress}
        settings={settings}
        onInstall={onInstall}
        onSwitchLlm={onSwitchLlm}
      />
      <FaceCard
        skills={skills}
        state={state}
        progress={progress}
        settings={settings}
        currentFace={currentFace}
        onInstall={onInstall}
        onSwitchFace={onSwitchFace}
        onPatchSettings={onPatchSettings}
      />
      <YoloCard
        skills={skills}
        state={state}
        progress={progress}
        settings={settings}
        currentYolo={currentYolo}
        onInstall={onInstall}
        onSwitchYolo={onSwitchYolo}
        onPatchSettings={onPatchSettings}
      />
      <AlprCard
        skills={skills}
        state={state}
        progress={progress}
        currentRegion={currentAlprRegion}
        onInstall={onInstall}
        onSwitchAlpr={onSwitchAlpr}
      />
      <SearchCard
        skills={skills}
        state={state}
        progress={progress}
        currentModel={currentSearchModel}
        onInstall={onInstall}
        onSwitchSearch={onSwitchSearch}
      />
      <DepthCard
        skills={skills}
        settings={settings}
        onPatchSettings={onPatchSettings}
        onRefresh={onRefresh}
      />
      <EnhancementsCard
        skills={skills}
        state={state}
        progress={progress}
        settings={settings}
        onInstall={onInstall}
      />
    </div>
  );
}

// ── Enhancements card — install-only skills (audio, deep Re-ID) ──────────────

const ENHANCEMENT_SKILLS = [
  { id: "audio_yamnet", label: "Audio detection", hint: "scream / glass / alarm — enable in Settings" },
  { id: "reid_osnet",   label: "Deep person Re-ID", hint: "OSNet · durable cross-camera tracking" },
] as const;

function EnhancementsCard({
  skills, state, progress, settings, onInstall,
}: {
  skills:    SkillStatus[];
  state:     Record<string, InstallState>;
  progress:  Record<string, { pct: number; downloaded?: number; total?: number | null }>;
  settings:  Settings | null;
  onInstall: (def: SkillDef) => void;
}) {
  const installedSet = new Set(skills.filter(s => s.installed).map(s => s.id));
  // Which enhancement is actively being used right now (drives the "In use" pill).
  const isActive = (id: string) =>
    id === "audio_yamnet" ? !!settings?.audio_detection
  : id === "reid_osnet"   ? !!settings?.auto_reid
  : false;
  return (
    <Card title="Enhancements"
      subtitle="Audio events + deep person Re-ID.">
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {ENHANCEMENT_SKILLS.map(row => {
          const def = findSkill(row.id);
          if (!def) return null;
          const isInstalled   = installedSet.has(row.id);
          const isDownloading = state[row.id] === "downloading";
          const prog          = progress[row.id];
          const pct           = prog?.pct ?? 0;
          const dlBytes       = prog?.downloaded;
          const totBytes      = prog?.total;
          return (
            <div key={row.id} style={{
              display: "flex", alignItems: "center", gap: 8,
              padding: "8px 10px", borderRadius: 10,
              background: isInstalled ? "color-mix(in srgb, var(--accent) 10%, transparent)" : "rgb(var(--ink) / 0.02)",
              border: `1px solid ${isInstalled ? "var(--accent)" : "var(--border)"}`,
            }}>
              <div style={{ minWidth: 130, display: "flex", flexDirection: "column" }}>
                <span style={{ fontWeight: 700, fontSize: 12, color: isInstalled ? "var(--accent)" : "var(--text-primary)" }}>{row.label}</span>
                <span style={{ fontSize: 10, color: "var(--text-muted)" }}>{def.sizeLabel} · {row.hint}</span>
                {/* Licence of the WEIGHTS, at the point of choice. Nothing is
                    bundled or preselected, so installing a model is the user's
                    decision — and a copyleft licence can place real obligations
                    on whatever they build around it. */}
                {def.license && (
                  <span title={def.licenseNote ?? def.license}
                    style={{
                      marginTop: 2, fontSize: 9, fontWeight: 700, letterSpacing: 0.02,
                      color: def.license.startsWith("AGPL") ? "var(--status-warn)" : "var(--text-muted)",
                    }}>
                    {def.license}{def.license.startsWith("AGPL") ? " · commercial use needs a licence" : ""}
                  </span>
                )}
              </div>
              <div style={{ flex: 1 }} />
              <RemoveButton skillId={row.id} />
              {isDownloading ? (
                <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 10, color: "var(--text-secondary)" }}>
                  <Download size={11} />
                  {totBytes && totBytes > 0
                    ? `${pct}% · ${formatBytes(dlBytes)} / ${formatBytes(totBytes)}`
                    : dlBytes ? `${formatBytes(dlBytes)} downloaded…` : "Starting…"}
                </div>
              ) : isInstalled && isActive(row.id) ? (
                <span style={{ display: "inline-flex", alignItems: "center", gap: 4, fontSize: 11, fontWeight: 700, color: "var(--accent)" }}>
                  <Check size={11} /> In use
                </span>
              ) : isInstalled ? (
                <span style={{ display: "inline-flex", alignItems: "center", gap: 4, fontSize: 11, fontWeight: 700, color: "var(--accent)" }}>
                  <Check size={11} /> Installed
                </span>
              ) : (
                <button onClick={() => onInstall(def)}
                  style={{
                    display: "inline-flex", alignItems: "center", gap: 5,
                    padding: "4px 10px", borderRadius: 999,
                    border: "1px solid var(--border-strong)",
                    background: "transparent", color: "var(--text-primary)",
                    fontSize: 11, fontWeight: 700, cursor: "pointer",
                  }}>
                  <Download size={10} /> Install
                </button>
              )}
            </div>
          );
        })}
      </div>
    </Card>
  );
}

// ── ALPR card — three regional tiers (mature NVRs / fast-plate-ocr pattern) ─────

const ALPR_TIERS = [
  { region: "global",       id: "alpr_global",       label: "Global",       hint: "Worldwide default" },
  { region: "european",     id: "alpr_european",     label: "European",     hint: "EU / UK plates" },
  { region: "argentinian",  id: "alpr_argentinian",  label: "Argentinian",  hint: "Mercosur / AR" },
] as const;

function AlprCard({
  skills, state, progress, currentRegion, onInstall, onSwitchAlpr,
}: {
  skills:        SkillStatus[];
  state:         Record<string, InstallState>;
  progress:      Record<string, { pct: number; downloaded?: number; total?: number | null }>;
  currentRegion: string;
  onInstall:     (def: SkillDef) => void;
  onSwitchAlpr:  (region: "global" | "european" | "argentinian") => void;
}) {
  const installedSet = new Set(skills.filter(s => s.installed).map(s => s.id));
  // Legacy v6 install (`skills/alpr/`) is reported as `alpr_global` by the
  // backend, so we don't need a special case here.

  return (
    <Card title="License plate recognition"
      subtitle="Reads plates on cars/motorcycles and tags the event.">
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {ALPR_TIERS.map(row => {
          const def = findSkill(row.id);
          if (!def) return null;
          const isInstalled = installedSet.has(row.id);
          const isActive    = currentRegion === row.region;
          const isDownloading = state[row.id] === "downloading";
          const prog          = progress[row.id];
          const pct           = prog?.pct ?? 0;
          const dlBytes       = prog?.downloaded;
          const totBytes      = prog?.total;
          return (
            <div key={row.id} style={{
              display: "flex", alignItems: "center", gap: 8,
              padding: "8px 10px", borderRadius: 10,
              background: isActive ? "color-mix(in srgb, var(--accent) 10%, transparent)" : "rgb(var(--ink) / 0.02)",
              border: `1px solid ${isActive ? "var(--accent)" : "var(--border)"}`,
            }}>
              <div style={{ minWidth: 100, display: "flex", flexDirection: "column" }}>
                <span style={{
                  fontWeight: 700, fontSize: 12,
                  color: isActive ? "var(--accent)" : "var(--text-primary)",
                }}>{row.label}</span>
                <span style={{ fontSize: 10, color: "var(--text-muted)" }}>{row.hint}</span>
              </div>
              <div style={{ flex: 1 }} />
              <RemoveButton skillId={row.id} />
              {isDownloading ? (
                <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 10, color: "var(--text-secondary)" }}>
                  <Download size={11} />
                  {totBytes && totBytes > 0
                    ? `${pct}% · ${formatBytes(dlBytes)} / ${formatBytes(totBytes)}`
                    : dlBytes
                      ? `${formatBytes(dlBytes)} downloaded…`
                      : "Starting…"}
                </div>
              ) : isInstalled ? (
                isActive ? (
                  <span style={{ display: "inline-flex", alignItems: "center", gap: 4, fontSize: 11, fontWeight: 700, color: "var(--accent)" }}>
                    <Check size={11} /> In use
                  </span>
                ) : (
                  <button onClick={() => onSwitchAlpr(row.region)}
                    style={{
                      padding: "4px 10px", borderRadius: 999,
                      border: "1px solid var(--border-strong)",
                      background: "transparent", color: "var(--text-primary)",
                      fontSize: 11, fontWeight: 700, cursor: "pointer",
                    }}>Use</button>
                )
              ) : (
                <button onClick={() => onInstall(def)}
                  style={{
                    display: "inline-flex", alignItems: "center", gap: 5,
                    padding: "4px 10px", borderRadius: 999,
                    border: "1px solid var(--border-strong)",
                    background: "transparent", color: "var(--text-primary)",
                    fontSize: 11, fontWeight: 700, cursor: "pointer",
                  }}>
                  <Download size={10} /> Install
                </button>
              )}
            </div>
          );
        })}
      </div>
    </Card>
  );
}

// ── Semantic search card — three CLIP tiers (resource-picker, like AlprCard) ──

const SEARCH_TIERS = [
  { id: "mobileclip_s0", label: "MobileCLIP-S0", hint: "207 MB · Apple · low-resource" },
  { id: "clip_b32",      label: "CLIP ViT-B/32", hint: "579 MB · OpenAI · balanced" },
  { id: "jina_clip",     label: "Jina-CLIP-v1",  hint: "850 MB · most accurate" },
] as const;

function SearchCard({
  skills, state, progress, currentModel, onInstall, onSwitchSearch,
}: {
  skills:         SkillStatus[];
  state:          Record<string, InstallState>;
  progress:       Record<string, { pct: number; downloaded?: number; total?: number | null }>;
  currentModel:   string;
  onInstall:      (def: SkillDef) => void;
  onSwitchSearch: (model: "off" | "mobileclip_s0" | "clip_b32" | "jina_clip") => void;
}) {
  const installedSet = new Set(skills.filter(s => s.installed).map(s => s.id));

  return (
    <Card title="Semantic event search"
      subtitle="Search footage by meaning, plus “Find similar”.">
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {SEARCH_TIERS.map(row => {
          const def = findSkill(row.id);
          if (!def) return null;
          const isInstalled   = installedSet.has(row.id);
          const isActive      = currentModel === row.id;
          const isDownloading = state[row.id] === "downloading";
          const prog          = progress[row.id];
          const pct           = prog?.pct ?? 0;
          const dlBytes       = prog?.downloaded;
          const totBytes      = prog?.total;
          return (
            <div key={row.id} style={{
              display: "flex", alignItems: "center", gap: 8,
              padding: "8px 10px", borderRadius: 10,
              background: isActive ? "color-mix(in srgb, var(--accent) 10%, transparent)" : "rgb(var(--ink) / 0.02)",
              border: `1px solid ${isActive ? "var(--accent)" : "var(--border)"}`,
            }}>
              <div style={{ minWidth: 120, display: "flex", flexDirection: "column" }}>
                <span style={{ fontWeight: 700, fontSize: 12, color: isActive ? "var(--accent)" : "var(--text-primary)" }}>{row.label}</span>
                <span style={{ fontSize: 10, color: "var(--text-muted)" }}>{row.hint}</span>
              </div>
              <div style={{ flex: 1 }} />
              <RemoveButton skillId={row.id} />
              {isDownloading ? (
                <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 10, color: "var(--text-secondary)" }}>
                  <Download size={11} />
                  {totBytes && totBytes > 0
                    ? `${pct}% · ${formatBytes(dlBytes)} / ${formatBytes(totBytes)}`
                    : dlBytes ? `${formatBytes(dlBytes)} downloaded…` : "Starting…"}
                </div>
              ) : isInstalled ? (
                isActive ? (
                  <span style={{ display: "inline-flex", alignItems: "center", gap: 4, fontSize: 11, fontWeight: 700, color: "var(--accent)" }}>
                    <Check size={11} /> In use
                  </span>
                ) : (
                  <button onClick={() => onSwitchSearch(row.id)}
                    style={{
                      padding: "4px 10px", borderRadius: 999,
                      border: "1px solid var(--border-strong)",
                      background: "transparent", color: "var(--text-primary)",
                      fontSize: 11, fontWeight: 700, cursor: "pointer",
                    }}>Use</button>
                )
              ) : (
                <button onClick={() => onInstall(def)}
                  style={{
                    display: "inline-flex", alignItems: "center", gap: 5,
                    padding: "4px 10px", borderRadius: 999,
                    border: "1px solid var(--border-strong)",
                    background: "transparent", color: "var(--text-primary)",
                    fontSize: 11, fontWeight: 700, cursor: "pointer",
                  }}>
                  <Download size={10} /> Install
                </button>
              )}
            </div>
          );
        })}
        {currentModel !== "off" && (
          <button onClick={() => onSwitchSearch("off")}
            style={{
              alignSelf: "flex-start", marginTop: 2, padding: "3px 8px",
              fontSize: 10, fontWeight: 600, color: "var(--text-muted)",
              background: "transparent", border: "none", cursor: "pointer",
            }}>
            Turn off semantic search
          </button>
        )}
      </div>
    </Card>
  );
}

// ── Advanced disclosure (shared by FaceCard + YoloCard) ─────────────────────

function AdvancedDisclosure({ children }: { children: React.ReactNode }) {
  const [open, setOpen] = React.useState(false);
  return (
    <div style={{ marginTop: 10 }}>
      <button onClick={() => setOpen(v => !v)}
        style={{
          display: "inline-flex", alignItems: "center", gap: 4,
          padding: "4px 0", border: "none", background: "transparent",
          color: "var(--text-muted)", fontSize: 11, fontWeight: 600, cursor: "pointer",
        }}>
        <ChevronRight size={11} style={{
          transition: "transform 120ms ease",
          transform: open ? "rotate(90deg)" : "rotate(0deg)",
        }} />
        Advanced
      </button>
      {open && (
        <div style={{
          marginTop: 8, padding: "10px 12px",
          borderRadius: 10, border: "1px solid var(--border)",
          background: "rgb(var(--ink) / 0.02)",
          display: "flex", flexDirection: "column", gap: 12,
        }}>
          {children}
        </div>
      )}
    </div>
  );
}

function SliderRow({
  label, help, value, min, max, step, onChange,
}: {
  label:    string;
  help?:    string;
  value:    number;
  min:      number;
  max:      number;
  step:     number;
  onChange: (v: number) => void;
}) {
  return (
    <div>
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", marginBottom: 4 }}>
        <span style={{ fontSize: 11, fontWeight: 700 }}>{label}</span>
        <span style={{ fontSize: 11, color: "var(--text-secondary)", fontFeatureSettings: "'tnum'" }}>
          {value.toFixed(step < 1 ? 2 : 0)}
        </span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(parseFloat(e.target.value))}
        style={{ width: "100%" }} />
      {help && (
        <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 2 }}>{help}</div>
      )}
    </div>
  );
}

// ── Face card — two tiers, identical row pattern to YOLO/ALPR/Search ─────────

const FACE_TIERS = [
  { tier: "small", id: "face_small", label: "Small",  size: "~37 MB"  },
  { tier: "large", id: "face_large", label: "Large",  size: "~262 MB" },
] as const;

function FaceCard({
  skills, state, progress, settings,
  currentFace, onInstall, onSwitchFace, onPatchSettings,
}: {
  skills:          SkillStatus[];
  state:           Record<string, InstallState>;
  progress:        Record<string, { pct: number; downloaded?: number; total?: number | null }>;
  settings:        Settings | null;
  currentFace:     string;
  onInstall:       (def: SkillDef) => void;
  onSwitchFace:    (tier: "off" | "small" | "large") => void;
  onPatchSettings: (patch: Partial<Settings>) => void;
}) {
  const installedSet = new Set(skills.filter(s => s.installed).map(s => s.id));

  return (
    <Card title="Face recognition"
      subtitle="Names known persons in event captions and notifications.">
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {FACE_TIERS.map(row => {
          const def = findSkill(row.id);
          if (!def) return null;
          const isInstalled   = installedSet.has(row.id);
          const isActive      = currentFace === row.tier;
          const isDownloading = state[row.id] === "downloading";
          const prog          = progress[row.id];
          const pct           = prog?.pct ?? 0;
          const dlBytes       = prog?.downloaded;
          const totBytes      = prog?.total;
          return (
            <div key={row.id} style={{
              display: "flex", alignItems: "center", gap: 8,
              padding: "8px 10px", borderRadius: 10,
              background: isActive ? "color-mix(in srgb, var(--accent) 10%, transparent)" : "rgb(var(--ink) / 0.02)",
              border: `1px solid ${isActive ? "var(--accent)" : "var(--border)"}`,
            }}>
              <span style={{ fontWeight: 700, fontSize: 12, minWidth: 60,
                color: isActive ? "var(--accent)" : "var(--text-primary)" }}>{row.label}</span>
              <span style={{ fontSize: 10, color: "var(--text-muted)" }}>{row.size}</span>
              <div style={{ flex: 1 }} />
              <RemoveButton skillId={row.id} />
              {isDownloading ? (
                <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 10, color: "var(--text-secondary)" }}>
                  <Download size={11} />
                  {totBytes && totBytes > 0
                    ? `${pct}% · ${formatBytes(dlBytes)} / ${formatBytes(totBytes)}`
                    : dlBytes ? `${formatBytes(dlBytes)} downloaded…` : "Starting…"}
                </div>
              ) : isInstalled ? (
                isActive ? (
                  <span style={{ display: "inline-flex", alignItems: "center", gap: 4, fontSize: 11, fontWeight: 700, color: "var(--accent)" }}>
                    <Check size={11} /> In use
                  </span>
                ) : (
                  <button onClick={() => onSwitchFace(row.tier)}
                    style={{
                      padding: "4px 10px", borderRadius: 999,
                      border: "1px solid var(--border-strong)",
                      background: "transparent", color: "var(--text-primary)",
                      fontSize: 11, fontWeight: 700, cursor: "pointer",
                    }}>Use</button>
                )
              ) : (
                <button onClick={() => onInstall(def)}
                  style={{
                    display: "inline-flex", alignItems: "center", gap: 5,
                    padding: "4px 10px", borderRadius: 999,
                    border: "1px solid var(--border-strong)",
                    background: "transparent", color: "var(--text-primary)",
                    fontSize: 11, fontWeight: 700, cursor: "pointer",
                  }}>
                  <Download size={10} /> Install
                </button>
              )}
            </div>
          );
        })}
        {currentFace !== "off" && (
          <button onClick={() => onSwitchFace("off")}
            style={{
              alignSelf: "flex-start", marginTop: 2, padding: "3px 8px",
              fontSize: 10, fontWeight: 600, color: "var(--text-muted)",
              background: "transparent", border: "none", cursor: "pointer",
            }}>
            Turn off face recognition
          </button>
        )}
      </div>
      {/* The four face thresholds used to be duplicated here, on the LEGACY
          probability scale (0.7 / 0.9 / 0.8) with a "Recognition confidence"
          range of 0.70–0.99. The backend compares RAW ArcFace cosine and
          defaults to 0.5 — anything above ~0.65 stops recognition entirely, so
          every position on that slider was worse than never touching it. They
          live in Settings → Faces → Advanced now, on the real scale, once. */}
    </Card>
  );
}

// ── YOLO card — five tiers (mature NVRs / edge-AI NVRs pattern) ──────────────────

/// The on-device language model, as a card in the same grid as every other model.
///
/// It used to be a bespoke banner bolted above the grid (`InAppEngineBanner`),
/// which is why local AI felt like it lived in two places: a provider tab AND an
/// install strip, in a shape nothing else used. Same tier-row layout as YOLO now,
/// so "install a model" means one thing everywhere.
const LLM_TIERS = [
  { tier: "fast",     id: "local_llm_fast",   label: "Fast",     size: "~230 MB", note: "instant, terse" },
  { tier: "balanced", id: "local_llm",        label: "Balanced", size: "~731 MB", note: "the default" },
  { tier: "vision",   id: "local_llm_vision", label: "Vision",   size: "~1.3 GB", note: "can see footage" },
] as const;

function LocalAiCard({
  skills, state, progress, settings, onInstall, onSwitchLlm,
}: {
  skills:      SkillStatus[];
  state:       Record<string, InstallState>;
  progress:    Record<string, { pct: number; downloaded?: number; total?: number | null }>;
  settings:    Settings | null;
  onInstall:   (def: SkillDef) => void;
  onSwitchLlm: (tier: "fast" | "balanced" | "vision") => void;
}) {
  const installedSet = new Set(skills.filter(s => s.installed).map(s => s.id));
  const current  = settings?.local_llm_tier ?? "balanced";
  // `""` also means on-device (the settings v3/v4 default), same as elsewhere.
  const onDevice = (settings?.ai_provider ?? "local") === "local" || settings?.ai_provider === "";

  return (
    <Card title="Local AI"
      subtitle="Runs the chat inside the app — no server, no API key, nothing resident when idle.">
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {LLM_TIERS.map(row => {
          const def = findSkill(row.id);
          if (!def) return null;
          const isInstalled   = installedSet.has(row.id);
          // "In use" needs BOTH: the tier is picked AND on-device is the active
          // provider. Showing it while the user is on OpenAI would be a lie.
          const isActive      = onDevice && current === row.tier;
          const isDownloading = state[row.id] === "downloading";
          const prog          = progress[row.id];
          const pct           = prog?.pct ?? 0;
          const dlBytes       = prog?.downloaded;
          const totBytes      = prog?.total;
          return (
            <div key={row.id} style={{
              display: "flex", alignItems: "center", gap: 8,
              padding: "8px 10px", borderRadius: 10,
              background: isActive ? "color-mix(in srgb, var(--accent) 10%, transparent)" : "rgb(var(--ink) / 0.02)",
              border: `1px solid ${isActive ? "var(--accent)" : "var(--border)"}`,
            }}>
              <span style={{
                fontWeight: 700, fontSize: 12,
                color: isActive ? "var(--accent)" : "var(--text-primary)",
                minWidth: 66,
              }}>{row.label}</span>
              <span style={{ fontSize: 10, color: "var(--text-muted)" }}>{row.size}</span>
              <span style={{ fontSize: 10, color: "var(--text-muted)", opacity: 0.75 }}>· {row.note}</span>
              <div style={{ flex: 1 }} />
              <RemoveButton skillId={row.id} />
              {isDownloading ? (
                <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 10, color: "var(--text-secondary)" }}>
                  <Download size={11} />
                  {totBytes && totBytes > 0
                    ? `${pct}% · ${formatBytes(dlBytes)} / ${formatBytes(totBytes)}`
                    : dlBytes
                      ? `${formatBytes(dlBytes)} downloaded…`
                      : "Starting…"}
                </div>
              ) : isInstalled ? (
                isActive ? (
                  <span style={{ display: "inline-flex", alignItems: "center", gap: 4, fontSize: 11, fontWeight: 700, color: "var(--accent)" }}>
                    <Check size={11} /> In use
                  </span>
                ) : (
                  <button onClick={() => onSwitchLlm(row.tier)}
                    style={{
                      padding: "4px 10px", borderRadius: 999,
                      border: "1px solid var(--border-strong)",
                      background: "transparent", color: "var(--text-primary)",
                      fontSize: 11, fontWeight: 700, cursor: "pointer",
                    }}>Use</button>
                )
              ) : (
                <button onClick={() => onInstall(def)}
                  style={{
                    display: "inline-flex", alignItems: "center", gap: 5,
                    padding: "4px 10px", borderRadius: 999,
                    border: "1px solid var(--border-strong)",
                    background: "transparent", color: "var(--text-primary)",
                    fontSize: 11, fontWeight: 700, cursor: "pointer",
                  }}>
                  <Download size={10} /> Install
                </button>
              )}
            </div>
          );
        })}
      </div>
      <div style={{ fontSize: 10.5, color: "var(--text-muted)", marginTop: 10, lineHeight: 1.5 }}>
        Weights are downloaded from Liquid AI at your request under the
        <strong> LFM Open License v1.0</strong> — their own licence, not Apache-2.0 —
        and are not distributed with this app.
      </div>
    </Card>
  );
}

const YOLO_TIERS = [
  { tier: "nano",   id: "yolo26n", label: "Nano",   size: "~10 MB"  },
  { tier: "small",  id: "yolo26s", label: "Small",  size: "~37 MB"  },
  { tier: "medium", id: "yolo26m", label: "Medium", size: "~78 MB"  },
  { tier: "large",  id: "yolo26l", label: "Large",  size: "~95 MB"  },
  { tier: "xlarge", id: "yolo26x", label: "XLarge", size: "~175 MB" },
] as const;

function YoloCard({
  skills, state, progress, settings,
  currentYolo, onInstall, onSwitchYolo, onPatchSettings,
}: {
  skills:          SkillStatus[];
  state:           Record<string, InstallState>;
  progress:        Record<string, { pct: number; downloaded?: number; total?: number | null }>;
  settings:        Settings | null;
  currentYolo:     string;
  onInstall:       (def: SkillDef) => void;
  onSwitchYolo:    (tier: "nano" | "small" | "medium" | "large" | "xlarge") => void;
  onPatchSettings: (patch: Partial<Settings>) => void;
}) {
  const installedSet = new Set(skills.filter(s => s.installed).map(s => s.id));

  return (
    <Card title="Object detection"
      subtitle="YOLO 2026 — detects what's in every frame.">
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {YOLO_TIERS.map(row => {
          const def = findSkill(row.id);
          if (!def) return null;
          const isInstalled = installedSet.has(row.id);
          const isActive    = currentYolo === row.tier;
          const isDownloading = state[row.id] === "downloading";
          const prog          = progress[row.id];
          const pct           = prog?.pct ?? 0;
          const dlBytes       = prog?.downloaded;
          const totBytes      = prog?.total;
          return (
            <div key={row.id} style={{
              display: "flex", alignItems: "center", gap: 8,
              padding: "8px 10px", borderRadius: 10,
              background: isActive ? "color-mix(in srgb, var(--accent) 10%, transparent)" : "rgb(var(--ink) / 0.02)",
              border: `1px solid ${isActive ? "var(--accent)" : "var(--border)"}`,
            }}>
              <span style={{
                fontWeight: 700, fontSize: 12,
                color: isActive ? "var(--accent)" : "var(--text-primary)",
                minWidth: 60,
              }}>{row.label}</span>
              <span style={{ fontSize: 10, color: "var(--text-muted)" }}>{row.size}</span>
              <div style={{ flex: 1 }} />
              <RemoveButton skillId={row.id} />
              {isDownloading ? (
                <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 10, color: "var(--text-secondary)" }}>
                  <Download size={11} />
                  {totBytes && totBytes > 0
                    ? `${pct}% · ${formatBytes(dlBytes)} / ${formatBytes(totBytes)}`
                    : dlBytes
                      ? `${formatBytes(dlBytes)} downloaded…`
                      : "Starting…"}
                </div>
              ) : isInstalled ? (
                isActive ? (
                  <span style={{ display: "inline-flex", alignItems: "center", gap: 4, fontSize: 11, fontWeight: 700, color: "var(--accent)" }}>
                    <Check size={11} /> In use
                  </span>
                ) : (
                  <button onClick={() => onSwitchYolo(row.tier)}
                    style={{
                      padding: "4px 10px", borderRadius: 999,
                      border: "1px solid var(--border-strong)",
                      background: "transparent", color: "var(--text-primary)",
                      fontSize: 11, fontWeight: 700, cursor: "pointer",
                    }}>Use</button>
                )
              ) : (
                <button onClick={() => onInstall(def)}
                  style={{
                    display: "inline-flex", alignItems: "center", gap: 5,
                    padding: "4px 10px", borderRadius: 999,
                    border: "1px solid var(--border-strong)",
                    background: "transparent", color: "var(--text-primary)",
                    fontSize: 11, fontWeight: 700, cursor: "pointer",
                  }}>
                  <Download size={10} /> Install
                </button>
              )}
            </div>
          );
        })}
      </div>
      {/* Advanced — global confidence threshold + COCO-group class filter.
          Class chips map "person", "vehicle", "animal", "package" to their
          COCO class names; selection writes to settings.yolo_class_filter
          as a comma-separated whitelist. Empty selection = no filter. */}
      <AdvancedDisclosure>
        {/* Range and fallback track state.rs's default (0.40). They used to read
            0.10–0.80 / 0.20, so opening the disclosure and touching nothing else
            still showed a threshold the detector was not using. */}
        <SliderRow label="Confidence threshold"
          value={settings?.yolo_confidence_threshold ?? 0.40}
          min={0.05} max={0.95} step={0.05}
          onChange={v => onPatchSettings({ yolo_confidence_threshold: v })} />
        <YoloClassFilter
          value={settings?.yolo_class_filter ?? ""}
          onChange={v => onPatchSettings({ yolo_class_filter: v })} />
      </AdvancedDisclosure>
    </Card>
  );
}

// ── YOLO class-group filter chips ───────────────────────────────────────────
// standard: users pick broad categories, not individual COCO classes.
// Each group expands to its full COCO labels when written to settings.

const YOLO_CLASS_GROUPS: { id: string; label: string; classes: string[] }[] = [
  { id: "person",  label: "Person",  classes: ["person"] },
  { id: "vehicle", label: "Vehicle", classes: ["car","truck","bus","motorcycle","bicycle","train","boat","airplane"] },
  { id: "animal",  label: "Animal",  classes: ["cat","dog","bird","horse","sheep","cow","elephant","bear","zebra","giraffe"] },
  { id: "package", label: "Package", classes: ["backpack","suitcase","handbag"] },
];

function YoloClassFilter({ value, onChange }: { value: string; onChange: (csv: string) => void }) {
  const selected = new Set(value.split(",").map(s => s.trim().toLowerCase()).filter(Boolean));
  const isEmpty = selected.size === 0;
  const isGroupActive = (g: typeof YOLO_CLASS_GROUPS[number]) =>
    !isEmpty && g.classes.every(c => selected.has(c));

  const toggle = (g: typeof YOLO_CLASS_GROUPS[number]) => {
    // If "all" is on (empty filter), selecting any group should restrict to that group.
    let next = isEmpty
      ? new Set<string>(g.classes)
      : new Set(selected);
    if (!isEmpty) {
      if (isGroupActive(g)) {
        g.classes.forEach(c => next.delete(c));
      } else {
        g.classes.forEach(c => next.add(c));
      }
    }
    // Empty filter means "track everything" — that's the intuitive default
    // when the user toggles every group off.
    const csv = Array.from(next).sort().join(",");
    onChange(csv);
  };

  return (
    <div>
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", marginBottom: 4 }}>
        <span style={{ fontSize: 11, fontWeight: 700 }}>Classes to track</span>
        <span style={{ fontSize: 10, color: "var(--text-muted)" }}>
          {isEmpty ? "All" : `${selected.size} class${selected.size === 1 ? "" : "es"}`}
        </span>
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
        {YOLO_CLASS_GROUPS.map(g => {
          // `isEmpty ||` used to be here, so an empty filter painted all four
          // chips as SELECTED. Clicking a lit chip then restricted to that group
          // instead of deselecting it, and switching the last group off flipped
          // silently from "none" back to "all". Empty means "no filter", which
          // the "All" label above already says.
          const active = isGroupActive(g);
          return (
            <button key={g.id} onClick={() => toggle(g)}
              style={{
                padding: "3px 9px", borderRadius: 999,
                border: `1px solid ${active ? "var(--accent)" : "var(--border)"}`,
                background: active ? "color-mix(in srgb, var(--accent) 10%, transparent)" : "transparent",
                color: active ? "var(--accent)" : "var(--text-muted)",
                fontSize: 10, fontWeight: 700, cursor: "pointer",
              }}>
              {g.label}
            </button>
          );
        })}
      </div>
      <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 4 }}>
        Turn all off to track every detected class.
      </div>
    </div>
  );
}

// ── Card primitive ───────────────────────────────────────────────────────────

// Uniform card shell: fixed header block (title + one-line subtitle) so every
// card's action area starts at the SAME y-offset, then the body. `marginTop:
// auto` bottom-aligns the action content across a grid row.
function Card({
  title, subtitle, children,
}: {
  title:    string;
  subtitle: string;
  children: React.ReactNode;
}) {
  return (
    <div className="glass" style={{ padding: 16, display: "flex", flexDirection: "column" }}>
      <div style={{ marginBottom: 12, minHeight: 34 }}>
        <div style={{ fontWeight: 700, fontSize: 13, letterSpacing: -0.01 }}>{title}</div>
        <div style={{ fontSize: 11, color: "var(--text-muted)", lineHeight: 1.4, marginTop: 2,
          overflow: "hidden", display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical" }}>{subtitle}</div>
      </div>
      <div style={{ marginTop: "auto" }}>{children}</div>
    </div>
  );
}

function formatBytes(n?: number): string {
  if (!n) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let v = n; let i = 0;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
  return `${v.toFixed(v < 10 ? 1 : 0)} ${units[i]}`;
}

// ── Identity footer ──────────────────────────────────────────────────────────

function IdentityFooter({
  settings, onSave,
}: {
  settings: Settings | null;
  onSave: (patch: Partial<Settings>) => Promise<void>;
}) {
  const [name, setName] = useState(settings?.agent_persona_name ?? "");
  const [text, setText] = useState(settings?.agent_persona_text ?? "");
  useEffect(() => {
    setName(settings?.agent_persona_name ?? "");
    setText(settings?.agent_persona_text ?? "");
  }, [settings?.agent_persona_name, settings?.agent_persona_text]);

  return (
    <div className="glass" style={{ padding: 18 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 14 }}>
        <Bot size={14} style={{ color: "var(--accent)" }} />
        <span style={{ fontWeight: 700, fontSize: 13 }}>Agent identity</span>
        <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
          How the Guardian introduces itself in alerts and chat.
        </span>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 2fr", gap: 12 }}>
        <label style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          <span style={{ fontSize: 10, fontWeight: 700, color: "var(--text-muted)", letterSpacing: 0.04, textTransform: "uppercase" }}>
            Name
          </span>
          <input value={name} onChange={e => setName(e.target.value)}
            onBlur={() => name !== (settings?.agent_persona_name ?? "") && onSave({ agent_persona_name: name })}
            style={input} />
        </label>
        <label style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          <span style={{ fontSize: 10, fontWeight: 700, color: "var(--text-muted)", letterSpacing: 0.04, textTransform: "uppercase" }}>
            Personality
          </span>
          <input value={text} onChange={e => setText(e.target.value)}
            onBlur={() => text !== (settings?.agent_persona_text ?? "") && onSave({ agent_persona_text: text })}
            placeholder="e.g. calm, terse, focuses on actionable risk."
            style={input} />
        </label>
      </div>
    </div>
  );
}

const input: React.CSSProperties = {
  padding: "10px 14px", borderRadius: 12, fontSize: 13,
  border: "1px solid var(--border-strong)",
  background: "rgb(var(--ink) / 0.04)",
  color: "var(--text-primary)", outline: "none",
  fontFamily: "inherit",
};

// Used for unused-import suppression — these come from the registry.
void SKILL_REGISTRY;
void (null as unknown as SkillId);
