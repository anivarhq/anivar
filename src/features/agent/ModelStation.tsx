/**
 * ModelStation — the single panel that owns AI-model lifecycle.
 *
 * One-glance contract:
 *   • Provider segmented control (On-device / LM Studio / Custom / cloud)
 *   • Per-provider connection block (nothing for on-device, URL/API key otherwise)
 *   • Test Connection button → `api.testAiProvider`
 *   • "Use" writes the picked tag to `vision_model` — the single model field.
 *
 * The Ollama tab, its model browser and its pull/delete plumbing were REMOVED
 * with the daemon (2026-07-28): the language model now runs in-process via
 * llama.cpp (`agent::local_llm`). Anyone who still wants a self-hosted server —
 * including Ollama — points "Custom (OpenAI-Compatible)" at its /v1 endpoint.
 */

import { useCallback, useEffect, useState } from "react";
import {
  Cpu, Sparkles, Brain, Zap, Globe,
  ServerCog, Plug, CheckCircle, AlertCircle, Eye, EyeOff,
  Check, RefreshCw, Loader,
} from "lucide-react";
import { api, Settings } from "../../api";
import { useStore } from "../../store";
import { useShallow } from "zustand/react/shallow";

type ProviderId =
  | "local"
  | "lmstudio"
  | "openai_compatible"
  | "openai"
  | "anthropic"
  | "groq"
  | "xai"
  | "gemini";

/** How the connection block + model list behave for this provider. */
type ProviderKind = "on-device" | "local-openai" | "cloud";

interface ProviderDef {
  id:          ProviderId;
  label:       string;
  icon:        React.ReactNode;
  kind:        ProviderKind;
  /** Suggested URL for `local-openai` providers — UI pre-fills on first focus. */
  defaultUrl?: string;
}

const PROVIDERS: ProviderDef[] = [
  // The default. Runs inside anivar.exe via llama.cpp — no server, no port,
  // no install. Listed first because it needs no configuration at all.
  { id: "local",             label: "On-device",                  icon: <Cpu size={12} />,       kind: "on-device" },
  { id: "lmstudio",          label: "LM Studio",                  icon: <ServerCog size={12} />, kind: "local-openai", defaultUrl: "http://localhost:1234/v1" },
  { id: "openai_compatible", label: "Custom (OpenAI-Compatible)", icon: <Plug size={12} />,      kind: "local-openai", defaultUrl: "" },
  { id: "openai",            label: "OpenAI",                     icon: <Sparkles size={12} />,  kind: "cloud" },
  { id: "anthropic",         label: "Claude",                     icon: <Brain size={12} />,     kind: "cloud" },
  { id: "groq",              label: "Groq",                       icon: <Zap size={12} />,       kind: "cloud" },
  { id: "xai",               label: "xAI (Grok)",                 icon: <Zap size={12} />,       kind: "cloud" },
  { id: "gemini",            label: "Gemini",                     icon: <Globe size={12} />,     kind: "cloud" },
];

function findProvider(id: string): ProviderDef {
  return PROVIDERS.find(p => p.id === id) ?? PROVIDERS[0];
}

interface TestResult {
  ok:      boolean;
  count?:  number;
  models?: string[];
  error?:  string;
  /** WHICH provider produced this result. Load-bearing: without it a result from
   *  the previously-viewed tab can populate the newly-viewed tab's model list —
   *  that is how "LFM2.5-350M (on-device)" ended up listed under LM Studio and
   *  got committed as `lmstudio` + an on-device model name, sending every chat
   *  to localhost:1234. Always compare against `viewProvider` before using it. */
  provider: ProviderId;
}

export function ModelStation() {
  const { settings, setSettings, showToast } = useStore(useShallow(s => ({ settings: s.settings, setSettings: s.setSettings, showToast: s.showToast })));
  // The COMMITTED / active engine — what actually runs inference right now.
  const activeProvider = (settings?.ai_provider as ProviderId) || "local";
  const activeModel    = settings?.vision_model || "";

  // The provider the user is BROWSING. Switching tabs only changes this — it does
  // NOT touch the running engine. The active model keeps running until the user
  // explicitly picks a model (Use), which commits provider + model together.
  // (Open WebUI / LM Studio / LibreChat all separate "browse" from "load".)
  const [viewProvider, setViewProvider] = useState<ProviderId>(activeProvider);
  // Follow the active provider when it changes externally (initial settings load,
  // or right after we commit a new model). Never yanks the user off a tab they're
  // browsing, since browsing alone doesn't change activeProvider.
  useEffect(() => { setViewProvider(activeProvider); }, [activeProvider]);

  // A model row is "in use" only when we're viewing the provider that's actually
  // active — the same model string could exist under two providers.
  const inUseModel  = viewProvider === activeProvider ? activeModel : "";
  const viewLabel   = findProvider(viewProvider).label;
  const activeLabel = findProvider(activeProvider).label;

  // ── Connection test ─────────────────────────────────────────────────────
  const [testing, setTesting] = useState(false);
  const [testRes, setTestRes] = useState<TestResult | null>(null);
  const runTest = async () => {
    // Capture the provider being tested: the await below can outlive the tab, and
    // a late reply must not be mistaken for the new tab's result.
    const tested = viewProvider;
    setTesting(true);
    setTestRes(null);
    try {
      const r = await api.testAiProvider(tested);
      setTestRes({ ok: r.ok, count: r.count, models: r.models, error: r.error, provider: tested });
    } catch (e: any) {
      setTestRes({ ok: false, error: String(e), provider: tested });
    } finally {
      setTesting(false);
    }
  };
  // Reset + AUTO-PROBE the connection whenever the provider changes (and on mount), so
  // the status chip is always live — the user shouldn't have to click "Test" to learn
  // whether the selected provider actually works. Debounced so a quick switch doesn't
  // fire a burst of requests. A manual Test still works for re-checking after edits.
  useEffect(() => {
    setTestRes(null);
    const id = setTimeout(() => { runTest(); }, 500);
    return () => clearTimeout(id);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [viewProvider]);

  // ── Model list (provider-dependent) ─────────────────────────────────────
  const [cloudModels,     setCloudModels]     = useState<Array<{ id: string; vision: boolean }>>([]);

  const viewKind = findProvider(viewProvider).kind;

  const loadModels = useCallback(async () => {
    if (viewKind === "cloud") {
      try {
        const list = await api.listProviderModels(viewProvider);
        setCloudModels((list ?? []).map(m => ({ id: m.id, vision: m.category === "vision" })));
      } catch { setCloudModels([]); }
    } else {
      // local-openai: live model list is populated by Test response. Do not
      // call list_provider_models — it returns the curated cloud lists.
      setCloudModels([]);
    }
  }, [viewKind, viewProvider]);
  useEffect(() => { loadModels(); }, [loadModels]);
  // For local-openai providers, surface whatever Test returned as the model list.
  // These come from a live `/v1/models` call, so vision-capability is unknown.
  useEffect(() => {
    // `testRes.provider === viewProvider` is the guard. On a tab switch this effect
    // re-runs with the PREVIOUS tab's result still in scope (the sibling effect's
    // setTestRes(null) only queues a re-render), so without it the old provider's
    // models get listed under the new one.
    if (viewKind === "local-openai" && testRes?.ok && testRes.provider === viewProvider
        && Array.isArray(testRes.models)) {
      setCloudModels((testRes.models ?? []).map(id => ({ id, vision: false })));
    }
  }, [viewKind, viewProvider, testRes]);

  // (The Ollama pull-progress listener, startPull and deleteModel lived here.
  //  They drove the model browser, which went with the daemon — the on-device
  //  model installs through the normal skill flow instead.)

  const useModel = async (tag: string) => {
    if (!settings) return;
    // Commit BOTH the browsed provider and the chosen model atomically — this is
    // the ONLY action that changes the running engine. Browsing tabs never does.
    // Also ENABLE the agent: picking a model IS the intent to use it — before
    // this, `agent_enabled` had no UI control at all, so the Guardian stayed
    // "resting" forever even with a model selected (the "I'm resting" bug).
    const next: Settings = { ...settings, ai_provider: viewProvider, vision_model: tag, agent_enabled: true };
    await api.saveSettings(next);
    setSettings(next);
    // On-device has no model NAME to name (the engine is compiled in), so `tag`
    // is empty there — say something meaningful instead of "Now using  · …".
    showToast(
      tag ? `Now using ${tag} · ${findProvider(viewProvider).label} — Guardian active`
          : `Now using ${findProvider(viewProvider).label} AI — Guardian active`,
      "success");
  };

  /** Stop using AI altogether. There is no "no engine" provider — every provider
   *  value names a real engine — so the off switch is `agent_enabled`, matching the
   *  "Turn off face recognition" precedent. The installed model is left on disk;
   *  Remove is the separate, destructive action. */
  const turnOffAi = async () => {
    if (!settings) return;
    const next: Settings = { ...settings, agent_enabled: false };
    await api.saveSettings(next);
    setSettings(next);
    showToast("Guardian AI turned off — the model stays installed", "info");
  };

  // ── Custom tag input ───────────────────────────────────────────────────
    return (
    <div className="glass" style={{ padding: 18, display: "flex", flexDirection: "column", gap: 14 }}>
      {/* ── Header ───────────────────────────────────────────────────────── */}
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <div>
          <div style={{ fontWeight: 800, fontSize: 14, letterSpacing: -0.02 }}>Model</div>
          <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
            Powers vision &amp; chat for the Guardian.
          </div>
        </div>
        <div style={{ flex: 1 }} />
        {/* Persistent ACTIVE badge — always shows the running provider + model,
            no matter which provider tab you're browsing.
            On-device is "configured" with an EMPTY `vision_model` (the engine is
            compiled in and has no model name), so keying the badge off that field
            alone rendered a working on-device engine as "No model selected". */}
        {activeModel || activeProvider === "local" ? (
          <span title={`Active engine: ${activeLabel}`} style={{
            display: "inline-flex", alignItems: "center", gap: 6,
            padding: "4px 10px", borderRadius: 999, maxWidth: 300,
            background: "var(--accent-glow)", color: "var(--accent)",
            fontSize: 11, fontWeight: 700,
          }}>
            <CheckCircle size={11} style={{ flexShrink: 0 }} />
            <span style={{ opacity: 0.7 }}>{activeLabel}</span>
            {/* On-device sets `vision_model` to "" by design (the engine is compiled
                in and has no model name), so this used to render "On-device · "
                with a separator and nothing after it. */}
            {!!activeModel && (
              <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>· {activeModel}</span>
            )}
          </span>
        ) : (
          <span style={{
            display: "inline-flex", alignItems: "center", gap: 5,
            padding: "4px 10px", borderRadius: 999,
            background: "rgb(var(--ink) / 0.05)", color: "var(--text-muted)",
            fontSize: 11, fontWeight: 700,
          }}>
            <AlertCircle size={11} /> No model selected
          </span>
        )}
      </div>

      {/* The off switch. `turnOffAi` was DEFINED here and never rendered, so
          `agent_enabled` was written true by this component and by Arsenal's
          local-AI card, and false nowhere reachable. Ten sites in the Rust tree
          gate on it, which meant that once a user picked any model, on-device
          analysis ran forever with nothing in the app able to stop it. Shown
          only while the agent is on, next to the badge that says so. */}
      {settings?.agent_enabled && (
        <button type="button" onClick={turnOffAi}
          title="Stops background analysis. The installed model stays on disk."
          style={{
            alignSelf: "flex-start", padding: "3px 8px", fontSize: 10, fontWeight: 600,
            color: "var(--text-muted)", background: "transparent",
            border: "none", cursor: "pointer",
          }}>
          Turn off Guardian AI
        </button>
      )}

      {/* ── Provider segmented control ──────────────────────────────────── */}
      <div style={{ display: "inline-flex", gap: 4, padding: 4, borderRadius: 999,
        background: "rgb(var(--ink) / 0.04)", flexWrap: "wrap" }}>
        {PROVIDERS.map(p => {
          const on       = viewProvider === p.id;   // currently browsing
          const isActive = activeProvider === p.id;  // currently running
          const tip =
            p.kind === "on-device"    ? "Runs inside the app — no server, nothing to configure"
          : p.kind === "cloud"        ? `Browse ${p.label} via API key`
          : p.kind === "local-openai" ? `Connect to a ${p.label} server`
          :                              `Connect to ${p.label} on this machine`;
          return (
            <button key={p.id} type="button" onClick={() => setViewProvider(p.id)}
              title={isActive ? `${tip} · active engine` : tip}
              style={{
                display: "inline-flex", alignItems: "center", gap: 6,
                padding: "6px 12px", borderRadius: 999, border: "none",
                background: on ? "var(--accent)" : "transparent",
                color:      on ? "var(--on-accent)"     : "var(--text-secondary)",
                fontSize: 11, fontWeight: 700, cursor: "pointer",
              }}>
              {p.icon}{p.label}
              {/* Dot marks the running engine when you've browsed away from it. */}
              {isActive && !on && (
                <span title="Active engine" style={{
                  width: 5, height: 5, borderRadius: 999,
                  background: "var(--accent)", marginLeft: 1,
                }} />
              )}
            </button>
          );
        })}
      </div>

      {/* ── Browsing-vs-active hint ─────────────────────────────────────── */}
      {viewProvider !== activeProvider && (
        <div style={{
          display: "flex", alignItems: "center", gap: 8,
          padding: "7px 11px", borderRadius: 10,
          background: "rgb(var(--ink) / 0.03)", border: "1px dashed var(--border-strong)",
          fontSize: 11, color: "var(--text-secondary)",
        }}>
          <Eye size={12} style={{ color: "var(--text-muted)", flexShrink: 0 }} />
          <span>
            Browsing <strong>{viewLabel}</strong>.{" "}
            {activeModel
              ? <><strong>{activeLabel} · {activeModel}</strong> stays active until you pick a model here.</>
              : <>Pick a model to make it active.</>}
          </span>
        </div>
      )}

      {/* ── Connection block ────────────────────────────────────────────── */}
      <ConnectionBlock
        key={viewProvider}
        provider={viewProvider}
        settings={settings}
        onChange={async (patch) => {
          if (!settings) return;
          const next: Settings = { ...settings, ...patch };
          await api.saveSettings(next);
          setSettings(next);
          // Re-probe after a credential/URL edit so the status chip reflects reality
          // immediately (type key → blur → saves → auto-tests → ✓/✗).
          setTimeout(() => { runTest(); }, 300);
        }}
        onTest={runTest}
        testing={testing}
        testRes={testRes}
      />

      {/* ── On-device engine: install state is the entire configuration ── */}

      {/* ── Model list / browser ────────────────────────────────────────── */}
      {viewKind === "local-openai" && (
        <CloudList
          models={cloudModels}
          activeModel={inUseModel}
          onUse={useModel}
          providerLabel={viewLabel}
          emptyHint={
            cloudModels.length === 0 && !testRes
              ? "No models yet"
              : cloudModels.length === 0 && testRes && !testRes.ok
              ? "Server unreachable — check the Base URL above."
              : undefined
          }
        />
      )}
      {viewKind === "cloud" && (
        <CloudList
          models={cloudModels}
          activeModel={inUseModel}
          onUse={useModel}
          providerLabel={viewLabel}
        />
      )}

    </div>
  );
}

// ─── Connection block ──────────────────────────────────────────────────────

// On-device AI banner. The app used to DOWNLOAD AND RUN OLLAMA here — a separate
// server it then had to watchdog, which on a 16 GB machine left a 6 GB orphaned
// llama-server that Ollama itself had lost track of. Now the model runs INSIDE
// anivar.exe (llama.cpp, see agent/local_llm.rs): ~230 MB, loaded on demand,
// released after idle, and it cannot outlive the app.
// The on-device install/use/remove strip used to live here. It is now a normal
// card in Arsenal's model grid (`LocalAiCard`), so local AI is described in the
// same shape as every other model instead of having its own bespoke surface —
// which is what made it look like it lived in two places.
function ConnectionBlock({
  provider, settings, onChange, onTest, testing, testRes,
}: {
  provider: ProviderId;
  settings: Settings | null;
  onChange: (patch: Partial<Settings>) => Promise<void>;
  onTest:   () => Promise<void>;
  testing:  boolean;
  testRes:  TestResult | null;
}) {
  const [showKey, setShowKey] = useState(false);
  const def = findProvider(provider);

  // ── On-device: nothing to connect to ────────────────────────────────────
  // The engine is compiled into the app, so the only "connection" state is which
  // model is installed — and that lives in the Local AI card below, alongside
  // every other model, rather than in a second install surface up here.
  if (def.kind === "on-device") {
    return (
      <div style={connWrap}>
        <span style={connLabel}>Engine</span>
        <span style={{ flex: 1, fontSize: 12, color: "var(--text-muted)" }}>
          Runs inside the app — no server, no API key. Pick a model in <strong>Local AI</strong> below.
        </span>
        <TestPill testing={testing} testRes={testRes} onTest={onTest} />
      </div>
    );
  }

  // ── LM Studio / OpenAI-Compatible: URL + optional key ───────────────────
  if (def.kind === "local-openai") {
    const urlCurrent = settings?.openai_compatible_url ?? "";
    const keyCurrent = settings?.openai_compatible_key ?? "";
    const placeholder = def.defaultUrl || "https://your-server.example.com/v1";
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        <div style={connWrap}>
          <span style={connLabel}>Base URL</span>
          <input
            defaultValue={urlCurrent}
            placeholder={placeholder}
            onBlur={e => {
              const v = e.target.value.trim();
              if (v !== urlCurrent) onChange({ openai_compatible_url: v });
            }}
            style={{ ...connInput, fontFamily: "var(--font-mono)" }} />
          <TestPill testing={testing} testRes={testRes} onTest={onTest} />
        </div>
        <div style={connWrap}>
          <span style={connLabel}>API key</span>
          <input
            defaultValue={keyCurrent}
            type={showKey ? "text" : "password"}
            placeholder="optional — most local servers leave this blank"
            onBlur={e => { if (e.target.value !== keyCurrent) onChange({ openai_compatible_key: e.target.value }); }}
            style={{ ...connInput, fontFamily: "var(--font-mono)" }} />
          <button type="button" onClick={() => setShowKey(s => !s)}
            title={showKey ? "Hide key" : "Show key"} style={iconBtn}>
            {showKey ? <EyeOff size={12} /> : <Eye size={12} />}
          </button>
        </div>
        {provider === "lmstudio" && (
          <div style={{ fontSize: 10, color: "var(--text-muted)", paddingLeft: 12 }}>
            Models are managed in the LM Studio app.
          </div>
        )}
      </div>
    );
  }

  // ── Cloud: API key only ─────────────────────────────────────────────────
  const keyField: keyof Settings = (
    provider === "openai" ? "openai_api_key" :
    provider === "anthropic" ? "anthropic_api_key" :
    provider === "groq" ? "groq_api_key" :
    provider === "xai" ? "xai_api_key" : "gemini_api_key"
  );
  const current = (settings?.[keyField] as string | undefined) ?? "";

  return (
    <div style={connWrap}>
      <span style={connLabel}>API key</span>
      <input
        defaultValue={current}
        type={showKey ? "text" : "password"}
        placeholder={`Paste your ${def.label} API key`}
        onBlur={e => { if (e.target.value !== current) onChange({ [keyField]: e.target.value } as Partial<Settings>); }}
        style={{ ...connInput, fontFamily: "var(--font-mono)" }} />
      <button type="button" onClick={() => setShowKey(s => !s)}
        title={showKey ? "Hide key" : "Show key"} style={iconBtn}>
        {showKey ? <EyeOff size={12} /> : <Eye size={12} />}
      </button>
      <TestPill testing={testing} testRes={testRes} onTest={onTest} />
    </div>
  );
}

function TestPill({ testing, testRes, onTest }: {
  testing: boolean; testRes: TestResult | null; onTest: () => void;
}) {
  return (
    <div style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
      <button onClick={onTest} disabled={testing} style={btnGhost(!testing)}>
        {testing ? <Loader size={11} className="spin" /> : <RefreshCw size={11} />} Test
      </button>
      {testRes && (
        testRes.ok ? (
          <span style={pill("var(--accent)")}>
            <CheckCircle size={10} /> {testRes.count ?? 0} models
          </span>
        ) : (
          <span style={pill("var(--accent-red)")} title={testRes.error}>
            <AlertCircle size={10} /> {testRes.error?.slice(0, 60) ?? "Failed"}
          </span>
        )
      )}
    </div>
  );
}
// ─── Cloud list ────────────────────────────────────────────────────────────

function CloudList({
  models, activeModel, onUse, providerLabel, emptyHint,
}: {
  models:        Array<{ id: string; vision: boolean }>;
  activeModel:   string;
  onUse:         (tag: string) => void;
  providerLabel: string;
  emptyHint?:    string;
}) {
  const [customId, setCustomId] = useState("");
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      {models.length === 0 && (
        <div style={{ padding: 12, fontSize: 11, color: "var(--text-muted)", textAlign: "center" }}>
          {emptyHint ?? <>Add your API key above and click <strong>Test</strong> to verify.</>}
        </div>
      )}
      {models.map(m => {
        const active = m.id === activeModel;
        return (
          <div key={m.id} style={modelRow(active)}>
            <span style={tagText(active)}>{m.id}</span>
            {m.vision && (
              <span style={pill("var(--accent)")} title="Can analyse camera frames">
                <Eye size={10} /> Vision
              </span>
            )}
            <button onClick={() => onUse(m.id)} disabled={active} style={btnPrimary(!active)}>
              {active ? <><Check size={11} /> In use</> : "Use"}
            </button>
          </div>
        );
      })}
      {/* Custom model ID — Open WebUI's escape hatch for any provider model. */}
      <div style={{
        display: "flex", gap: 8, alignItems: "center",
        padding: "8px 12px", borderRadius: 12,
        background: "rgb(var(--ink) / 0.02)",
        border: "1px dashed var(--border)",
      }}>
        <span style={{ fontSize: 10, fontWeight: 700, color: "var(--text-muted)",
          textTransform: "uppercase", letterSpacing: 0.04, whiteSpace: "nowrap" }}>+ Custom</span>
        <input value={customId} onChange={e => setCustomId(e.target.value)}
          placeholder={`${providerLabel} model ID, e.g. gpt-4o-2024-08-06`}
          onKeyDown={e => { if (e.key === "Enter" && customId.trim()) { onUse(customId.trim()); setCustomId(""); } }}
          style={{
            flex: 1, border: "none", background: "transparent", outline: "none",
            fontFamily: "var(--font-mono)", fontSize: 12, color: "var(--text-primary)",
          }} />
        <button onClick={() => { if (customId.trim()) { onUse(customId.trim()); setCustomId(""); } }}
          disabled={!customId.trim()}
          style={btnPrimary(!!customId.trim())}>
          Use
        </button>
      </div>
    </div>
  );
}

function tagText(active: boolean): React.CSSProperties {
  return {
    fontFamily: "var(--font-mono)", fontSize: 12,
    color: active ? "var(--accent)" : "var(--text-primary)",
    flex: 1, overflow: "hidden", textOverflow: "ellipsis",
  };
}
// ─── Helpers ───────────────────────────────────────────────────────────────

// ─── Style tokens ──────────────────────────────────────────────────────────

const connWrap: React.CSSProperties = {
  display: "flex", alignItems: "center", gap: 8,
  padding: "10px 12px", borderRadius: 12,
  border: "1px solid var(--border)",
  background: "rgb(var(--ink) / 0.03)",
};
const connLabel: React.CSSProperties = {
  fontSize: 10, fontWeight: 700, color: "var(--text-muted)",
  letterSpacing: 0.04, textTransform: "uppercase",
};
const connInput: React.CSSProperties = {
  flex: 1, border: "none", background: "transparent", outline: "none",
  fontSize: 12, color: "var(--text-primary)",
};
const iconBtn: React.CSSProperties = {
  display: "inline-flex", alignItems: "center", justifyContent: "center",
  width: 26, height: 26, borderRadius: 8, border: "1px solid var(--border)",
  background: "transparent", color: "var(--text-muted)", cursor: "pointer",
};

function modelRow(active: boolean): React.CSSProperties {
  return {
    display: "flex", alignItems: "center", gap: 8,
    padding: "8px 12px", borderRadius: 12,
    background: active ? "var(--hl)" : "rgb(var(--ink) / 0.02)",
    border: `1px solid ${active ? "var(--border-accent)" : "var(--border)"}`,
  };
}
function btnGhost(enabled: boolean): React.CSSProperties {
  return {
    display: "inline-flex", alignItems: "center", gap: 5,
    padding: "5px 12px", borderRadius: 999,
    border: "1px solid var(--border-strong)",
    background: "transparent",
    color: enabled ? "var(--text-primary)" : "var(--text-muted)",
    fontSize: 11, fontWeight: 700, cursor: enabled ? "pointer" : "not-allowed",
    opacity: enabled ? 1 : 0.5,
  };
}
function btnPrimary(enabled: boolean): React.CSSProperties {
  return {
    display: "inline-flex", alignItems: "center", gap: 5,
    padding: "5px 12px", borderRadius: 999, border: "none",
    background: enabled ? "var(--accent)" : "rgb(var(--ink) / 0.06)",
    color:      enabled ? "var(--on-accent)"      : "var(--text-muted)",
    fontSize: 11, fontWeight: 700, cursor: enabled ? "pointer" : "default",
  };
}
function pill(color: string): React.CSSProperties {
  return {
    display: "inline-flex", alignItems: "center", gap: 4,
    padding: "2px 8px", borderRadius: 999,
    background: `color-mix(in oklab, ${color} 15%, transparent)`,
    color, fontSize: 10, fontWeight: 700, whiteSpace: "nowrap",
  };
}
