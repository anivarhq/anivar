// Floating liquid-glass camera-settings window. Opened from the focus Camera
// view's "Settings" button. Consolidates the per-camera bits (name / source /
// enable / remove), the detection knobs that govern this camera's events, and
// the mask/zone editor — all in one floating pane so the user never leaves the
// live view to tune a camera.
//
// Texture matches the focus header buttons + the global `.glass` material so it
// reads as the same liquid glass used everywhere else.

import { useCallback, useEffect, useState } from "react";
import {
  Camera, SlidersHorizontal, Shapes, X, Trash2, Check,
  Info, Activity, Wifi, Film, RefreshCw, Loader2,
} from "lucide-react";

import { api, CameraConfig, StreamProbe } from "../../api";
import type { Settings, CameraTelemetry } from "../../types";
import { useStore } from "../../store";
import { useShallow } from "zustand/react/shallow";
import { MaskEditor } from "../cameras/MaskEditor";

import styles from "./CameraSettingsModal.module.css";

type Tab = "camera" | "info" | "masks";

// ── Telemetry formatters ──────────────────────────────────────────────────────
function fmtAgo(secs: number | null): string {
  if (secs == null) return "never";
  if (secs < 60) return `${secs}s ago`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m ago`;
  return `${Math.floor(secs / 3600)}h ago`;
}
// Pull host + credentials out of a stream URL for a best-effort ONVIF lookup.
function parseStreamHost(url: string): { host: string; user?: string; pass?: string } | null {
  const m = url.match(/^[a-z]+:\/\/(?:([^:/@]+)(?::([^@]*))?@)?([^:/?#]+)/i);
  if (!m) return null;
  return { user: m[1], pass: m[2], host: m[3] };
}

// ── Device-info presentational helpers ────────────────────────────────────────
function InfoRow({ k, v, mono }: { k: string; v: React.ReactNode; mono?: boolean }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 12, padding: "5px 0" }}>
      <span style={{ fontSize: 11.5, color: "var(--text-muted)", flexShrink: 0 }}>{k}</span>
      <span style={{ fontSize: 12, color: "var(--text-primary)", fontWeight: 600, textAlign: "right", wordBreak: "break-all", fontFamily: mono ? "var(--font-mono)" : "inherit" }}>{v}</span>
    </div>
  );
}
function InfoCard({ icon, title, children }: { icon: React.ReactNode; title: string; children: React.ReactNode }) {
  return (
    <div style={{ background: "var(--bg-base)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", padding: "10px 12px" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 10, fontWeight: 800, letterSpacing: "0.06em", textTransform: "uppercase", color: "var(--text-muted)", marginBottom: 4 }}>
        {icon}{title}
      </div>
      {children}
    </div>
  );
}
function StatusPill({ on, onText, offText }: { on: boolean; onText: string; offText: string }) {
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 5, fontSize: 11, fontWeight: 700,
      padding: "3px 9px", borderRadius: 20,
      background: on ? "var(--hl)" : "var(--bg-elevated)",
      color: on ? "var(--accent)" : "var(--text-muted)",
      border: `1px solid ${on ? "var(--border-accent)" : "var(--border)"}`,
    }}>
      {/* The dot that used to sit here encoded the same boolean as the label
          next to it, in the same colour. The word does the job. */}
      {on ? onText : offText}
    </span>
  );
}

interface Props {
  camId: number;
  /** Close the window. */
  onClose: () => void;
  /** Disable + remove this camera. Parent exits focus mode + reloads cams. */
  onRemoveCamera: () => void;
}

export function CameraSettingsModal({ camId, onClose, onRemoveCamera }: Props) {
  const { showToast, settings: storeSettings, setSettings } = useStore(useShallow(s => ({ showToast: s.showToast, settings: s.settings, setSettings: s.setSettings })));
  const [tab, setTab] = useState<Tab>("camera");

  // ── Per-camera config ──────────────────────────────────────────────────────
  const [cfg, setCfg]       = useState<CameraConfig | null>(null);
  const [savingCfg, setSavingCfg] = useState(false);

  // ── Detection (global Settings, but these govern THIS camera's events) ─────

  // ── Device info / telemetry ────────────────────────────────────────────────
  const [tel, setTel]         = useState<CameraTelemetry | null>(null);
  const [stream, setStream]   = useState<StreamProbe | null>(null);
  const [probing, setProbing] = useState(false);
  const [onvif, setOnvif]     = useState<{ manufacturer: string; model: string; firmware_version: string; serial_number: string } | null>(null);

  const loadTelemetry = useCallback(async () => {
    try { setTel(await api.getCameraTelemetry(camId)); } catch { /* ignore */ }
  }, [camId]);

  // Live stream stats (codec/res/fps) + best-effort ONVIF identity — on demand so
  // the panel paints instantly, then fills in.
  const probeLive = useCallback(async (url: string) => {
    if (!url) return;
    setProbing(true);
    try {
      const r = await api.probeStream(url, (cfg?.transport as "tcp" | "udp") ?? "tcp");
      setStream(r);
    } catch { setStream(null); }
    finally { setProbing(false); }
    // Best-effort ONVIF device info from the host in the URL.
    const h = parseStreamHost(url);
    if (h) {
      try {
        const info = await api.getOnvifDeviceInfo(`http://${h.host}/onvif/device_service`, h.user, h.pass);
        if (info && (info.manufacturer || info.model)) setOnvif(info);
      } catch { /* many cams won't answer — fine */ }
    }
  }, [cfg]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    api.getCameraConfigs()
      .then(cfgs => setCfg(cfgs.find(c => c.cam_id === camId) ?? {
        cam_id: camId, name: `Camera ${camId + 1}`,
        source_type: "native", source_url: "", device_id: "", enabled: true,
      }))
      .catch(() => {});
    return () => window.removeEventListener("keydown", onKey);
  }, [camId]);

  // Load telemetry whenever the Device Info tab is shown; probe the live stream
  // once we know the source. USB/native can't be probed (device is busy capturing).
  useEffect(() => {
    if (tab !== "info") return;
    loadTelemetry();
    if (cfg && (cfg.source_type === "rtsp" || cfg.source_type === "mjpeg") && cfg.source_url && !stream && !probing) {
      probeLive(cfg.source_url);
    }
  }, [tab, cfg, loadTelemetry, probeLive]); // eslint-disable-line react-hooks/exhaustive-deps

  const saveCfg = async () => {
    if (!cfg) return;
    setSavingCfg(true);
    try {
      await api.setCameraConfig(cfg);
      showToast("Camera settings saved", "success");
    } catch (e: any) { showToast(e.message ?? "Failed to save", "error"); }
    finally { setSavingCfg(false); }
  };


  // Masks tab embeds the existing full-screen MaskEditor (it has its own glass
  // shell + Esc handling). Returned at the TOP LEVEL so it replaces the whole
  // modal — rendering it INSIDE the glass window makes its `position:fixed`
  // resolve against the window's backdrop-filter containing block (squished).
  if (tab === "masks") {
    return <MaskEditor camId={camId} onClose={() => setTab("camera")} />;
  }

  return (
    <div className={styles.backdrop} onClick={e => { if (e.target === e.currentTarget) onClose(); }}>
      <div className={styles.window} role="dialog" aria-modal="true">
        {/* ── Title bar ── */}
        <div className={styles.titleBar}>
          <span className={styles.title}>
            <Camera size={14} /> Camera {camId + 1} settings
          </span>
          <button className={styles.closeBtn} onClick={onClose} aria-label="Close"><X size={15} /></button>
        </div>

        {/* ── Tabs ── */}
        <div className={styles.tabs}>
          <button className={`${styles.tab} ${tab === "camera" ? styles.tabActive : ""}`}
            onClick={() => setTab("camera")}>
            <Camera size={13} /> Camera
          </button>
          <button className={`${styles.tab} ${tab === "info" ? styles.tabActive : ""}`}
            onClick={() => setTab("info")}>
            <Info size={13} /> Device info
          </button>
          <button className={`${styles.tab}`} onClick={() => setTab("masks")}>
            <Shapes size={13} /> Masks & zones
          </button>
        </div>

        {/* ── Body ── */}
        <div className={styles.body}>
          {tab === "camera" && cfg && (
            <div className={styles.section}>
              <label className={styles.field}>
                <span className={styles.fieldLabel}>Name</span>
                <input className={styles.input} value={cfg.name}
                  onChange={e => setCfg({ ...cfg, name: e.target.value })} />
              </label>

              <label className={styles.field}>
                <span className={styles.fieldLabel}>Source type</span>
                <select className={styles.input} value={cfg.source_type === "browser" ? "native" : cfg.source_type}
                  onChange={e => setCfg({ ...cfg, source_type: e.target.value })}>
                  <option value="native">USB / Integrated camera</option>
                  <option value="rtsp">RTSP / IP camera</option>
                  <option value="mjpeg">MJPEG</option>
                </select>
              </label>

              <label className={styles.field}>
                <span className={styles.fieldLabel}>Brand / make <em style={{ fontStyle: "normal", color: "var(--text-muted)", fontWeight: 400 }}>(optional)</em></span>
                <input className={styles.input} value={cfg.brand ?? ""}
                  placeholder="e.g. Reolink, Hikvision, Lorex…"
                  onChange={e => setCfg({ ...cfg, brand: e.target.value })} />
              </label>

              {(cfg.source_type === "rtsp" || cfg.source_type === "mjpeg") && (
                <label className={styles.field}>
                  <span className={styles.fieldLabel}>Source URL</span>
                  <input className={styles.input} value={cfg.source_url}
                    placeholder="rtsp://user:pass@192.168.1.50:554/stream"
                    onChange={e => setCfg({ ...cfg, source_url: e.target.value })} />
                </label>
              )}
              {cfg.source_type === "rtsp" && (
                <label className={styles.field}>
                  <span className={styles.fieldLabel}>
                    Low-res detect stream <em style={{ fontStyle: "normal", color: "var(--text-muted)", fontWeight: 400 }}>(optional)</em>
                  </span>
                  <input className={styles.input} value={cfg.detect_url ?? ""}
                    placeholder="rtsp://…/stream2"
                    onChange={e => setCfg({ ...cfg, detect_url: e.target.value })} />
                </label>
              )}
              {(cfg.source_type === "browser" || cfg.source_type === "native") && (
                <label className={styles.field}>
                  <span className={styles.fieldLabel}>Device ID</span>
                  <input className={styles.input} value={cfg.device_id}
                    placeholder="device id / native index"
                    onChange={e => setCfg({ ...cfg, device_id: e.target.value })} />
                </label>
              )}

              <label className={styles.toggleRow}>
                <span className={styles.fieldLabel}>Enabled</span>
                <input type="checkbox" checked={cfg.enabled}
                  onChange={e => setCfg({ ...cfg, enabled: e.target.checked })} />
              </label>

              <div className={styles.actions}>
                <button className={`${styles.btn} ${styles.btnDanger}`}
                  onClick={() => {
                    if (window.confirm("Remove this camera? You can re-add it later in Settings.")) onRemoveCamera();
                  }}>
                  <Trash2 size={13} /> Remove camera
                </button>
                <button className={`${styles.btn} ${styles.btnPrimary}`} onClick={saveCfg} disabled={savingCfg}>
                  <Check size={13} /> {savingCfg ? "Saving…" : "Save"}
                </button>
              </div>
            </div>
          )}

          {tab === "info" && (
            <div className={styles.section} style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              {!tel ? (
                <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 8, padding: "30px 0", color: "var(--text-muted)", fontSize: 12 }}>
                  <Loader2 size={15} style={{ animation: "spin 1s linear infinite" }} /> Loading device info…
                </div>
              ) : (
                <>
                  {/* Status strip */}
                  <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
                    <StatusPill on={tel.online} onText="Online" offText="Offline" />
                    <StatusPill on={tel.recording} onText="Recording" offText="Not recording" />
                    <span style={{ marginLeft: "auto", fontSize: 11, color: "var(--text-muted)" }}>
                      <Activity size={11} style={{ verticalAlign: "-1px", marginRight: 4 }} />
                      last frame {fmtAgo(tel.last_frame_secs)}
                    </span>
                  </div>

                  {/* Identity */}
                  <InfoCard icon={<Info size={11} />} title="Identity">
                    <InfoRow k="Name" v={tel.name || `Camera ${camId + 1}`} />
                    <InfoRow k="Brand / make" v={onvif?.manufacturer || tel.brand || "—"} />
                    {onvif?.model && <InfoRow k="Model" v={onvif.model} />}
                    {onvif?.firmware_version && <InfoRow k="Firmware" v={onvif.firmware_version} mono />}
                    {onvif?.serial_number && <InfoRow k="Serial" v={onvif.serial_number} mono />}
                    <InfoRow k="Source" v={tel.source_type === "native" || tel.source_type === "browser" ? "USB / Integrated" : tel.source_type.toUpperCase()} />
                  </InfoCard>

                  {/* Live stream */}
                  <InfoCard icon={<Film size={11} />} title="Live stream">
                    {tel.source_type === "native" || tel.source_type === "browser" ? (
                      <div style={{ fontSize: 12, color: "var(--text-muted)", padding: "3px 0" }}>Local capture device — managed by the app.</div>
                    ) : probing ? (
                      <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12, color: "var(--text-muted)", padding: "3px 0" }}>
                        <Loader2 size={13} style={{ animation: "spin 1s linear infinite" }} /> Probing stream…
                      </div>
                    ) : stream?.ok ? (
                      <>
                        <InfoRow k="Resolution" v={`${stream.width}×${stream.height}`} />
                        <InfoRow k="Codec" v={stream.codec.toUpperCase()} />
                        <InfoRow k="Frame rate" v={stream.fps > 0 ? `${Math.round(stream.fps)} fps` : "—"} />
                        <InfoRow k="Audio" v={stream.has_audio ? (stream.audio_codec ? stream.audio_codec.toUpperCase() : "Yes") : "None"} />
                      </>
                    ) : (
                      <div style={{ fontSize: 12, color: stream ? "var(--accent-red)" : "var(--text-muted)", padding: "3px 0" }}>
                        {stream?.error || "Not probed yet."}
                      </div>
                    )}
                    {(tel.source_type === "rtsp" || tel.source_type === "mjpeg") && (
                      <button onClick={() => probeLive(cfg!.source_url)} disabled={probing}
                        style={{ marginTop: 6, display: "inline-flex", alignItems: "center", gap: 5, padding: "5px 10px", borderRadius: 16, border: "1px solid var(--border)", background: "var(--bg-elevated)", fontSize: 11, fontWeight: 600, color: "var(--text-secondary)", cursor: "pointer" }}>
                        <RefreshCw size={11} style={{ animation: probing ? "spin 1s linear infinite" : "none" }} /> Refresh
                      </button>
                    )}
                  </InfoCard>

                  {/* Network */}
                  <InfoCard icon={<Wifi size={11} />} title="Network">
                    {tel.source_url_masked
                      ? <InfoRow k="URL" v={tel.source_url_masked} mono />
                      : <InfoRow k="Device" v={tel.device_id || "auto"} mono />}
                    {(tel.source_type === "rtsp") && <InfoRow k="Transport" v={tel.transport.toUpperCase()} />}
                  </InfoCard>
                </>
              )}
            </div>
          )}

          {/* The "Detection" tab lived here. Every control on it — the sensitivity
              preset, "Continuous recording (NVR)" and segment length — wrote a
              GLOBAL settings key from a dialog titled "Camera N settings", and
              its own closing note admitted "these knobs are shared across all
              cameras". Changing one camera's sensitivity changed every camera's.
              They live in Settings, which is where a global belongs; segment
              length moved there too (it had no home there at all). Masks & zones
              stays — that one really is per-camera. */}
          </div>
      </div>
    </div>
  );
}
