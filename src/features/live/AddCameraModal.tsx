// Add-Camera modal — guided onboarding for USB / RTSP / MJPEG / ONVIF cameras
// (brand catalog with URL templates, LAN discovery, connection test). Extracted
// from LivePanel: it is a self-contained 3-prop modal with no grid coupling.
import { useEffect, useState } from "react";
import { useStore } from "../../store";
import { useShallow } from "zustand/react/shallow";
import { api } from "../../api";
import { writeCamSource } from "../../lib/camSource";
import { Wifi, Search, Camera, Monitor, CheckCircle2, Loader, RefreshCw, Link, X } from "lucide-react";
import styles from "./LivePanel.module.css";


// ── Brand catalog ─────────────────────────────────────────────────────────────
// Names only (no logos). The brand just preselects the right RTSP URL template +
// setup note; the connection type lives in `type`.

interface Brand {
  id: string; name: string;
  type: "rtsp" | "mjpeg" | "onvif";
  urlTemplate?: string;   // "" → user pastes a URL (custom / bridge)
  note?: string;          // setup gotcha shown when selected
}

const BRANDS: Brand[] = [
  { id: "reolink",   name: "Reolink",         type: "rtsp",  urlTemplate: "rtsp://{user}:{pass}@{ip}:554/h264Preview_01_main", note: "Main stream shown. Sub-stream: …/h264Preview_01_sub. Enable RTSP in the Reolink app if it's off." },
  { id: "hikvision", name: "Hikvision",       type: "rtsp",  urlTemplate: "rtsp://{user}:{pass}@{ip}:554/Streaming/Channels/101", note: "Sub-stream: Channels/102. Use a camera user that has RTSP rights." },
  { id: "dahua",     name: "Dahua",           type: "rtsp",  urlTemplate: "rtsp://{user}:{pass}@{ip}:554/cam/realmonitor?channel=1&subtype=0", note: "Sub-stream: subtype=1." },
  { id: "amcrest",   name: "Amcrest",         type: "rtsp",  urlTemplate: "rtsp://{user}:{pass}@{ip}:554/cam/realmonitor?channel=1&subtype=0", note: "Sub-stream: subtype=1." },
  { id: "axis",      name: "Axis",            type: "rtsp",  urlTemplate: "rtsp://{user}:{pass}@{ip}:554/axis-media/media.amp" },
  { id: "tapo",      name: "TP-Link Tapo",    type: "rtsp",  urlTemplate: "rtsp://{user}:{pass}@{ip}:554/stream1", note: "Create a Camera Account in the Tapo app (Advanced → Camera Account) and use THOSE credentials — not your TP-Link login. Sub-stream: stream2." },
  { id: "wyze",      name: "Wyze",            type: "rtsp",  urlTemplate: "rtsp://{user}:{pass}@{ip}:554/live", note: "Requires the official Wyze RTSP firmware, then enable RTSP in the app (V2/V3 only)." },
  { id: "uniview",   name: "Uniview",         type: "rtsp",  urlTemplate: "rtsp://{user}:{pass}@{ip}:554/unicast/c1/s0/live" },
  { id: "hanwha",    name: "Hanwha / Samsung",type: "rtsp",  urlTemplate: "rtsp://{user}:{pass}@{ip}:554/profile2/media.smp" },
  { id: "blink",     name: "Blink",           type: "rtsp",  urlTemplate: "", note: "Blink is cloud-only (no local RTSP). 1) Add Blink to Scrypted or Home Assistant. 2) Enable its RTSP/RTSP-rebroadcast. 3) Paste that rtsp:// URL below." },
  { id: "ring",      name: "Ring",            type: "rtsp",  urlTemplate: "", note: "Ring is cloud-only (no official RTSP). 1) Add Ring to Scrypted (or Home Assistant + ring-mqtt). 2) Enable the RTSP rebroadcast. 3) Paste that rtsp:// URL below." },
  { id: "onvif",     name: "ONVIF Generic",   type: "onvif", urlTemplate: "" },
  { id: "rtsp",      name: "Custom RTSP",     type: "rtsp",  urlTemplate: "" },
  { id: "mjpeg",     name: "MJPEG / HTTP",    type: "mjpeg", urlTemplate: "" },
  { id: "other",     name: "Other…",          type: "rtsp",  urlTemplate: "", note: "Enter your camera's brand name and its RTSP (or MJPEG) URL below." },
];

// ── Smart Add Camera Modal ────────────────────────────────────────────────────

export type ModalTab = "discover" | "ip" | "usb";

export function AddCameraModal({ nextSlotId, onClose, onAdded, initialTab }: {
  nextSlotId: number;
  onClose: () => void;
  onAdded: (cfg: { cam_id: number; name: string; source_type: string; source_url: string; device_id: string; enabled: boolean }) => void;
  /** Open straight onto one path. The empty Live grid offers all three as
   *  cards, so picking one there should not land the user on a fourth screen
   *  they then have to navigate out of. */
  initialTab?: ModalTab;
}) {
  const { showToast } = useStore(useShallow(s => ({ showToast: s.showToast })));
  const [tab, setTab] = useState<ModalTab>(initialTab ?? "discover");

  // Discover tab state
  const [discovering, setDiscovering] = useState(false);
  const [discovered,  setDiscovered]  = useState<any[]>([]);
  const [discoverErr, setDiscoverErr] = useState("");
  // Optional credentials applied to ONVIF discovery so found cameras come back with
  // a ready-to-use RTSP URL (most cameras require auth to hand out their stream URI).
  const [discUser, setDiscUser] = useState("");
  const [discPass, setDiscPass] = useState("");

  // IP tab state
  const [brand,    setBrand]    = useState<Brand | null>(null);
  const [draft,    setDraft]    = useState({ name: "", ip: "", user: "admin", pass: "", url: "", mjpegUser: "", mjpegPass: "", customBrand: "" });
  const [testing,  setTesting]  = useState(false);
  const [testOk,   setTestOk]   = useState<boolean | null>(null);
  const [testInfo, setTestInfo] = useState("");

  // USB tab state — integrated/USB cameras are captured server-side (native) only.
  const [nativeCams,  setNativeCams]  = useState<{ index: number; name: string; description: string }[]>([]);
  /** Devices exist but every one is already claimed by an enabled slot. */
  const [allClaimed,  setAllClaimed]  = useState(false);
  const [selDevice,   setSelDevice]   = useState({ deviceId: "", name: "", isNative: false, nativeIndex: 0 });

  const [saving, setSaving] = useState(false);

  // ── Setup ────────────────────────────────────────────────────────────────────
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);

    // Load integrated/USB cameras — captured SERVER-SIDE (native), which records
    // 24/7 regardless of which page is open. Devices already claimed by an
    // ENABLED slot are filtered OUT of the picker: a physical camera has one
    // owner — a second capture would just fail with "device in use". Claims are
    // matched by device_id (index or name — dshow resolves both) AND by name,
    // since indexes can shift as devices are plugged/unplugged.
    Promise.all([
      api.listNativeCameras().catch(() => [] as { index: number; name: string; description: string }[]),
      api.getCameraConfigs().catch(() => []),
    ]).then(([nc, cfgs]) => {
      const claimed = new Set<string>();
      for (const c of cfgs) {
        if (!c.enabled || (c.source_type !== "native" && c.source_type !== "browser")) continue;
        if (c.device_id) claimed.add(String(c.device_id).toLowerCase());
        if (c.name) claimed.add(c.name.toLowerCase());
      }
      const avail = nc.filter(cam =>
        !claimed.has(String(cam.index)) && !claimed.has(cam.name.toLowerCase()));
      setAllClaimed(nc.length > 0 && avail.length === 0);
      setNativeCams(avail);
      if (avail.length > 0) {
        setSelDevice(prev => prev.deviceId || prev.isNative ? prev
          : { deviceId: "", name: avail[0].name, isNative: true, nativeIndex: avail[0].index });
      }
    }).catch(() => {});

    // Auto-scan on open
    runDiscover();

    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // ── Discover ─────────────────────────────────────────────────────────────────
  // One scan, two methods: ONVIF WS-Discovery (returns a real, ready-to-use RTSP
  // URL + make/model) AND a network port scan (finds non-ONVIF devices). Merged and
  // de-duped by IP, ONVIF preferred. ONVIF hits add in one click; everything else
  // hands off to manual setup (so we never save a guessed/broken URL).
  const runDiscover = async () => {
    setDiscovering(true); setDiscoverErr(""); setDiscovered([]);
    try {
      const [onvifRes, scanRes] = await Promise.allSettled([
        api.discoverAndConfigureOnvif(discUser.trim() || undefined, discPass || undefined, 4000),
        api.discoverCameras(),
      ]);

      const items: any[] = [];
      const onvifIps = new Set<string>();
      if (onvifRes.status === "fulfilled") {
        for (const s of onvifRes.value) {
          onvifIps.add(s.source_ip);
          const title = [s.manufacturer, s.model].filter(Boolean).join(" ") || "ONVIF Camera";
          items.push({ kind: "onvif", title, ip: s.source_ip, sub: `${s.source_ip} · ONVIF`, url: s.rtsp_url, needsAuth: !discUser.trim() });
        }
      }
      if (scanRes.status === "fulfilled") {
        for (const c of scanRes.value) {
          if (onvifIps.has(c.ip)) continue; // ONVIF already gave us a better entry for this device
          items.push({ kind: c.kind === "mjpeg" || c.kind === "http" ? "mjpeg" : "rtsp", title: `Device @ ${c.ip}`, ip: c.ip, sub: `${c.ip}:${c.port} · ${c.kind.toUpperCase()}`, port: c.port });
        }
      }
      setDiscovered(items);
      if (items.length === 0) {
        const allFailed = onvifRes.status === "rejected" && scanRes.status === "rejected";
        setDiscoverErr(allFailed
          ? "Scan failed — make sure cameras are on the same network."
          : "No cameras found. Add yours manually on the IP Camera tab.");
      }
    } catch {
      setDiscoverErr("Scan failed — make sure cameras are on the same network.");
    } finally { setDiscovering(false); }
  };

  // Send a discovered (non-ONVIF) device to the manual tab with its IP prefilled.
  const setUpDiscovered = (item: any) => {
    setDraft(d => ({ ...d, ip: item.ip, url: item.kind === "mjpeg" && item.port ? `http://${item.ip}:${item.port}/` : "" }));
    setBrand(null);
    setTab("ip");
  };

  // ── Save helpers ─────────────────────────────────────────────────────────────
  const buildUrl = () => {
    if (!brand?.urlTemplate) return draft.url;
    return brand.urlTemplate
      .replace("{user}", encodeURIComponent(draft.user))
      .replace("{pass}", encodeURIComponent(draft.pass))
      .replace("{ip}",   draft.ip)
      .replace("{ch}",   "1");
  };

  const testConnection = async () => {
    const url = buildUrl();
    if (!url) return;
    setTesting(true); setTestOk(null); setTestInfo("");
    try {
      // ffprobe-based test: fast, and reports the real codec/resolution/fps.
      const r = await api.probeStream(url, "tcp");
      setTestOk(r.ok);
      setTestInfo(r.ok
        ? `${r.width}×${r.height}${r.fps > 0 ? ` · ${Math.round(r.fps)}fps` : ""} · ${r.codec.toUpperCase()}${r.has_audio ? " · audio" : ""}`
        : r.error);
    } catch { setTestOk(false); setTestInfo("Test failed"); }
    finally { setTesting(false); }
  };

  const saveCamera = async (overrides?: Partial<{ name: string; source_type: string; source_url: string; device_id: string; brand: string }>) => {
    setSaving(true);
    try {
      // USB/integrated cameras are captured server-side (native) only — no browser path.
      let source_type = "rtsp", source_url = buildUrl(), device_id = "";
      if (tab === "usb") {
        source_type = "native"; source_url = ""; device_id = String(selDevice.nativeIndex);
      }
      // Brand: the picked make, the typed name for "Other…", blank for generic/USB.
      const resolvedBrand = tab !== "ip" ? ""
        : brand?.id === "other" ? draft.customBrand.trim()
        : (brand && !["rtsp", "mjpeg", "onvif", "other"].includes(brand.id)) ? brand.name
        : "";
      const cfg = {
        cam_id:      nextSlotId,
        name:        overrides?.name ?? (draft.name || (brand?.name ?? `Camera ${nextSlotId + 1}`)),
        source_type: overrides?.source_type ?? source_type,
        source_url:  overrides?.source_url  ?? source_url,
        device_id:   overrides?.device_id   ?? device_id,
        enabled:     true,
        brand:       overrides?.brand ?? resolvedBrand,
      };
      await api.setCameraConfig(cfg);
      // Auto-start via localStorage (SSOT mapping in lib/camSource). MJPEG cameras
      // also stash Basic-Auth creds, which only exist at add time (not in the config).
      const extra: Record<string, unknown> = {};
      if (cfg.source_type === "mjpeg") {
        const authUser = tab === "ip" && brand?.type === "mjpeg" ? draft.mjpegUser : draft.user;
        const authPass = tab === "ip" && brand?.type === "mjpeg" ? draft.mjpegPass : draft.pass;
        if (authUser) extra.authUser = authUser;
        if (authPass) extra.authPass = authPass;
      }
      writeCamSource(nextSlotId, cfg, extra);
      showToast(`${cfg.name} added`, "success");
      onAdded(cfg); onClose();
    } catch (e: any) { showToast(e.message ?? "Failed to add camera", "error"); }
    finally { setSaving(false); }
  };

  const maskedUrl = brand ? buildUrl().replace(encodeURIComponent(draft.pass || ""), "•••") : "";

  // ── Render ───────────────────────────────────────────────────────────────────
  return (
    <div className={styles.modalBackdrop} onClick={e => { if (e.target === e.currentTarget) onClose(); }}>
      <div className={styles.modal} style={{ maxWidth: 560 }}>

        {/* Header */}
        <div className={styles.modalHeader}>
          <span className={styles.modalTitle}>
            <Camera size={14} style={{ display: "inline", marginRight: 6, verticalAlign: "middle" }} />
            Add Camera — Slot {nextSlotId + 1}
          </span>
          <button className={styles.modalClose} onClick={onClose}><X size={15} /></button>
        </div>

        {/* Tab bar */}
        <div style={{ display: "flex", borderBottom: "1px solid var(--border)", background: "var(--bg-surface)" }}>
          {([
            { id: "discover", icon: <Wifi size={13} />,    label: "Auto-Discover" },
            { id: "ip",       icon: <Link size={13} />,    label: "IP Camera" },
            { id: "usb",      icon: <Monitor size={13} />, label: "Integrated / USB" },
          ] as const).map(t => (
            <button key={t.id}
              onClick={() => setTab(t.id)}
              style={{
                flex: 1, display: "flex", alignItems: "center", justifyContent: "center", gap: 6,
                padding: "10px 0", fontSize: 12, fontWeight: 700,
                border: "none", background: "none", cursor: "pointer",
                color: tab === t.id ? "var(--accent)" : "var(--text-muted)",
                borderBottom: tab === t.id ? "2px solid var(--accent)" : "2px solid transparent",
                transition: "color 0.15s, border-color 0.15s",
              }}>
              {t.icon} {t.label}
            </button>
          ))}
        </div>

        <div className={styles.modalBody}>

          {/* ══ AUTO-DISCOVER TAB ══════════════════════════════════════════════ */}
          {tab === "discover" && (
            <div>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10, marginBottom: 10 }}>
                <div style={{ fontSize: 12, color: "var(--text-muted)" }}>
                  Finds cameras on your network via ONVIF + a port scan.
                </div>
                <button onClick={runDiscover} disabled={discovering}
                  style={{ display: "flex", alignItems: "center", gap: 5, padding: "5px 12px", borderRadius: 20, border: "1px solid var(--border)", background: "var(--bg-elevated)", fontSize: 12, fontWeight: 600, cursor: "pointer", flexShrink: 0 }}>
                  <RefreshCw size={12} style={{ animation: discovering ? "spin 1s linear infinite" : "none" }} />
                  {discovering ? "Scanning…" : "Scan Again"}
                </button>
              </div>

              {/* Optional credentials → ONVIF returns ready-to-use stream URLs */}
              <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
                <input className={styles.cfgIn} placeholder="Camera username (optional)" value={discUser}
                  autoComplete="off" onChange={e => setDiscUser(e.target.value)} style={{ flex: 1 }} />
                <input className={styles.cfgIn} type="password" placeholder="Password (optional)" value={discPass}
                  autoComplete="off" onChange={e => setDiscPass(e.target.value)} style={{ flex: 1 }} />
              </div>

              {discovering && (
                <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 12, padding: "28px 0" }}>
                  <div style={{ width: 40, height: 40, borderRadius: "50%", border: "3px solid var(--border)", borderTopColor: "var(--accent)", animation: "spin 0.8s linear infinite" }} />
                  <div style={{ fontSize: 13, color: "var(--text-muted)" }}>Scanning network… this takes 5–10 seconds</div>
                </div>
              )}

              {!discovering && discoverErr && (
                <div style={{ padding: "18px 0", textAlign: "center" }}>
                  <Wifi size={26} color="var(--text-muted)" style={{ marginBottom: 8 }} />
                  <div style={{ fontSize: 13, color: "var(--text-muted)", lineHeight: 1.6 }}>{discoverErr}</div>
                  <button onClick={() => setTab("ip")} style={{ marginTop: 12, padding: "6px 16px", borderRadius: 20, background: "var(--accent-glow)", border: "1px solid var(--border-accent)", color: "var(--accent)", fontSize: 12, fontWeight: 700, cursor: "pointer" }}>
                    Add manually →
                  </button>
                </div>
              )}

              {!discovering && discovered.length > 0 && (
                <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                  {discovered.map((cam, i) => (
                    <div key={i} style={{
                      display: "flex", alignItems: "center", gap: 12,
                      padding: "10px 14px", borderRadius: "var(--radius-md)",
                      background: "var(--bg-elevated)", border: "1px solid var(--border)",
                    }}>
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ fontWeight: 700, fontSize: 13, display: "flex", alignItems: "center", gap: 6 }}>
                          {cam.title}
                          {cam.kind === "onvif" && <span style={{ fontSize: 9, fontWeight: 800, letterSpacing: "0.04em", padding: "1px 5px", borderRadius: 4, background: "var(--hl)", color: "var(--text-secondary)" }}>ONVIF</span>}
                        </div>
                        <div style={{ fontSize: 11, color: "var(--text-muted)", fontFamily: "var(--font-mono)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{cam.sub}</div>
                        {cam.kind === "onvif" && cam.needsAuth && (
                          <div style={{ fontSize: 10, color: "var(--accent-amber)", marginTop: 2 }}>Needs a password? Enter it above and Scan Again.</div>
                        )}
                      </div>
                      {cam.kind === "onvif" ? (
                        <button
                          onClick={() => saveCamera({ name: cam.title, source_type: "rtsp", source_url: cam.url })}
                          disabled={saving || !cam.url}
                          style={{ flexShrink: 0, padding: "6px 14px", borderRadius: 20, background: "var(--accent-fill)", color: "var(--on-accent)", fontSize: 12, fontWeight: 700, border: "none", cursor: "pointer" }}>
                          {saving ? "…" : "Add"}
                        </button>
                      ) : (
                        <button
                          onClick={() => setUpDiscovered(cam)}
                          style={{ flexShrink: 0, padding: "6px 14px", borderRadius: 20, background: "var(--bg-base)", color: "var(--text-primary)", fontSize: 12, fontWeight: 700, border: "1px solid var(--border)", cursor: "pointer" }}>
                          Set up →
                        </button>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* ══ IP CAMERA TAB ══════════════════════════════════════════════════ */}
          {tab === "ip" && (
            <div className={styles.configForm}>

              {/* Brand picker — scannable grid of name cards */}
              <div className={styles.cfgLbl} style={{ marginBottom: 0 }}>Camera Brand / Type</div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 8, marginTop: 4 }}>
                {BRANDS.map(b => {
                  const active = brand?.id === b.id;
                  return (
                    <button key={b.id}
                      onClick={() => { setBrand(b); setDraft(d => ({ ...d, name: d.name || b.name })); setTestOk(null); setTestInfo(""); }}
                      title={b.name}
                      style={{
                        position: "relative", display: "flex", alignItems: "center", justifyContent: "center",
                        padding: "12px 8px", borderRadius: "var(--radius-md)",
                        border: `1px solid ${active ? "var(--accent)" : "var(--border)"}`,
                        background: active ? "var(--hl)" : "var(--bg-elevated)",
                        cursor: "pointer", minHeight: 46,
                        fontSize: 12, fontWeight: 600, lineHeight: 1.2, textAlign: "center",
                        color: active ? "var(--accent)" : "var(--text-primary)",
                      }}>
                      {b.name}
                      {active && <CheckCircle2 size={12} color="var(--accent)" style={{ position: "absolute", top: 5, right: 5 }} />}
                    </button>
                  );
                })}
              </div>

              {brand?.note && (
                <div style={{ padding: "10px 12px", borderRadius: "var(--radius-sm)", background: "rgba(251,146,60,0.08)", border: "1px solid rgba(251,146,60,0.25)", fontSize: 12, color: "var(--accent-amber)", lineHeight: 1.6 }}>
                  ⚠ {brand.note}
                </div>
              )}

              {brand?.id === "other" && (
                <label className={styles.cfgLbl}>Brand name
                  <input className={styles.cfgIn} value={draft.customBrand}
                    onChange={e => setDraft(d => ({ ...d, customBrand: e.target.value }))}
                    placeholder="e.g. Lorex, Swann, Annke…" />
                </label>
              )}

              <label className={styles.cfgLbl}>Camera Name
                <input className={styles.cfgIn} value={draft.name} autoFocus
                  onChange={e => setDraft(d => ({ ...d, name: e.target.value }))}
                  placeholder={brand?.name ?? "e.g. Front Door"} />
              </label>

              {/* IP + credentials (branded cameras) */}
              {brand && brand.urlTemplate !== undefined && brand.urlTemplate !== "" && (
                <>
                  <label className={styles.cfgLbl}>IP Address
                    <input className={styles.cfgIn} value={draft.ip}
                      onChange={e => setDraft(d => ({ ...d, ip: e.target.value }))}
                      placeholder="192.168.1.100" />
                  </label>
                  <div className={styles.cfgRow}>
                    <label className={styles.cfgLbl}>Username
                      <input className={styles.cfgIn} value={draft.user}
                        onChange={e => setDraft(d => ({ ...d, user: e.target.value }))} />
                    </label>
                    <label className={styles.cfgLbl}>Password
                      <input className={styles.cfgIn} type="password" value={draft.pass}
                        onChange={e => setDraft(d => ({ ...d, pass: e.target.value }))} />
                    </label>
                  </div>
                  {draft.ip && (
                    <div className={styles.generatedUrl}>
                      <span className={styles.generatedUrlLbl}>Stream URL (auto-generated)</span>
                      <code className={styles.generatedUrlCode}>{maskedUrl}</code>
                    </div>
                  )}
                </>
              )}

              {/* Manual URL (custom RTSP, MJPEG, ONVIF without template) */}
              {brand && (brand.id === "rtsp" || brand.id === "mjpeg" || brand.id === "onvif" || brand.urlTemplate === "") && (
                <>
                  <label className={styles.cfgLbl}>Stream URL
                    <input className={styles.cfgIn} value={draft.url}
                      onChange={e => setDraft(d => ({ ...d, url: e.target.value }))}
                      placeholder={brand.type === "mjpeg" ? "http://192.168.1.100:8080" : "rtsp://192.168.1.100:554/stream"} />
                  </label>
                  {/* Optional credentials for password-protected cameras */}
                  <div style={{ display: "flex", gap: 8, alignItems: "flex-end" }}>
                    <label className={styles.cfgLbl} style={{ flex: 1 }}>Username <span style={{ color: "var(--text-muted)", fontWeight: 400 }}>(optional)</span>
                      <input className={styles.cfgIn}
                        value={brand.type === "mjpeg" ? draft.mjpegUser : draft.user}
                        onChange={e => brand.type === "mjpeg"
                          ? setDraft(d => ({ ...d, mjpegUser: e.target.value }))
                          : setDraft(d => ({ ...d, user: e.target.value }))}
                        placeholder="admin" />
                    </label>
                    <label className={styles.cfgLbl} style={{ flex: 1 }}>Password <span style={{ color: "var(--text-muted)", fontWeight: 400 }}>(optional)</span>
                      <input className={styles.cfgIn} type="password"
                        value={brand.type === "mjpeg" ? draft.mjpegPass : draft.pass}
                        onChange={e => brand.type === "mjpeg"
                          ? setDraft(d => ({ ...d, mjpegPass: e.target.value }))
                          : setDraft(d => ({ ...d, pass: e.target.value }))}
                        placeholder="••••••" />
                    </label>
                  </div>
                </>
              )}

              {/* Test + Add */}
              {brand && (
                <div style={{ display: "flex", gap: 8, marginTop: 4 }}>
                  <button onClick={testConnection} disabled={testing || !buildUrl()}
                    style={{ flex: "0 0 auto", display: "flex", alignItems: "center", gap: 6, padding: "8px 14px", borderRadius: 20, border: "1px solid var(--border)", background: "var(--bg-elevated)", fontSize: 12, fontWeight: 600, cursor: "pointer", color: testOk === true ? "var(--accent)" : testOk === false ? "var(--accent-red)" : "var(--text-secondary)" }}>
                    {testing ? <Loader size={12} style={{ animation: "spin 1s linear infinite" }} /> : testOk === true ? <CheckCircle2 size={12} /> : <Search size={12} />}
                    {testing ? "Testing…" : testOk === true ? "Connected!" : testOk === false ? "Failed" : "Test"}
                  </button>
                  <button className={styles.addBtn} style={{ flex: 1 }} onClick={() => saveCamera()} disabled={saving || !brand || (brand.urlTemplate !== "" && !draft.ip && !draft.url)}>
                    {saving ? "Adding…" : `Add ${draft.name || brand.name}`}
                  </button>
                </div>
              )}

              {brand && testInfo && (
                <div style={{ fontSize: 11.5, lineHeight: 1.5, color: testOk ? "var(--accent)" : "var(--accent-red)", fontFamily: testOk ? "var(--font-mono)" : "inherit" }}>
                  {testOk ? "✓ " : "✕ "}{testInfo}
                </div>
              )}

              {!brand && (
                <div style={{ padding: "20px 0", textAlign: "center", color: "var(--text-muted)", fontSize: 13 }}>
                  ↑ Select a brand to get started
                </div>
              )}
            </div>
          )}

          {/* ══ INTEGRATED / USB TAB ═══════════════════════════════════════════ */}
          {tab === "usb" && (
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              <div style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 4 }}>
                Select an integrated or USB camera:
              </div>

              {/* Integrated / USB cameras — captured server-side (records 24/7) */}
              {nativeCams.length > 0 && (
                <>
                  {nativeCams.map(cam => (
                    <div key={cam.index} onClick={() => setSelDevice({ deviceId: String(cam.index), name: cam.name, isNative: true, nativeIndex: cam.index })}
                      style={{ display: "flex", alignItems: "center", gap: 12, padding: "10px 14px", borderRadius: "var(--radius-md)", border: `1px solid ${selDevice.isNative && selDevice.nativeIndex === cam.index ? "var(--accent)" : "var(--border)"}`, background: selDevice.isNative && selDevice.nativeIndex === cam.index ? "var(--hl)" : "var(--bg-elevated)", cursor: "pointer" }}>
                      <Camera size={18} color="var(--accent)" />
                      <div style={{ flex: 1 }}>
                        <div style={{ fontWeight: 600, fontSize: 13 }}>{cam.name}</div>
                        <div style={{ fontSize: 11, color: "var(--text-muted)" }}>{cam.description || "Records 24/7 — low latency"}</div>
                      </div>
                      {selDevice.isNative && selDevice.nativeIndex === cam.index && <CheckCircle2 size={16} color="var(--accent)" />}
                    </div>
                  ))}
                </>
              )}

              {nativeCams.length === 0 && (
                <div style={{ padding: "28px 0", textAlign: "center" }}>
                  <Camera size={30} color="var(--text-muted)" style={{ marginBottom: 8 }} />
                  <div style={{ fontSize: 13, color: "var(--text-muted)" }}>
                    {allClaimed
                      ? "Your integrated/USB camera is already added to a slot — a physical camera can only be used by one slot at a time."
                      : "No integrated or USB cameras detected. Connect one and try again."}
                  </div>
                </div>
              )}

              {(selDevice.isNative) && (
                <>
                  <label className={styles.cfgLbl} style={{ marginTop: 8 }}>Camera Name
                    <input className={styles.cfgIn} value={draft.name}
                      onChange={e => setDraft(d => ({ ...d, name: e.target.value }))}
                      placeholder={selDevice.name || "Integrated Camera"} autoFocus />
                  </label>
                  <button className={styles.addBtn} onClick={() => saveCamera({ name: draft.name || selDevice.name })} disabled={saving}>
                    {saving ? "Adding…" : `Add ${draft.name || selDevice.name || "Camera"}`}
                  </button>
                </>
              )}
            </div>
          )}

        </div>
      </div>
    </div>
  );
}
