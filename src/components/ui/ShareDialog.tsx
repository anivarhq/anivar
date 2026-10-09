/**
 * Share — a signed, expiring link (share_cmds.rs) to a camera's live view or an
 * event clip. Links open through remote access (Tailscale Funnel), so when that
 * isn't set up the dialog says so instead of offering a link that can't work.
 */
import { useEffect, useId, useState } from "react";
import { Check, Link2, X } from "lucide-react";
import { Modal } from "./Modal";
import { api } from "../../api";
import { useShallow } from "zustand/react/shallow";
import { useStore } from "../../store";
import { openExternal } from "../../lib/openExternal";

/** How long a link lives — the same choices as Settings' default. */
export const SHARE_EXPIRY = [
  { mins: 15,   label: "15 minutes" },
  { mins: 30,   label: "30 minutes" },
  { mins: 60,   label: "1 hour" },
  { mins: 1440, label: "24 hours" },
  { mins: 0,    label: "Until app restart" },
] as const;

export function ShareDialog({ kind, resourceId, title, onClose }: {
  kind: "live" | "clip";
  resourceId: string;
  title: string;
  onClose: () => void;
}) {
  const titleId = useId();
  const { defaultMins, setTab } = useStore(useShallow(s => ({ defaultMins: s.settings?.live_share_default_minutes, setTab: s.setTab })));
  const [remote, setRemote] = useState<"checking" | "on" | "off">("checking");
  const [mins, setMins] = useState<number>(defaultMins ?? 30);
  const [busy, setBusy] = useState(false);
  const [link, setLink] = useState<{ url: string; expiresAt: number; copied: boolean } | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.tailscaleStatus()
      .then(s => setRemote(s.installed && s.logged_in ? "on" : "off"))
      .catch(() => setRemote("off"));
  }, []);

  const create = async () => {
    setBusy(true); setError(null); setLink(null);
    try {
      const r = await api.generateShareLink(kind, resourceId, mins);
      const copied = await navigator.clipboard.writeText(r.url).then(() => true, () => false);
      setLink({ url: r.url, expiresAt: r.expires_at, copied });
    } catch (e) {
      setError(typeof e === "string" ? e : (e as Error)?.message ?? "Couldn't create the link.");
    } finally { setBusy(false); }
  };

  const setupUrl = error?.match(/https:\/\/\S+/)?.[0];
  const lasts = link &&(link.expiresAt === 0
    ? "Works until the app restarts."
    : `Works until ${new Date(link.expiresAt * 1000).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}.`);

  return (
    <Modal onClose={onClose} layer="top" labelledBy={titleId}>
      <div className="glass-strong"
        style={{ width: 380, maxWidth: "100%", padding: 18, display: "flex", flexDirection: "column", gap: 14 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <Link2 size={16} />
          <div id={titleId} style={{ flex: 1, fontWeight: 700, fontSize: 14 }}>{title}</div>
          <button onClick={onClose} aria-label="Close" style={{ background: "none", border: "none", cursor: "pointer",
            color: "var(--text-muted)", padding: 4 }}><X size={18} /></button>
        </div>

        {remote === "checking" && (
          <div style={{ fontSize: 12.5, color: "var(--text-muted)" }}>Checking remote access…</div>
        )}

        {remote === "off" && (
          <>
            <div style={{ fontSize: 12.5, lineHeight: 1.55, color: "var(--text-secondary)" }}>
              Share links open through remote access, which isn't on yet. Turn it on in
              Settings → Remote access, then share again.
            </div>
            <button className="btn-primary" style={{ padding: "9px 16px", justifyContent: "center" }}
              onClick={() => { setTab("settings"); onClose(); }}>
              Open Settings
            </button>
          </>
        )}

        {remote === "on" && (
          <>
            <div role="radiogroup" aria-label="Link lasts" style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
              {SHARE_EXPIRY.map(o => (
                <button key={o.mins} role="radio" aria-checked={mins === o.mins}
                  className={mins === o.mins ? "btn-primary" : "btn-secondary"}
                  style={{ padding: "6px 11px", fontSize: 12 }}
                  onClick={() => { setMins(o.mins); setLink(null); }}>
                  {o.label}
                </button>
              ))}
            </div>
            {link ? (
              <>
                <input readOnly value={link.url} aria-label="Share link" onFocus={e => e.currentTarget.select()}
                  style={{ width: "100%", padding: "8px 10px", borderRadius: 8, fontSize: 12,
                    border: "1px solid var(--border)", background: "var(--bg-elevated)", color: "var(--text-primary)" }} />
                <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, color: "var(--text-secondary)" }}>
                  {link.copied ? <><Check size={13} /> Copied.</> : "Select the link to copy it."} {lasts} Anyone with the link can open it.
                </div>
              </>
            ) : (
              <button className="btn-primary" disabled={busy} style={{ padding: "9px 16px", justifyContent: "center" }}
                onClick={create}>
                {busy ? "Creating link…" : "Copy link"}
              </button>
            )}
            {error && (
              <div role="alert" style={{ fontSize: 12, lineHeight: 1.5, color: "var(--status-alert)", whiteSpace: "pre-wrap" }}>
                {error}
                {/* Tailscale Funnel's one-time consent page, when that's what failed. */}
                {setupUrl && (
                  <button className="btn-secondary" style={{ display: "block", marginTop: 8, padding: "6px 12px", fontSize: 12 }}
                    onClick={() => openExternal(setupUrl)}>
                    Open setup page
                  </button>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </Modal>
  );
}
