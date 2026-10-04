import { useEffect, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { getVersion } from "@tauri-apps/api/app";
import { AlertCircle, ArrowDownCircle, CheckCircle2, ExternalLink, RefreshCw } from "lucide-react";
import { api, type UpdateInfo } from "../../api";
import { openExternal } from "../../lib/openExternal";
import styles from "./SettingsPanel.module.css";

type Phase =
  | { k: "idle" }
  | { k: "checking" }
  | { k: "current"; info: UpdateInfo; at: Date }
  | { k: "available"; info: UpdateInfo }
  | { k: "downloading"; info?: UpdateInfo; got: number; total: number | null }
  | { k: "installing"; latest?: string }
  | { k: "error"; title: string; detail: string; info?: UpdateInfo };

// The last result outlives the Settings panel, so reopening it doesn't ask
// GitHub again or forget an update it already found.
let remembered: Phase = { k: "idle" };

/** The changelog's top-level bullets, by section ("Added" → "New"). The
 *  manifest's notes are the whole release page — download steps, the
 *  SmartScreen walkthrough — so everything before "## Changes" is dropped. */
export function releaseHighlights(notes: string): { heading: string; items: string[] }[] {
  const at = notes.indexOf("## Changes");
  const body = (at >= 0 ? notes.slice(at) : notes).replace(/\r\n/g, "\n");
  const groups: { heading: string; items: string[] }[] = [];
  for (const m of body.matchAll(/^(?:###\s+(.+)|- (?:\*\*([\s\S]+?)\*\*|(.+)))/gm)) {
    if (m[1]) { groups.push({ heading: m[1].trim() === "Added" ? "New" : m[1].trim(), items: [] }); continue; }
    const text = (m[2] ?? m[3]).replace(/\s+/g, " ").replace(/[*`]/g, "").replace(/[.:]$/, "").trim();
    if (!groups.length) groups.push({ heading: "", items: [] });
    groups[groups.length - 1].items.push(text);
  }
  return groups.filter(g => g.items.length);
}

/** Splits "Couldn't check for updates: <reason>" and says a network failure
 *  in words a person can act on. */
function failure(e: unknown, info?: UpdateInfo): Phase {
  const msg = String((e as any)?.message ?? e);
  const cut = msg.indexOf(": ");
  const title = cut > 0 ? msg.slice(0, cut) : "Something went wrong";
  const reason = cut > 0 ? msg.slice(cut + 2) : msg;
  const offline = /sending request|connect|dns|timed? ?out|network|resolve/i.test(reason);
  return { k: "error", title, detail: offline ? "Can't reach GitHub. Check your internet connection." : reason, info };
}

const mb = (b: number) => (b / 1048576).toFixed(1);
const releaseUrl = (v?: string) =>
  v ? `https://github.com/anivarhq/anivar/releases/tag/v${v}` : "https://github.com/anivarhq/anivar/releases/latest";
const released = (d?: string | null) => {
  const [y, m, day] = (d ?? "").split("-").map(Number);
  return y ? new Date(y, m - 1, day).toLocaleDateString(undefined, { day: "numeric", month: "short" }) : null;
};

const MAX_ITEMS = 8;

export function UpdateStatus({ autoCheck }: { autoCheck: boolean }) {
  const [phase, setPhaseState] = useState<Phase>(remembered);
  const [version, setVersion] = useState("");
  const setPhase = (p: Phase) => { remembered = p; setPhaseState(p); };

  const check = async () => {
    setPhase({ k: "checking" });
    try {
      const info = await api.updateCheck();
      setPhase(info.available ? { k: "available", info } : { k: "current", info, at: new Date() });
    } catch (e) { setPhase(failure(e)); }
  };

  // On success the app exits into the installer (Windows) or restarts, so
  // only a failure ever comes back here.
  const install = async (info: UpdateInfo) => {
    setPhase({ k: "downloading", info, got: 0, total: null });
    try { await api.updateInstall(); }
    catch (e) { setPhase(failure(e, info)); }
  };

  useEffect(() => {
    getVersion().then(setVersion).catch(() => {});
    if (autoCheck && remembered.k === "idle") check();
    // The automatic path (update_cmds.rs) downloads and installs on its own;
    // its progress shows here too, whoever started it.
    const known = () => ("info" in remembered ? remembered.info : undefined);
    const offs = [
      listen<{ downloaded: number; total: number | null }>("update:progress", e =>
        setPhase({ k: "downloading", info: known(), got: e.payload.downloaded, total: e.payload.total })),
      listen<{ latest: string }>("update:installing", e => setPhase({ k: "installing", latest: e.payload.latest })),
    ];
    return () => { offs.forEach(p => p.then(f => f())); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const info = "info" in phase ? phase.info : undefined;
  const latest = phase.k === "installing" ? phase.latest : info?.latest;
  const busy = phase.k === "checking" || phase.k === "downloading" || phase.k === "installing";
  const pct = phase.k === "downloading" && phase.total ? Math.min(100, Math.round(phase.got / phase.total * 100)) : null;

  let icon, title: string, sub: string;
  switch (phase.k) {
    case "idle":
      icon = <RefreshCw size={18} />;
      title = `Anivar NVR ${version}`;
      sub = "Not checked for updates yet";
      break;
    case "checking":
      icon = <RefreshCw size={18} style={{ animation: "spin 1s linear infinite" }} />;
      title = `Anivar NVR ${version}`;
      sub = "Checking for updates…";
      break;
    case "current":
      icon = <CheckCircle2 size={18} color="var(--status-ok)" />;
      title = "Anivar NVR is up to date";
      sub = `Version ${phase.info.current} · checked at ${phase.at.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" })}`;
      break;
    case "available": {
      const date = released(phase.info.date);
      icon = <ArrowDownCircle size={18} color="var(--accent)" />;
      title = `Version ${phase.info.latest} is available`;
      sub = `You have ${phase.info.current}${date ? ` · released ${date}` : ""}`;
      break;
    }
    case "downloading":
      icon = <ArrowDownCircle size={18} color="var(--accent)" />;
      title = latest ? `Downloading ${latest}` : "Downloading the update";
      sub = phase.total ? `${pct}% · ${mb(phase.got)} of ${mb(phase.total)} MB`
          : phase.got ? `${mb(phase.got)} MB` : "Starting…";
      break;
    case "installing":
      icon = <RefreshCw size={18} color="var(--accent)" style={{ animation: "spin 1s linear infinite" }} />;
      title = latest ? `Installing ${latest}` : "Installing the update";
      sub = "Anivar will close, install and reopen by itself.";
      break;
    case "error":
      icon = <AlertCircle size={18} color="var(--status-alert)" />;
      title = phase.title;
      sub = phase.detail;
      break;
  }

  const highlights = info?.available && phase.k !== "installing" ? releaseHighlights(info.notes ?? "") : [];
  const total = highlights.reduce((n, g) => n + g.items.length, 0);
  let left = MAX_ITEMS;

  return (
    <div className={styles.updCard}>
      <div className={styles.updRow}>
        <span className={styles.updIcon}>{icon}</span>
        <div className={styles.updText} aria-live="polite">
          <span className={styles.updTitle}>{title}</span>
          <span className={styles.updSub}>{sub}</span>
        </div>
        {phase.k === "available" || (phase.k === "error" && info?.available) ? (
          <button className={styles.saveBtn} onClick={() => install(info!)}>
            {phase.k === "error" ? "Try again" : "Install and restart"}
          </button>
        ) : !busy && (
          <button className={styles.ghostBtn} onClick={check}>
            {phase.k === "error" ? "Try again" : phase.k === "current" ? "Check again" : "Check for updates"}
          </button>
        )}
      </div>

      {phase.k === "downloading" && (
        <div className={styles.updBar}>
          <div className={pct === null ? styles.updBarIndeterminate : undefined}
            style={pct === null ? undefined : { width: `${pct}%` }} />
        </div>
      )}

      {total > 0 && (
        <div className={styles.updNotes}>
          {highlights.map(g => {
            const items = g.items.slice(0, Math.max(0, left));
            left -= items.length;
            return items.length > 0 && (
              <div key={g.heading}>
                {g.heading && <div className={styles.updNotesHeading}>{g.heading}</div>}
                <ul>{items.map(t => <li key={t}>{t}</li>)}</ul>
              </div>
            );
          })}
          <button className={styles.updLink} onClick={() => openExternal(releaseUrl(latest))}>
            {total > MAX_ITEMS ? `All ${total} changes` : "Full release notes"} <ExternalLink size={11} />
          </button>
        </div>
      )}
    </div>
  );
}
