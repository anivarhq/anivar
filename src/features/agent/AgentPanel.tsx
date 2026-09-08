/**
 * Guardian — Benji Taylor / Coinbase design language.
 * Layered dark surfaces, glowing accents, premium card hierarchy.
 */
import { useEffect, useState, useRef, useCallback, type ReactNode } from "react";
import { usePanelCache } from "../../lib/panelCache";
import { createPortal } from "react-dom";
import { useStore } from "../../store";
import { useShallow } from "zustand/react/shallow";
import { api, AgentStatus, ChatPart, EventCard } from "../../api";
import { listen } from "@tauri-apps/api/event";
import styles from "./AgentPanel.module.css";
import cardStyles from "../review/ReviewFeed.module.css";
import { MAX_RETRIES, retryDelayMs, bustUrl } from "../live/clipRetry";
import { SEVERITY_FG, severityOfRisk, tint } from "../../lib/palette";
import { Fluffy, type Expression } from "../../components/ui/Fluffy";
import {
  Shield,
  Brain, Send, Loader,
  Sparkles, Search, X, Play, Camera as CameraIcon,
  Check, Copy, RotateCcw, ArrowDown, Trash2, Link as LinkIcon,
} from "lucide-react";

// ── Types ─────────────────────────────────────────────────────────────────────
// NOTE: the chat is a CONVERSATION surface only — alert notifications live in
// the Telegram channel (and Review), never here.

type MsgSender = "guardian" | "user";
// Every message is a chat turn now. "event_list" was a separate message kind
// created by a global Tauri event; evidence rides on the answer it belongs to.
type MsgKind   = "chat";

/** How a message is answered. All three only ever READ — Guardian is
 *  monitor-only, and no mode changes that. */
export type ChatMode = "ask" | "investigate" | "brief";

const MODES: { id: ChatMode; label: string; hint: string; cls: string; dot: string }[] = [
  { id: "ask",         label: "Ask",         hint: "answers in seconds",        cls: "modeAsk",         dot: "var(--text-muted)" },
  { id: "investigate", label: "Investigate", hint: "sweeps the whole archive",  cls: "modeInvestigate", dot: "var(--status-warn)" },
  { id: "brief",       label: "Brief",       hint: "summarises a period",       cls: "modeBrief",       dot: "var(--status-idle)" },
];

/** Context fullness as a 14px ring. Quiet under 70%, amber past it, red at 90%. */
function ContextRing({ used, limit }: { used: number; limit: number }) {
  if (!limit) return null;
  const pct = Math.min(100, Math.round((used / limit) * 100));
  const r = 6, c = 2 * Math.PI * r;
  const colour = pct >= 90 ? "var(--status-alert)" : pct >= 70 ? "var(--status-warn)" : "var(--accent)";
  const k = (n: number) => n >= 1000 ? `${(n / 1000).toFixed(n >= 10000 ? 0 : 1)}k` : `${n}`;
  return (
    <span className={styles.ctxRing}
      title={`${k(used)} / ${k(limit)} tokens (estimated) — older turns drop first`}>
      <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden>
        <circle cx="7" cy="7" r={r} fill="none" stroke="var(--border)" strokeWidth="2.5" />
        <circle cx="7" cy="7" r={r} fill="none" stroke={colour} strokeWidth="2.5"
          strokeDasharray={`${(pct / 100) * c} ${c}`} strokeLinecap="round"
          transform="rotate(-90 7 7)" />
      </svg>
      <span className={styles.ctxPct} style={{ color: colour }}>{pct}%</span>
    </span>
  );
}

// EventCard is THE shared card contract (src/types) — the same shape Telegram
// renders into an album. It used to be redeclared here, which is how the two
// surfaces quietly disagreed about which fields an event has.

interface Msg {
  id:       string;
  sender:   MsgSender;
  kind:     MsgKind;
  ts:       number;
  text?:    string;
  pending?: boolean;
  parts?:   ChatPart[];
  /** The events this answer is about — the same cards Telegram renders. */
  events?:  EventCard[];
  /** Live "what the agent is doing" steps (Searching footage…, Reviewing
   *  sounds…) — streamed while pending, kept as a muted trail on the answer. */
  activity?: string[];
  /** Partial reply arriving token by token while `pending`. Discarded once the
   *  turn completes: the backend may substitute its own findings for a draft
   *  that failed verification, so the final text is authoritative, not this. */
  streamed?: string;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

const uid = () => Math.random().toString(36).slice(2, 10);

function greeting() {
  const h = new Date().getHours();
  if (h < 5)  return "Good evening";
  if (h < 12) return "Good morning";
  if (h < 18) return "Good afternoon";
  return "Good evening";
}

// One-tap questions — shown on the empty hero AND as a persistent chip row
// above the input. Each maps to the agent's grounded tools (events, people,
// vehicles, sounds, summary), so answers come with playable evidence.
// Empty-state suggestions. Four, not five, and chosen to SHOW the range rather
// than list synonyms of "what happened": a plain day summary, a weekday + time
// of day, a description-driven search, and a named person. Each exercises a
// different part of the resolver, so the first thing a new user tries is also
// the thing that demonstrates what this can do.
const QUICK_ASKS = [
  "What happened today?",
  "Anything last night?",
  "Who came on Tuesday evening?",
  "Find the person in the red jacket",
];

function fmtTime(ts: number) {
  return new Date(ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}
function fmtRelative(iso: string) {
  try {
    const diff = Date.now() - new Date(iso).getTime();
    if (diff < 60_000)       return "just now";
    if (diff < 3_600_000)    return `${Math.floor(diff / 60_000)}m ago`;
    if (diff < 86_400_000)   return `${Math.floor(diff / 3_600_000)}h ago`;
    return new Date(iso).toLocaleDateString([], { month: "short", day: "numeric" });
  } catch { return iso; }
}

const RISK_COLOR: Record<string, string> = new Proxy({}, {
  get: (_t, k: string) => SEVERITY_FG[severityOfRisk(k)],
}) as Record<string, string>;
const RISK_LABEL: Record<string, string> = {
  critical: "Critical", high: "High", medium: "Medium", low: "Low",
};
function riskColor(r: string) { return RISK_COLOR[r] ?? RISK_COLOR.low; }

// ── Lightweight markdown for the agent's chat replies ──────────────────────────
// Dependency-free: handles **bold**, *italic*, `code`, [label](url), bullet and
// numbered lists, and paragraphs — enough to make LLM narrative replies readable
// without pulling in a markdown library.
function renderInline(text: string): ReactNode[] {
  const nodes: ReactNode[] = [];
  const re = /(\*\*([^*]+)\*\*|\*([^*]+)\*|`([^`]+)`|\[([^\]]+)\]\((https?:\/\/[^\s)]+)\))/g;
  let last = 0, k = 0, m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) nodes.push(text.slice(last, m.index));
    if (m[2] !== undefined)      nodes.push(<strong key={k++}>{m[2]}</strong>);
    else if (m[3] !== undefined) nodes.push(<em key={k++}>{m[3]}</em>);
    else if (m[4] !== undefined) nodes.push(<code key={k++} className={styles.mdCode}>{m[4]}</code>);
    else if (m[5] !== undefined) nodes.push(
      <a key={k++} href={m[6]} target="_blank" rel="noreferrer" className={styles.mdLink}>{m[5]}</a>);
    last = m.index + m[0].length;
  }
  if (last < text.length) nodes.push(text.slice(last));
  return nodes;
}

/** Render a ```mermaid fence as a real diagram (grok-build pattern: charts as
 *  first-class agent output). The library loads lazily on first use; a render
 *  failure just hides the block (the deterministic backend charts always
 *  parse, so this is belt-and-suspenders). */
function MermaidBlock({ code }: { code: string }) {
  const [svg, setSvg] = useState<string>("");
  useEffect(() => {
    let alive = true;
    import("mermaid").then(async m => {
      try {
        m.default.initialize({ startOnLoad: false, theme: "dark", securityLevel: "strict" });
        const { svg } = await m.default.render(
          `mmd-${Math.random().toString(36).slice(2, 9)}`, code.trim());
        if (alive) setSvg(svg);
      } catch { if (alive) setSvg(""); }
    }).catch(() => {});
    return () => { alive = false; };
  }, [code]);
  if (!svg) return null;
  return <div style={{ overflowX: "auto", margin: "6px 0", maxWidth: "100%" }}
    dangerouslySetInnerHTML={{ __html: svg }} />;
}

function MarkdownText({ text }: { text: string }) {
  // Mermaid fences render as diagrams; the segments around them keep the
  // normal lightweight markdown treatment.
  const fence = /```mermaid\s*\n([\s\S]*?)```/g;
  if (fence.test(text)) {
    fence.lastIndex = 0;
    const out: ReactNode[] = [];
    let last = 0, i = 0, mm: RegExpExecArray | null;
    while ((mm = fence.exec(text)) !== null) {
      const before = text.slice(last, mm.index).trim();
      if (before) out.push(<MarkdownText key={`t${i}`} text={before} />);
      out.push(<MermaidBlock key={`m${i}`} code={mm[1]} />);
      last = mm.index + mm[0].length; i++;
    }
    const after = text.slice(last).trim();
    if (after) out.push(<MarkdownText key={`t${i}end`} text={after} />);
    return <>{out}</>;
  }
  const lines = text.split("\n");
  const blocks: ReactNode[] = [];
  let items: string[] = [];
  let listKind: "ul" | "ol" | null = null;
  let k = 0;
  const flush = () => {
    if (items.length === 0) return;
    const lis = items.map((li, i) => <li key={i}>{renderInline(li)}</li>);
    blocks.push(listKind === "ol"
      ? <ol key={k++} className={styles.mdList}>{lis}</ol>
      : <ul key={k++} className={styles.mdList}>{lis}</ul>);
    items = []; listKind = null;
  };
  for (const raw of lines) {
    const line = raw.trimEnd();
    const bullet  = line.match(/^\s*[-*•]\s+(.*)/);
    const ordered = line.match(/^\s*\d+[.)]\s+(.*)/);
    if (bullet)       { if (listKind !== "ul") flush(); listKind = "ul"; items.push(bullet[1]); }
    else if (ordered) { if (listKind !== "ol") flush(); listKind = "ol"; items.push(ordered[1]); }
    else { flush(); if (line.trim() !== "") blocks.push(<p key={k++} className={styles.mdP}>{renderInline(line)}</p>); }
  }
  flush();
  return <>{blocks}</>;
}

// ── Evidence parts (playable proof inside guardian replies) ───────────────────

/** One structured evidence part: inline snapshot / person card / share link /
 *  activity chart. Events are NOT parts — they render as the card strip below,
 *  from the same data Telegram turns into an album. */
function EvidencePart({ part }: { part: ChatPart }) {
  const { setTab, setLiveView, setFocusedCam } = useStore(useShallow(s => ({
    setTab: s.setTab, setLiveView: s.setLiveView, setFocusedCam: s.setFocusedCam })));
  const [snap, setSnap] = useState<string | null>(null);
  const [snapErr, setSnapErr] = useState(false);

  useEffect(() => {
    if (part.type !== "snapshot") return;
    let alive = true;
    api.getCameraSnapshot(part.cam ?? 0)
      .then(b64 => { if (alive) { b64 ? setSnap(b64) : setSnapErr(true); } })
      .catch(() => { if (alive) setSnapErr(true); });
    return () => { alive = false; };
  }, [part.type, part.cam]);

  // A real, minted share URL. The app used to render share tags as a plain
  // inline card and mint nothing, while the same tag produced a working
  // Tailscale link on Telegram.
  if (part.type === "link" && part.url) {
    return (
      <a href={part.url} target="_blank" rel="noreferrer"
        style={{ display: "inline-flex", alignItems: "center", gap: 8, padding: "8px 12px",
          borderRadius: 10, border: "1px solid var(--border)", width: "fit-content",
          background: "rgb(var(--ink) / 0.03)", fontSize: 12, fontWeight: 700,
          color: "var(--accent)", textDecoration: "none" }}>
        <LinkIcon size={12} /> {part.label ?? "Open link"}
      </a>
    );
  }
  if (part.type === "chart" && part.body) {
    return <MarkdownText text={part.body} />;
  }
  if (part.type === "snapshot") {
    const cam = part.cam ?? 0;
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {snap ? (
          <img src={`data:image/jpeg;base64,${snap}`} alt={`Camera ${cam + 1} snapshot`}
            style={{ maxWidth: 320, width: "100%", borderRadius: 10, border: "1px solid var(--border)" }} />
        ) : (
          <div style={{ width: 320, maxWidth: "100%", height: 120, borderRadius: 10,
            border: "1px solid var(--border)", display: "flex", alignItems: "center",
            justifyContent: "center", gap: 6, fontSize: 11, color: "var(--text-muted)",
            background: "rgb(var(--ink) / 0.03)" }}>
            {snapErr ? <>Camera {cam + 1} isn't live right now</> :
              <><Loader size={11} style={{ animation: "spin 1s linear infinite" }} /> Loading snapshot…</>}
          </div>
        )}
        <button onClick={() => { setFocusedCam(cam); setLiveView("camera"); setTab("live"); }}
          style={{ display: "flex", alignItems: "center", gap: 6, padding: "5px 10px",
            borderRadius: 8, cursor: "pointer", width: "fit-content", fontSize: 11, fontWeight: 700,
            border: "1px solid var(--border)", background: "rgb(var(--ink) / 0.05)",
            color: "var(--text-secondary)" }}>
          <CameraIcon size={11} /> Open Live · CAM {cam + 1}
        </button>
      </div>
    );
  }
  if (part.type === "person") {
    return (
      <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "8px 12px",
        borderRadius: 10, border: "1px solid var(--border)", width: "fit-content",
        background: "rgb(var(--ink) / 0.03)" }}>
        {part.thumbnail ? (
          <img src={`data:image/jpeg;base64,${part.thumbnail}`} alt={part.name ?? "person"}
            style={{ width: 40, height: 40, borderRadius: 10, objectFit: "cover" }} />
        ) : (
          <span style={{ width: 40, height: 40, borderRadius: 10, display: "flex",
            alignItems: "center", justifyContent: "center", fontSize: 18,
            background: "rgb(var(--ink) / 0.06)" }}>👤</span>
        )}
        <span style={{ fontSize: 12.5, fontWeight: 700 }}>{part.name}</span>
      </div>
    );
  }
  return null;
}

/** Mini evidence player — streams /footage/:id/clip with the proven cache-busted
 *  auto-retry so a clip still encoding server-side heals instead of dead-ending. */
function MiniClipPlayer({ eventId, onClose }: { eventId: string; onClose: () => void }) {
  const streamInfo = useStore(s => s.streamInfo);
  const [attempt, setAttempt] = useState(0);
  const [failed, setFailed] = useState(false);
  const timerRef = useRef<number | null>(null);
  useEffect(() => () => { if (timerRef.current) window.clearTimeout(timerRef.current); }, []);
  useEffect(() => { setAttempt(0); setFailed(false); }, [eventId]);

  if (!streamInfo) return null;
  const base = `http://localhost:${streamInfo.port}/footage/${eventId}/clip?token=${streamInfo.auth_token}`;
  const src = attempt > 0 ? bustUrl(base, attempt) : base;

  const onError = () => {
    setAttempt(a => {
      if (a < MAX_RETRIES) {
        const next = a + 1;
        if (timerRef.current) window.clearTimeout(timerRef.current);
        timerRef.current = window.setTimeout(() => setAttempt(next), retryDelayMs(next));
        return a; // bump happens after the backoff
      }
      setFailed(true);
      return a;
    });
  };

  return createPortal(
    <div onClick={e => { if (e.target === e.currentTarget) onClose(); }}
      style={{ position: "fixed", inset: 0, zIndex: 1300, background: "rgba(5,4,4,0.75)",
        display: "flex", alignItems: "center", justifyContent: "center", padding: 24 }}>
      <div className="glass-strong" style={{ width: "100%", maxWidth: 720, padding: 14,
        display: "flex", flexDirection: "column", gap: 10 }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <span style={{ fontSize: 12.5, fontWeight: 700, display: "inline-flex", alignItems: "center", gap: 6 }}>
            <Play size={12} style={{ color: "var(--accent)" }} /> Event clip
          </span>
          <button onClick={onClose} aria-label="Close"
            style={{ background: "none", border: "none", cursor: "pointer", color: "var(--text-muted)", padding: 4 }}>
            <X size={15} />
          </button>
        </div>
        {failed ? (
          <div style={{ padding: "36px 0", textAlign: "center", fontSize: 12, color: "var(--text-secondary)" }}>
            Couldn't load this clip — it may still be processing.{" "}
            <button onClick={() => { setFailed(false); setAttempt(a => a + 1); }}
              style={{ background: "none", border: "none", color: "var(--accent)",
                cursor: "pointer", fontWeight: 700, fontSize: 12 }}>
              Try again
            </button>
          </div>
        ) : (
          <video key={`${eventId}#${attempt}`} src={src} controls autoPlay playsInline
            onError={onError}
            style={{ width: "100%", borderRadius: 10, background: "#000", maxHeight: "70vh" }} />
        )}
      </div>
    </div>,
    document.body,
  );
}

// ── Main panel ────────────────────────────────────────────────────────────────

export function AgentPanel() {
  const { settings, setSettings, showToast } = useStore(useShallow(s => ({ settings: s.settings, setSettings: s.setSettings, showToast: s.showToast })));

  // Cached at module scope so the conversation survives tab switches (the
  // active-only mounting unmounts this panel; losing the chat felt like a bug).
  const [msgs,        setMsgs]        = usePanelCache<Msg[]>("agent.msgs", []);
  const [input,       setInput]       = useState("");
  const [sending,     setSending]     = useState(false);
  const [status,      setStatus]      = useState<AgentStatus | null>(null);
  // Event id currently playing in the mini evidence player (null = closed).
  const [playingEvent, setPlayingEvent] = useState<string | null>(null);
  // Conversation history for context (last 8 exchanges)
  const historyRef = useRef<{ role: string; content: string }[]>([]);

  const feedRef  = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const [focused, setFocused] = useState(false);

  // Fluffy's face is driven by what Guardian is ACTUALLY doing — that is the whole
  // point of it (motion that carries information, not decoration). `delighted` is a
  // brief flash when an answer lands with evidence attached; everything else is a
  // steady state derived below.
  const [delighted, setDelighted] = useState(false);
  const lastAnsweredRef = useRef<string | null>(null);
  useEffect(() => {
    const last = msgs[msgs.length - 1];
    if (!last || last.sender !== "guardian" || last.pending) return;
    if (lastAnsweredRef.current === last.id) return;
    lastAnsweredRef.current = last.id;
    if (!last.events?.length) return;          // only when it found something
    setDelighted(true);
    const t = setTimeout(() => setDelighted(false), 2600);
    return () => clearTimeout(t);
  }, [msgs]);
  const [mode, setMode] = useState<ChatMode>("ask");
  // Reported by the backend with each reply; 0/0 until the first one lands.
  const [ctx, setCtx] = useState<{ used: number; limit: number }>({ used: 0, limit: 0 });
  const cycleMode = useCallback(() =>
    setMode(m => MODES[(MODES.findIndex(x => x.id === m) + 1) % MODES.length].id), []);

  // Restore the DURABLE conversation on first mount (backend-owned chat_log —
  // survives app restarts and is shared with Telegram, so the discussion
  // continues instead of starting from a blank slate every session).
  useEffect(() => {
    api.getChatLog(30).then(rows => {
      if (rows.length === 0) return;
      historyRef.current = rows.slice(-8).map(r => ({ role: r.role, content: r.content }));
      setMsgs(prev => prev.length > 0 ? prev : rows.map((r, i) => ({
        id: uid(),
        sender: (r.role === "user" ? "user" : "guardian") as MsgSender,
        kind: "chat" as MsgKind,
        ts: new Date(r.created_at).getTime() || Date.now() + i,
        text: r.content,
        // Restored turns carry their cards, same as live ones. `content` used to
        // arrive with its brackets still in it and printed `[SHOW_EVENTS:ids=…]`
        // as prose; the backend resolves both halves now.
        events: r.events?.length ? r.events : undefined,
      })));
    }).catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Bootstrap ───────────────────────────────────────────────────────────────
  // Status pill only. Deliberately NO alert loading: notifications are the
  // channel's job (Telegram) — this surface is the investigator, not a feed.
  useEffect(() => {
    api.getAgentStatus().then(setStatus).catch(() => {});
  }, []);


  // ── Live events ─────────────────────────────────────────────────────────────
  // ONLY conversation-driven listeners. The old `agent:analyzed` (alert cards)
  // and `intelligence:alert` (system lines) feeds are gone — notifications are
  // handled by the Telegram channel, not this surface.
  useEffect(() => {
    const push = (m: Msg) => setMsgs(p => [...p, m].slice(-200));

    const subs = [
      listen<any>("guardian:heartbeat", () => {
        // Background memory work continues; we just refresh the status pill.
        // Skip while hidden — no one is looking at the pill.
        if (document.hidden) return;
        api.getAgentStatus().then(setStatus).catch(() => {});
      }),
      // Live agent-work feed: each tool the agent runs streams one line
      // ("Searching footage for 'person'…") into the CURRENT pending bubble,
      // so the user watches the work happen before the answer lands.
      listen<any>("guardian:activity", ({ payload }) => {
        const line = payload?.text;
        if (!line) return;
        setMsgs(p => {
          const ridx = [...p].reverse().findIndex(m => m.pending);
          if (ridx === -1) return p;
          const idx = p.length - 1 - ridx;
          return p.map((m, i) => i === idx
            ? { ...m, activity: [...(m.activity ?? []).filter(a => a !== line), line].slice(-8) }
            : m);
        });
      }),
      // Live token stream from the on-device engine's phrasing pass. Appended
      // to the pending bubble as it arrives; `chat_app`'s return value replaces
      // it when the turn completes — which matters, because a draft that fails
      // the grounding check is discarded and the findings ship instead.
      listen<any>("guardian:chat-token", ({ payload }) => {
        const { content, done } = payload ?? {};
        if (done || !content) return;
        setMsgs(p => {
          const ridx = [...p].reverse().findIndex(m => m.pending);
          if (ridx === -1) return p;
          const idx = p.length - 1 - ridx;
          return p.map((m, i) => i === idx
            ? { ...m, streamed: (m.streamed ?? "") + content }
            : m);
        });
      }),
      // NOTE: there is no "guardian:show-events" listener any more. Events used
      // to arrive on a GLOBAL Tauri event that triggered a second IPC round trip
      // to fetch them, which meant the cards belonged to the panel rather than
      // to the message that produced them. They now ride on the reply itself.
    ];
    return () => { subs.forEach(s => s.then(f => f())); };
  }, []);

  // ── Scroll ──────────────────────────────────────────────────────────────────
  // Follow the newest content ONLY when the user is already at the bottom.
  // Unconditional auto-scroll yanked them back down mid-sentence every time an
  // activity line arrived, which made reading an earlier answer impossible.
  const [atBottom, setAtBottom] = useState(true);
  const onFeedScroll = useCallback(() => {
    const el = feedRef.current;
    if (!el) return;
    setAtBottom(el.scrollHeight - el.scrollTop - el.clientHeight < 80);
  }, []);
  const scrollToLatest = useCallback(() => {
    const el = feedRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, []);
  useEffect(() => {
    if (atBottom) scrollToLatest();
  }, [msgs, atBottom, scrollToLatest]);

  // ── Send ─────────────────────────────────────────────────────────────────────
  const sendText = useCallback(async (text: string) => {
    const q = text.trim();
    if (!q || sending) return;
    setInput("");
    setSending(true);
    const userMsg: Msg    = { id: uid(), sender: "user",     kind: "chat", ts: Date.now(),     text: q };
    const pending: Msg    = { id: uid(), sender: "guardian", kind: "chat", ts: Date.now() + 1, pending: true };
    setMsgs(p => [...p, userMsg, pending]);
    try {
      // Full tool-calling brain (same as Telegram) + structured evidence parts
      // (playable clips, inline snapshots, person cards). Bounded server-side.
      const reply = await api.chatApp(historyRef.current, q, mode);
      setCtx({ used: reply.context_used ?? 0, limit: reply.context_limit ?? 0 });
      setMsgs(p => p.map(m => m.id === pending.id
        ? { ...m, text: reply.text, parts: reply.parts,
            events: reply.events?.length ? reply.events : undefined, pending: false } : m));
      historyRef.current = [
        ...historyRef.current,
        { role: "user",      content: q          },
        { role: "assistant", content: reply.text },
      ].slice(-8);
    } catch (e: any) {
      setMsgs(p => p.map(m => m.id === pending.id
        ? { ...m, text: `I couldn't process that. ${e.message ?? e ?? ""}`.trim(), pending: false } : m));
    } finally { setSending(false); }
    // `mode` belongs in the deps: without it the closure kept the mode selected
    // when the callback was last built, so switching to Investigate and sending
    // straight away silently ran the PREVIOUS mode.
  }, [sending, mode]);

  const send = useCallback(() => sendText(input), [input, sendText]);

  // ── Message actions ─────────────────────────────────────────────────────────
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const copyMsg = useCallback((id: string, text: string) => {
    navigator.clipboard.writeText(text).then(() => {
      setCopiedId(id);
      setTimeout(() => setCopiedId(c => (c === id ? null : c)), 1600);
    }).catch(() => {});
  }, []);

  /// Wipe the durable conversation. This is also the repair when the agent gets
  /// stuck reissuing a stale answer: the old reply lives in `chat_log` and comes
  /// back as context until the thread is cleared.
  const clearHistory = useCallback(async () => {
    if (!window.confirm(
      "Delete this conversation?\n\nOnly the chat is removed — learned memory, events, footage and enrolled people are untouched."
    )) return;
    try {
      await api.clearChatLog();
      historyRef.current = [];
      setMsgs([]);
    } catch (e: any) {
      // Clearing the view but not the log would put them back on the next open.
      window.alert(`Could not clear the conversation: ${e}`);
    }
  }, []);

  /// Re-ask the question that produced this answer — the nearest preceding user
  /// turn. Cheaper than remembering a prompt for every reply, and correct even
  /// after the feed has been trimmed.
  const retry = useCallback((answerId: string) => {
    setMsgs(prev => {
      const i = prev.findIndex(m => m.id === answerId);
      const asked = i > 0
        ? [...prev.slice(0, i)].reverse().find(m => m.sender === "user")?.text
        : undefined;
      if (asked) setTimeout(() => sendText(asked), 0);
      return prev;
    });
  }, [sendText]);

  // ── Status ───────────────────────────────────────────────────────────────────
  const isActive  = !!(status?.enabled && (status?.provider_ready ?? true));

  const isEnabled = !!status?.enabled;

  // Order matters: the most specific state wins. Every branch is something the
  // app actually knows, so the face never claims a mood it cannot justify.
  const fluffyMood: Expression =
      !isEnabled  ? "sleeping"    // no model chosen — Guardian is resting
    : sending     ? "thinking"    // a request is in flight
    : !isActive   ? "concerned"   // enabled, but the engine isn't answering
    : delighted   ? "happy"       // just returned an answer WITH evidence
    : focused     ? "listening"   // the input has focus
    : "idle";

  // ── Evidence card strip ──────────────────────────────────────────────────────
  // The events an answer is ABOUT, from the same cards Telegram turns into a photo
  // album. This used to be a message of its own (`kind: "event_list"`) pushed by a
  // global Tauri event, which meant the cards belonged to the panel rather than to
  // the answer — scroll up and they were somewhere else entirely.
  const EventStrip = ({ events }: { events: EventCard[] }) => (
    <div className={styles.eventListCard}>
      <div className={styles.eventListHead}>
        <Search size={11} style={{ color: "var(--accent)" }} />
        <span>{events.length} event{events.length !== 1 ? "s" : ""}</span>
        <span className={styles.eventListFilter}>from this answer</span>
      </div>
      {/* Events-feed-size tiles (150px, 16:9) in a HORIZONTAL scroll strip —
          the same visual language as the Review section's cards. */}
      <div style={{ display: "flex", gap: 10, overflowX: "auto", paddingBottom: 6,
        scrollbarWidth: "thin" }}>
        {events.map(ev => {
          const color = riskColor(ev.risk_level);
          return (
            <button key={ev.id} className={`${cardStyles.card} ${styles.evCard}`}
              onClick={() => setPlayingEvent(ev.id)}
              // `has_clip` only means a clip FILE was exported; playback slices
              // the continuous recording on demand, so every card is playable.
              title={ev.summary ? ev.summary
                : ev.has_clip ? "Play this event's clip"
                : "Play — sliced from the continuous recording"}
              style={{ width: 150, flexShrink: 0 }}>
              <div className={cardStyles.cardImg}>
                {ev.thumbnail
                  ? <img src={`data:image/jpeg;base64,${ev.thumbnail}`} alt="" loading="lazy"
                      onError={e => { e.currentTarget.style.display = "none"; }} />
                  : <div className={cardStyles.cardImgBlank}><Play size={14} /></div>}
                <div className={cardStyles.cardRisk} style={{ background: tint(color, 87) }}>
                  {ev.risk_level}
                </div>
                <span className={cardStyles.cardPlay}><Play size={12} fill="#fff" /></span>
              </div>
              <div className={cardStyles.cardTime} style={{ padding: "2px 4px 4px" }}>
                {ev.ts}{ev.duration ? ` · ${ev.duration}` : ""}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );

  // ── Message renderer ─────────────────────────────────────────────────────────
  const renderMsg = (msg: Msg) => {
    // ── Turns ─────────────────────────────────────────────────────────────────
    // The user's question is contained; the answer is page content. See the
    // `.turn` rules in the stylesheet for why.
    const isUser = msg.sender === "user";
    if (isUser) {
      return (
        <div key={msg.id} className={`${styles.turn} ${styles.turnUser}`}>
          <div className={styles.userText}>{msg.text}</div>
          <span className={styles.bubbleTime}>{fmtTime(msg.ts)}</span>
        </div>
      );
    }

    return (
      <div key={msg.id} className={styles.turn}>
        {msg.pending ? (
          <>
            {/* Steps already done stay visible above the live one, so the user
                can see what was actually consulted rather than a blank spinner. */}
            {(msg.activity ?? []).slice(0, -1).map((a, i) => (
              <div key={i} className={styles.trailStep}>
                <Check size={11} style={{ opacity: 0.6 }} /> {a}
              </div>
            ))}
            {msg.streamed ? (
              // Once words start arriving, show them instead of the step line —
              // the work is visibly done and the answer is being written.
              <div className={styles.bubbleText}>
                {msg.streamed}<span className={styles.caret} />
              </div>
            ) : (
              <span className={styles.thinking}>
                <Loader size={13} style={{ animation: "spin 1s linear infinite" }} />
                {(msg.activity ?? []).length > 0
                  ? msg.activity![msg.activity!.length - 1] + "…"
                  : "Thinking…"}
              </span>
            )}
          </>
        ) : (
          <>
            {/* Work trail as a real disclosure — collapsed by default, because
                how the answer was reached is secondary to the answer. */}
            {(msg.activity ?? []).length > 0 && (
              <details className={styles.trail}>
                <summary className={styles.trailSummary}>
                  <Search size={11} />
                  Worked on this · {msg.activity!.length} step
                  {msg.activity!.length !== 1 ? "s" : ""}
                </summary>
                <div className={styles.trailSteps}>
                  {msg.activity!.map((a, i) => (
                    <div key={i} className={styles.trailStep}>
                      <Check size={11} style={{ opacity: 0.6 }} /> {a}
                    </div>
                  ))}
                </div>
              </details>
            )}
            <div className={styles.bubbleText}>
              <MarkdownText text={msg.text ?? ""} />
            </div>
            {msg.events && msg.events.length > 0 && (
              <EventStrip events={msg.events} />
            )}
            {msg.parts && msg.parts.length > 0 && (
              <div className={styles.evidence}>
                {msg.parts.map((p, i) => (
                  <EvidencePart key={i} part={p} />
                ))}
              </div>
            )}
            <div className={styles.turnActions}>
              <button className={styles.turnAction}
                onClick={() => copyMsg(msg.id, msg.text ?? "")}
                title="Copy this answer">
                {copiedId === msg.id
                  ? <><Check size={12} /> Copied</>
                  : <><Copy size={12} /> Copy</>}
              </button>
              <button className={styles.turnAction}
                onClick={() => retry(msg.id)} disabled={sending}
                title="Ask again">
                <RotateCcw size={12} /> Retry
              </button>
              <span className={styles.bubbleTime} style={{ alignSelf: "center", marginTop: 0 }}>
                {fmtTime(msg.ts)}
              </span>
            </div>
          </>
        )}
      </div>
    );
  };

  // ── Render ────────────────────────────────────────────────────────────────────
  return (
    <div className={styles.root}>

      {/* ── Chat feed ───────────────────────────────────────────────── */}
      {(
        <>
          {/* Thread controls sit at the TOP, out of the way of the answer.
              Only shown once there is a conversation to act on. */}
          {msgs.length > 0 && (
            <div className={styles.threadBar}>
              <button className={styles.turnAction} onClick={clearHistory}
                title="Delete this conversation">
                <Trash2 size={12} /> Clear conversation
              </button>
            </div>
          )}

          {/* ── Feed ──────────────────────────────────────────────────── */}
          <div className={styles.feed} ref={feedRef} onScroll={onFeedScroll}>
            {msgs.length === 0 ? (
              // Welcoming hero — calm, glass, no header clutter.
              <div className={styles.welcome}>
                <div className={styles.welcomeOrb}>
                  <Fluffy size={72} expression={fluffyMood} />
                </div>
                <h2 className={styles.welcomeTitle}>{greeting()}</h2>
                <p className={styles.welcomeSub}>
                  {isActive
                    ? "Every answer comes with the clips to prove it."
                    : isEnabled
                      ? "Reconnecting to the AI engine…"
                      : "I'm resting. Pick a model in Arsenal → Model (\"Use\") and I'll start watching."}
                </p>
                {isActive && (
                  <div className={styles.welcomeChips}>
                    {QUICK_ASKS.map(s => (
                      <button key={s} className={styles.welcomeChip}
                        onClick={() => sendText(s)} disabled={sending}>
                        {s}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            ) : null}
            {msgs.length > 0 && <div className={styles.thread}>{msgs.map(renderMsg)}</div>}
          </div>

          {/* Only offered when the user has scrolled away — otherwise it is a
              button that does nothing, permanently in the way. */}
          {!atBottom && msgs.length > 0 && (
            <button className={styles.jumpLatest} onClick={scrollToLatest}>
              <ArrowDown size={12} /> Latest
            </button>
          )}

          {/* ── Composer ──────────────────────────────────────────────── */}
          <div className={styles.inputBar}>
            <div className={styles.composerWrap}>
              <div className={`${styles.composer} ${focused ? styles.composerFocus : ""}`}>
                {/* Fluffy sits WITH the composer once the conversation starts —
                    the welcome hero scrolls away, and the character carrying
                    Guardian's state has to stay where you're looking. Hidden
                    while the hero is up so there is never two of him. */}
                {msgs.length > 0 && (
                  <span className={styles.composerFluffy} aria-hidden>
                    <Fluffy size={30} expression={fluffyMood} />
                  </span>
                )}
                <textarea
                  ref={inputRef}
                  className={styles.input}
                  placeholder="Ask about anything your cameras saw or heard…"
                  value={input}
                  rows={1}
                  onFocus={() => setFocused(true)}
                  onBlur={() => setFocused(false)}
                  onChange={e => {
                    setInput(e.target.value);
                    // Grow to fit, up to the CSS max-height, then scroll.
                    const el = e.target;
                    el.style.height = "auto";
                    el.style.height = `${Math.min(el.scrollHeight, 180)}px`;
                  }}
                  onKeyDown={e => {
                    // Shift+Tab cycles the mode — the same gesture Claude Code
                    // uses, and it keeps hands on the keyboard mid-thought.
                    if (e.key === "Tab" && e.shiftKey) { e.preventDefault(); cycleMode(); return; }
                    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
                  }}
                  disabled={sending}
                />
                <button className={`${styles.sendBtn} ${input.trim() ? styles.sendBtnActive : ""}`}
                  onClick={send} disabled={sending || !input.trim()}
                  title="Send (Enter)">
                  {sending
                    ? <Loader size={14} style={{ animation: "spin 1s linear infinite" }} />
                    : <Send size={14} />}
                </button>
              </div>
              {/* Mode left, context right — everything that changes what a
                  message does, next to where the message is written. */}
              <div className={styles.composerFoot}>
                <ContextRing used={ctx.used} limit={ctx.limit} />
                <span className={styles.modeSpacer} />
                {(() => {
                  const m = MODES.find(x => x.id === mode)!;
                  return (
                    <button type="button" onClick={cycleMode}
                      className={`${styles.modeBtn} ${styles[m.cls]}`}
                      title="Shift+Tab to switch mode">
                      <span className={styles.modeDot} style={{ background: m.dot }} />
                      {m.label}
                      <span className={styles.modeHint}>· {m.hint}</span>
                    </button>
                  );
                })()}
              </div>
              {input.includes("\n") && (
                <div className={styles.inputHint}>Enter to send · Shift+Enter for a new line</div>
              )}
            </div>
          </div>
        </>
      )}

      {/* ── Mini evidence player ─────────────────────────────────────── */}
      {playingEvent && (
        <MiniClipPlayer eventId={playingEvent} onClose={() => setPlayingEvent(null)} />
      )}

    </div>
  );
}


// (ConfigDrawer + SkillsTab + Section/FieldRow/Toggle helpers removed —
//  their job moved to ./Arsenal.tsx, the Notifications section in Settings,
//  and the shared ./skillDownload.ts registry.)
