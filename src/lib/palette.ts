/**
 * The semantic palette — the ONE place a colour gets a meaning.
 *
 * Before this file there were four independent severity scales
 * (`NVRPanel.AI_RISK_COLOR`, `EventsPanel.RISK_COLOR`, `AgentPanel.RISK_COLOR`,
 * `eventFormat.riskColor`) plus a fifth for timeline bands, using two rival
 * palettes — Apple's `#FF3B30` family in some files and Tailwind's `#ef4444`
 * family in others — to express the *same* four states. The same event could be
 * three different colours depending on which panel you were looking at.
 *
 * Rules:
 *   1. Everything resolves to a CSS token from index.css. No literals here.
 *      That keeps a single accent switch honest across the whole app.
 *   2. Four states only: ok / warn / alert / idle. If something needs a fifth,
 *      it almost certainly wants an icon, not a colour.
 *   3. Colour is never the sole signal — every consumer pairs these with a
 *      label or icon, so the palette stays usable for colour-blind users and
 *      survives being rendered over arbitrary video.
 */

export type Severity = "ok" | "warn" | "alert" | "idle";

/** Foreground/mark colour for a severity. */
export const SEVERITY_FG: Record<Severity, string> = {
  ok:    "var(--status-ok)",
  warn:  "var(--status-warn)",
  alert: "var(--status-alert)",
  idle:  "var(--status-idle)",
};

/** Tinted background + hairline for chips and cards. */
export const SEVERITY_BG: Record<Severity, string> = {
  ok:    "color-mix(in srgb, var(--status-ok) 12%, transparent)",
  warn:  "color-mix(in srgb, var(--status-warn) 12%, transparent)",
  alert: "color-mix(in srgb, var(--status-alert) 14%, transparent)",
  idle:  "color-mix(in srgb, var(--status-idle) 10%, transparent)",
};

export const SEVERITY_BORDER: Record<Severity, string> = {
  ok:    "color-mix(in srgb, var(--status-ok) 30%, transparent)",
  warn:  "color-mix(in srgb, var(--status-warn) 30%, transparent)",
  alert: "color-mix(in srgb, var(--status-alert) 36%, transparent)",
  idle:  "color-mix(in srgb, var(--status-idle) 26%, transparent)",
};

/**
 * A severity colour at partial alpha.
 *
 * Every colour in this file resolves to a CSS token — a `var(...)` string. The
 * old way to fade one was to append a hex-alpha suffix, `${color}BB`, which was
 * correct back when these were literals and became `var(--status-alert)BB` after
 * the token migration: invalid CSS, so the browser silently dropped the whole
 * declaration. That is how every risk pill in the app, and every event marker on
 * the NVR timeline, came to render with no fill at all.
 *
 * `color-mix` composes with `var()`, so this works for tokens AND for the real
 * hex literals the depictive palettes below still use.
 */
export const tint = (color: string, pct: number) =>
  `color-mix(in srgb, ${color} ${pct}%, transparent)`;

/**
 * Map every risk word the DB has ever stored onto the four states.
 *
 * The vocabularies were never unified across panels — Agies wrote
 * normal/monitor/suspicious/critical, the agent wrote low/medium/high/critical,
 * and older rows used low/medium/high. All of them land here.
 */
const RISK_TO_SEVERITY: Record<string, Severity> = {
  normal: "ok",      low: "ok",
  monitor: "warn",   medium: "warn",
  suspicious: "alert", high: "alert", critical: "alert",
};

export function severityOfRisk(risk: string | null | undefined): Severity {
  if (!risk) return "idle";
  return RISK_TO_SEVERITY[risk.toLowerCase()] ?? "idle";
}

/** Numeric risk score (0..1) → severity. Thresholds unchanged from `riskColor`. */
export function severityOfScore(score: number): Severity {
  if (score > 0.5) return "alert";
  if (score > 0.3) return "warn";
  return "ok";
}

/**
 * Detection-box colours for the live overlay.
 *
 * Replaces `hsl(stringToHue(label), 90%, 60%)`, which assigned a hue by hashing
 * the YOLO class name: any label could land on any hue, including ones that
 * collided with the semantic colours above, and at 90% saturation everything
 * vibrated over real footage. These four are fixed, muted, and chosen to stay
 * legible on top of arbitrary video.
 */
/*
 * DELIBERATELY LITERAL, like the depictive maps below — these are the one part of
 * the palette that must NOT follow the theme.
 *
 * They are stroked onto a canvas over live video. Routed through `--text-primary`
 * / `--status-*` they would follow whatever the chrome is doing; these must
 * follow the FOOTAGE, which is dark. Keeping them literal is what stops a future
 * token change from drawing near-black boxes on night video.
 */
export const BOX_COLOR = {
  recognised: "#6FA97C",   // a known, named person   (= dark --status-ok)
  person:     "#F5F2EF",   // an unidentified person — highest legibility
  object:     "#8C99A8",   // everything else YOLO reports (= dark --status-idle)
  alert:      "#C9605C",   // the peak-scored subject of an active event
} as const;

/**
 * Event categories are distinguished by their ICON, not by colour.
 *
 * There were three parallel definitions of these four categories
 * (`NVRPanel.CATEGORY_META`, `ReviewPanel.CategoryBadge`, `HistoryDrawer`), each
 * with different hues for the same thing — person was `#3DA5FF` in one place and
 * `#5E9FFF` in another, and animal carried an `#A855F7`/`#A755F7` typo. Giving
 * every category its own colour also spent the palette on information the emoji
 * already carries, and collided with the severity colours that share the view.
 *
 * So: one neutral chip treatment, the icon does the identifying work.
 */
export const CATEGORY_ICON: Record<string, string> = {
  person: "👤", vehicle: "🚗", animal: "🐾", package: "📦",
};

export const CATEGORY_CHIP = {
  bg:     "color-mix(in srgb, var(--text-primary) 8%, transparent)",
  fg:     "var(--text-secondary)",
  border: "var(--border)",
} as const;

/**
 * Swatches for the 11 colour words the HSV classifier votes on (clothing and
 * vehicle bodies).
 *
 * These are DEPICTIVE, not semantic: a red car's dot has to look red. They are
 * deliberately exempt from the token system — routing them through
 * `--status-*` made an orange car and a yellow car render identically, and
 * painted every red car in the UI's alert colour.
 *
 * Muted to match the rest of the palette, but kept mutually distinguishable,
 * which is the entire job of a swatch. Previously duplicated in PersonsPanel
 * (`OUTFIT_DOT`) and VehiclesView (`VEHICLE_COLOR_HEX`) with all 11 values
 * different between the two.
 */
export const OBJECT_COLOR_SWATCH: Record<string, string> = {
  black:  "#1F1F22",
  white:  "#ECEAE7",
  silver: "#C2C6CC",
  gray:   "#83878E",
  red:    "#C0453F",
  orange: "#C97A3C",
  yellow: "#C9B24B",
  green:  "#5E9A6B",
  blue:   "#5B7FA8",
  purple: "#8B6FA8",
  brown:  "#7A5638",
};

/**
 * Zone identity colours, and per-mask-type colours, for the mask editor.
 *
 * Also exempt from the semantic tokens, for the same reason as the swatches
 * above: their whole job is telling zone 1 from zone 2 and a motion mask from a
 * speed line. Routing them through `--status-*` collapsed four of the six zone
 * colours into one and made object and line masks identical.
 *
 * Muted to match the palette, but chosen for mutual separation and to stay
 * visible drawn over arbitrary camera footage.
 */
export const ZONE_PALETTE = [
  "#5B7FA8", "#8B6FA8", "#B0688A", "#4E9A93", "#C97A3C", "#C0453F",
];

export const MASK_TYPE_COLOR: Record<string, string> = {
  motion: "#C0453F",
  object: "#C97A3C",
  speed:  "#5E9A6B",
  line:   "#C9B24B",
};
