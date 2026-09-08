/**
 * Fluffy — the Anivar mascot. Entirely code, no assets.
 *
 * Built on the technique behind the Grok Bot icon (Benji Taylor): a **fixed
 * primitive vocabulary** where every expression is the same shapes with
 * different numbers, so any state morphs into any other. Nothing is ever added,
 * removed or crossfaded — the numbers are interpolated and the geometry is
 * regenerated from them each frame.
 *
 * Two design decisions worth keeping:
 *
 * 1. **Fluff is irregularity.** A scalloped circle reads as a flower or a cog —
 *    tested at 12/14/16/18/20 lobes and it never read as fur. Overlapping circles
 *    of deliberately uneven size DO read as a pom-pom, because they union into an
 *    organic silhouette with no repeating rhythm. `TUFTS` is a fixed table on
 *    purpose: a random silhouette would shimmer between renders and break morphing.
 *
 * 2. **"Cute" is Kindchenschema, not vibes.** Eyes large, wide apart, and set
 *    BELOW the body's midline so a big domed forehead of fluff sits above them;
 *    small mouth close underneath; no sharp vertices; and a specular glint offset
 *    up-left in each eye — the cheapest, strongest cuteness lever there is,
 *    because it makes an eye read as wet and alive.
 */
import { useEffect, useRef, useState } from "react";

export type Expression =
  | "idle" | "listening" | "thinking" | "happy" | "concerned" | "sleeping";

/** The morphable parameter set. Every expression is one of these. */
interface Params {
  rx: number;      // eye half-width
  apex: number;    // eye upper peak (height above centre)
  base: number;    // eye lower peak — negative bends UP, making a crescent
  gx: number; gy: number;   // pupil offset (gaze)
  brow: number;    // brow opacity
  lift: number;    // brow inner-end lift. POSITIVE = inner raised = worried.
                   // Inverting this is the difference between worried and angry —
                   // both "thinking" and "concerned" read as cross until it's +ve.
  curve: number;   // mouth curvature, -1 frown … +1 smile
  open: number;    // mouth opening
  blush: number;   // cheek opacity
  tilt: number;    // head tilt, degrees
  mw: number;      // mouth half-width
}

/**
 * Tuned for the size it is actually SEEN at.
 *
 * The first pass used values that looked right on a 200px contact sheet and were
 * invisible in the app: the `thinking` gaze was 1.5 units in a 100-unit viewBox,
 * which at the composer's 30px is 0.45 of a pixel. Nothing moved.
 *
 * So the per-state differences now lean on what survives at 30px — eye SHAPE,
 * head TILT and gross gaze — rather than on fine offsets.
 */
const EXPRESSIONS: Record<Expression, Params> = {
  idle:      { rx: 6,   apex: 6.5, base: 6.5,  gx: 0,  gy: 0,    brow: 0,   lift: 0,   curve:  0.35, open: 0.15, blush: 0.25, tilt:   0, mw: 5   },
  // Eyes markedly wider than idle — at small size that reads instantly.
  listening: { rx: 8,   apex: 9.5, base: 9.5,  gx: 0,  gy: 0.5,  brow: 0,   lift: 0,   curve:  0.20, open: 0.22, blush: 0.20, tilt:   7, mw: 4.4 },
  // No brow: raised read as *surprised*, lowered as *angry*. The gaze carries it,
  // so the gaze has to be big enough to see — pupils pushed to the eye's edge.
  thinking:  { rx: 5.5, apex: 5,   base: 5,    gx: -3, gy: -3.4, brow: 0,   lift: 0,   curve:  0.05, open: 0.08, blush: 0.15, tilt: -12, mw: 3.2 },
  happy:     { rx: 6.4, apex: 5,   base: -1.4, gx: 0,  gy: 0,    brow: 0,   lift: 0,   curve:  0.95, open: 0.60, blush: 0.75, tilt:   0, mw: 5.6 },
  concerned: { rx: 6.5, apex: 7,   base: 7,    gx: 0,  gy: 1.6,  brow: 0.95,lift: 3.4, curve: -0.55, open: 0.25, blush: 0.10, tilt:   0, mw: 4   },
  sleeping:  { rx: 6,   apex: 0.8, base: -0.6, gx: 0,  gy: 0,    brow: 0,   lift: 0,   curve:  0.15, open: 0.05, blush: 0.30, tilt:  11, mw: 4   },
};

/** Seeded tufts: [angle°, distance from centre, radius]. Fixed, never random. */
const TUFTS: [number, number, number][] = [
  [0, 22, 12], [32, 24, 9], [70, 21, 13], [105, 23, 10], [140, 22, 12], [168, 24, 8],
  [200, 21, 13], [232, 23, 10], [262, 22, 11], [292, 24, 9], [325, 21, 13],
];

const CX = 50, CY = 50, EYE_DX = 13, EYE_DY = 8, MOUTH_DY = 21;

/** Eye as ONE path from three numbers — this is what lets a closed eye morph
 *  from an open one without swapping primitives. */
const eyePath = (cx: number, cy: number, rx: number, apex: number, base: number) =>
  `M ${cx - rx} ${cy} Q ${cx} ${cy - 2 * apex} ${cx + rx} ${cy} ` +
  `Q ${cx} ${cy + 2 * base} ${cx - rx} ${cy} Z`;

/** Mouth as ONE path. `+2` keeps a minimum thickness so a closed smile still reads. */
const mouthPath = (cx: number, cy: number, w: number, curve: number, open: number) => {
  const top = 2 * curve * 6;
  const bot = top + 2 * (open * 7) + 2;
  return `M ${cx - w} ${cy} Q ${cx} ${cy + top} ${cx + w} ${cy} ` +
         `Q ${cx} ${cy + bot} ${cx - w} ${cy} Z`;
};

const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

/**
 * Tween the parameter set toward a target.
 *
 * The numbers are animated, NOT the paths: setting the SVG `d` *attribute* does
 * not transition in CSS, and interpolating parameters is both simpler and what
 * makes "any state to any state" true by construction.
 */
function useMorph(target: Params, instant: boolean): Params {
  const [, force] = useState(0);
  const cur = useRef<Params>({ ...target });
  const raf = useRef(0);

  useEffect(() => {
    if (instant) { cur.current = { ...target }; force(n => n + 1); return; }
    const keys = Object.keys(target) as (keyof Params)[];
    const step = () => {
      let moving = false;
      for (const k of keys) {
        const next = lerp(cur.current[k], target[k], 0.18);
        if (Math.abs(target[k] - next) > 0.002) moving = true;
        cur.current[k] = moving ? next : target[k];
      }
      force(n => n + 1);
      raf.current = moving ? requestAnimationFrame(step) : 0;   // stop when settled
    };
    cancelAnimationFrame(raf.current);
    raf.current = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf.current);
  }, [target, instant]);

  return cur.current;
}

export function Fluffy({ expression = "idle", size = 96, className }: {
  expression?: Expression;
  size?: number;
  className?: string;
}) {
  const reduced = typeof window !== "undefined"
    && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

  // Blink: a brief eye-close on the states where the eyes are open. It is just a
  // target override, so it morphs like everything else.
  const [blinking, setBlinking] = useState(false);
  const canBlink = expression === "idle" || expression === "listening" || expression === "concerned";
  useEffect(() => {
    if (!canBlink || reduced) return;
    let t: ReturnType<typeof setTimeout>;
    const loop = () => {
      t = setTimeout(() => {
        setBlinking(true);
        setTimeout(() => { setBlinking(false); loop(); }, 110);
      }, 2600 + Math.random() * 3200);
    };
    loop();
    return () => clearTimeout(t);
  }, [canBlink, reduced]);

  const base = EXPRESSIONS[expression];
  const target: Params = blinking ? { ...base, apex: 0.7, base: -0.4 } : base;
  const p = useMorph(target, !!reduced);

  // A closed eye has no pupil to show; fade rather than unmount so it morphs.
  const openness = Math.max(0, Math.min(1, (p.apex - 1.2) / 3));

  return (
    <svg width={size} height={size} viewBox="0 0 100 100" className={className}
      aria-hidden="true" focusable="false">
      <style>{"@keyframes fluffyBob{0%,100%{translate:0 0}50%{translate:0 -3px}}"}</style>
      <g style={{
        transformOrigin: "50px 50px",
        transform: `rotate(${p.tilt}deg)`,
        // A slow bob while working. Motion is the only cue that survives at any
        // size, and "is it doing anything?" is the question being answered.
        animation: expression === "thinking" && !reduced
          ? "fluffyBob 1.5s ease-in-out infinite" : undefined,
      }}>
        {/* Body: overlapping circles union into a pom-pom. */}
        <g fill="var(--fluffy-body, #E8DCCF)">
          <circle cx={CX} cy={CY + 2} r={24} />
          {TUFTS.map(([a, d, r], i) => (
            <circle key={i} r={r}
              cx={CX + d * Math.cos((a * Math.PI) / 180)}
              cy={CY + 2 + d * Math.sin((a * Math.PI) / 180)} />
          ))}
        </g>

        {[-1, 1].map(sx => (
          <ellipse key={`b${sx}`} cx={CX + sx * 20} cy={CY + 16} rx={6} ry={3.8}
            fill="var(--accent, #C9605C)" opacity={p.blush} />
        ))}

        {[-1, 1].map(sx => {
          const ex = CX + sx * EYE_DX, ey = CY + EYE_DY;
          const by = ey - p.rx - 4.5;
          return (
            <g key={`e${sx}`}>
              <path d={eyePath(ex, ey, p.rx, p.apex, p.base)} fill="var(--fluffy-ink, #241F1C)" />
              <circle cx={ex + p.gx} cy={ey + p.gy} r={2.6}
                fill="var(--fluffy-ink, #241F1C)" opacity={openness} />
              <circle cx={ex + p.gx - 1.9} cy={ey + p.gy - 2.1} r={1.7}
                fill="#fff" opacity={openness} />
              <line x1={ex - sx * 5} y1={by - p.lift} x2={ex + sx * 5} y2={by + p.lift * 0.5}
                stroke="var(--fluffy-ink, #241F1C)" strokeWidth={2.2} strokeLinecap="round"
                opacity={p.brow} />
            </g>
          );
        })}

        <path d={mouthPath(CX, CY + MOUTH_DY, p.mw, p.curve, p.open)}
          fill="var(--fluffy-ink, #241F1C)" />
      </g>
    </svg>
  );
}
