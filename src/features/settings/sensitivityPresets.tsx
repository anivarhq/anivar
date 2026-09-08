// Single source of truth for the Detection-sensitivity presets, shared by the
// main Settings panel and the per-camera Camera-settings modal so the two can
// never drift. One preset writes the six verified-live "open an event" knobs so
// the user never has to reason about raw motion physics.
//
// The "Balanced" row mirrors the Rust backend Default (src-tauri/src/state.rs).
// Low = fewer alerts (require a confirmed object + higher bars); High = catch
// everything (lower bars, more false alarms). The backend clamps
// motion_open_score_mult.max(1.0) and motion_min_frames.max(1), so every value
// below stays within what the engine will actually honour.

export type SensPreset = "low" | "balanced" | "high";

export interface PresetKnobs {
  sensitivity: number;
  motion_threshold: number;
  motion_min_frames: number;
  motion_open_score_mult: number;
  yolo_confidence_threshold: number;
  require_object_to_open_event: boolean;
}

export const SENSITIVITY_PRESETS: Record<SensPreset, PresetKnobs> = {
  low:      { sensitivity: 0.06, motion_threshold: 12, motion_min_frames: 5, motion_open_score_mult: 1.5, yolo_confidence_threshold: 0.55, require_object_to_open_event: true  },
  balanced: { sensitivity: 0.04, motion_threshold: 8,  motion_min_frames: 3, motion_open_score_mult: 1.0, yolo_confidence_threshold: 0.40, require_object_to_open_event: false },
  high:     { sensitivity: 0.02, motion_threshold: 4,  motion_min_frames: 1, motion_open_score_mult: 1.0, yolo_confidence_threshold: 0.25, require_object_to_open_event: false },
};

export const PRESET_META: { id: SensPreset; label: string; desc: string }[] = [
  { id: "low",      label: "Low",      desc: "Fewer alerts · only confirmed objects open events" },
  { id: "balanced", label: "Balanced", desc: "Recommended for most cameras" },
  { id: "high",     label: "High",     desc: "Catch everything · expect more false alarms" },
];

/** Which preset (if any) a settings object currently matches. "custom" once any
 *  of the six knobs has been hand-edited away from a preset row. */
export function matchPreset(f: Partial<PresetKnobs>): SensPreset | "custom" {
  const near = (a: number | undefined, b: number) => Math.abs((a ?? b) - b) < 1e-6;
  for (const k of ["low", "balanced", "high"] as SensPreset[]) {
    const p = SENSITIVITY_PRESETS[k];
    if (near(f.sensitivity, p.sensitivity) &&
        (f.motion_threshold ?? p.motion_threshold) === p.motion_threshold &&
        (f.motion_min_frames ?? p.motion_min_frames) === p.motion_min_frames &&
        near(f.motion_open_score_mult, p.motion_open_score_mult) &&
        near(f.yolo_confidence_threshold, p.yolo_confidence_threshold) &&
        !!f.require_object_to_open_event === p.require_object_to_open_event) {
      return k;
    }
  }
  return "custom";
}

/** Segmented Low · Balanced · High card picker. Caller owns the surrounding
 *  chrome (label / CUSTOM badge / help text). */
export function SensitivityPresetPicker({ active, onPick }: {
  active: SensPreset | "custom";
  onPick: (p: SensPreset) => void;
}) {
  return (
    <div style={{ display: "flex", gap: 8 }}>
      {PRESET_META.map(p => {
        const on = active === p.id;
        return (
          <button key={p.id} type="button" onClick={() => onPick(p.id)}
            style={{
              flex: 1, display: "flex", flexDirection: "column", gap: 4, alignItems: "flex-start",
              padding: "10px 12px", borderRadius: 12, cursor: "pointer", textAlign: "left",
              border: `1.5px solid ${on ? "var(--accent)" : "var(--border)"}`,
              background: on ? "var(--hl)" : "transparent",
              transition: "border-color 0.15s, background 0.15s",
            }}>
            <span style={{ fontSize: 13, fontWeight: 700, color: on ? "var(--accent)" : "var(--text-primary)" }}>{p.label}</span>
            <span style={{ fontSize: 10, color: "var(--text-muted)", lineHeight: 1.35 }}>{p.desc}</span>
          </button>
        );
      })}
    </div>
  );
}
