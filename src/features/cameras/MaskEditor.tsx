/**
 * MaskEditor — NVR-parity polygon editor for motion masks, object masks,
 * and zones.
 *
 * mature NVRs model (mirrored here):
 *  • Motion mask  → excludes pixels from MOTION detection (timestamps, trees,
 *    flags). Used sparingly — over-masking degrades object tracking.
 *  • Object mask  → discards a detection when the BOTTOM-CENTER of its bbox
 *    falls inside the polygon (kills stubborn fixed false positives).
 *  • Zone         → named region; an object is "in" the zone when its bbox
 *    bottom-center is inside. Carries object filters + inertia + loitering.
 *
 * Coordinates are normalized 0–1 (resolution-independent, exactly like mature NVRs).
 *
 * The drawing stage is ASPECT-LOCKED to the video's native aspect ratio, so the
 * full frame is always shown with NO letterbox and clicks map 1:1 to normalized
 * coords — which is what makes zone drawing accurate and convenient.
 */
import { useEffect, useRef, useState, useCallback } from "react";
import { useStore } from "../../store";
import { useShallow } from "zustand/react/shallow";
import { api, CameraMask, CameraMaskMap, MaskType, parseMasks, serializeMasks } from "../../api";
import { X, Plus, Trash2, Save, Pencil } from "lucide-react";

import styles from "./MaskEditor.module.css";
import { ZONE_PALETTE, MASK_TYPE_COLOR } from "../../lib/palette";

// ── Helpers ───────────────────────────────────────────────────────────────────

function uuid(): string { return Math.random().toString(36).slice(2, 10); }

type Pt = { x: number; y: number };

function parsePoints(s: string): Pt[] {
  const nums = s.split(",").map(Number).filter(n => !isNaN(n));
  const pts: Pt[] = [];
  for (let i = 0; i + 1 < nums.length; i += 2) pts.push({ x: nums[i], y: nums[i + 1] });
  return pts;
}
function stringifyPoints(pts: Pt[]): string {
  return pts.map(p => `${p.x.toFixed(4)},${p.y.toFixed(4)}`).join(",");
}
function centroid(pts: Pt[]): Pt {
  if (pts.length === 0) return { x: 0.5, y: 0.5 };
  return { x: pts.reduce((a, p) => a + p.x, 0) / pts.length, y: pts.reduce((a, p) => a + p.y, 0) / pts.length };
}
// Closest point on segment ab to p, plus the parametric t — used for inserting
// a vertex when the user double-clicks an edge.
function projectToSegment(p: Pt, a: Pt, b: Pt): { t: number; dist: number } {
  const dx = b.x - a.x, dy = b.y - a.y;
  const len2 = dx * dx + dy * dy || 1e-9;
  let t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / len2;
  t = Math.max(0, Math.min(1, t));
  const cx = a.x + t * dx, cy = a.y + t * dy;
  return { t, dist: Math.hypot(p.x - cx, p.y - cy) };
}

const ZONE_COLORS = ZONE_PALETTE;
const ZONE_OBJECTS = ["person", "car", "animal", "package"];

function maskStroke(m: CameraMask, idx: number): string {
  const byType = MASK_TYPE_COLOR[m.type];
  if (byType) return byType;
  return m.color ?? ZONE_COLORS[idx % ZONE_COLORS.length];
}
function maskFill(m: CameraMask, idx: number): string {
  const c = maskStroke(m, idx);
  // 22 ≈ 13% alpha
  return c + "22";
}

// ── MaskEditor ────────────────────────────────────────────────────────────────

interface MaskEditorProps {
  camId: number;
  onClose: () => void;
}

type EditorMode = "idle" | "drawing";

export function MaskEditor({ camId, onClose }: MaskEditorProps) {
  const { settings, setSettings, showToast, streamInfo } = useStore(useShallow(s => ({ settings: s.settings, setSettings: s.setSettings, showToast: s.showToast, streamInfo: s.streamInfo })));

  const [allMasks, setAllMasks] = useState<CameraMaskMap>(() => parseMasks(settings?.camera_masks));
  const masks: CameraMask[] = allMasks[String(camId)] ?? [];

  const [mode,        setMode]        = useState<EditorMode>("idle");
  const [drawType,    setDrawType]    = useState<MaskType>("motion");
  const [newName,     setNewName]     = useState("");
  const [draftPoints, setDraftPoints] = useState<Pt[]>([]);
  const [hoverPoint,  setHoverPoint]  = useState<Pt | null>(null);
  const [selectedId,  setSelectedId]  = useState<string | null>(null);
  const [saving,      setSaving]      = useState(false);
  // Aspect ratio of the source video — drives the aspect-locked canvas.
  const [aspect,      setAspect]      = useState(16 / 9);

  // drag = moving an existing vertex; polyDrag = moving a whole polygon.
  const [drag,     setDrag]     = useState<{ maskId: string; ptIdx: number } | null>(null);
  const polyDrag = useRef<{ maskId: string; last: Pt } | null>(null);
  // True once a press has actually moved — so the trailing click doesn't also
  // toggle/clear the selection after a drag.
  const didDrag = useRef(false);

  const svgRef = useRef<SVGSVGElement>(null);

  // The live MJPEG `/stream` route was removed; `/snapshot` is the current full
  // frame. We refresh it every few seconds so the background feels live while
  // you draw, without depending on the deleted streaming route.
  const [snapTick, setSnapTick] = useState(() => Date.now());
  useEffect(() => {
    const id = window.setInterval(() => setSnapTick(Date.now()), 4000);
    return () => window.clearInterval(id);
  }, []);
  const feedUrl = streamInfo
    ? `http://localhost:${streamInfo.port}/snapshot?cam=${camId}&token=${streamInfo.auth_token}&t=${snapTick}`
    : null;

  const selected = masks.find(m => m.id === selectedId) ?? null;

  // ── coords: clientX/Y → normalized 0–1 over the canvas (no letterbox math
  //    needed because the canvas IS the video aspect). ───────────────────────
  const coords = useCallback((e: { clientX: number; clientY: number }): Pt => {
    const r = svgRef.current!.getBoundingClientRect();
    return {
      x: Math.max(0, Math.min(1, (e.clientX - r.left) / Math.max(1, r.width))),
      y: Math.max(0, Math.min(1, (e.clientY - r.top) / Math.max(1, r.height))),
    };
  }, []);

  // Mutate the selected camera's mask list immutably.
  const updateMask = useCallback((id: string, fn: (m: CameraMask) => CameraMask) => {
    setAllMasks(prev => {
      const cam = [...(prev[String(camId)] ?? [])];
      const i = cam.findIndex(m => m.id === id);
      if (i < 0) return prev;
      cam[i] = fn(cam[i]);
      return { ...prev, [String(camId)]: cam };
    });
  }, [camId]);

  // ── drawing ───────────────────────────────────────────────────────────────
  const onSvgClick = (e: React.MouseEvent<SVGSVGElement>) => {
    if (didDrag.current) { didDrag.current = false; return; } // swallow drag-end click
    if (mode !== "drawing") { setSelectedId(null); return; }
    e.stopPropagation();
    const pt = coords(e);
    // Close the polygon by clicking near the first vertex (mature NVRs behaviour).
    if (draftPoints.length >= 3) {
      const first = draftPoints[0];
      const r = svgRef.current!.getBoundingClientRect();
      const distPx = Math.hypot((pt.x - first.x) * r.width, (pt.y - first.y) * r.height);
      if (distPx < 14) { closeDraft(); return; }
    }
    setDraftPoints(p => [...p, pt]);
  };

  const onSvgMove = (e: React.MouseEvent<SVGSVGElement>) => {
    if (mode === "drawing") setHoverPoint(coords(e));
    if (drag && e.buttons === 1) {
      didDrag.current = true;
      const pt = coords(e);
      updateMask(drag.maskId, m => {
        const pts = parsePoints(m.points);
        pts[drag.ptIdx] = pt;
        return { ...m, points: stringifyPoints(pts) };
      });
    } else if (polyDrag.current && e.buttons === 1) {
      didDrag.current = true;
      const pt = coords(e);
      const { maskId, last } = polyDrag.current;
      const dx = pt.x - last.x, dy = pt.y - last.y;
      polyDrag.current = { maskId, last: pt };
      updateMask(maskId, m => {
        const pts = parsePoints(m.points).map(p => ({
          x: Math.max(0, Math.min(1, p.x + dx)),
          y: Math.max(0, Math.min(1, p.y + dy)),
        }));
        return { ...m, points: stringifyPoints(pts) };
      });
    }
  };

  const onSvgUp = () => { setDrag(null); polyDrag.current = null; };

  const closeDraft = useCallback(() => {
    if (drawType === "line") {
      if (draftPoints.length !== 2) { showToast("A line needs exactly 2 points", "error"); return; }
    } else if (drawType === "speed") {
      if (draftPoints.length !== 4) { showToast("A speed zone needs exactly 4 corner points (in order)", "error"); return; }
    } else if (draftPoints.length < 3) {
      showToast("Need at least 3 points", "error"); return;
    }
    const zoneCount = masks.filter(m => m.type === "zone").length;
    const name = newName.trim() ||
      (drawType === "motion" ? `Motion mask ${masks.length + 1}`
        : drawType === "object" ? `Object mask ${masks.length + 1}`
          : drawType === "speed" ? `Speed zone ${masks.filter(m => m.type === "speed").length + 1}`
          : drawType === "line" ? `Line ${masks.filter(m => m.type === "line").length + 1}`
          : `Zone ${zoneCount + 1}`);
    const entry: CameraMask = {
      id: uuid(), name, points: stringifyPoints(draftPoints), type: drawType,
      ...(drawType === "zone" ? {
        color: ZONE_COLORS[zoneCount % ZONE_COLORS.length],
        objects: [], inertia: 3, loitering_secs: 0, alert_on_enter: true,
      } : {}),
      ...(drawType === "speed" ? { width_m: 5, height_m: 5 } : {}),
    };
    setAllMasks(prev => ({ ...prev, [String(camId)]: [...(prev[String(camId)] ?? []), entry] }));
    setDraftPoints([]); setNewName(""); setMode("idle"); setSelectedId(entry.id);
  }, [draftPoints, drawType, masks, newName, camId, showToast]);

  // Insert a vertex on the nearest edge (double-click) of the selected polygon.
  const onSvgDoubleClick = (e: React.MouseEvent<SVGSVGElement>) => {
    if (mode === "drawing" || !selected) return;
    const pt = coords(e);
    const pts = parsePoints(selected.points);
    let best = { i: -1, t: 0, dist: Infinity };
    for (let i = 0; i < pts.length; i++) {
      const a = pts[i], b = pts[(i + 1) % pts.length];
      const { t, dist } = projectToSegment(pt, a, b);
      if (dist < best.dist) best = { i, t, dist };
    }
    if (best.i >= 0 && best.dist < 0.04) {
      const a = pts[best.i], b = pts[(best.i + 1) % pts.length];
      const np = { x: a.x + (b.x - a.x) * best.t, y: a.y + (b.y - a.y) * best.t };
      pts.splice(best.i + 1, 0, np);
      updateMask(selected.id, m => ({ ...m, points: stringifyPoints(pts) }));
    }
  };

  const deleteVertex = (maskId: string, idx: number) => {
    updateMask(maskId, m => {
      const pts = parsePoints(m.points);
      if (pts.length <= 3) { showToast("A polygon needs at least 3 points", "error"); return m; }
      pts.splice(idx, 1);
      return { ...m, points: stringifyPoints(pts) };
    });
  };

  const deleteMask = (id: string) => {
    setAllMasks(prev => ({ ...prev, [String(camId)]: (prev[String(camId)] ?? []).filter(m => m.id !== id) }));
    if (selectedId === id) setSelectedId(null);
  };

  const save = async () => {
    if (!settings) return;
    setSaving(true);
    try {
      const updated = { ...settings, camera_masks: serializeMasks(allMasks) };
      await api.saveSettings(updated);
      setSettings(updated);
      showToast("Masks & zones saved", "success");
      onClose();
    } catch (e: any) { showToast(e.message ?? "Save failed", "error"); }
    finally { setSaving(false); }
  };

  // Esc closes (or cancels drawing).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      if (mode === "drawing") { setMode("idle"); setDraftPoints([]); }
      else onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [mode, onClose]);

  // Helpers to convert normalized → percentage for SVG (viewport 0..100).
  const px = (p: Pt) => ({ x: p.x * 100, y: p.y * 100 });

  return (
    <div className={styles.backdrop} onClick={e => { if (e.target === e.currentTarget) onClose(); }}>
      <div className={styles.window} role="dialog" aria-modal="true">

        {/* ── Title bar ── */}
        <div className={styles.titleBar}>
          <span className={styles.title}>Mask &amp; Zone editor — CAM {camId + 1}</span>
          {mode === "drawing" && (
            <button className={`${styles.glassBtn} ${styles.glassBtnDanger}`}
              onClick={() => { setDraftPoints([]); setMode("idle"); }}>
              Cancel drawing
            </button>
          )}
          <button className={`${styles.glassBtn} ${styles.glassBtnPrimary}`} onClick={save} disabled={saving}>
            <Save size={13} /> {saving ? "Saving…" : "Save"}
          </button>
          <button className={styles.closeBtn} onClick={onClose} aria-label="Close"><X size={16} /></button>
        </div>

        <div className={styles.main}>

          {/* ── Drawing stage (aspect-locked to the video) ── */}
          <div className={styles.stage}>
            <div className={styles.canvas} style={{ aspectRatio: String(aspect) }}>
              <img
                className={styles.canvasImg}
                src={feedUrl ?? ""}
                alt="camera"
                crossOrigin="anonymous"
                onLoad={e => {
                  const im = e.currentTarget;
                  if (im.naturalWidth && im.naturalHeight) setAspect(im.naturalWidth / im.naturalHeight);
                }}
              />
              <svg
                ref={svgRef}
                className={styles.svg}
                viewBox="0 0 100 100"
                preserveAspectRatio="none"
                style={{ cursor: mode === "drawing" ? "crosshair" : "default" }}
                onClick={onSvgClick}
                onDoubleClick={onSvgDoubleClick}
                onMouseMove={onSvgMove}
                onMouseUp={onSvgUp}
                onMouseLeave={onSvgUp}
              >
                {/* Existing masks/zones */}
                {masks.map((mask, idx) => {
                  const pts = parsePoints(mask.points);
                  if (pts.length < 2) return null;
                  const sp = pts.map(px);
                  const isSel = selectedId === mask.id;
                  const stroke = maskStroke(mask, idx);
                  return (
                    <g key={mask.id}>
                      <polygon
                        points={sp.map(p => `${p.x},${p.y}`).join(" ")}
                        fill={maskFill(mask, idx)}
                        stroke={stroke}
                        strokeWidth={isSel ? 0.6 : 0.4}
                        vectorEffect="non-scaling-stroke"
                        style={{ cursor: mode === "idle" ? "move" : "default" }}
                        onClick={e => { e.stopPropagation(); if (mode === "idle") setSelectedId(isSel ? null : mask.id); }}
                        onMouseDown={e => {
                          if (mode !== "idle") return;
                          e.stopPropagation();
                          didDrag.current = false;
                          setSelectedId(mask.id);
                          polyDrag.current = { maskId: mask.id, last: coords(e) };
                        }}
                      />
                      {(() => {
                        const c = px(centroid(pts));
                        return (
                          <text x={c.x} y={c.y} textAnchor="middle" dominantBaseline="middle"
                            style={{ pointerEvents: "none", userSelect: "none", fontWeight: 700, fill: "#fff", fontSize: 3.4, paintOrder: "stroke", stroke: "#000", strokeWidth: 0.6 }}>
                            {mask.name}
                          </text>
                        );
                      })()}
                      {isSel && sp.map((p, i) => (
                        <circle key={i} cx={p.x} cy={p.y} r={1.4}
                          fill="#fff" stroke={stroke} strokeWidth={0.5}
                          vectorEffect="non-scaling-stroke"
                          style={{ cursor: "grab" }}
                          onMouseDown={e => { e.stopPropagation(); didDrag.current = false; setDrag({ maskId: mask.id, ptIdx: i }); }}
                          onContextMenu={e => { e.preventDefault(); e.stopPropagation(); deleteVertex(mask.id, i); }}
                        />
                      ))}
                    </g>
                  );
                })}

                {/* Draft polygon */}
                {draftPoints.length > 0 && (() => {
                  const sp = draftPoints.map(px);
                  const hov = hoverPoint ? px(hoverPoint) : null;
                  const stroke = MASK_TYPE_COLOR[drawType] ?? ZONE_PALETTE[0];
                  const all = hov ? [...sp, hov] : sp;
                  return (
                    <g>
                      {all.length >= 3 && (
                        <polygon points={all.map(p => `${p.x},${p.y}`).join(" ")}
                          fill={stroke + "1F"} stroke={stroke} strokeWidth={0.4}
                          strokeDasharray="1.5 1" vectorEffect="non-scaling-stroke" />
                      )}
                      {all.slice(0, -1).map((p, i) => (
                        <line key={i} x1={p.x} y1={p.y} x2={all[i + 1].x} y2={all[i + 1].y}
                          stroke={stroke} strokeWidth={0.4} strokeDasharray="1.5 1" vectorEffect="non-scaling-stroke" />
                      ))}
                      {sp.map((p, i) => (
                        <circle key={i} cx={p.x} cy={p.y} r={i === 0 ? 1.8 : 1.3}
                          fill={i === 0 ? stroke : "#fff"} stroke={stroke} strokeWidth={0.5}
                          vectorEffect="non-scaling-stroke" />
                      ))}
                    </g>
                  );
                })()}
              </svg>

              <div className={styles.hint}>
                {mode === "drawing"
                  ? (draftPoints.length < 3
                      ? `Click to place point ${draftPoints.length + 1} (need 3+)`
                      : "Click the first point ● to close")
                  : selected
                    ? "Drag polygon to move · drag dots to reshape · double-click edge to add · right-click dot to remove"
                    : ""}
              </div>
            </div>
          </div>

          {/* ── Sidebar ── */}
          <div className={styles.sidebar}>

            {/* Add new */}
            <div className={styles.sideSection}>
              <div className={styles.sideLabel}>Add new</div>
              <div className={styles.typeRow}>
                {(["motion", "object", "zone", "speed", "line"] as MaskType[]).map(t => {
                  // Third copy of the mask-type palette; all three now read MASK_TYPE_COLOR.
                  const base = MASK_TYPE_COLOR[t] ?? ZONE_PALETTE[0];
                  const pal = { b: `color-mix(in srgb, ${base} 50%, transparent)`,
                                bg: `color-mix(in srgb, ${base} 14%, transparent)`,
                                fg: base };
                  const active = drawType === t;
                  return (
                    <button key={t} className={styles.typeBtn}
                      onClick={() => setDrawType(t)}
                      title={t === "motion" ? "Exclude this area from motion detection"
                        : t === "object" ? "Discard detections whose bottom-center falls here"
                        : t === "speed" ? "4-point ground quad for speed estimation — set its real-world size below"
                        : t === "line" ? "2-point tripwire — fires when something crosses it"
                          : "Named region — alerts when a tracked object enters"}
                      style={active ? { borderColor: pal.b, background: pal.bg, color: pal.fg } : undefined}>
                      {t === "motion" ? "Motion" : t === "object" ? "Object" : t === "speed" ? "Speed" : t === "line" ? "Line" : "Zone"}
                    </button>
                  );
                })}
              </div>
              <input className={styles.input}
                placeholder={drawType === "zone" ? "Zone name (e.g. Driveway)" : "Name (optional)"}
                value={newName} onChange={e => setNewName(e.target.value)} />
              <button className={`${styles.startBtn} ${mode === "drawing" ? styles.startBtnActive : ""}`}
                onClick={() => mode === "drawing" ? closeDraft() : (setMode("drawing"), setDraftPoints([]), setSelectedId(null))}>
                {mode === "drawing" ? <>Finish polygon</> : <><Plus size={13} /> Start drawing</>}
              </button>
            </div>

            {/* List */}
            <div className={styles.list}>
              {masks.length === 0 && (
                <div className={styles.empty}>No masks or zones yet.<br />Pick a type and start drawing.</div>
              )}
              {masks.map((mask, idx) => (
                <div key={mask.id}
                  className={`${styles.row} ${selectedId === mask.id ? styles.rowActive : ""}`}
                  style={selectedId === mask.id ? { borderLeftColor: maskStroke(mask, idx) } : undefined}
                  onClick={() => setSelectedId(mask.id === selectedId ? null : mask.id)}>
                  <span className={styles.swatch} style={{ background: maskStroke(mask, idx) }} />
                  <div className={styles.rowMain}>
                    <div className={styles.rowName}>{mask.name}</div>
                    <div className={styles.rowMeta}>
                      {mask.type === "motion" ? "Motion mask" : mask.type === "object" ? "Object mask"
                        : mask.type === "speed" ? "Speed zone" : mask.type === "line" ? "Line" : "Zone"}
                      {" · "}{parsePoints(mask.points).length} pts
                    </div>
                  </div>
                  <button className={styles.iconBtn} title="Delete"
                    onClick={e => { e.stopPropagation(); deleteMask(mask.id); }}>
                    <Trash2 size={13} />
                  </button>
                </div>
              ))}
            </div>

            {/* Zone properties (selected zone only) */}
            {selected && (
              <div className={styles.zoneProps}>
                <label className={styles.propLabel}>
                  <span><Pencil size={10} style={{ verticalAlign: "middle", marginRight: 4 }} />Name</span>
                  <input className={styles.input} style={{ marginBottom: 0 }} value={selected.name}
                    onChange={e => updateMask(selected.id, m => ({ ...m, name: e.target.value }))} />
                </label>

                {selected.type === "speed" && (
                  <div className={styles.propLabel}>
                    <span>Real-world size of this ground area</span>
                    <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                      <input type="number" min={0.5} max={500} step={0.5} className={styles.input}
                        style={{ marginBottom: 0, width: 76 }}
                        value={selected.width_m ?? 5}
                        onChange={e => updateMask(selected.id, m => ({ ...m, width_m: Number(e.target.value) || 0 }))} />
                      <span style={{ fontSize: 11, color: "var(--text-muted)" }}>m wide ×</span>
                      <input type="number" min={0.5} max={500} step={0.5} className={styles.input}
                        style={{ marginBottom: 0, width: 76 }}
                        value={selected.height_m ?? 5}
                        onChange={e => updateMask(selected.id, m => ({ ...m, height_m: Number(e.target.value) || 0 }))} />
                      <span style={{ fontSize: 11, color: "var(--text-muted)" }}>m deep</span>
                    </div>
                    <span style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 4, display: "block" }}>
                      4 corners on flat ground, in order.
                    </span>
                  </div>
                )}

                {selected.type === "zone" && (
                  <>
                    <div className={styles.propLabel}>
                      <span>Objects that count <em>{(selected.objects?.length ?? 0) === 0 ? "any" : selected.objects!.join(", ")}</em></span>
                      <div className={styles.chipRow}>
                        {ZONE_OBJECTS.map(o => {
                          const on = selected.objects?.includes(o);
                          return (
                            <button key={o} className={`${styles.chip} ${on ? styles.chipActive : ""}`}
                              onClick={() => updateMask(selected.id, m => {
                                const cur = new Set(m.objects ?? []);
                                if (cur.has(o)) cur.delete(o); else cur.add(o);
                                return { ...m, objects: [...cur] };
                              })}>{o}</button>
                          );
                        })}
                      </div>
                    </div>

                    <label className={styles.propLabel}>
                      <span>Inertia (frames) <em>{selected.inertia ?? 3}</em></span>
                      <input type="range" min={1} max={10} step={1} value={selected.inertia ?? 3}
                        onChange={e => updateMask(selected.id, m => ({ ...m, inertia: Number(e.target.value) }))} />
                    </label>

                    <label className={styles.propLabel}>
                      <span>Loitering before counted <em>{selected.loitering_secs ?? 0}s</em></span>
                      <input type="range" min={0} max={60} step={1} value={selected.loitering_secs ?? 0}
                        onChange={e => updateMask(selected.id, m => ({ ...m, loitering_secs: Number(e.target.value) }))} />
                    </label>
                  </>
                )}
              </div>
            )}

            {/* Help + mature NVRs "use sparingly" guidance */}
            <div className={styles.help}>
              <strong style={{ color: MASK_TYPE_COLOR.motion }}>Motion mask</strong> — ignore motion here (timestamps, swaying trees, sky).<br />
              <strong style={{ color: MASK_TYPE_COLOR.object }}>Object mask</strong> — drop detections whose feet land here.<br />
              <strong style={{ color: ZONE_PALETTE[0] }}>Zone</strong> — alert when an object enters a named area.
              <div className={styles.warn}>
                Use masks sparingly — heavy masking hurts tracking.
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
