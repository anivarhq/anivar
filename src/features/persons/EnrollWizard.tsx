/**
 * Guided face enrollment.
 *
 * Self-contained by design: it owns its own webcam stream, its own model gate
 * and its own capture buffer, and reports back through `onDone` when the person
 * is saved. Nothing in the People panel reads its state, which is why it was the
 * cleanest 460 lines to lift out of a 2,900-line file.
 */
import { useCallback, useEffect, useRef, useState, type CSSProperties } from "react";
import { UserPlus, Camera, RefreshCw, Check, X, Sparkles } from "lucide-react";
import { api, FaceCapture, FaceDebug } from "../../api";
import { downloadSkill, findSkill } from "../agent/skillDownload";

const ENROLL_TARGET = 6;

const ENROLL_MIN_QUALITY = 0.15;  // lenient — guided capture should accept any real face; Save picks the sharpest

const ENROLL_MIN_AREA = 1600;     // ~40×40 px face minimum

const ENROLL_DIVERSITY = 0.92;    // keep a frame only if cosine to every kept one < this (a new angle)

/** Cosine of two L2-normalised ArcFace embeddings = dot product. */
function faceCosine(a: number[], b: number[]): number {
  if (a.length !== b.length) return 0;
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

const POSE_PROMPTS = [
  "Look straight at the camera",
  "Turn your head slightly left",
  "Turn your head slightly right",
  "Tilt your head up a little",
  "Tilt your head down a little",
  "One more — any angle",
];

/** Guided multi-angle enrollment from the LOCAL WEBCAM. Shows a live self-view
 *  with a face-oval guide; auto-captures sharp, large-enough faces at NEW angles
 *  (cosine-different) by drawing the video frame → backend ArcFace `embed_face`,
 *  then saves them all under one name (`enroll_person_multi`) — the same 512-d
 *  space as live + event recognition, so the person is recognised everywhere. */
export function EnrollWizard({ onDone, showToast }: {
  onDone: () => void;
  showToast: (msg: string, type?: "success" | "error" | "info") => void;
}) {
  const [name, setName] = useState("");
  const [role, setRole] = useState("resident");
  const [captures, setCaptures] = useState<FaceCapture[]>([]);
  const [capturing, setCapturing] = useState(false);
  const [saving, setSaving] = useState(false);
  const [hint, setHint] = useState("");
  const [modelReady, setModelReady] = useState<boolean | null>(null);
  const [installing, setInstalling] = useState(false);
  const [installPct, setInstallPct] = useState(0);
  const [camReady, setCamReady] = useState(false);
  const [camError, setCamError] = useState<string | null>(null);
  const [statusErr, setStatusErr] = useState<string | null>(null);
  // Default to UPLOAD — no camera/inference spins up until the user explicitly
  // picks live capture, which saves resources for the common "add from a photo" case.
  const [mode, setMode] = useState<"live" | "upload">("upload");
  const [uploadNote, setUploadNote] = useState<string | null>(null);
  const [faceDbg, setFaceDbg] = useState<FaceDebug | null>(null);

  const videoRef  = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const capturesRef = useRef<FaceCapture[]>([]);
  useEffect(() => { capturesRef.current = captures; }, [captures]);
  const noFaceTicks = useRef(0);
  const diagnosed = useRef(false);

  // SERVER-FEED fallback: the USB webcam is now owned server-side (dshow ffmpeg), so
  // getUserMedia can't open it. When that fails we capture frames from the backend's
  // camera feed over IPC (get_camera_snapshot) instead of failing with "Camera
  // unavailable". serverFeed drives the UI; serverB64Ref holds the latest raw frame.
  const [serverFeed, setServerFeed]   = useState(false);
  const [serverFrame, setServerFrame] = useState<string | null>(null);
  const serverB64Ref  = useRef<string | null>(null);
  const serverFeedRef = useRef(false);
  useEffect(() => { serverFeedRef.current = serverFeed; }, [serverFeed]);
  // Which backend camera feeds the enrollment preview. Was hardcoded to cam 0 —
  // multi-camera households couldn't enroll from any other camera.
  const [enrollCam, setEnrollCam]   = useState(0);
  const [activeCams, setActiveCams] = useState<number[]>([]);
  useEffect(() => {
    if (mode !== "live") return;
    api.getActiveCameras().then(ids => {
      setActiveCams(ids);
      // If cam 0 isn't live but others are, start on a live one.
      if (ids.length > 0 && !ids.includes(0)) setEnrollCam(ids[0]);
    }).catch(() => {});
  }, [mode]);

  // Does the face pipeline actually LOAD? `face_debug` reports per-model status
  // (detector + ArcFace) so a missing/partial/incompatible model shows a clear
  // reason — and feeds the visible "Face model" status line below.
  const checkPipeline = useCallback(() => {
    api.faceDebug()
      .then(d => {
        setFaceDbg(d);
        const ok = d.detector_loaded && d.embedder_loaded;
        setModelReady(ok);
        setStatusErr(ok ? null : (d.note || "Face model not loaded"));
      })
      .catch(e => { setModelReady(false); setStatusErr(String(e)); });
  }, []);
  useEffect(() => { checkPipeline(); }, [checkPipeline]);

  // Start the LOCAL WEBCAM once the model is ready. Always a fresh feed (unlike
  // the shared store frame, which is only live on the Live tab). Stop every track
  // on cleanup so the camera light goes off when leaving.
  useEffect(() => {
    if (modelReady !== true || mode !== "live") return;
    let cancelled = false;
    (async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false,
        });
        if (cancelled) { stream.getTracks().forEach(t => t.stop()); return; }
        streamRef.current = stream;
        const v = videoRef.current;
        if (v) { v.srcObject = stream; await v.play().catch(() => {}); }
        setServerFeed(false);
        setCamReady(true);
      } catch {
        // getUserMedia failed — almost always because the USB cam is held by the
        // server-side dshow capture. Fall back to the backend's camera feed (IPC).
        if (!cancelled) setServerFeed(true);
      }
    })();
    return () => {
      cancelled = true;
      streamRef.current?.getTracks().forEach(t => t.stop());
      streamRef.current = null;
      setCamReady(false);
    };
  }, [modelReady, mode]);

  // SERVER-FEED poll: when getUserMedia isn't available, pull the latest frame from the
  // backend camera (user-selectable; defaults to the first LIVE camera) over IPC for
  // both the self-view and capture. Only surfaces "camera unavailable" if NO frames
  // arrive for a few seconds.
  useEffect(() => {
    if (!serverFeed || mode !== "live") return;
    let alive = true, misses = 0;
    const poll = async () => {
      try {
        const b64 = await api.getCameraSnapshot(enrollCam);
        if (!alive) return;
        if (b64) {
          serverB64Ref.current = b64;
          setServerFrame(`data:image/jpeg;base64,${b64}`);
          setCamReady(true); setCamError(null); misses = 0;
        } else if (++misses >= 20) {           // ~3s with no frames → genuinely off
          setCamError("No camera feed");
        }
      } catch (e) {
        if (alive && ++misses >= 20) setCamError(String(e));
      }
    };
    poll();
    const iv = setInterval(poll, 250); // ~4fps preview — plenty for enrollment, half the IPC
    return () => { alive = false; clearInterval(iv); };
  }, [serverFeed, mode, enrollCam]);

  // Capture loop — grab the live VIDEO frame, embed via ArcFace, keep diverse angles.
  useEffect(() => {
    if (!capturing) return;
    let alive = true, busy = false;
    const tick = async () => {
      if (!alive || busy) return;
      // Grab the current frame from whichever source is live: the server camera feed
      // (dshow, over IPC) or the local getUserMedia <video> drawn to canvas.
      let b64: string | null = null;
      if (serverFeedRef.current) {
        b64 = serverB64Ref.current;
      } else {
        const v = videoRef.current, cnv = canvasRef.current;
        if (v && cnv && v.readyState >= 2 && v.videoWidth) {
          cnv.width = v.videoWidth; cnv.height = v.videoHeight;
          const ctx = cnv.getContext("2d");
          if (ctx) {
            ctx.drawImage(v, 0, 0, cnv.width, cnv.height); // true (un-mirrored) frame
            b64 = cnv.toDataURL("image/jpeg", 0.85).split(",")[1];
          }
        }
      }
      if (!b64) return;
      busy = true;
      try {
        const cap = await api.embedFace(b64);
        const cur = capturesRef.current;
        if (cur.length >= ENROLL_TARGET) return;
        if (!cap) {
          setHint("No face — look at the camera.");
          noFaceTicks.current += 1;
          // After ~3 s with no detection, run the diagnostic ONCE on this frame so
          // the status line says exactly why (0 faces / low conf / not embedded).
          if (noFaceTicks.current === 6 && !diagnosed.current) {
            diagnosed.current = true;
            api.faceDebug(b64).then(setFaceDbg).catch(() => {});
          }
          return;
        }
        noFaceTicks.current = 0; // a face was found — detector is working
        if (cap.quality < ENROLL_MIN_QUALITY)  { setHint("Hold still — a little blurry."); return; }
        if (cap.area < ENROLL_MIN_AREA)        { setHint("Move closer to the camera."); return; }
        const dup = cur.some(c => faceCosine(c.embedding, cap.embedding) >= ENROLL_DIVERSITY);
        if (dup) { setHint(POSE_PROMPTS[Math.min(cur.length, POSE_PROMPTS.length - 1)]); return; }
        setCaptures(cs => [...cs, cap]);
        const nextLen = cur.length + 1;
        setHint(nextLen >= ENROLL_TARGET
          ? "Got enough — tap Save."
          : `Captured! ${POSE_PROMPTS[Math.min(nextLen, POSE_PROMPTS.length - 1)]}`);
      } catch { /* transient frame */ } finally { busy = false; }
    };
    const iv = setInterval(tick, 500);
    return () => { alive = false; clearInterval(iv); };
  }, [capturing]);

  // Auto-stop once we have enough angles.
  useEffect(() => {
    if (capturing && captures.length >= ENROLL_TARGET) {
      setCapturing(false);
      setHint("Got enough angles — review and save.");
    }
  }, [captures.length, capturing]);

  const installModel = async () => {
    const skill = findSkill("face_small");
    if (!skill) { showToast("Face model unavailable", "error"); return; }
    setInstalling(true); setInstallPct(0);
    try {
      await downloadSkill(skill, pct => setInstallPct(pct));
      checkPipeline(); // verify it actually loads, not just that files landed
      showToast("Face model installed", "success");
    } catch (e) {
      showToast(`Install failed: ${e}`, "error");
    } finally { setInstalling(false); }
  };

  // Upload photos → embed each via the SAME ArcFace model → collect as captures.
  const onFiles = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    setUploadNote(`Processing ${files.length} photo${files.length === 1 ? "" : "s"}…`);
    let added = 0, noFace = 0;
    for (const file of Array.from(files)) {
      try {
        const dataUrl: string = await new Promise((res, rej) => {
          const r = new FileReader();
          r.onload = () => res(String(r.result));
          r.onerror = rej;
          r.readAsDataURL(file);
        });
        const b64 = dataUrl.includes(",") ? dataUrl.split(",")[1] : dataUrl;
        const cap = await api.embedFace(b64);
        if (cap) { setCaptures(cs => [...cs, cap]); added++; }
        else noFace++;
      } catch { noFace++; }
    }
    setUploadNote(
      `${added} face${added === 1 ? "" : "s"} added` +
      (noFace > 0 ? ` · ${noFace} photo${noFace === 1 ? "" : "s"} had no detectable face` : ""));
  };

  const save = async () => {
    if (!name.trim() || captures.length === 0) return;
    setSaving(true);
    try {
      const best = [...captures].sort((a, b) => b.quality - a.quality)[0];
      await api.enrollPersonMulti(
        name.trim(), role,
        captures.map(c => c.embedding),
        best ? `data:image/jpeg;base64,${best.thumbnail_b64}` : null,
      );
      showToast(`${name.trim()} enrolled (${captures.length} angle${captures.length === 1 ? "" : "s"})`, "success");
      onDone();
    } catch (e) {
      showToast(String(e), "error");
      setSaving(false);
    }
  };

  // ── Model-install gate ──────────────────────────────────────────────────
  if (modelReady === false) {
    return (
      <div className="glass" style={{ padding: 28, maxWidth: 460, margin: "0 auto", textAlign: "center",
        display: "flex", flexDirection: "column", alignItems: "center", gap: 16 }}>
        <Sparkles size={36} style={{ opacity: 0.4, color: "var(--accent)" }} />
        <div>
          <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 6 }}>Face model required</div>
          {statusErr && (
            <div style={{ fontSize: 11, color: "var(--accent-red)", marginTop: 8, maxWidth: 340, lineHeight: 1.5 }}>
              {statusErr.replace(/^.*Error:\s*/, "")}
            </div>
          )}
        </div>
        <button onClick={installModel} disabled={installing} className="btn-primary" style={{ padding: "10px 18px" }}>
          {installing ? `Installing… ${installPct}%` : "Install face model"}
        </button>
      </div>
    );
  }

  const progressPct = Math.round((captures.length / ENROLL_TARGET) * 100);

  return (
    <div className="glass" style={{ padding: 24, maxWidth: 480, margin: "0 auto" }}>
      <div style={{ marginBottom: 16 }}>
        <label style={enrollLabel}>Full Name</label>
        <input value={name} onChange={e => setName(e.target.value)} placeholder="e.g. John Smith" style={enrollInput} />
      </div>
      <div style={{ marginBottom: 18 }}>
        <label style={enrollLabel}>Role</label>
        <select value={role} onChange={e => setRole(e.target.value)} style={enrollInput}>
          <option value="resident">Resident / Family</option>
          <option value="employee">Employee / Staff</option>
          <option value="visitor">Trusted Visitor</option>
        </select>
      </div>

      {/* Mode toggle — live webcam vs photo upload (mature NVRs "Add Face") */}
      <div style={{ display: "inline-flex", gap: 4, padding: 4, borderRadius: 999,
        background: "rgb(var(--ink) / 0.04)", marginBottom: 14 }}>
        {([["upload", "Upload photos"], ["live", "Live capture"]] as const).map(([m, label]) => (
          <button key={m} type="button" onClick={() => { setMode(m); setCapturing(false); }}
            style={{ padding: "6px 14px", borderRadius: 999, border: "none", fontSize: 12, fontWeight: 600,
              cursor: "pointer", background: mode === m ? "var(--accent)" : "transparent",
              color: mode === m ? "var(--on-accent)" : "var(--text-secondary)" }}>
            {label}
          </button>
        ))}
      </div>

      {/* Face-model status — the two bundled models by name, with live dots */}
      {faceDbg && (
        <div style={{ marginBottom: 12, padding: "8px 10px", borderRadius: 10,
          background: "rgb(var(--ink) / 0.03)", border: "1px solid var(--border)" }}>
          <div style={{ fontSize: 9, fontWeight: 700, letterSpacing: 0.05, textTransform: "uppercase",
            color: "var(--text-tertiary)", marginBottom: 6 }}>Face model</div>
          {([["yolov8n-face — detector", faceDbg.detector_loaded] as const,
             ["ArcFace — recognizer",    faceDbg.embedder_loaded] as const]).map(([label, ok]) => (
            <div key={label} style={{ display: "flex", alignItems: "center", gap: 7, fontSize: 11, padding: "1px 0" }}>
              <span style={{ width: 7, height: 7, borderRadius: 999, background: ok ? "var(--accent)" : "var(--accent-red)" }} />
              <span style={{ color: "var(--text-secondary)" }}>{label} · {ok ? "loaded" : "not loaded"}</span>
            </div>
          ))}
          {faceDbg.note && <div style={{ fontSize: 10, color: "var(--text-tertiary)", marginTop: 4 }}>{faceDbg.note}</div>}
        </div>
      )}

      {mode === "live" ? (<>
      {/* Camera picker — only when the SERVER feed is in use and several cameras
          are live (getUserMedia already picks its own device). Was hardcoded cam 0. */}
      {serverFeed && activeCams.length > 1 && (
        <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 10, flexWrap: "wrap" }}>
          <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: 0.04, textTransform: "uppercase",
            color: "var(--text-tertiary)" }}>Camera</span>
          {activeCams.map(id => (
            <button key={id}
              onClick={() => { setEnrollCam(id); setCamError(null); }}
              style={{
                fontSize: 11, fontWeight: 700, padding: "4px 12px", borderRadius: 999, cursor: "pointer",
                border: `1px solid ${enrollCam === id ? "var(--accent)" : "var(--border)"}`,
                background: enrollCam === id ? "var(--hl)" : "transparent",
                color: enrollCam === id ? "var(--accent)" : "var(--text-secondary)",
              }}>
              CAM {id + 1}
            </button>
          ))}
        </div>
      )}
      {/* Live self-view + face-oval guide */}
      <div style={{ position: "relative", borderRadius: 18, marginBottom: 14, overflow: "hidden",
        background: "#000", border: "1px solid var(--border)", aspectRatio: "4 / 3" }}>
        {serverFeed ? (
          // Server-side camera feed (dshow) over IPC — getUserMedia couldn't open the
          // USB cam because ffmpeg owns it. Mirrored for a natural selfie view.
          serverFrame && <img src={serverFrame} alt="" draggable={false}
            style={{ width: "100%", height: "100%", objectFit: "cover", transform: "scaleX(-1)" }} />
        ) : (
          <video ref={videoRef} autoPlay muted playsInline
            style={{ width: "100%", height: "100%", objectFit: "cover", transform: "scaleX(-1)" }} />
        )}
        <canvas ref={canvasRef} style={{ display: "none" }} />
        {/* Spotlight oval — darkens outside, highlights where to put your face */}
        <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", pointerEvents: "none" }}>
          <div style={{ width: "48%", height: "80%", borderRadius: "50%",
            border: `3px solid ${capturing ? "var(--accent)" : "rgb(var(--ink) / 0.55)"}`,
            boxShadow: "0 0 0 9999px rgba(0,0,0,0.38)", transition: "border-color 200ms" }} />
        </div>
        {!camReady && !camError && (
          <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", color: "#fff", fontSize: 12 }}>
            Starting camera…
          </div>
        )}
        {capturing && (
          <div style={{ position: "absolute", top: 10, left: 10, padding: "4px 10px", borderRadius: 999,
            fontSize: 11, fontWeight: 700, background: "rgba(220,40,40,0.85)", color: "#fff",
            display: "flex", alignItems: "center", gap: 6 }}>
            <span style={{ width: 7, height: 7, borderRadius: 999, background: "#fff" }} /> CAPTURING
          </div>
        )}
      </div>

      {camError && (
        <div style={{ fontSize: 11, color: "var(--accent-red)", marginBottom: 12, textAlign: "center", lineHeight: 1.5 }}>
          Camera unavailable — grant camera permission, or name people your security cameras saw in the <strong>Train</strong> tab.
        </div>
      )}

      {/* Progress + hint */}
      <div style={{ marginBottom: 12 }}>
        <div style={{ display: "flex", justifyContent: "space-between", gap: 10, fontSize: 11, color: "var(--text-tertiary)", marginBottom: 5 }}>
          <span style={{ fontWeight: 700, color: "var(--text-secondary)" }}>{captures.length} / {ENROLL_TARGET} angles</span>
          <span style={{ textAlign: "right" }}>{hint}</span>
        </div>
        <div style={{ height: 5, borderRadius: 999, background: "rgb(var(--ink) / 0.06)", overflow: "hidden" }}>
          <div style={{ height: "100%", width: `${progressPct}%`,
            background: progressPct >= 100 ? "var(--accent)" : "var(--accent-amber)",
            borderRadius: 999, transition: "width 300ms" }} />
        </div>
      </div>
      </>) : (<>
        {/* Upload photos (mature NVRs "Add Face") — embeds each via the same ArcFace model */}
        <label style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
          gap: 10, padding: "28px 18px", marginBottom: 12, borderRadius: 18, cursor: "pointer",
          border: "1.5px dashed var(--border-strong)", background: "rgb(var(--ink) / 0.02)", textAlign: "center" }}>
          <UserPlus size={28} style={{ opacity: 0.4 }} />
          <div style={{ fontSize: 13, fontWeight: 600 }}>Choose photos of this person</div>
          <div style={{ fontSize: 11, color: "var(--text-tertiary)", maxWidth: 320, lineHeight: 1.5 }}>
            3–6 clear, front-facing photos
          </div>
          <input type="file" accept="image/*" multiple style={{ display: "none" }}
            onChange={e => { onFiles(e.target.files); e.currentTarget.value = ""; }} />
        </label>
        {uploadNote && (
          <div style={{ fontSize: 11, color: "var(--text-secondary)", marginBottom: 12, textAlign: "center" }}>{uploadNote}</div>
        )}
      </>)}

      {/* Captured crops */}
      {captures.length > 0 && (
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 14 }}>
          {captures.map((c, i) => (
            <div key={i} style={{ position: "relative", width: 52, height: 52, borderRadius: 10, overflow: "hidden", border: "1px solid var(--border)" }}>
              <img src={`data:image/jpeg;base64,${c.thumbnail_b64}`} alt="" style={{ width: "100%", height: "100%", objectFit: "cover" }} />
              <button onClick={() => setCaptures(cs => cs.filter((_, j) => j !== i))}
                style={{ position: "absolute", top: 2, right: 2, width: 16, height: 16, borderRadius: 999, border: "none",
                  background: "rgba(0,0,0,0.6)", color: "#fff", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center" }}>
                <X size={9} />
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Actions */}
      <div style={{ display: "flex", gap: 10 }}>
        {mode === "live" && (!capturing ? (
          <button onClick={() => { setHint(POSE_PROMPTS[0]); setCapturing(true); }}
            disabled={!camReady || captures.length >= ENROLL_TARGET}
            style={{ flex: 1, padding: "10px 0", borderRadius: 999, border: "1px solid var(--border-strong)",
              background: "transparent", color: "var(--text-primary)", cursor: camReady ? "pointer" : "not-allowed", fontSize: 13,
              display: "flex", alignItems: "center", justifyContent: "center", gap: 6,
              opacity: !camReady ? 0.5 : 1 }}>
            <Camera size={14} /> {captures.length === 0 ? "Start capture" : "Capture more"}
          </button>
        ) : (
          <button onClick={() => { setCapturing(false); setHint("Paused — resume or save."); }}
            style={{ flex: 1, padding: "10px 0", borderRadius: 999, border: "1px solid var(--accent-amber)",
              background: "transparent", color: "var(--accent-amber)", cursor: "pointer", fontSize: 13 }}>
            Pause
          </button>
        ))}
        <button onClick={save} disabled={saving || !name.trim() || captures.length === 0} className="btn-primary"
          style={{ flex: 1, padding: "10px 0", opacity: saving || !name.trim() || captures.length === 0 ? 0.5 : 1 }}>
          <UserPlus size={14} /> {saving ? "Saving…" : `Save${captures.length ? ` (${captures.length})` : ""}`}
        </button>
      </div>

      <p style={{ fontSize: 11, color: "var(--text-tertiary)", marginTop: 14, textAlign: "center", lineHeight: 1.5 }}>
        {mode === "live"
          ? "Keep your face in the oval and turn your head slowly — front, left, right, up, down."
          : "A few varied photos — different angles and lighting."}
      </p>
    </div>
  );
}

// ── Helpers ─────────────────────────────────────────────────────────────────

const enrollLabel: CSSProperties = { display: "block", fontSize: 11, fontWeight: 600, color: "var(--text-tertiary)", marginBottom: 6 };

const enrollInput: CSSProperties = {
  width: "100%", padding: "10px 14px", borderRadius: 12, fontSize: 14,
  border: "1px solid var(--border-strong)", background: "rgb(var(--ink) / 0.04)",
  color: "var(--text-primary)", outline: "none",
};
