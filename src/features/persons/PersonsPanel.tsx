/**
 * PersonsPanel — the ONE identity home (mature NVRs Face Library model).
 *
 * A single vertically-composed section (no tabs):
 *   1. Roster — enrolled people grid + "Recently recognized" strip (each
 *      recognition shows its provenance: method, score, "why this name?").
 *   2. "Needs your review" — unknown faces/clusters awaiting a tag (mature NVRs
 *      Train): everything in `face_embeddings` without a `person_id`.
 *   3. "Tracked across cameras" — body Re-ID tracks; body matches only
 *      PROPOSE a name, a face (or the user) confirms.
 * Enrollment is a modal launched from the header. Vehicles/Sounds moved to
 * the Review section (they're event browsers, not identity).
 *
 * All suggestion/correction actions bind by PERSON ID, never display name
 * (renames and duplicate names mis-resolve by name).
 *
 * Visual layer: `.glass` utility classes from `src/index.css`.
 */

import { useEffect, useRef, useState, useCallback, useMemo, type ReactNode, type CSSProperties } from "react";
import { createPortal } from "react-dom";
import { UserPlus, Trash2, Camera, RefreshCw, Users, Sparkles, Check, X, Layers, Clock, Activity, MapPin, Play, Pencil } from "lucide-react";
import { api, KnownPerson, UnknownFace, TrackedPerson, TrackedCluster, FaceShot, Recognition, FaceCapture, UnknownCluster, FaceDebug, FaceClassifierStatus, PersonSighting, PersonEvent, PersonStats } from "../../api";
import { listen } from "@tauri-apps/api/event";
import { useStore } from "../../store";
import { eventThumbSrc, faceCropSrc, bodyCropSrc } from "../../lib/eventThumb";
import { useShallow } from "zustand/react/shallow";
import { downloadSkill, findSkill } from "../agent/skillDownload";
import { usePanelCache } from "../../lib/panelCache";
import { Modal, useDismiss } from "../../components/ui/Modal";
import { fmtWhen } from "../../lib/time";
import { OBJECT_COLOR_SWATCH } from "../../lib/palette";
import {
  RemovePersonDialog, PersonPickList, SectionHeader, ZoomableImg, FaceContextZoom,
  Stat, Section, Empty, summaryText, methodLabel, whyNamed, formatRelative,
  fmtDay, OutfitLine,
} from "./shared";
import { CardGrid, ProfileCard, ProfileMedia } from "../review/Card";
import { EnrollWizard } from "./EnrollWizard";
import { TrackedSection } from "./TrackedSection";
import { ReviewSection } from "./ReviewSection";
import { PersonDetail } from "./PersonDetail";

export function PersonsPanel() {
  // Enrollment is a MODAL launched from the People header (it was a whole tab).
  const [enrollOpen, setEnrollOpen] = useState(false);
  // Stale-while-revalidate: the fan-out lists survive tab switches, so a
  // revisit paints instantly while refresh() revalidates in the background.
  const [persons, setPersons] = usePanelCache<KnownPerson[]>("people.persons", []);
  const [unknowns, setUnknowns] = usePanelCache<UnknownFace[]>("people.unknowns", []);
  const [tracked, setTracked] = usePanelCache<TrackedPerson[]>("people.tracked", []);
  const [recognitions, setRecognitions] = usePanelCache<Recognition[]>("people.recognitions", []);
  const [clusters, setClusters] = usePanelCache<UnknownCluster[]>("people.clusters", []);
  const [personStats, setPersonStats] = usePanelCache<PersonStats[]>("people.personStats", []);
  const [detailPerson, setDetailPerson] = useState<KnownPerson | null>(null);
  const [loading, setLoading] = useState(false);
  /** Has the fan-out completed at least once this session?
   *
   *  `loading` only ever spun the header icon, so a cold start painted
   *  RosterSection's "No people enrolled yet", ReviewSection's "No unidentified
   *  faces" and TrackedSection's "No tracked people yet" all at once, with
   *  `persons=[]`, before the first fetch resolved. usePanelCache hides
   *  that on REVISITS only — every genuinely cold start flashed three
   *  "you have nothing" cards at someone who has plenty. */
  const [everLoaded, setEverLoaded] = useState(false);
  // Never silently swallow a backend failure — a model-load crash, DB lock, or
  // embedding dim-mismatch must look DIFFERENT from "no people yet" (the old
  // `.catch(() => [])` made them identical). Capture per-loader errors + the
  // face-model health so the panel shows a banner + Retry instead of a blank tab.
  const [loadError, setLoadError] = useState<string | null>(null);
  const [health, setHealth] = useState<FaceDebug | null>(null);
  const [classifier, setClassifier] = useState<FaceClassifierStatus | null>(null);
  const [reidBackend, setReidBackend] = useState<string | null>(null);
  const { showToast } = useStore(useShallow(s => ({ showToast: s.showToast })));

  const refresh = useCallback(async () => {
    setLoading(true);
    const errs: string[] = [];
    // Each loader resolves independently — one failure shouldn't blank the rest —
    // but every failure is recorded so the UI can surface it.
    const guard = async <T,>(p: Promise<T>, fallback: T, label: string): Promise<T> => {
      try { return await p; }
      catch (e) { errs.push(`${label}: ${String(e).replace(/^.*Error:\s*/, "")}`); return fallback; }
    };
    try {
      const [p, u, t, r, cl, ps] = await Promise.all([
        guard(api.listKnownPersons(), [] as KnownPerson[], "Roster"),
        guard(api.listRecentUnknownFaces(60, 14, 0.10), [] as UnknownFace[], "Unidentified faces"),
        guard(api.listTrackedPersons(), [] as TrackedPerson[], "Tracked"),
        guard(api.listRecentRecognitions(30, 7), [] as Recognition[], "Recognitions"),
        guard(api.listUnknownClusters(30, 0.10), [] as UnknownCluster[], "Clusters"),
        guard(api.getPersonStats(), [] as PersonStats[], "Activity patterns"),
      ]);
      setPersons(p);
      setUnknowns(u);
      setTracked(t);
      setRecognitions(r);
      setClusters(cl);
      setPersonStats(ps);
      setLoadError(errs.length ? errs.join(" · ") : null);
      // Face-model health (independent of the lists) — drives the status strip.
      setHealth(await api.faceDebug().catch(() => null));
      // Hybrid matching head — drives the "Recognition mode" line + Trained chips.
      setClassifier(await api.faceClassifierStatus().catch(() => null));
      // Active Re-ID backbone — drives the Tracked-tab engine caption.
      setReidBackend(await api.reidBackendStatus().catch(() => null));
    } finally {
      setEverLoaded(true);
      setLoading(false);
    }
  }, []);
  useEffect(() => { refresh(); }, [refresh]);
  // Returning after the window was hidden → refresh, but throttled: Alt-Tabbing
  // repeatedly used to re-issue the whole 9-command fan-out every single time.
  const lastRefreshRef = useRef(0);
  useEffect(() => {
    const onVis = () => {
      if (document.hidden) return;
      if (Date.now() - lastRefreshRef.current < 60_000) return;
      lastRefreshRef.current = Date.now();
      refresh();
    };
    document.addEventListener("visibilitychange", onVis);
    return () => document.removeEventListener("visibilitychange", onVis);
  }, [refresh]);

  /** How many things are actually waiting for the user, counted ONCE.
   *
   *  There used to be three of these on screen simultaneously — the header
   *  subtitle counted unknown faces, the header badge counted unknown faces
   *  plus unnamed body tracks, and the section header counted unknown faces
   *  plus clusters. All three were labelled as the same thing. */
  /** Configured camera names, for the movement trail. Same one-liner Review
   *  uses; a trail that reads "camera 1 -> camera 3" is a trail nobody reads. */
  const [camConfigs, setCamConfigs] = useState<Array<{ cam_id: number; name: string }>>([]);
  useEffect(() => { api.getCameraConfigs().then(setCamConfigs).catch(() => {}); }, []);
  const cameraName = useCallback(
    (id: number) => camConfigs.find(c => c.cam_id === id)?.name || `Camera ${id + 1}`,
    [camConfigs]);

  /** Stats keyed by PERSON ID, built once and shared by the roster cards and the
   *  detail view — the detail would otherwise re-fetch data already in memory. */
  const statsById = useMemo(() => {
    const m = new Map<string, PersonStats>();
    for (const st of personStats) if (st.person_id) m.set(st.person_id, st);
    return m;
  }, [personStats]);

  const reviewCount = unknowns.length + clusters.length
    + tracked.filter(tp => !tp.known_name).length;

  const [removeTarget, setRemoveTarget] = useState<{ id: string; name: string } | null>(null);
  const handleDelete = (id: string, name: string) => setRemoveTarget({ id, name });

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", overflow: "hidden" }}>
      {/* Glass header */}
      <div className="glass" style={{
        margin: 16, marginBottom: 0,
        padding: "14px 18px",
        display: "flex", alignItems: "center", gap: 14,
      }}>
        <Users size={16} style={{ color: "var(--accent)" }} />
        <span style={{ fontWeight: 700, fontSize: 14, letterSpacing: -0.01 }}>People</span>
        <span style={{ fontSize: 12, color: "var(--text-tertiary)" }}>
          {persons.length} enrolled · {reviewCount} to review
        </span>

        {/* ONE review count, computed once above. There used to be three of
            them on screen at the same time — this subtitle, the badge below,
            and the section header — all labelled the same and all disagreeing. */}
        {(() => {
          return reviewCount > 0 ? (
            <span title="People waiting for your review below" style={{
              display: "inline-flex", alignItems: "center", justifyContent: "center",
              minWidth: 18, height: 18, padding: "0 6px",
              fontSize: 10, fontWeight: 700, borderRadius: 999,
              background: "var(--accent-glow)", color: "var(--accent)",
            }}>{reviewCount}</span>
          ) : null;
        })()}
        <div style={{ marginLeft: "auto" }} />
        <button
          onClick={refresh}
          title="Refresh"
          style={{ background: "none", border: "none", color: "var(--text-tertiary)", cursor: "pointer", padding: 4 }}
        >
          <RefreshCw size={14} className={loading ? "spin" : ""} />
        </button>
      </div>

      <div style={{ flex: 1, overflow: "auto", padding: 16, paddingTop: 12 }}>
        {/* Face-model health — shown on every tab so "recognition is off" is never
            mistaken for "no people". Stays quiet when the pipeline is healthy. */}
        <FaceModelStrip health={health} onChanged={refresh} showToast={showToast} />
        {/* A loader failed — say so (with Retry) instead of pretending it's empty. */}
        {loadError && (
          <div className="glass" style={{
            padding: "11px 15px", marginBottom: 14, display: "flex", alignItems: "center", gap: 12,
            border: "1px solid color-mix(in srgb, var(--status-alert) 34%, transparent)",
          }}>
            <X size={15} style={{ color: "var(--accent-red)", flexShrink: 0 }} />
            <div style={{ flex: 1, fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.5 }}>
              <strong style={{ color: "var(--text-primary)" }}>Couldn't load some People data.</strong>
              <div style={{ fontSize: 10.5, color: "var(--text-tertiary)", marginTop: 3 }}>{loadError}</div>
            </div>
            <button onClick={refresh} className="btn-primary" style={{ flexShrink: 0, padding: "7px 14px", fontSize: 12 }}>Retry</button>
          </div>
        )}
        {!everLoaded && persons.length === 0 ? (
          <div style={{ display: "flex", alignItems: "center", justifyContent: "center",
            gap: 8, padding: "60px 0", color: "var(--text-tertiary)", fontSize: 12 }}>
            <RefreshCw size={14} className="spin" /> Loading people…
          </div>
        ) : (
          <>
            {/* ── 1. Roster: enrolled people + recognition strips + corrections ── */}
            <RosterSection persons={persons} recognitions={recognitions} onDelete={handleDelete}
              onOpenDetail={setDetailPerson}
              onJumpToReview={() => document.getElementById("needs-review")?.scrollIntoView({ behavior: "smooth", block: "start" })}
              onJumpToEnroll={() => setEnrollOpen(true)}
              hasUnknowns={unknowns.length > 0}
              classifier={classifier}
              stats={statsById}
              onChanged={refresh}
              showToast={showToast}
              onRetrain={async () => {
                try {
                  const s = await api.retrainFaceClassifier();
                  setClassifier(s);
                  showToast(s.active ? `Smart match retrained · ${s.trained_people} people` : "Not enough enrolled angles yet for smart match", s.active ? "success" : "info");
                } catch (e) { showToast(`Retrain failed: ${String(e)}`, "error"); }
              }} />

            {/* ── 2. Review queue: faces the agent saw but couldn't identify ── */}
            <div id="needs-review" style={{ marginTop: 26 }}>
              <SectionHeader icon={<Sparkles size={13} />} title="Needs your review"
                subtitle="Tag the people the cameras saw but couldn't identify — every tag makes recognition smarter."
                count={reviewCount} />
              <ReviewSection unknowns={unknowns} clusters={clusters} persons={persons}
                onTagged={() => { refresh(); showToast("Tagged", "success"); }} showToast={showToast} />
            </div>

            {/* ── 3. Cross-camera body tracking (proposals + named tracks) ── */}
            <div style={{ marginTop: 26 }}>
              <SectionHeader icon={<Layers size={13} />} title="Tracked across cameras"
                subtitle="People followed by body appearance (Re-ID). Body matches only PROPOSE a name — a face or you confirms." />
              <TrackedSection tracked={tracked} backend={reidBackend} persons={persons}
                onChanged={refresh} showToast={showToast} />
            </div>
          </>
        )}
      </div>

      {removeTarget && (
        <RemovePersonDialog
          person={removeTarget}
          showToast={showToast}
          onCancel={() => setRemoveTarget(null)}
          onDone={() => { setRemoveTarget(null); refresh(); }} />
      )}

      {/* Enrollment modal (was a whole tab) — the wizard card is self-contained. */}
      {enrollOpen && (
        <div role="dialog" aria-modal="true"
          onClick={e => { if (e.target === e.currentTarget) setEnrollOpen(false); }}
          style={{
            position: "fixed", inset: 0, zIndex: 3000,
            background: "rgba(0,0,0,0.55)", backdropFilter: "blur(4px)",
            display: "flex", alignItems: "flex-start", justifyContent: "center",
            overflow: "auto", padding: "40px 16px",
          }}>
          <div style={{ position: "relative", width: "100%", maxWidth: 520 }}>
            <button type="button" onClick={() => setEnrollOpen(false)} title="Close"
              style={{ position: "absolute", top: -8, right: -8, zIndex: 1, width: 30, height: 30,
                borderRadius: 999, border: "1px solid var(--border-strong)", cursor: "pointer",
                background: "var(--bg-elevated)", color: "var(--text-primary)",
                display: "flex", alignItems: "center", justifyContent: "center" }}>
              <X size={15} />
            </button>
            <EnrollWizard onDone={() => { setEnrollOpen(false); refresh(); }} showToast={showToast} />
          </div>
        </div>
      )}

      {detailPerson && (
        <PersonDetail
          person={detailPerson}
          stats={statsById.get(detailPerson.id)}
          cameraName={cameraName}
          onClose={() => setDetailPerson(null)}
          onDeleted={() => { setDetailPerson(null); refresh(); }}
          // One label, one destination. This used to scroll to the review queue
          // when `unknowns.length > 0` and open the enrollment wizard otherwise
          // — same button, same position, two entirely different outcomes
          // decided by state the user cannot see. "Add more angles" means
          // capture more angles, so it opens the wizard, always.
          onAddAngles={() => { setDetailPerson(null); setEnrollOpen(true); }}
          showToast={showToast}
        />
      )}
    </div>
  );
}

/**
 * Face-model health strip — shown on every People tab. Quiet (a single muted
 * line) when the pipeline is ready; a prominent, actionable banner when the
 * model is missing or won't load. This is the difference between "the app is
 * broken and I can't tell" and "oh, I need to install the face model".
 */
function FaceModelStrip({ health, onChanged, showToast }: {
  health: FaceDebug | null;
  onChanged: () => void;
  showToast: (msg: string, type?: "success" | "error" | "info") => void;
}) {
  const [installing, setInstalling] = useState(false);
  const [pct, setPct] = useState(0);
  if (!health) return null;

  const installed = health.detector_installed && health.embedder_installed;
  const loaded = health.detector_loaded && health.embedder_loaded;
  const ready = installed && loaded;

  const install = async () => {
    const skill = findSkill("face_small");
    if (!skill) { showToast("Face model unavailable", "error"); return; }
    setInstalling(true); setPct(0);
    try {
      await downloadSkill(skill, p => setPct(Math.round(p)));
      showToast("Face model installed", "success");
      onChanged();
    } catch (e) {
      showToast(`Install failed: ${String(e).replace(/^.*Error:\s*/, "")}`, "error");
    } finally { setInstalling(false); }
  };

  if (ready) {
    const stale = health.enrolled_dim_mismatch > 0;
    // Recognition working is the expected steady state — no status line needed.
    // Only the actionable stale-embeddings warning earns pixels here.
    if (!stale) return null;
    return (
      <div style={{ margin: "0 2px 12px" }}>
        {/* Legacy embeddings made by a different model are silently un-matchable —
            tell the user to re-enroll instead of leaving them stuck as "unknown". */}
        {stale && (
          <div className="glass" style={{
            marginTop: 8, padding: "10px 14px", display: "flex", alignItems: "center", gap: 10,
            border: "1px solid rgba(255,176,32,0.34)", fontSize: 11.5, color: "var(--text-secondary)", lineHeight: 1.5,
          }}>
            <Sparkles size={13} style={{ color: "var(--status-warn)", flexShrink: 0 }} />
            <span>
              <strong style={{ color: "var(--text-primary)" }}>
                {health.enrolled_dim_mismatch} {health.enrolled_dim_mismatch === 1 ? "person needs" : "people need"} re-enrollment.
              </strong>{" "}
              Their saved face data was made with a different model and can't match — re-enroll them on the current model.
            </span>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="glass" style={{
      padding: "12px 16px", marginBottom: 14, display: "flex", alignItems: "center", gap: 12,
      border: "1px solid rgba(255,176,32,0.34)",
    }}>
      <Sparkles size={15} style={{ color: "var(--status-warn)", flexShrink: 0 }} />
      <div style={{ flex: 1, fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.5 }}>
        <strong style={{ color: "var(--text-primary)" }}>
          {!installed ? "Face model not installed" : "Face model failed to load"}
        </strong>
        {" — recognition, the live overlay, and event names stay off until this is fixed."}
        {health.note && <div style={{ fontSize: 10.5, color: "var(--text-tertiary)", marginTop: 3 }}>{health.note}</div>}
      </div>
      {!installed ? (
        <button onClick={install} disabled={installing} className="btn-primary" style={{ flexShrink: 0, padding: "7px 14px", fontSize: 12 }}>
          {installing ? `Installing… ${pct}%` : "Install"}
        </button>
      ) : (
        <button onClick={onChanged} className="btn-primary" style={{ flexShrink: 0, padding: "7px 14px", fontSize: 12 }}>Retry</button>
      )}
    </div>
  );
}

// ── Tracked (body Re-ID — cross-camera, appearance-based) ────────────────────


// The component names below used to end in "Tab" — leftovers from an era when
// this panel had six of them. It has had none since the consolidation to one
// scrolling column with three sections and a modal wizard, and the names were
// the last thing still claiming otherwise.
function RosterSection({ persons, recognitions, onDelete, onOpenDetail, onJumpToReview, onJumpToEnroll, hasUnknowns, classifier, stats, onRetrain, onChanged, showToast }: {
  persons: KnownPerson[];
  recognitions: Recognition[];
  onDelete: (id: string, name: string) => void;
  onOpenDetail: (p: KnownPerson) => void;
  onJumpToReview: () => void;
  onJumpToEnroll: () => void;
  hasUnknowns: boolean;
  classifier: FaceClassifierStatus | null;
  stats: Map<string, PersonStats>;
  onRetrain: () => void;
  onChanged: () => void;
  showToast: (msg: string, type?: "success" | "error" | "info") => void;
}) {
  const trainedIds = useMemo(() => new Set(classifier?.person_ids ?? []), [classifier]);
  // Stats join by PERSON ID, matching the rule in this module's docblock.
  //
  // This used to key on lowercase display name and then SUPPRESS the stat
  // entirely whenever two people shared a name — a wrong answer traded for no
  // answer, because `get_person_stats` grouped by name and could not tell them
  // apart. It groups by `person_id` now, so both Alexes get their own numbers.
  /** A recognition the user rejected from the confirm queue; RecognizedStrip
   *  renders the correction dialog for it. */
  const [correctingMarginal, setCorrectingMarginal] = useState<Recognition | null>(null);
  const [query, setQuery] = useState("");
  const [roleFilter, setRoleFilter] = useState("all");
  const filtered = useMemo(() => persons.filter(p => {
    if (roleFilter !== "all" && (p.role || "").toLowerCase() !== roleFilter) return false;
    if (query.trim() && !p.name.toLowerCase().includes(query.trim().toLowerCase())) return false;
    return true;
  }), [persons, query, roleFilter]);

  if (persons.length === 0) {
    return (
      <div className="glass" style={{
        padding: "40px 28px", textAlign: "center",
        display: "flex", flexDirection: "column", alignItems: "center", gap: 16,
      }}>
        <Users size={42} style={{ opacity: 0.35 }} />
        <div>
          <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 6 }}>No people enrolled yet</div>
          <div style={{ fontSize: 12, color: "var(--text-secondary)", maxWidth: 340, lineHeight: 1.55 }}>
            Once your cameras spot faces they'll appear in the <strong>Train</strong> tab — tap one to give it a name.
            Or enroll yourself manually from a live frame.
          </div>
        </div>
        <div style={{ display: "flex", gap: 10 }}>
          {hasUnknowns && (
            <button onClick={onJumpToReview} className="btn-primary" style={{ padding: "8px 16px" }}>
              <Sparkles size={13} /> Tag from recent events
            </button>
          )}
          <button onClick={onJumpToEnroll} style={{
            padding: "8px 16px", borderRadius: 999, fontSize: 12, fontWeight: 600,
            border: "1px solid var(--border-strong)", background: "transparent",
            color: "var(--text-primary)", cursor: "pointer",
            display: "inline-flex", alignItems: "center", gap: 6,
          }}>
            <UserPlus size={13} /> Enroll from camera
          </button>
        </div>
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <RecognitionModeStrip classifier={classifier} personCount={persons.length} onRetrain={onRetrain} />

      {/* Narrowly-decided matches first: they are the ones that go wrong. */}
      <ConfirmQueue
        recognitions={recognitions}
        persons={persons}
        onCorrect={setCorrectingMarginal}
        onConfirm={async (r) => {
          // Re-assert the SAME person. `correct_face` appends the shot to their
          // gallery, so a confirmation does not merely get recorded — it makes
          // the next match against this face less marginal.
          try {
            await api.correctFace(r.id, r.person_id);
            showToast(`Confirmed ${r.name} — added to their gallery`, "success");
            onChanged();
          } catch (e) { showToast(String(e), "error"); }
        }} />

      {recognitions.length > 0 && (
        <RecognizedStrip recognitions={recognitions} persons={persons} onOpenDetail={onOpenDetail}
          onChanged={onChanged} showToast={showToast} correcting={correctingMarginal}
          onCloseCorrecting={() => setCorrectingMarginal(null)} />
      )}

      {/* Filter bar — search by name + role chips */}
      <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
        <input value={query} onChange={e => setQuery(e.target.value)} placeholder="Search name…"
          style={{ flex: "1 1 160px", minWidth: 140, padding: "8px 12px", borderRadius: 999, fontSize: 12,
            border: "1px solid var(--border-strong)", background: "rgb(var(--ink) / 0.04)",
            color: "var(--text-primary)", outline: "none" }} />
        {/* "family" was missing, while roleColor has handled it all along — so
            a person with that role was reachable only through "all". */}
        {["all", "resident", "family", "employee", "visitor"].map(r => (
          <button key={r} type="button" onClick={() => setRoleFilter(r)}
            style={{ padding: "6px 12px", borderRadius: 999, fontSize: 11, fontWeight: 600, cursor: "pointer",
              border: `1px solid ${roleFilter === r ? "var(--accent)" : "var(--border)"}`,
              background: roleFilter === r ? "var(--hl)" : "transparent",
              color: roleFilter === r ? "var(--accent)" : "var(--text-secondary)", textTransform: "capitalize" }}>
            {r}
          </button>
        ))}
      </div>

      {/* Same grid primitive as the cluster and stranger lists, so a roster
          sitting above them lines up instead of using its own column maths. */}
      <CardGrid scroll={false} min={200}>
        {filtered.map(p => (
          <PersonCard key={p.id} person={p} onDelete={onDelete} onOpen={() => onOpenDetail(p)}
            trained={trainedIds.has(p.id)} stats={stats.get(p.id)} />
        ))}
      </CardGrid>
      {filtered.length === 0 && (
        <div style={{ fontSize: 12, color: "var(--text-tertiary)", textAlign: "center", padding: 18 }}>
          No people match.
        </div>
      )}
    </div>
  );
}

// ── Recognition mode (hybrid head: cosine cold-start vs trained classifier) ──
// Tells the user WHICH matcher is live. Cosine-NN works from the first angle but
// gets fuzzy as the roster grows; once ≥2 people have enough angles the trained
// "smart match" classifier takes over (better at separating look-alikes). This
// makes the upgrade legible so enrolling more angles feels like it does something.
function RecognitionModeStrip({ classifier, personCount, onRetrain }: {
  classifier: FaceClassifierStatus | null;
  personCount: number;
  onRetrain: () => void;
}) {
  if (personCount === 0) return null;
  const active = !!classifier?.active;
  return (
    <div className="glass" style={{
      padding: "9px 13px", display: "flex", alignItems: "center", gap: 10,
      fontSize: 11.5, color: "var(--text-secondary)",
    }}>
      <Sparkles size={13} style={{ color: active ? "var(--accent)" : "var(--text-muted)", flexShrink: 0 }} />
      <div style={{ flex: 1, lineHeight: 1.4 }}>
        {active ? (
          <>
            <strong style={{ color: "var(--text-primary)" }}>Smart match</strong>
            {" "}· trained on {classifier!.trained_people} {classifier!.trained_people === 1 ? "person" : "people"}
            {classifier!.has_reject_class ? " · learns strangers too" : ""}
          </>
        ) : (
          <>
            <strong style={{ color: "var(--text-primary)" }}>Cosine match</strong>
            {" "}· learning — enroll ≥3 angles for 2+ people to unlock smart match
          </>
        )}
      </div>
      <button type="button" onClick={onRetrain} title="Retrain the recognition model from the current roster"
        style={{ flexShrink: 0, display: "inline-flex", alignItems: "center", gap: 5, padding: "5px 11px",
          borderRadius: 999, fontSize: 11, fontWeight: 600, cursor: "pointer",
          border: "1px solid var(--border-strong)", background: "transparent", color: "var(--text-primary)" }}>
        <RefreshCw size={11} /> Retrain
      </button>
    </div>
  );
}

// ── Recently recognized (mature NVRs "Recent Recognitions") ─────────────────────

/**
 * How much better the winning face scored than the runner-up, below which a
 * recognition is worth a human glance.
 *
 * The matcher already refuses to name anyone whose margin is under 0.05 — it
 * demotes those to "unknown" rather than guessing (`face.rs`, the look-alike
 * ambiguity guard). So every recognition that reaches this UI cleared that bar.
 * What it does NOT distinguish is "cleared it decisively" from "cleared it by a
 * hair", and the second case is exactly where two family members get swapped.
 *
 * 0.10 is twice the matcher's own floor: comfortably above the reject line,
 * comfortably below a confident match.
 */
const CONFIRM_MARGIN = 0.10;

/**
 * Recognitions the matcher got right *narrowly* — the ones worth confirming.
 *
 * `match_margin` is the gap to the runner-up PERSON. It has been computed,
 * stored and shipped over IPC on every recognition, and the UI rendered only
 * `match_score` — which is the number that looks reassuring and says nothing
 * about whether it was a coin flip between two people who look alike.
 *
 * Confirming re-runs `correct_face` against the SAME person, which appends the
 * shot to their gallery: the answer is not just recorded, it makes the next
 * match less marginal.
 */
function ConfirmQueue({ recognitions, persons, onCorrect, onConfirm }: {
  recognitions: Recognition[];
  persons: KnownPerson[];
  onCorrect: (r: Recognition) => void;
  onConfirm: (r: Recognition) => void;
}) {
  const marginal = useMemo(() => recognitions.filter(
    r => r.match_margin != null && r.match_margin < CONFIRM_MARGIN && r.person_id), [recognitions]);
  if (marginal.length === 0) return null;

  return (
    <div className="glass-accent" style={{ padding: "12px 14px" }}>
      <div style={{
        display: "flex", alignItems: "center", gap: 7, marginBottom: 4,
        fontSize: 11, fontWeight: 700, letterSpacing: 0.04, textTransform: "uppercase",
        color: "var(--text-secondary)",
      }}>
        <Sparkles size={12} style={{ color: "var(--accent)" }} /> Worth confirming ({marginal.length})
      </div>
      <div style={{ fontSize: 11.5, color: "var(--text-secondary)", lineHeight: 1.5, marginBottom: 10 }}>
        These were named, but only just \u2014 someone else scored almost as high. Confirming
        one teaches the matcher; correcting it teaches it more.
      </div>
      <div style={{ display: "flex", gap: 10, overflowX: "auto", paddingBottom: 2 }}>
        {marginal.map(r => {
          const gap = Math.round((r.match_margin ?? 0) * 100);
          return (
            <div key={r.id} style={{ flexShrink: 0, width: 112 }}>
              <div style={{
                width: 112, height: 112, borderRadius: 12, overflow: "hidden",
                border: "1px solid var(--border-strong)", position: "relative", marginBottom: 6,
              }}>
                <img src={`data:image/jpeg;base64,${r.thumbnail_b64}`} alt={r.name}
                  style={{ width: "100%", height: "100%", objectFit: "cover" }} />
                <div title={whyNamed(r.match_method, r.match_score, r.match_margin)}
                  style={{
                    position: "absolute", bottom: 4, left: 4, right: 4,
                    padding: "2px 6px", borderRadius: 999, fontSize: 9, fontWeight: 700,
                    background: "rgba(0,0,0,0.66)", color: "var(--accent-amber)",
                    textAlign: "center",
                  }}>
                  +{gap} over runner-up
                </div>
              </div>
              <div style={{ fontSize: 11.5, fontWeight: 700, textAlign: "center", marginBottom: 5,
                whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{r.name}</div>
              <div style={{ display: "flex", gap: 4 }}>
                <button type="button" onClick={() => onConfirm(r)} className="btn-primary"
                  style={{ flex: 1, padding: "4px 0", fontSize: 10.5, borderRadius: 999,
                    display: "inline-flex", alignItems: "center", justifyContent: "center", gap: 3 }}>
                  <Check size={10} /> Yes
                </button>
                <button type="button" onClick={() => onCorrect(r)}
                  style={{ flex: 1, padding: "4px 0", fontSize: 10.5, borderRadius: 999, cursor: "pointer",
                    border: "1px solid var(--border-strong)", background: "transparent",
                    color: "var(--text-secondary)" }}>
                  No
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function RecognizedStrip({ recognitions, persons, onOpenDetail, onChanged, showToast,
                          correcting: external, onCloseCorrecting }: {
  recognitions: Recognition[];
  persons: KnownPerson[];
  onOpenDetail: (p: KnownPerson) => void;
  onChanged: () => void;
  showToast: (msg: string, type?: "success" | "error" | "info") => void;
  /** Correction requested from outside (the confirm queue). This component owns
   *  CorrectionModal, so driving it from here beats a second copy of the dialog. */
  correcting?: Recognition | null;
  onCloseCorrecting?: () => void;
}) {
  const [ownCorrecting, setOwnCorrecting] = useState<Recognition | null>(null);
  const correcting = external ?? ownCorrecting;
  const setCorrecting = (r: Recognition | null) => {
    if (r === null && external) onCloseCorrecting?.();
    setOwnCorrecting(r);
  };
  const correct = async (r: Recognition, correctId: string | null, label: string) => {
    try {
      await api.correctFace(r.id, correctId);
      showToast(correctId ? `Corrected to ${label} — the model learned from it` : `Marked "not ${r.name}" — won't repeat`, "success");
      setCorrecting(null);
      onChanged();
    } catch (e) { showToast(String(e), "error"); }
  };
  return (
    <div className="glass" style={{ padding: "12px 14px" }}>
      <div style={{
        display: "flex", alignItems: "center", gap: 7, marginBottom: 10,
        fontSize: 11, fontWeight: 700, letterSpacing: 0.04, textTransform: "uppercase",
        color: "var(--text-tertiary)",
      }}>
        <Activity size={12} style={{ color: "var(--accent)" }} /> Recently recognized
      </div>
      <div style={{ display: "flex", gap: 10, overflowX: "auto", paddingBottom: 2 }}>
        {recognitions.map(r => (
          <div key={r.id} style={{ flexShrink: 0, width: 92, textAlign: "center", position: "relative" }}>
            <button type="button"
              onClick={() => { const p = persons.find(p => p.id === r.person_id); if (p) onOpenDetail(p); }}
              title={`${r.name} · CAM ${r.cam_id + 1} · ${fmtWhen(r.seen_at)}\n${whyNamed(r.match_method, r.match_score, r.match_margin)}${r.event_id ? ` · event ${r.event_id.slice(0, 8)}` : ""}`}
              style={{ width: "100%", padding: 0, border: "none", cursor: "pointer", background: "transparent", textAlign: "center" }}>
              <div style={{
                width: 92, height: 92, borderRadius: 14, overflow: "hidden",
                border: "1px solid var(--border)", marginBottom: 6, position: "relative",
              }}>
                <img src={`data:image/jpeg;base64,${r.thumbnail_b64}`} alt={r.name}
                  style={{ width: "100%", height: "100%", objectFit: "cover" }} />
                <div style={{
                  position: "absolute", bottom: 4, right: 4,
                  padding: "1px 5px", borderRadius: 999, fontSize: 9, fontWeight: 700,
                  background: "rgba(0,0,0,0.62)", color: r.quality > 0.6 ? "var(--accent)" : "rgb(var(--ink) / 0.85)",
                }}>{Math.round(r.quality * 100)}</div>
                {/* Naming provenance: match score, hover for method + margin. */}
                {r.match_score != null && (
                  <div style={{
                    position: "absolute", bottom: 4, left: 4,
                    padding: "1px 5px", borderRadius: 999, fontSize: 9, fontWeight: 700,
                    background: "rgba(0,0,0,0.62)", color: "var(--accent)",
                  }}>{Math.round(r.match_score * 100)}%</div>
                )}
              </div>
              <div style={{ fontSize: 11, fontWeight: 700, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{r.name}</div>
              <div style={{ fontSize: 9, color: "var(--text-tertiary)" }}>CAM {r.cam_id + 1} · {fmtWhen(r.seen_at)}</div>
            </button>
            {/* Correction: this recognition is wrong → fix it (trains the model). */}
            <button type="button" onClick={() => setCorrecting(r)}
              title="Wrong person? Correct it"
              style={{ position: "absolute", top: 4, left: 4, padding: "1px 6px", borderRadius: 999,
                fontSize: 9, fontWeight: 700, cursor: "pointer", border: "none",
                background: "rgba(0,0,0,0.62)", color: "rgb(var(--ink) / 0.92)" }}>
              Wrong?
            </button>
          </div>
        ))}
      </div>
      {correcting && (
        <CorrectionModal recognition={correcting} persons={persons}
          onCorrect={correct} onClose={() => setCorrecting(null)} />
      )}
    </div>
  );
}

// Fix a WRONG recognition: pick the right person (trains the model + records a hard
// negative so it won't repeat), or mark "not them".
function CorrectionModal({ recognition, persons, onCorrect, onClose }: {
  recognition: Recognition;
  persons: KnownPerson[];
  onCorrect: (r: Recognition, correctId: string | null, label: string) => void;
  onClose: () => void;
}) {
  useDismiss(onClose);
  return createPortal(
    <div onClick={onClose} style={{
      position: "fixed", inset: 0, background: "rgba(0,0,0,0.6)", backdropFilter: "blur(4px)",
      display: "flex", alignItems: "center", justifyContent: "center", zIndex: 1000, padding: 20,
    }}>
      <div onClick={e => e.stopPropagation()} className="glass" style={{ width: 440, maxWidth: "100%", padding: 20 }}>
        <div style={{ display: "flex", gap: 12, alignItems: "center", marginBottom: 12 }}>
          <img src={`data:image/jpeg;base64,${recognition.thumbnail_b64}`} alt=""
            style={{ width: 56, height: 56, borderRadius: 12, objectFit: "cover", border: "1px solid var(--border)" }} />
          <div>
            <div style={{ fontWeight: 700, fontSize: 15 }}>Who is this really?</div>
            <div style={{ fontSize: 11.5, color: "var(--text-secondary)", lineHeight: 1.5 }}>
              Matched as <strong>{recognition.name}</strong>. Pick the correct person — the model learns the boundary
              so it stops repeating the mistake.
            </div>
          </div>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 6, maxHeight: 300, overflowY: "auto" }}>
          {persons.filter(p => p.id !== recognition.person_id).map(k => (
            <button key={k.id} type="button" onClick={() => onCorrect(recognition, k.id, k.name)}
              style={{ display: "flex", alignItems: "center", gap: 10, padding: "8px 10px", borderRadius: 10,
                cursor: "pointer", textAlign: "left",
                border: "1px solid var(--border)", background: "rgb(var(--ink) / 0.03)", color: "var(--text-primary)" }}>
              {k.thumbnail
                ? <img src={k.thumbnail.startsWith("data:") ? k.thumbnail : `data:image/jpeg;base64,${k.thumbnail}`}
                    alt={k.name} style={{ width: 30, height: 30, borderRadius: 8, objectFit: "cover" }} />
                : <span style={{ fontSize: 20 }}>👤</span>}
              <span style={{ fontWeight: 700, fontSize: 13 }}>{k.name}</span>
              <span style={{ fontSize: 10, color: "var(--text-tertiary)", textTransform: "capitalize" }}>{k.role}</span>
            </button>
          ))}
        </div>
        <button type="button" onClick={() => onCorrect(recognition, null, "")}
          style={{ marginTop: 10, width: "100%", padding: "8px 0", borderRadius: 10, fontSize: 12, fontWeight: 700,
            cursor: "pointer", border: "1px solid var(--accent-amber)", background: "transparent", color: "var(--accent-amber)" }}>
          None of these — not {recognition.name}
        </button>
        <button type="button" onClick={onClose}
          style={{ marginTop: 8, width: "100%", padding: "8px 0", borderRadius: 10, fontSize: 12, fontWeight: 600,
            cursor: "pointer", border: "1px solid var(--border-strong)", background: "transparent", color: "var(--text-secondary)" }}>
          Cancel
        </button>
      </div>
    </div>,
    document.body,
  );
}

function PersonCard({ person, onDelete, onOpen, trained, stats }: {
  person: KnownPerson;
  onDelete: (id: string, name: string) => void;
  onOpen: () => void;
  trained?: boolean;
  stats?: PersonStats;
}) {
  const roleColor: Record<string, string> = {
    resident: "var(--accent)",
    family:   "var(--accent)",
    employee: "var(--accent-amber)",
    visitor:  "var(--status-idle)",
  };
  // Count enrolled shots — more *diverse* angles = stronger model. Mature NVRs guidance:
  // 5–10 minimum, 20–30 good, diversity over volume. Bar hits "Strong" at ~10.
  // (Server-computed: listings no longer ship the ~190KB embeddings JSON.)
  const shotCount = person.embedding_count;
  const strengthPct = Math.min(100, Math.round((shotCount / 12) * 100));
  const lastSeen = person.last_seen_at ? new Date(person.last_seen_at) : null;
  const lastSeenLabel = lastSeen ? formatRelative(lastSeen) : "never seen";

  // The last hand-rolled shape in People. It was a fixed 168px avatar while every
  // other identity card in the app - vehicles, clusters, tracked bodies - is
  // aspect-ratio driven, so a roster next to a cluster list never lined up.
  return (
    <ProfileCard
      onClick={onOpen}
      title={`View ${person.name}'s gallery & history`}
      media={
        <ProfileMedia aspect="4 / 3" fallback={<Users size={26} />}
          src={person.thumbnail
            ? (person.thumbnail.startsWith("data:") ? person.thumbnail : `data:image/jpeg;base64,${person.thumbnail}`)
            : undefined}>
          {/* Passive marks only, per the card layout rule: state, not actions. */}
          <span style={{
            position: "absolute", bottom: 8, left: 8,
            padding: "3px 9px", borderRadius: 999, fontSize: 10, fontWeight: 600,
            background: "rgba(0,0,0,0.55)", backdropFilter: "blur(10px)", color: "#fff",
          }}>{lastSeenLabel}</span>
          {trained && (
            <span title="Covered by smart match (trained classifier)" style={{
              position: "absolute", top: 8, right: 8,
              padding: "3px 8px", borderRadius: 999, fontSize: 9.5, fontWeight: 700,
              background: "var(--accent-glow)", color: "var(--accent)",
              border: "1px solid var(--accent)", backdropFilter: "blur(10px)",
              display: "inline-flex", alignItems: "center", gap: 4,
            }}>
              <Check size={10} /> Trained
            </span>
          )}
        </ProfileMedia>
      }>
        <div style={{
          fontWeight: 700, fontSize: 14, letterSpacing: -0.01,
          whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
          marginBottom: 4,
        }}>{person.name}</div>
        <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap", marginBottom: 10 }}>
          <span style={{
            fontSize: 10, fontWeight: 700, letterSpacing: 0.04, textTransform: "uppercase",
            color: roleColor[person.role] ?? "var(--text-tertiary)",
          }}>
            {person.role}
          </span>
          {shotCount === 0 && (
            <span title="Body-only identity — recognised by appearance. Link a face to make it durable."
              style={{ fontSize: 9, fontWeight: 700, padding: "2px 7px", borderRadius: 999,
                background: "color-mix(in srgb, var(--status-idle) 14%, transparent)", color: "var(--status-idle)", border: "1px solid var(--status-idle)",
                display: "inline-flex", alignItems: "center", gap: 3 }}>
              <Layers size={9} /> no face yet
            </span>
          )}
        </div>

        {/* Activity pattern — the self-learned routine (sightings / days / usual hour) */}
        {stats && stats.sightings_30d > 0 && (
          <div title="This person's activity pattern over the last 30 days"
            style={{ display: "flex", alignItems: "center", gap: 5, flexWrap: "wrap",
              fontSize: 10.5, color: "var(--text-secondary)", marginBottom: 10 }}>
            <Activity size={10} style={{ color: "var(--accent)", flexShrink: 0 }} />
            <span>{stats.sightings_30d}× / {stats.days_active} day{stats.days_active === 1 ? "" : "s"}</span>
            {stats.peak_hour != null && (
              <span style={{ color: "var(--text-tertiary)" }}>· usually ~{String(stats.peak_hour).padStart(2, "0")}:00</span>
            )}
            {stats.cameras.length > 0 && (
              <span style={{ color: "var(--text-tertiary)" }}>· cam {stats.cameras.map(c => c + 1).join(", ")}</span>
            )}
          </div>
        )}

        {/* Embedding strength bar */}
        <div style={{ marginBottom: 12 }} title={`${shotCount} enrolled angles — add more for higher accuracy`}>
          <div style={{
            display: "flex", justifyContent: "space-between", alignItems: "center",
            fontSize: 10, color: "var(--text-tertiary)", marginBottom: 4,
          }}>
            <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
              <Layers size={9} /> {shotCount} angle{shotCount === 1 ? "" : "s"}
            </span>
            <span style={{ color: strengthPct >= 80 ? "var(--accent)" : "var(--text-tertiary)" }}>
              {strengthPct >= 80 ? "Strong" : strengthPct >= 40 ? "OK" : "Weak"}
            </span>
          </div>
          <div style={{
            height: 4, borderRadius: 999, overflow: "hidden",
            background: "rgb(var(--ink) / 0.06)",
          }}>
            <div style={{
              height: "100%", width: `${strengthPct}%`,
              background: strengthPct >= 80 ? "var(--accent)"
                : strengthPct >= 40 ? "var(--accent-amber)" : "var(--accent-red)",
              borderRadius: 999,
              transition: "width 380ms cubic-bezier(0.16,1,0.3,1)",
            }} />
          </div>
        </div>

        {/* A full-width red "Remove" used to sit here on EVERY card, with the
            same visual weight as opening the person — one click from browsing
            to permanently destroying every face descriptor, crop, appearance
            vector and sighting for that person. It still exists inside
            PersonDetail, which is where a destructive action belongs: behind
            the deliberate act of opening the record you want to delete. */}
    </ProfileCard>
  );
}

// ── Person detail (gallery + history + stats) ───────────────────────────────
