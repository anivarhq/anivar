/**
 * PersonsPanel — the ONE identity home.
 *
 * Three places, one question each:
 *   • Today  — who was here, as visits (one continuous stay across cameras).
 *   • People — the people you've named.
 *   • Review — "Who is this?": people the cameras keep seeing but can't name.
 * The header search box finds anyone by description or appearance. Enrollment
 * is a modal launched from the header.
 *
 * What is deliberately NOT here, because real systems don't ask users to do it:
 * confirming body-appearance matches, confirming narrow face recognitions, or
 * reading engine internals (match scores, quality numbers, which matcher is
 * live). Appearance matching works silently — linking visits and "Find similar";
 * a wrong name is fixed from that person's own gallery ("Not {name}").
 *
 * All naming and correction binds by PERSON ID, never display name (renames and
 * duplicate names mis-resolve by name).
 */

import { useEffect, useRef, useState, useCallback, useMemo, type ReactNode } from "react";
import { UserPlus, RefreshCw, RotateCcw, Users, Sparkles, X, Layers, Clock, Activity, Search, Calendar } from "lucide-react";
import { api, KnownPerson, UnknownCluster, FaceDebug, PersonStats } from "../../api";
import { useStore } from "../../store";
import { useShallow } from "zustand/react/shallow";
import { downloadSkill, findSkill } from "../agent/skillDownload";
import { usePanelCache } from "../../lib/panelCache";
import { localDateStr } from "../../lib/time";
import { formatRelative, fmtLocalDay } from "./shared";
import { CardGrid, ProfileCard, ProfileMedia, CardEmpty, CARD_MIN } from "../review/Card";
import { FilterDropdown } from "../review/FilterDropdown";
import { GlassCalendar } from "../../components/ui/GlassCalendar";
import { useRecordedDays } from "../../lib/useRecordedDays";
import styles from "../review/ReviewFeed.module.css";
import { EnrollWizard } from "./EnrollWizard";
import { ReviewSection } from "./ReviewSection";
import { PersonDetail } from "./PersonDetail";
import { TodayView } from "./TodayView";
import { PeopleSearch } from "./PeopleSearch";

type PeopleTab = "today" | "people" | "review";
const PEOPLE_TABS: { id: PeopleTab; label: string; icon: ReactNode }[] = [
  { id: "today", label: "Today", icon: <Clock size={13} /> },
  { id: "people", label: "People", icon: <Users size={13} /> },
  { id: "review", label: "Review", icon: <Sparkles size={13} /> },
];

// "family" was missing from the old chip row while roleColor handled it all
// along, so a person with that role was reachable only through "all".
const ROLES = ["resident", "family", "employee", "visitor"];

export function PersonsPanel() {
  // Enrollment is a MODAL launched from the People header (it was a whole tab).
  const [enrollOpen, setEnrollOpen] = useState(false);
  // Set when "Add more angles" opened the wizard for an existing person.
  const [enrollTarget, setEnrollTarget] = useState<KnownPerson | null>(null);
  const [tab, setTab] = useState<PeopleTab>(() => {
    try {
      const t = localStorage.getItem("sc.peopleTab");
      if (t && PEOPLE_TABS.some(x => x.id === t)) return t as PeopleTab;
    } catch { /* storage blocked */ }
    return "today";   // also where a stored "search" lands — that tab is gone
  });
  // Track id for "looks like this person".
  const [similarTo, setSimilarTo] = useState<string | null>(null);
  const pickTab = (t: PeopleTab) => {
    setTab(t);
    setSimilarTo(null);
    try { localStorage.setItem("sc.peopleTab", t); } catch { /* storage blocked */ }
  };

  /** ONE search box, in the header, on every tab — Review's model exactly.
   *
   *  It means two different things depending on the tab, as Review's does: a
   *  whole-archive PEOPLE search on Today and Review, and a live client-side
   *  filter of the roster on People (what Review's box does on Vehicles and
   *  Sounds). */
  const [query, setQuery] = useState("");
  const [debounced, setDebounced] = useState("");
  useEffect(() => {
    const t = setTimeout(() => setDebounced(query.trim()), 250);
    return () => clearTimeout(t);
  }, [query]);
  const rosterMode = tab === "people";
  /** Results take over the body: a typed description, or "looks like this one". */
  const searchMode = !!similarTo || (!rosterMode && debounced.length > 0);
  /** "Looks like this person" replaces any typed query — they are two different
   *  questions, and leaving the words in the box would cancel the match on the
   *  next debounce tick. `debounced` is set here too, so there is no window
   *  where the stale query is still live. */
  const findSimilar = (trackId: string) => {
    setQuery(""); setDebounced(""); setCalOpen(false); setSimilarTo(trackId);
  };

  // Today's day — the header date button owns it, so TodayView has no day-nav.
  const [day, setDay] = useState(() => localDateStr());
  const [calOpen, setCalOpen] = useState(false);
  const calAnchor = useRef<HTMLButtonElement>(null);
  const recordedDays = useRecordedDays();

  // Toolbar filters. Empty set = no filter, matching Review's dropdowns.
  const [camFilter, setCamFilter] = useState<Set<string>>(new Set());
  const [roleFilter, setRoleFilter] = useState<Set<string>>(new Set());
  const toggleIn = (set: (fn: (p: Set<string>) => Set<string>) => void) => (v: string) =>
    set(prev => { const n = new Set(prev); n.has(v) ? n.delete(v) : n.add(v); return n; });

  // Stale-while-revalidate: the lists survive tab switches, so a revisit paints
  // instantly while refresh() revalidates in the background.
  const [persons, setPersons] = usePanelCache<KnownPerson[]>("people.persons", []);
  const [clusters, setClusters] = usePanelCache<UnknownCluster[]>("people.clusters", []);
  const [personStats, setPersonStats] = usePanelCache<PersonStats[]>("people.personStats", []);
  const [detailPerson, setDetailPerson] = useState<KnownPerson | null>(null);
  const [loading, setLoading] = useState(false);
  /** Has the fan-out completed at least once this session? Without it a cold
   *  start paints "nobody named yet" before the first fetch resolves. */
  const [everLoaded, setEverLoaded] = useState(false);
  // A backend failure must look DIFFERENT from "no people yet": capture it and
  // show a banner with Retry instead of a blank tab.
  const [loadError, setLoadError] = useState<string | null>(null);
  const [health, setHealth] = useState<FaceDebug | null>(null);
  const { showToast } = useStore(useShallow(s => ({ showToast: s.showToast })));

  const refresh = useCallback(async () => {
    setLoading(true);
    const errs: string[] = [];
    // Each loader resolves independently — one failure shouldn't blank the rest.
    const guard = async <T,>(p: Promise<T>, fallback: T, label: string): Promise<T> => {
      try { return await p; }
      catch (e) { errs.push(`${label}: ${String(e).replace(/^.*Error:\s*/, "")}`); return fallback; }
    };
    try {
      const [p, cl, ps] = await Promise.all([
        guard(api.listKnownPersons(), [] as KnownPerson[], "People"),
        guard(api.listUnknownClusters(30, 0.10), [] as UnknownCluster[], "People to name"),
        guard(api.getPersonStats(), [] as PersonStats[], "Activity patterns"),
      ]);
      setPersons(p);
      setClusters(cl);
      setPersonStats(ps);
      setLoadError(errs.length ? errs.join(" · ") : null);
      // Face-model health — the strip that says "install the face model".
      setHealth(await api.faceDebug().catch(() => null));
    } finally {
      setEverLoaded(true);
      setLoading(false);
    }
  }, []);
  useEffect(() => { refresh(); }, [refresh]);
  // Returning after the window was hidden → refresh, but throttled.
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

  /** Every configured camera — names for visits, including a camera that has
   *  since been disabled but still has history. */
  const [camConfigs, setCamConfigs] = useState<Array<{ cam_id: number; name: string; enabled: boolean }>>([]);
  useEffect(() => { api.getCameraConfigs().then(setCamConfigs).catch(() => {}); }, []);
  const cameraName = useCallback(
    (id: number) => camConfigs.find(c => c.cam_id === id)?.name || `Camera ${id + 1}`,
    [camConfigs]);
  /** The Cameras filter lists real, enabled cameras only. `get_camera_configs`
   *  always returns 16 slots, empty placeholders included — the same rule Review
   *  applies, which People was missing. */
  const cams = useMemo(
    () => camConfigs.filter(c => c.enabled).map(c => String(c.cam_id)),
    [camConfigs]);

  /** Stats keyed by PERSON ID, shared by the roster cards and the detail view. */
  const statsById = useMemo(() => {
    const m = new Map<string, PersonStats>();
    for (const st of personStats) if (st.person_id) m.set(st.person_id, st);
    return m;
  }, [personStats]);

  /** Decisions waiting, not rows: one per person to name. */
  const reviewCount = clusters.length;

  /** Result count for the header status line; null while a search is in flight. */
  const [searchCount, setSearchCount] = useState<number | null>(null);
  const hasFilters = searchMode || tab === "today" || tab === "people";

  return (
    <div className={styles.root}>
      {/* ── Top toolbar: tabs · search · date — the same one row as Review ── */}
      <div className={styles.header}>
        <div className={`lg ${styles.segTabs}`}>
          {PEOPLE_TABS.map(t => (
            <button key={t.id} className={`${styles.segTab} ${tab === t.id ? styles.segTabActive : ""}`}
              onClick={() => pickTab(t.id)}>
              {t.icon} {t.label}
              {t.id === "review" && reviewCount > 0 && <span className={styles.count}>{reviewCount}</span>}
            </button>
          ))}
        </div>

        <div className={`lg ${styles.searchRow}`}>
          <Search size={14} className={styles.searchIcon} />
          <input
            className={styles.searchInput}
            placeholder={rosterMode
              ? "Filter people — name…"
              : "Describe someone — blue top with a backpack · unfamiliar, no hat"}
            value={query}
            // Searching hides the date button, so a calendar left open would
            // reappear by itself when the query is cleared.
            onChange={e => { setQuery(e.target.value); setCalOpen(false); }}
            onKeyDown={e => { if (e.key === "Escape") setQuery(""); }}
            spellCheck={false}
          />
          {searchMode && !similarTo && (
            <span className={styles.searchStatus}>
              {searchCount == null ? "searching…"
                : `${searchCount} result${searchCount === 1 ? "" : "s"} · all dates`}
            </span>
          )}
          {query && (
            <button className={styles.searchClear} onClick={() => setQuery("")}
              title="Clear search" aria-label="Clear search">
              <X size={14} />
            </button>
          )}
        </div>

        <div className={styles.headerRight}>
          {/* The day lives here, not on a chevron pair inside Today. */}
          {tab === "today" && !searchMode && (<>
            <button ref={calAnchor} className={`lg ${styles.glassBtn}`} onClick={() => setCalOpen(o => !o)}>
              <Calendar size={12} /> {fmtLocalDay(day)}
            </button>
            <GlassCalendar value={day} onChange={d => { setDay(d); setCalOpen(false); }}
              max={localDateStr()} open={calOpen} onClose={() => setCalOpen(false)} anchorRef={calAnchor}
              recordedDays={recordedDays} />
          </>)}
          <button className={`lg ${styles.glassBtn}`}
            onClick={() => { setEnrollTarget(null); setEnrollOpen(true); }}>
            <UserPlus size={12} /> Enroll
          </button>
          <button className={`lg ${styles.iconBtn}`} onClick={refresh} disabled={loading} title="Refresh">
            <RotateCcw size={13} className={loading ? styles.spin : ""} />
          </button>
        </div>
      </div>

      {/* ── Filters — Review's dropdown row, only where there is something to filter ── */}
      {hasFilters && (
        <div className={styles.filters}>
          {(searchMode || tab === "today") && (
            <FilterDropdown label="Cameras" options={cams} selected={camFilter}
              onToggle={toggleIn(setCamFilter)}
              onSelectAll={() => setCamFilter(new Set(cams))}
              onClear={() => setCamFilter(new Set())}
              format={v => cameraName(Number(v))} emptyText="No cameras" />
          )}
          {!searchMode && tab === "people" && (
            <FilterDropdown label="Role" options={ROLES} selected={roleFilter}
              onToggle={toggleIn(setRoleFilter)}
              onSelectAll={() => setRoleFilter(new Set(ROLES))}
              onClear={() => setRoleFilter(new Set())} />
          )}
        </div>
      )}

      {/* Pinned strips — these scrolled away with the content before. */}
      {(((tab === "people" || tab === "review") && !searchMode) || loadError) && (
        <div style={{ flexShrink: 0, padding: "0 16px" }}>
          {/* Face-model health, so "recognition is off" is never mistaken for
              "no people". Stays quiet when the pipeline is healthy. */}
          {(tab === "people" || tab === "review") && !searchMode && (
            <FaceModelStrip health={health} onChanged={refresh} showToast={showToast} />
          )}
          {loadError && (
            <div className="glass" style={{
              padding: "11px 15px", marginBottom: 12, display: "flex", alignItems: "center", gap: 12,
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
        </div>
      )}

      {!everLoaded && persons.length === 0 ? (
        <CardEmpty icon={<RefreshCw size={30} className="spin" />}>Loading people…</CardEmpty>
      ) : searchMode ? (
        <PeopleSearch cameraName={cameraName} persons={persons}
          query={debounced} cams={[...camFilter].map(Number)} similarTo={similarTo}
          onClearSimilar={() => setSimilarTo(null)} onFindSimilar={findSimilar}
          onCount={setSearchCount} onChanged={refresh} showToast={showToast} />
      ) : tab === "today" ? (
        <TodayView day={day} cams={camFilter} cameraName={cameraName} persons={persons}
          toName={reviewCount} onReview={() => pickTab("review")}
          onChanged={refresh} onFindSimilar={findSimilar} showToast={showToast} />
      ) : tab === "people" ? (
        <RosterSection persons={persons} stats={statsById}
          query={debounced} roleFilter={roleFilter}
          hasPeopleToName={reviewCount > 0}
          onOpenDetail={setDetailPerson}
          onJumpToReview={() => pickTab("review")}
          onJumpToEnroll={() => setEnrollOpen(true)} />
      ) : (
        <ReviewSection clusters={clusters} persons={persons} onChanged={refresh} showToast={showToast} />
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
            <EnrollWizard person={enrollTarget ?? undefined}
              onDone={() => { setEnrollOpen(false); setEnrollTarget(null); refresh(); }} showToast={showToast} />
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
          // "Add more angles" means capture more angles, so it opens the wizard, always.
          onAddAngles={() => { setEnrollTarget(detailPerson); setDetailPerson(null); setEnrollOpen(true); }}
          showToast={showToast}
        />
      )}
    </div>
  );
}

/**
 * Face-model health strip — shown on People and Review. Silent when the pipeline
 * is ready; a prominent, actionable banner when the model is missing or won't
 * load. The difference between "the app is broken and I can't tell" and "oh, I
 * need to install the face model".
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
    // Working is the expected steady state — only a stale roster earns pixels:
    // face data made by a different model can't match, silently.
    if (health.enrolled_dim_mismatch <= 0) return null;
    return (
      <div className="glass" style={{
        marginBottom: 12, padding: "10px 14px", display: "flex", alignItems: "center", gap: 10,
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

/** The people you've named — one card each, and nothing else. */
function RosterSection({ persons, stats, query, roleFilter, hasPeopleToName, onOpenDetail, onJumpToReview, onJumpToEnroll }: {
  persons: KnownPerson[];
  stats: Map<string, PersonStats>;
  /** Name filter from the header search box (debounced, already trimmed). */
  query: string;
  /** Roles from the toolbar dropdown; empty = every role. */
  roleFilter: Set<string>;
  /** Review has people waiting for a name. */
  hasPeopleToName: boolean;
  onOpenDetail: (p: KnownPerson) => void;
  onJumpToReview: () => void;
  onJumpToEnroll: () => void;
}) {
  const filtered = useMemo(() => {
    const q = query.toLowerCase();
    return persons.filter(p =>
      (roleFilter.size === 0 || roleFilter.has((p.role || "").toLowerCase()))
      && (!q || p.name.toLowerCase().includes(q)));
  }, [persons, query, roleFilter]);

  if (persons.length === 0) {
    return (
      <CardEmpty icon={<Users size={32} />}>
        Nobody named yet. Name the people your cameras keep seeing, or enroll someone from a live camera.
        <span style={{ display: "flex", gap: 8, justifyContent: "center", marginTop: 12 }}>
          {hasPeopleToName && (
            <button className={`lg ${styles.glassBtn}`} onClick={onJumpToReview}>
              <Sparkles size={12} /> Name people
            </button>
          )}
          <button className={`lg ${styles.glassBtn}`} onClick={onJumpToEnroll}>
            <UserPlus size={12} /> Enroll from camera
          </button>
        </span>
      </CardEmpty>
    );
  }

  return (
    <CardGrid min={CARD_MIN}>
      {filtered.length === 0
        ? <CardEmpty icon={<Users size={32} />}>Nobody matches those filters.</CardEmpty>
        : filtered.map(p => (
          <PersonCard key={p.id} person={p} onOpen={() => onOpenDetail(p)} stats={stats.get(p.id)} />
        ))}
    </CardGrid>
  );
}

function PersonCard({ person, onOpen, stats }: {
  person: KnownPerson;
  onOpen: () => void;
  stats?: PersonStats;
}) {
  const roleColor: Record<string, string> = {
    resident: "var(--accent)",
    family:   "var(--accent)",
    employee: "var(--accent-amber)",
    visitor:  "var(--status-idle)",
  };
  // Enrolled angles — more *diverse* angles = stronger recognition (Frigate's
  // guidance: 20–30 varied images per person). Bar reads "Strong" at ~10.
  const shotCount = person.embedding_count;
  const strengthPct = Math.min(100, Math.round((shotCount / 12) * 100));
  const lastSeen = person.last_seen_at ? new Date(person.last_seen_at) : null;
  const lastSeenLabel = lastSeen ? formatRelative(lastSeen) : "never seen";

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
            <span title="Recognised by appearance only. Add a face to make it reliable."
              style={{ fontSize: 9, fontWeight: 700, padding: "2px 7px", borderRadius: 999,
                background: "color-mix(in srgb, var(--status-idle) 14%, transparent)", color: "var(--status-idle)", border: "1px solid var(--status-idle)",
                display: "inline-flex", alignItems: "center", gap: 3 }}>
              <Layers size={9} /> no face yet
            </span>
          )}
        </div>

        {/* Routine over the last 30 days: how often, and usually when. */}
        {stats && stats.sightings_30d > 0 && (
          <div title="This person's activity over the last 30 days"
            style={{ display: "flex", alignItems: "center", gap: 5, flexWrap: "wrap",
              fontSize: 10.5, color: "var(--text-secondary)", marginBottom: 10 }}>
            <Activity size={10} style={{ color: "var(--accent)", flexShrink: 0 }} />
            <span>Seen on {stats.days_active} day{stats.days_active === 1 ? "" : "s"}</span>
            {stats.peak_hour != null && (
              <span style={{ color: "var(--text-tertiary)" }}>· usually ~{String(stats.peak_hour).padStart(2, "0")}:00</span>
            )}
          </div>
        )}

        {/* Recognition strength — actionable: "Weak" means add more angles. */}
        <div style={{ marginBottom: 4 }} title={`${shotCount} enrolled angles — add more for higher accuracy`}>
          <div style={{
            display: "flex", justifyContent: "space-between", alignItems: "center",
            fontSize: 10, color: "var(--text-tertiary)", marginBottom: 4,
          }}>
            <span>Recognition</span>
            <span style={{ color: strengthPct >= 80 ? "var(--accent)" : "var(--text-tertiary)" }}>
              {strengthPct >= 80 ? "Strong" : strengthPct >= 40 ? "OK" : "Weak — add angles"}
            </span>
          </div>
          <div style={{ height: 4, borderRadius: 999, overflow: "hidden", background: "rgb(var(--ink) / 0.06)" }}>
            <div style={{
              height: "100%", width: `${strengthPct}%`,
              background: strengthPct >= 80 ? "var(--accent)"
                : strengthPct >= 40 ? "var(--accent-amber)" : "var(--accent-red)",
              borderRadius: 999,
              transition: "width 380ms cubic-bezier(0.16,1,0.3,1)",
            }} />
          </div>
        </div>
    </ProfileCard>
  );
}
