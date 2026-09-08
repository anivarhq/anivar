/**
 * What the archive already knows about a person, beyond "here are their photos".
 *
 * Both of these read data the backend has always produced and the UI has never
 * shown:
 *
 *   - `get_person_stats` builds a 24-bucket local-hour histogram and, until now,
 *     serialised only its argmax. "Home every evening" and "here exactly once, at
 *     18:00" reached the UI as the same single number.
 *   - `get_camera_correlations` returns the full ordered camera trail per person.
 *     It has been wired through `api/index.ts` and called by nothing at all.
 *
 * Neither is a new capability. Both are presentation over answers already
 * computed on every call.
 */
import { useEffect, useState } from "react";
import { Clock, MapPin, ArrowRight } from "lucide-react";
import { api, PersonStats } from "../../api";
import { fmtWhen } from "../../lib/time";

/* ── When ──────────────────────────────────────────────────────────────────── */

/**
 * Hour-of-day histogram.
 *
 * 24 bars, local hours, normalised to the busiest. Deliberately unlabelled except
 * at 00 / 06 / 12 / 18 — the shape is the information ("evenings", "overnight",
 * "school run"), not the exact counts, and a full axis at this size is noise.
 */
export function HourHistogram({ hours, peak }: { hours: number[]; peak: number | null }) {
  const max = Math.max(1, ...hours);
  const total = hours.reduce((a, b) => a + b, 0);
  if (total === 0) return null;

  return (
    <div>
      <div style={{ display: "flex", alignItems: "flex-end", gap: 2, height: 44 }}>
        {hours.map((n, h) => (
          <div key={h} title={`${String(h).padStart(2, "0")}:00 — ${n} sighting${n === 1 ? "" : "s"}`}
            style={{
              flex: 1,
              height: `${Math.max(n > 0 ? 8 : 2, (n / max) * 100)}%`,
              borderRadius: 2,
              // The peak hour is the one fact the card already showed; keeping it
              // accented ties the two views together.
              background: h === peak ? "var(--accent)" : "rgb(var(--ink) / 0.18)",
              transition: "background 140ms var(--ease-out, ease)",
            }} />
        ))}
      </div>
      <div style={{
        display: "flex", justifyContent: "space-between", marginTop: 4,
        fontSize: 9, fontFamily: "var(--font-mono)", color: "var(--text-tertiary)",
      }}>
        <span>00</span><span>06</span><span>12</span><span>18</span><span>23</span>
      </div>
    </div>
  );
}

/* ── Where ─────────────────────────────────────────────────────────────────── */

type Hop = { camera_id: number; seen_at: string; confidence: number; event_id: string | null };

/**
 * The movement trail: which cameras saw this person, in what order.
 *
 * Consecutive sightings on the SAME camera collapse into one hop. Without that
 * a person standing in the driveway for two minutes produces forty identical
 * entries and the actual path — the thing worth seeing — is buried.
 */
export function MovementTrail({ personName, cameraName, onOpenEvent }: {
  personName: string;
  cameraName: (id: number) => string;
  onOpenEvent?: (eventId: string) => void;
}) {
  const [hops, setHops] = useState<Hop[] | null>(null);
  const [hours, setHours] = useState(24);

  useEffect(() => {
    let live = true;
    api.getCameraCorrelations(hours)
      .then(rows => {
        if (!live) return;
        const mine = rows.find(r => r.person_name === personName);
        const raw = (mine?.sightings ?? []).slice().reverse(); // oldest first: a path
        const collapsed: Hop[] = [];
        for (const s of raw) {
          const prev = collapsed[collapsed.length - 1];
          if (prev && prev.camera_id === s.camera_id) continue;
          collapsed.push(s);
        }
        setHops(collapsed);
      })
      .catch(() => { if (live) setHops([]); });
    return () => { live = false; };
  }, [personName, hours]);

  if (hops === null) {
    return <div style={{ fontSize: 11, color: "var(--text-tertiary)" }}>Loading movement…</div>;
  }

  return (
    <div>
      <div style={{ display: "flex", gap: 4, marginBottom: 8 }}>
        {[24, 72, 168].map(h => (
          <button key={h} type="button" onClick={() => { setHops(null); setHours(h); }}
            style={{
              padding: "2px 9px", borderRadius: 999, fontSize: 10, fontWeight: 700,
              cursor: "pointer",
              border: `1px solid ${h === hours ? "var(--accent)" : "var(--border)"}`,
              background: h === hours ? "color-mix(in srgb, var(--accent) 12%, transparent)" : "transparent",
              color: h === hours ? "var(--accent)" : "var(--text-secondary)",
            }}>
            {h === 24 ? "24h" : h === 72 ? "3d" : "7d"}
          </button>
        ))}
      </div>

      {hops.length === 0 ? (
        <div style={{ fontSize: 11, color: "var(--text-tertiary)" }}>
          No camera sightings in this window.
        </div>
      ) : (
        <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 6 }}>
          {hops.map((h, i) => (
            <span key={`${h.camera_id}-${h.seen_at}`} style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
              <button
                type="button"
                disabled={!h.event_id || !onOpenEvent}
                onClick={() => h.event_id && onOpenEvent?.(h.event_id)}
                title={h.event_id ? "Open this event" : "No event recorded for this sighting"}
                style={{
                  display: "inline-flex", alignItems: "center", gap: 5,
                  padding: "4px 9px", borderRadius: 999,
                  border: "1px solid var(--border-strong)",
                  background: "rgb(var(--ink) / 0.04)",
                  color: "var(--text-primary)", fontSize: 11, fontWeight: 600,
                  cursor: h.event_id && onOpenEvent ? "pointer" : "default",
                }}>
                <MapPin size={10} style={{ color: "var(--accent)" }} />
                {cameraName(h.camera_id)}
                <span style={{ fontFamily: "var(--font-mono)", fontSize: 10, color: "var(--text-secondary)" }}>
                  {fmtWhen(h.seen_at)}
                </span>
              </button>
              {i < hops.length - 1 && <ArrowRight size={11} style={{ color: "var(--text-tertiary)" }} />}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

/* ── When, as prose + chart ────────────────────────────────────────────────── */

/** The activity block: the histogram plus the numbers that frame it. */
export function ActivityPattern({ stats }: { stats: PersonStats | undefined }) {
  if (!stats || stats.sightings_30d === 0) {
    return <div style={{ fontSize: 11, color: "var(--text-tertiary)" }}>
      No sightings in the last 30 days.
    </div>;
  }
  const peakLabel = stats.peak_hour === null
    ? null
    : `${String(stats.peak_hour).padStart(2, "0")}:00–${String((stats.peak_hour + 1) % 24).padStart(2, "0")}:00`;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <div style={{ fontSize: 11.5, color: "var(--text-secondary)", lineHeight: 1.5 }}>
        <strong style={{ color: "var(--text-primary)" }}>{stats.sightings_30d}</strong> sighting
        {stats.sightings_30d === 1 ? "" : "s"} across{" "}
        <strong style={{ color: "var(--text-primary)" }}>{stats.days_active}</strong> day
        {stats.days_active === 1 ? "" : "s"}
        {peakLabel && <> · usually <Clock size={10} style={{ display: "inline", verticalAlign: -1 }} /> {peakLabel}</>}
      </div>
      <HourHistogram hours={stats.hours ?? []} peak={stats.peak_hour} />
    </div>
  );
}
