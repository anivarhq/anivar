/**
 * When a person tends to be here, beyond "here are their photos".
 *
 * `get_person_stats` builds a 24-bucket local-hour histogram: "home every
 * evening" and "here exactly once, at 18:00" used to reach the UI as the same
 * single number. (Where they move lives in visits — PersonDetail / TodayView.)
 */
import { Clock } from "lucide-react";
import type { PersonStats } from "../../api";

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
