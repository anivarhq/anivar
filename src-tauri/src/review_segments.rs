//! Server-side Review Items (`ReviewSegment` parity).
//!
//! A *review item* is a per-camera time window that bundles overlapping
//! `motion_events` into ONE reviewable unit with a single severity
//! (alert/detection) and a DB-backed reviewed flag. Windows do not overlap
//! WITHIN A KIND — video items never overlap other video items — but a sound and
//! a video item may cover the same seconds, because they are different things
//! that genuinely happened at once. (Before that split, a five-minute "Speech"
//! event spanning two person events welded all three into a single card.) This
//! is the canonical grouping the Review page + the NVR timeline both read —
//! replacing the old
//! client-only `ReviewFeed.buildReviewItems`, so the feed, the timeline, and
//! notifications all agree on what an "event" is.
//!
//! `upsert_review_segment` is **idempotent**: it always re-aggregates the
//! segment from its member events, so calling it repeatedly (on event close,
//! again after AI analysis, during backfill) converges to the same row.

use std::sync::Arc;

use sqlx::SqlitePool;
use tauri::State;
use uuid::Uuid;

use crate::AppState;

/// Events within this many seconds of a segment's edge merge into it (mature NVRs
/// uses a comparable cutoff; mirrors the client `GAP_MS = 30_000`).
const GAP_SECS: i64 = 30;
/// Past this span, a segment stops reaching ACROSS THE GAP for its next event.
///
/// Merging is transitive and never shrinks: every merge widens the window, so the
/// next event lands inside the widened window too and the chain runs away. A
/// camera with continuous activity collapsed a whole busy period into ONE card -
/// 11 events rendered as 2 cards, with no hint on the card that it held the other
/// nine. Capping the GAP reach (not the window itself) breaks the chain while
/// keeping the non-overlap invariant: see the filter in `upsert_review_segment`.
///
/// ponytail: one global cap. Make it a per-camera setting if a site wants
/// coarser or finer review items than 2 minutes.
const MAX_SEGMENT_SECS: i64 = 120;
/// Broad categories that make a segment an ALERT (mirrors client `ALERT_CATEGORIES`).
const ALERT_CATEGORIES: [&str; 2] = ["person", "vehicle"];

// ── Aggregated payload stored in `review_segments.data` ────────────────────────

#[derive(serde::Serialize, serde::Deserialize, Default, Clone)]
pub struct ReviewSegmentData {
    pub member_ids:    Vec<String>,
    pub labels:        Vec<String>,   // specific objects (dog/car/person…)
    pub categories:    Vec<String>,   // broad buckets (drive severity)
    pub sub_label:     Option<String>,
    pub plate:         Option<String>,
    pub zones:         Vec<String>,
    pub peak:          f32,
    pub audio:         Option<String>,
    pub fall:          Option<String>,
    pub crossing:      Option<String>,
    pub speed:         Option<String>,
    pub summary:       Option<String>, // raw ai_summary of the best member
    pub clip_event_id: Option<String>,
}

/// Flat DTO sent to the frontend (columns + flattened `data`).
#[derive(serde::Serialize)]
pub struct ReviewSegmentDto {
    pub id:         String,
    pub cam_id:     u8,
    pub start_time: String,
    pub end_time:   Option<String>,
    pub severity:   String,
    pub thumbnail:  Option<String>,
    pub reviewed:   bool,
    #[serde(flatten)]
    pub data:       ReviewSegmentData,
}

// ── Member-event row (the columns the aggregation needs) ───────────────────────

struct EvRow {
    id:              String,
    started_at:      String,
    ended_at:        Option<String>,
    duration_secs:   Option<f64>,
    peak_score:      f32,
    thumbnail:       Option<String>,
    event_category:  Option<String>,
    dominant_label:  Option<String>,
    sub_label:       Option<String>,
    recognized_plate: Option<String>,
    zones_entered:   Option<String>,
    ai_summary:      Option<String>,
    top_speed_kmh:   Option<f64>,
    clip_path:       Option<String>,
    cam_id:          i64,
}

type EvTuple = (
    String, String, Option<String>, Option<f64>, f32, Option<String>, Option<String>,
    Option<String>, Option<String>, Option<String>, Option<String>, Option<String>,
    Option<f64>, Option<String>, i64,
);

const EV_COLS: &str = "id, started_at, ended_at, duration_secs, peak_score, thumbnail, \
    event_category, dominant_label, sub_label, recognized_plate, zones_entered, ai_summary, \
    top_speed_kmh, clip_path, cam_id";

fn ev_from_tuple(t: EvTuple) -> EvRow {
    EvRow {
        id: t.0, started_at: t.1, ended_at: t.2, duration_secs: t.3, peak_score: t.4,
        thumbnail: t.5, event_category: t.6, dominant_label: t.7, sub_label: t.8,
        recognized_plate: t.9, zones_entered: t.10, ai_summary: t.11, top_speed_kmh: t.12,
        clip_path: t.13, cam_id: t.14,
    }
}

async fn load_event(db: &SqlitePool, id: &str) -> Option<EvRow> {
    let q = format!("SELECT {EV_COLS} FROM motion_events WHERE id=?");
    sqlx::query_as::<_, EvTuple>(&q).bind(id).fetch_optional(db).await.ok().flatten().map(ev_from_tuple)
}

fn epoch_secs(rfc3339: &str) -> i64 {
    chrono::DateTime::parse_from_rfc3339(rfc3339).map(|d| d.timestamp()).unwrap_or(0)
}

/// End of an event in epoch seconds — `ended_at` if closed, else a floor of
/// start + max(duration, 10s) so an open/short event still has a sane window.
fn event_end_secs(ev: &EvRow) -> i64 {
    match &ev.ended_at {
        Some(ea) => epoch_secs(ea),
        None => epoch_secs(&ev.started_at) + (ev.duration_secs.unwrap_or(0.0).max(10.0) as i64),
    }
}

// ── Aggregation (server port of ReviewFeed.makeItem) ───────────────────────────

struct Agg {
    start_time: String,
    end_time: String,
    severity: String,
    thumbnail: Option<String>,
    data: ReviewSegmentData,
}

fn aggregate(members: &[EvRow]) -> Agg {
    // Start = earliest member; end = latest member end.
    let start_member = members.iter().min_by_key(|m| epoch_secs(&m.started_at)).unwrap();
    let (mut end_secs, mut end_str) = (i64::MIN, String::new());
    for m in members {
        let e = event_end_secs(m);
        if e > end_secs {
            end_secs = e;
            end_str = m.ended_at.clone().unwrap_or_else(|| {
                chrono::DateTime::from_timestamp(e, 0).map(|d| d.to_rfc3339()).unwrap_or_else(|| m.started_at.clone())
            });
        }
    }

    // Peak from NON-AUDIO members only: an audio event's peak_score is YAMNet
    // confidence, and a loud sound must never escalate a segment to "alert"
    // (red) or drive the feed's risk pill. Audio-only segment => peak 0.
    let peak = members.iter()
        .filter(|m| m.event_category.as_deref() != Some("audio"))
        .map(|m| m.peak_score).fold(0.0_f32, f32::max);

    let mut labels: Vec<String> = Vec::new();
    for m in members {
        let l = m.dominant_label.clone().or_else(|| m.event_category.clone()).unwrap_or_default();
        if !l.is_empty() && l != "other" && !labels.contains(&l) { labels.push(l); }
    }
    // Categories drive severity, so they must be derived the SAME way the alert
    // filter derives them — from (dominant_label, event_category) together, not
    // from `event_category` alone. Clip analysis writes 'other' whenever its
    // detections parse empty, so a person event is routinely stored as
    // ('other', 'person'); reading the column raw dropped it and left a segment
    // full of people categorised as audio-only, which then demoted the card to a
    // detection and hid its timeline band.
    let mut categories: Vec<String> = Vec::new();
    for m in members {
        let c = crate::agent::conditions::event_category_of(
            m.dominant_label.as_deref().unwrap_or(""),
            m.event_category.as_deref().unwrap_or(""),
        );
        if c != "other" && !categories.iter().any(|e| e == c) { categories.push(c.to_string()); }
    }
    let sub_label = members.iter().find_map(|m| m.sub_label.clone().filter(|s| !s.is_empty()));
    let plate = members.iter().find_map(|m| m.recognized_plate.clone().filter(|s| !s.is_empty()));
    let mut zones: Vec<String> = Vec::new();
    for m in members {
        if let Some(z) = &m.zones_entered {
            for part in z.split(',') {
                let p = part.trim().to_string();
                if !p.is_empty() && !zones.contains(&p) { zones.push(p); }
            }
        }
    }
    let audio = members.iter().find(|m| m.event_category.as_deref() == Some("audio"))
        .map(|m| m.dominant_label.clone().filter(|s| !s.is_empty()).unwrap_or_else(|| "Sound".into()));
    let fall = if members.iter().any(|m| m.event_category.as_deref() == Some("fall")) {
        Some("Person down".to_string())
    } else { None };
    let crossing = members.iter().find(|m| m.event_category.as_deref() == Some("crossing"))
        .and_then(|m| m.dominant_label.clone());
    let top_speed = members.iter().filter_map(|m| m.top_speed_kmh).fold(0.0_f64, f64::max);
    let speed = if top_speed >= 1.0 { Some(format!("{} km/h", top_speed.round() as i64)) } else { None };
    let summary = members.iter().find_map(|m| m.ai_summary.clone().filter(|s| !s.is_empty()));

    // Best thumbnail = highest-peak member that has one.
    let thumbnail = members.iter().filter(|m| m.thumbnail.is_some())
        .max_by(|a, b| a.peak_score.partial_cmp(&b.peak_score).unwrap_or(std::cmp::Ordering::Equal))
        .and_then(|m| m.thumbnail.clone());
    // A clip-able member, else just the first (footage_clip slices NVR regardless).
    let clip_event_id = members.iter().find(|m| m.clip_path.is_some()).map(|m| m.id.clone())
        .or_else(|| members.first().map(|m| m.id.clone()));

    // NOTE: audio presence deliberately does NOT alert — a sound event alone is
    // informational (it also must not paint red severity bands on the timeline).
    let is_alert = categories.iter().any(|c| ALERT_CATEGORIES.contains(&c.as_str()))
        || peak > 0.5 || !zones.is_empty() || fall.is_some() || crossing.is_some();

    Agg {
        start_time: start_member.started_at.clone(),
        end_time: end_str,
        severity: if is_alert { "alert".into() } else { "detection".into() },
        thumbnail,
        data: ReviewSegmentData {
            member_ids: members.iter().map(|m| m.id.clone()).collect(),
            labels, categories, sub_label, plate, zones, peak, audio, fall, crossing, speed,
            summary, clip_event_id,
        },
    }
}

// ── Upsert ─────────────────────────────────────────────────────────────────────

/// Group `event_id` into its review segment, creating or extending one. Loads
/// the event, finds an overlapping segment for the camera (within GAP), merges
/// + re-aggregates from all members, or creates a fresh segment. Idempotent.
pub async fn upsert_review_segment(db: &SqlitePool, event_id: &str) {
    let ev = match load_event(db, event_id).await { Some(e) => e, None => return };
    let cam_id = ev.cam_id;
    let ev_start = epoch_secs(&ev.started_at);
    let ev_end = event_end_secs(&ev);
    let lo = chrono::DateTime::from_timestamp(ev_start - GAP_SECS, 0)
        .map(|d| d.to_rfc3339()).unwrap_or_else(|| ev.started_at.clone());
    let hi = chrono::DateTime::from_timestamp(ev_end + GAP_SECS, 0)
        .map(|d| d.to_rfc3339()).unwrap_or_else(|| ev.started_at.clone());

    // ONE TRANSACTION around find-merge-write.
    //
    // This used to be a bare SELECT followed by an UPDATE-or-INSERT with nothing
    // holding them together. Two events closing in the same moment — which is
    // exactly what a busy scene produces — both read "no segment here" and both
    // INSERTed. The live database grew two pairs of segments sharing an identical
    // start time: that race, on disk. It is also why the module's stated
    // invariant (a non-overlapping per-camera window) stopped holding.
    let mut tx = match db.begin().await {
        Ok(t) => t,
        Err(e) => { tracing::warn!("review segment: begin failed: {e}"); return; }
    };

    // EVERY overlapping segment, not just the newest.
    //
    // The old query took `ORDER BY start_time DESC LIMIT 1`, so an event bridging
    // two existing segments merged into one and left the other overlapping for
    // good. Overlaps accumulated and nothing ever healed them. Taking all of them
    // and collapsing to one is what makes the invariant self-repairing.
    //
    // `end_time` is decoded as Option: the column is nullable and the WHERE clause
    // right here matches `end_time IS NULL`. Typed as String, one NULL row made
    // `fetch_all` return Err, `unwrap_or_default()` swallowed it, and the code fell
    // into the "no segment here" branch and INSERTed a duplicate - silently, forever.
    let mut existing: Vec<(String, String, String, Option<String>)> = sqlx::query_as(
        "SELECT id, data, start_time, end_time FROM review_segments
         WHERE cam_id=? AND start_time <= ? AND (end_time IS NULL OR end_time >= ?)
         ORDER BY start_time ASC"
    ).bind(cam_id).bind(&hi).bind(&lo).fetch_all(&mut *tx).await.unwrap_or_default();

    // Reject candidates this event must NOT be folded into:
    //
    //  * a different KIND. A five-minute "Speech" event overlapping two separate
    //    person events welded all three into one card. Sounds are their own
    //    review items (and their own Sounds tab); they never bridge video.
    //  * one that is ALREADY at the cap and that this event only reaches through
    //    the GAP_SECS slack - see MAX_SEGMENT_SECS.
    //
    // The "only reaches through slack" half is what keeps the non-overlap
    // invariant intact. An event that genuinely overlaps the stored window is
    // always merged, however wide that window has grown: splitting it out would
    // create two cards covering the same seconds, which is the duplicate-card bug
    // this module already fixed once. Only an event sitting in the gap BEYOND the
    // window can safely start a fresh segment - it cannot overlap what it follows.
    let ev_is_audio = ev.event_category.as_deref() == Some("audio");
    existing.retain(|(_, data, st, en)| {
        let seg_is_audio = serde_json::from_str::<ReviewSegmentData>(data)
            .map(|d| !d.categories.is_empty() && d.categories.iter().all(|c| c == "audio"))
            .unwrap_or(false);
        if seg_is_audio != ev_is_audio { return false; }

        let (seg_start, seg_end) = (epoch_secs(st), en.as_deref().map(epoch_secs).unwrap_or(i64::MAX));
        let overlaps = ev_start <= seg_end && ev_end >= seg_start;
        overlaps || (seg_end - seg_start) < MAX_SEGMENT_SECS
    });

    let now = chrono::Utc::now().to_rfc3339();

    if existing.is_empty() {
        let agg = aggregate(std::slice::from_ref(&ev));
        let (start_time, end_time) = with_min_window(&agg.start_time, &agg.end_time);
        let data = serde_json::to_string(&agg.data).unwrap_or_default();
        let id = format!("{}-{}", ev_start, &Uuid::new_v4().simple().to_string()[..6]);
        if let Err(e) = sqlx::query(
            "INSERT INTO review_segments(id, cam_id, start_time, end_time, severity, thumbnail, data, reviewed, created_at, updated_at)
             VALUES(?,?,?,?,?,?,?,0,?,?)"
        ).bind(&id).bind(cam_id).bind(&start_time).bind(&end_time).bind(&agg.severity)
         .bind(&agg.thumbnail).bind(&data).bind(&now).bind(&now).execute(&mut *tx).await {
            tracing::warn!("review segment: insert failed: {e}");
            return; // tx drops → rollback
        }
    } else {
        // Union the members of every overlapping segment, plus this event.
        let mut ids: Vec<String> = Vec::new();
        for (_, data_json, _, _) in &existing {
            if let Ok(d) = serde_json::from_str::<ReviewSegmentData>(data_json) {
                for id in d.member_ids {
                    if !ids.contains(&id) { ids.push(id); }
                }
            }
        }
        if !ids.contains(&ev.id) { ids.push(ev.id.clone()); }

        let mut members: Vec<EvRow> = Vec::with_capacity(ids.len());
        for id in &ids {
            if let Some(m) = load_event(db, id).await { members.push(m); }
        }
        if members.is_empty() { return; } // tx drops → rollback
        let agg = aggregate(&members);

        // NEVER SHRINK. The window is the union of what is already stored and what
        // the members say — re-aggregating from members alone overwrote the stored
        // span with a narrower one whenever a member had been deleted, which is how
        // one-second segments came to exist despite the 10 s floor in
        // `event_end_secs`. Compared as epochs, not strings: stored timestamps carry
        // differing sub-second precision and would not sort reliably as text.
        let mut start_time = agg.start_time.clone();
        let mut end_time = agg.end_time.clone();
        for (_, _, st, en) in &existing {
            if epoch_secs(st) < epoch_secs(&start_time) { start_time = st.clone(); }
            if let Some(en) = en {
                if epoch_secs(en) > epoch_secs(&end_time) { end_time = en.clone(); }
            }
        }
        let (start_time, end_time) = with_min_window(&start_time, &end_time);

        // Keep the earliest segment, fold the rest into it.
        let keep_id = existing[0].0.clone();
        for (id, _, _, _) in existing.iter().skip(1) {
            if let Err(e) = sqlx::query("DELETE FROM review_segments WHERE id=?")
                .bind(id).execute(&mut *tx).await {
                tracing::warn!("review segment: coalesce delete failed: {e}");
                return; // tx drops → rollback
            }
        }

        let data = serde_json::to_string(&agg.data).unwrap_or_default();
        if let Err(e) = sqlx::query(
            "UPDATE review_segments SET start_time=?, end_time=?, severity=?, thumbnail=?, data=?, updated_at=? WHERE id=?"
        ).bind(&start_time).bind(&end_time).bind(&agg.severity).bind(&agg.thumbnail)
         .bind(&data).bind(&now).bind(&keep_id).execute(&mut *tx).await {
            tracing::warn!("review segment: update failed: {e}");
            return; // tx drops → rollback
        }
    }

    if let Err(e) = tx.commit().await {
        tracing::warn!("review segment: commit failed: {e}");
    }
}

/// Enforce the documented minimum window on EVERY write, not just on create.
///
/// `event_end_secs` floors an open event at start + 10 s, but the merge path
/// wrote `aggregate`'s raw span straight back, so a segment could be stored
/// one second wide and then render as a junk card and a hairline timeline band.
fn with_min_window(start: &str, end: &str) -> (String, String) {
    const MIN_WINDOW_SECS: i64 = 10;
    let s = epoch_secs(start);
    let e = epoch_secs(end);
    if e - s >= MIN_WINDOW_SECS { return (start.to_string(), end.to_string()); }
    let widened = chrono::DateTime::from_timestamp(s + MIN_WINDOW_SECS, 0)
        .map(|d| d.to_rfc3339())
        .unwrap_or_else(|| end.to_string());
    (start.to_string(), widened)
}

// ── Commands ───────────────────────────────────────────────────────────────────

/// All review segments overlapping the range, newest first. The canonical feed
/// source for the Review day-view + the timeline review-band overlay.
#[tauri::command]
pub async fn get_review_segments(
    state: State<'_, Arc<AppState>>,
    range_start: String,
    range_end: String,
) -> Result<Vec<ReviewSegmentDto>, String> {
    let rows: Vec<(String, i64, String, Option<String>, String, Option<String>, String, i64)> = sqlx::query_as(
        // OVERLAP, not "started inside". Bounding on start_time alone dropped a
        // segment that opened at 23:58 and ran past midnight from BOTH days -
        // the day it started (range hadn't begun) and the day it ran into.
        "SELECT id, cam_id, start_time, end_time, severity, thumbnail, data, reviewed
         FROM review_segments
         WHERE start_time <= ? AND COALESCE(end_time, start_time) >= ?
         ORDER BY start_time DESC LIMIT 5000"
    ).bind(&range_end).bind(&range_start).fetch_all(&state.db).await.map_err(|e| e.to_string())?;

    Ok(rows.into_iter().map(|(id, cam, st, et, sev, th, data, rev)| ReviewSegmentDto {
        id, cam_id: cam as u8, start_time: st, end_time: et, severity: sev, thumbnail: th,
        reviewed: rev != 0,
        data: serde_json::from_str(&data).unwrap_or_default(),
    }).collect())
}

#[tauri::command]
pub async fn set_review_segment_reviewed(
    state: State<'_, Arc<AppState>>,
    id: String,
    reviewed: bool,
) -> Result<(), String> {
    sqlx::query("UPDATE review_segments SET reviewed=? WHERE id=?")
        .bind(reviewed as i64).bind(&id).execute(&state.db).await.map_err(|e| e.to_string())?;
    Ok(())
}

/// One-shot backfill: group existing events (last 30 days) that aren't yet in a
/// segment. Guarded by a settings flag so it runs once. Processing ASC lets each
/// event create-or-merge naturally. Live hooks handle everything after.
pub async fn backfill_review_segments(db: &SqlitePool) {
    let done: Option<(String,)> = sqlx::query_as("SELECT value FROM settings WHERE key='review_backfill_v1'")
        .fetch_optional(db).await.ok().flatten();
    if done.is_some() { return; }

    let since = (chrono::Utc::now() - chrono::Duration::days(30)).to_rfc3339();
    let ids: Vec<(String,)> = sqlx::query_as(
        "SELECT id FROM motion_events WHERE started_at >= ? ORDER BY started_at ASC LIMIT 50000"
    ).bind(&since).fetch_all(db).await.unwrap_or_default();
    let n = ids.len();
    for (id,) in ids {
        upsert_review_segment(db, &id).await;
    }
    sqlx::query("INSERT OR REPLACE INTO settings(key,value) VALUES('review_backfill_v1','1')")
        .execute(db).await.ok();
    if n > 0 {
        tracing::info!("Review segments backfill: grouped {} events from the last 30 days.", n);
    }
}

/// Rebuild the whole derived table from `motion_events`.
///
/// `upsert_review_segment` can only ever MERGE — it has no code path that splits
/// a segment. So fixing the writer does nothing for rows already on disk, and
/// every existing install has bad ones: overlapping siblings from the old race,
/// one-second slivers from the shrinking merge, and (the reason this is a v2)
/// runaway segments where a whole busy period, or an audio event bridging two
/// unrelated video events, collapsed into a single card.
///
/// Dropping the table first is what makes the split possible. This is SAFE and is
/// NOT event deletion: `review_segments` is derived entirely from `motion_events`,
/// which this never touches — no events, no clips, no face data. It deliberately
/// does not route through `delete_events`, which is for removing real events.
pub async fn rebuild_review_segments(db: &SqlitePool) {
    let done: Option<(String,)> = sqlx::query_as("SELECT value FROM settings WHERE key='review_rebuild_v2'")
        .fetch_optional(db).await.ok().flatten();
    if done.is_some() { return; }

    let before: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM review_segments")
        .fetch_one(db).await.unwrap_or(0);
    if let Err(e) = sqlx::query("DELETE FROM review_segments").execute(db).await {
        tracing::warn!("Review segments rebuild: clear failed, skipping: {e}");
        return; // leave the old rows rather than half-rebuild
    }

    // Paged by (started_at, id) rather than one LIMIT 50000. The old cap silently
    // rebuilt only the OLDEST 50k events on a big archive and left recent days —
    // the ones anyone actually looks at — unrepaired.
    const PAGE: i64 = 5000;
    let mut cursor = (String::new(), String::new());
    let mut n: usize = 0;
    loop {
        let page: Vec<(String, String)> = sqlx::query_as(
            "SELECT id, started_at FROM motion_events
             WHERE (started_at, id) > (?, ?)
             ORDER BY started_at ASC, id ASC LIMIT ?"
        ).bind(&cursor.0).bind(&cursor.1).bind(PAGE)
         .fetch_all(db).await.unwrap_or_default();
        if page.is_empty() { break; }
        for (id, started_at) in &page {
            upsert_review_segment(db, id).await;
            cursor = (started_at.clone(), id.clone());
        }
        n += page.len();
    }

    let after: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM review_segments")
        .fetch_one(db).await.unwrap_or(0);
    sqlx::query("INSERT OR REPLACE INTO settings(key,value) VALUES('review_rebuild_v2','1')")
        .execute(db).await.ok();
    tracing::info!("Review segments rebuild: {} segments -> {} over {} events.", before, after, n);
}

// ── Tests ─────────────────────────────────────────────────────────────────────
//
// The invariant this module claims in its own header — "a non-overlapping
// per-camera time window" — had no test, and the live database had been
// violating it for as long as the feature existed. These assert the invariant
// directly rather than testing the helpers around it.

#[cfg(test)]
mod tests {
    use super::*;

    async fn mem_db() -> SqlitePool {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        crate::db::init_db(&pool).await.unwrap();
        pool
    }

    async fn add_event(pool: &SqlitePool, id: &str, cam: i64, start: &str, end: &str) {
        sqlx::query(
            "INSERT INTO motion_events(id, cam_id, started_at, ended_at, peak_score)
             VALUES(?,?,?,?,0.5)")
            .bind(id).bind(cam).bind(start).bind(end)
            .execute(pool).await.unwrap();
    }

    /// Segments for one camera, ordered, as (start_secs, end_secs).
    async fn spans(pool: &SqlitePool, cam: i64) -> Vec<(i64, i64)> {
        let rows: Vec<(String, String)> = sqlx::query_as(
            "SELECT start_time, end_time FROM review_segments WHERE cam_id=? ORDER BY start_time ASC")
            .bind(cam).fetch_all(pool).await.unwrap();
        rows.iter().map(|(a, b)| (epoch_secs(a), epoch_secs(b))).collect()
    }

    /// A one-second span is widened to the documented floor; a real span is not
    /// touched. Segments a second wide are what produced junk cards and hairline
    /// timeline bands.
    #[test]
    fn min_window_widens_a_sliver_only() {
        let (s, e) = with_min_window("2026-09-04T00:27:33+00:00", "2026-09-04T00:27:34+00:00");
        assert_eq!(epoch_secs(&e) - epoch_secs(&s), 10, "one-second sliver widened to the floor");

        let (s2, e2) = with_min_window("2026-09-04T00:00:00+00:00", "2026-09-04T00:05:00+00:00");
        assert_eq!(epoch_secs(&e2) - epoch_secs(&s2), 300, "a real span is left alone");
    }

    /// THE invariant. Events inside the gap window must collapse into exactly one
    /// segment per camera, and segments must never overlap — which is what the
    /// old `LIMIT 1` lookup and the untransacted read-then-write both broke.
    #[tokio::test]
    async fn segments_never_overlap_within_a_camera() {
        let pool = mem_db().await;
        // Four overlapping/adjacent events on cam 0, all inside GAP_SECS of each
        // other, plus one on cam 1 at the same time.
        add_event(&pool, "a", 0, "2026-09-04T00:24:32+00:00", "2026-09-04T00:25:57+00:00").await;
        add_event(&pool, "b", 0, "2026-09-04T00:24:32+00:00", "2026-09-04T00:29:43+00:00").await;
        add_event(&pool, "c", 0, "2026-09-04T00:27:33+00:00", "2026-09-04T00:27:34+00:00").await;
        add_event(&pool, "d", 0, "2026-09-04T00:29:33+00:00", "2026-09-04T00:34:54+00:00").await;
        add_event(&pool, "x", 1, "2026-09-04T00:24:32+00:00", "2026-09-04T00:25:00+00:00").await;
        for id in ["a", "b", "c", "d", "x"] {
            upsert_review_segment(&pool, id).await;
        }

        let cam0 = spans(&pool, 0).await;
        assert_eq!(cam0.len(), 1,
            "four events within the gap must coalesce to ONE segment, got {cam0:?}");
        for w in cam0.windows(2) {
            assert!(w[0].1 < w[1].0, "segments overlap: {:?} then {:?}", w[0], w[1]);
        }
        // The surviving window must cover every member.
        assert_eq!(cam0[0].0, epoch_secs("2026-09-04T00:24:32+00:00"), "earliest start kept");
        assert_eq!(cam0[0].1, epoch_secs("2026-09-04T00:34:54+00:00"), "latest end kept");

        // Per-camera: cam 1 keeps its own segment.
        assert_eq!(spans(&pool, 1).await.len(), 1, "cam 1 is not merged into cam 0");
    }

    /// Merging must only ever widen. Re-aggregating from members alone let a
    /// stored window be overwritten with a narrower one — the mechanism behind
    /// the one-second segments found in the live database.
    #[tokio::test]
    async fn merging_never_shrinks_the_window() {
        let pool = mem_db().await;
        add_event(&pool, "long", 0, "2026-09-04T04:12:28+00:00", "2026-09-04T04:20:17+00:00").await;
        upsert_review_segment(&pool, "long").await;
        let before = spans(&pool, 0).await;
        assert_eq!(before.len(), 1);

        // A short event wholly inside the existing window.
        add_event(&pool, "short", 0, "2026-09-04T04:12:28+00:00", "2026-09-04T04:12:38+00:00").await;
        upsert_review_segment(&pool, "short").await;

        let after = spans(&pool, 0).await;
        assert_eq!(after.len(), 1, "the short event must merge, not create a sibling: {after:?}");
        assert!(after[0].0 <= before[0].0, "start moved forward: {:?} -> {:?}", before[0], after[0]);
        assert!(after[0].1 >= before[0].1, "end moved backward: {:?} -> {:?}", before[0], after[0]);
    }

    /// Same helper as `add_event`, with a category (audio events need one).
    async fn add_cat_event(pool: &SqlitePool, id: &str, cam: i64, start: &str, end: &str,
                           cat: &str, label: &str) {
        sqlx::query(
            "INSERT INTO motion_events(id, cam_id, started_at, ended_at, peak_score,
                                       event_category, dominant_label)
             VALUES(?,?,?,?,0.5,?,?)")
            .bind(id).bind(cam).bind(start).bind(end).bind(cat).bind(label)
            .execute(pool).await.unwrap();
    }

    /// The direction nothing asserted before: a busy period must NOT collapse into
    /// one card. Three back-to-back five-minute person events, each starting a
    /// second after the last one ended, chained through the 30 s gap into a single
    /// ten-minute segment holding all of them — with nothing on the card saying so.
    /// They are separated by real gaps, so splitting them cannot overlap.
    #[tokio::test]
    async fn a_busy_period_splits_instead_of_collapsing() {
        let pool = mem_db().await;
        add_event(&pool, "p1", 0, "2026-08-29T00:24:32+00:00", "2026-08-29T00:29:32+00:00").await;
        add_event(&pool, "p2", 0, "2026-08-29T00:29:33+00:00", "2026-08-29T00:34:33+00:00").await;
        add_event(&pool, "p3", 0, "2026-08-29T00:34:44+00:00", "2026-08-29T00:34:54+00:00").await;
        for id in ["p1", "p2", "p3"] { upsert_review_segment(&pool, id).await; }

        let cam0 = spans(&pool, 0).await;
        assert!(cam0.len() > 1, "a busy period must not collapse into one card: {cam0:?}");
        // The invariant still holds across the split.
        for w in cam0.windows(2) {
            assert!(w[0].1 < w[1].0, "split produced overlapping segments: {:?} then {:?}", w[0], w[1]);
        }
    }

    /// An event that OVERLAPS an over-cap segment still merges. Splitting it out
    /// would put two cards on the same seconds — the duplicate-card bug. The cap
    /// only stops a fat segment reaching across a gap, never a real overlap.
    #[tokio::test]
    async fn the_cap_never_creates_an_overlap() {
        let pool = mem_db().await;
        add_event(&pool, "long",  0, "2026-09-04T04:12:28+00:00", "2026-09-04T04:20:17+00:00").await;
        add_event(&pool, "inside", 0, "2026-09-04T04:15:00+00:00", "2026-09-04T04:15:10+00:00").await;
        for id in ["long", "inside"] { upsert_review_segment(&pool, id).await; }

        let cam0 = spans(&pool, 0).await;
        assert_eq!(cam0.len(), 1, "an overlapping event must merge even past the cap: {cam0:?}");
    }

    /// A long sound overlapping two separate video events must not weld them into
    /// one card. This is what made 11 events render as 2: five-minute "Speech"
    /// events spanning the quiet second between two person events.
    #[tokio::test]
    async fn audio_does_not_bridge_two_video_events() {
        let pool = mem_db().await;
        add_cat_event(&pool, "v1", 0, "2026-08-29T00:20:00+00:00", "2026-08-29T00:20:30+00:00",
                      "person", "person").await;
        add_cat_event(&pool, "v2", 0, "2026-08-29T00:22:00+00:00", "2026-08-29T00:22:30+00:00",
                      "person", "person").await;
        // Speech covering the whole span including the 90 s hole between them.
        add_cat_event(&pool, "a1", 0, "2026-08-29T00:20:00+00:00", "2026-08-29T00:22:30+00:00",
                      "audio", "Speech").await;
        for id in ["v1", "v2", "a1"] { upsert_review_segment(&pool, id).await; }

        let members: Vec<Vec<String>> = sqlx::query_scalar::<_, String>(
            "SELECT data FROM review_segments WHERE cam_id=0 ORDER BY start_time ASC")
            .fetch_all(&pool).await.unwrap()
            .iter()
            .map(|d| serde_json::from_str::<ReviewSegmentData>(d).unwrap().member_ids)
            .collect();
        assert!(!members.iter().any(|m| m.contains(&"v1".to_string()) && m.contains(&"v2".to_string())),
                "a sound welded two separate video events into one card: {members:?}");
    }

    /// Re-running the upsert for the same event must be a no-op, not a new
    /// segment. The repair pass re-drives every event, so idempotence is what
    /// makes that safe to run on a live database.
    #[tokio::test]
    async fn upsert_is_idempotent() {
        let pool = mem_db().await;
        add_event(&pool, "e", 0, "2026-09-04T01:00:00+00:00", "2026-09-04T01:01:00+00:00").await;
        for _ in 0..3 { upsert_review_segment(&pool, "e").await; }
        assert_eq!(spans(&pool, 0).await.len(), 1, "repeat upserts must not fan out");
    }
}
