//! Retrieve first, then answer.
//!
//! The Guardian used to work the other way round: the model wrote prose, and only
//! afterwards did Rust splice query results into the gaps
//! (`tools::execute_query_tags`). On the on-device provider — which is the DEFAULT
//! and which has no tool loop — that meant the model **never saw the data it was
//! writing about**. It was guessing on every turn, by construction, and no amount
//! of prompt tuning could fix it.
//!
//! This module inverts the order:
//!
//! 1. [`pre_resolve`] turns the question into a [`Query`] with pure string work —
//!    camera names, enrolled people, dates, "how many", "show me". No model, no
//!    latency, nothing to hallucinate.
//! 2. [`retrieve`] runs real SQL and returns [`Evidence`]: dated lines, FULL event
//!    ids, camera names, counts.
//! 3. The model gets a SHORT prompt — the question plus that evidence — and writes
//!    the answer. It cannot invent the facts, because they are in front of it.
//! 4. If the model produces nothing usable, [`Evidence::fallback_answer`] ships
//!    instead. **A correct, dated answer arrives even with the model deleted from
//!    disk** — which is the property that makes this worth building.
//!
//! Division of labour, stated once: Rust computes, the model phrases. That plays
//! to what a small model is actually good at (its own card says "data extraction,
//! structured outputs, tool use"; "not recommended for knowledge-intensive tasks")
//! instead of asking it to be an analyst.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::AppState;

/// What the user is asking for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Kind {
    /// The clock. Asked constantly, and the one question a language model has no
    /// business answering: it cannot read a clock, so it copies a plausible-looking
    /// time out of whatever else is in its context. Observed live — asked "what is
    /// the time now" it replied "19:01 (local time, UTC-05:00)", which was a
    /// person's last-seen time from an unrelated line, in an invented timezone.
    Clock,
    /// "who are you" / "what can you do". Also answered from facts rather than by
    /// asking the model to describe itself: handed its own system prompt and asked
    /// what it can do, a small model reads the prompt back — observed live, it
    /// recited the COCO class list and the literal line "Guardian agent (you):".
    About,
    /// "hi", "thanks". Not a question at all, and the one thing a retrieval agent
    /// must not do is answer it with retrieval: "hello" came back as "No new
    /// events recorded in the past 6 hours."
    Greeting,
    /// Events matching the slots — the default shape of "what happened".
    Events,
    /// A number, not a list ("how many people came yesterday").
    Count,
    /// The archive itself: how far back it goes, which days hold anything.
    ///
    /// Distinct from `Count` because the answer is about the RECORDING, not
    /// about events in a window — and because "how many days of footage do you
    /// have" was being answered "nothing recorded today".
    Coverage,
    /// One named person's activity.
    Person,
    /// Vehicles / plates.
    Vehicles,
    /// Audio events.
    Sounds,
    /// System + camera state.
    Status,
    /// Keyword search across the archive.
    Search,
    /// Which models/engines are running. Answered from `Settings`, because the
    /// model cannot know and will invent one — asked "which vision model is
    /// running" it replied "the standard video processing model", which is not a
    /// thing that exists.
    Models,
    /// The user's saved events.
    Bookmarks,
    /// "show me those" — whatever was last put in front of them.
    Recall,
}

impl Kind {
    /// Does an answer of this kind ASSERT things that were recorded?
    ///
    /// Clock/About/Greeting/Models/Status are answered from the clock and the
    /// config — they legitimately carry no rows. Everything else is a claim about
    /// footage, and a claim about footage with no rows behind it is a fabrication.
    fn asserts_footage(self) -> bool {
        !matches!(self, Kind::Clock | Kind::About | Kind::Greeting | Kind::Models | Kind::Status)
    }
}

/// The time window. Every variant resolves to a **local**-calendar predicate:
/// asking "today" at 23:00 must not silently mean "since 05:30 UTC".
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum When {
    Today,
    Yesterday,
    /// Evening onwards within the last 12 hours.
    LastNight,
    /// A rolling window, for "last hour" / "this week".
    Hours(u32),
    /// An explicit calendar day, `YYYY-MM-DD` local.
    Day(String),
    /// An inclusive local date range, used when a single day found nothing and
    /// the refinement loop widened to its neighbours.
    Range(String, String),
    /// No time bound at all. Bookmarks with no date named mean ALL of them.
    Ever,
}

impl When {
    /// How this window should be described to the user. Always concrete — never
    /// "recently", which is the phrasing that makes a security report useless.
    fn label(&self) -> String {
        let d = |off: i64| {
            (chrono::Local::now() + chrono::Duration::days(off))
                .format("%Y-%m-%d").to_string()
        };
        match self {
            When::Today       => format!("today, {}", d(0)),
            When::Yesterday   => format!("yesterday, {}", d(-1)),
            When::LastNight   => format!("last night (evening of {})", d(-1)),
            When::Hours(1)    => "the last hour".into(),
            When::Hours(24)   => "the last 24 hours".into(),
            When::Hours(168)  => "the last 7 days".into(),
            When::Hours(n)    => format!("the last {n} hours"),
            // Weekday too, not just the digits — "on Tuesday 28 July" is what a
            // person can check against their own memory of the week.
            When::Day(day)    => match chrono::NaiveDate::parse_from_str(day, "%Y-%m-%d") {
                Ok(d)  => format!("on {}", d.format("%A %-d %B")),
                Err(_) => format!("on {day}"),
            },
            When::Range(a, b) => format!("between {a} and {b}"),
            When::Ever        => "of all time".into(),
        }
    }

    /// Can results from this window fall on more than one calendar day?
    ///
    /// Decides whether each evidence row has to carry its date. A single named
    /// day does not need one repeated twelve times — the headline says it — but
    /// anything wider does, or the model is left to infer dates it cannot know.
    fn spans_multiple_days(&self) -> bool {
        match self {
            When::Today | When::Yesterday | When::Day(_) => false,
            // Evening-onwards can cross midnight.
            When::LastNight => true,
            When::Hours(n)  => *n > 24,
            When::Range(a, b) => a != b,
            When::Ever      => true,
        }
    }
}

/// A resolved question: what to fetch, and over what.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Query {
    pub kind: Kind,
    pub when: When,
    pub cam: Option<i64>,
    pub who: Option<String>,
    /// One of `event_category_of`'s buckets — person/vehicle/animal/audio — so the
    /// filter vocabulary and the stored classification can never drift apart.
    pub what: Option<&'static str>,
    pub risk: Option<&'static str>,
    /// Local hour window `[start, end)`; `start > end` wraps midnight.
    pub hours: Option<(u32, u32)>,
    /// Garment colours: `("top"|"bottom", colour)`, matched against `outfit`.
    pub outfit: Vec<(&'static str, &'static str)>,
    /// Residual free-text terms.
    pub keywords: Vec<String>,
    pub limit: i64,
    /// The user asked to SEE something, not just be told. Drives `[SEND_CLIP]`.
    pub wants_media: bool,
    /// Restrict to events the user has saved.
    pub bookmarked: bool,
    /// EXCLUDE this category — "show events with no people".
    pub exclude_what: Option<&'static str>,
}

impl Query {
    fn new(kind: Kind, when: When) -> Self {
        Query { kind, when, cam: None, who: None, what: None, risk: None,
                hours: None, outfit: Vec::new(), keywords: Vec::new(),
                limit: 12, wants_media: false, bookmarked: false, exclude_what: None }
    }
}

// ─── The one SQL builder ─────────────────────────────────────────────────────

/// "Today" in the user's timezone. The single definition in the codebase —
/// `chat::build_situation_ctx` and `conditions::explore_events` both resolve
/// "today" this way, and two different answers to "what is today" is exactly the
/// bug class that made timeline pins vanish after 19:00 local.
pub(super) const TODAY_SQL: &str = "date(started_at,'localtime') = date('now','localtime')";

/// Build the `WHERE` predicate and its binds.
///
/// Returns `(sql, binds)`. **Every user-derived value is a bind**, never formatted
/// into the string — this runs on text that reaches us from Telegram, which is
/// untrusted input by definition.
pub(super) fn where_sql(q: &Query) -> (String, Vec<String>) {
    let mut sql = String::new();
    let mut binds: Vec<String> = Vec::new();

    // A single named day PLUS a window that wraps midnight is the one case the
    // obvious predicate gets quietly wrong: `date = Tue AND (H>=22 OR H<02)` also
    // matches Tuesday 00:00-01:59, which is the night BEFORE the one meant. Emit
    // the two-date form instead, and skip the generic clauses below.
    if let (When::Day(day), Some((a, b))) = (&q.when, q.hours) {
        if a > b {
            let next = chrono::NaiveDate::parse_from_str(day, "%Y-%m-%d")
                .map(|d| (d + chrono::Duration::days(1)).format("%Y-%m-%d").to_string())
                .unwrap_or_else(|_| day.clone());
            sql.push_str(
                "((date(started_at,'localtime') = ? AND strftime('%H',started_at,'localtime') >= ?) \
                 OR (date(started_at,'localtime') = ? AND strftime('%H',started_at,'localtime') < ?))");
            binds.push(day.clone());
            binds.push(format!("{a:02}"));
            binds.push(next);
            binds.push(format!("{b:02}"));
            push_common(q, &mut sql, &mut binds);
            return (sql, binds);
        }
    }

    match &q.when {
        When::Today     => sql.push_str(TODAY_SQL),
        When::Yesterday => sql.push_str(
            "date(started_at,'localtime') = date('now','-1 day','localtime')"),
        // Same clause `explore_events` has always used for "last night".
        When::LastNight => sql.push_str(
            "started_at > datetime('now','-12 hours') \
             AND cast(strftime('%H', started_at,'localtime') as int) >= 20"),
        When::Hours(n)  => {
            sql.push_str("started_at > datetime('now', ?)");
            binds.push(format!("-{n} hours"));
        }
        When::Day(day)  => {
            sql.push_str("date(started_at,'localtime') = ?");
            binds.push(day.clone());
        }
        When::Range(a, b) => {
            sql.push_str("date(started_at,'localtime') BETWEEN ? AND ?");
            binds.push(a.clone());
            binds.push(b.clone());
        }
        When::Ever => sql.push_str("1=1"),
    }

    // Hour window. Compared as STRINGS, deliberately.
    //
    // `binds` is `Vec<String>`, and SQLite orders values by type before value —
    // so `cast(strftime('%H',…) as int) >= '20'` compares INTEGER against TEXT,
    // is always false, and every hour filter would silently return nothing.
    // `strftime('%H')` is zero-padded, so lexicographic order equals numeric
    // order across 00..23 and a string comparison is simply correct.
    if let Some((a, b)) = q.hours {
        if a < b {
            sql.push_str(" AND strftime('%H',started_at,'localtime') >= ? \
                          AND strftime('%H',started_at,'localtime') < ?");
        } else {
            sql.push_str(" AND (strftime('%H',started_at,'localtime') >= ? \
                          OR strftime('%H',started_at,'localtime') < ?)");
        }
        binds.push(format!("{a:02}"));
        binds.push(format!("{b:02}"));
    }

    push_common(q, &mut sql, &mut binds);
    (sql, binds)
}

/// SQL for one `event_category_of` bucket, expanded back to the stored labels.
///
/// Literal, closed sets — no user text reaches the SQL. The `COALESCE` matters:
/// the columns are nullable, and `NOT (col = 'person')` evaluates to NULL rather
/// than true for an uncategorised row — so without it a negated filter would
/// silently drop every event the classifier never labelled, which is most of what
/// "no people" is asking for.
fn category_predicate(what: &str) -> &'static str {
    match what {
        "audio"   => "(COALESCE(event_category,'') = 'audio')",
        "person"  => "(COALESCE(event_category,'') = 'person' \
                      OR COALESCE(dominant_label,'') = 'person')",
        "vehicle" => "(COALESCE(event_category,'') = 'vehicle' \
                      OR COALESCE(dominant_label,'') IN \
                         ('car','truck','bus','motorcycle','bicycle','vehicle'))",
        "animal"  => "(COALESCE(dominant_label,'') IN \
                      ('dog','cat','bird','horse','sheep','cow','bear'))",
        _ => "(1=1)",
    }
}

/// The slot predicates that don't depend on how the time clause was built.
fn push_common(q: &Query, sql: &mut String, binds: &mut Vec<String>) {

    if let Some(cam) = q.cam {
        // Bound like everything else even though it is already an integer —
        // one rule, no exceptions to remember.
        sql.push_str(" AND cam_id = ?");
        binds.push(cam.to_string());
    }
    if let Some(who) = &q.who {
        sql.push_str(" AND sub_label = ? COLLATE NOCASE");
        binds.push(who.clone());
    }
    if let Some(what) = q.what {
        sql.push_str(" AND ");
        sql.push_str(category_predicate(what));
    }
    // The same predicate, inverted. `NOT (…)` on its own would drop rows whose
    // columns are NULL — SQL three-valued logic — and an uncategorised event is
    // exactly what "no people" should include, hence the COALESCE inside
    // `category_predicate`.
    if let Some(what) = q.exclude_what {
        sql.push_str(" AND NOT ");
        sql.push_str(category_predicate(what));
    }
    if let Some(risk) = q.risk {
        sql.push_str(match risk {
            "critical" => " AND EXISTS (SELECT 1 FROM agent_alerts aa \
                             WHERE aa.event_id = motion_events.id AND aa.risk_level = 'critical')",
            // Quiet events: no alert at all, or one graded below suspicious. An
            // event with no `agent_alerts` row IS low risk — that is precisely
            // what not raising an alert means.
            "low"      => " AND NOT EXISTS (SELECT 1 FROM agent_alerts aa \
                             WHERE aa.event_id = motion_events.id \
                               AND aa.risk_level IN ('suspicious','high','critical'))",
            _          => " AND EXISTS (SELECT 1 FROM agent_alerts aa \
                             WHERE aa.event_id = motion_events.id \
                               AND aa.risk_level IN ('suspicious','high','critical'))",
        });
    }
    // Saved events only. The EXISTS runs FROM `motion_events`, so a bookmark row
    // left behind by a deleted event (there is no FK cascade) can never surface a
    // ghost — it simply matches nothing.
    if q.bookmarked {
        sql.push_str(" AND EXISTS (SELECT 1 FROM event_bookmarks b                        WHERE b.event_id = motion_events.id)");
    }
    // Garment colours. `json_extract` returns NULL when the column is NULL or the
    // band is absent, and `NULL = 'red'` is NULL — so an event with no clothing
    // data never matches, which is right: unknown is not "no". The count of those
    // is reported separately, because "no red jacket" and "I couldn't see what
    // anyone was wearing" are very different answers.
    for (band, colour) in &q.outfit {
        sql.push_str(" AND json_extract(outfit, ?) = ?");
        binds.push(format!("$.{band}"));
        binds.push((*colour).to_string());
    }
    // Free text, one AND group per keyword over the searchable columns — the
    // same column set `nvr_recording::keyword_search` uses.
    for kw in &q.keywords {
        sql.push_str(
            " AND (LOWER(COALESCE(ai_summary,'')) LIKE ? \
               OR LOWER(COALESCE(dominant_label,'')) LIKE ? \
               OR LOWER(COALESCE(sub_label,'')) LIKE ? \
               OR LOWER(COALESCE(recognized_plate,'')) LIKE ? \
               OR LOWER(COALESCE(zones_entered,'')) LIKE ? \
               OR LOWER(COALESCE(event_category,'')) LIKE ?)");
        for _ in 0..6 { binds.push(format!("%{}%", kw.to_lowercase())); }
    }
}

// ─── Camera names ────────────────────────────────────────────────────────────

/// Configured camera names by slot. The agent used to talk about "Camera 0"
/// because `camera_configs.name` was reachable only from the Telegram menus —
/// so it could neither report a camera by name nor understand a question that
/// used one.
pub(super) async fn camera_names(db: &sqlx::SqlitePool) -> BTreeMap<i64, String> {
    // Every CONFIGURED camera, named or not. Filtering on `name <> ''` reported
    // "Cameras: none set up yet" to a user with two cameras that simply had no
    // names typed in — a configured camera exists whether or not it was named.
    let rows: Vec<(i64, String)> = sqlx::query_as(
        "SELECT cam_id, name FROM camera_configs ORDER BY cam_id"
    ).fetch_all(db).await.unwrap_or_default();
    rows.into_iter()
        .map(|(id, name)| {
            let n = name.trim();
            (id, if n.is_empty() { format!("Camera {id}") } else { n.to_string() })
        })
        .collect()
}

/// How to name a camera in output: its configured name, else the slot number.
fn cam_label(names: &BTreeMap<i64, String>, cam: i64) -> String {
    names.get(&cam).cloned().unwrap_or_else(|| format!("Camera {cam}"))
}

// ─── Evidence ────────────────────────────────────────────────────────────────

/// What the database actually says. This is what the model is shown, and what the
/// deterministic fallback is built from — the two never diverge because they read
/// the same struct.
pub(super) struct Evidence {
    pub headline: String,
    pub lines: Vec<String>,
    /// FULL event uuids in display order. Truncated ids were why every clip the
    /// agent offered to send resolved to nothing.
    pub ids: Vec<String>,
    pub span: String,
    /// `lines` are a SAMPLE of a longer result and may be trimmed with an "…and N
    /// more". False when they are the complete answer — "…and 1 more." tacked onto
    /// a description of what the assistant does is nonsense, and shipped once.
    pub sampled: bool,
    /// How many events MATCH, as opposed to how many are in `lines`.
    ///
    /// The two diverge whenever the window is bigger than the row budget, and
    /// conflating them is how "12 events" came to read as the whole truth when
    /// it was one page of forty-seven. `answer()` states both.
    pub total: usize,
}

impl Evidence {
    fn empty(span: String, what: &str) -> Self {
        Evidence {
            headline: format!("Nothing {what} {span}."),
            lines: Vec::new(), ids: Vec::new(), span, sampled: true, total: 0,
        }
    }

    /// Evidence that is complete as written — shown in full, never trimmed.
    fn whole(headline: String, lines: Vec<String>, span: String) -> Self {
        Evidence { total: lines.len(), headline, lines, ids: Vec::new(), span, sampled: false }
    }

    pub fn is_empty(&self) -> bool { self.lines.is_empty() }

    /// The answer to ship when the model gives us nothing usable.
    ///
    /// Templated, always correct, and **always carrying its date range** — an
    /// undated security report is worthless, so no branch here may omit the span.
    pub fn fallback_answer(&self) -> String {
        if self.lines.is_empty() { return self.headline.clone(); }
        if !self.sampled {
            return format!("{}\n{}", self.headline, self.lines.join("\n"));
        }
        let shown = self.lines.iter().take(5).cloned().collect::<Vec<_>>().join("\n");
        let more = self.lines.len().saturating_sub(5);
        let tail = if more > 0 { format!("\n…and {more} more.") } else { String::new() };
        format!("{}\n{shown}{tail}", self.headline)
    }
}

// ─── Deterministic pre-resolver ──────────────────────────────────────────────

/// Turn the question into a [`Query`] using nothing but string matching.
///
/// Returns `None` when the intent is genuinely unclear — the caller then falls
/// back to the conversational path rather than confidently answering the wrong
/// question. The miss rate is logged, so whether a model-based extractor is worth
/// adding on top is a measurement rather than an opinion.
pub(super) fn pre_resolve(
    question: &str,
    cams: &BTreeMap<i64, String>,
    people: &[String],
) -> Option<Query> {
    let q = question.to_lowercase();
    let has = |w: &str| q.contains(w);

    // ── Meta and smalltalk, answered from facts ─────────────────────────────
    //
    // Checked before everything else, and matched on WORDS rather than whole
    // phrases. The first version listed phrases ("what time", "the time now") and
    // "what is the time" matched none of them — it fell through to the small
    // model, which echoed the question back verbatim. Typos do the same to any
    // phrase list: "wha5 date is it?" is a real thing a real user typed.
    let words: std::collections::HashSet<&str> = q
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect();
    let word = |w: &str| words.contains(w);

    // Greeting: contains a greeting word AND asks for no data.
    //
    // The first version capped it at three words, so "hii how are you" — four —
    // fell through to the small model and came back as "The cameras haven't
    // detected any activity lately." A greeting is not defined by its length;
    // it is defined by there being nothing to look up. So test for the absence
    // of data intent instead, which also lets "hi, what happened today" through
    // to the query path where it belongs.
    const HELLOS: &[&str] = &["hi", "hii", "hiii", "hey", "heya", "hello", "helo",
                              "yo", "sup", "greetings", "howdy", "hola", "namaste"];
    const THANKS: &[&str] = &["thanks", "thank", "ty", "thx", "cheers", "nice",
                              "great", "cool", "ok", "okay", "awesome", "perfect"];
    // Any of these means the user wants something looked up, so it is not smalltalk.
    const DATA_INTENT: &[&str] = &[
        "happen", "happened", "event", "events", "alert", "alerts", "clip", "clips",
        "footage", "video", "camera", "cameras", "person", "people", "someone",
        "anyone", "car", "cars", "vehicle", "plate", "sound", "sounds", "noise",
        "today", "yesterday", "night", "week", "hour", "time", "date", "show",
        "find", "search", "many", "count", "status", "who", "when", "where",
    ];
    let smalltalk = !words.is_empty()
        && !words.iter().any(|w| DATA_INTENT.contains(w))
        && (words.iter().any(|w| HELLOS.contains(w))
            || words.iter().all(|w| THANKS.contains(w) || w == &"you" || w == &"it"
                                    || w == &"is" || w == &"that"));
    // "how are you" / "how's it going" carry no data intent either.
    if smalltalk || has("how are you") || has("how are ya") || has("how is it going")
        || has("how's it going") || has("hows it going")
    {
        return Some(Query::new(Kind::Greeting, When::Today));
    }

    // The clock. One word is enough — no phrasing of "what is the time" that a
    // human will type should miss, and no event query says "time" or "date"
    // without also naming something to look for.
    if word("time") || word("date") || word("clock")
        || (word("day") && (word("what") || word("which") || word("todays")))
    {
        return Some(Query::new(Kind::Clock, When::Today));
    }

    if has("who are you") || has("what are you") || has("what can you do")
        || has("what do you do") || has("your name") || word("capabilities")
        || word("capable") || has("how do you work") || q.trim() == "help"
        || has("what should i ask") || has("what can i ask")
    {
        return Some(Query::new(Kind::About, When::Today));
    }

    // ── when ────────────────────────────────────────────────────────────────
    // Precedence: an explicit ISO date, then a named weekday, then relative
    // phrases. The weekday MUST come before the "past|previous|old" branch below
    // or "previous Tuesday" is swallowed by it and silently becomes 30 days.
    let toks = super::slots::tokens(&q);
    let today_local = chrono::Local::now().date_naive();
    let weekday = super::slots::weekday_day(&toks, today_local);
    // ISO first (unambiguous), then written-out dates. Both beat the relative
    // phrases below: "what happened on august 3rd" must not fall through to today.
    let explicit_day = find_iso_date(&q).or_else(|| find_natural_date(&q, today_local));
    let when = if let Some(d) = explicit_day {
        Some(When::Day(d))
    } else if let Some((d, _clamped)) = &weekday {
        Some(When::Day(d.clone()))
    } else if has("last hour") || has("past hour") {
        Some(When::Hours(1))
    } else if has("yesterday") {
        Some(When::Yesterday)
    } else if has("last night") || has("overnight") || has("tonight") {
        Some(When::LastNight)
    } else if has("this week") || has("past week") || has("last 7") {
        Some(When::Hours(168))
    } else if has("last 24") || has("24 hour") {
        Some(When::Hours(24))
    } else if has("today") || has("so far") {
        Some(When::Today)
    } else if has("old") || has("past") || has("previous") || has("earlier")
        || has("history") || has("archive") || word("ever") || has("all time")
    {
        // `has("ever")` was a substring test — it fired on *every*, *whatever*,
        // *however*. Whole-token only.
        // "show me the old footage" must not mean today. Asked for past footage
        // it reported on the current day, found nothing, and said so — technically
        // true and completely useless.
        Some(When::Hours(24 * 30))
    } else {
        None
    };

    // ── kind ────────────────────────────────────────────────────────────────
    //
    // Every bucket goes through `slots::hits`, which matches whole tokens or a
    // single typo — never substrings. The previous ladder was a raw keyword list
    // and eight of nine real messages missed it entirely, because real people
    // type "vlips", "shiw", "ebents", "whuch" and "modela".
    use super::slots as sl;
    // Phrases are matched against the tokens rejoined, not the raw text, so
    // hyphens and stray punctuation cannot hide a phrase ("book-marks").
    let normalised = toks.join(" ");
    let hit = |v: &[&str]| sl::hits(&toks, &normalised, v);

    let counting      = hit(sl::COUNT);
    let vehicle_words = hit(sl::VEHICLE);
    let sound_words   = hit(sl::SOUND);
    let person_words  = hit(sl::PERSON) || word("who");
    let media_words   = hit(sl::MEDIA);
    let event_words   = hit(sl::EVENTS);
    // `has("what ")` — with a trailing space — was false for "what?", "whats",
    // and any message ENDING in "what".
    let listing = word("what") || word("whats") || has("which") || has("list");

    // A named, enrolled person outranks every generic shape: "when was Ranjith
    // last here" carries none of the keywords below, and is unmistakeably about
    // one person.
    // WHOLE-TOKEN match, minimum three characters. `q.contains(name)` with a
    // 2-char floor made someone enrolled as "Al" match *alarm*, *always* and
    // *personal* — and it would make a "Rose" turn every question about roses
    // into a person lookup.
    let named = people.iter().find(|n| {
        let n = n.trim().to_lowercase();
        n.len() >= 3 && toks.contains(&n)
    });

    let kind = if let Some(_name) = named {
        Kind::Person
    } else if hit(sl::CONFIG) {
        // Above events: "what us going on whuch models are selected" is asking
        // about the SYSTEM, and deserves the specific answer.
        Kind::Models
    } else if hit(sl::BOOKMARK) {
        Kind::Bookmarks
    } else if hit(sl::STATUS) {
        Kind::Status
    } else if hit(sl::COVERAGE) {
        // Before Count: "how many days of footage" matches both, and it is a
        // question about the archive, not about today.
        Kind::Coverage
    } else if counting {
        Kind::Count
    } else if vehicle_words && listing {
        Kind::Vehicles
    } else if sound_words && listing {
        Kind::Sounds
    } else if media_words && hit(sl::DEICTIC) && !vehicle_words && !sound_words
        && !person_words && !event_words && when.is_none()
    {
        // "can you show them" — a reference to whatever was last shown, carrying
        // no subject of its own.
        Kind::Recall
    } else if vehicle_words || sound_words || person_words || event_words || media_words {
        Kind::Events
    } else if when.is_some() {
        // A bare date word IS a data question. "anything yesterday?" carries no
        // other vocabulary but is unmistakeably about the footage.
        //
        // Keyed on the DATE, deliberately — never on an interrogative. That is
        // what keeps "what is the user of this application" out of the event
        // query: a question word alone never resolves a kind.
        Kind::Events
    } else {
        return None;
    };

    let mut query = Query::new(kind, when.clone().unwrap_or(When::Today));
    // Media intent is read BEFORE the miss exit above could have discarded it. It
    // used to be computed 35 lines later, so "send me the clips" — a message that
    // is nothing but a media request — never reached it.
    query.wants_media = media_words;
    if kind == Kind::Bookmarks {
        query.bookmarked = true;
        // "my bookmarks" means all of them, not today's.
        if when.is_none() { query.when = When::Ever; }
    }
    if let Some(name) = named {
        query.who = Some(name.clone());
        // One person's pattern is a 30-day question, not a today question —
        // unless the user actually named a window.
        if when.is_none() { query.when = When::Hours(24 * 30); }
    }

    // ── slots ───────────────────────────────────────────────────────────────
    if let Some((&cam, _)) = cams.iter().find(|(_, name)| {
        !name.trim().is_empty() && q.contains(&name.to_lowercase())
    }) {
        query.cam = Some(cam);
    }
    // The SAME buckets the kind ladder uses. This was left on a stale raw list
    // when the ladder moved to `slots::hits`, and that list has no "audio" in it —
    // so "show the audio events" filtered nothing and came back full of motion and
    // person rows. One vocabulary, or they drift apart exactly like this.
    if query.what.is_none() {
        query.what = if vehicle_words {
            Some("vehicle")
        } else if sound_words {
            Some("audio")
        } else if person_words {
            Some("person")
        } else {
            None
        };
    }
    // Risk. "high" was missing entirely, so "show all the high risk events today"
    // filtered NOTHING, returned every low-risk event, and the model dutifully
    // called them high-risk. A severity filter that silently does nothing is
    // worse than one that isn't offered.
    query.risk = if hit(&["critical", "severe", "emergency"]) {
        Some("critical")
    // "risky" is deliberately absent: it is one edit from the NEUTRAL word
    // "risk", so listing it made "show the low risk events" resolve to high.
    } else if hit(&["high", "suspicious", "important", "serious", "urgent",
                    "danger", "dangerous", "concerning", "high risk"]) {
        Some("high")
    } else if hit(&["low", "normal", "routine", "quiet", "uneventful", "low risk"]) {
        Some("low")
    } else {
        None
    };

    // Negation — "show events with NO people", "without a car".
    //
    // Ignored before: the answer said "where no people were detected" and then
    // listed the person events. Inverting a filter is not a nuance, it is the
    // opposite of what was asked.
    let negated = has(" no ") || has("without") || has("not a ") || has("except")
        || has("other than") || has("apart from") || has("excluding")
        || normalised.starts_with("no ");
    if negated {
        query.exclude_what = query.what.take();
    }
    // ── description slots ───────────────────────────────────────────────────
    query.hours = super::slots::hour_window(&toks);

    // Tokens already claimed by something more specific must not be re-read as a
    // colour — an enrolled "Rose" or "Amber" above all.
    let mut claimed: std::collections::HashSet<String> = std::collections::HashSet::new();
    if let Some(n) = &query.who { claimed.insert(n.to_lowercase()); }
    let (outfit, garments) = super::slots::outfit_slots(&toks, &claimed);

    // A vehicle question with a garment in it is a mis-parse, not a refinement:
    // cars do not wear jackets. Drop the outfit and let `describe()` say so.
    if query.what == Some("vehicle") {
        query.outfit.clear();
    } else {
        query.outfit = outfit;
        // Describing what someone was WEARING is describing a person.
        if !query.outfit.is_empty() && query.what.is_none() { query.what = Some("person"); }
    }
    for (_, c) in &query.outfit { claimed.insert((*c).to_string()); }
    for g in &garments { claimed.insert(g.clone()); }

    // Free text is only worth carrying on an explicit search; on a plain "what
    // happened today" every leftover noun would narrow the answer to nothing.
    if query.kind == Kind::Events && (has("find") || has("search") || has("looking for")) {
        query.kind = Kind::Search;
        query.keywords = super::slots::keywords(&toks, &claimed);
    }
    for g in garments {
        if !query.keywords.contains(&g) { query.keywords.push(g); }
    }

    Some(query)
}

/// First `YYYY-MM-DD` in the text, if it is a real calendar date.
/// Parse a written-out date: "August 3rd", "aug 3", "3 August", "Aug 3 2026",
/// "the 3rd of this month", "on the 3rd".
///
/// `find_iso_date` below only ever matched `YYYY-MM-DD`, so every one of these
/// resolved to nothing and the question silently fell through to *today*. Asked
/// "what happened on august 3rd" the agent retrieved today's rows, found none,
/// and answered "Nothing recorded today, 2026-08-13" — the right answer to a
/// question nobody asked. (The model itself drafted the correct sentence; the
/// grounding check then rejected it, correctly, because Aug 3 appeared nowhere
/// in the evidence it had been handed. The gate was fine. The window was wrong.)
///
/// An NVR only ever holds the PAST, so a date that would land in the future is
/// read as the most recent one that already happened: "december 25" asked in
/// August means last December, not four months from now.
fn find_natural_date(q: &str, today: chrono::NaiveDate) -> Option<String> {
    use chrono::Datelike;
    let toks = super::slots::tokens(q);

    // A day number, with or without an ordinal suffix ("3", "3rd", "23rd").
    let day_of = |s: &str| -> Option<u32> {
        let digits: String = s.chars().take_while(|c| c.is_ascii_digit()).collect();
        if digits.is_empty() || digits.len() > 2 { return None; }
        let rest = &s[digits.len()..];
        if !(rest.is_empty() || matches!(rest, "st" | "nd" | "rd" | "th")) { return None; }
        match digits.parse::<u32>() { Ok(d) if (1..=31).contains(&d) => Some(d), _ => None }
    };
    let month_of = |s: &str| -> Option<u32> {
        if s.len() < 3 { return None; }
        MONTHS.iter().position(|m| s.starts_with(m)).map(|i| i as u32 + 1)
    };
    let year_of = |s: &str| -> Option<i32> {
        if s.len() != 4 || !s.bytes().all(|c| c.is_ascii_digit()) { return None; }
        match s.parse::<i32>() { Ok(y) if (2000..2100).contains(&y) => Some(y), _ => None }
    };

    // Build the date, rolling back a year rather than pointing at the future.
    let build = |y: Option<i32>, m: u32, d: u32| -> Option<String> {
        let year = y.unwrap_or_else(|| today.year());
        let made = chrono::NaiveDate::from_ymd_opt(year, m, d)?;
        let made = if y.is_none() && made > today {
            chrono::NaiveDate::from_ymd_opt(year - 1, m, d)?
        } else { made };
        Some(made.format("%Y-%m-%d").to_string())
    };

    // ── month + day, in either order, adjacent or one filler token apart
    //    ("august 3rd", "3 august", "3rd of august", "aug 3 2026") ──────────
    for (i, tok) in toks.iter().enumerate() {
        let Some(m) = month_of(tok) else { continue };
        for j in [i + 1, i + 2, i.wrapping_sub(1), i.wrapping_sub(2)] {
            let Some(cand) = toks.get(j) else { continue };
            // Only step over a filler word, never over another number.
            if j == i + 2 && !matches!(toks.get(i + 1).map(String::as_str), Some("the" | "of")) { continue; }
            if j == i.wrapping_sub(2) && !matches!(toks.get(i.wrapping_sub(1)).map(String::as_str), Some("the" | "of")) { continue; }
            let Some(d) = day_of(cand) else { continue };
            // A year may sit on either side of the day.
            let y = toks.get(j + 1).and_then(|s| year_of(s))
                .or_else(|| toks.get(i + 1).and_then(|s| year_of(s)))
                .or_else(|| toks.get(i.wrapping_sub(1)).and_then(|s| year_of(s)));
            if let Some(iso) = build(y, m, d) { return Some(iso); }
        }
    }

    // ── "the 3rd of this month" / "on the 3rd" — no month named ────────────
    //
    // Requires the ordinal form ("3rd", not "3") so a bare number in "last 3
    // hours" or "camera 3" can never be read as a day.
    let this_month = q.contains("this month") || q.contains("of the month");
    for (i, tok) in toks.iter().enumerate() {
        let ordinal = tok.len() > 2 && matches!(&tok[tok.len() - 2..], "st" | "nd" | "rd" | "th");
        if !ordinal { continue; }
        let Some(d) = day_of(tok) else { continue };
        let preceded = i > 0 && matches!(toks[i - 1].as_str(), "the" | "on");
        if !(this_month || preceded) { continue; }
        let made = chrono::NaiveDate::from_ymd_opt(today.year(), today.month(), d)
            // That day hasn't arrived this month yet → the last one that did.
            .filter(|dt| *dt <= today)
            .or_else(|| {
                let (y, m) = if today.month() == 1 { (today.year() - 1, 12) } else { (today.year(), today.month() - 1) };
                chrono::NaiveDate::from_ymd_opt(y, m, d)
            })?;
        return Some(made.format("%Y-%m-%d").to_string());
    }
    None
}

fn find_iso_date(q: &str) -> Option<String> {
    let b = q.as_bytes();
    for i in 0..b.len().saturating_sub(9) {
        let s = &q[i..i + 10];
        if s.as_bytes().iter().enumerate().all(|(j, c)| match j {
            4 | 7 => *c == b'-',
            _ => c.is_ascii_digit(),
        }) && chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d").is_ok()
        {
            return Some(s.to_string());
        }
    }
    None
}

// ─── Retrieval ───────────────────────────────────────────────────────────────

/// The ids of the events this query matches, newest first.
///
/// These are the exact rows the cards will show. `extra` is a LITERAL predicate
/// chosen by the caller — no user text reaches it — and everything else comes
/// from `where_sql`, so every slot the question carried (camera, person,
/// category, risk, hours, outfit, keywords) is honoured. The old path rebuilt a
/// lossy filter string instead and routinely showed a different window.
async fn ids_for(db: &sqlx::SqlitePool, q: &Query, extra: &str, limit: i64) -> Vec<String> {
    let (pred, binds) = where_sql(q);
    let sql = format!(
        "SELECT id FROM motion_events WHERE {pred}{extra}          ORDER BY started_at DESC LIMIT {}", limit.clamp(1, 40));
    let mut qy = sqlx::query_scalar::<_, String>(&sql);
    for b in &binds { qy = qy.bind(b); }
    qy.fetch_all(db).await.unwrap_or_default()
}

/// Run the query and describe what came back.
pub(super) async fn retrieve(
    state: &Arc<AppState>,
    settings: &crate::Settings,
    q: &Query,
) -> Evidence {
    let db = &state.db;
    let names = camera_names(db).await;
    let span = q.when.label();
    let (pred, binds) = where_sql(q);

    match q.kind {
        // Answered from the system clock and the config. No model involved, so
        // there is nothing to hallucinate — see `Kind::Clock` / `Kind::About`.
        Kind::Clock => {
            let now = chrono::Local::now();
            Evidence::whole(
                format!("It's {} on {}, UTC{}.",
                    now.format("%H:%M"), now.format("%A %-d %B %Y"), now.format("%:z")),
                Vec::new(), span)
        }

        Kind::Greeting => Evidence::whole(
            "Hello. Ask me what happened today, on a particular day, or last night — \
             or about a person, a vehicle or a sound, and I'll pull the footage."
                .to_string(),
            Vec::new(), span),

        // What is actually configured. Answered from `Settings`, never by the
        // model: asked "which vision model is running" it invented "the standard
        // video processing model", which is not a thing that exists.
        Kind::Models => {
            let on_device = !super::llm::provider_supports_vision(settings);
            let off_or = |v: &str| {
                let v = v.trim();
                if v.is_empty() || v == "off" { "off".to_string() } else { v.to_string() }
            };
            Evidence::whole("Here's what's running.".to_string(), vec![
                format!("Chat: {}.", if on_device {
                    format!("{} — on-device, inside the app",
                        super::local_llm::tier_label(&settings.local_llm_tier))
                } else {
                    format!("{} via {}", off_or(&settings.vision_model), settings.ai_provider)
                }),
                format!("Scene descriptions: {}.", if on_device && settings.local_llm_tier != "vision" {
                    "none — this engine is text-only".to_string()
                } else if on_device {
                    "on-device vision".to_string()
                } else { off_or(&settings.vision_model) }),
                format!("Object detection: YOLO26 {}.", settings.yolo_variant),
                format!("Face recognition: {}.", off_or(&settings.face_model)),
                format!("Clip search: {}.", off_or(&settings.search_model)),
                format!("Number plates: ALPR, {} region.", settings.alpr_region),
                format!("Audio detection: {}.",
                    if settings.audio_detection { "on" } else { "off" }),
            ], span)
        }

        // Handled before retrieval — see `answer`. Nothing to fetch.
        Kind::Recall => Evidence::whole(String::new(), Vec::new(), span),

        // `answer()` short-circuits About before it gets here, because the
        // capability manifest is returned verbatim rather than phrased by the
        // model. This arm exists for exhaustiveness and delegates to the SAME
        // function, so if that short-circuit is ever removed the answer stays
        // right instead of quietly becoming a stub.
        Kind::About => Evidence::whole(
            capabilities(state, settings, &names).await, Vec::new(), span),

        // How much footage there IS. Deliberately ignores `q.when` — the whole
        // point of the question is the extent of the archive, so scoping it to
        // today (which is what happens when this falls through to Count) answers
        // a different question and answers it "nothing".
        Kind::Coverage => {
            let row: Option<(i64, i64, Option<String>, Option<String>)> = sqlx::query_as(
                "SELECT COUNT(*), COUNT(DISTINCT date(started_at,'localtime')),
                        MIN(started_at), MAX(started_at) FROM motion_events"
            ).fetch_optional(db).await.ok().flatten();
            let Some((total, days, first, last)) = row.filter(|r| r.0 > 0) else {
                return Evidence::whole(
                    "There are no recorded events at all yet.".to_string(),
                    Vec::new(), span);
            };
            let d = |s: &Option<String>| s.as_deref()
                .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                .map(|t| t.with_timezone(&chrono::Local).format("%A %-d %B").to_string())
                .unwrap_or_else(|| "unknown".into());

            // Per-day breakdown, newest first — "which days" wants the list, and
            // "how many days" is answered by the headline above it.
            let per_day: Vec<(String, i64)> = sqlx::query_as(
                "SELECT date(started_at,'localtime'), COUNT(*) FROM motion_events
                  GROUP BY 1 ORDER BY 1 DESC LIMIT 14"
            ).fetch_all(db).await.unwrap_or_default();

            Evidence::whole(
                format!("{total} recorded event(s) across {days} day(s), from {} to {}. \
                         Recordings are kept for {} days.",
                    d(&first), d(&last), settings.nvr_retain_days),
                per_day.into_iter()
                    .map(|(day, n)| format!("{day} · {n} event(s)"))
                    .collect(),
                span)
        }

        Kind::Status => {
            let ctx = super::chat::build_situation_ctx(db, None).await;
            let lines: Vec<String> =
                ctx.lines().map(str::to_string).filter(|l| !l.is_empty()).collect();
            Evidence {
                headline: format!("System status as of {}.", super::chat::now_local_str()),
                total: lines.len(),
                lines,
                // Today's events, so "what is going on" comes with the footage.
                ids: ids_for(db, q, "", row_budget(settings) as i64).await,
                span,
                sampled: true,
            }
        }

        Kind::Count => {
            let sql = format!("SELECT COUNT(*) FROM motion_events WHERE {pred}");
            let mut qy = sqlx::query_scalar::<_, i64>(&sql);
            for b in &binds { qy = qy.bind(b); }
            let n = qy.fetch_one(db).await.unwrap_or(0);
            let subject = match q.what {
                Some("person")  => "person events",
                Some("vehicle") => "vehicle events",
                Some("audio")   => "sound events",
                Some("animal")  => "animal events",
                _ => "events",
            };
            let where_ = q.cam.map(|c| format!(" on {}", cam_label(&names, c))).unwrap_or_default();
            let mut ev = Evidence::whole(format!("{n} {subject}{where_} {span}."), Vec::new(), span);
            // The prose states the TRUE total; the cards are honestly the newest 12.
            ev.ids = ids_for(db, q, "", 12).await;
            ev
        }

        Kind::Person => {
            let who = q.who.clone().unwrap_or_default();
            person_evidence(db, &who, &names, span).await
        }

        Kind::Vehicles => {
            let sql = format!(
                "SELECT recognized_plate, COUNT(*), MAX(started_at), MAX(cam_id)
                   FROM motion_events
                  WHERE {pred} AND recognized_plate IS NOT NULL AND recognized_plate <> ''
                  GROUP BY recognized_plate ORDER BY COUNT(*) DESC LIMIT 20");
            let mut qy = sqlx::query_as::<_, (String, i64, String, i64)>(&sql);
            for b in &binds { qy = qy.bind(b); }
            let rows = qy.fetch_all(db).await.unwrap_or_default();
            if rows.is_empty() {
                // No READABLE PLATE is not the same as no vehicle — ALPR needs a
                // clear, close, well-lit plate and abstains constantly. Answering
                // "no vehicles today" when three cars drove past is a false
                // negative on the actual question, so fall back to the vehicles
                // the detector saw and say why there are no plates.
                let ids = ids_for(db, q, "", 12).await;
                let mut vq = q.clone();
                vq.what = Some("vehicle");
                let seen = stats_for(db, &vq).await.total;
                return if seen == 0 {
                    Evidence::empty(span, "involving a vehicle")
                } else {
                    Evidence {
                        headline: format!(
                            "{seen} vehicle event(s) {span}, but none with a plate clear enough to read."),
                        lines: Vec::new(), ids: ids.clone(), span, sampled: true, total: ids.len(),
                    }
                };
            }
            let lines = rows.iter().map(|(plate, n, last, cam)| format!(
                "{plate} — seen {n}×, last {} on {}",
                super::chat::rel_time(last), cam_label(&names, *cam))).collect::<Vec<_>>();
            // The prose is ordered by how often each plate appeared; the cards are
            // the footage behind it, newest first. Each line carries its own
            // "last seen", so the two orders don't mislead.
            let ids = ids_for(db, q,
                " AND recognized_plate IS NOT NULL AND recognized_plate <> ''", 12).await;
            Evidence {
                headline: format!("{} distinct plate(s) {span}:", rows.len()),
                total: lines.len(), lines, ids, span, sampled: true,
            }
        }

        Kind::Sounds => {
            let sql = format!(
                "SELECT COALESCE(NULLIF(dominant_label,''),'sound'), COUNT(*), MAX(started_at)
                   FROM motion_events
                  WHERE {pred} AND event_category = 'audio'
                  GROUP BY 1 ORDER BY 2 DESC LIMIT 10");
            let mut qy = sqlx::query_as::<_, (String, i64, String)>(&sql);
            for b in &binds { qy = qy.bind(b); }
            let rows = qy.fetch_all(db).await.unwrap_or_default();
            if rows.is_empty() { return Evidence::empty(span, "heard"); }
            let total: i64 = rows.iter().map(|(_, n, _)| n).sum();
            let lines = rows.iter().map(|(label, n, last)| format!(
                "{label} — {n}×, last {}", super::chat::rel_time(last))).collect::<Vec<_>>();
            let ids = ids_for(db, q, " AND event_category = 'audio'", 12).await;
            Evidence {
                headline: format!("{total} sound event(s) {span}:"),
                total: lines.len(), lines, ids, span, sampled: true,
            }
        }

        // Events, Search and Bookmarks share a shape — Bookmarks is a WHERE
        // clause on the same query, not a different kind of question.
        Kind::Events | Kind::Search | Kind::Bookmarks => {
            let sql = format!(
                "SELECT id, started_at, cam_id, COALESCE(event_category,''),
                        COALESCE(dominant_label,''), COALESCE(sub_label,''),
                        duration_secs, ai_summary
                   FROM motion_events
                  WHERE {pred}
                  ORDER BY started_at DESC LIMIT {}",
                q.limit.max(row_budget(settings) as i64).clamp(1, 60));
            #[allow(clippy::type_complexity)]
            let mut qy = sqlx::query_as::<_, (String, String, i64, String, String, String,
                                              Option<f64>, Option<String>)>(&sql);
            for b in &binds { qy = qy.bind(b); }
            let rows = qy.fetch_all(db).await.unwrap_or_default();
            if rows.is_empty() {
                let what = match q.what {
                    Some("person")  => "involving a person",
                    Some("vehicle") => "involving a vehicle",
                    Some("audio")   => "heard",
                    _ => "recorded",
                };
                return Evidence::empty(span, what);
            }

            let mut ids = Vec::with_capacity(rows.len());
            let mut lines = Vec::with_capacity(rows.len());
            // A window that can span more than one day must SAY the day on every
            // row. Rows used to carry a bare "%H:%M", so a question about older
            // footage handed the model twelve clock times and no dates — while
            // the prompt asked it to state which date it was covering. It had
            // nothing to read, so it invented: observed live producing three
            // events dated 2026-08-06/07/08, two of them in the future, for rows
            // that were all Aug 03. Give it the date and the guessing stops.
            let multiday = q.when.spans_multiple_days();
            for (id, started, cam, cat, label, sub, dur, summary) in &rows {
                ids.push(id.clone());
                let time = chrono::DateTime::parse_from_rfc3339(started)
                    .map(|t| {
                        let t = t.with_timezone(&chrono::Local);
                        if multiday { t.format("%b %d %H:%M").to_string() }
                        else        { t.format("%H:%M").to_string() }
                    })
                    .unwrap_or_else(|_| started.clone());
                // Most specific label available: a recognised name beats a class.
                let subject = if !sub.is_empty() { sub.clone() }
                    else if !label.is_empty() { label.clone() }
                    else if !cat.is_empty() { cat.clone() }
                    else { "motion".to_string() };
                let secs = dur.map(|d| format!(", {d:.0}s")).unwrap_or_default();
                let note = summary.as_deref()
                    .map(super::memory::extract_summary_text)
                    // Mark the cut. A note sliced at 90 characters mid-sentence
                    // reads to the model as a complete thought that simply ends,
                    // and it will summarise the fragment as though that were all
                    // there was.
                    .map(|s| {
                        let cut: String = s.chars().take(90).collect();
                        if cut.chars().count() < s.chars().count() { cut + "\u{2026}" } else { cut }
                    })
                    .filter(|s| !s.trim().is_empty())
                    .map(|s| format!(" — {s}"))
                    .unwrap_or_default();
                lines.push(format!("{time} · {} · {subject}{secs}{note}",
                    cam_label(&names, *cam)));
            }
            let where_ = q.cam.map(|c| format!(" on {}", cam_label(&names, c))).unwrap_or_default();

            // The facts a summary is made of, counted HERE rather than left to
            // the model. Asking a 1.2B model to tally twelve rows is asking for a
            // wrong number stated confidently; asking it to write a sentence
            // around numbers it was given is what it is good at.
            let st = stats_for(db, q).await;
            let named = named_in(db, q).await;
            let mut notes: Vec<String> = Vec::new();
            if st.people > 0   { notes.push(format!("{} involving a person", st.people)); }
            if st.vehicles > 0 { notes.push(format!("{} a vehicle", st.vehicles)); }
            if st.sounds > 0   { notes.push(format!("{} a sound", st.sounds)); }
            if !named.is_empty() { notes.push(format!("recognised: {}", named.join(", "))); }
            if st.longest >= 120.0 {
                notes.push(format!("longest ran {:.0} minutes", st.longest / 60.0));
            }
            // What connects this window to the rest of the week. Counted over a
            // WIDER window than the question, which is the whole point — it is
            // the difference between "a van was here" and "the third time that
            // van has been here".
            notes.extend(recurring_in(db, q).await);

            // The ends of the window ACTUALLY covered, from MIN/MAX over every
            // match — the span label is what was asked for, which is not the
            // same thing. Dated when the range crosses days, because "between
            // 19:36 and 22:47" over a month is a nonsense a reader cannot catch.
            let ends = match (&st.first, &st.last) {
                (Some(f), Some(l)) if f != l => {
                    let fmt = |s: &str| chrono::DateTime::parse_from_rfc3339(s)
                        .map(|t| {
                            let t = t.with_timezone(&chrono::Local);
                            if multiday { t.format("%b %d %H:%M").to_string() }
                            else        { t.format("%H:%M").to_string() }
                        })
                        .unwrap_or_default();
                    format!(", between {} and {}", fmt(f), fmt(l))
                }
                _ => String::new(),
            };
            let detail = if notes.is_empty() { String::new() } else { format!(" — {}", notes.join(", ")) };

            // A failed count is not a count of zero. Say so, rather than
            // reporting an empty archive the archive never confirmed.
            if st.failed && rows.is_empty() {
                return Evidence {
                    headline: format!(
                        "I couldn't read the event archive just now, so I can't tell you what happened {span}. \
                         This is a fault on my side, not an empty camera — try again in a moment."),
                    total: 0, lines: Vec::new(), ids: Vec::new(), span, sampled: false,
                };
            }

            let total = (st.total as usize).max(rows.len());
            let shown = if total > rows.len() {
                format!(" (the {} most recent are listed)", rows.len())
            } else { String::new() };

            Evidence {
                headline: format!("{total} event(s){where_} {span}{ends}{detail}{shown}."),
                total, lines, ids, span, sampled: true,
            }
        }
    }
}

/// One person's activity, aggregated.
///
/// Carries the small-sample guard that `tools::person_activity` lacked: below
/// three sightings a "usual hour" is noise, and stating it as a pattern is the
/// agent inventing a routine out of two data points.
async fn person_evidence(
    db: &sqlx::SqlitePool,
    who: &str,
    names: &BTreeMap<i64, String>,
    span: String,
) -> Evidence {
    // `event_id` comes along free — `face_sightings` has carried it all along, and
    // it is what lets a person answer attach the actual clips.
    let rows: Vec<(String, i64, Option<String>)> = sqlx::query_as(
        "SELECT seen_at, camera_id, event_id FROM face_sightings
          WHERE person_name = ? COLLATE NOCASE AND seen_at > datetime('now','-30 days')
          ORDER BY seen_at DESC LIMIT 500"
    ).bind(who).fetch_all(db).await.unwrap_or_default();

    if rows.is_empty() {
        return Evidence {
            headline: format!(
                "No sightings of {who} in the last 30 days. \
                 (If that name isn't what they're enrolled under, the People tab has the list.)"),
            lines: Vec::new(), ids: Vec::new(), span, sampled: true, total: 0,
        };
    }

    use chrono::Timelike as _;
    let mut days = std::collections::BTreeSet::new();
    let mut cams = std::collections::BTreeSet::new();
    let mut hours = [0usize; 24];
    for (at, cam, _) in &rows {
        if let Ok(t) = chrono::DateTime::parse_from_rfc3339(at) {
            let l = t.with_timezone(&chrono::Local);
            days.insert(l.format("%Y-%m-%d").to_string());
            hours[l.hour() as usize] += 1;
        }
        cams.insert(*cam);
    }
    let peak = if rows.len() >= 3 {
        hours.iter().enumerate().max_by_key(|(_, &n)| n)
            .filter(|(_, &n)| n > 0)
            .map(|(h, _)| format!(" Usually around {h:02}:00."))
            .unwrap_or_default()
    } else {
        String::new() // two sightings is not a routine
    };
    let cam_list = cams.iter().map(|c| cam_label(names, *c)).collect::<Vec<_>>().join(", ");

    // COUNT(*), not rows.len(). The query is `LIMIT 500`, so a frequently-seen
    // person came back as exactly "500 sightings" — the cap reported as a fact,
    // and a suspiciously round one. Observed live on "find ranjith".
    let sightings: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM face_sightings
          WHERE person_name = ? COLLATE NOCASE AND seen_at > datetime('now','-30 days')")
        .bind(who).fetch_one(db).await.unwrap_or(rows.len() as i64);
    // Days are counted the same way, or "500 across 3 days" understates it too.
    let day_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(DISTINCT date(seen_at,'localtime')) FROM face_sightings
          WHERE person_name = ? COLLATE NOCASE AND seen_at > datetime('now','-30 days')")
        .bind(who).fetch_one(db).await.unwrap_or(days.len() as i64);

    Evidence {
        headline: format!(
            "{who}: {sightings} sighting(s) across {day_count} day(s) in the last 30 days. \
             Last seen {}.{peak} Cameras: {cam_list}.",
            super::chat::rel_time(&rows[0].0)),
        lines: rows.iter().take(8).map(|(at, cam, _)| format!(
            "{} · {}", super::chat::rel_time(at), cam_label(names, *cam))).collect(),
        ids: {
            let mut seen: Vec<String> = Vec::new();
            for (_, _, eid) in &rows {
                if let Some(e) = eid.as_deref().filter(|e| !e.is_empty()) {
                    if !seen.iter().any(|s| s == e) { seen.push(e.to_string()); }
                }
                if seen.len() == 12 { break; }
            }
            seen
        },
        span,
        sampled: true,
        // Sightings, not events — this answer counts appearances of one person.
        total: rows.len(),
    }
}

// ─── The answer ──────────────────────────────────────────────────────────────

/// Answer a question from the database, using the model only to phrase the result.
///
/// Returns `None` when the question isn't a data question at all ("who are you?",
/// "thanks") — the caller then uses the conversational path.
pub(super) async fn answer(
    state: &Arc<AppState>,
    settings: &crate::Settings,
    question: &str,
    // Recent conversation as prompt text (see `super::chat::recent_turns`). Empty
    // for a cold first turn. This path used to receive nothing at all, so every
    // follow-up — "and yesterday?", "what about camera 2" — resolved against a
    // blank slate.
    history_ctx: &str,
    mode: super::chat::Mode,
) -> Option<String> {
    let cams = camera_names(&state.db).await;
    let people: Vec<String> = sqlx::query_scalar("SELECT name FROM known_persons")
        .fetch_all(&state.db).await.unwrap_or_default();

    // Brief answers a PERIOD, not a question, so an unresolvable message is not
    // a miss — "brief me" carries no slots at all and still has a right answer.
    // Resolve it as an events query over the window the user named, or today.
    let resolved = pre_resolve(question, &cams, &people).or_else(|| {
        // Today, unconditionally: this branch only runs when `pre_resolve` found
        // no slots at all, which means the message named no window either. A
        // "brief me on last night" DOES resolve, and keeps its own window.
        (mode == super::chat::Mode::Brief)
            .then(|| Query::new(Kind::Events, When::Today))
    });

    // `pre_resolve` is string matching, so it misses any phrasing it wasn't
    // written for — "what is going on", "work with me as an investigator". Those
    // used to fall through to open conversation and come back generic.
    //
    // So: ask the model to ROUTE, not to answer. One call, output constrained to
    // a known `Kind`, and the answer that follows is still built by Rust and
    // still checked by `grounded()`. This is the one job the small model is
    // documented to be good at — the same comment that forbids it from judging
    // risk credits it with 81% tool use — and a bad classification degrades to a
    // wrong-but-grounded answer, never to a fabrication.
    let resolved = match resolved {
        Some(q) => Some(q),
        None => route_with_model(settings, question).await,
    };

    let Some(q) = resolved else {
        // Short messages used to get a canned "I'm not sure what you're asking"
        // instead of a model call, because a small model handed "what" plus a
        // previous answer reissues that answer verbatim. That was a symptom of
        // this path running with NO history: the model had the last reply and no
        // sense that a new question was being asked. History now reaches both
        // paths, so "shorter", "explain why", "and the garage" are ordinary
        // follow-ups and go to the conversational path like anything else.
        tracing::debug!(question, "guardian: pre-resolver miss — conversational path");
        return None;
    };
    tracing::debug!(?q, "guardian: resolved");

    // "show me those" — hand back the SAME events, without a query or a model call.
    //
    // Read here rather than in `pre_resolve` so that stays a pure, synchronous
    // function with a table test. Ten minutes is long enough to finish reading a
    // paragraph and ask, short enough that yesterday's dangling "those" cannot
    // resurface as today's answer.
    let mut q = q;

    // ── What each mode changes ──────────────────────────────────────────────
    //
    // All three read the same tables through the same query; they differ in how
    // wide they look and how the finding is written. Nothing here grants a mode
    // any capability another does not have — Guardian stays monitor-only.
    match mode {
        super::chat::Mode::Investigate => {
            // Sweep: take the full row budget, and if the user named no window,
            // search the whole archive rather than assuming today. "Someone
            // stole my phone" is not a question about the last six hours.
            q.limit = row_budget(settings) as i64;
            if q.when == When::Today && !mentions_a_time(question) {
                q.when = When::Hours(720);
            }
            step(state, "Investigating — widening to the full archive");
        }
        super::chat::Mode::Brief => {
            q.limit = row_budget(settings) as i64;
            step(state, "Building a brief");
        }
        super::chat::Mode::Ask => {}
    }

    if q.kind == Kind::Recall {
        const RECALL_TTL: std::time::Duration = std::time::Duration::from_secs(600);
        let fresh = state.last_shown.read().await.clone()
            .filter(|(at, ids, _)| at.elapsed() < RECALL_TTL && !ids.is_empty());
        if let Some((_, ids, span)) = fresh {
            step(state, &format!("Showing the same {} event(s) again", ids.len()));
            return Some(format!("Here they are again — the {} event(s) from {span}.\n{}",
                ids.len(),
                media_tag(&ids, false, row_budget(settings)).unwrap_or_default()));
        }
        // Nothing to recall (fresh start, or expired). Degrade to today rather
        // than answering "I don't know what you mean".
        step(state, "Nothing to show from earlier — showing today instead");
        q.kind = Kind::Events;
        q.wants_media = true;
    }

    // Narrate the work. The user asked to see what the agent is doing rather
    // than a bare "Thinking…" — and a step list is also the honest record of
    // where the answer came from, which is what makes it checkable.
    step(state, &format!("Reading the question — {}", describe(&q)));

    // A described search gets the refinement ladder; the fixed shapes (clock,
    // greeting, status) have nothing to relax.
    // "What can you do?" is answered from live configuration, NOT by the model.
    //
    // It used to go through the phrasing pass with "describe what you can do in
    // your own words, in two or three sentences" — so asking three times gave
    // three different vague paragraphs, none of them a list, and none of them
    // aware of what is actually switched on. A capability manifest is not a
    // conversation: it should be complete, specific to this install, and the
    // same every time it is asked.
    if q.kind == Kind::About {
        step(state, "Reading what's configured");
        return Some(capabilities(state, settings, &cams).await);
    }

    let (q, given_up) = if matches!(q.kind, Kind::Events | Kind::Search | Kind::Count)
        && (!q.outfit.is_empty() || q.hours.is_some() || !q.keywords.is_empty())
    {
        let (relaxed, gave) = widen_until_found(&state.db, &q).await;
        for g in &gave { step(state, &format!("Nothing matched — relaxed {g}")); }
        (relaxed, gave)
    } else {
        (q, Vec::new())
    };

    let mut ev = retrieve(state, settings, &q).await;
    step(state, &if ev.is_empty() {
        format!("Searched {} — nothing found", ev.span)
    } else {
        format!("Searched {} — {} result(s)", ev.span, ev.lines.len().max(1))
    });

    // Investigate is a SWEEP, not one wider query: follow the leads the first
    // pass turned up until a pass stops adding anything.
    if mode == super::chat::Mode::Investigate && !ev.is_empty() {
        ev = investigate(state, settings, &q, ev).await;
    }

    // What we gave up goes into the HEADLINE, not just the model's prompt — so
    // the deterministic fallback carries it too. The model must never be the
    // only thing that knows the search was widened.
    if !given_up.is_empty() {
        ev.headline = format!("{} (I couldn't match {}, so I relaxed it.)",
            ev.headline.trim_end_matches('.'), given_up.join(", then "));
    }

    step(state, "Writing the answer");

    // The model's entire job: turn the findings into a natural reply.
    //
    // EVERY kind goes through here, including "hi" and "what's the time". A
    // hardcoded reply is reliably correct and reliably robotic, and the user's
    // objection to that is right — so the model writes the words, and the facts
    // it needs are placed in front of it rather than left to memory. Short
    // prompt, no tool catalogue, no memory dump: the old design buried a small
    // model under 3,300 tokens of instructions and got its own prompt read back.
    let tone = match q.kind {
        Kind::Greeting => "The user is just saying hello. Greet them back warmly in one \
                           short sentence and invite them to ask about the cameras. \
                           Do not report any events.",
        Kind::About    => "Describe what you can do, in your own words, in two or three \
                           sentences. Use the capability notes below — do not list them \
                           verbatim, and do not invent capabilities you were not given.",
        // The old wording here forbade mentioning models at all — written to stop
        // the model reciting its own prompt, back when the prompt WAS the input.
        // It is not any more, and the ban made "which models are running"
        // unanswerable. Same protection, aimed at the right failure.
        Kind::Models   => "Say what is running, plainly, in a sentence or two. Use ONLY \
                           the lines below — never name a model that is not there.",
        Kind::Clock    => "Tell them the time and date conversationally. Use EXACTLY the \
                           time and date given below — never any other.",
        // SUMMARISE, don't transcribe.
        //
        // Given a dozen display-ready rows and told to "use the findings", the
        // model reprinted them — "23:23: Motion detected. 23:21: Motion detected."
        // twelve times over. That reads like a database dump because it is one,
        // and the user can already see every row as a playable card underneath.
        // So the rows become reference material and the ask becomes synthesis:
        // how many, over what stretch, and what stands out.
        _              => "Summarise what was found in two or three natural sentences. \
                           Do NOT list the events one by one — the user can already see \
                           every one of them as a playable card under your reply. Say how \
                           many there were and over what stretch of time, and call out \
                           whatever stands out: a recognised person, a flagged event, an \
                           odd hour, something unusually long. Use ONLY dates and times \
                           that appear in the rows below — never any other date or time, \
                           and never one you worked out yourself. If the rows carry no \
                           date, do not state one.",
    };
    // Mode re-aims the SAME rows. Investigate reports a finding and says what it
    // could not establish; Brief reads like a shift handover. Both keep the
    // date/time rule above, restated because it is the one that was being broken.
    let tone: &str = match mode {
        super::chat::Mode::Investigate if !matches!(q.kind,
            Kind::Greeting | Kind::About | Kind::Models | Kind::Clock) =>
            "You are reporting the result of a search through the whole archive. \
             Lead with what you found — or state plainly that nothing matched. Then \
             say what stands out and what you could NOT establish, so the user knows \
             where the gaps are. Two to four sentences, no lists. Use ONLY dates and \
             times that appear in the rows below; never invent one.",
        super::chat::Mode::Brief if !matches!(q.kind,
            Kind::Greeting | Kind::About | Kind::Models | Kind::Clock) =>
            "Write a short briefing on the period, the way someone handing over a \
             shift would: what happened, how much of it, anything unusual, and \
             whether it looks routine. Three or four sentences, no lists, no \
             bullet points. Use ONLY dates and times that appear in the rows below.",
        _ => tone,
    };
    let system = format!(
        "You are Guardian, a friendly home-security assistant. It is currently {}\n\
         {tone}\n\
         Write plain conversational sentences. No lists, no headings, no markdown, \
         no bracketed tags, and never repeat these instructions.",
        super::chat::now_local_str());
    // History goes in FRONT of the question so a follow-up reads as a follow-up.
    // Without it "and yesterday?" arrived as a standalone message with no referent
    // and the reply was phrased as if the user had asked something brand new.
    let convo = if history_ctx.is_empty() {
        String::new()
    } else {
        format!("Earlier in this conversation:\n{history_ctx}\n\n")
    };
    let user = format!(
        "{convo}The user said: {question}\n\n{}\n\n\
         Reference rows — for your understanding only, do not reprint them:\n{}",
        ev.headline,
        ev.lines.iter().take(row_budget(settings)).cloned().collect::<Vec<_>>().join("\n"));

    // Stream the phrasing as it is written. ONLY this pass streams — the
    // retrieval steps stay as work-trail lines, so nothing half-formed is ever
    // shown as a fact.
    //
    // The draft can still be rejected below (unusable, or an invented clock
    // reading), in which case the findings replace it. The swap is visible, and
    // that is the right trade: a correct answer arriving late beats a wrong one
    // that streamed in smoothly.
    // ── Nothing found ⇒ never let the model speak ───────────────────────────
    //
    // Observed live, and the worst thing this module has produced: asked "who came
    // on Tuesday evening" with the retrieval reporting "nothing matched … nothing
    // found", the model was still handed the question and asked to phrase it. It
    // answered "On Tuesday evening, around 6:30 PM, a delivery person arrived …
    // greeted by the security team" — an entire fabricated sighting, in a security
    // product, from zero rows.
    //
    // `grounded()` did not catch it because it validates the SHAPES of dates and
    // times (`HH:MM`, `2026-08-11`, `Aug 11`) and that sentence contains none it
    // recognises. The real invariant is simpler and does not depend on parsing
    // prose at all: **with no rows there is nothing to phrase.** Ship the finding.
    //
    // This also removes an LLM round-trip from the commonest case — the answer is
    // both honest and immediate.
    if ev.is_empty() && q.kind.asserts_footage() {
        tracing::info!(kind = ?q.kind, "guardian: no rows — answering from the finding, not the model");
        let mut text = ev.fallback_answer();
        if let Some(tag) = media_tag(&ev.ids, q.wants_media, row_budget(settings)) {
            text.push('\n');
            text.push_str(&tag);
        }
        return Some(text);
    }

    let app = state.app_handle.clone();
    let on_token: Option<super::local_llm::OnToken> = Some(Box::new(move |piece: &str| {
        use tauri::Emitter;
        let _ = app.emit("guardian:chat-token",
            serde_json::json!({ "content": piece, "done": false }));
    }));
    let drafted = super::llm::call_llm_streaming(settings, &system, &user, on_token).await;
    {
        // Close the stream whatever happened, so the UI stops showing a caret.
        use tauri::Emitter;
        let _ = state.app_handle.emit("guardian:chat-token",
            serde_json::json!({ "content": "", "done": true }));
    }
    let mut text = match drafted {
        // `grounded` is the price of letting it speak freely: a natural reply is
        // welcome, an invented clock reading is not.
        Ok(t) if super::llm::usable(&t) && grounded(&t, &q, &given_up, &ev) => t.trim().to_string(),
        Ok(t) => {
            tracing::warn!(len = t.len(), draft = %t.chars().take(200).collect::<String>(),
                "guardian: draft rejected — using the findings directly");
            ev.fallback_answer()
        }
        Err(e) => {
            // Not an error bubble: we already know the answer, the model was only
            // going to word it. This is the whole point of retrieving first.
            tracing::warn!("guardian: phrasing failed ({e}) — using the findings directly");
            ev.fallback_answer()
        }
    };

    // Tags are appended by US, from the evidence that produced the answer — never
    // emitted by the model. Truncated ids, invented filters and tags no surface
    // executes all become impossible here rather than merely unlikely.
    //
    // Attached whenever there ARE ids, not only when the user said "show": an
    // answer about footage should arrive WITH the footage. `wants_media` now only
    // chooses the shape — one clip inline, or the card strip.
    if let Some(tag) = media_tag(&ev.ids, q.wants_media, row_budget(settings)) {
        text.push('\n');
        text.push_str(&tag);
    }

    // Remember what we put in front of them, so "show me those" means these.
    if !ev.ids.is_empty() {
        let keep: Vec<String> = ev.ids.iter().take(SHOW_CAP).cloned().collect();
        *state.last_shown.write().await =
            Some((std::time::Instant::now(), keep, ev.span.clone()));
    }
    Some(text)
}

/// Does this clause DENY rather than assert?
///
/// Used to tell "there was a person in the red jacket" — a false identification
/// when the colour was never matched — apart from "I couldn't find anyone in
/// red", which is the truthful report of exactly that failure. Substring
/// matching is enough here: the cost of a miss is one rejected draft, and the
/// cost of over-matching is a fabricated sighting, so the list stays short and
/// unambiguous.
pub(super) fn negated(clause: &str) -> bool {
    const DENIALS: &[&str] = &[
        "n't", " no ", " not ", " none", " nobody", " nothing", " never",
        " without", " couldn", " didn", " wasn", " weren", " unable",
        "no one", "failed to", "unmatched", "unconfirmed",
    ];
    let padded = format!(" {clause} ");
    DENIALS.iter().any(|m| padded.contains(m))
}

/// Does this draft contradict a fact we handed the model?
///
/// The model is allowed to phrase things freely — that is the point — but not to
/// invent the one class of fact it has no way to know. Asked the time, it
/// answered "19:01 (local time, UTC-05:00)": a number copied from an unrelated
/// line, in a timezone that isn't the user's. So for clock answers every `HH:MM`
/// and every four-digit year in the reply must match reality.
///
/// Deliberately narrow. Over-checking would reject good prose for saying "this
/// morning", and a false reject costs a natural answer.
fn grounded(draft: &str, q: &Query, relaxed: &[String], ev: &Evidence) -> bool {
    // A slot the search GAVE UP on must not be asserted as found.
    //
    // Observed live: asked to find "the person in the red jacket", the ladder
    // relaxed the colour (no clothing data matched) and the model still answered
    // "there was a person in the red jacket around 19:10". The relaxation was in
    // the prompt; it wrote past it. A dropped colour that comes back as a
    // confirmed sighting is the worst failure this module can produce — it is a
    // false identification.
    //
    // But only when it is ASSERTED. This used to reject on the mere PRESENCE of
    // the colour anywhere in the draft, which threw away the most honest answer
    // available — "I couldn't find anyone in red on Aug 3, but there were 12
    // person events" — and left a template that never mentions red at all,
    // reading as though the model ignored half the question. A colour inside a
    // negated clause is a report about the search, not a false identification.
    if !relaxed.is_empty() {
        let d = draft.to_lowercase();
        let mut colours: Vec<String> = q.outfit.iter().map(|(_, c)| c.to_lowercase()).collect();
        // The colours were already removed from `q.outfit` by the time we get
        // here, so also check what the relaxation notes named.
        for note in relaxed {
            for word in note.to_lowercase().split_whitespace() {
                if super::slots::is_colour(word) { colours.push(word.to_string()); }
            }
        }
        for sentence in d.split(|c| c == '.' || c == '!' || c == '?' || c == ';') {
            if !colours.iter().any(|c| sentence.contains(c.as_str())) { continue; }
            if !negated(sentence) { return false; }
        }
    }
    // ── Every date and time asserted must come from the rows ────────────────
    //
    // This used to be `if q.kind != Kind::Clock { return true; }` — the check ran
    // for clock questions ONLY, so every footage answer was unverified. That is
    // how "the first occurred on 2026-08-06 at 01:49, the second on 2026-08-07 at
    // 03:12, and the third on 2026-08-08 at 05:45" reached the user: three events
    // that do not exist, two of them dated in the future.
    //
    // The rows are the only evidence there is. If a timestamp is not in them it
    // was invented, and an invented sighting at an invented time is the worst
    // thing this module can emit.
    // Clock answers are exempt from the row check and get the STRICTER rule
    // below instead: their entire content is the present moment, which appears
    // in no row by definition, so "must be in the evidence" is the wrong test —
    // "must be now" is the right one.
    if q.kind != Kind::Clock {
        return cites_only_known(draft, ev);
    }
    let now = chrono::Local::now();
    let hhmm = now.format("%H:%M").to_string();
    let year = now.format("%Y").to_string();
    let b = draft.as_bytes();

    for i in 0..b.len() {
        // A four-digit run that looks like a year must BE this year.
        if b[i].is_ascii_digit()
            && (i == 0 || !b[i - 1].is_ascii_digit())
            && i + 4 <= b.len()
            && b[i..i + 4].iter().all(u8::is_ascii_digit)
            && (i + 4 == b.len() || !b[i + 4].is_ascii_digit())
        {
            let n = &draft[i..i + 4];
            if (2000..2100).contains(&n.parse::<i32>().unwrap_or(0)) && n != year {
                return false;
            }
        }
        // Any HH:MM must be the current time. Within-the-minute drift between
        // building the prompt and checking the reply is tolerated.
        if i + 5 <= b.len() && b[i + 2] == b':'
            && b[i..i + 2].iter().all(u8::is_ascii_digit)
            && b[i + 3..i + 5].iter().all(u8::is_ascii_digit)
            && (i == 0 || !b[i - 1].is_ascii_digit())
        {
            let t = &draft[i..i + 5];
            if t != hhmm && t != (now - chrono::Duration::minutes(1)).format("%H:%M").to_string() {
                return false;
            }
        }
    }
    true
}

/// What this install can actually do, right now.
///
/// Built from live configuration rather than described by the model, so it is
/// complete, honest about what is switched OFF, and identical every time it is
/// asked. Ends with what Guardian may not do, because a monitoring assistant
/// that quietly cannot act is worse than one that says so.
async fn capabilities(
    state: &Arc<AppState>,
    settings: &crate::Settings,
    cams: &BTreeMap<i64, String>,
) -> String {
    let off = |v: &str| v.is_empty() || v == "off";
    let mut out = String::from("Here's what I can do on this system right now.\n\n");

    out.push_str("**Answering questions about your footage**\n");
    out.push_str("• What happened today, last night, on a date, or in the last N hours\n");
    out.push_str("• How many events, on which camera, at what time of day\n");
    out.push_str("• Search by keyword — a name, a plate, \"package\", \"dog\", \"glass breaking\"\n");
    out.push_str("• Show or send the actual clips, as playable cards\n");
    out.push_str("• Your saved bookmarks\n\n");

    out.push_str("**Recognition that's switched on**\n");
    let people: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM known_persons")
        .fetch_one(&state.db).await.unwrap_or(0);
    out.push_str(&if off(&settings.face_model) {
        "• Faces: off — I can see that a person was there, but not who\n".to_string()
    } else {
        format!("• Faces: on ({}) — {people} {} enrolled\n",
            settings.face_model,
            if people == 1 { "person" } else { "people" })
    });
    out.push_str(&if off(&settings.alpr_region) {
        "• Number plates: off\n".to_string()
    } else {
        format!("• Number plates: on ({} region)\n", settings.alpr_region)
    });
    let sounds = settings.audio_listen.split(',').filter(|s| !s.trim().is_empty()).count();
    out.push_str(&if sounds == 0 {
        "• Sound: off\n".to_string()
    } else {
        format!("• Sound: listening for {sounds} kinds ({})\n",
            settings.audio_listen.replace(',', ", "))
    });
    out.push_str(&if off(&settings.search_model) {
        "• Meaning-based search: off — keywords only\n".to_string()
    } else {
        format!("• Meaning-based search: on ({}) — I can find \"a person in red\" \
                 even when nobody wrote that down\n", settings.search_model)
    });
    out.push_str(&format!("• Object detection: {}\n\n", settings.yolo_variant));

    let cam_list = if cams.is_empty() { "none set up yet".to_string() }
        else { cams.values().cloned().collect::<Vec<_>>().join(", ") };
    out.push_str(&format!("**Cameras**\n• {cam_list}\n\n"));

    out.push_str("**How to ask**\n");
    out.push_str("• Ask — quick answers, one pass\n");
    out.push_str("• Investigate — I sweep the whole archive and report what I find, \
                  and what I couldn't establish\n");
    out.push_str("• Brief — a summary of a period rather than an answer to a question\n\n");

    // Alert rules are the one thing the agent genuinely CAN change, so they are
    // advertised as a capability instead of denied here. The old copy read
    // "Change any setting ... I watch and report only", which contradicted both
    // the alert tools it is handed and what it does when asked to set one up.
    out.push_str("• Set up an alert — tell me what to watch for and I'll add the rule\n\n");

    out.push_str("**What I can't do**\n");
    out.push_str("• Change settings, or turn a camera on or off — beyond alert rules, \
                  I watch and report only\n");
    if settings.telegram_bot_token.trim().is_empty() {
        out.push_str("• Message you outside this window — Telegram isn't set up yet\n");
    }
    if !settings.agent_enabled {
        out.push_str("• Watch on my own — background monitoring is currently off, \
                      so I only answer when asked\n");
    }
    out
}

/// Did the user name a time at all?
///
/// `pre_resolve` defaults an unqualified question to today, which is right when
/// answering and wrong when investigating: "someone stole my phone" carries no
/// window, and searching only the last six hours for it is how you find nothing.
/// This tells the two cases apart without re-parsing the question.
fn mentions_a_time(question: &str) -> bool {
    let q = question.to_lowercase();
    // Phrases need a substring test; single words need a TOKEN test.
    //
    // This was one flat `contains` over both, so "someday", "everyday",
    // "birthday" and "holiday" all matched "day" and told an Investigate sweep
    // that the user had named a window. The sweep then stayed pinned to today
    // instead of searching the 720-hour archive it was asked for.
    const PHRASES: &[&str] = &["last night", "this morning", "this week", "this month"];
    if PHRASES.iter().any(|p| q.contains(p)) { return true; }
    const WORDS: &[&str] = &[
        "today", "tonight", "yesterday", "hour", "hours", "minute", "minutes",
        "day", "days", "week", "weeks", "month", "months", "morning",
        "afternoon", "evening", "night", "ago", "monday", "tuesday", "wednesday",
        "thursday", "friday", "saturday", "sunday"];
    q.split(|c: char| !c.is_alphanumeric()).any(|t| WORDS.contains(&t))
}

/// How many event rows a prompt may carry, given the model behind it.
///
/// This used to be a literal 12 everywhere, which meant an 8k on-device model
/// and a 200k cloud model were both shown twelve events — the small one strained
/// and the large one wasted. A row is ~90 characters, and roughly a fifth of the
/// window is a sane share to spend on evidence with the rest left for the system
/// prompt, history, memory and the reply.
///
/// Floored at 12 so behaviour never regresses below what shipped, and capped at
/// 40 because past that the model is summarising a spreadsheet — and the counts,
/// span and outliers in the headline are computed in Rust anyway, so extra rows
/// buy detail, not correctness.
///
/// A SIXTEENTH of the window, not a fifth. The binding constraint is not context
/// space, it is what the model can actually hold in its head: the on-device tier
/// is a 1.2B, and handing it sixty rows to tally is how you get a confident wrong
/// number. This gives it ~22 and a large cloud model the full 40.
pub(super) fn row_budget(settings: &crate::Settings) -> usize {
    const ROW_CHARS: usize = 90;
    let evidence_tokens = super::llm::context_limit(settings) / 16;
    let rows = evidence_tokens * 4 / ROW_CHARS;
    rows.clamp(12, 40)
}

const MONTHS: [&str; 12] = ["jan", "feb", "mar", "apr", "may", "jun",
                            "jul", "aug", "sep", "oct", "nov", "dec"];

/// Full month names, used to confirm that a three-letter hit is actually a month
/// and not the opening of an ordinary word. `"separate 4"` used to be read as
/// `sep 04`, `"marked 3"` as `mar 03`, `"decided 5"` as `dec 05` — each one
/// rejecting a perfectly good summary for citing a date it never mentioned.
const MONTHS_FULL: [&str; 12] = ["january", "february", "march", "april", "may", "june",
                                 "july", "august", "september", "october", "november", "december"];

/// Does the draft cite only clock times and dates that appear in the evidence?
///
/// Deliberately checks the *shapes a reader would act on* — an exact `HH:MM`,
/// `Aug 03`, `2026-08-07` — against the headline, the span and the rows. Prose
/// like "late last night" is untouched: vagueness is not a false claim, and
/// rejecting it would push the model toward the specificity it gets wrong. For
/// the same reason a rounded time is allowed through — see the branch below.
fn cites_only_known(draft: &str, ev: &Evidence) -> bool {
    // Everything the model was actually shown, normalised once.
    let mut hay = format!("{} {}", ev.headline, ev.span).to_lowercase();
    for l in &ev.lines { hay.push(' '); hay.push_str(&l.to_lowercase()); }

    // NOTE: today's date is deliberately NOT added here. It is citable only when
    // the evidence itself mentions it — a "today" query's headline already reads
    // "today, 2026-08-06", so the honest path is open. Blanket-allowing it would
    // re-admit the original bug in a subtler form: rows from Aug 03 summarised as
    // "12 events on Aug 06" is a false report even though the date is real.
    let d = draft.to_lowercase();
    let b = d.as_bytes();
    let year_now = chrono::Local::now().format("%Y").to_string();

    let digits_at = |i: usize, n: usize| {
        i + n <= b.len() && b[i..i + n].iter().all(u8::is_ascii_digit)
    };
    let boundary_before = |i: usize| i == 0 || !b[i - 1].is_ascii_digit();

    for i in 0..b.len() {
        // HH:MM — the shape a person reads as "this happened at".
        //
        // A ROUNDED time is exempt. The prompt asks for a summary "over what
        // stretch of time", and the natural way to write one is "activity ran
        // from about 19:00 to 23:00" — quarter-hour figures that summarise real
        // rows without copying any single one. Rejecting those threw away the
        // exact answer the prompt asked for. Arbitrary minutes are still checked,
        // which is what catches the original bug: a clock reading of "19:01"
        // lifted from an unrelated line.
        if digits_at(i, 2) && i + 5 <= b.len() && b[i + 2] == b':' && digits_at(i + 3, 2)
            && boundary_before(i) && (i + 5 == b.len() || !b[i + 5].is_ascii_digit())
            && !hay.contains(&d[i..i + 5])
        {
            let rounded = matches!(&d[i + 3..i + 5], "00" | "15" | "30" | "45");
            if !rounded { return false; }
        }
        // YYYY-MM-DD, and bare four-digit years.
        if digits_at(i, 4) && boundary_before(i) {
            let iso_len = 10;
            if i + iso_len <= b.len() && b[i + 4] == b'-' && b[i + 7] == b'-'
                && digits_at(i + 5, 2) && digits_at(i + 8, 2)
            {
                if !hay.contains(&d[i..i + iso_len]) { return false; }
            } else if i + 4 == b.len() || !b[i + 4].is_ascii_digit() {
                let y = &d[i..i + 4];
                if (2000..2100).contains(&y.parse::<i32>().unwrap_or(0))
                    && y != year_now && !hay.contains(y)
                {
                    return false;
                }
            }
        }
    }

    // "Aug 03" / "August 3" — how a row actually reads once it carries a date,
    // and the exact form the model got wrong (Aug 06 for rows dated Aug 03).
    for (idx, mon) in MONTHS.iter().enumerate() {
        let mut from = 0;
        while let Some(rel) = d[from..].find(mon) {
            let at = from + rel;
            from = at + mon.len();
            // The hit must BE a month word, not the opening of another word.
            //
            // This was a raw substring scan, so "separate 4" matched `sep` and
            // then demanded the rows contain "sep 04"; "marked 3" and "decided 5"
            // did the same. Each one threw away a correct summary. Require a word
            // boundary before the match, and require the whole alphabetic run to
            // be a prefix of the real month name.
            if at > 0 && b[at - 1].is_ascii_alphabetic() { continue; }
            let mut j = from;
            while j < b.len() && b[j].is_ascii_alphabetic() { j += 1; }
            if !MONTHS_FULL[idx].starts_with(&d[at..j]) { continue; }
            while j < b.len() && (b[j] == b' ' || b[j] == b'.') { j += 1; }
            if !(j < b.len() && b[j].is_ascii_digit()) { continue; }
            let mut k = j;
            while k < b.len() && b[k].is_ascii_digit() { k += 1; }
            let Ok(day) = d[j..k].parse::<u32>() else { continue };
            if !(1..=31).contains(&day) { continue; }
            // Accept either rendering the rows might use.
            let padded = format!("{mon} {day:02}");
            let bare   = format!("{mon} {day}");
            let iso    = format!("-{:02}-{day:02}", idx + 1);
            if !hay.contains(&padded) && !hay.contains(&bare) && !hay.contains(&iso) {
                return false;
            }
        }
    }
    true
}

/// Aggregates over EVERY matching event, not the page that was fetched.
///
/// This exists because mixing the two produced a headline that was false in a
/// way no reader could detect: "269 event(s) the last 720 hours, between 19:36
/// and 22:47 — 16 involving a person". The 269 was real; the span and every
/// sub-count came from the 22 rows that fit, which all happened to be one
/// evening. A total next to sampled detail reads as detail about the total.
///
/// So the numbers now come from SQL over the same predicate the rows do.
#[derive(Default)]
struct Stats {
    total:    i64,
    people:   i64,
    vehicles: i64,
    sounds:   i64,
    longest:  f64,
    first:    Option<String>,
    last:     Option<String>,
    /// The query itself FAILED — as opposed to matching nothing.
    ///
    /// These are not the same answer and must never read the same. A broken
    /// database used to come back as a confident "nothing recorded today",
    /// which is the same class of falsehood as an invented event: an assertion
    /// about the archive that the archive never made.
    failed:   bool,
}

async fn stats_for(db: &sqlx::SqlitePool, q: &Query) -> Stats {
    let (pred, binds) = where_sql(q);
    let sql = format!(
        "SELECT COUNT(*),
                COALESCE(SUM(CASE WHEN event_category='audio' THEN 1 ELSE 0 END),0),
                COALESCE(SUM(CASE WHEN event_category='person'
                                    OR dominant_label='person' THEN 1 ELSE 0 END),0),
                COALESCE(SUM(CASE WHEN event_category='vehicle'
                                    OR dominant_label IN ('car','truck','bus','motorcycle','bicycle')
                              THEN 1 ELSE 0 END),0),
                -- 0.0, not 0: SQLite types the literal, and an INTEGER zero here
                -- fails to decode as f64, which made the whole query error and
                -- `stats_for` return zeros. That silently told the widening
                -- ladder nothing ever matched.
                COALESCE(MAX(duration_secs), 0.0),
                MIN(started_at), MAX(started_at)
           FROM motion_events WHERE {pred}");
    let mut qy = sqlx::query_as::<_, (i64, i64, i64, i64, f64, Option<String>, Option<String>)>(&sql);
    for b in &binds { qy = qy.bind(b); }
    match qy.fetch_one(db).await {
        Ok((total, sounds, people, vehicles, longest, first, last)) =>
            Stats { total, people, vehicles, sounds, longest, first, last, failed: false },
        Err(e) => {
            tracing::warn!("stats_for: {e}");
            Stats { failed: true, ..Default::default() }
        }
    }
}

/// Which of the named entities in this match set are REGULARS.
///
/// "The third time that van has been here this week" is a *count over a wider
/// window than the question asked about*, which is why it never appeared in an
/// answer before: every other number in `Evidence` describes the match set.
///
/// Counted in SQL, deliberately. A 1.2B model asked to tally rows returns a
/// confident wrong number, and in a security log a wrong count is worse than no
/// count. Reported as sightings-across-days rather than an average hour: an
/// average of 23:00 and 01:00 is noon, and nobody would catch it.
async fn recurring_in(db: &sqlx::SqlitePool, q: &Query) -> Vec<String> {
    /// Below this a "pattern" is a coincidence. Two visits is not a routine —
    /// the same floor `person_evidence` already applies to peak hours.
    const REGULAR: i64 = 3;

    let mut out: Vec<String> = Vec::new();
    for (col, who) in [("sub_label", named_in(db, q).await),
                       ("recognized_plate", plates_in(db, q).await)] {
        for name in who.into_iter().take(3) {
            // The column is a LITERAL from the loop above; the value is bound.
            let sql = format!(
                "SELECT COUNT(*), COUNT(DISTINCT date(started_at,'localtime'))
                   FROM motion_events
                  WHERE {col} = ? COLLATE NOCASE
                    AND started_at > datetime('now','-7 days')");
            let row: Option<(i64, i64)> = sqlx::query_as(&sql)
                .bind(&name).fetch_optional(db).await.ok().flatten();
            let Some((seen, days)) = row else { continue };
            if seen < REGULAR { continue; }
            out.push(if days <= 1 {
                format!("{name} has been seen {seen} times today")
            } else {
                format!("{name} has been seen {seen} times across {days} days this week")
            });
        }
    }
    out
}

/// The distinct recognised plates in the whole match set.
async fn plates_in(db: &sqlx::SqlitePool, q: &Query) -> Vec<String> {
    let (pred, binds) = where_sql(q);
    let sql = format!(
        "SELECT DISTINCT recognized_plate FROM motion_events
          WHERE {pred} AND recognized_plate IS NOT NULL AND recognized_plate <> '' LIMIT 6");
    let mut qy = sqlx::query_scalar::<_, String>(&sql);
    for b in &binds { qy = qy.bind(b); }
    qy.fetch_all(db).await.unwrap_or_default()
}

/// The distinct recognised names in the whole match set — again, not just the page.
async fn named_in(db: &sqlx::SqlitePool, q: &Query) -> Vec<String> {
    let (pred, binds) = where_sql(q);
    let sql = format!(
        "SELECT DISTINCT sub_label FROM motion_events
          WHERE {pred} AND sub_label IS NOT NULL AND sub_label <> '' LIMIT 6");
    let mut qy = sqlx::query_scalar::<_, String>(&sql);
    for b in &binds { qy = qy.bind(b); }
    qy.fetch_all(db).await.unwrap_or_default()
}

/// Classify a question `pre_resolve` didn't recognise into a `Kind` + window.
///
/// The model ROUTES; it does not answer. Its entire output budget is one word
/// from a fixed list plus one window from another — so the worst it can do is
/// send a question down the wrong (but real) query, which `grounded()` then
/// checks against actual rows. Compare the alternative it replaces: handing the
/// question to open conversation, where a small model answers from its own
/// prompt.
///
/// Deliberately NOT given `Clock`, `About`, `Models`, `Recall` or `Greeting` —
/// those are answered from configuration or from state, and `pre_resolve`
/// already recognises them reliably. Routing exists for the shapes that vary:
/// what happened, how many, who, which vehicle, what sound.
async fn route_with_model(settings: &crate::Settings, question: &str) -> Option<Query> {
    // A one-word answer is cheap on any engine, but not free, and a fragment
    // this short carries nothing to route on. Returning None sends it to the
    // conversational path, which now has the conversation history to read it
    // against — "and yesterday?" is only meaningless in isolation.
    if question.split_whitespace().count() < 4 { return None; }

    let system = "You label a home-security question. Reply with exactly two words, \
                  lowercase, separated by a space: a TOPIC and a WINDOW. \
                  TOPIC is one of: events, count, person, vehicles, sounds, search, \
                  status, coverage, bookmarks. \
                  WINDOW is one of: today, yesterday, lastnight, week, ever. \
                  No punctuation, no explanation. If unsure, reply: events today";
    let user = format!("Question: {question}\nTopic and window?");

    let reply = super::llm::call_llm(settings, system, &user, None, false).await.ok()?;
    let lower = reply.trim().to_lowercase();
    let mut words = lower.split_whitespace();
    // Scan rather than index: a small model may answer "topic: events today".
    let kind = words.clone().find_map(|w| match w.trim_matches(|c: char| !c.is_alphabetic()) {
        "events"    => Some(Kind::Events),
        "count"     => Some(Kind::Count),
        "person"    => Some(Kind::Person),
        "vehicles"  => Some(Kind::Vehicles),
        "sounds"    => Some(Kind::Sounds),
        "search"    => Some(Kind::Search),
        "status"    => Some(Kind::Status),
        "coverage"  => Some(Kind::Coverage),
        "bookmarks" => Some(Kind::Bookmarks),
        _ => None,
    })?;
    let when = words.find_map(|w| match w.trim_matches(|c: char| !c.is_alphabetic()) {
        "today"     => Some(When::Today),
        "yesterday" => Some(When::Yesterday),
        "lastnight" => Some(When::LastNight),
        "week"      => Some(When::Hours(168)),
        "ever"      => Some(When::Ever),
        _ => None,
    }).unwrap_or(When::Today);

    // `Person` without a name has nothing to look up, and `Search` without terms
    // matches everything — neither is an answer, so fall through to conversation.
    if matches!(kind, Kind::Person | Kind::Search) { return None; }

    tracing::debug!(question, ?kind, ?when, "guardian: routed by model");
    let mut q = Query::new(kind, when);
    // Media intent is read from the QUESTION, not from the model — "show me"
    // is a word match, and asking a 1.2B model to also decide that is spending
    // reliability on something `slots::MEDIA` already gets right.
    let toks = super::slots::tokens(question);
    q.wants_media = super::slots::hits(&toks, &question.to_lowercase(), super::slots::MEDIA);
    if q.kind == Kind::Bookmarks { q.bookmarked = true; }
    Some(q)
}

/// Investigate: keep looking while each pass turns up events the last one didn't.
///
/// The first pass answers the question as asked. Each pass after it follows a
/// LEAD out of what was found — a recognised person gets their other sightings
/// across the same window — and the sweep stops the moment a pass adds nothing
/// new. This is what `Mode::Investigate` always claimed to do; before this it
/// was a wider row budget and a longer window, one query, no iteration.
///
/// Bounded by pass count AND wall clock: a busy archive must not be able to
/// make the chat sit there while the whole thing is walked.
async fn investigate(
    state: &Arc<AppState>,
    settings: &crate::Settings,
    q: &Query,
    mut ev: Evidence,
) -> Evidence {
    const MAX_PASSES: usize = 4;
    const DEADLINE: std::time::Duration = std::time::Duration::from_secs(45);
    let started = std::time::Instant::now();

    // Only follow leads we have not already been given. If the user asked about
    // one person, chasing that same person is not a second pass.
    let leads: Vec<String> = named_in(&state.db, q).await.into_iter()
        .filter(|n| q.who.as_deref().map(|w| !w.eq_ignore_ascii_case(n)).unwrap_or(true))
        .take(MAX_PASSES)
        .collect();
    if leads.is_empty() { return ev; }

    let mut seen: std::collections::HashSet<String> = ev.ids.iter().cloned().collect();
    let mut gained = 0usize;
    for name in leads {
        if started.elapsed() > DEADLINE {
            step(state, "Investigation hit its time limit — reporting what I have");
            break;
        }
        step(state, &format!("Following a lead — everything involving {name}"));
        let mut lead = q.clone();
        lead.who = Some(name.clone());
        lead.outfit.clear();       // the lead is the person, not what they wore once
        lead.keywords.clear();
        let found = retrieve(state, settings, &lead).await;

        let mut added = 0usize;
        for (i, id) in found.ids.iter().enumerate() {
            if !seen.insert(id.clone()) { continue; }
            ev.ids.push(id.clone());
            if let Some(line) = found.lines.get(i) { ev.lines.push(line.clone()); }
            added += 1;
        }
        // A pass that adds nothing ends that lead — that is the stopping rule.
        step(state, &if added == 0 {
            format!("{name} — nothing new")
        } else {
            format!("{name} — {added} more event(s)")
        });
        gained += added;
    }

    // The headline must describe what the sweep ACTUALLY covered: `grounded()`
    // checks the answer against headline + rows, so a total the headline does
    // not admit to would get the answer rejected.
    if gained > 0 {
        ev.total += gained;
        ev.headline = format!("{} (investigated further: {} event(s) in total)",
            ev.headline.trim_end_matches('.'), ev.total);
    }
    ev
}

/// Run the query, relaxing it a step at a time until something comes back.
///
/// Returns the evidence and **the list of what was given up to get it**. That
/// list is the point: "no red jacket on Tuesday" and "I have no idea what anyone
/// was wearing on Tuesday" are completely different answers, and a search that
/// quietly widens until it finds something is worse than one that finds nothing.
///
/// Terminates by construction — the rungs are walked forward once, and there are
/// a fixed number of them.
/// Takes the pool rather than `AppState` so it is testable, and so narration
/// stays the caller's job — this function decides, it does not talk.
async fn widen_until_found(
    db: &sqlx::SqlitePool,
    q: &Query,
) -> (Query, Vec<String>) {
    const MAX_RUNGS: usize = 3;
    let mut cur = q.clone();
    let mut given_up: Vec<String> = Vec::new();

    for _ in 0..MAX_RUNGS {
        if stats_for(db, &cur).await.total > 0 { break; }

        // Weakest slot first — ordered by how likely it is to be ABSENT rather
        // than false. `when` is deliberately last and never silent: an answer
        // about the wrong day is worse than no answer at all.
        if !cur.outfit.is_empty() {
            // With both bands, drop the BOTTOM first: the legs band abstains far
            // more often (occlusion, and dark trousers read as shadow).
            let unknown = count_unknown_outfit(db, &cur).await;
            if cur.outfit.len() > 1 {
                cur.outfit.retain(|(band, _)| *band == "top");
                given_up.push("what they were wearing below the waist".into());
            } else {
                let (_, colour) = cur.outfit.remove(0);
                given_up.push(match unknown {
                    0 => format!("the {colour} clothing"),
                    n => format!("the {colour} clothing ({n} of those events have no clothing recorded)"),
                });
            }
        } else if let Some((a, b)) = cur.hours {
            // Widen the window before abandoning it — "around 9" is a memory,
            // not a timestamp.
            let width = (b + 24 - a) % 24;
            if width <= 2 {
                cur.hours = Some(((a + 22) % 24, (b + 2) % 24));
                given_up.push("widened the time to a few hours either side".into());
            } else {
                cur.hours = None;
                given_up.push("the time of day".into());
            }
        } else if !cur.keywords.is_empty() {
            let dropped = cur.keywords.remove(cur.keywords.len() - 1);
            given_up.push(format!("the word \"{dropped}\""));
        } else if cur.cam.is_some() {
            cur.cam = None;
            given_up.push("the camera".into());
        } else if let When::Day(d) = cur.when.clone() {
            // Last resort, and only to the neighbouring days — never to "the
            // last month", which would answer a different question entirely.
            if let Ok(day) = chrono::NaiveDate::parse_from_str(&d, "%Y-%m-%d") {
                cur.when = When::Range(
                    (day - chrono::Duration::days(1)).format("%Y-%m-%d").to_string(),
                    (day + chrono::Duration::days(1)).format("%Y-%m-%d").to_string());
                given_up.push(format!("widened to the days either side of {d}"));
            } else { break; }
        } else {
            break; // nothing left to relax
        }
    }
    (cur, given_up)
}

/// How many events in this window have NO clothing recorded.
///
/// The difference between "nobody wore red" and "I couldn't see what anyone was
/// wearing". NULL means unknown, and saying so is the honest answer.
async fn count_unknown_outfit(db: &sqlx::SqlitePool, q: &Query) -> i64 {
    let mut bare = q.clone();
    bare.outfit.clear();
    let (pred, binds) = where_sql(&bare);
    let sql = format!("SELECT COUNT(*) FROM motion_events WHERE {pred} AND outfit IS NULL");
    let mut qy = sqlx::query_scalar::<_, i64>(&sql);
    for b in &binds { qy = qy.bind(b); }
    qy.fetch_one(db).await.unwrap_or(0)
}

/// Push one line into the chat's live work trail.
fn step(state: &Arc<AppState>, text: &str) {
    use tauri::Emitter;
    let _ = state.app_handle.emit("guardian:activity", serde_json::json!({ "text": text }));
}

/// Plain-English restatement of a resolved query, shown in the work trail so the
/// user can see how their words were understood — and catch it when they weren't.
fn describe(q: &Query) -> String {
    let subject = match q.kind {
        Kind::Clock    => return "the current date and time".into(),
        Kind::About    => return "what I can do".into(),
        Kind::Greeting => return "a greeting".into(),
        Kind::Models   => return "which models are running".into(),
        Kind::Recall   => return "the events I showed you a moment ago".into(),
        Kind::Status   => return "system and camera status".into(),
        Kind::Coverage => return "how much footage there is".into(),
        Kind::Bookmarks => "your saved events",
        Kind::Count    => "a count of events",
        Kind::Person   => "one person's activity",
        Kind::Vehicles => "vehicles and plates",
        Kind::Sounds   => "sounds",
        Kind::Events | Kind::Search => "events",
    };
    let mut out = format!("{subject} for {}", q.when.label());
    if let Some(who) = &q.who { out.push_str(&format!(", {who}")); }
    if let Some(what) = q.what { out.push_str(&format!(", {what} only")); }
    // Both of these were invisible in the work trail, which is where a
    // misreading is supposed to become obvious.
    if let Some(what) = q.exclude_what { out.push_str(&format!(", NO {what}")); }
    if let Some(r) = q.risk {
        out.push_str(match r {
            "critical" => ", critical only",
            "low"      => ", low risk only",
            _          => ", high risk only",
        });
    }
    if q.bookmarked { out.push_str(", saved only"); }
    if q.cam.is_some() { out.push_str(", one camera"); }
    out
}

/// Hard ceiling on cards in one answer, whatever the row budget says.
///
/// The strip is a horizontal scroller, so a large number is survivable, but past
/// this the answer stops being a reply and becomes a listing — that is Review's
/// job, and the headline now says how many there really were.
const SHOW_CAP: usize = 40;

/// The media tag for a set of event ids, if any.
///
/// One id AND an explicit ask ("send me the clip") → that clip inline. Otherwise
/// the card strip, addressed BY ID. The old code rebuilt a filter string from the
/// query, and `When::Day`, `When::Range` and the 30-day window all collapsed to
/// "last 24" — so the cards routinely covered a different period than the answer.
///
/// At most ONE `[SEND_CLIP]` is possible per turn, which is what keeps Telegram
/// from uploading a queue of videos: the strip is one message with buttons.
fn media_tag(ids: &[String], wants_media: bool, budget: usize) -> Option<String> {
    // Cards track the ROW BUDGET, so the strip is exactly the set of events the
    // model was shown. Letting them diverge would let the prose describe an
    // event with no card under it — the bug this invariant exists to prevent.
    let shown: Vec<&str> = ids.iter().take(budget.min(SHOW_CAP)).map(String::as_str).collect();
    match shown.len() {
        0 => None,
        1 if wants_media => Some(format!("[SEND_CLIP:{}]", shown[0])),
        _ => Some(format!("[SHOW_EVENTS:ids={}]", shown.join(","))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cams() -> BTreeMap<i64, String> {
        BTreeMap::from([(0, "Front Door".to_string()), (1, "Driveway".to_string())])
    }
    fn people() -> Vec<String> { vec!["Ranjith".into(), "Amma".into()] }

    fn resolve(q: &str) -> Query {
        pre_resolve(q, &cams(), &people()).unwrap_or_else(|| panic!("unresolved: {q}"))
    }

    #[test]
    fn resolves_the_questions_people_actually_ask() {
        let q = resolve("what happened today");
        assert_eq!(q.kind, Kind::Events);
        assert_eq!(q.when, When::Today);

        // No date word at all still means today — the prompt has always said so.
        assert_eq!(resolve("anything I should know about?").when, When::Today);

        // ── Empty evidence must never reach the model ────────────────────
        //
        // The live failure this pins: "who came on Tuesday evening" retrieved
        // NOTHING, the model was asked to phrase it anyway, and it reported a
        // delivery person arriving and being greeted by the security team. A
        // fabricated sighting in a security product.
        for k in [Kind::Events, Kind::Count, Kind::Person, Kind::Vehicles,
                  Kind::Sounds, Kind::Search, Kind::Bookmarks, Kind::Coverage] {
            assert!(k.asserts_footage(), "{k:?} claims footage — no rows must mean no answer");
        }
        // ...while the kinds answered from the clock and the config legitimately
        // carry no rows and must still be allowed to speak.
        for k in [Kind::Clock, Kind::About, Kind::Greeting, Kind::Models, Kind::Status] {
            assert!(!k.asserts_footage(), "{k:?} is answered from config, not footage");
        }
        // An empty Evidence must produce an honest finding, never silence.
        let empty = Evidence::empty("on Tuesday 11 August".into(), "recorded");
        let said = empty.fallback_answer();
        assert!(said.contains("Nothing"), "empty evidence must say so: {said}");
        assert!(said.contains("Tuesday 11 August"), "and must name the window asked about: {said}");

        // ── Written-out dates ────────────────────────────────────────────
        //
        // The bug these pin: only `YYYY-MM-DD` was ever recognised, so "what
        // happened on august 3rd" resolved to When::Today and the agent answered
        // "Nothing recorded today" to a question about a different day.
        let aug13 = chrono::NaiveDate::from_ymd_opt(2026, 8, 13).unwrap();
        for phrase in ["what happened on august 3rd", "what happened on aug 3",
                       "anything on 3 august", "events on the 3rd of august",
                       "what happened on august 3rd 2026"] {
            assert_eq!(find_natural_date(phrase, aug13).as_deref(), Some("2026-08-03"),
                       "should resolve to Aug 3: {phrase}");
        }
        // "the 3rd of this month" / "on the 3rd" — no month named.
        for phrase in ["what happened on the 3rd of this month", "anything on the 3rd"] {
            assert_eq!(find_natural_date(phrase, aug13).as_deref(), Some("2026-08-03"),
                       "bare ordinal should mean this month: {phrase}");
        }
        // An NVR holds only the past: a date not yet reached rolls back.
        assert_eq!(find_natural_date("what happened on december 25", aug13).as_deref(),
                   Some("2025-12-25"), "a future date must mean the last one that happened");
        assert_eq!(find_natural_date("anything on the 20th", aug13).as_deref(),
                   Some("2026-07-20"), "a day later than today means last month");
        // Bare numbers must NOT be read as days.
        for phrase in ["events in the last 3 hours", "what did camera 3 see", "last 24 hours"] {
            assert_eq!(find_natural_date(phrase, aug13), None, "must not parse a day: {phrase}");
        }
        // And the whole path end to end: the resolver must carry the day through.
        assert_eq!(resolve("what happened on august 3rd").when,
                   When::Day("2026-08-03".into()));

        let q = resolve("anyone at the front door last night");
        assert_eq!(q.when, When::LastNight);
        assert_eq!(q.cam, Some(0), "camera name must resolve to its slot");
        assert_eq!(q.what, Some("person"));

        let q = resolve("how many events yesterday");
        assert_eq!(q.kind, Kind::Count);
        assert_eq!(q.when, When::Yesterday);

        let q = resolve("what cars have you seen this week");
        assert_eq!(q.kind, Kind::Vehicles);
        assert_eq!(q.when, When::Hours(168));

        let q = resolve("show me today's events");
        assert!(q.wants_media, "'show me' must ask for cards, not prose");

        let q = resolve("what happened on 2026-07-14");
        assert_eq!(q.when, When::Day("2026-07-14".into()));
    }

    /// A named person beats the generic shapes — "when was Ranjith here" contains
    /// none of the event keywords.
    #[test]
    fn a_named_person_wins() {
        let q = resolve("when was Ranjith last here");
        assert_eq!(q.kind, Kind::Person);
        assert_eq!(q.who.as_deref(), Some("Ranjith"));
    }

    /// Not every message is a data question, and guessing is worse than saying
    /// so. Greetings and meta questions now DO resolve (to `Greeting`/`About`) —
    /// what must still fall through is genuine ambiguity.
    #[test]
    fn declines_when_the_intent_is_genuinely_unclear() {
        for q in ["why", "the blue one", "no, the other one",
                  // A lone interrogative fragment. Resolving it to "today's
                  // events" would be a guess dressed as an answer.
                  "wbat",
                  // The load-bearing one: the catch-all keys off a DATE, never off
                  // an interrogative, so questions about the app stay out of the
                  // event query. "user" is 4 chars and fuzzy-eligible, but none of
                  // its distance-1 neighbours is in any bucket.
                  "what is the user of this application",
                  "what is the point of all this"] {
            assert!(pre_resolve(q, &cams(), &people()).is_none(), "should not resolve: {q}");
        }
    }

    /// The nine messages from a real session. Eight of them used to resolve to
    /// nothing and fall through to a generic non-answer.
    #[test]
    fn the_messages_people_actually_typed() {
        let k = |q: &str| resolve(q).kind;
        assert_eq!(k("what is going on"), Kind::Events);
        assert_eq!(k("what is going on right now"), Kind::Events);
        assert_eq!(k("what us going on whuch models are selected"), Kind::Models);
        assert_eq!(k("which vision modela are running"), Kind::Models);
        assert_eq!(k("show ke thevrecent events"), Kind::Events);
        assert_eq!(k("what abkut audio ebents"), Kind::Sounds);
        assert_eq!(k("can you shiw them"), Kind::Recall);

        // "send me the clips" is nothing BUT a media request; `wants_media` used
        // to be computed after the only miss exit, so it never even got read.
        let q = resolve("send me yhe current vlips or video");
        assert_eq!(q.kind, Kind::Events);
        assert!(q.wants_media);
    }

    /// "show the audio events" came back full of MOTION and PERSON rows: the kind
    /// ladder moved to the shared vocabulary but the `what` slot was left on a
    /// stale list that had no "audio" in it, so nothing was filtered.
    #[test]
    fn the_subject_filter_uses_the_same_vocabulary_as_the_ladder() {
        assert_eq!(resolve("show the audio events").what, Some("audio"));
        assert_eq!(resolve("any sounds today").what, Some("audio"));
        assert_eq!(resolve("show the vehicles").what, Some("vehicle"));
        assert_eq!(resolve("anyone around").what, Some("person"));
        // …and the typo tolerance reaches the filter too, not just the kind.
        assert_eq!(resolve("what abkut audio ebents").what, Some("audio"));
    }

    /// People write it as two words, and neither half can be a single-word entry
    /// ("book" would swallow bookshelf). It fell through to the path that parrots
    /// the previous answer — which is how "the bookmarks were found in the audio
    /// events" happened.
    #[test]
    fn bookmarks_spelled_as_two_words() {
        for s in ["what about the book marks", "are there any book marks",
                  "show my book-marks"] {
            assert_eq!(resolve(s).kind, Kind::Bookmarks, "{s}");
        }
    }

    /// The worst failure available: the ladder gave up on "red", and the reply
    /// THE regression test. Observed live, asked to search older footage, the
    /// model answered:
    ///
    ///   "There were three events recorded in the old footage. The first occurred
    ///    on 2026-08-06 at 01:49, the second on 2026-08-07 at 03:12, and the third
    ///    on 2026-08-08 at 05:45."
    ///
    /// None of those events existed and two of the dates were in the FUTURE. It
    /// shipped because the date/time check ran for `Kind::Clock` only; every
    /// footage answer was unverified.
    #[test]
    fn an_invented_date_or_time_never_ships() {
        let q = Query::new(Kind::Events, When::Hours(720));
        let ev = rows_ev();

        for lie in [
            "The first occurred on 2026-08-06 at 01:49, the second on 2026-08-07 at 03:12.",
            "There were 12 events on Aug 06.",          // wrong day, rows are Aug 03
            "Someone was at the door at 03:12.",        // time absent from the rows
            "It all happened back in 2024.",            // year neither now nor cited
        ] {
            assert!(!grounded(lie, &q, &[], &ev), "should have been rejected: {lie}");
        }

        for ok in [
            "Twelve events between 22:12 and 22:47, the last at 22:47 on the front door.",
            "A person showed up at 19:01 on Aug 03.",
            "There was activity late in the evening.",  // vague is not a false claim
            "I found 12 events involving a person.",
        ] {
            assert!(grounded(ok, &q, &[], &ev), "should have been allowed: {ok}");
        }
    }

    /// The other half of the invented-date rule: prose the grader used to reject
    /// even though it asserted nothing false.
    ///
    /// Every one of these shipped as a five-row database dump instead, because a
    /// rejected draft is replaced by `fallback_answer()` — the exact
    /// event-by-event list the phrasing prompt forbids two lines above it. The
    /// grader was enforcing the opposite of what the prompt asked for.
    #[test]
    fn a_summary_is_not_an_invented_date() {
        let ev = rows_ev();

        for ok in [
            // Rounded times SUMMARISE the rows rather than copying one, which is
            // what "say over what stretch of time" asks for.
            "Activity ran from about 19:00 to 23:00.",
            "Things picked up around 22:30 and stayed busy.",
            // "sep"/"mar"/"dec" inside an ordinary word are not month citations.
            "Two separate 4-second clips, both on the drive.",
            "Nothing marked 3 of these as suspicious.",
            "It decided 5 minutes later that nothing was wrong.",
        ] {
            assert!(cites_only_known(ok, &ev), "should have been allowed: {ok}");
        }

        // ...and the real thing is still caught.
        for lie in [
            "Someone was at the door at 03:12.",   // arbitrary minutes, not in rows
            "There were 12 events on Aug 06.",     // real month word, wrong day
            "Two events on September 4.",          // spelled out, still wrong day
        ] {
            assert!(!cites_only_known(lie, &ev), "should have been rejected: {lie}");
        }
    }

    /// A relaxed colour may not be ASSERTED — but saying we could not find it is
    /// the most honest answer available, and the check used to reject that too.
    ///
    /// It tested for the PRESENCE of the colour anywhere in the draft, so the
    /// user asking about a red jacket got a template that never mentions red,
    /// which reads as the model ignoring half the question.
    #[test]
    fn a_relaxed_colour_may_still_be_reported_as_missing() {
        let mut q = Query::new(Kind::Search, When::Today);
        q.outfit = vec![("top", "red")];
        let gave_up = vec!["the red clothing (6 of those events have no clothing recorded)".to_string()];

        for ok in [
            "I couldn't find anyone in red on Aug 03, but there were 12 person events.",
            "No one in a red jacket turned up; 12 people were seen.",
            "Nothing matched the red clothing.",
        ] {
            assert!(grounded(ok, &q, &gave_up, &rows_ev()), "should have been allowed: {ok}");
        }

        // The false identification this check exists for is still rejected.
        assert!(!grounded("There was a person in the red jacket around 19:01.",
                          &q, &gave_up, &rows_ev()));
    }

    /// Rows must carry their date whenever the window can cross midnight —
    /// without it the model has nothing to cite and fills the gap itself, which
    /// is what produced the invented dates above.
    #[test]
    fn multiday_windows_put_the_date_on_every_row() {
        for wide in [When::Hours(720), When::Hours(48), When::Ever, When::LastNight,
                     When::Range("2026-08-01".into(), "2026-08-03".into())] {
            assert!(wide.spans_multiple_days(), "{wide:?} needs dated rows");
        }
        for narrow in [When::Today, When::Yesterday, When::Hours(6),
                       When::Day("2026-08-03".into()),
                       When::Range("2026-08-03".into(), "2026-08-03".into())] {
            assert!(!narrow.spans_multiple_days(), "{narrow:?} is one day; headline carries it");
        }
    }

    /// Evidence a draft may legitimately cite: two rows, one span, one headline.
    fn rows_ev() -> Evidence {
        Evidence::whole(
            "12 events involving a person between 22:12 and 22:47".to_string(),
            vec!["Aug 03 22:47 · Front Door · person, 1s".to_string(),
                 "Aug 03 19:01 · Drive · person, 8s".to_string()],
            "on Aug 03".to_string(),
        )
    }

    /// came back asserting a person in a red jacket. A dropped colour returning
    /// as a confirmed sighting is a false identification.
    #[test]
    fn a_relaxed_colour_may_not_be_asserted_as_found() {
        let mut q = Query::new(Kind::Search, When::Today);
        q.outfit = vec![("top", "red")];
        let gave_up = vec!["the red clothing (6 of those events have no clothing recorded)".to_string()];

        assert!(!grounded("There was a person in the red jacket around 19:10.", &q, &gave_up, &rows_ev()));
        // Honest phrasings are fine.
        assert!(grounded("I couldn't match the clothing, but 12 people were seen.", &q, &gave_up, &rows_ev()));
        // With nothing relaxed, the colour is a legitimate finding.
        assert!(grounded("A person in a red jacket at 19:01.", &q, &[], &rows_ev()));
    }

    /// "show all the high risk events today" returned every LOW-risk event and
    /// the model called them high-risk: "high" was in no vocabulary, so the
    /// filter silently did nothing.
    #[test]
    fn risk_words_actually_filter() {
        assert_eq!(resolve("show all the high risk events today").risk, Some("high"));
        assert_eq!(resolve("anything suspicious").risk, Some("high"));
        assert_eq!(resolve("any critical alerts").risk, Some("critical"));
        assert_eq!(resolve("show the low risk events").risk, Some("low"));
        assert_eq!(resolve("what happened today").risk, None);
    }

    /// "show events which no people" listed the person events. Inverting a filter
    /// is not a nuance — it is the opposite of what was asked.
    #[test]
    fn negation_inverts_the_subject() {
        let q = resolve("show events which no people");
        assert_eq!(q.exclude_what, Some("person"), "the subject must move to exclude");
        assert_eq!(q.what, None, "and must not ALSO be required");

        for s in ["events without a car", "show me anything except people",
                  "motion but not a person"] {
            assert!(resolve(s).exclude_what.is_some(), "{s}");
        }
        // A plain question is not a negation.
        assert!(resolve("show the people today").exclude_what.is_none());
    }

    /// The SQL-NULL trap: `NOT (col = 'person')` is NULL, not true, for an
    /// uncategorised row — so a naive negation drops exactly the events "no
    /// people" is asking for.
    #[tokio::test]
    async fn negation_keeps_uncategorised_events() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        crate::db::init_db(&pool).await.unwrap();
        for (id, cat) in [("p", Some("person")), ("v", Some("vehicle")), ("bare", None)] {
            sqlx::query("INSERT INTO motion_events(id, started_at, event_category)
                         VALUES(?, datetime('now'), ?)")
                .bind(id).bind(cat).execute(&pool).await.unwrap();
        }
        let mut q = Query::new(Kind::Events, When::Today);
        q.exclude_what = Some("person");
        let mut got = ids_for(&pool, &q, "", 12).await;
        got.sort();
        assert_eq!(got, vec!["bare".to_string(), "v".into()],
                   "an event with NO category is not a person event");
    }

    /// Saved events, all of time unless a date is named.
    #[test]
    fn bookmarks_resolve_and_default_to_all_time() {
        for s in ["show my bookmarks", "what have i saved", "my favourites"] {
            let q = resolve(s);
            assert_eq!(q.kind, Kind::Bookmarks, "{s}");
            assert!(q.bookmarked, "{s}");
            assert_eq!(q.when, When::Ever, "{s}");
        }
        // …but an explicit date still wins.
        assert_eq!(resolve("bookmarks from yesterday").when, When::Yesterday);
    }

    /// One id + an explicit ask plays inline; anything else is the card strip,
    /// addressed BY ID so the cards are exactly the events described.
    #[test]
    fn the_media_tag_is_exact() {
        let ids: Vec<String> = (0..30).map(|i| format!("id{i}")).collect();
        assert_eq!(media_tag(&[], true, 12), None);
        assert_eq!(media_tag(&[], false, 12), None);
        assert_eq!(media_tag(&ids[..1], true, 12).unwrap(), "[SEND_CLIP:id0]");
        // Without an explicit ask, even one event is a card, not an upload.
        assert_eq!(media_tag(&ids[..1], false, 12).unwrap(), "[SHOW_EVENTS:ids=id0]");

        // Cards track the row budget, so the strip is exactly what the model saw.
        let many = media_tag(&ids, true, 12).unwrap();
        assert!(many.starts_with("[SHOW_EVENTS:ids=id0,id1,"), "{many}");
        assert_eq!(many.matches(',').count(), 11, "12 cards for a 12-row budget");
        assert!(!many.contains(",]"), "no trailing comma: {many}");

        // A bigger model shows more — the whole point of the budget.
        let wide = media_tag(&ids, true, 40).unwrap();
        assert!(wide.matches(',').count() > 11, "a wider budget must show more");
        // …but never past the hard ceiling, whatever the budget claims.
        assert_eq!(media_tag(&ids, true, 999).unwrap().matches(',').count(),
                   ids.len().min(SHOW_CAP) - 1);
    }

    /// Headline numbers must describe EVERY match, not the page that fit.
    ///
    /// Observed live: "269 event(s) the last 720 hours, between 19:36 and 22:47 —
    /// 16 involving a person". The 269 was real; the span and the sub-counts came
    /// from the 22 rows that fit, which were all one evening. A total sitting next
    /// to sampled detail reads as detail about the total, and no reader can tell.
    #[tokio::test]
    async fn headline_numbers_cover_every_match_not_just_the_page() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        crate::db::init_db(&pool).await.unwrap();
        // 30 events over three days — more than any row budget will fetch.
        for d in 1..=3 {
            for i in 0..10 {
                sqlx::query("INSERT INTO motion_events(id, started_at, event_category, duration_secs)
                             VALUES(?,?,?,?)")
                    .bind(format!("e{d}_{i}"))
                    .bind(format!("2026-08-0{d}T1{i}:00:00Z"))
                    .bind(if i < 4 { "person" } else { "audio" })
                    .bind(if i == 9 { 300.0 } else { 10.0 })
                    .execute(&pool).await.unwrap();
            }
        }
        let q = Query::new(Kind::Events, When::Ever);
        let st = stats_for(&pool, &q).await;

        assert_eq!(st.total, 30, "total must be every match");
        assert_eq!(st.people, 12, "4 person events × 3 days — not 4 from one page");
        assert_eq!(st.sounds, 18);
        assert_eq!(st.longest, 300.0, "the longest anywhere, not the longest shown");
        // The ends must span all three days, not the newest day's rows.
        assert!(st.first.as_deref().unwrap().contains("2026-08-01"), "{:?}", st.first);
        assert!(st.last.as_deref().unwrap().contains("2026-08-03"),  "{:?}", st.last);
    }

    /// Investigate must not inherit "today" from a question that named no time —
    /// "someone stole my phone" searched over six hours finds nothing, which is
    /// the difference between a sweep and a shrug.
    #[test]
    fn a_question_with_no_time_is_not_a_question_about_today() {
        for no_time in ["someone stole my phone", "find the person in the red jacket",
                        "who took the parcel", "a package went missing"] {
            assert!(!mentions_a_time(no_time), "{no_time} names no window");
        }
        for timed in ["what happened today", "anyone here last night", "events this week",
                      "who came by on tuesday", "the last 3 hours"] {
            assert!(mentions_a_time(timed), "{timed} names a window");
        }
    }

    /// Every mode reads; none of them acts. Guardian is monitor-only and the
    /// modes are about how hard it looks, not what it may do.
    #[test]
    fn investigate_looks_wider_than_ask() {
        use super::super::chat::Mode;
        assert!(Mode::Investigate.max_hops() > Mode::Ask.max_hops());
        assert_eq!(Mode::default(), Mode::Ask, "Ask stays the default");
        assert_eq!(Mode::Brief.max_hops(), Mode::Ask.max_hops(),
            "Brief summarises a period; it does not need extra hops");
    }

    /// The budget is what stops an 8k model and a 200k model both being shown
    /// twelve rows. It must never drop below what shipped, or answers regress.
    #[test]
    fn the_row_budget_scales_but_never_shrinks() {
        let mut s = crate::Settings::default();

        s.ai_provider = "local".into();
        let on_device = row_budget(&s);
        assert!(on_device >= 12, "floor is the old behaviour, got {on_device}");

        s.ai_provider = "anthropic".into();
        s.vision_model = "claude-sonnet-4".into();
        let cloud = row_budget(&s);
        assert!(cloud > on_device, "200k window must beat 8k: {cloud} vs {on_device}");
        assert!(cloud <= 40, "capped so an answer stays an answer, got {cloud}");

        // An unknown model must not collapse to the floor.
        s.vision_model = "some-new-model-2027".into();
        assert!(row_budget(&s) >= 12);
    }

    /// "The third time that van has been here" is a count over a WIDER window
    /// than the question asked about, and it must come from SQL — a model
    /// tallying rows returns a confident wrong number.
    #[tokio::test]
    async fn a_regular_visitor_is_counted_across_the_week_not_the_window() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        crate::db::init_db(&pool).await.unwrap();
        // Four sightings of one plate spread over three days; one of another.
        // `-10 minutes`, not `-1 hours`: the window predicate is `started_at >
        // cutoff`, so an event sitting exactly on the boundary is excluded.
        for (id, at, plate) in [
            ("a", "-10 minutes", "AB12CDE"), ("b", "-1 days", "AB12CDE"),
            ("c", "-2 days",     "AB12CDE"), ("d", "-2 days", "AB12CDE"),
            ("e", "-10 minutes", "ZZ99ZZZ"),
        ] {
            sqlx::query("INSERT INTO motion_events(id, started_at, recognized_plate, event_category)
                         VALUES(?, datetime('now', ?), ?, 'vehicle')")
                .bind(id).bind(at).bind(plate).execute(&pool).await.unwrap();
        }

        // Ask only about the last hour — the recurrence still reaches back a week.
        let q = Query::new(Kind::Events, When::Hours(1));
        let notes = recurring_in(&pool, &q).await;

        assert!(notes.iter().any(|n| n.contains("AB12CDE") && n.contains('4')),
                "the regular is counted over the week, not the hour: {notes:?}");
        assert!(!notes.iter().any(|n| n.contains("ZZ99ZZZ")),
                "one sighting is not a pattern: {notes:?}");
        assert!(notes.iter().any(|n| n.contains("3 days")),
                "distinct DAYS, not an average hour: {notes:?}");
    }

    /// A query that FAILED and a query that matched nothing are different
    /// answers and must never read the same. "Nothing recorded today" asserts
    /// something about the archive; a broken read asserts nothing at all.
    #[tokio::test]
    async fn a_broken_archive_does_not_read_as_an_empty_one() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        // Deliberately NOT initialised: `motion_events` does not exist, so every
        // query errors — the shape of a corrupt or locked database.
        let q = Query::new(Kind::Events, When::Today);
        let broken = stats_for(&pool, &q).await;
        assert!(broken.failed, "a failed read must be marked as one");
        assert_eq!(broken.total, 0);

        // …and a healthy but empty archive must NOT be marked failed, or every
        // quiet day would report a fault.
        crate::db::init_db(&pool).await.unwrap();
        let empty = stats_for(&pool, &q).await;
        assert!(!empty.failed, "an empty archive is not a broken one");
        assert_eq!(empty.total, 0);
    }

    /// Investigate follows leads out of the first pass. The lead must be a
    /// DIFFERENT person than the one asked about — otherwise "when was Alex
    /// here?" spends its whole budget re-running the question it just answered.
    #[tokio::test]
    async fn an_investigation_does_not_chase_the_person_already_asked_about() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        crate::db::init_db(&pool).await.unwrap();
        for (id, who) in [("a", "Alex"), ("b", "Alex"), ("c", "Sam")] {
            sqlx::query("INSERT INTO motion_events(id, started_at, sub_label, event_category)
                         VALUES(?, datetime('now','-1 hour'), ?, 'person')")
                .bind(id).bind(who).execute(&pool).await.unwrap();
        }
        let mut q = Query::new(Kind::Events, When::Hours(24));
        q.who = Some("Alex".into());

        let leads: Vec<String> = named_in(&pool, &q).await.into_iter()
            .filter(|n| q.who.as_deref().map(|w| !w.eq_ignore_ascii_case(n)).unwrap_or(true))
            .collect();
        assert!(!leads.iter().any(|n| n == "Alex"),
                "the subject of the question is not a lead: {leads:?}");
    }

    /// The whole point of the sweep is that it TERMINATES: a lead that adds no
    /// new ids must not extend it, or a busy archive walks forever.
    #[tokio::test]
    async fn an_investigation_stops_when_a_pass_adds_nothing() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        crate::db::init_db(&pool).await.unwrap();
        sqlx::query("INSERT INTO motion_events(id, started_at, sub_label, event_category)
                     VALUES('a', datetime('now','-1 hour'), 'Sam', 'person')")
            .execute(&pool).await.unwrap();

        // Everything the query matches is already in `seen`, so a second pass
        // over the same rows contributes nothing new.
        let q = Query::new(Kind::Events, When::Hours(24));
        let all = stats_for(&pool, &q).await.total;
        assert_eq!(all, 1);

        let mut seen: std::collections::HashSet<String> = ["a".to_string()].into();
        let added = ["a"].iter().filter(|id| seen.insert(id.to_string())).count();
        assert_eq!(added, 0, "a repeat pass adds nothing and ends the lead");
    }

    /// Phrase lists miss. The first cut matched `"what time"` and `"the time
    /// now"`, so "what is the time" matched NOTHING, fell through to the small
    /// model, and came back as the question echoed verbatim. Typos break a phrase
    /// list the same way: "wha5 date is it?" is what a real user actually typed.
    #[test]
    fn clock_questions_resolve_however_they_are_typed() {
        for q in ["what is the time", "what's the time", "what is the time now",
                  "what time is it", "time?", "what is todays date",
                  "what's the date", "wha5 date is it?", "what day is it"] {
            assert_eq!(resolve(q).kind, Kind::Clock, "{q}");
        }
    }

    /// "hello" was answered with "No new events recorded in the past 6 hours."
    /// Retrieval is not a conversation.
    ///
    /// The length ceiling that replaced it was wrong too: "hii how are you" is
    /// four words and came back as "The cameras haven't detected any activity
    /// lately." A greeting is defined by having nothing to look up, not by being
    /// short.
    #[test]
    fn greetings_are_not_event_queries() {
        for q in ["hi", "hii", "hello", "hey", "yo", "thanks!", "thank you", "ok cool",
                  "hii how are you", "hey there, how are you doing", "how are you",
                  "hello!!", "yo whats up"] {
            assert_eq!(resolve(q).kind, Kind::Greeting, "{q}");
        }
        // …but a greeting wrapped around a real question is a real question.
        assert_eq!(resolve("hi, what happened today").kind, Kind::Events);
        assert_eq!(resolve("hey what time is it").kind, Kind::Clock);
        assert_eq!(resolve("hello, any cars today").kind, Kind::Events);
    }

    /// Asked what it could do, it recited the COCO class list out of its own prompt.
    #[test]
    fn identity_questions_resolve() {
        for q in ["who are you", "what can you do", "what are your capabilities",
                  "what do you do", "whats your name", "what are you doing"] {
            assert_eq!(resolve(q).kind, Kind::About, "{q}");
        }
    }

    /// The model may phrase the time however it likes — but the numbers have to
    /// be ours. This is what stops "19:01 (local time, UTC-05:00)" shipping again
    /// now that clock answers go through the model for natural wording.
    #[test]
    fn a_clock_answer_may_not_invent_the_numbers() {
        let q = Query::new(Kind::Clock, When::Today);
        let now = chrono::Local::now();
        let good = format!("It's just gone {} on {}.",
            now.format("%H:%M"), now.format("%A"));
        assert!(grounded(&good, &q, &[], &rows_ev()), "{good}");
        assert!(grounded("It's late morning here.", &q, &[], &rows_ev()), "vague phrasing is fine");

        assert!(!grounded("The time now is 19:01 (local time, UTC-05:00).", &q, &[], &rows_ev()));
        assert!(!grounded("It is 2024 and all is quiet.", &q, &[], &rows_ev()));

        // The guard applies ONLY to clock answers — an event at 19:01 is a fact.
        let ev = Query::new(Kind::Events, When::Today);
        assert!(grounded("Someone arrived at 19:01 last night.", &ev, &[], &rows_ev()));
    }

    /// Observed live: "show me the old footage" reported on TODAY, found nothing,
    /// and said so — true, and useless.
    #[test]
    fn old_footage_does_not_mean_today() {
        for q in ["show me the past footage", "old footage", "previous events",
                  "anything in the archive"] {
            let r = resolve(q);
            assert_ne!(r.when, When::Today, "{q} resolved to today");
        }
    }

    /// The security property: nothing user-derived is ever formatted into SQL.
    /// This runs on text that arrives from Telegram, so it is untrusted by
    /// definition — the value must travel as a bind, and only as a bind.
    #[test]
    fn user_values_are_bound_never_interpolated() {
        const EVIL: &str = "Robert'); DROP TABLE motion_events;--";
        let mut q = Query::new(Kind::Events, When::Day("2026-07-14".into()));
        q.cam = Some(3);
        q.who = Some(EVIL.into());
        let (sql, binds) = where_sql(&q);
        assert!(!sql.contains(EVIL), "user text reached the SQL string: {sql}");
        assert!(!sql.contains("DROP"), "user text reached the SQL string: {sql}");
        assert!(!sql.contains("2026-07-14"), "even a validated date is bound: {sql}");
        assert_eq!(sql.matches('?').count(), binds.len(), "one bind per placeholder");
        assert!(binds.iter().any(|b| b == EVIL), "the value must travel as a bind");
    }

    /// The whole point of the round: every slot in the sentence survives.
    #[test]
    fn a_described_memory_parses_into_every_slot() {
        let q = resolve("find the person in the red jacket who came around 9pm last tuesday");
        assert!(matches!(q.when, When::Day(_)), "weekday must resolve to a date: {:?}", q.when);
        assert_eq!(q.hours, Some((20, 22)), "9pm gets an hour of slack either side");
        assert_eq!(q.outfit, vec![("top", "red")]);
        assert_eq!(q.what, Some("person"), "describing clothing implies a person");
        assert_eq!(q.kind, Kind::Search);
    }

    /// Cars do not wear jackets. A garment in a vehicle question is a mis-parse.
    #[test]
    fn a_vehicle_question_drops_the_outfit() {
        let q = resolve("what cars came on tuesday");
        assert!(q.outfit.is_empty());
        assert_eq!(q.what, Some("vehicle"));
    }

    /// Hour predicates must compare STRING to STRING. `binds` is `Vec<String>`,
    /// and SQLite orders by type first — so casting the column to int would make
    /// `20 >= '20'` false and every hour filter would silently return nothing.
    #[test]
    fn hour_bounds_are_zero_padded_strings() {
        let mut q = Query::new(Kind::Events, When::Today);
        q.hours = Some((9, 17));
        let (sql, binds) = where_sql(&q);
        assert!(sql.contains("strftime('%H',started_at,'localtime') >= ?"), "{sql}");
        assert!(!sql.contains("as int) >="), "must not cast for the hour compare: {sql}");
        assert!(binds.contains(&"09".to_string()), "{binds:?}");
        assert!(binds.contains(&"17".to_string()), "{binds:?}");
    }

    /// A window that wraps midnight ON a named day means that day's evening into
    /// the NEXT morning. The obvious predicate also matches the same day's small
    /// hours — the night before the one meant.
    #[tokio::test]
    async fn a_midnight_window_on_a_day_spans_into_the_next() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        crate::db::init_db(&pool).await.unwrap();

        // Seeded as LOCAL wall-clock, stored as UTC — which is exactly what the
        // app does, and what makes this test mean the same thing in every
        // timezone. Seeding UTC literals would pass or fail depending on where
        // the machine is, which is no test at all.
        use chrono::TimeZone as _;
        let local_utc = |d: (i32, u32, u32), t: (u32, u32)| {
            let naive = chrono::NaiveDate::from_ymd_opt(d.0, d.1, d.2).unwrap()
                .and_hms_opt(t.0, t.1, 0).unwrap();
            chrono::Local.from_local_datetime(&naive).unwrap()
                .with_timezone(&chrono::Utc).to_rfc3339()
        };
        for (id, at) in [
            ("early", local_utc((2026, 7, 28), (0, 30))),  // Tue small hours — the WRONG night
            ("late",  local_utc((2026, 7, 28), (23, 30))), // Tue evening      — wanted
            ("after", local_utc((2026, 7, 29), (0, 30))),  // Wed small hours  — wanted
            ("noon",  local_utc((2026, 7, 28), (12, 0))),  // outside the window
        ] {
            sqlx::query("INSERT INTO motion_events(id, started_at) VALUES(?,?)")
                .bind(id).bind(&at).execute(&pool).await.unwrap();
        }
        let mut q = Query::new(Kind::Events, When::Day("2026-07-28".into()));
        q.hours = Some((22, 2));
        let (pred, binds) = where_sql(&q);
        let sql = format!("SELECT id FROM motion_events WHERE {pred} ORDER BY id");
        let mut qy = sqlx::query_scalar::<_, String>(&sql);
        for b in &binds { qy = qy.bind(b); }
        let got = qy.fetch_all(&pool).await.unwrap();
        assert_eq!(got, vec!["after".to_string(), "late".to_string()],
                   "must span Tue night into Wed morning, and exclude Tue's own small hours");
    }

    /// The refinement ladder: it must terminate, relax the WEAKEST slot first,
    /// and report what it gave up. A search that quietly widens until it finds
    /// something is worse than one that finds nothing.
    #[tokio::test]
    async fn the_ladder_drops_the_outfit_before_the_day() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        crate::db::init_db(&pool).await.unwrap();
                use chrono::TimeZone as _;
        // One person on Tuesday at 21:00 local, wearing blue — not the red we ask for.
        let naive = chrono::NaiveDate::from_ymd_opt(2026, 7, 28).unwrap()
            .and_hms_opt(21, 0, 0).unwrap();
        let at = chrono::Local.from_local_datetime(&naive).unwrap()
            .with_timezone(&chrono::Utc).to_rfc3339();
        sqlx::query(
            "INSERT INTO motion_events(id, started_at, dominant_label, outfit)
             VALUES('e1', ?, 'person', '{\"top\":\"blue\"}')")
            .bind(&at).execute(&pool).await.unwrap();

        let mut q = Query::new(Kind::Search, When::Day("2026-07-28".into()));
        q.hours = Some((20, 22));
        q.outfit = vec![("top", "red")];
        q.what = Some("person");

        let (relaxed, given_up) = widen_until_found(&pool, &q).await;
        assert!(relaxed.outfit.is_empty(), "the colour must be the first thing dropped");
        assert_eq!(relaxed.when, When::Day("2026-07-28".into()), "the DAY must survive");
        assert_eq!(relaxed.hours, Some((20, 22)), "the hour must survive too");
        assert_eq!(given_up.len(), 1, "stop as soon as it matches: {given_up:?}");
        assert!(given_up[0].contains("red"), "{given_up:?}");
    }

    /// A query that already matches must not be relaxed at all.
    #[tokio::test]
    async fn a_matching_query_is_left_alone() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        crate::db::init_db(&pool).await.unwrap();
                use chrono::TimeZone as _;
        let naive = chrono::NaiveDate::from_ymd_opt(2026, 7, 28).unwrap()
            .and_hms_opt(21, 0, 0).unwrap();
        let at = chrono::Local.from_local_datetime(&naive).unwrap()
            .with_timezone(&chrono::Utc).to_rfc3339();
        sqlx::query(
            "INSERT INTO motion_events(id, started_at, dominant_label, outfit)
             VALUES('e1', ?, 'person', '{\"top\":\"red\"}')")
            .bind(&at).execute(&pool).await.unwrap();

        let mut q = Query::new(Kind::Search, When::Day("2026-07-28".into()));
        q.hours = Some((20, 22));
        q.outfit = vec![("top", "red")];
        let (relaxed, given_up) = widen_until_found(&pool, &q).await;
        assert!(given_up.is_empty(), "nothing to relax: {given_up:?}");
        assert_eq!(relaxed.outfit, vec![("top", "red")]);
    }

    /// Nothing in the database at all: the ladder must still stop, not spin.
    #[tokio::test]
    async fn the_ladder_terminates_on_an_empty_database() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        crate::db::init_db(&pool).await.unwrap();
                let mut q = Query::new(Kind::Search, When::Day("2026-07-28".into()));
        q.hours = Some((20, 22));
        q.outfit = vec![("top", "red"), ("bottom", "black")];
        q.keywords = vec!["backpack".into()];
        let (_, given_up) = widen_until_found(&pool, &q).await;
        assert!(given_up.len() <= 3, "bounded at three rungs: {given_up:?}");
        assert!(!given_up.is_empty(), "it should have tried something");
    }

    /// A bookmark row left behind by a deleted event must never surface. The
    /// `EXISTS` runs FROM `motion_events`, so an orphan simply matches nothing —
    /// which matters because `event_bookmarks` has no FK cascade.
    #[tokio::test]
    async fn a_deleted_bookmarked_event_cannot_come_back() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        crate::db::init_db(&pool).await.unwrap();
        for id in ["keep", "gone", "plain"] {
            sqlx::query("INSERT INTO motion_events(id, started_at) VALUES(?, datetime('now'))")
                .bind(id).execute(&pool).await.unwrap();
            if id != "plain" {
                sqlx::query("INSERT INTO event_bookmarks(event_id, cam_id, created_at)
                             VALUES(?, 0, datetime('now'))")
                    .bind(id).execute(&pool).await.unwrap();
            }
        }
        // The event goes; its bookmark row is left dangling.
        sqlx::query("DELETE FROM motion_events WHERE id='gone'").execute(&pool).await.unwrap();

        let mut q = Query::new(Kind::Bookmarks, When::Ever);
        q.bookmarked = true;
        let got = ids_for(&pool, &q, "", 12).await;
        assert_eq!(got, vec!["keep".to_string()],
                   "only the surviving bookmark, never the orphan or the unsaved event");
    }

    /// Closed vocabularies are safe to inline — but they must stay closed.
    #[test]
    fn category_filters_use_literal_sets_only() {
        for what in ["person", "vehicle", "animal", "audio"] {
            let mut q = Query::new(Kind::Events, When::Today);
            q.what = Some(what);
            let (sql, binds) = where_sql(&q);
            assert!(sql.contains("AND"), "{what} produced no predicate");
            assert!(binds.is_empty(), "{what} should need no binds");
        }
    }

    /// An undated security answer is worthless. Every branch must carry its span.
    #[test]
    fn every_fallback_answer_is_dated() {
        let today = When::Today.label();
        assert!(today.contains(&chrono::Local::now().format("%Y-%m-%d").to_string()));

        let empty = Evidence::empty(today.clone(), "recorded");
        assert!(empty.fallback_answer().contains(&today));

        let full = Evidence {
            headline: format!("3 event(s) {today}:"),
            lines: (0..9).map(|i| format!("0{i}:00 · Front Door · person")).collect(),
            ids: vec!["a".into()], span: today.clone(), sampled: true, total: 1,
        };
        let a = full.fallback_answer();
        assert!(a.contains(&today));
        assert!(a.contains("and 4 more"), "long lists must be summarised: {a}");
    }

    /// A predicate that parses in my head but not in SQLite is a runtime error on
    /// a user's question. This builds the real schema in memory and asks SQLite to
    /// prepare every query shape the module can generate — column names, the
    /// `EXISTS` risk sub-select and the `COLLATE NOCASE` clause included.
    #[tokio::test]
    async fn every_generated_query_parses_against_the_real_schema() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.expect("open");
        crate::db::init_db(&pool).await.expect("schema");

        let whens = [
            When::Today, When::Yesterday, When::LastNight,
            When::Hours(1), When::Hours(168), When::Day("2026-07-14".into()),
        ];
        for when in whens {
            for what in [None, Some("person"), Some("vehicle"), Some("animal"), Some("audio")] {
                for risk in [None, Some("critical"), Some("high")] {
                    let mut q = Query::new(Kind::Events, when.clone());
                    q.what = what;
                    q.risk = risk;
                    q.cam = Some(0);
                    q.who = Some("Ranjith".into());
                    let (pred, binds) = where_sql(&q);

                    for sql in [
                        format!("SELECT COUNT(*) FROM motion_events WHERE {pred}"),
                        format!("SELECT id, started_at, cam_id, COALESCE(event_category,''), \
                                 COALESCE(dominant_label,''), COALESCE(sub_label,''), \
                                 duration_secs, ai_summary \
                                 FROM motion_events WHERE {pred} ORDER BY started_at DESC LIMIT 5"),
                        format!("SELECT recognized_plate, COUNT(*), MAX(started_at), MAX(cam_id) \
                                 FROM motion_events WHERE {pred} AND recognized_plate IS NOT NULL \
                                 GROUP BY recognized_plate"),
                        format!("SELECT COALESCE(NULLIF(dominant_label,''),'sound'), COUNT(*), \
                                 MAX(started_at) FROM motion_events \
                                 WHERE {pred} AND event_category = 'audio' GROUP BY 1"),
                    ] {
                        let mut qy = sqlx::query(&sql);
                        for b in &binds { qy = qy.bind(b); }
                        qy.fetch_all(&pool).await
                            .unwrap_or_else(|e| panic!("{when:?}/{what:?}/{risk:?}: {e}\n{sql}"));
                    }
                }
            }
        }

        // The two aggregate queries that don't route through `where_sql`.
        sqlx::query("SELECT cam_id, name FROM camera_configs WHERE name <> ''")
            .fetch_all(&pool).await.expect("camera_names");
        sqlx::query(
            "SELECT seen_at, camera_id FROM face_sightings
              WHERE person_name = ? COLLATE NOCASE AND seen_at > datetime('now','-30 days')
              ORDER BY seen_at DESC LIMIT 500")
            .bind("Ranjith").fetch_all(&pool).await.expect("person_evidence");
    }

    #[test]
    fn iso_dates_are_validated_not_just_shaped() {
        assert_eq!(find_iso_date("on 2026-07-14 please").as_deref(), Some("2026-07-14"));
        assert_eq!(find_iso_date("2026-13-99"), None, "impossible dates are not dates");
        assert_eq!(find_iso_date("no date here"), None);
    }
}
