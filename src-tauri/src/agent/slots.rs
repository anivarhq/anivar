//! Turning a described memory into query slots.
//!
//! "the person in the red jacket who came around 9pm last Tuesday" carries four
//! facts — a weekday, a clock time, a garment colour, and a subject — and until
//! this module existed the resolver dropped three of them and answered about
//! persons *today*.
//!
//! Everything here is a pure function over the question text. No clock reads
//! (the caller passes `today`), no database, no async — which is what lets the
//! large table of real phrasings run as an ordinary unit test.
//!
//! Two rules shape the vocabularies:
//!
//! * **Whole words only.** `monitor` contains `mon`, `satellite` contains `sat`,
//!   `personal` contains `son`. Substring matching on a weekday or a name turns
//!   ordinary sentences into confident nonsense.
//! * **Only what the pipeline can actually produce.** The colour lexicon is
//!   exactly the eleven bins `alpr::dominant_color_in_band` votes over. Accepting
//!   "teal" would parse beautifully and match nothing, forever.

use std::collections::HashSet;

/// Split a question into lowercase alphanumeric tokens, in order.
pub(super) fn tokens(q: &str) -> Vec<String> {
    q.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_string)
        .collect()
}

/// Do `a` and `b` differ by at most one insert, delete or substitution?
///
/// Real messages arrive as `vlips`, `shiw`, `ebents`, `whuch`, `modela` — a
/// keyword list loses to every one of them, and losing means the question falls
/// through to a generic non-answer. Two pointers, no allocation, no matrix: the
/// only question is whether ONE edit reconciles them.
fn within_one(a: &str, b: &str) -> bool {
    let (x, y) = (a.as_bytes(), b.as_bytes());
    let (long, short) = if x.len() >= y.len() { (x, y) } else { (y, x) };
    if long.len() - short.len() > 1 { return false; }

    let mut i = 0; // index into `long`
    let mut j = 0; // index into `short`
    let mut edited = false;
    while i < long.len() && j < short.len() {
        if long[i] == short[j] { i += 1; j += 1; continue; }
        if edited { return false; }
        edited = true;
        if long.len() == short.len() { i += 1; j += 1; } else { i += 1; } // substitute : delete
    }
    true // any single trailing leftover is the one permitted edit
}

/// Does the question mention any of `vocab`?
///
/// Multi-word entries match as substrings of `normalised` — the tokens rejoined
/// with single spaces, NOT the raw text. That is what makes "book-marks",
/// "book  marks" and "book marks!" all match the phrase `"book marks"`:
/// punctuation and spacing stop mattering once the tokens are the truth.
///
/// **Single words match whole-token or fuzzy — never substring.** That rule is
/// the important one: substring matching is what makes `scary` a vehicle (via
/// `car`) and `personal` a person, the bug class this module's header warns about.
///
/// Fuzzy applies only from four characters on both sides. Below that a single
/// edit is noise, not a typo — `car`/`cat`, `who`/`how`, `van`/`can` are all
/// distance 1 and all mean different things.
pub(super) fn hits(toks: &[String], normalised: &str, vocab: &[&str]) -> bool {
    vocab.iter().any(|v| {
        if v.contains(' ') { return normalised.contains(v); }
        toks.iter().any(|t| {
            t == v || (t.len() >= 4 && v.len() >= 4 && within_one(t, v))
        })
    })
}

// ─── What a question is about ────────────────────────────────────────────────
//
// Plurals are spelled out because single words no longer match as substrings.
// That is the trade: ~20 extra strings, and a whole class of false positive gone.

pub(super) const MEDIA: &[&str] = &[
    "show", "see", "watch", "view", "clip", "clips", "video", "videos",
    "footage", "play", "send", "share", "attach", "replay", "recording",
    "recordings", "watching",
];
pub(super) const EVENTS: &[&str] = &[
    "event", "events", "happen", "happened", "happening", "motion", "activity",
    "alert", "alerts", "detection", "detections", "anything", "update", "updates",
    "recent", "latest", "new", "going on", "whats up", "what's up", "any news",
];
pub(super) const VEHICLE: &[&str] = &[
    "vehicle", "vehicles", "car", "cars", "van", "vans", "truck", "trucks",
    "plate", "plates", "numberplate", "licence", "license", "motorbike",
];
pub(super) const SOUND: &[&str] = &[
    // "audio" was missing entirely, so "what about audio events" was never an
    // audio question no matter how carefully it was spelled.
    "sound", "sounds", "audio", "audible", "noise", "noises", "bark", "barking",
    "alarm", "alarms", "siren", "scream", "shout", "glass", "heard", "hearing",
];
pub(super) const PERSON: &[&str] = &[
    "person", "people", "someone", "somebody", "anyone", "stranger", "strangers",
    "intruder", "intruders", "visitor", "visitors", "delivery", "courier", "face",
    "faces",
];
pub(super) const BOOKMARK: &[&str] = &[
    "bookmark", "bookmarks", "bookmarked", "saved", "favourite", "favourites",
    "favorite", "favorites", "starred", "pinned", "marked",
    // People write it as two words. As single tokens neither "book" nor "marks"
    // can be listed — "book" would swallow bookshelf, booking, booked — so these
    // go in as PHRASES, which match as substrings.
    "book mark", "book marks",
];
/// Questions about the ARCHIVE ITSELF rather than what happened in it: how far
/// back the footage goes, which days have any, how much is kept.
///
/// Must be tested BEFORE `COUNT` in the ladder — "how many days of footage do
/// you have" contains "how many", and answering it as a count of today's events
/// produced "nothing recorded today" for a question about the whole archive.
pub(super) const COVERAGE: &[&str] = &[
    "how far back", "how many days", "how much footage", "how many days of footage",
    "which days", "what days", "since when", "how long have you been",
    "oldest", "earliest", "days of footage", "days of recording",
    "how much do you have", "how much history", "date range", "coverage",
];
pub(super) const COUNT: &[&str] = &[
    "how many", "how much", "count", "number of", "total", "tally",
];
pub(super) const STATUS: &[&str] = &[
    "status", "all clear", "everything ok", "everything okay", "any problem",
    "is anything", "healthy", "working", "online", "offline",
];
/// Questions about the SYSTEM rather than the footage.
pub(super) const CONFIG: &[&str] = &[
    "model", "models", "engine", "engines", "yolo", "detector", "vision",
    "provider", "llm", "running", "version", "config", "configured", "settings",
    "setup", "using",
];
/// Deictic references to whatever was last shown — "show me those".
pub(super) const DEICTIC: &[&str] = &[
    "them", "those", "these", "that", "it", "again",
];

// ─── Weekday ─────────────────────────────────────────────────────────────────

/// `(weekday token, chrono weekday)` — long and short forms people type.
const WEEKDAYS: &[(&str, chrono::Weekday)] = &[
    ("monday", chrono::Weekday::Mon), ("mon", chrono::Weekday::Mon),
    ("tuesday", chrono::Weekday::Tue), ("tue", chrono::Weekday::Tue), ("tues", chrono::Weekday::Tue),
    ("wednesday", chrono::Weekday::Wed), ("wed", chrono::Weekday::Wed), ("weds", chrono::Weekday::Wed),
    ("thursday", chrono::Weekday::Thu), ("thu", chrono::Weekday::Thu), ("thur", chrono::Weekday::Thu),
    ("thurs", chrono::Weekday::Thu),
    ("friday", chrono::Weekday::Fri), ("fri", chrono::Weekday::Fri),
    ("saturday", chrono::Weekday::Sat), ("sat", chrono::Weekday::Sat),
    ("sunday", chrono::Weekday::Sun), ("sun", chrono::Weekday::Sun),
];

/// Resolve a weekday mention to a concrete local date.
///
/// Returns `(YYYY-MM-DD, clamped_from_future)`. Resolving to a *date* rather than
/// carrying a weekday through the query means no new SQL branch: it reuses the
/// existing `When::Day`.
///
/// `today` is a parameter, not a clock read — the "last Tuesday when today is
/// Tuesday" case is impossible to test otherwise, and it is the case that is
/// always wrong in naive implementations.
pub(super) fn weekday_day(
    toks: &[String],
    today: chrono::NaiveDate,
) -> Option<(String, bool)> {
    use chrono::Datelike as _;
    let set: HashSet<&str> = toks.iter().map(String::as_str).collect();
    let (_, target) = WEEKDAYS.iter().find(|(w, _)| set.contains(w))?;

    let back = (today.weekday().num_days_from_monday() as i64
        - target.num_days_from_monday() as i64)
        .rem_euclid(7);

    let explicit_last = set.contains("last") || set.contains("previous") || set.contains("past");
    let future = set.contains("next") || set.contains("coming");

    // "last Tuesday" ON a Tuesday means a week ago, not today. Bare "Tuesday"
    // means today. This single line is the whole reason `today` is injected.
    let days = if explicit_last && back == 0 { 7 } else { back };
    let day = today - chrono::Duration::days(days);
    Some((day.format("%Y-%m-%d").to_string(), future))
}

// ─── Time of day ─────────────────────────────────────────────────────────────

/// Named parts of the day, as `[start, end)` local hours. `night` wraps midnight.
const BUCKETS: &[(&str, u32, u32)] = &[
    ("morning", 5, 12),
    ("afternoon", 12, 17),
    ("evening", 17, 22),
    ("night", 22, 5),
    ("midnight", 23, 2),
    ("noon", 11, 14),
    ("lunchtime", 11, 14),
];

/// An hour window `[start, end)` in local time; `start > end` wraps midnight.
///
/// A stated clock time gets **±1 hour**. People remember "about nine" and mean
/// somewhere in the 8-to-10 region; matching the 21:00 hour exactly is how a
/// search that should work returns nothing. An explicit range ("between 8 and
/// 10") is taken literally — the user already gave the width.
pub(super) fn hour_window(toks: &[String]) -> Option<(u32, u32)> {
    let set: HashSet<&str> = toks.iter().map(String::as_str).collect();

    // An explicit range wins over everything.
    if set.contains("between") || set.contains("from") {
        let nums: Vec<u32> = toks.iter().filter_map(|t| parse_hour(t, &set)).collect();
        if nums.len() >= 2 && nums[0] != nums[1] {
            return Some((nums[0] % 24, nums[1] % 24));
        }
    }
    // A specific clock time, fuzzed.
    for (i, t) in toks.iter().enumerate() {
        if let Some(h) = parse_clock(t, toks.get(i + 1).map(String::as_str)) {
            return Some(((h + 23) % 24, (h + 1) % 24));
        }
    }
    // A named part of the day.
    for (name, a, b) in BUCKETS {
        if set.contains(name) { return Some((*a, *b)); }
    }
    None
}

/// A bare hour inside a "between X and Y" phrase.
fn parse_hour(tok: &str, set: &HashSet<&str>) -> Option<u32> {
    let n: u32 = tok.trim_end_matches(|c: char| c.is_alphabetic()).parse().ok()?;
    if n > 24 { return None; }
    Some(meridiem(n, tok.ends_with("pm") || set.contains("pm"),
                     tok.ends_with("am") || set.contains("am")))
}

/// One clock reading: `9pm`, `21`, `21:00`, or `9` followed by `pm`.
fn parse_clock(tok: &str, next: Option<&str>) -> Option<u32> {
    let (digits, suffix) = tok.split_at(tok.find(|c: char| !c.is_ascii_digit()).unwrap_or(tok.len()));
    if digits.is_empty() { return None; }
    // "21:00" arrives as two tokens; the hour is the first.
    let n: u32 = digits.parse().ok()?;
    if n > 24 { return None; }
    let pm = suffix.starts_with("pm") || next == Some("pm");
    let am = suffix.starts_with("am") || next == Some("am");
    // A bare number is only a time if it is unambiguous or marked. "3 people"
    // must not become 15:00, so require a meridiem or a 24-hour-looking value.
    if !pm && !am && n < 13 { return None; }
    Some(meridiem(n, pm, am))
}

fn meridiem(n: u32, pm: bool, am: bool) -> u32 {
    match (n, pm, am) {
        (12, true, _) => 12,
        (12, _, true) => 0,
        (h, true, _) if h < 12 => h + 12,
        (h, _, _) => h % 24,
    }
}

// ─── Clothing ────────────────────────────────────────────────────────────────

/// Colours the classifier can actually vote for (`alpr.rs`'s eleven bins), plus
/// the everyday words that map onto them. Anything outside this list would parse
/// and then match nothing, which is worse than not parsing.
const COLOURS: &[(&str, &str)] = &[
    ("black", "black"), ("white", "white"), ("silver", "silver"),
    ("gray", "gray"), ("grey", "gray"),
    ("red", "red"), ("maroon", "red"), ("crimson", "red"), ("burgundy", "red"),
    ("orange", "orange"),
    ("yellow", "yellow"), ("gold", "yellow"),
    ("green", "green"), ("olive", "green"),
    ("blue", "blue"), ("navy", "blue"), ("teal", "blue"),
    ("purple", "purple"), ("violet", "purple"),
    ("brown", "brown"), ("tan", "brown"), ("beige", "brown"), ("khaki", "brown"),
];

const TOPS: &[&str] = &[
    "jacket", "coat", "hoodie", "hoody", "shirt", "tshirt", "top", "jumper",
    "sweater", "sweatshirt", "blouse", "vest", "anorak", "parka", "blazer",
    "cardigan", "puffer", "kurta", "uniform",
];
const BOTTOMS: &[&str] = &[
    "trousers", "pants", "jeans", "shorts", "skirt", "leggings", "chinos",
    "joggers", "trouser",
];

/// Is this word one of the colours the classifier can vote for?
///
/// Used by the answer's grounding check: if the search gave up on "red", the
/// reply must not come back claiming a red jacket was found.
pub(super) fn is_colour(w: &str) -> bool {
    COLOURS.iter().any(|(k, v)| *k == w || *v == w)
}

/// Garment colours mentioned in the question.
///
/// Returns `(slots, garment_keywords)`. A colour with no garment ("the man in
/// red") is taken as a **top**, because that is what people describe — and it is
/// the first slot the refinement loop drops. A garment with no colour becomes a
/// keyword instead, so it can still match the summary text for free.
///
/// `skip` holds tokens already claimed by something more specific — an enrolled
/// person's name, above all. Someone called Rose or Amber must not turn their own
/// question into a colour filter.
pub(super) fn outfit_slots(
    toks: &[String],
    skip: &HashSet<String>,
) -> (Vec<(&'static str, &'static str)>, Vec<String>) {
    let mut slots: Vec<(&'static str, &'static str)> = Vec::new();
    let mut garments: Vec<String> = Vec::new();

    let band_of = |w: &str| -> Option<&'static str> {
        if TOPS.contains(&w) { Some("top") } else if BOTTOMS.contains(&w) { Some("bottom") } else { None }
    };

    for (i, tok) in toks.iter().enumerate() {
        if skip.contains(tok) { continue; }
        let Some((_, colour)) = COLOURS.iter().find(|(w, _)| *w == tok) else {
            // A garment on its own is still a useful keyword.
            if band_of(tok).is_some() && !garments.contains(tok) { garments.push(tok.clone()); }
            continue;
        };
        // Look ahead a couple of tokens: "red jacket", "red winter coat".
        let band = toks.iter().skip(i + 1).take(3).find_map(|w| band_of(w)).unwrap_or("top");
        if !slots.iter().any(|(b, _)| *b == band) {
            slots.push((band, colour));
        }
    }
    (slots, garments)
}

/// Residual search terms: what is left once every other slot has taken its words.
pub(super) fn keywords(toks: &[String], consumed: &HashSet<String>) -> Vec<String> {
    const STOP: &[&str] = &[
        "the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "for", "with",
        "who", "what", "when", "where", "which", "was", "were", "is", "are", "did",
        "do", "does", "me", "my", "you", "your", "i", "show", "find", "search",
        "any", "all", "some", "that", "this", "there", "here", "came", "come",
        "about", "around", "near", "please", "can", "could", "would", "get",
        "person", "people", "someone", "anyone", "somebody", "man", "woman", "guy",
    ];
    let mut out: Vec<String> = Vec::new();
    for t in toks {
        if t.len() < 3 || STOP.contains(&t.as_str()) || consumed.contains(t) { continue; }
        if !out.contains(t) { out.push(t.clone()); }
        if out.len() == 4 { break; }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tk(q: &str) -> Vec<String> { tokens(q) }
    fn day(y: i32, m: u32, d: u32) -> chrono::NaiveDate {
        chrono::NaiveDate::from_ymd_opt(y, m, d).unwrap()
    }

    /// 2026-07-28 is a Tuesday; 2026-07-30 a Thursday.
    #[test]
    fn weekday_resolves_to_a_concrete_date() {
        let thu = day(2026, 7, 30);
        assert_eq!(weekday_day(&tk("last tuesday"), thu).unwrap().0, "2026-07-28");
        assert_eq!(weekday_day(&tk("on tuesday"), thu).unwrap().0, "2026-07-28");
        assert_eq!(weekday_day(&tk("tues"), thu).unwrap().0, "2026-07-28");
        // Yesterday by name.
        assert_eq!(weekday_day(&tk("wednesday"), thu).unwrap().0, "2026-07-29");
    }

    /// The case every naive implementation gets wrong.
    #[test]
    fn last_tuesday_on_a_tuesday_is_a_week_ago() {
        let tue = day(2026, 7, 28);
        assert_eq!(weekday_day(&tk("last tuesday"), tue).unwrap().0, "2026-07-21");
        // …but bare "Tuesday" on a Tuesday means today.
        assert_eq!(weekday_day(&tk("tuesday"), tue).unwrap().0, "2026-07-28");
    }

    /// There is no footage from the future, so a forward reference is clamped —
    /// and flagged, so the answer can say what it did.
    #[test]
    fn next_weekday_is_clamped_to_the_past() {
        let thu = day(2026, 7, 30);
        let (d, clamped) = weekday_day(&tk("next tuesday"), thu).unwrap();
        assert_eq!(d, "2026-07-28");
        assert!(clamped);
    }

    /// Substring matching would make these all weekdays. They are not.
    #[test]
    fn ordinary_words_are_not_weekdays() {
        let thu = day(2026, 7, 30);
        for q in ["check the monitor", "satellite dish", "the sunroof", "personal items",
                  "weds out the noise is fine"] {
            if q.starts_with("weds") { continue; } // "weds" IS a weekday token
            assert!(weekday_day(&tk(q), thu).is_none(), "{q}");
        }
    }

    #[test]
    fn clock_times_get_an_hour_of_slack() {
        assert_eq!(hour_window(&tk("around 9pm")), Some((20, 22)));
        assert_eq!(hour_window(&tk("at 9 pm")), Some((20, 22)));
        assert_eq!(hour_window(&tk("21:00")), Some((20, 22)));
        // Midnight wraps.
        assert_eq!(hour_window(&tk("at 12am")), Some((23, 1)));
    }

    #[test]
    fn explicit_ranges_are_taken_literally() {
        assert_eq!(hour_window(&tk("between 8 and 10 pm")), Some((20, 22)));
        assert_eq!(hour_window(&tk("from 14 to 16")), Some((14, 16)));
    }

    #[test]
    fn named_parts_of_the_day() {
        assert_eq!(hour_window(&tk("tuesday evening")), Some((17, 22)));
        assert_eq!(hour_window(&tk("in the morning")), Some((5, 12)));
        assert_eq!(hour_window(&tk("last night")), Some((22, 5)), "night wraps midnight");
        assert_eq!(hour_window(&tk("what happened today")), None);
    }

    /// A bare small number is not a time — "3 people" must stay a count.
    #[test]
    fn bare_small_numbers_are_not_times() {
        assert_eq!(hour_window(&tk("were there 3 people")), None);
        assert_eq!(hour_window(&tk("how many cars")), None);
    }

    #[test]
    fn garment_colours_map_to_bands() {
        let none = HashSet::new();
        assert_eq!(outfit_slots(&tk("red jacket"), &none).0, vec![("top", "red")]);
        assert_eq!(outfit_slots(&tk("grey hoodie"), &none).0, vec![("top", "gray")],
                   "grey and gray are the same bin");
        assert_eq!(outfit_slots(&tk("navy top and black jeans"), &none).0,
                   vec![("top", "blue"), ("bottom", "black")]);
        // A colour with no garment is a torso — that's what people describe.
        assert_eq!(outfit_slots(&tk("the man in red"), &none).0, vec![("top", "red")]);
    }

    /// A garment with no colour is still worth something, as a keyword.
    #[test]
    fn a_bare_garment_becomes_a_keyword() {
        let none = HashSet::new();
        let (slots, garments) = outfit_slots(&tk("someone in a jacket"), &none);
        assert!(slots.is_empty());
        assert_eq!(garments, vec!["jacket".to_string()]);
    }

    /// Someone enrolled as "Rose" must not turn their own name into a filter.
    #[test]
    fn enrolled_names_are_not_colours() {
        let skip: HashSet<String> = ["rose".to_string(), "amber".to_string()].into();
        // "rose" isn't in the lexicon anyway, but "amber" would be if we'd added it —
        // the guard is what makes adding colour words safe later.
        let (slots, _) = outfit_slots(&tk("when did amber arrive"), &skip);
        assert!(slots.is_empty());
    }

    #[test]
    fn keywords_are_the_residue() {
        let consumed: HashSet<String> = ["red".into(), "jacket".into(), "tuesday".into()].into();
        let got = keywords(&tk("find the person in the red jacket with a backpack on tuesday"), &consumed);
        assert_eq!(got, vec!["backpack".to_string()]);
    }
}

#[cfg(test)]
mod fuzzy_tests {
    use super::*;

    /// The typos in this test are all real — copied from a live session.
    #[test]
    fn one_edit_typos_still_match() {
        let hit = |q: &str, v: &[&str]| { let tk = tokens(q); let n = tk.join(" "); hits(&tk, &n, v) };
        assert!(hit("send me yhe current vlips or video", MEDIA), "vlips -> clips");
        assert!(hit("can you shiw them", MEDIA), "shiw -> show");
        assert!(hit("what abkut audio ebents", EVENTS), "ebents -> events");
        assert!(hit("which vision modela are running", CONFIG), "modela -> models");
        assert!(hit("what us going on whuch models are selected", CONFIG));
    }

    /// Below four characters a single edit is a different word, not a typo.
    /// These pairs are exactly why the floor exists.
    #[test]
    fn short_words_never_match_fuzzily() {
        let hit = |q: &str, v: &[&str]| { let tk = tokens(q); let n = tk.join(" "); hits(&tk, &n, v) };
        assert!(!hit("the cat sat", VEHICLE), "cat must not reach car");
        assert!(!hit("how do i", PERSON), "how must not reach who");
        assert!(!hit("can i", VEHICLE), "can must not reach van");
        assert!(!hit("i saw a bat", SOUND), "bat must not reach bark");
    }

    /// Single words match whole tokens, never substrings — the rule that stops
    /// "scary" becoming a vehicle and "personal" becoming a person.
    #[test]
    fn single_words_do_not_match_as_substrings() {
        let hit = |q: &str, v: &[&str]| { let tk = tokens(q); let n = tk.join(" "); hits(&tk, &n, v) };
        assert!(!hit("that was scary", VEHICLE));
        assert!(!hit("my personal notes", PERSON));
        assert!(!hit("the carpet is wet", VEHICLE));
        // …but multi-word entries are still substrings, which is how phrases work.
        assert!(hit("what is going on", EVENTS));
    }

    #[test]
    fn edit_distance_is_exactly_one() {
        assert!(within_one("show", "shiw"));   // substitute
        assert!(within_one("clips", "vlips")); // substitute
        assert!(within_one("events", "event")); // delete
        assert!(within_one("event", "events")); // insert
        assert!(within_one("model", "model"));  // identical
        assert!(!within_one("recent", "thevrecent"), "distance 4 is not a typo we chase");
        assert!(!within_one("show", "slow!"), "two edits");
    }
}
