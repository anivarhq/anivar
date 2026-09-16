//! People search, find-similar and the day's visits — all over `person_tracks`.
//!
//! "the woman in a blue top with a backpack, no hat" is answered by STRUCTURED
//! evidence: colour shares per track, pedestrian-attribute probabilities,
//! behaviours, names, cameras. CLIP only sees the words left over — it binds
//! colour to garment poorly and ignores "no/without" (ARO, NegBench), so those
//! never reach it. Unknown evidence is neutral: a track with no colour reading
//! ranks below a match and above a contradiction, never out.
//!
//! Results group by WHO (a face-named person, a body-proposed "maybe", or an
//! anonymous body identity) so one person is one row, not forty frames.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::Serialize;
use tauri::State;

use crate::agent::slots;
use crate::alpr::COLOR_NAMES;
use crate::AppState;

// ─── Query ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Attr { Index(usize), AnyBag }

#[derive(Debug, Default, PartialEq)]
pub(crate) struct PersonQuery {
    /// (band "top"|"bottom", colour)
    pub colors: Vec<(&'static str, &'static str)>,
    /// (attribute, wanted?) — `false` came from "no"/"without"/"not".
    pub attrs: Vec<(Attr, bool)>,
    pub behaviours: Vec<&'static str>,
    pub person_id: Option<String>,
    pub unfamiliar: bool,
    pub cams: Vec<i64>,
    /// Leftover words for CLIP.
    pub residual: Vec<String>,
    /// Human chips of what was understood ("blue top", "no hat").
    pub understood: Vec<String>,
}

impl PersonQuery {
    /// Does the query ask something a person TRACK answers better than event text?
    pub(crate) fn is_personal(&self) -> bool {
        !self.colors.is_empty() || !self.attrs.is_empty() || !self.behaviours.is_empty()
            || self.person_id.is_some() || self.unfamiliar
    }

    fn push_attr(&mut self, attr: Attr, want: bool, label: String) {
        if self.attrs.iter().any(|(a, _)| *a == attr) { return; }
        self.attrs.push((attr, want));
        self.understood.push(label);
    }
}

const ATTR_WORDS: &[(&str, Attr, &str)] = &[
    ("hat", Attr::Index(0), "hat"), ("cap", Attr::Index(0), "hat"), ("beanie", Attr::Index(0), "hat"),
    ("helmet", Attr::Index(0), "hat"),
    ("glasses", Attr::Index(1), "glasses"), ("sunglasses", Attr::Index(1), "glasses"),
    ("spectacles", Attr::Index(1), "glasses"),
    ("coat", Attr::Index(10), "long coat"), ("overcoat", Attr::Index(10), "long coat"),
    ("trousers", Attr::Index(11), "trousers"), ("pants", Attr::Index(11), "trousers"),
    ("jeans", Attr::Index(11), "trousers"),
    ("shorts", Attr::Index(12), "shorts"),
    ("skirt", Attr::Index(13), "skirt or dress"), ("dress", Attr::Index(13), "skirt or dress"),
    ("boots", Attr::Index(14), "boots"),
    ("handbag", Attr::Index(15), "handbag"), ("purse", Attr::Index(15), "handbag"),
    ("backpack", Attr::Index(17), "backpack"), ("rucksack", Attr::Index(17), "backpack"),
    ("bag", Attr::AnyBag, "bag"),
    ("carrying", Attr::Index(18), "carrying something"), ("holding", Attr::Index(18), "carrying something"),
    ("child", Attr::Index(19), "child"), ("kid", Attr::Index(19), "child"), ("children", Attr::Index(19), "child"),
    ("elderly", Attr::Index(21), "older person"), ("senior", Attr::Index(21), "older person"),
];
const FEMALE: &[&str] = &["woman", "women", "female", "lady", "ladies", "girl"];
const MALE: &[&str] = &["man", "men", "male", "guy", "gentleman", "boy"];
const NEGATORS: &[&str] = &["no", "without", "not", "nothing"];
const BEHAVIOURS: &[(&str, &str)] = &[
    ("loitering", "loitering"), ("lingering", "loitering"), ("loiter", "loitering"),
    ("running", "running"), ("ran", "running"),
    ("climbing", "climbing"), ("climbed", "climbing"),
    ("fell", "person_down"), ("fallen", "person_down"), ("lying", "person_down"), ("collapsed", "person_down"),
    ("intruder", "intrusion"), ("trespassing", "intrusion"), ("entered", "intrusion"),
];
const UNFAMILIAR: &[&str] = &["unfamiliar", "stranger", "strangers", "unknown", "unrecognised", "unrecognized"];

fn attr_word(w: &str) -> Option<(Attr, &'static str)> {
    ATTR_WORDS.iter().find(|(a, _, _)| *a == w).map(|(_, a, l)| (*a, *l))
}

/// Parse a people query. `known` = (person id, name); `cams` = id → camera name.
pub(crate) fn parse(query: &str, known: &[(String, String)], cams: &BTreeMap<i64, String>) -> PersonQuery {
    let toks = slots::tokens(query);
    let tokset: HashSet<&str> = toks.iter().map(String::as_str).collect();
    let mut q = PersonQuery::default();
    // Tokens a name or camera took — nothing else may reinterpret them.
    let mut claimed: HashSet<String> = HashSet::new();
    // Everything understood, so CLIP only gets what is left.
    let mut consumed: HashSet<String> = HashSet::new();

    // Names first, longest first, so someone called Rose never becomes a colour.
    let mut names: Vec<&(String, String)> = known.iter().collect();
    names.sort_by_key(|(_, n)| std::cmp::Reverse(n.len()));
    for (id, name) in names {
        let parts = slots::tokens(name);
        if !parts.is_empty() && parts.iter().all(|p| tokset.contains(p.as_str())) {
            q.person_id = Some(id.clone());
            q.understood.push(name.clone());
            claimed.extend(parts);
            break;
        }
    }
    for (id, name) in cams {
        let parts = slots::tokens(name);
        if !parts.is_empty() && parts.iter().all(|p| tokset.contains(p.as_str())) && !q.cams.contains(id) {
            q.cams.push(*id);
            q.understood.push(format!("{name} camera"));
            claimed.extend(parts);
        }
    }
    consumed.extend(claimed.iter().cloned());

    let (colours, _) = slots::outfit_slots(&toks, &claimed);
    for (band, colour) in colours {
        q.colors.push((band, colour));
        q.understood.push(format!("{colour} {band}"));
    }
    // A colour word and the garment it describes are both understood.
    for (i, t) in toks.iter().enumerate() {
        if claimed.contains(t) || !slots::is_colour(t) { continue; }
        consumed.insert(t.clone());
        if let Some(g) = toks.iter().skip(i + 1).take(3).find(|w| slots::garment_band(w).is_some()) {
            consumed.insert(g.clone());
        }
    }

    // "no hat, backpack": a negator reaches forward only until the next thing it
    // could describe, so the backpack stays wanted.
    let negated = |i: usize| {
        for w in toks[i.saturating_sub(3)..i].iter().rev() {
            if NEGATORS.contains(&w.as_str()) { return true; }
            if attr_word(w).is_some() || slots::garment_band(w).is_some() { return false; }
        }
        false
    };

    for (i, t) in toks.iter().enumerate() {
        if claimed.contains(t) { continue; }
        let next = toks.get(i + 1).map(String::as_str);
        let prev = i.checked_sub(1).and_then(|p| toks.get(p)).map(String::as_str);
        if t == "bag" && prev == Some("shoulder") { continue; } // taken with "shoulder"
        let two_word = match (t.as_str(), next) {
            ("shoulder", Some("bag")) => Some((Attr::Index(16), "shoulder bag")),
            ("long", Some("sleeve" | "sleeves" | "sleeved")) => Some((Attr::Index(3), "long sleeves")),
            ("short", Some("sleeve" | "sleeves" | "sleeved")) => Some((Attr::Index(2), "short sleeves")),
            _ => None,
        };
        if let (Some(_), Some(n)) = (two_word, next) { consumed.insert(n.to_string()); }
        if let Some((attr, label)) = two_word.or_else(|| attr_word(t)) {
            let want = !negated(i);
            q.push_attr(attr, want, if want { label.to_string() } else { format!("no {label}") });
            consumed.insert(t.clone());
            continue;
        }
        let female = FEMALE.contains(&t.as_str());
        if female || MALE.contains(&t.as_str()) {
            q.push_attr(Attr::Index(22), female, if female { "woman".into() } else { "man".into() });
            if t == "girl" || t == "boy" { q.push_attr(Attr::Index(19), true, "child".into()); }
            consumed.insert(t.clone());
            continue;
        }
        if let Some((_, b)) = BEHAVIOURS.iter().find(|(w, _)| w == t) {
            if !q.behaviours.contains(b) { q.behaviours.push(b); q.understood.push(b.replace('_', " ")); }
            consumed.insert(t.clone());
            continue;
        }
        if UNFAMILIAR.contains(&t.as_str()) {
            if !q.unfamiliar { q.unfamiliar = true; q.understood.push("unfamiliar".into()); }
            consumed.insert(t.clone());
            continue;
        }
        if NEGATORS.contains(&t.as_str()) { consumed.insert(t.clone()); }
    }
    q.residual = slots::keywords(&toks, &consumed);
    q
}

// ─── Scoring ──────────────────────────────────────────────────────────────────

/// Evidence we don't have scores between a match and a contradiction.
const UNKNOWN_COLOR: f32 = 0.15;
const UNKNOWN_ATTR: f32 = 0.3;
const FLOOR: f32 = 0.02;
/// Results scoring this far below the best are dropped (weak tail).
const TAIL: f32 = 3.0;

#[derive(Clone, Debug, Default)]
pub(crate) struct TrackRow {
    pub id: String,
    pub cam_id: i64,
    pub event_id: Option<String>,
    pub started_at: String,
    pub ended_at: String,
    pub body_person_id: Option<String>,
    pub known_person_id: Option<String>,
    pub identity_method: Option<String>,
    pub top_color: Option<String>,
    pub bottom_color: Option<String>,
    pub colors: Option<Vec<f32>>,
    pub par: Option<Vec<f32>>,
    pub behaviours: Vec<String>,
    pub reid: Option<Vec<f32>>,
    pub clip: Option<Vec<f32>>,
}

/// Log-likelihood-style score of one track against the structured slots.
/// `None` = excluded (a requested behaviour that never happened on this track).
pub(crate) fn slot_score(q: &PersonQuery, t: &TrackRow) -> Option<f32> {
    if q.behaviours.iter().any(|b| !t.behaviours.iter().any(|x| x == b)) { return None; }
    let mut s = 0.0f32;
    for (band, colour) in &q.colors {
        let base = if *band == "bottom" { 11 } else { 0 };
        let idx = COLOR_NAMES.iter().position(|c| c == colour)?;
        let p = match &t.colors {
            Some(v) if v.len() == 22 && v[base..base + 11].iter().sum::<f32>() > 0.0 => v[base + idx].max(FLOOR),
            _ => UNKNOWN_COLOR,
        };
        s += p.ln();
    }
    for (attr, want) in &q.attrs {
        let p = match (&t.par, attr) {
            (Some(v), Attr::Index(i)) if v.len() == crate::par::N => Some(v[*i]),
            (Some(v), Attr::AnyBag) if v.len() == crate::par::N => Some(v[15].max(v[16]).max(v[17])),
            _ => None,
        };
        s += match p {
            Some(p) => (if *want { p } else { 1.0 - p }).max(FLOOR).ln(),
            None => UNKNOWN_ATTR.ln(),
        };
    }
    Some(s)
}

// ─── Results ──────────────────────────────────────────────────────────────────

#[derive(Serialize, Clone, Debug)]
pub struct TrackHit {
    pub id: String,
    pub cam_id: i64,
    pub event_id: Option<String>,
    /// `body_*` (anonymous) or `kp_<id>` (proposed) — what "This is…" binds.
    pub body_person_id: Option<String>,
    pub started_at: String,
    pub ended_at: String,
    pub top_color: Option<String>,
    pub bottom_color: Option<String>,
    /// What they visibly wear/carry (never gender or age).
    pub evidence: Vec<&'static str>,
    pub behaviours: Vec<String>,
    pub identity_method: Option<String>,
    pub score: f32,
}

#[derive(Serialize, Clone, Debug)]
pub struct PersonGroup {
    pub key: String,
    /// Face-confirmed identity.
    pub person_id: Option<String>,
    pub name: Option<String>,
    /// Body-appearance proposal only ("Maybe X?").
    pub maybe_name: Option<String>,
    pub score: f32,
    pub first_seen: String,
    pub last_seen: String,
    pub cameras: Vec<i64>,
    /// Best first.
    pub tracks: Vec<TrackHit>,
}

#[derive(Serialize, Debug, Default)]
pub struct PeopleSearchResult {
    pub groups: Vec<PersonGroup>,
    pub understood: Vec<String>,
    /// Words the search could not use (e.g. no CLIP model for "hoodie").
    pub ignored: Vec<String>,
}

/// Which visit a track belongs to: a known person, else a body identity, else
/// "someone anonymous on this camera".
///
/// The last case used to be the track's own id, so every fragment was its own
/// visit. On real footage one man at a desk broke into 164 such fragments in a
/// day, and Today showed ~170 "Unfamiliar" tiles of him. Keyed by camera,
/// `build_visits`' gap rule merges back-to-back fragments into one stay.
/// ponytail: two strangers overlapping on one camera share a visit — fine for
/// "who was here, when"; split by appearance if that ever matters.
fn group_key(t: &TrackRow) -> String {
    match (&t.known_person_id, &t.body_person_id) {
        (Some(k), _) => format!("p:{k}"),
        (None, Some(b)) => format!("b:{b}"),
        _ => format!("u:{}", t.cam_id),
    }
}

fn hit(t: &TrackRow, score: f32) -> TrackHit {
    let evidence = t.par.as_ref().filter(|v| v.len() == crate::par::N)
        .map(|v| crate::par::evidence(&std::array::from_fn(|i| v[i]))).unwrap_or_default();
    TrackHit {
        id: t.id.clone(), cam_id: t.cam_id, event_id: t.event_id.clone(), body_person_id: t.body_person_id.clone(),
        started_at: t.started_at.clone(), ended_at: t.ended_at.clone(),
        top_color: t.top_color.clone(), bottom_color: t.bottom_color.clone(),
        evidence, behaviours: t.behaviours.clone(), identity_method: t.identity_method.clone(), score,
    }
}

fn name_fields(t: &TrackRow, names: &HashMap<String, String>) -> (Option<String>, Option<String>) {
    let name = t.known_person_id.as_ref().and_then(|k| names.get(k).cloned());
    let maybe = if t.known_person_id.is_none() {
        t.body_person_id.as_deref().and_then(|b| b.strip_prefix("kp_")).and_then(|k| names.get(k).cloned())
    } else { None };
    (name, maybe)
}

/// Group scored tracks by who they were; groups ordered by their best track.
pub(crate) fn group(scored: Vec<(TrackRow, f32)>, names: &HashMap<String, String>) -> Vec<PersonGroup> {
    let mut by: HashMap<String, PersonGroup> = HashMap::new();
    for (t, score) in scored {
        let key = group_key(&t);
        let (name, maybe_name) = name_fields(&t, names);
        let g = by.entry(key.clone()).or_insert_with(|| PersonGroup {
            key, person_id: t.known_person_id.clone(), name, maybe_name, score,
            first_seen: t.started_at.clone(), last_seen: t.ended_at.clone(), cameras: Vec::new(), tracks: Vec::new(),
        });
        g.score = g.score.max(score);
        if t.started_at < g.first_seen { g.first_seen = t.started_at.clone(); }
        if t.ended_at > g.last_seen { g.last_seen = t.ended_at.clone(); }
        if !g.cameras.contains(&t.cam_id) { g.cameras.push(t.cam_id); }
        g.tracks.push(hit(&t, score));
    }
    let mut out: Vec<PersonGroup> = by.into_values().collect();
    for g in &mut out {
        g.tracks.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.started_at.cmp(&a.started_at)));
    }
    out.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| b.last_seen.cmp(&a.last_seen)));
    out
}

// ─── Visits ───────────────────────────────────────────────────────────────────

#[derive(Serialize, Clone, Debug)]
pub struct Visit {
    pub key: String,
    pub person_id: Option<String>,
    pub name: Option<String>,
    pub maybe_name: Option<String>,
    pub start: String,
    pub end: String,
    /// In order of first appearance.
    pub cameras: Vec<i64>,
    pub behaviours: Vec<String>,
    /// Chronological hops.
    pub tracks: Vec<TrackHit>,
}

#[derive(Serialize, Debug, Default)]
pub struct PeopleDay {
    pub visits: Vec<Visit>,
    pub known_visits: usize,
    pub unfamiliar_visits: usize,
    /// Unfamiliar identities seen on two or more separate visits in the range.
    pub unfamiliar_repeat: usize,
}

fn ts(s: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(s).ok().map(|d| d.with_timezone(&Utc))
}

/// Chain one identity's tracks into visits: a gap of `gap_secs` or less between
/// one track's end and the next one's start is the same visit — which is what
/// turns Door → Hall → Garden into one row instead of three.
pub(crate) fn build_visits(mut rows: Vec<TrackRow>, gap_secs: i64, names: &HashMap<String, String>) -> Vec<Visit> {
    rows.sort_by(|a, b| a.started_at.cmp(&b.started_at));
    let mut open: HashMap<String, Visit> = HashMap::new();
    let mut done: Vec<Visit> = Vec::new();
    for t in rows {
        let key = group_key(&t);
        let continues = open.get(&key).is_some_and(|v| match (ts(&v.end), ts(&t.started_at)) {
            (Some(end), Some(start)) => (start - end).num_seconds() <= gap_secs,
            _ => false,
        });
        if !continues {
            if let Some(v) = open.remove(&key) { done.push(v); }
            let (name, maybe_name) = name_fields(&t, names);
            open.insert(key.clone(), Visit {
                key: key.clone(), person_id: t.known_person_id.clone(), name, maybe_name,
                start: t.started_at.clone(), end: t.ended_at.clone(),
                cameras: Vec::new(), behaviours: Vec::new(), tracks: Vec::new(),
            });
        }
        let Some(v) = open.get_mut(&key) else { continue };
        if t.ended_at > v.end { v.end = t.ended_at.clone(); }
        if !v.cameras.contains(&t.cam_id) { v.cameras.push(t.cam_id); }
        for b in &t.behaviours { if !v.behaviours.contains(b) { v.behaviours.push(b.clone()); } }
        v.tracks.push(hit(&t, 0.0));
    }
    done.extend(open.into_values());
    done.sort_by(|a, b| b.start.cmp(&a.start));
    done
}

// ─── Data access ──────────────────────────────────────────────────────────────

const COLS: &str = "id, cam_id, event_id, started_at, ended_at, body_person_id, known_person_id, \
                    identity_method, top_color, bottom_color, colors, par, behaviours, reid, clip, clip_model";

type Row = (String, i64, Option<String>, String, String, Option<String>, Option<String>, Option<String>,
            Option<String>, Option<String>, Option<Vec<u8>>, Option<Vec<u8>>, Option<String>,
            Option<Vec<u8>>, Option<Vec<u8>>, Option<String>);

fn to_track(r: Row, clip_model: &str) -> TrackRow {
    let blob = |b: Option<Vec<u8>>| b.map(|b| crate::embed::blob_to_vec(&b));
    TrackRow {
        id: r.0, cam_id: r.1, event_id: r.2, started_at: r.3, ended_at: r.4,
        body_person_id: r.5, known_person_id: r.6, identity_method: r.7,
        top_color: r.8, bottom_color: r.9, colors: blob(r.10), par: blob(r.11),
        behaviours: r.12.and_then(|j| serde_json::from_str(&j).ok()).unwrap_or_default(),
        reid: blob(r.13),
        clip: if !clip_model.is_empty() && r.15.as_deref() == Some(clip_model) { blob(r.14) } else { None },
    }
}

async fn load_track(db: &sqlx::SqlitePool, id: &str, clip_model: &str) -> Option<TrackRow> {
    sqlx::query_as::<_, Row>(&format!("SELECT {COLS} FROM person_tracks WHERE id=?"))
        .bind(id).fetch_optional(db).await.ok().flatten().map(|r| to_track(r, clip_model))
}

async fn load_tracks(
    db: &sqlx::SqlitePool, from: &str, to: &str, cams: &[i64],
    person: Option<&str>, unfamiliar: bool, clip_model: &str, limit: i64,
) -> Vec<TrackRow> {
    let mut sql = format!("SELECT {COLS} FROM person_tracks WHERE started_at >= ? AND started_at <= ?");
    if !cams.is_empty() { sql.push_str(&format!(" AND cam_id IN ({})", vec!["?"; cams.len()].join(","))); }
    if person.is_some() { sql.push_str(" AND known_person_id = ?"); }
    if unfamiliar { sql.push_str(" AND known_person_id IS NULL"); }
    sql.push_str(" ORDER BY started_at DESC LIMIT ?");
    let mut q = sqlx::query_as::<_, Row>(&sql).bind(from).bind(to);
    for c in cams { q = q.bind(c); }
    if let Some(p) = person { q = q.bind(p); }
    q.bind(limit).fetch_all(db).await.unwrap_or_default()
        .into_iter().map(|r| to_track(r, clip_model)).collect()
}

async fn known_names(db: &sqlx::SqlitePool) -> Vec<(String, String)> {
    sqlx::query_as("SELECT id, name FROM known_persons").fetch_all(db).await.unwrap_or_default()
}

fn window(from: Option<String>, to: Option<String>, days: i64) -> (String, String) {
    let now = Utc::now();
    (from.unwrap_or_else(|| (now - chrono::Duration::days(days)).to_rfc3339()),
     to.unwrap_or_else(|| now.to_rfc3339()))
}

/// Run a parsed query over the window. Shared by the People search box and the
/// Review search (which turns the result into events).
async fn run(state: &Arc<AppState>, q: &PersonQuery, from: &str, to: &str) -> PeopleSearchResult {
    let model = state.settings.read().await.search_model.clone();
    let rows = load_tracks(&state.db, from, to, &q.cams, q.person_id.as_deref(), q.unfamiliar, &model, 5000).await;

    let mut scored: Vec<(TrackRow, f32)> = rows.into_iter()
        .filter_map(|t| slot_score(q, &t).map(|s| (t, s))).collect();

    // CLIP only for the words structure couldn't take.
    let mut ignored = Vec::new();
    if !q.residual.is_empty() {
        let text = q.residual.join(" ");
        let (dd, m) = (state.data_dir.clone(), model.clone());
        let tvec = if crate::embed::is_installed(&dd, &m) {
            tokio::task::spawn_blocking(move || crate::embed::with_model(&dd, &m, |enc| {
                let a = enc.encode_text(&format!("a photo of a person, {text}")).ok()?;
                let b = enc.encode_text(&format!("a person wearing {text}")).ok()?;
                Some(a.iter().zip(&b).map(|(x, y)| (x + y) / 2.0).collect::<Vec<f32>>())
            }).flatten()).await.ok().flatten()
        } else { None };
        match tvec {
            Some(tv) => {
                let sims: Vec<Option<f32>> = scored.iter()
                    .map(|(t, _)| t.clip.as_ref().map(|c| crate::embed::cosine(&tv, c))).collect();
                let vals: Vec<f32> = sims.iter().flatten().copied().collect();
                if vals.len() >= 2 {
                    let mean = vals.iter().sum::<f32>() / vals.len() as f32;
                    let sd = (vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / vals.len() as f32).sqrt().max(1e-3);
                    for ((_, s), sim) in scored.iter_mut().zip(sims) {
                        if let Some(sim) = sim { *s += 0.7 * (sim - mean) / sd; }
                    }
                }
            }
            None => ignored = q.residual.clone(),
        }
    }

    if q.is_personal() || (!q.residual.is_empty() && ignored.is_empty()) {
        let best = scored.iter().map(|(_, s)| *s).fold(f32::MIN, f32::max);
        scored.retain(|(_, s)| *s >= best - TAIL);
    }
    let names: HashMap<String, String> = known_names(&state.db).await.into_iter().collect();
    let mut groups = group(scored, &names);
    groups.truncate(60);
    PeopleSearchResult { groups, understood: q.understood.clone(), ignored }
}

/// Event ids for a Review search, best person match first — `None` when the query
/// isn't about a person's appearance, identity or behaviour (keyword search then).
pub(crate) async fn event_ids_for_query(state: &Arc<AppState>, query: &str) -> Option<Vec<String>> {
    let known = known_names(&state.db).await;
    let cams = crate::agent::retrieve::camera_names(&state.db).await;
    let q = parse(query, &known, &cams);
    if !q.is_personal() { return None; }
    let (from, to) = window(None, None, 90);
    let res = run(state, &q, &from, &to).await;
    let mut seen = HashSet::new();
    let mut hits: Vec<(f32, String)> = res.groups.iter().flat_map(|g| g.tracks.iter())
        .filter_map(|t| t.event_id.clone().map(|e| (t.score, e))).collect();
    hits.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    Some(hits.into_iter().filter_map(|(_, e)| seen.insert(e.clone()).then_some(e)).collect())
}

// ─── Commands ─────────────────────────────────────────────────────────────────

#[tauri::command]
pub async fn search_people(
    state: State<'_, Arc<AppState>>,
    query: String,
    from: Option<String>,
    to: Option<String>,
    cams: Option<Vec<i64>>,
) -> Result<PeopleSearchResult, String> {
    let st = state.inner();
    let known = known_names(&st.db).await;
    let cam_names = crate::agent::retrieve::camera_names(&st.db).await;
    let mut q = parse(&query, &known, &cam_names);
    for c in cams.unwrap_or_default() { if !q.cams.contains(&c) { q.cams.push(c); } }
    let (from, to) = window(from, to, 30);
    Ok(run(st, &q, &from, &to).await)
}

/// "Find this person": nearest tracks by appearance (Re-ID mean), boosted when a
/// face ties them to the same person, never the same camera at the same time
/// (the tracker already said those are different people).
#[tauri::command]
pub async fn find_similar_person(
    state: State<'_, Arc<AppState>>,
    track_id: String,
    days: Option<i64>,
) -> Result<PeopleSearchResult, String> {
    let st = state.inner();
    let src = load_track(&st.db, &track_id, "").await.ok_or("track not found")?;
    let understood = vec!["looks like this person".to_string()];
    let Some(src_reid) = src.reid.clone() else {
        return Ok(PeopleSearchResult { understood,
            ignored: vec!["no appearance reading for this track".into()], ..Default::default() });
    };
    let (from, to) = window(None, None, days.unwrap_or(30));
    // ponytail: brute force over the window (≤20k × 256-d); a usearch index pays off past ~200k tracks.
    let rows = load_tracks(&st.db, &from, &to, &[], None, false, "", 20_000).await;
    let mut scored: Vec<(TrackRow, f32)> = rows.into_iter().filter(|t| t.id != src.id).filter_map(|t| {
        let r = t.reid.as_ref()?;
        if r.len() != src_reid.len() { return None; }
        let overlap = t.cam_id == src.cam_id && t.started_at <= src.ended_at && t.ended_at >= src.started_at;
        if overlap { return None; }
        let mut s = crate::embed::cosine(&src_reid, r);
        if src.known_person_id.is_some() && t.known_person_id == src.known_person_id { s += 0.1; }
        (s >= 0.5).then_some((t, s))
    }).collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(200);
    let names: HashMap<String, String> = known_names(&st.db).await.into_iter().collect();
    let mut groups = group(scored, &names);
    groups.truncate(40);
    Ok(PeopleSearchResult { groups, understood, ignored: Vec::new() })
}

/// Everyone who was here between `from` and `to`, as visits — or one person's
/// visits when `person_id` is given (People → person detail).
#[tauri::command]
pub async fn get_people_day(
    state: State<'_, Arc<AppState>>,
    from: String,
    to: String,
    person_id: Option<String>,
) -> Result<PeopleDay, String> {
    let st = state.inner();
    let rows = load_tracks(&st.db, &from, &to, &[], person_id.as_deref(), false, "", 5000).await;
    let names: HashMap<String, String> = known_names(&st.db).await.into_iter().collect();
    let visits = build_visits(rows, 180, &names);
    let known_visits = visits.iter().filter(|v| v.person_id.is_some()).count();
    let mut per_key: HashMap<&str, usize> = HashMap::new();
    for v in visits.iter().filter(|v| v.person_id.is_none()) { *per_key.entry(v.key.as_str()).or_default() += 1; }
    let unfamiliar_visits = per_key.values().sum();
    let unfamiliar_repeat = per_key.values().filter(|n| **n >= 2).count();
    Ok(PeopleDay { known_visits, unfamiliar_visits, unfamiliar_repeat, visits })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cams() -> BTreeMap<i64, String> { BTreeMap::from([(0, "Front Door".to_string()), (1, "Garden".to_string())]) }
    fn known() -> Vec<(String, String)> { vec![("p1".into(), "Rose".into()), ("p2".into(), "Ravi Kumar".into())] }

    #[test]
    fn parses_colours_attributes_negation_and_gender() {
        let q = parse("woman in a blue top and black jeans with a backpack, no hat", &known(), &cams());
        assert!(q.colors.contains(&("top", "blue")));
        assert!(q.colors.contains(&("bottom", "black")));
        assert!(q.attrs.contains(&(Attr::Index(17), true)), "backpack");
        assert!(q.attrs.contains(&(Attr::Index(0), false)), "no hat");
        assert!(q.attrs.contains(&(Attr::Index(22), true)), "woman");
        assert!(q.attrs.contains(&(Attr::Index(11), true)), "jeans → trousers, even with a colour on it");
        assert!(q.understood.contains(&"no hat".to_string()));
        assert!(q.residual.is_empty(), "garments bound to a colour don't go to CLIP: {:?}", q.residual);
        assert!(q.is_personal());
    }

    #[test]
    fn a_negation_stops_at_the_next_described_thing() {
        let q = parse("no hat, backpack", &[], &BTreeMap::new());
        assert!(q.attrs.contains(&(Attr::Index(0), false)));
        assert!(q.attrs.contains(&(Attr::Index(17), true)), "the backpack is still wanted");
        let m = parse("man without glasses", &[], &BTreeMap::new());
        assert!(m.attrs.contains(&(Attr::Index(22), false)));
        assert!(m.understood.contains(&"man".to_string()), "never 'no man'");
        assert!(m.attrs.contains(&(Attr::Index(1), false)));
    }

    #[test]
    fn names_cameras_behaviours_and_leftovers() {
        let q = parse("Rose loitering at the Front Door in a hoodie", &known(), &cams());
        assert_eq!(q.person_id.as_deref(), Some("p1"), "Rose is a person, not a colour");
        assert!(q.colors.is_empty());
        assert_eq!(q.cams, vec![0]);
        assert_eq!(q.behaviours, vec!["loitering"]);
        assert_eq!(q.residual, vec!["hoodie".to_string()]);

        let q = parse("unfamiliar person with a shoulder bag", &known(), &cams());
        assert!(q.unfamiliar);
        assert!(q.attrs.contains(&(Attr::Index(16), true)));
        assert!(!q.attrs.contains(&(Attr::AnyBag, true)), "shoulder bag is not also 'any bag'");
        assert!(q.residual.is_empty());

        let plain = parse("delivery van", &known(), &cams());
        assert!(!plain.is_personal());
    }

    fn row(colors: Option<Vec<f32>>, par: Option<Vec<f32>>) -> TrackRow {
        TrackRow { id: "t".into(), started_at: "2026-09-15T10:00:00Z".into(), ended_at: "2026-09-15T10:01:00Z".into(),
                   colors, par, ..Default::default() }
    }

    #[test]
    fn a_match_beats_unknown_which_beats_a_contradiction() {
        let q = parse("blue top", &[], &BTreeMap::new());
        let mut blue = vec![0.0f32; 22]; blue[8] = 0.8; blue[0] = 0.2;
        let mut red = vec![0.0f32; 22]; red[4] = 0.9; red[0] = 0.1;
        let m = slot_score(&q, &row(Some(blue), None)).unwrap();
        let u = slot_score(&q, &row(None, None)).unwrap();
        let c = slot_score(&q, &row(Some(red), None)).unwrap();
        assert!(m > u && u > c, "{m} > {u} > {c}");
    }

    #[test]
    fn negation_scores_the_absence() {
        let q = parse("no backpack", &[], &BTreeMap::new());
        let mut has = vec![0.0f32; 26]; has[17] = 0.9;
        let none = vec![0.0f32; 26];
        assert!(slot_score(&q, &row(None, Some(none))).unwrap() > slot_score(&q, &row(None, Some(has))).unwrap());
    }

    #[test]
    fn a_behaviour_is_a_hard_filter() {
        let q = parse("running", &[], &BTreeMap::new());
        let mut ran = row(None, None); ran.behaviours = vec!["running".into()];
        assert!(slot_score(&q, &ran).is_some());
        assert!(slot_score(&q, &row(None, None)).is_none());
    }

    #[test]
    fn visits_chain_hops_within_the_gap_and_split_after_it() {
        let mk = |id: &str, cam: i64, s: &str, e: &str, body: &str| TrackRow {
            id: id.into(), cam_id: cam, started_at: s.into(), ended_at: e.into(),
            body_person_id: Some(body.into()), ..Default::default()
        };
        let rows = vec![
            mk("a", 0, "2026-09-15T08:00:00Z", "2026-09-15T08:01:00Z", "body_x"),
            mk("b", 1, "2026-09-15T08:02:30Z", "2026-09-15T08:04:00Z", "body_x"), // 90 s later → same visit
            mk("c", 0, "2026-09-15T12:00:00Z", "2026-09-15T12:01:00Z", "body_x"), // hours later → new visit
            mk("d", 1, "2026-09-15T08:02:00Z", "2026-09-15T08:03:00Z", "body_y"), // someone else
        ];
        let v = build_visits(rows, 180, &HashMap::new());
        assert_eq!(v.len(), 3);
        let morning = v.iter().find(|v| v.key == "b:body_x" && v.start.starts_with("2026-09-15T08")).unwrap();
        assert_eq!(morning.cameras, vec![0, 1], "Door → Garden, in order");
        assert_eq!(morning.tracks.len(), 2);
        assert_eq!(morning.end, "2026-09-15T08:04:00Z");
    }

    #[test]
    fn anonymous_fragments_on_one_camera_are_one_visit() {
        // One unidentified person at a desk: the tracker drops and re-acquires
        // them every few seconds, and no fragment carries a person or body id.
        let mk = |id: &str, cam: i64, s: &str, e: &str| TrackRow {
            id: id.into(), cam_id: cam, started_at: s.into(), ended_at: e.into(), ..Default::default()
        };
        let rows = vec![
            mk("f1", 0, "2026-09-16T09:00:00Z", "2026-09-16T09:00:04Z"),
            mk("f2", 0, "2026-09-16T09:00:06Z", "2026-09-16T09:00:09Z"),
            mk("f3", 0, "2026-09-16T09:01:00Z", "2026-09-16T09:01:30Z"),
            mk("f4", 0, "2026-09-16T15:00:00Z", "2026-09-16T15:00:05Z"), // hours later → new visit
            mk("g1", 1, "2026-09-16T09:00:02Z", "2026-09-16T09:00:08Z"), // other camera → its own visit
        ];
        let v = build_visits(rows, 180, &HashMap::new());
        assert_eq!(v.len(), 3, "not five tiles");
        let morning = v.iter().find(|v| v.key == "u:0" && v.start.starts_with("2026-09-16T09")).unwrap();
        assert_eq!(morning.tracks.len(), 3);
        assert_eq!((morning.name.as_deref(), morning.maybe_name.as_deref()), (None, None), "still unfamiliar, not guessed");
    }

    #[test]
    fn a_body_proposal_is_a_maybe_not_a_name() {
        let names = HashMap::from([("p1".to_string(), "Rose".to_string())]);
        let t = TrackRow { id: "t".into(), body_person_id: Some("kp_p1".into()), ..Default::default() };
        assert_eq!(name_fields(&t, &names), (None, Some("Rose".to_string())));
        let f = TrackRow { known_person_id: Some("p1".into()), ..t };
        assert_eq!(name_fields(&f, &names), (Some("Rose".to_string()), None));
    }
}
