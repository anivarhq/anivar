//! THE evidence resolver — the agent's reply, parsed once.
//!
//! Guardian used to return a string with `[BRACKET]` tags and let each surface
//! re-read it. The app used the registry scanner [`super::tools::parse_tags`];
//! Telegram hand-rolled `next_tag()` across twelve separate blocks in
//! `dispatch::execute_telegram_commands`. Nothing forced the two to agree, and
//! they drifted badly:
//!
//! * Both surfaces already called `conditions::explore_events`, which returns a
//!   full card — time, camera, duration, AI summary, risk, thumbnail. Telegram
//!   read three of those fields and threw the other five away, so the phone got
//!   a row of text where the desktop got a picture.
//! * `[SHARE_CLIP]` minted a real Tailscale URL on Telegram and minted *nothing*
//!   in the app — even though the system prompt tells the model to use that tag
//!   whenever the user asks for "a link".
//! * The app showed 20 events, Telegram showed 8.
//!
//! So the tags are resolved HERE, once, into typed [`Evidence`]. Surfaces render
//! `Vec<Evidence>`; they never see the tag string. A field added to a card now
//! reaches both faces or neither — it can no longer reach one.
//!
//! What this module deliberately does NOT do: execute query tools. Those are
//! model-facing text, spliced by `tools::execute_query_tags` before the reply
//! ever gets here.

use std::sync::Arc;

use crate::AppState;

/// One event as every surface renders it — the typed form of
/// `conditions::card()`. Telegram lost five of these fields for a year because
/// it read the JSON by hand; now the compiler carries them across.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EventCard {
    pub id:         String,
    pub started_at: String,
    /// Pre-formatted local time, e.g. "Aug 03 21:14".
    pub ts:         String,
    #[serde(default)] pub cam:         i64,
    #[serde(default)] pub duration:    Option<String>,
    #[serde(default)] pub summary:     Option<String>,
    #[serde(default)] pub risk_level:  String,
    #[serde(default)] pub threat_type: String,
    #[serde(default)] pub has_clip:    bool,
    /// Bare base64 JPEG (blob refs already resolved by `card()`).
    #[serde(default)] pub thumbnail:   Option<String>,
}

impl EventCard {
    /// The one-line identity of an event: when, where, what. Used for button
    /// labels and album captions, so a list of twelve "person" events is
    /// actually distinguishable.
    pub fn headline(&self, cam_name: &str) -> String {
        let dur = self.duration.as_deref().map(|d| format!(" · {d}")).unwrap_or_default();
        format!("{} · {}{} · {}", self.ts, cam_name, dur, self.label())
    }

    /// What was seen. Prefers the AI summary, falls back to the threat type.
    pub fn label(&self) -> String {
        let s = super::memory::extract_summary_text(self.summary.as_deref().unwrap_or(""));
        if s.is_empty() {
            if self.threat_type.is_empty() { "motion".into() } else { self.threat_type.clone() }
        } else {
            s.chars().take(90).collect()
        }
    }

    /// Risk badge, or empty for the unremarkable majority.
    pub fn risk_badge(&self) -> &'static str {
        match self.risk_level.as_str() {
            "critical"              => "🚨 critical",
            "suspicious" | "high"   => "⚠️ suspicious",
            _                       => "",
        }
    }

    /// Decoded thumbnail bytes, ready for a Telegram upload.
    pub fn thumb_bytes(&self) -> Option<Vec<u8>> {
        use base64::Engine;
        let t = self.thumbnail.as_deref()?.trim();
        if t.is_empty() { return None; }
        base64::engine::general_purpose::STANDARD.decode(t).ok().filter(|b| !b.is_empty())
    }
}

/// One piece of resolved evidence. Every surface renders the same set at the
/// same fidelity; only the medium differs (cards vs a photo album).
#[derive(Debug, Clone)]
pub enum Evidence {
    /// Matching events, in display order. `play` means the user asked for the
    /// footage ITSELF (`[SEND_CLIP]`), not just to see what matched — Telegram
    /// uploads the video, the app opens the player.
    Events { cards: Vec<EventCard>, play: bool, label: String },
    /// A live look at a camera. Deliberately unresolved: a snapshot should be
    /// as fresh as the moment it is sent, not as fresh as the moment the model
    /// decided to send one.
    Snapshot { cam: u8, burst: bool },
    /// An enrolled person's photo.
    Person { name: String, thumbnail: Option<String> },
    /// A minted share URL that works from outside the LAN.
    Link { kind: &'static str, url: String, expires_at: i64, label: String },
    /// Today's activity by hour. Rendered as a mermaid diagram in the app and
    /// as ASCII on Telegram, from the same SQL.
    Chart,
}

/// Parse the agent's reply ONCE and resolve every action tag against live data.
///
/// Returns the prose with all tags stripped, plus the evidence they stood for.
///
/// This runs DOWNSTREAM of persistence: `chat_log` deliberately stores the reply
/// with its tags INTACT, because each surface post-processes them differently.
/// So anything reading a stored reply back must go through [`replay`] or
/// [`strip_tags`] — this comment used to claim a tag could never reach the log,
/// and on the strength of that the restore path rendered `content` raw and showed
/// the user a wall of `[SHOW_EVENTS:ids=…]`.
pub async fn resolve(state: &Arc<AppState>, reply: &str) -> (String, Vec<Evidence>) {
    let mut text = reply.to_string();
    let mut out: Vec<Evidence> = Vec::new();

    // Right to left, so each splice leaves the earlier spans valid.
    let calls = super::tools::parse_tags(&text);
    for call in calls.into_iter().rev() {
        let arg = |k: &str| call.args.get(k).and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
        let num = |k: &str| call.args.get(k).and_then(|v| v.as_i64());
        // Query tools are model-facing text and already ran upstream.
        if call.tool.tag.is_empty() { continue; }

        // A tool the user switched off in Settings renders on NO surface. This
        // gate used to exist only on Telegram, so a disabled camera-snapshot
        // tool still put live pictures in the desktop chat.
        if let Some(key) = gate_key(call.tool.name) {
            if !super::dispatch::tool_enabled(&state.db, key).await {
                text.replace_range(call.span.clone(), "");
                continue;
            }
        }

        let ev: Option<Evidence> = match call.tool.name {
            "show_events" => {
                let filter = { let f = arg("filter"); if f.is_empty() { "today".into() } else { f } };
                let cards = cards_for(state, &filter).await;
                (!cards.is_empty()).then(|| Evidence::Events {
                    label: describe(&filter), cards, play: false,
                })
            }
            // Search results ARE events. They used to be a plain-text splice in
            // the app and a separate button list on Telegram — two renderings of
            // the same rows, neither carrying a picture.
            "search_events" => {
                let q = arg("query");
                let cards = cards_for_ids(state, &search_ids(state, &q).await).await;
                if cards.is_empty() {
                    // A search that found nothing must SAY so. Dropping the tag
                    // silently leaves prose promising results that never arrive.
                    text.replace_range(call.span.clone(), &format!("(nothing matched “{q}”)"));
                    continue;
                }
                Some(Evidence::Events { label: format!("matches for “{q}”"), cards, play: false })
            }
            "send_clip" => {
                let id = arg("event_id");
                let cards = cards_for(state, &format!("ids={id}")).await;
                if cards.is_empty() {
                    text.replace_range(call.span.clone(), "(that event is no longer in the archive)");
                    continue;
                }
                Some(Evidence::Events { label: "the clip".into(), cards, play: true })
            }
            "snapshot"   => Some(Evidence::Snapshot { cam: num("camera").unwrap_or(0) as u8, burst: false }),
            "live_video" => Some(Evidence::Snapshot { cam: num("camera").unwrap_or(0) as u8, burst: true }),
            "send_person" => {
                let name = arg("name");
                if name.is_empty() { None } else {
                    Some(Evidence::Person { thumbnail: person_thumb(state, &name).await, name })
                }
            }
            // Both share tags mint through the SAME bounded, idempotent path the
            // Telegram buttons use — so the app finally gets a real URL instead
            // of silently degrading to an inline card.
            "share_clip" | "share_live" => {
                let (kind, resource) = if call.tool.name == "share_clip" {
                    ("clip", arg("event_id"))
                } else {
                    ("live", num("camera").unwrap_or(0).to_string())
                };
                let mins = num("minutes").map(|m| m as u32).unwrap_or(
                    state.settings.read().await.live_share_default_minutes);
                if resource.is_empty() { None } else {
                    match super::dispatch::mint_link_bounded(state, kind, &resource, mins).await {
                        Ok((url, expires_at)) => Some(Evidence::Link {
                            kind: if kind == "clip" { "clip" } else { "live" },
                            url, expires_at,
                            label: if kind == "clip" { "Private clip link".into() }
                                   else { format!("Live view · camera {resource}") },
                        }),
                        // Link setup failures carry actionable guidance (the
                        // Tailscale consent URL). Surface it as prose, not silence.
                        Err(msg) => { text.replace_range(call.span.clone(), &msg); continue; }
                    }
                }
            }
            "day_chart" => Some(Evidence::Chart),
            // Alert-rule tools answer in words. Splice their result in place.
            "subscribe_alert" | "unsubscribe_alert" | "list_rules" => {
                let out = super::run_tool(state, call.tool.name, &call.args).await;
                text.replace_range(call.span.clone(), out.trim());
                continue;
            }
            // `remember` is honoured upstream by `process_remember_tags`; here it
            // is only stripped so the user never sees the bracket.
            _ => None,
        };

        if let Some(e) = ev { out.push(e); }
        text.replace_range(call.span.clone(), "");
    }

    // Collected right-to-left; restore document order.
    out.reverse();
    // Merge adjacent event groups so twelve events asked for in two tags still
    // render as ONE album rather than two.
    merge_event_groups(&mut out);
    (text.trim().to_string(), out)
}

/// Settings key gating a tool, if any. The person key deliberately keeps its
/// historical `send_person_photo` spelling so existing user toggles still apply.
fn gate_key(tool: &str) -> Option<&'static str> {
    match tool {
        "snapshot"    => Some("snapshot"),
        "live_video"  => Some("live_video"),
        "send_clip"   => Some("send_clip"),
        "send_person" => Some("send_person_photo"),
        _ => None,
    }
}

/// One event as a card, for the surfaces that start from an id (a Telegram
/// button, an alert) rather than from a tag.
pub async fn card_for(state: &Arc<AppState>, event_id: &str) -> Option<EventCard> {
    cards_for(state, &format!("ids={event_id}")).await.into_iter().next()
}

/// Cards for an exact id set, in the order given.
pub async fn cards_for_ids(state: &Arc<AppState>, ids: &[String]) -> Vec<EventCard> {
    if ids.is_empty() { return Vec::new(); }
    cards_for(state, &format!("ids={}", ids.join(","))).await
}

/// Cards for a filter — the single card query both surfaces already shared.
async fn cards_for(state: &Arc<AppState>, filter: &str) -> Vec<EventCard> {
    super::conditions::explore_events(state, filter).await
        .into_iter()
        .filter_map(|v| serde_json::from_value(v).ok())
        .collect()
}

/// Strip every action tag from a reply, leaving only prose.
///
/// `chat_log` stores replies with their tags INTACT (see [`resolve`]), so every
/// reader of a stored reply needs this. Without it the model is fed its own tag
/// syntax back as conversation history — the one thing the prompt forbids it to
/// emit.
pub fn strip_tags(text: &str) -> String {
    let mut out = text.to_string();
    // Right to left, so each splice leaves the earlier spans valid.
    for call in super::tools::parse_tags(text).into_iter().rev() {
        if call.tool.tag.is_empty() { continue; }
        out.replace_range(call.span.clone(), "");
    }
    out.trim().to_string()
}

/// Re-render a PERSISTED reply for the conversation restore.
///
/// NOT [`resolve`]: that one has side effects — it mints Tailscale share URLs —
/// so replaying history through it would hand out a fresh link every time the
/// app reopened. This strips every tag and re-hydrates the one kind that is safe
/// to replay: events, BY ID, straight from the database.
///
/// Only the `ids=` form comes back. A bare `[SHOW_EVENTS:today]` re-run on load
/// would put *today's* events under a week-old message, which is worse than
/// showing none. Snapshots are deliberately not replayed either (a snapshot is
/// as fresh as the moment it was asked for), and links are never re-minted.
///
/// `budget` is the shared card allowance across the whole restored conversation,
/// decremented as it is spent — cards carry base64 thumbnails, and an uncapped
/// restore is a fat payload on every mount. Rows past the budget still get their
/// prose; they just come back without pictures.
pub async fn replay(state: &Arc<AppState>, reply: &str, budget: &mut usize) -> (String, Vec<EventCard>) {
    let text = strip_tags(reply);
    if *budget == 0 { return (text, Vec::new()); }

    let mut ids = replay_ids(reply);
    ids.truncate(*budget);
    // Events deleted since the message was written simply come back as fewer
    // cards — the archive is the truth, not the transcript.
    let cards = cards_for_ids(state, &ids).await;
    *budget -= cards.len().min(*budget);
    (text, cards)
}

/// The event ids a persisted reply is safe to replay, in order, deduplicated.
///
/// Split out from [`replay`] because it is the whole decision — which tags may
/// come back and which must not — and it is pure string work, testable without
/// a database.
fn replay_ids(reply: &str) -> Vec<String> {
    let mut ids: Vec<String> = Vec::new();
    for call in super::tools::parse_tags(reply) {
        let arg = |k: &str| call.args.get(k).and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
        match call.tool.name {
            "show_events" => {
                // Same prefix `describe` keys the id form on.
                let f = arg("filter");
                if f.get(..4).is_some_and(|p| p.eq_ignore_ascii_case("ids=")) {
                    ids.extend(f[4..].split(',').map(str::trim).filter(|s| !s.is_empty()).map(String::from));
                }
            }
            "send_clip" => {
                let id = arg("event_id");
                if !id.is_empty() { ids.push(id); }
            }
            _ => {}
        }
    }
    let mut seen = std::collections::HashSet::new();
    ids.retain(|id| seen.insert(id.clone()));
    ids
}

/// Event ids matching a search, via THE search backend (six label columns plus
/// the CLIP semantic pass) — then re-read as cards so search results carry the
/// same picture and metadata as every other event.
pub async fn search_ids(state: &Arc<AppState>, query: &str) -> Vec<String> {
    if query.is_empty() { return Vec::new(); }
    crate::nvr_recording::search_events_core(state, query, Some(20)).await
        .unwrap_or_default()
        .into_iter().map(|e| e.id).collect()
}

async fn person_thumb(state: &Arc<AppState>, name: &str) -> Option<String> {
    let t: Option<String> = sqlx::query_scalar(
        "SELECT thumbnail FROM known_persons WHERE name = ? COLLATE NOCASE LIMIT 1"
    ).bind(name).fetch_optional(&state.db).await.ok().flatten().flatten();
    t.map(|t| crate::blobstore::resolve(&state.data_dir, &t)).filter(|t| !t.is_empty())
}

/// Human name for a filter, for the album header.
///
/// `str::get`, not `&f[..4]`: this string comes straight out of a model, and a
/// filter starting with a multi-byte character (`[SHOW_EVENTS:日本語]`) makes the
/// byte slice land mid-character and PANIC.
fn describe(filter: &str) -> String {
    let f = filter.trim();
    if f.get(..4).is_some_and(|p| p.eq_ignore_ascii_case("ids=")) { return "events".into(); }
    f.to_string()
}

/// Fold consecutive `Events` groups into one, dropping duplicate ids. Two tags
/// describing overlapping sets should not produce the same event twice.
fn merge_event_groups(items: &mut Vec<Evidence>) {
    let mut i = 0;
    while i + 1 < items.len() {
        let mergeable = matches!((&items[i], &items[i + 1]),
            (Evidence::Events { play: a, .. }, Evidence::Events { play: b, .. }) if a == b);
        if !mergeable { i += 1; continue; }
        let Evidence::Events { cards: next, .. } = items.remove(i + 1) else { unreachable!() };
        let Evidence::Events { cards, .. } = &mut items[i] else { unreachable!() };
        for c in next {
            if !cards.iter().any(|e| e.id == c.id) { cards.push(c); }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(id: &str) -> EventCard {
        EventCard {
            id: id.into(), started_at: "2026-08-03T21:14:00Z".into(), ts: "Aug 03 21:14".into(),
            cam: 0, duration: Some("12s".into()), summary: None,
            risk_level: "normal".into(), threat_type: "person".into(),
            has_clip: true, thumbnail: None,
        }
    }

    /// A card is the SAME shape on both surfaces, and the fields Telegram used
    /// to drop survive the round trip. This is the regression test for the
    /// drift that motivated the module.
    #[test]
    fn a_card_survives_the_json_round_trip() {
        let json = serde_json::json!({
            "id": "abc", "started_at": "2026-08-03T21:14:00Z", "ts": "Aug 03 21:14",
            "cam": 2, "duration": "12s", "summary": "Person approached the door",
            "risk_level": "suspicious", "threat_type": "person",
            "has_clip": true, "thumbnail": "AAAA",
        });
        let c: EventCard = serde_json::from_value(json).expect("card decodes");
        assert_eq!(c.cam, 2);
        assert_eq!(c.duration.as_deref(), Some("12s"));
        assert_eq!(c.risk_badge(), "⚠️ suspicious");
        assert!(c.headline("Front Door").contains("Front Door"),
                "the camera name reaches the button label");
        assert!(c.headline("Front Door").contains("12s"),
                "duration reaches the button label — Telegram used to drop it");
    }

    /// The restore-path regression: a persisted reply carries its brackets, and
    /// rendering `content` raw put a wall of ids in the chat bubble.
    #[test]
    fn a_stored_reply_strips_to_prose() {
        let raw = "Two people at the door.
[SHOW_EVENTS:ids=abc,def]";
        assert_eq!(strip_tags(raw), "Two people at the door.");
        assert!(!strip_tags(raw).contains('['), "no bracket survives into the bubble");
        // Prose containing a non-tag bracket is not a tag and must come through.
        assert_eq!(strip_tags("seen [maybe] once"), "seen [maybe] once");
    }

    /// Only the `ids=` form replays. A bare filter re-run at restore time would
    /// hang TODAY's events under a week-old message.
    #[test]
    fn only_id_addressed_tags_replay() {
        let ids = |r: &str| replay_ids(r);
        assert_eq!(ids("here[SHOW_EVENTS:ids=abc, def]"), vec!["abc", "def"]);
        assert_eq!(ids("here[SEND_CLIP:xyz]"), vec!["xyz"]);
        assert!(ids("here[SHOW_EVENTS:today]").is_empty(), "a filter must not re-run");
        assert!(ids("here[SNAPSHOT:0]").is_empty(), "a snapshot is never replayed");
        assert!(ids("here[SHARE_CLIP:abc:30]").is_empty(), "a link is never re-minted");
        assert_eq!(ids("[SHOW_EVENTS:ids=a,a][SEND_CLIP:a]"), vec!["a"], "deduplicated");
    }

    /// The exact reply the user was shown as gibberish: eleven ids in one tag,
    /// on its own line after prose. Verbatim on purpose — the earlier fix was
    /// reasoned about rather than run against a real stored row.
    #[test]
    fn the_reported_reply_comes_back_as_prose_and_eleven_ids() {
        let raw = concat!(
            "There were three events yesterday, between 19:24 and 19:35, involving a ",
            "person named Ranjith. The longest duration recorded was 300 seconds.

",
            "[SHOW_EVENTS:ids=75450c66-c423-4191-83b7-0dc6d59ba268,",
            "63454c62-583f-4399-b11b-48e65ac0e2b5,f8fed76b-6a57-4475-a74c-1485d0263f47,",
            "dd370a3a-f479-462d-8cbc-654277936aa0,5d2f6a1b-5210-4263-aad0-78e42a9d781c,",
            "5da5d74c-559e-411c-8cf5-6827110f5f6c,a852df40-f8ee-4470-8132-67cb6139f316,",
            "d61c16c5-cc3c-497a-a873-83d424b9888a,87ab5eef-81d5-4dc3-854d-60c72a202a01,",
            "dd8367d6-0510-4a11-9f83-40ed8a975195,584ec548-4021-4a60-af24-ad71f65c3888]");
        let text = strip_tags(raw);
        assert!(text.ends_with("300 seconds."), "prose survives whole: {text:?}");
        assert!(!text.contains("SHOW_EVENTS"), "the bracket is gone: {text:?}");
        assert!(!text.contains("75450c66"), "no id leaks into the bubble: {text:?}");
        let ids = replay_ids(raw);
        assert_eq!(ids.len(), 11, "every id is recovered as a card: {ids:?}");
        assert_eq!(ids[0], "75450c66-c423-4191-83b7-0dc6d59ba268");
        assert_eq!(ids[10], "584ec548-4021-4a60-af24-ad71f65c3888");
    }

    /// An older reply, or a surface that has not been redeployed, must not blow
    /// up on a card that predates the `cam` column.
    #[test]
    fn a_card_without_the_new_fields_still_decodes() {
        let json = serde_json::json!({
            "id": "abc", "started_at": "2026-08-03T21:14:00Z", "ts": "Aug 03 21:14",
        });
        let c: EventCard = serde_json::from_value(json).expect("legacy card decodes");
        assert_eq!(c.cam, 0);
        assert_eq!(c.label(), "motion");
    }

    /// Filters come out of a MODEL. A byte slice through a multi-byte character
    /// panics, and `[SHOW_EVENTS:日本語]` is one token away at any time.
    #[test]
    fn a_non_ascii_filter_does_not_panic() {
        assert_eq!(describe("日本語"), "日本語");
        assert_eq!(describe("ids=abc,def"), "events");
        assert_eq!(describe("IDS=abc"), "events");
        assert_eq!(describe("é"), "é");
        assert_eq!(describe("today"), "today");
        assert_eq!(describe(""), "");
    }

    #[test]
    fn overlapping_event_groups_merge_into_one_album() {
        let mut items = vec![
            Evidence::Events { cards: vec![ev("a"), ev("b")], play: false, label: "today".into() },
            Evidence::Events { cards: vec![ev("b"), ev("c")], play: false, label: "today".into() },
        ];
        merge_event_groups(&mut items);
        assert_eq!(items.len(), 1, "one album, not two");
        let Evidence::Events { cards, .. } = &items[0] else { panic!() };
        assert_eq!(cards.iter().map(|c| c.id.as_str()).collect::<Vec<_>>(), ["a", "b", "c"],
                   "duplicate id dropped, order kept");
    }

    /// A clip request and a card list are different intents and must not merge —
    /// one uploads a video, the other shows an album.
    #[test]
    fn a_clip_request_does_not_merge_into_a_card_list() {
        let mut items = vec![
            Evidence::Events { cards: vec![ev("a")], play: true,  label: "the clip".into() },
            Evidence::Events { cards: vec![ev("b")], play: false, label: "today".into() },
        ];
        merge_event_groups(&mut items);
        assert_eq!(items.len(), 2);
    }
}
