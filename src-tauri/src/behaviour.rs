//! Person behaviours from tracks (and pose, where it earns its cost) — the rules
//! half of PP-Human's action layer. PP-Human's learned action models (ST-GCN
//! falls, PP-TSM fights) are not ported: their training data is non-commercial,
//! there is no ONNX, and upstream reports false falls on sitting and false fights
//! on dancing. Rules on stable tracks are explainable and tunable.
//!
//! Every rule is a per-(camera, track) state machine: evidence accumulates, the
//! rule CONFIRMS, fires ONCE for that track, and stays quiet. Events already
//! record on first evidence elsewhere; an alert here requires confirmation.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use crate::pose::Pose;
use crate::tracking::{point_in_poly, TrackPoint};

/// A track unseen this long is forgotten (a returning person is a new visit).
const GONE: Duration = Duration::from_secs(60);
/// Running: median foot speed in body-heights per second (walking ≈ 0.8).
const RUN_HEIGHTS_PER_SEC: f32 = 1.8;
/// Person down: torso this far from vertical, after the track was seen upright…
const DOWN_DEG: f32 = 60.0;
const UPRIGHT_DEG: f32 = 30.0;
/// …held this long across at least this many pose samples.
const DOWN_HOLD: Duration = Duration::from_secs(10);
const DOWN_SAMPLES: u32 = 3;
/// Crowd: over threshold for this long, with no gap longer than CROWD_GAP.
const CROWD_HOLD: Duration = Duration::from_secs(5);
const CROWD_GAP: Duration = Duration::from_secs(3);
/// A fired crowd re-arms after the scene has been under threshold this long.
const CROWD_REARM: Duration = Duration::from_secs(30);
/// Pose sampling cadence per track.
const POSE_EVERY: Duration = Duration::from_secs(1);

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Zone {
    pub name: String,
    pub polygon: Vec<(f32, f32)>,
    pub alert_on_enter: bool,
    /// 0 = no zone loitering rule.
    pub loitering_secs: u32,
    /// Consecutive frames inside before an entry counts.
    pub inertia: u32,
    /// Sofa / bed / lawn: lying down here is not an incident.
    pub ignore_down: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Fence { pub name: String, pub a: (f32, f32), pub b: (f32, f32) }

#[derive(Clone, Debug, Default)]
pub(crate) struct Rules {
    pub zones: Vec<Zone>,
    pub fences: Vec<Fence>,
    pub intrusion: bool,
    /// Whole-frame dwell threshold; 0 = off.
    pub loiter_secs: u32,
    pub running: bool,
    pub down: bool,
    pub climbing: bool,
    /// People at once; 0 = off.
    pub crowd_threshold: u32,
}

impl Rules {
    pub(crate) fn wants_pose(&self) -> bool { self.down || (self.climbing && !self.fences.is_empty()) }
}

/// One tracked person this frame. Coordinates normalised 0..1 to the frame.
pub(crate) struct PersonFrame<'a> {
    pub track_id: u64,
    pub foot: (f32, f32),
    /// Box height in pixels, and the frame size, to turn motion into body-heights.
    pub box_h: f32,
    pub frame_w: f32,
    pub frame_h: f32,
    pub history: &'a [TrackPoint],
    pub pose: Option<&'a Pose>,
    /// Highest visible ankle, normalised to the frame (from `pose`).
    pub ankle_y: Option<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Kind { Intrusion, Loitering, Running, Down, Climbing, Crowd }

impl Kind {
    /// `dispatch_intelligence_alert` type — also the alert-settings category key.
    pub(crate) fn alert_type(self) -> &'static str {
        match self {
            Kind::Intrusion => "intrusion",
            Kind::Loitering => "loitering",
            Kind::Running   => "running",
            Kind::Down      => "person_down",
            Kind::Climbing  => "climbing",
            Kind::Crowd     => "crowd",
        }
    }
}

#[derive(Debug, PartialEq)]
pub(crate) struct Fired { pub track_id: u64, pub kind: Kind, pub detail: String }

#[derive(Default)]
struct ZoneState { inside: u32, dwell: f32, last_in: Option<Instant>, entered: bool, loitered: bool }

#[derive(Default)]
struct TrackState {
    first: Option<Instant>,
    last: Option<Instant>,
    max_h: f32,
    zones: HashMap<String, ZoneState>,
    loitered: bool,
    ran: bool,
    upright: bool,
    down_since: Option<Instant>,
    down_samples: u32,
    down_fired: bool,
    below_fence: bool,
    above_fence: u32,
    climbed: bool,
    last_pose: Option<Instant>,
}

#[derive(Default)]
pub(crate) struct CamState {
    tracks: HashMap<u64, TrackState>,
    crowd_since: Option<Instant>,
    crowd_last: Option<Instant>,
    crowd_fired: bool,
}

/// Should this track get a pose sample now? Only behaviour candidates do, at
/// most once per `POSE_EVERY`: fresh tracks (to learn the upright baseline),
/// boxes that shrank (bending, falling), anyone already "down", and everyone
/// while the camera has a fence line.
pub(crate) fn pose_due(cam: &mut CamState, rules: &Rules, track_id: u64, box_h: f32, now: Instant) -> bool {
    if !rules.wants_pose() || track_id == 0 { return false; }
    let t = cam.tracks.entry(track_id).or_default();
    if t.last_pose.is_some_and(|p| now.duration_since(p) < POSE_EVERY) { return false; }
    let young = t.first.is_none_or(|f| now.duration_since(f) < Duration::from_secs(3));
    let shrank = t.max_h > 0.0 && box_h < t.max_h * 0.65;
    let due = young || shrank || t.down_since.is_some() || (rules.climbing && !rules.fences.is_empty());
    if due { t.last_pose = Some(now); }
    due
}

/// Advance every rule by one frame. Returns what confirmed THIS frame.
pub(crate) fn step(cam: &mut CamState, rules: &Rules, people: &[PersonFrame], now: Instant) -> Vec<Fired> {
    let mut fired = Vec::new();
    cam.tracks.retain(|_, t| t.last.is_none_or(|l| now.duration_since(l) < GONE));

    for p in people.iter().filter(|p| p.track_id != 0) {
        let t = cam.tracks.entry(p.track_id).or_default();
        let dt = t.last.map_or(0.0, |l| now.duration_since(l).as_secs_f32().min(2.0));
        t.first.get_or_insert(now);
        t.last = Some(now);
        t.max_h = t.max_h.max(p.box_h);

        // ── Zones: intrusion (N consecutive frames inside) + zone dwell ──
        for z in &rules.zones {
            let zs = t.zones.entry(z.name.clone()).or_default();
            if point_in_poly(p.foot.0, p.foot.1, &z.polygon) {
                zs.inside += 1;
                if zs.last_in.is_some() { zs.dwell += dt; }
                zs.last_in = Some(now);
                if rules.intrusion && z.alert_on_enter && !zs.entered && zs.inside >= z.inertia.max(1) {
                    zs.entered = true;
                    fired.push(Fired { track_id: p.track_id, kind: Kind::Intrusion, detail: z.name.clone() });
                }
                if z.loitering_secs > 0 && !zs.loitered && zs.dwell >= z.loitering_secs as f32 {
                    zs.loitered = true;
                    fired.push(Fired { track_id: p.track_id, kind: Kind::Loitering,
                                       detail: format!("{} for {:.0}s", z.name, zs.dwell) });
                }
            } else {
                zs.inside = 0;
                zs.last_in = None;
            }
        }

        // ── Whole-frame dwell ──
        if rules.loiter_secs > 0 && !t.loitered {
            let dwell = t.first.map_or(0.0, |f| now.duration_since(f).as_secs_f32());
            if dwell >= rules.loiter_secs as f32 {
                t.loitered = true;
                fired.push(Fired { track_id: p.track_id, kind: Kind::Loitering, detail: format!("in view for {dwell:.0}s") });
            }
        }

        // ── Running ──
        if rules.running && !t.ran {
            if let Some(v) = heights_per_sec(p.history, p.box_h, p.frame_w, p.frame_h, now) {
                if v >= RUN_HEIGHTS_PER_SEC {
                    t.ran = true;
                    fired.push(Fired { track_id: p.track_id, kind: Kind::Running, detail: format!("{v:.1} body-heights/s") });
                }
            }
        }

        // ── Pose rules (only on frames that carry a pose sample) ──
        if let Some(pose) = p.pose {
            if rules.down && !t.down_fired {
                if let Some(angle) = pose.torso_angle() {
                    let excused = rules.zones.iter().any(|z| z.ignore_down && point_in_poly(p.foot.0, p.foot.1, &z.polygon));
                    if angle <= UPRIGHT_DEG {
                        t.upright = true;
                        t.down_since = None;
                        t.down_samples = 0;
                    } else if angle >= DOWN_DEG && t.upright && !excused {
                        let since = *t.down_since.get_or_insert(now);
                        t.down_samples += 1;
                        if t.down_samples >= DOWN_SAMPLES && now.duration_since(since) >= DOWN_HOLD {
                            t.down_fired = true;
                            fired.push(Fired { track_id: p.track_id, kind: Kind::Down,
                                               detail: format!("{:.0}s", now.duration_since(since).as_secs_f32()) });
                        }
                    } else if angle < 45.0 {
                        t.down_since = None;
                        t.down_samples = 0;
                    }
                }
            }
            if rules.climbing && !t.climbed {
                if let Some(ay) = p.ankle_y {
                    for f in &rules.fences {
                        let Some(ly) = line_y_at(f, p.foot.0) else { continue };
                        if ay > ly + 0.01 { t.below_fence = true; t.above_fence = 0; }
                        else if ay < ly - 0.01 && t.below_fence {
                            t.above_fence += 1;
                            if t.above_fence >= 2 {
                                t.climbed = true;
                                fired.push(Fired { track_id: p.track_id, kind: Kind::Climbing, detail: f.name.clone() });
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    // ── Crowd: sustained count, once per episode ──
    if rules.crowd_threshold > 0 {
        let count = people.iter().filter(|p| p.track_id != 0).count() as u32;
        if count >= rules.crowd_threshold {
            let since = match (cam.crowd_since, cam.crowd_last) {
                (Some(s), Some(l)) if now.duration_since(l) <= CROWD_GAP => s,
                _ => now,
            };
            cam.crowd_since = Some(since);
            cam.crowd_last = Some(now);
            if !cam.crowd_fired && now.duration_since(since) >= CROWD_HOLD {
                cam.crowd_fired = true;
                fired.push(Fired { track_id: 0, kind: Kind::Crowd, detail: format!("{count} people") });
            }
        } else if cam.crowd_last.is_none_or(|l| now.duration_since(l) >= CROWD_REARM) {
            cam.crowd_fired = false;
            cam.crowd_since = None;
        }
    }
    fired
}

/// Median foot speed over the last 1.5 s in body-heights per second. `None`
/// until at least a second of motion history exists.
fn heights_per_sec(history: &[TrackPoint], box_h: f32, fw: f32, fh: f32, now: Instant) -> Option<f32> {
    if box_h < 24.0 { return None; }
    let pts: Vec<&TrackPoint> = history.iter()
        .filter(|p| now.duration_since(p.t) <= Duration::from_millis(1500)).collect();
    if pts.len() < 3 { return None; }
    let span = pts.last()?.t.duration_since(pts.first()?.t).as_secs_f32();
    if span < 1.0 { return None; }
    let mut v: Vec<f32> = pts.windows(2).filter_map(|w| {
        let dt = w[1].t.duration_since(w[0].t).as_secs_f32();
        if dt <= 0.01 { return None; }
        let (dx, dy) = ((w[1].x - w[0].x) * fw, (w[1].y - w[0].y) * fh);
        Some((dx * dx + dy * dy).sqrt() / box_h / dt)
    }).collect();
    if v.is_empty() { return None; }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(v[v.len() / 2])
}

/// The fence line's y at `x`, when `x` lies within its horizontal span.
fn line_y_at(f: &Fence, x: f32) -> Option<f32> {
    let (x0, x1) = (f.a.0.min(f.b.0), f.a.0.max(f.b.0));
    if x < x0 || x > x1 || (f.b.0 - f.a.0).abs() < 1e-4 { return None; }
    Some(f.a.1 + (f.b.1 - f.a.1) * (x - f.a.0) / (f.b.0 - f.a.0))
}

/// Zones and fence lines for one camera from `settings.camera_masks` (MaskEditor
/// JSON). Zone options the editor always offered — alert on enter, loitering
/// seconds, inertia — used to be ignored by the backend.
pub(crate) fn parse_masks(masks_json: &str, cam_id: u8) -> (Vec<Zone>, Vec<Fence>) {
    let (mut zones, mut fences) = (Vec::new(), Vec::new());
    let Ok(map) = serde_json::from_str::<serde_json::Value>(masks_json) else { return (zones, fences) };
    let Some(arr) = map.get(cam_id.to_string()).and_then(|v| v.as_array()) else { return (zones, fences) };
    for m in arr {
        let nums: Vec<f32> = m.get("points").and_then(|p| p.as_str()).unwrap_or("")
            .split(',').filter_map(|s| s.trim().parse().ok()).collect();
        let name = m.get("name").and_then(|n| n.as_str()).unwrap_or("zone").to_string();
        match m.get("type").and_then(|t| t.as_str()) {
            Some("zone") if nums.len() >= 6 => zones.push(Zone {
                name,
                polygon: nums.chunks_exact(2).map(|c| (c[0], c[1])).collect(),
                alert_on_enter: m.get("alert_on_enter").and_then(|v| v.as_bool()).unwrap_or(true),
                loitering_secs: m.get("loitering_secs").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
                inertia: m.get("inertia").and_then(|v| v.as_u64()).unwrap_or(3) as u32,
                ignore_down: m.get("ignore_down").and_then(|v| v.as_bool()).unwrap_or(false),
            }),
            Some("line") if nums.len() >= 4 && m.get("fence").and_then(|v| v.as_bool()).unwrap_or(false) =>
                fences.push(Fence { name, a: (nums[0], nums[1]), b: (nums[2], nums[3]) }),
            _ => {}
        }
    }
    (zones, fences)
}

static CAMS: OnceLock<Mutex<HashMap<u8, CamState>>> = OnceLock::new();

/// Run `f` against one camera's behaviour state (process-wide).
pub(crate) fn with_cam<R>(cam_id: u8, f: impl FnOnce(&mut CamState) -> R) -> Option<R> {
    let mut map = CAMS.get_or_init(Default::default).lock().ok()?;
    Some(f(map.entry(cam_id).or_default()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame<'a>(id: u64, foot: (f32, f32), history: &'a [TrackPoint], pose: Option<&'a Pose>) -> PersonFrame<'a> {
        PersonFrame { track_id: id, foot, box_h: 200.0, frame_w: 1280.0, frame_h: 720.0, history, pose, ankle_y: None }
    }

    fn square() -> Vec<(f32, f32)> { vec![(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)] }

    fn zone(loiter: u32) -> Zone {
        Zone { name: "Gate".into(), polygon: square(), alert_on_enter: true, loitering_secs: loiter, inertia: 3, ignore_down: false }
    }

    fn torso(angle_deg: f32) -> Pose {
        let mut kp = [[0.0f32; 3]; 17];
        let (dx, dy) = (angle_deg.to_radians().sin() * 60.0, angle_deg.to_radians().cos() * 60.0);
        for (i, x, y) in [(5, 100.0, 100.0), (6, 100.0, 100.0), (11, 100.0 + dx, 100.0 + dy), (12, 100.0 + dx, 100.0 + dy)] {
            kp[i] = [x, y, 0.9];
        }
        Pose { kp }
    }

    #[test]
    fn intrusion_waits_for_inertia_and_fires_once() {
        let rules = Rules { zones: vec![zone(0)], intrusion: true, ..Default::default() };
        let mut cam = CamState::default();
        let t0 = Instant::now();
        let mut kinds = Vec::new();
        for i in 0..6 {
            let out = step(&mut cam, &rules, &[frame(7, (0.5, 0.5), &[], None)], t0 + Duration::from_millis(200 * i));
            kinds.push(out.len());
        }
        assert_eq!(kinds, vec![0, 0, 1, 0, 0, 0], "fires on the 3rd frame inside, then never again");
        // A foot outside the polygon is not an intrusion.
        let mut cam = CamState::default();
        for i in 0..6 {
            assert!(step(&mut cam, &rules, &[frame(8, (0.9, 0.9), &[], None)], t0 + Duration::from_millis(200 * i)).is_empty());
        }
    }

    #[test]
    fn zone_loitering_needs_real_dwell() {
        let rules = Rules { zones: vec![zone(20)], ..Default::default() };
        let mut cam = CamState::default();
        let t0 = Instant::now();
        let mut fired_at = None;
        for s in 0..30u64 {
            let out = step(&mut cam, &rules, &[frame(1, (0.5, 0.5), &[], None)], t0 + Duration::from_secs(s));
            if out.iter().any(|f| f.kind == Kind::Loitering) { fired_at = Some(s); }
        }
        assert_eq!(fired_at, Some(20));
    }

    #[test]
    fn running_is_measured_in_body_heights() {
        let rules = Rules { running: true, ..Default::default() };
        let t0 = Instant::now();
        // 200 px tall person moving 120 px per 0.25 s = 2.4 heights/s → running.
        let fast: Vec<TrackPoint> = (0..6).map(|i| TrackPoint { x: 0.1 + i as f32 * 120.0 / 1280.0, y: 0.8, t: t0 + Duration::from_millis(250 * i) }).collect();
        let now = t0 + Duration::from_millis(1250);
        let mut cam = CamState::default();
        assert!(step(&mut cam, &rules, &[frame(1, (0.5, 0.8), &fast, None)], now).iter().any(|f| f.kind == Kind::Running));
        // Walking: 40 px per 0.25 s = 0.8 heights/s.
        let walk: Vec<TrackPoint> = (0..6).map(|i| TrackPoint { x: 0.1 + i as f32 * 40.0 / 1280.0, y: 0.8, t: t0 + Duration::from_millis(250 * i) }).collect();
        let mut cam = CamState::default();
        assert!(step(&mut cam, &rules, &[frame(2, (0.5, 0.8), &walk, None)], now).is_empty());
    }

    #[test]
    fn person_down_needs_an_upright_baseline_and_a_long_hold() {
        let rules = Rules { down: true, ..Default::default() };
        let (up, flat, seated) = (torso(5.0), torso(85.0), torso(35.0));
        let t0 = Instant::now();
        let at = |s: u64| t0 + Duration::from_secs(s);

        // Seen upright, then flat for 12 s → fires after the hold.
        let mut cam = CamState::default();
        step(&mut cam, &rules, &[frame(1, (0.5, 0.5), &[], Some(&up))], at(0));
        let mut fired = false;
        for s in 1..=12 {
            fired |= step(&mut cam, &rules, &[frame(1, (0.5, 0.5), &[], Some(&flat))], at(s)).iter().any(|f| f.kind == Kind::Down);
            if s < 11 { assert!(!fired, "fired early at {s}s"); }
        }
        assert!(fired);

        // Never seen upright (first seen lying on a sofa) → never fires.
        let mut cam = CamState::default();
        for s in 0..20 {
            assert!(step(&mut cam, &rules, &[frame(2, (0.5, 0.5), &[], Some(&flat))], at(s)).is_empty());
        }

        // Sitting (35°) is not down.
        let mut cam = CamState::default();
        step(&mut cam, &rules, &[frame(3, (0.5, 0.5), &[], Some(&up))], at(0));
        for s in 1..20 {
            assert!(step(&mut cam, &rules, &[frame(3, (0.5, 0.5), &[], Some(&seated))], at(s)).is_empty());
        }

        // Lying in an ignore_down zone is excused.
        let mut z = zone(0); z.ignore_down = true;
        let rules = Rules { down: true, zones: vec![z], ..Default::default() };
        let mut cam = CamState::default();
        step(&mut cam, &rules, &[frame(4, (0.5, 0.5), &[], Some(&up))], at(0));
        for s in 1..20 {
            assert!(step(&mut cam, &rules, &[frame(4, (0.5, 0.5), &[], Some(&flat))], at(s)).is_empty());
        }
    }

    #[test]
    fn climbing_needs_the_ankles_to_go_from_below_to_above_the_fence() {
        let fence = Fence { name: "Back fence".into(), a: (0.0, 0.5), b: (1.0, 0.5) };
        let rules = Rules { climbing: true, fences: vec![fence], ..Default::default() };
        let pose = torso(5.0);
        let t0 = Instant::now();
        let mk = |ay: f32| PersonFrame { ankle_y: Some(ay), ..frame(1, (0.5, 0.6), &[], Some(&pose)) };

        // Someone standing BEHIND the fence (ankles above it from the start) → nothing.
        let mut cam = CamState::default();
        for s in 0..5 { assert!(step(&mut cam, &rules, &[mk(0.4)], t0 + Duration::from_secs(s)).is_empty()); }

        // Near side, then ankles up over the line for two samples → climbing.
        let mut cam = CamState::default();
        step(&mut cam, &rules, &[mk(0.7)], t0);
        step(&mut cam, &rules, &[mk(0.45)], t0 + Duration::from_secs(1));
        let out = step(&mut cam, &rules, &[mk(0.44)], t0 + Duration::from_secs(2));
        assert!(out.iter().any(|f| f.kind == Kind::Climbing));
    }

    #[test]
    fn crowd_must_be_sustained_and_rearms() {
        let rules = Rules { crowd_threshold: 3, ..Default::default() };
        let mut cam = CamState::default();
        let t0 = Instant::now();
        let people =[frame(1, (0.1, 0.1), &[], None), frame(2, (0.2, 0.2), &[], None), frame(3, (0.3, 0.3), &[], None)];
        let mut hits = 0;
        for s in 0..8 { hits += step(&mut cam, &rules, &people, t0 + Duration::from_secs(s)).len(); }
        assert_eq!(hits, 1, "one alert per crowd episode");
        // Quiet for 30 s, then a new crowd → alerts again.
        step(&mut cam, &rules, &[], t0 + Duration::from_secs(40));
        let mut again = 0;
        for s in 41..48 { again += step(&mut cam, &rules, &people, t0 + Duration::from_secs(s)).len(); }
        assert_eq!(again, 1);
    }

    #[test]
    fn masks_parse_zone_options_and_only_fence_lines() {
        let json = r#"{"2":[
            {"type":"zone","name":"Porch","points":"0.1,0.1,0.9,0.1,0.9,0.9","loitering_secs":15,"inertia":5,"alert_on_enter":false,"ignore_down":true},
            {"type":"line","name":"Gate","points":"0,0.5,1,0.5"},
            {"type":"line","name":"Wall","points":"0,0.3,1,0.3","fence":true},
            {"type":"mask","points":"0,0,1,0,1,1"}]}"#;
        let (zones, fences) = parse_masks(json, 2);
        assert_eq!(zones.len(), 1);
        let z = &zones[0];
        assert_eq!((z.loitering_secs, z.inertia, z.alert_on_enter, z.ignore_down), (15, 5, false, true));
        assert_eq!(fences.len(), 1, "a plain line is a tripwire, not a fence");
        assert_eq!(fences[0].name, "Wall");
        assert!(parse_masks(json, 3).0.is_empty());
    }
}
