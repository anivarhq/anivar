//! Audio event detection — YAMNet (AudioSet) sound classifier (NVR-parity).
//!
//! Mature NVRs run a CPU audio detector over 500+ AudioSet classes (scream, glass
//! breaking, smoke/fire alarm, gunshot, dog bark, speech, …) and raises events
//! when a listened class exceeds a threshold. We do the same with **YAMNet**
//! (the standard AudioSet model) installed as the `audio_yamnet` skill:
//!   * `model.onnx`     — waveform float32 (16 kHz mono) → scores `[frames, 521]`.
//!   * `class_map.csv`  — the 521-row AudioSet ontology (index,mid,display_name)
//!                        shipped with YAMNet, so we match by class NAME (never a
//!                        hard-coded index).
//!
//! Lazy-loaded into a process-wide cache like [`crate::embed`]. Absent skill →
//! `with_detector` returns `None` and audio detection is silently off.
//!
//! Defensive: input/output names + output rank (`[frames, C]` or `[C]`) are
//! resolved from the loaded session, because exports vary.

use std::path::Path;
use std::sync::{Mutex, OnceLock};

use ort::session::Session as OrtSession;
use ort::value::Tensor;


const SKILL_ID: &str = "audio_yamnet";
/// YAMNet expects 16 kHz mono. One inference window ≈ 0.975 s minimum; we feed ~1 s.
pub const SAMPLE_RATE: u32 = 16_000;

static MODEL: OnceLock<Mutex<Option<Yamnet>>> = OnceLock::new();

pub struct Yamnet {
    session:     OrtSession,
    input_name:  String,
    score_out:   usize,
    class_names: Vec<String>,
}

impl Yamnet {
    pub fn try_load(data_dir: &Path) -> anyhow::Result<Self> {
        let dir = data_dir.join("skills").join(SKILL_ID);
        let model_path = dir.join("model.onnx");
        let map_path   = dir.join("class_map.csv");
        if !model_path.exists() { anyhow::bail!("audio_yamnet: missing {}", model_path.display()); }
        if !map_path.exists()   { anyhow::bail!("audio_yamnet: missing {}", map_path.display()); }

        let class_names = load_class_map(&map_path)?;
        if class_names.is_empty() { anyhow::bail!("audio_yamnet: empty class_map.csv"); }

        // YAMNet is a mobile-class model — CPU is plenty at audio-window cadence,
        // and this removes its holds on the serialized GPU lock (enrichment lane).
        let session = crate::inference::build_ort_session_cpu(&model_path)?;
        let input_name = session.inputs().first()
            .map(|i| i.name().to_string()).unwrap_or_else(|| "waveform".into());
        // The scores output is the one whose trailing dim == class count; fall back to 0.
        let score_out = session.outputs().iter()
            .position(|o| o.name().to_lowercase().contains("score"))
            .unwrap_or(0);

        tracing::info!("audio_yamnet loaded — input={input_name} score_out#{score_out} classes={}", class_names.len());
        Ok(Self { session, input_name, score_out, class_names })
    }

    /// Classify one mono 16 kHz window. Returns `(class_name, score)` per class,
    /// taking the max over time frames, sorted high→low. Empty on failure.
    pub fn detect(&mut self, pcm: &[f32]) -> Vec<(String, f32)> {
        if pcm.len() < (SAMPLE_RATE as usize) / 2 { return Vec::new(); } // < ~0.5 s — too short
        let tensor = match Tensor::<f32>::from_array(([pcm.len()], pcm.to_vec())) {
            Ok(t) => t, Err(e) => { tracing::debug!("yamnet tensor: {e}"); return Vec::new(); }
        };
        let run_result = {
            let _t = crate::inference::infer_timer("audio");
            self.session.run(ort::inputs![self.input_name.as_str() => tensor])
        };
        let outputs = match run_result {
            Ok(o) => o, Err(e) => { tracing::debug!("yamnet run: {e}"); return Vec::new(); }
        };
        let (shape, data) = match outputs[self.score_out].try_extract_tensor::<f32>() {
            Ok((s, d)) => (s, d), Err(e) => { tracing::debug!("yamnet extract: {e}"); return Vec::new(); }
        };
        let dims: Vec<usize> = shape.iter().map(|&v| v.max(0) as usize).collect();
        // Resolve [frames, C] (max-pool over frames) or [C].
        let n_classes = self.class_names.len();
        let per_class_max: Vec<f32> = match dims.as_slice() {
            [c] if *c == n_classes => data[..n_classes].to_vec(),
            [frames, c] if *c == n_classes => {
                let mut m = vec![f32::MIN; n_classes];
                for f in 0..*frames {
                    let base = f * c;
                    for k in 0..n_classes { if data[base + k] > m[k] { m[k] = data[base + k]; } }
                }
                m
            }
            // Unknown layout but flat length is a multiple of C — treat as frames-major.
            _ if !data.is_empty() && data.len() % n_classes == 0 => {
                let frames = data.len() / n_classes;
                let mut m = vec![f32::MIN; n_classes];
                for f in 0..frames {
                    let base = f * n_classes;
                    for k in 0..n_classes { if data[base + k] > m[k] { m[k] = data[base + k]; } }
                }
                m
            }
            _ => return Vec::new(),
        };
        let mut out: Vec<(String, f32)> = self.class_names.iter().cloned()
            .zip(per_class_max).collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        out
    }
}

/// Parse YAMNet's `class_map.csv` (header `index,mid,display_name`) → display
/// names indexed by row. Tolerates quoted names with commas.
fn load_class_map(path: &Path) -> anyhow::Result<Vec<String>> {
    let text = std::fs::read_to_string(path)?;
    let mut names = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if i == 0 && line.to_lowercase().contains("display_name") { continue; } // header
        // display_name is the last field; handle a trailing quoted field.
        let name = if let Some(idx) = line.find('"') {
            line[idx..].trim_matches('"').to_string()
        } else {
            line.rsplit(',').next().unwrap_or("").to_string()
        };
        let name = name.trim().trim_matches('"').to_string();
        if !name.is_empty() { names.push(name); }
    }
    Ok(names)
}

// ─── Process-wide accessors ──────────────────────────────────────────────────

pub fn with_detector<R>(data_dir: &Path, f: impl FnOnce(&mut Yamnet) -> R) -> Option<R> {
    let cell = MODEL.get_or_init(|| Mutex::new(None));
    let mut guard = cell.lock().ok()?;
    if guard.is_none() {
        match Yamnet::try_load(data_dir) {
            Ok(m) => *guard = Some(m),
            Err(e) => { tracing::debug!("audio_yamnet unavailable: {e}"); return None; }
        }
    }
    guard.as_mut().map(f)
}

pub fn is_installed(data_dir: &Path) -> bool {
    let dir = data_dir.join("skills").join(SKILL_ID);
    // A real YAMNet ONNX is several MB. A tiny file is a FAILED download — observed in
    // the wild as a 29-byte "Invalid username or password." auth-error page saved as
    // model.onnx. Treat anything under 100 KB as NOT installed so the UI prompts a
    // re-download instead of the detector silently failing to load it every cycle.
    let model_ok = std::fs::metadata(dir.join("model.onnx"))
        .map(|m| m.len() > 100_000).unwrap_or(false);
    model_ok && dir.join("class_map.csv").exists()
}

/// RMS of a PCM window in [0,1] — used as the cheap "is anything happening?" gate
/// before paying for YAMNet inference (mature NVRs' `min_volume`).
/// AudioSet classes in the HIGH-PITCH / alert register — screams, alarms,
/// sirens, glass, whistles. Used by the Audio tab's "high-pitch only" card
/// filter: YAMNet's own semantics decide (a scream flags, loud speech doesn't).
const HIGH_PITCH_CLASSES: &[&str] = &[
    "scream", "screaming", "yell", "shout",
    "siren", "civil defense siren", "police car", "ambulance", "fire engine",
    "smoke detector", "fire alarm", "alarm", "alarm clock", "car alarm", "buzzer",
    "beep", "bleep", "whistle", "whistling", "squeal", "screech",
    "glass", "shatter",
    "baby cry", "infant cry", "crying",
    "telephone bell",
];

/// Case-insensitive substring match against the high-pitch set (same matching
/// style as the user's listen-list).
/// Sound-category filter sets (AudioSet-grouped, standard label matching).
/// Each entry is a lowercase substring matched against dominant_label.
pub(crate) fn category_label_patterns(cat: &str) -> Option<&'static [&'static str]> {
    match cat {
        "human"   => Some(&["speech", "conversation", "laugh", "crying", "cough", "shout",
                            "whisper", "singing", "yell", "sneeze", "narration"]),
        "alarm"   => Some(&["siren", "alarm", "smoke detector", "buzzer", "beep", "bleep",
                            "bell", "telephone", "chime"]),
        "animal"  => Some(&["dog", "bark", "cat", "meow", "bird", "chirp", "growl",
                            "animal", "howl", "purr"]),
        "vehicle" => Some(&["car", "engine", "horn", "truck", "motorcycle", "traffic",
                            "aircraft", "vehicle", "train"]),
        "impact"  => Some(&["glass", "shatter", "bang", "gunshot", "explosion", "knock",
                            "slam", "thump", "crash", "smash", "breaking"]),
        "music"   => Some(&["music", "instrument", "guitar", "piano", "drum", "song"]),
        _ => None,
    }
}

pub(crate) fn is_high_pitch_class(name: &str) -> bool {
    let n = name.to_lowercase();
    HIGH_PITCH_CLASSES.iter().any(|c| n.contains(c))
}

pub fn rms(pcm: &[f32]) -> f32 {
    if pcm.is_empty() { return 0.0; }
    (pcm.iter().map(|x| x * x).sum::<f32>() / pcm.len() as f32).sqrt()
}
