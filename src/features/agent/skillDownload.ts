/**
 * Shared skill-download flow used by Cookbook (and historically by the
 * AgentPanel Skills tab). Handles both single-URL skills (legacy YOLO26)
 * and multi-file skills (face_small / face_large need detector + embedder).
 *
 * Why a separate module:
 *   • Cookbook and AgentPanel both want the same Install / Uninstall buttons
 *     with identical state-machine transitions.
 *   • Multi-file walking + progress aggregation is non-trivial — we want one
 *     copy.
 *   • Test surface: pure functions on `SkillDef`, no React.
 *
 * The `SkillDef` registry below is the single source of truth for what we
 * call a "skill". Adding a new one is purely declarative.
 */

import { api } from "../../api";

export type YoloTier = "nano" | "small" | "medium" | "large" | "xlarge";

export type AlprRegion = "global" | "european" | "argentinian";

/// On-device language-model tiers. Mirrors `YoloTier`: several registry entries
/// share `slot: "llm"` and this picks which one `settings.local_llm_tier` runs.
export type LlmTier = "fast" | "balanced" | "vision";

export type SkillId =
  // YOLO 2026 tiers — one skill ID per variant, all share the `slot: "yolo"`.
  | "yolo26n" | "yolo26s" | "yolo26m" | "yolo26l" | "yolo26x"
  // Face tiers — mature NVRs' small/large split.
  | "face_small" | "face_large"
  // License plate recognition — region-specific Mobile-ViT v2 OCR variants.
  // Three tiers mirror fast-plate-ocr's release tags; users pick whichever
  // matches their camera's geography for best accuracy.
  | "alpr_global" | "alpr_european" | "alpr_argentinian"
  // Semantic event search — CLIP-family image + text encoders (NVR parity),
  // tiered by size so resource-constrained hosts can pick a small model.
  | "mobileclip_s0" | "clip_b32" | "jina_clip"
  // Audio event detection — YAMNet (AudioSet) sound classifier.
  | "audio_yamnet"
  // Deep person Re-ID — NVIDIA TAO ReIdentificationNet (commercial-use terms).
  | "reid_tao"
  // Body pose — MoveNet SinglePose Lightning (Apache-2.0) for behaviour alerts.
  | "pose_movenet"
  // Person attributes — PaddleClas PULC (Apache-2.0, PA-100K) for People search.
  | "par_pulc"
  | "depth_anything"
  // On-device language model (llama.cpp in-process) — replaces the Ollama daemon.
  | "local_llm"
  | "local_llm_fast"
  | "local_llm_vision"
  // Back-compat: existing installs that landed in `skills/yolo26/` (pre-tier
  // rollout) or `skills/alpr/` (pre-region split). We never create new skills
  // under these ids; the registry exposes them so the UI can show legacy installs.
  | "yolo26" | "alpr";
export type SkillStatus = "not_installed" | "downloading" | "installed" | "error";

export interface SkillFile {
  url:      string;
  filename: string;   // stored at <data>/skills/<id>/<filename>
  label:    string;
  /** Expected SHA-256 (hex). When set, a download that doesn't match is deleted
   *  and the install fails — for URLs that redirect through signed storage. */
  sha256?:  string;
}

export interface SkillDef {
  id:           SkillId;
  name:         string;
  /** Use-case tag — drives which Cookbook slot the skill belongs to. */
  slot:         "yolo" | "face" | "alpr" | "search" | "audio" | "reid" | "privacy" | "llm";
  description:  string;
  /** Human size, e.g. "~37 MB". */
  sizeLabel:    string;
  /** Used by the legacy single-file flow (Yolo26). Empty for multi-file. */
  installUrl:   string;
  /** Multi-file alternative — preferred for face skills. */
  files?:       SkillFile[];
  /** Subtitle badge — "CPU" / "GPU" / "RECOMMENDED" etc. */
  badge?:       string;
  badgeColor?:  string;
  /** YOLO tier this skill represents — set on all five `yolo26{n,s,m,l,x}`
   *  entries. Used by the Cookbook YoloCard to render a tier-picker row and
   *  to write `settings.yolo_variant` when the user activates one. */
  yoloTier?:    YoloTier;
  /** ALPR region this skill represents — set on all three `alpr_*` entries.
   *  Used by the Cookbook AlprCard to render a region-picker and to write
   *  `settings.alpr_region` when the user activates one. */
  alprRegion?:  AlprRegion;
  /** On-device LLM tier this skill represents — set on the three `llm` entries.
   *  Written to `settings.local_llm_tier` when the user activates one. */
  llmTier?:     LlmTier;
  /** Upstream licence of the WEIGHTS (not of this app). Shown at the point of
   *  choice so installing a model is an informed act: some licences place real
   *  obligations on whatever you build around them. See THIRD-PARTY-NOTICES.md. */
  license?:     string;
  /** Plain-language consequence of `license`, surfaced next to it when the
   *  licence is one a user could get caught out by. */
  licenseNote?: string;
  /** Minimum believable size of the installed weights, in bytes.
   *
   *  Set this when a skill's model is REPLACED by a bigger one at the same path:
   *  the generic "is there a weight file here" check would otherwise keep
   *  reporting the superseded model as installed, and the upgrade would never be
   *  offered. Not a substitute for a digest — it catches supersession, not
   *  tampering. */
  minBytes?:    number;
}

/// Same note on all three local-model tiers — the licence is the model family's,
/// not the tier's.
const LFM_LICENCE_NOTE =
  "Liquid AI's own licence, not Apache-2.0 or MIT. Fine for personal and evaluation " +
  "use; read the terms before shipping a commercial product on it. The weights are " +
  "downloaded from Liquid AI at your request and are not distributed with this app.";

/** Authoritative skill registry. Keep aligned with the Rust `list_installed_skills` IDs. */
export const SKILL_REGISTRY: SkillDef[] = [
  // ── YOLO 2026 tiers — five separate ONNX repos on Hugging Face. All use
  //    the same DETR-style output (handled in `inference.rs::decode_yolo_output_detr`).
  {
    id:          "yolo26n",
    name:        "YOLO26 — Nano",
    slot:        "yolo",
    yoloTier:    "nano",
    description: "Smallest tier. Real-time on a modest CPU. Good for low-power hosts.",
    sizeLabel:   "~10 MB",
    badge:       "CPU",
    badgeColor:  "var(--status-idle)",
    installUrl:  "https://huggingface.co/onnx-community/yolo26n-ONNX/resolve/main/onnx/model.onnx",
    license:     "AGPL-3.0",
    licenseNote: "Ultralytics requires a commercial licence for proprietary or commercial use — choosing this model places obligations on whatever you build around it.",
  },
  {
    id:          "yolo26s",
    name:        "YOLO26 — Small",
    slot:        "yolo",
    yoloTier:    "small",
    description: "Balanced accuracy at modest disk + memory cost. Sweet spot under 4 GB VRAM.",
    sizeLabel:   "~37 MB",
    badge:       "BALANCED",
    badgeColor:  "var(--status-idle)",
    installUrl:  "https://huggingface.co/onnx-community/yolo26s-ONNX/resolve/main/onnx/model.onnx",
    license:     "AGPL-3.0",
    licenseNote: "Ultralytics requires a commercial licence for proprietary or commercial use — choosing this model places obligations on whatever you build around it.",
  },
  {
    id:          "yolo26m",
    name:        "YOLO26 — Medium",
    slot:        "yolo",
    yoloTier:    "medium",
    description: "Better accuracy than Small. Comfortable with 4+ GB VRAM.",
    sizeLabel:   "~78 MB",
    installUrl:  "https://huggingface.co/onnx-community/yolo26m-ONNX/resolve/main/onnx/model.onnx",
    license:     "AGPL-3.0",
    licenseNote: "Ultralytics requires a commercial licence for proprietary or commercial use — choosing this model places obligations on whatever you build around it.",
  },
  {
    id:          "yolo26l",
    name:        "YOLO26 — Large",
    slot:        "yolo",
    yoloTier:    "large",
    description: "Strong accuracy. Needs 6+ GB VRAM for real-time, but tolerates lower.",
    sizeLabel:   "~95 MB",
    badge:       "ACCURATE",
    badgeColor:  "var(--status-ok)",
    installUrl:  "https://huggingface.co/onnx-community/yolo26l-ONNX/resolve/main/onnx/model.onnx",
    license:     "AGPL-3.0",
    licenseNote: "Ultralytics requires a commercial licence for proprietary or commercial use — choosing this model places obligations on whatever you build around it.",
  },
  {
    id:          "yolo26x",
    name:        "YOLO26 — XLarge",
    slot:        "yolo",
    yoloTier:    "xlarge",
    description: "Highest accuracy. Requires 8+ GB VRAM for real-time inference.",
    sizeLabel:   "~175 MB",
    badge:       "BEST",
    badgeColor:  "var(--status-ok)",
    installUrl:  "https://huggingface.co/onnx-community/yolo26x-ONNX/resolve/main/onnx/model.onnx",
    license:     "AGPL-3.0",
    licenseNote: "Ultralytics requires a commercial licence for proprietary or commercial use — choosing this model places obligations on whatever you build around it.",
  },
  // ── Face recognition (separate skill from YOLO 2026, intentional) ───────
  // Both reference NVRs ship face recognition as its own pipeline distinct from
  // object detection:
  //   • edge-AI NVRs — MTCNN (face detection) + InsightFace ArcFace + SVM.
  //   • mature NVRs    — FaceNet (small / CPU) or ArcFace (large / GPU).
  // YOLO 2026's COCO 80-class output does NOT include a face class — only
  // "person" — and ArcFace-style embedding matching needs an aligned face crop
  // with 5-point landmarks. Keeping these skills separate lets a host install
  // YOLO without paying the face-model cost (or vice versa).
  {
    id:          "face_small",
    name:        "Face Recognition — Small",
    slot:        "face",
    description: "standard face pipeline tuned for CPU. yolov8n-face detector + INT8 ArcFace embedder. Good for low-power hosts.",
    sizeLabel:   "~37 MB",
    badge:       "CPU",
    badgeColor:  "var(--status-idle)",
    installUrl:  "",
    files: [
      { url: "https://huggingface.co/deepghs/yolo-face/resolve/main/yolov8n-face/model.onnx",
        filename: "detector.onnx", label: "yolov8n-face — detector" },
      { url: "https://huggingface.co/onnxmodelzoo/arcfaceresnet100-11-int8/resolve/main/arcfaceresnet100-11-int8.onnx",
        filename: "embedder.onnx", label: "ArcFace INT8 — recognizer" },
    ],
  },
  {
    id:          "face_large",
    name:        "Face Recognition — Large",
    slot:        "face",
    description: "standard face pipeline at full FP32 precision. Same detector as Small. Best accuracy; real-time speed needs a discrete GPU or NPU.",
    sizeLabel:   "~262 MB",
    badge:       "ACCURATE",
    badgeColor:  "var(--status-ok)",
    installUrl:  "",
    files: [
      { url: "https://huggingface.co/deepghs/yolo-face/resolve/main/yolov8n-face/model.onnx",
        filename: "detector.onnx", label: "yolov8n-face — detector" },
      { url: "https://huggingface.co/garavv/arcface-onnx/resolve/main/arc.onnx",
        filename: "embedder.onnx", label: "ArcFace FP32 — recognizer" },
    ],
  },
  // ── License plate recognition (mature NVRs 0.16+ shipped ALPR as a first-class
  //    feature; we model it the same way — a separate skill that runs after a
  //    vehicle is detected in an event). Model is the Mobile-ViT v2 OCR from
  //    ankandrew/fast-plate-ocr, distributed via cnn-ocr-lp release tags.
  //    Three regional variants — global (default, multi-region), european
  //    (EU-format plates), argentinian (AR plates). Pick whichever matches the
  //    cameras' geography for best accuracy.
  {
    id:          "alpr_global",
    name:        "License Plates — Global",
    slot:        "alpr",
    alprRegion:  "global",
    description: "Default region. Multi-format plates, works worldwide. Best starting point if you don't know which region matches your cameras.",
    sizeLabel:   "~8 MB",
    badge:       "DEFAULT",
    badgeColor:  "var(--status-idle)",
    installUrl:  "",
    // Force the saved filename so `alpr.rs::find_alpr_model` (which looks for
    // `skills/alpr_global/model.onnx`) finds it without a directory scan.
    files: [
      { url:      "https://github.com/ankandrew/cnn-ocr-lp/releases/download/arg-plates/global_mobile_vit_v2_ocr.onnx",
        filename: "model.onnx",
        label:    "Global Mobile-ViT v2 OCR" },
    ],
  },
  {
    id:          "alpr_european",
    name:        "License Plates — European",
    slot:        "alpr",
    alprRegion:  "european",
    description: "Tuned for European-format plates (yellow rear / white front, country code stripe). More accurate than Global in the EU/UK.",
    sizeLabel:   "~8 MB",
    badge:       "EU",
    badgeColor:  "var(--status-warn)",
    installUrl:  "",
    files: [
      { url:      "https://github.com/ankandrew/cnn-ocr-lp/releases/download/eu-plates/european_mobile_vit_v2_ocr.onnx",
        filename: "model.onnx",
        label:    "European Mobile-ViT v2 OCR" },
    ],
  },
  {
    id:          "alpr_argentinian",
    name:        "License Plates — Argentinian",
    slot:        "alpr",
    alprRegion:  "argentinian",
    description: "Tuned for Argentinian plates (Mercosur and pre-Mercosur formats). Pick this if your cameras are in Argentina.",
    sizeLabel:   "~8 MB",
    badge:       "AR",
    badgeColor:  "var(--status-warn)",
    installUrl:  "",
    files: [
      { url:      "https://github.com/ankandrew/cnn-ocr-lp/releases/download/arg-plates/argentinian_mobile_vit_v2_ocr.onnx",
        filename: "model.onnx",
        label:    "Argentinian Mobile-ViT v2 OCR" },
    ],
  },
  // ── Semantic event search (mature NVRs 0.14+ shipped natural-language search via
  //    CLIP; we model it identically). Each tier is a two-tower ONNX model +
  //    tokenizer from the transformers.js exports. The Rust encoder (`embed.rs`)
  //    resolves input/output tensor names + preprocessing per model, so all
  //    three share one code path. Filenames are fixed so `embed::try_load` finds
  //    them: vision_model.onnx + text_model.onnx + tokenizer.json. Pick by host
  //    resources — MobileCLIP-S0 is the small/fast default.
  {
    id:          "mobileclip_s0",
    name:        "Semantic Search — MobileCLIP-S0",
    slot:        "search",
    description: "Apple's MobileCLIP-S0 — the small, fast default. ~5× smaller than CLIP-B/16 at similar accuracy. Best for low-resource hosts.",
    sizeLabel:   "~207 MB",
    badge:       "LITE",
    badgeColor:  "var(--status-ok)",
    installUrl:  "",
    files: [
      { url: "https://huggingface.co/Xenova/mobileclip_s0/resolve/main/onnx/vision_model.onnx",
        filename: "vision_model.onnx", label: "Vision encoder" },
      { url: "https://huggingface.co/Xenova/mobileclip_s0/resolve/main/onnx/text_model.onnx",
        filename: "text_model.onnx", label: "Text encoder" },
      { url: "https://huggingface.co/Xenova/mobileclip_s0/resolve/main/tokenizer.json",
        filename: "tokenizer.json", label: "Tokenizer" },
    ],
  },
  {
    id:          "depth_anything",
    name:        "Depth Anonymization — Depth-Anything-v2",
    slot:        "privacy",
    description: "edge-AI-style privacy: the camera feed becomes a colorized depth map (near = warm, far = cool) — movement stays trackable while faces and identities never exist in recordings, streams or alerts. Local AI still analyzes raw frames in memory.",
    sizeLabel:   "~50 MB",   // FP16 default (see DEPTH_VARIANTS for the tier picker)
    badge:       "PRIVACY",
    badgeColor:  "var(--status-idle)",
    installUrl:  "",
    // Default = FP16 (half the old FP32, GPU-native). CameraView's auto-install
    // on first toggle uses this; the Arsenal Depth card offers all tiers.
    files: [
      { url: "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model_fp16.onnx",
        filename: "model_fp16.onnx", label: "Depth model (FP16)" },
    ],
  },
  {
    id:          "clip_b32",
    name:        "Semantic Search — CLIP ViT-B/32",
    slot:        "search",
    description: "OpenAI's classic CLIP ViT-B/32. A well-understood, balanced baseline. Bigger than MobileCLIP but broadly compatible.",
    sizeLabel:   "~579 MB",
    badge:       "STD",
    badgeColor:  "var(--status-idle)",
    installUrl:  "",
    files: [
      { url: "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/onnx/vision_model.onnx",
        filename: "vision_model.onnx", label: "Vision encoder" },
      { url: "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/onnx/text_model.onnx",
        filename: "text_model.onnx", label: "Text encoder" },
      { url: "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/tokenizer.json",
        filename: "tokenizer.json", label: "Tokenizer" },
    ],
  },
  {
    id:          "jina_clip",
    name:        "Semantic Search — Jina-CLIP",
    slot:        "search",
    description: "Jina-CLIP-v1 (768-d). Largest/most accurate tier; multilingual-leaning. Runs after an event closes; CPU-friendly but a heavier download.",
    sizeLabel:   "~850 MB",
    badge:       "MAX",
    badgeColor:  "var(--status-idle)",
    installUrl:  "",
    files: [
      { url:      "https://huggingface.co/jinaai/jina-clip-v1/resolve/main/onnx/vision_model.onnx",
        filename: "vision_model.onnx",
        label:    "Vision encoder (EVA02)" },
      { url:      "https://huggingface.co/jinaai/jina-clip-v1/resolve/main/onnx/text_model.onnx",
        filename: "text_model.onnx",
        label:    "Text encoder (JinaBERT)" },
      { url:      "https://huggingface.co/jinaai/jina-clip-v1/resolve/main/tokenizer.json",
        filename: "tokenizer.json",
        label:    "Tokenizer" },
    ],
  },
  // ── Audio event detection (NVR-parity). YAMNet (AudioSet, 521 classes)
  //    classifies scream / glass-breaking / smoke+fire alarm / gunshot / dog bark
  //    / speech from the camera's audio. Needs an RTSP camera with an audio track
  //    (browser/USB audio is a future frontend path). NOTE: the model + class_map
  //    URLs below are best-guess and should be VERIFIED before release (like the
  //    Jina lesson) — the feature degrades to off if the skill isn't present.
  {
    id:          "audio_yamnet",
    name:        "Audio Detection — YAMNet",
    slot:        "audio",
    description: "Detects scream, glass breaking, smoke/fire alarm, gunshot, dog bark and more from camera audio (RTSP). Raises events even when nothing is visible.",
    sizeLabel:   "~17 MB",
    badge:       "AUDIO",
    badgeColor:  "var(--status-warn)",
    installUrl:  "",
    files: [
      // VERIFIED tf2onnx export of TF-Hub YAMNet (16 MB): input `waveform` (variable-
      // length 1-D f32 @16 kHz), output_0 = [frames, 521] AudioSet scores — exactly what
      // audio.rs feeds/reads (downloaded + inference-tested before shipping). The old
      // `onnx-community/yamnet` URL 401'd and saved a 29-byte "Invalid username or
      // password." page as the model. jafet21/yamnetonnx is an equivalent fallback.
      { url: "https://huggingface.co/zeropointnine/yamnet-onnx/resolve/main/yamnet.onnx",
        filename: "model.onnx", label: "YAMNet classifier" },
      { url: "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv",
        filename: "class_map.csv", label: "AudioSet class map" },
    ],
  },
  // ── Local language model — the in-process brain for chat and alert wording.
  //    Runs inside anivar.exe via llama.cpp: no daemon, no port, nothing to
  //    install separately; loaded on demand, released when idle.
  //
  //    Three tiers sharing `slot: "llm"`, exactly like the five YOLO entries.
  //    `settings.local_llm_tier` picks which one runs; each has its own
  //    `skills/<id>/` directory, so installing one never disturbs another.
  {
    id:          "local_llm_fast",
    name:        "Local AI — Fast (350M)",
    slot:        "llm",
    llmTier:     "fast",
    description: "Answers instantly on any CPU. Liquid's own card recommends it for extraction and tool use and warns against knowledge-heavy work — so it reads terse rather than chatty. Best on low-power hosts.",
    sizeLabel:   "~230 MB",
    badge:       "FASTEST",
    badgeColor:  "var(--status-idle)",
    installUrl:  "",
    files: [
      { url:      "https://huggingface.co/LiquidAI/LFM2.5-350M-GGUF/resolve/main/LFM2.5-350M-Q4_K_M.gguf",
        filename: "model.gguf",
        label:    "LFM2.5-350M (Q4_K_M)" },
    ],
    license:     "LFM Open License v1.0",
    licenseNote: LFM_LICENCE_NOTE,
  },
  {
    id:          "local_llm",
    name:        "Local AI — Balanced (1.2B)",
    slot:        "llm",
    llmTier:     "balanced",
    description: "The default. Writes proper sentences while still running comfortably on a CPU, and answers from the database rather than guessing.",
    sizeLabel:   "~731 MB",
    badge:       "RECOMMENDED",
    badgeColor:  "var(--status-ok)",
    installUrl:  "",
    files: [
      { url:      "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF/resolve/main/LFM2.5-1.2B-Instruct-Q4_K_M.gguf",
        filename: "model.gguf",
        label:    "LFM2.5-1.2B-Instruct (Q4_K_M)" },
    ],
    // The 350M used to live at THIS id's path (~230 MB), so without a floor the
    // superseded file reads as installed forever and the upgrade is never offered.
    minBytes:    500 * 1024 * 1024,
    license:     "LFM Open License v1.0",
    licenseNote: LFM_LICENCE_NOTE,
  },
  {
    id:          "local_llm_vision",
    name:        "Local AI — Vision (1.6B)",
    slot:        "llm",
    llmTier:     "vision",
    description: "Can actually LOOK at your footage and describe what it sees, on-device. Two files: the model and its vision projector. Risk levels still come from the detector, not from this — a wrong risk level is a missed alert.",
    sizeLabel:   "~1.3 GB",
    badge:       "SEES FOOTAGE",
    badgeColor:  "var(--status-idle)",
    installUrl:  "",
    files: [
      { url:      "https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-GGUF/resolve/main/LFM2.5-VL-1.6B-Q4_K_M.gguf",
        filename: "model.gguf",
        label:    "LFM2.5-VL-1.6B (Q4_K_M)" },
      { url:      "https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-GGUF/resolve/main/mmproj-LFM2.5-VL-1.6b-Q8_0.gguf",
        filename: "mmproj.gguf",
        label:    "Vision projector (Q8_0)" },
    ],
    license:     "LFM Open License v1.0",
    licenseNote: LFM_LICENCE_NOTE,
  },
  // ── Deep person Re-ID — NVIDIA TAO ReIdentificationNet v1.2 (ResNet-50,
  //    256-d, RGB 256×128). Replaced OSNet x0.25, whose MSMT17 training data is
  //    research-only; NVIDIA states this model is ready for commercial use.
  //    VERIFIED 2026-09-15: anonymous NGC download (302 → signed storage),
  //    dynamic batch, output un-normalised (reid.rs L2s it). 16% of its weights
  //    are subnormal floats — ORT flush-to-zero takes CPU from 3.3 s to 21 ms.
  {
    id:          "reid_tao",
    name:        "Person Re-ID — NVIDIA ReIdentificationNet",
    slot:        "reid",
    description: "Deep person re-identification (ResNet-50). Follows people across cameras by appearance and powers Find similar person. Falls back to the built-in colour matcher when absent.",
    sizeLabel:   "~92 MB",
    badge:       "RE-ID",
    badgeColor:  "var(--status-idle)",
    installUrl:  "",
    files: [
      { url: "https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/reidentificationnet/deployable_v1.2/files?redirect=true&path=resnet50_market1501_aicity156.onnx",
        filename: "model.onnx", label: "ReIdentificationNet v1.2 (ResNet-50)",
        sha256: "0e21d09278508ec835955f422a9fdd3cd59b2a6ecdef98d705f388f33cebac2b" },
    ],
    license:     "NVIDIA TAO model terms",
    licenseNote: "NVIDIA states this model is ready for commercial use. Downloaded from NVIDIA NGC at your request; not distributed with this app.",
  },
  // ── Body pose — MoveNet SinglePose Lightning (Apache-2.0; COCO + Google's own
  //    "Active" set). Top-down on person crops, only for behaviour candidates and
  //    clothing-colour regions. VERIFIED 2026-09-15: int32 [1,192,192,3] →
  //    [1,1,17,3] (y, x, score); ~6 ms on CPU.
  {
    id:          "pose_movenet",
    name:        "Body Pose — MoveNet",
    slot:        "reid",
    description: "Body keypoints for person-down and climbing alerts, and sharper shirt and trouser colours. Runs only on people a rule is watching.",
    sizeLabel:   "~9 MB",
    badge:       "POSE",
    badgeColor:  "var(--status-idle)",
    installUrl:  "",
    files: [
      { url: "https://huggingface.co/Xenova/movenet-singlepose-lightning/resolve/main/onnx/model.onnx",
        filename: "model.onnx", label: "MoveNet SinglePose Lightning",
        sha256: "1ad4f8d6c2f776a9967db3993c9ca740bc350104f9d37c151dc183fc29a464ad" },
    ],
    license:     "Apache-2.0",
  },
  // ── Person attributes — PaddleClas PULC person_attribute (PP-LCNet x1.0).
  //    Apache-2.0 weights trained on PA-100K only (CC-BY 4.0) — NOT PP-Human's own
  //    attribute weights, which also saw research-only data. Baidu publishes only
  //    the Paddle format, so the ONNX conversion is hosted on anivarhq/models with
  //    its licence and model card. VERIFIED 2026-09-16: the anonymous public
  //    download matches the sha256 below; input `x` [N,3,256,192] → [N,26] sigmoid.
  {
    id:          "par_pulc",
    name:        "Person Attributes — PP-LCNet",
    slot:        "reid",
    description: "Clothing, bags and accessories on each person — the evidence on a visit, and search terms like \"backpack\" or \"no hat\". Runs once per person, not every frame.",
    sizeLabel:   "~7 MB",
    badge:       "ATTR",
    badgeColor:  "var(--status-idle)",
    installUrl:  "",
    files: [
      { url: "https://github.com/anivarhq/models/releases/download/pulc-person-attribute-v1/model.onnx",
        filename: "model.onnx", label: "PULC person attribute (PP-LCNet x1.0)",
        sha256: "8f180a2e58e7c582feb5eac031657c38e06f7df4271e3e846b0f0fa4e57667f5" },
    ],
    license:     "Apache-2.0",
    licenseNote: "PaddleClas model (© PaddlePaddle Authors), trained on PA-100K (CC-BY 4.0). Converted to ONNX by Anivar; not affiliated with or endorsed by Baidu.",
  },
];

export function findSkill(id: SkillId): SkillDef | undefined {
  return SKILL_REGISTRY.find(s => s.id === id);
}

/** Depth-anonymization SPEED presets. One FP16 model (~50 MB, half the old
 *  FP32 and faster on GPU); the preset only changes the INFERENCE RESOLUTION —
 *  the real efficiency lever (the model input is dynamic). Lower res = fewer
 *  ViT tokens = faster, with no quality cost that matters for a privacy depth
 *  map. (Quantized INT8/Q4 variants were rejected: smaller download but slower
 *  on GPU — those ops don't accelerate on DirectML/TensorRT.) */
export const DEPTH_PRESETS = [
  { preset: "fast",     label: "Fast",     res: "252px", note: "~4.5× faster",  badgeColor: "var(--status-ok)" },
  { preset: "balanced", label: "Balanced", res: "378px", note: "recommended",  badgeColor: "var(--status-idle)" },
  { preset: "quality",  label: "Quality",  res: "518px", note: "sharpest",      badgeColor: "var(--status-idle)" },
] as const;

/** The single FP16 depth model to install (shared by all presets). */
export function depthModelSkill(): SkillDef {
  return findSkill("depth_anything")!; // FP16 files[] — see SKILL_REGISTRY
}

/**
 * Download a skill, multi-file or single-URL. Aggregates progress into a
 * single 0-100 percentage and feeds it through `onProgress` along with the
 * raw byte counters when available. For multi-file skills the per-file
 * progress is normalised across the whole batch so the UI sees a smooth
 * 0..100 bar.
 */
export async function downloadSkill(
  skill: SkillDef,
  onProgress?: (pct: number, downloaded?: number, total?: number | null) => void,
): Promise<void> {
  if (skill.files && skill.files.length > 0) {
    const fileCount = skill.files.length;
    try {
      // Carry the last known byte counts across the file boundary.
      //
      // The per-file completion tick used to call `onProgress(pct)` with no
      // bytes at all, so every multi-file install (face_small, face_large,
      // depth_anything) dropped back to the UI's "Starting…" fallback at each
      // file boundary — i.e. at the 50% and 100% marks of the download. It read
      // as though the transfer had restarted or hung.
      let lastDl: number | undefined;
      let lastTot: number | null | undefined;
      for (let i = 0; i < fileCount; i++) {
        const f = skill.files[i];
        await api.downloadSkill(skill.id, f.url, (pct, dl, tot) => {
          // Each file occupies an equal slice of the overall progress. Bytes
          // pass through unchanged so the UI shows the active file's MB.
          const overall = Math.round(((i + pct / 100) / fileCount) * 100);
          lastDl = dl; lastTot = tot;
          onProgress?.(overall, dl, tot);
        }, f.filename, f.sha256);
        onProgress?.(Math.round(((i + 1) / fileCount) * 100), lastDl, lastTot);
      }
    } catch (e) {
      // ALL-OR-NOTHING. A multi-file skill is only usable with every file: if
      // face_small got its detector but not its embedder, "installed" checks
      // (any .onnx in the dir) said yes and recognition then failed at load
      // with no way for the user to see why. Roll the partial install back so
      // the card stays on "Install" and one retry fixes it.
      await api.removeSkill(skill.id).catch(() => {});
      throw e;
    }
    return;
  }
  await api.downloadSkill(skill.id, skill.installUrl, onProgress);
}

export async function removeSkill(skill: SkillDef): Promise<void> {
  await api.removeSkill(skill.id);
}
