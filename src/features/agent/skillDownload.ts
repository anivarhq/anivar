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

export type AlprRegion = "global" | "european";

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
  | "alpr_global" | "alpr_european"
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
  /** What to download into `skills/<id>/`, each file pinned to a revision and
   *  a SHA-256 (a mismatch fails the install). */
  files:        SkillFile[];
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
    files: [
      { url: "https://huggingface.co/onnx-community/yolo26n-ONNX/resolve/a8dc7e14743e1cea8ccd493bd99b4c2827de1acf/onnx/model.onnx",
        filename: "model.onnx", label: "YOLO26 N",
        sha256: "cda08d9440217e243e075ee839f40383c59b3f973e493f9f6c7452922a69436e" },
    ],
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
    files: [
      { url: "https://huggingface.co/onnx-community/yolo26s-ONNX/resolve/37669b009f416cb1df28751257d6ec5f8e4b4e20/onnx/model.onnx",
        filename: "model.onnx", label: "YOLO26 S",
        sha256: "c72bc5ad4e7f7c87666a051d0f01fc02d084688c1ca78e825031827a0590efb1" },
    ],
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
    files: [
      { url: "https://huggingface.co/onnx-community/yolo26m-ONNX/resolve/a1db4877f0a3ed68554c231cdae958e2280087e3/onnx/model.onnx",
        filename: "model.onnx", label: "YOLO26 M",
        sha256: "7cd89faaa164887c3c33ee0fe5d63723bf262f75dd90088c4cb3be5a310aabff" },
    ],
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
    files: [
      { url: "https://huggingface.co/onnx-community/yolo26l-ONNX/resolve/7a72d097ea1719e10eda2d89f75b6e9c009f7f6b/onnx/model.onnx",
        filename: "model.onnx", label: "YOLO26 L",
        sha256: "60938f3c91e456337439f962d1185bffc5f7ababa902dbaf4686ae86a66952bd" },
    ],
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
    files: [
      { url: "https://huggingface.co/onnx-community/yolo26x-ONNX/resolve/032eec52a570b81e5f9d66485b9322bd9b7f4053/onnx/model.onnx",
        filename: "model.onnx", label: "YOLO26 X",
        sha256: "73970a0227222a6e454b7b12f8453bd228183ebc688a62699e7ba57b3e105c42" },
    ],
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
    files: [
      { url: "https://huggingface.co/deepghs/yolo-face/resolve/e3662574830c534dfcc9c3b7ea4d89272f8aae4e/yolov8n-face/model.onnx",
        filename: "detector.onnx", label: "yolov8n-face — detector",
        sha256: "fd27189bfe5750a017648445700473459a6d02e7c3b0a3bfd8a54af77dd3b046" },
      { url: "https://huggingface.co/onnxmodelzoo/arcfaceresnet100-11-int8/resolve/c0ec783c5907f34e089495d6d0428e847fcededa/arcfaceresnet100-11-int8.onnx",
        filename: "embedder.onnx", label: "ArcFace INT8 — recognizer",
        sha256: "c625ca68a422418c48aa84f73341337e0a92b111f327909005d1eec07c95f936" },
    ],
    license:     "GPL-3.0 + Apache-2.0",
    licenseNote: "The face detector is YOLOv8n-face from akanametov/yolo-face (GPL-3.0, built on Ultralytics YOLOv8): copyleft terms reach whatever you build around it. The recognizer is the ONNX Model Zoo's ArcFace (Apache-2.0); its training data, MS-Celeb-1M, was released for non-commercial research only.",
  },
  {
    id:          "face_large",
    name:        "Face Recognition — Large",
    slot:        "face",
    description: "standard face pipeline at full FP32 precision. Same detector as Small. Best accuracy; real-time speed needs a discrete GPU or NPU.",
    sizeLabel:   "~262 MB",
    badge:       "ACCURATE",
    badgeColor:  "var(--status-ok)",
    files: [
      { url: "https://huggingface.co/deepghs/yolo-face/resolve/e3662574830c534dfcc9c3b7ea4d89272f8aae4e/yolov8n-face/model.onnx",
        filename: "detector.onnx", label: "yolov8n-face — detector",
        sha256: "fd27189bfe5750a017648445700473459a6d02e7c3b0a3bfd8a54af77dd3b046" },
      { url: "https://huggingface.co/garavv/arcface-onnx/resolve/224c23cbbdc27a22add7fa538dee5b2aa9304b83/arc.onnx",
        filename: "embedder.onnx", label: "ArcFace FP32 — recognizer",
        sha256: "ffe014a45c9488506719d37fd578ece6661bb385535b36e8039975fa5d4683db" },
    ],
    license:     "GPL-3.0 + unstated",
    licenseNote: "The face detector is YOLOv8n-face from akanametov/yolo-face (GPL-3.0, built on Ultralytics YOLOv8): copyleft terms reach whatever you build around it. The recognizer (garavv/arcface-onnx) states no licence at all, so its author has granted no rights beyond downloading it: avoid it for anything commercial.",
  },
  // ── License plate recognition (mature NVRs 0.16+ shipped ALPR as a first-class
  //    feature; we model it the same way — a separate skill that runs after a
  //    vehicle is detected in an event). Model is the Mobile-ViT v2 OCR from
  //    ankandrew/fast-plate-ocr, distributed via cnn-ocr-lp release tags.
  //    Two regional variants: global (default, multi-region) and european
  //    (EU-format plates). Pick whichever matches the cameras' geography.
  {
    id:          "alpr_global",
    name:        "License Plates — Global",
    slot:        "alpr",
    alprRegion:  "global",
    description: "Default region. Multi-format plates, works worldwide. Best starting point if you don't know which region matches your cameras.",
    sizeLabel:   "~8 MB",
    badge:       "DEFAULT",
    badgeColor:  "var(--status-idle)",
    // Force the saved filename so `alpr.rs::find_alpr_model` (which looks for
    // `skills/alpr_global/model.onnx`) finds it without a directory scan.
    files: [
      { url:      "https://github.com/ankandrew/cnn-ocr-lp/releases/download/arg-plates/global_mobile_vit_v2_ocr.onnx",
        filename: "model.onnx",
        label:    "Global Mobile-ViT v2 OCR",
        sha256: "e2fe5946c02e73901f73d21b3b7223a0c3e3b02059d4964fbf279b01af38d63c" },
    ],
    license:     "MIT",
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
    files: [
      { url:      "https://github.com/ankandrew/cnn-ocr-lp/releases/download/arg-plates/european_mobile_vit_v2_ocr.onnx",
        filename: "model.onnx",
        label:    "European Mobile-ViT v2 OCR",
        sha256: "5f388f57ddec318d38d17e420d292f5a049595bec93f111838903f6617f6943f" },
    ],
    license:     "MIT",
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
    files: [
      { url: "https://huggingface.co/Xenova/mobileclip_s0/resolve/757d59c9c6870a76a4b0306f05f5061bca15c39f/onnx/vision_model.onnx",
        filename: "vision_model.onnx", label: "Vision encoder",
        sha256: "17d3c037b1d488c10c50e09f6009ea5a198caef4e0e8f4ea5617b7cb2d067ac0" },
      { url: "https://huggingface.co/Xenova/mobileclip_s0/resolve/757d59c9c6870a76a4b0306f05f5061bca15c39f/onnx/text_model.onnx",
        filename: "text_model.onnx", label: "Text encoder",
        sha256: "f6e9bd5742bfc515889e901634d8a2ff2a57fab8564e4ad3760e800b1a51b77c" },
      { url: "https://huggingface.co/Xenova/mobileclip_s0/resolve/757d59c9c6870a76a4b0306f05f5061bca15c39f/tokenizer.json",
        filename: "tokenizer.json", label: "Tokenizer",
        sha256: "72ed5c96db5729294468543e4bc75fce14ca63f58e37300290189ba1c1e52b85" },
    ],
    license:     "Apple ML Research licence",
    licenseNote: "Apple released MobileCLIP's weights for non-commercial research only: commercial use, including in a commercial product or service, isn't permitted. CLIP ViT-B/32 (MIT) and Jina-CLIP (Apache-2.0) have no such limit.",
  },
  {
    id:          "depth_anything",
    name:        "Depth Anonymization — Depth-Anything-v2",
    slot:        "privacy",
    description: "edge-AI-style privacy: the camera feed becomes a colorized depth map (near = warm, far = cool) — movement stays trackable while faces and identities never exist in recordings, streams or alerts. Local AI still analyzes raw frames in memory.",
    sizeLabel:   "~50 MB",   // FP16 default (see DEPTH_VARIANTS for the tier picker)
    badge:       "PRIVACY",
    badgeColor:  "var(--status-idle)",
    // Default = FP16 (half the old FP32, GPU-native). CameraView's auto-install
    // on first toggle uses this; the Arsenal Depth card offers all tiers.
    files: [
      { url: "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/4472b7362082ad9968fee890ca0f1e5aca36b93d/onnx/model_fp16.onnx",
        filename: "model_fp16.onnx", label: "Depth model (FP16)",
        sha256: "2df6223f206b5164e21f664ace61dabeb9bb6a49b8b5a3e00510b4807d0f5b04" },
    ],
    license:     "Apache-2.0",
  },
  {
    id:          "clip_b32",
    name:        "Semantic Search — CLIP ViT-B/32",
    slot:        "search",
    description: "OpenAI's classic CLIP ViT-B/32. A well-understood, balanced baseline. Bigger than MobileCLIP but broadly compatible.",
    sizeLabel:   "~579 MB",
    badge:       "STD",
    badgeColor:  "var(--status-idle)",
    files: [
      { url: "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/d15189d7028b43f1d3e65039190477f6af591c2a/onnx/vision_model.onnx",
        filename: "vision_model.onnx", label: "Vision encoder",
        sha256: "fd6e1402a588279d1723c7534d4bcba5bc0b14b47dfab0e46f8c47b8270d7d40" },
      { url: "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/d15189d7028b43f1d3e65039190477f6af591c2a/onnx/text_model.onnx",
        filename: "text_model.onnx", label: "Text encoder",
        sha256: "3f6571f5bad13a97c469c1622e1cfc4d9aef78b79fdbfcff804ca357bfada8cc" },
      { url: "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/d15189d7028b43f1d3e65039190477f6af591c2a/tokenizer.json",
        filename: "tokenizer.json", label: "Tokenizer",
        sha256: "f7f3b7af117d467b58374797691a6438d3e6b9e9cef800dfd5dced7f697a90cd" },
    ],
    license:     "MIT",
  },
  {
    id:          "jina_clip",
    name:        "Semantic Search — Jina-CLIP",
    slot:        "search",
    description: "Jina-CLIP-v1 (768-d). Largest/most accurate tier; multilingual-leaning. Runs after an event closes; CPU-friendly but a heavier download.",
    sizeLabel:   "~850 MB",
    badge:       "MAX",
    badgeColor:  "var(--status-idle)",
    files: [
      { url:      "https://huggingface.co/jinaai/jina-clip-v1/resolve/ceb3e44ca4d6eceaa4f3fb58b1c1a5748b3f29b6/onnx/vision_model.onnx",
        filename: "vision_model.onnx",
        label:    "Vision encoder (EVA02)",
        sha256: "3032bf5f20bc39e23e9d2fff51ed1406341f1490bade128ffdcc6307af63c246" },
      { url:      "https://huggingface.co/jinaai/jina-clip-v1/resolve/ceb3e44ca4d6eceaa4f3fb58b1c1a5748b3f29b6/onnx/text_model.onnx",
        filename: "text_model.onnx",
        label:    "Text encoder (JinaBERT)",
        sha256: "d7b2e732e685dcaf5ca2a36e7f5d9169906859f4590766689dbf7b00b20da01b" },
      { url:      "https://huggingface.co/jinaai/jina-clip-v1/resolve/ceb3e44ca4d6eceaa4f3fb58b1c1a5748b3f29b6/tokenizer.json",
        filename: "tokenizer.json",
        label:    "Tokenizer",
        sha256: "b36ee0ed6d20d181de65ce729dea6169658a9cfa4dede6717ba9fa2e4fbd3bc7" },
    ],
    license:     "Apache-2.0",
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
    files: [
      // VERIFIED tf2onnx export of TF-Hub YAMNet (16 MB): input `waveform` (variable-
      // length 1-D f32 @16 kHz), output_0 = [frames, 521] AudioSet scores — exactly what
      // audio.rs feeds/reads (downloaded + inference-tested before shipping). The old
      // `onnx-community/yamnet` URL 401'd and saved a 29-byte "Invalid username or
      // password." page as the model. jafet21/yamnetonnx is an equivalent fallback.
      { url: "https://huggingface.co/zeropointnine/yamnet-onnx/resolve/ac2ca3bd45d12ec1f19f1144205ea529b4e9dedf/yamnet.onnx",
        filename: "model.onnx", label: "YAMNet classifier",
        sha256: "1510041dce24a2e9e84ec546807ac408ae496da6d1ed41bc3ccba649623f8e19" },
      { url: "https://raw.githubusercontent.com/tensorflow/models/dfffd623b6be8d1d9744b8e261fbac370d17c46d/research/audioset/yamnet/yamnet_class_map.csv",
        filename: "class_map.csv", label: "AudioSet class map",
        sha256: "cdf24d193e196d9e95912a2667051ae203e92a2ba09449218ccb40ef787c6df2" },
    ],
    license:     "Apache-2.0",
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
    files: [
      { url:      "https://huggingface.co/LiquidAI/LFM2.5-350M-GGUF/resolve/657e078c94084481950a2d555a941481f715536b/LFM2.5-350M-Q4_K_M.gguf",
        filename: "model.gguf",
        label:    "LFM2.5-350M (Q4_K_M)",
        sha256: "7e6f72643caafc9a68256686638c4d7916f2cec76d1df478d4c3ddcd95a6aed4" },
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
    files: [
      { url:      "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF/resolve/8ed288026e23958ad9dfa92d53ed773a8eee7125/LFM2.5-1.2B-Instruct-Q4_K_M.gguf",
        filename: "model.gguf",
        label:    "LFM2.5-1.2B-Instruct (Q4_K_M)",
        sha256: "b1b3de114215d9507409a662a501a631095a479a419584e8a2ded6304b19b4f5" },
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
    files: [
      { url:      "https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-GGUF/resolve/36fc16bc95133424921bcc3da009e83b2f23ffb5/LFM2.5-VL-1.6B-Q4_K_M.gguf",
        filename: "model.gguf",
        label:    "LFM2.5-VL-1.6B (Q4_K_M)",
        sha256: "aefc3c97c9eb30d9c0dd6af4c38250f5f5106b57c8cf92de7914c7d0a9c94da2" },
      { url:      "https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-GGUF/resolve/36fc16bc95133424921bcc3da009e83b2f23ffb5/mmproj-LFM2.5-VL-1.6b-Q8_0.gguf",
        filename: "mmproj.gguf",
        label:    "Vision projector (Q8_0)",
        sha256: "2ce89e610c56f3198ece2b86cf61743a08b9307279c89125eb2412ebb908689d" },
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
    files: [
      { url: "https://huggingface.co/Xenova/movenet-singlepose-lightning/resolve/ed0f314bb7356fd1dbf1e4f52c2d40791bf6534f/onnx/model.onnx",
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
}

export async function removeSkill(skill: SkillDef): Promise<void> {
  await api.removeSkill(skill.id);
}
