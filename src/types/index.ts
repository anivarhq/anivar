/**
 * Shared TypeScript types and small helpers for Anivar.
 *
 * Anything that describes a value crossing the Tauri command boundary or
 * shared between two features lives here. Keeping API wrappers
 * (`src/api/index.ts`) free of type definitions lets features import types
 * without pulling in the entire `invoke` surface.
 *
 * Backwards compatibility: `src/api/index.ts` re-exports everything from
 * this module, so `import { MotionEvent } from "../../api"` still works.
 */

export interface Settings {
  sensitivity: number;
  motion_threshold: number;
  record_on_motion: boolean;
  record_pre_buffer_secs: number;
  record_post_buffer_secs: number;
  record_max_event_secs: number;
  stream_port: number;
  stream_quality: number;
  retention_days: number;
  ai_model: string;
  // Guardian agent
  agent_enabled: boolean;
  vision_model: string;
  agent_poll_secs: number;
  alert_min_risk: string;
  /** Alert categories the user muted ("person"|"vehicle"|"animal"|"audio"|"other").
   *  Muted = still recorded/analyzed, just no channel message. Synced with the
   *  Telegram 👁 Alert filter (same settings field). */
  alert_muted_categories?: string[];
  camera_name: string;
  telegram_bot_token: string;
  telegram_chat_id: string;
  github_repo: string;
  device_name: string;
  auth_username: string;
  auth_password_hash: string;
  // AI Provider
  ai_provider: string;
  openai_api_key: string;
  anthropic_api_key: string;
  groq_api_key: string;
  xai_api_key: string;
  gemini_api_key: string;
  openai_compatible_url: string;
  openai_compatible_key: string;
  // Auto Re-ID
  auto_reid: boolean;
  reid_threshold: number;
  // Semantic event search (standard CLIP) — active embedding model.
  search_model?: "off" | "mobileclip_s0" | "clip_b32" | "jina_clip";
  // ONNX hardware acceleration: "auto"/"gpu" → DirectML (Windows) w/ CPU fallback, "cpu" → force CPU.
  inference_device?: "auto" | "gpu" | "cpu";
  // Face recognition (standard; native Rust pipeline)
  face_model?: "off" | "small" | "large";    // "off" | FaceNet 128-d CPU | ArcFace 512-d GPU/NPU
  face_detection_threshold?: number;         // Mature NVRs default 0.7
  face_recognition_threshold?: number;       // Mature NVRs default 0.9
  face_unknown_score?: number;               // Mature NVRs default 0.8
  face_class_confidence?: number;            // hybrid head: min softmax prob (0..1) to name
  face_liveness?: boolean;                   // anti-spoofing gate (needs face_liveness skill)
  // NVR
  nvr_enabled: boolean;
  nvr_segment_mins: number;
  nvr_max_gb: number;
  nvr_record_mode?: "always" | "motion_only" | "events_only"; // default: always
  nvr_retain_days?: number;
  nvr_retain_event_days?: number;
  /** Save each event's short clip before its raw footage is pruned. */
  keep_event_clips?: boolean;
  /** Appliance mode: keep-alive Scheduled Task relaunches the app (headless)
   *  within 5 min after a hard crash. Opt-in, Windows-only. */
  relaunch_after_crash?: boolean;
  // Masking & zones (JSON-encoded per camera)
  camera_masks?: string;
  /** Depth Map Anonymization per cam: JSON {"<camId>": true}. Server-enforced. */
  depth_anonymize?: string;
  /** Depth model variant for anonymization: "fp16" (default) | "int8" | "q4f16" | "fp32". */
  depth_model?: string;
  /** Client depth VIEW prefs per cam: JSON {"<camId>":{mode,opacity}}. Cosmetic. */
  depth_privacy?: string;
  // on-device assistants-inspired intelligence features
  loitering_detection?: boolean;     // alert when person lingers > threshold
  loitering_threshold_secs?: number; // seconds before loitering alert (default 30)
  crowd_detection?: boolean;         // alert when > N people in frame
  crowd_threshold?: number;          // person count threshold (default 3)
  repeat_visitor_detection?: boolean;
  repeat_visitor_threshold?: number; // appearances in 24h before alert (default 3)
  // assistant-parity / Cookbook fields. These exist in the Rust Settings struct;
  // we surface them so the Cookbook footer (persona) and Settings Notifications
  // section can edit them with type safety.
  quiet_hours_enabled?: boolean;
  quiet_hours_start?:   string;   // "HH:MM"
  quiet_hours_end?:     string;   // "HH:MM"
  agent_persona_name?:  string;
  agent_persona_text?:  string;
  strobe_frames?:       number;   // user floor for adaptive picker (v7)
  /** YOLO 2026 tier picker — chooses which `skills/yolo26{n,s,m,l,x}/` the
   *  inference loop loads. Default `"xlarge"` (back-compat with users who
   *  already installed the original yolo26 skill). */
  yolo_variant?:        "nano" | "small" | "medium" | "large" | "xlarge";
  /** Which on-device language model runs. Mirrors `yolo_variant`. */
  local_llm_tier?:      "fast" | "balanced" | "vision";
  // ── v7 richness knobs ───────────────────────────────────────────────────
  /** Strobe-extraction profile — duration→frame-count table picker.
   *  Trades VLM cost for clip understanding on long clips. */
  strobe_profile?:             "conservative" | "balanced" | "aggressive";
  /** Laplacian blur floor below which face crops are dropped before embedding.
   *  0.0 = keep everything; 0.5+ = only sharp faces. Default 0.20. */
  face_quality_floor?:         number;
  /** YOLO 2026 confidence threshold. Lower = catches more, more false positives.
   *  Default 0.20 (matches the pre-v7 hardcoded value). */
  yolo_confidence_threshold?:  number;
  /** Comma-separated COCO class names to keep, e.g. `"person,car"`.
   *  Empty = no filter. */
  yolo_class_filter?:          string;
  /** ALPR regional model picker. Maps to `skills/alpr_{region}/model.onnx`. */
  alpr_region?:                "global" | "european" | "argentinian";
  known_plates?:               string;   // "PLATE=Name" per line → names matched plates
  // Audio event detection (YAMNet) — RTSP cameras with an audio track.
  settings_version?:           number;  // backend migration marker — round-trip, never edit
  lan_access?:                 boolean; // serve on LAN (0.0.0.0) vs localhost-only; applies on restart
  audio_detection?:            boolean;
  audio_listen?:               string;   // comma-separated AudioSet class substrings
  audio_threshold?:            number;   // 0..1 per-class score to fire
  // ── v8 motion hysteresis ────────────────────────────────────────────────
  /** Consecutive motion-detected frames required to open / sustain an event.
   *  Default 3. The lifecycle-bug fix: single noisy frames no longer trip
   *  events open or refresh the close timer. */
  motion_min_frames?:           number;
  /** Multiplier on `sensitivity` for the OPEN threshold (sustain uses raw
   *  `sensitivity`). Default 1.5×. Higher = harder to open events. */
  motion_open_score_mult?:      number;
  /** mature NVRs lightning_threshold: ignore a frame when >this fraction changed at
   *  once (lighting / IR / exposure shift). 0 disables. Default 0.85. */
  motion_lightning_threshold?:  number;
  /** Opt-in edge-AI-style gate: drop events where YOLO never confirms a
   *  tracked class (person/vehicle/animal/package) within post-buffer. */
  require_object_to_open_event?: boolean;
  // ── v9 mature NVRs-model knobs ──────────────────────────────────────────────
  /** Frames of bbox absence before the tracked-object presence signal goes
   *  false. Drives the object-driven event close. Default 15. */
  detect_max_disappeared_frames?: number;
  /** Re-run the agent's clip analysis every N seconds during a long open
   *  event so mid-event activity actually alerts the user. 0 disables.
   *  Default 60. */
  re_analysis_interval_secs?: number;
  // ── v11 channels-first toggles ─────────────────────────────────────────
  /** When true (default), motion alerts to Telegram/Discord/Pushover/HA
   *  carry the event thumbnail inline. */
  attach_snapshot_to_alerts?: boolean;
  /** When true (default), Telegram + Discord clip uploads ride the same
   *  alert message. Larger clips are transcoded to fit. */
  attach_clip_to_alerts?: boolean;
  /** Default expiry in minutes for "Share Live View" / "Share Clip" links.
   *  0 means "until app restart". */
  live_share_default_minutes?: number;
  /** When true, the public tunnel auto-stops once all shares expire. */
  tunnel_auto_stop?: boolean;
}

/** v11 share-link result returned by `generateShareLink`. */
export interface ShareLinkResult {
  url:         string;
  kind:        "live" | "clip";
  resource_id: string;
  /** Unix epoch seconds. 0 = never expires (until app restart). */
  expires_at:  number;
}

/** v11 active-share tracker entry returned by `listActiveShares`. */
export interface ShareEntry {
  kind:        "live" | "clip";
  resource_id: string;
  expires_at:  number;
  created_at:  string;
  /** The minted /redeem URL (empty for entries persisted before this field). */
  url:         string;
}

/** v9 pull-state for the YOLO badge. Matches the Rust struct
 *  `crate::state::InferenceStatusSnapshot`. */
export interface InferenceStatus {
  state:   "not_installed" | "loading" | "ready" | "error";
  variant: string | null;
  fps:     number;
  since:   string | null;
}

// ── Mask / Zone types ─────────────────────────────────────────────────────────

/** Mask classes (standard). `motion` suppresses raw motion pixels;
 *  `object` suppresses YOLO detections by bottom-center test; `zone` is
 *  informational (named region, used as event metadata + agent context).
 *  v8: `object` is new — pre-v8 installs only had motion+zone.  */
export type MaskType = "motion" | "object" | "zone" | "speed" | "line";

export interface CameraMask {
  id:     string;
  name:   string;
  points: string;   // normalized "x1,y1,x2,y2,..." (0–1 range, like mature NVRs)
  type:   MaskType;
  color?: string;   // zone display colour
  alert_on_enter?: boolean; // zone only
  // ── Speed zone (type "speed") — a 4-point ground quad + real-world size. ──
  width_m?:  number;   // real width of the quad in metres
  height_m?: number;   // real height (depth) of the quad in metres
  // ── NVR-parity zone properties (zone type only; optional, JSON-rider so
  //    no backend schema change is needed). ─────────────────────────────────
  /** Which object classes count for this zone. Empty/undefined = any object. */
  objects?: string[];          // e.g. ["person","car"]
  /** Consecutive frames an object's bottom-center must stay inside before the
   *  zone registers it (mature NVRs default 3). Suppresses bbox jitter. */
  inertia?: number;
  /** Minimum seconds an object must linger inside before it counts as "in zone"
   *  (mature NVRs loitering_time). 0 = register immediately. */
  loitering_secs?: number;
}

export type CameraMaskMap = Record<string, CameraMask[]>; // camId → masks

export function parseMasks(raw: string | undefined): CameraMaskMap {
  if (!raw) return {};
  try { return JSON.parse(raw) as CameraMaskMap; } catch { return {}; }
}

export function serializeMasks(map: CameraMaskMap): string {
  return JSON.stringify(map);
}

export interface AgentAlert {
  id: string;
  event_id: string;
  risk_level: "low" | "medium" | "high" | "critical";
  threat_type: string;
  summary: string;
  is_false_positive: boolean;
  actions_taken: string | null;
  created_at: string;
  feedback: string | null;
}

export interface AgentStatus {
  enabled: boolean;
  /** The ACTIVE provider is usable: on-device model on disk, or a cloud key set. */
  provider_ready: boolean;
  last_run_at: string | null;
  model: string;
  vision_model: string;
  pending_events: number;
  total_analyzed: number;
}

/** One piece of playable evidence in an in-app Guardian chat reply.
 *  Events are NOT parts — they arrive as `ChatAppReply.events`. */
export interface ChatPart {
  type: "snapshot" | "person" | "link" | "chart";
  cam?: number | null;
  name?: string | null;
  thumbnail?: string | null; // bare base64 (person cards)
  url?: string | null;       // share link
  label?: string | null;     // share link
  body?: string | null;      // chart (mermaid fence)
}

/** `chat_app` reply: display text + structured evidence. */
export interface ChatAppReply {
  text: string;
  parts: ChatPart[];
  /** The events this answer is about, as the same cards Telegram renders into
   *  an album. They ride on the reply so cards belong to the message that
   *  produced them, rather than arriving through a global event. */
  events: EventCard[];
  /** Estimated prompt tokens in play and the model's window — drives the
   *  context ring in the composer. Approximate by design. */
  context_used: number;
  context_limit: number;
}

/** One event card — the shape BOTH surfaces render, from `conditions::card()`. */
export interface EventCard {
  id: string;
  started_at: string;
  ts: string;
  cam: number;
  duration?: string | null;
  summary?: string | null;
  risk_level: string;
  threat_type: string;
  has_clip: boolean;
  thumbnail?: string | null;
}

export interface FrameResult {
  motion_detected: boolean;
  motion_score: number;
  recording: boolean;
  event_id: string | null;
  /** Normalised [x1,y1,x2,y2] motion bounding boxes from Rust's background subtractor.
   *  Passed to the YOLO worker to crop inference to changed regions only. */
  motion_regions: [number, number, number, number][];
}

export interface StreamInfo {
  local_ip: string;
  port: number;
  url: string;
  url_with_token: string;
  auth_token: string;
  qr_data_url: string; // PNG data URI — safe for <img src> without XSS risk
}

export interface Detection {
  label: string;
  score: number;
  box: { xmin: number; ymin: number; xmax: number; ymax: number };
}

export interface MotionEvent {
  id: string;
  started_at: string;
  ended_at: string | null;
  duration_secs: number | null;
  peak_score: number;
  clip_path: string | null;
  thumbnail: string | null;
  detections: string | null;   // JSON-encoded Detection[]
  ai_summary: string | null;
  cam_id?: number;             // camera index (0-based); may be absent on old rows
  /** standard category derived from YOLO detections — "person", "vehicle",
   *  "animal", "package", or "other". Used by the Events row to render a type
   *  badge and filter chips. */
  event_category?: EventCategory | null;
  /** License-plate text recognised by the `alpr` skill when a vehicle was in
   *  frame. Null when the skill isn't installed or no plate was visible. */
  recognized_plate?: string | null;
  /** Top ground speed (km/h) seen inside a calibrated speed zone during the event. */
  top_speed_kmh?: number | null;
  /** CSV of named zones (v8) whose polygons contained at least one
   *  detection's bottom-center during the event. Populated by
   *  `analyze_event_clip`. Used by ReviewPanel to render zone chips. */
  zones_entered?: string | null;
  /** v27 mature NVRs per-label: the specific dominant COCO object — "person",
   *  "dog", "car", etc. The real object shown as the Review chip, distinct from
   *  the broad `event_category` bucket. */
  dominant_label?: string | null;
  /** v27 mature NVRs sub-label refinement: known face name, recognised plate, or
   *  delivery brand attached to the parent object. */
  sub_label?: string | null;
  /** v28 RFC3339 wall-clock when YOLO first confirmed an object. `started_at` is
   *  the earlier motion onset; the event clip is anchored here (minus the
   *  pre-buffer) so the playhead lands on the subject, not the empty lead-in. */
  first_object_at?: string | null;
  /** NVR-parity structured attributes — JSON `[{type,value,score,known_name?}]`
   *  (type ∈ face|plate|object). The typed, scored label set for the event. */
  attributes?: string | null;
  /** Confidence (0–1) of `sub_label` (top face score, else plate score). */
  sub_label_score?: number | null;
  /** Confidence (0–1) of `recognized_plate`. */
  plate_score?: number | null;
  /** Whether the agent judged this event a false positive (filterable). */
  false_positive?: boolean | null;
}

export type EventCategory = "person" | "vehicle" | "animal" | "package" | "audio" | "fall" | "crossing" | "other";

/** One entry in an event's lifecycle timeline (mature NVRs `Timeline` parity).
 *  Returned ordered by occurrence by `api.getEventTimeline`. */
export interface TimelineEntry {
  ts: string;
  /** appeared | entered_zone | recognized | lpr | attribute | speed | crossing | audio | fall | gone */
  class_type: string;
  label: string | null;
  value: string | null;
  score: number | null;
}

export interface KnownPerson {
  id: string;
  name: string;
  role: "resident" | "employee" | "visitor" | string;
  /** Listings ship a stub "[]" — the arrays are matcher-internal (~190KB/person).
   *  Use `embedding_count` for the enrolled-angle count. */
  embeddings: string;
  embedding_count: number;
  thumbnail: string | null;
  created_at: string;
  last_seen_at: string | null;
}

export interface CameraConfig {
  cam_id: number;
  name: string;
  source_type: "browser" | "rtsp" | "mjpeg" | "native" | string;
  source_url: string;
  device_id: string;
  enabled: boolean;
  /** RTSP transport — "tcp" (default) or "udp". Optional; backend defaults to tcp. */
  transport?: "tcp" | "udp" | string;
  /** Make/brand the user picked or typed (e.g. "Reolink", "Lorex"). Optional. */
  brand?: string;
  /** Optional low-res sub-stream URL used ONLY for detection (record stays on
   *  the main stream) — mature NVRs' detect-substream model. */
  detect_url?: string;
}

/** One GPU adapter's live usage (integrated or discrete). */
export interface GpuMetric {
  name: string;
  util: number;        // 3D-engine %, 0 when idle
  mem_mb: number;      // dedicated GPU memory in use
  is_discrete: boolean;
}

/** Live host utilization for the System Monitor. */
export interface SystemMetrics {
  cpu_total: number;
  per_core: number[];
  mem_used_mb: number;
  mem_total_mb: number;
  gpus: GpuMetric[];   // every adapter — integrated + discrete
  accelerator: string; // active ONNX EP (DirectML / TensorRT-RTX / CoreML / CPU)
  infer_stats: InferStatRow[]; // per-model latency (avg incl. GPU-lock wait)
}

export interface InferStatRow {
  model: string;
  count: number;
  avg_ms: number;
  p95_ms: number;
}

/** One execution provider and why it is (or isn't) running inference here. */
export interface AccelRow {
  ep: string;
  /** armed = running · ready = usable but another lane won · unavailable = can't
   *  run here · n/a = not applicable to this hardware. */
  state: "armed" | "ready" | "unavailable" | "n/a" | string;
  detail: string;
}

export interface TrtxStatus {
  supported: boolean;       // Windows + NVIDIA adapter
  provisioned: boolean;     // TensorRT-RTX pack on disk
  trt_provisioned: boolean; // classic TensorRT pack on disk
  active: boolean;          // some NVIDIA EP passed its canary
  active_ep: "nvrtx" | "tensorrt" | "cuda" | "none" | string;
  /** TensorRT-RTX runtime version the shipped provider links (e.g. "1.3"), read
   *  from its import table — the SDK picker names this instead of a hardcoded
   *  number that silently rots on the next ONNX Runtime bump. "" if unknown. */
  trtx_required_runtime: string;
}

/** Tailscale Funnel remote-access status (compliant live/clip sharing). */
export interface TailscaleStatus {
  installed: boolean;
  logged_in: boolean;
  dns_name: string;      // machine.tailnet.ts.net
  base_url: string;      // https://<dns_name>
  funnel_active: boolean;
  enable_url: string;    // one-time consent URL when Funnel isn't enabled yet
}

/** Measured recording rate vs the disk cap (Storage settings projection). */
export interface DiskProjection {
  gb_per_day: number;            // measured over the last 24h of indexed segments
  cap_gb: number;                // nvr_max_gb
  retention_days_setting: number;
  projected_days_at_cap: number; // days the cap holds at the measured rate
  recording_cams: number;        // cams that recorded in the last 24h
  record_mode: string;           // active retain mode
}

/** Detailed per-camera telemetry for the Device Info panel. */
export interface CameraTelemetry {
  cam_id: number;
  name: string;
  brand: string;
  source_type: string;
  source_url_masked: string;
  device_id: string;
  transport: string;
  online: boolean;
  last_frame_secs: number | null;
  nvr_enabled: boolean;
  recording: boolean;
  segments_total: number;
  bytes_total: number;
  duration_total_secs: number;
  segments_today: number;
  bytes_today: number;
  oldest_at: string | null;
  newest_at: string | null;
}

/** Result of a pre-save connection test (ffprobe) against a camera URL. */
export interface StreamProbe {
  ok: boolean;
  codec: string;        // h264 / hevc / mjpeg …
  width: number;
  height: number;
  fps: number;
  has_audio: boolean;
  audio_codec: string;
  error: string;        // human-readable when ok=false
}

export interface GpuInfo {
  name: string;
  is_discrete: boolean;
}

export interface DiscoveredCamera {
  ip: string;
  port: number;
  kind: "rtsp" | "mjpeg" | "http" | "onvif";
  url: string;
  name: string;
}

export interface NativeCameraDevice {
  index: number;
  name: string;
  description: string;
}

export interface BrowserCameraInfo {
  device_id: string;
  label: string;
}

export interface CameraInventory {
  native: NativeCameraDevice[];
  network: DiscoveredCamera[];
  browser: BrowserCameraInfo[];
}

export interface StorageInfo {
  total_bytes: number;
  clip_count: number;
  event_count: number;
  orphaned_clips: number;
  oldest_event: string | null;
  newest_event: string | null;
  nvr_bytes: number;
  nvr_count: number;
}
