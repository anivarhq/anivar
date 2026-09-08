import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

// Shared types live in src/types/. Re-exported here so existing imports keep
// working (`import type { MotionEvent } from "../../api"`).
export * from "../types";
import type {
  AgentAlert,
  AgentStatus,
  BrowserCameraInfo,
  CameraConfig,
  CameraInventory,
  DiscoveredCamera,
  EventCard,
  FrameResult,
  GpuInfo,
  InferenceStatus,
  KnownPerson,
  ShareEntry,
  ShareLinkResult,
  MotionEvent,
  NativeCameraDevice,
  Settings,
  StorageInfo,
  StreamInfo,
  StreamProbe,
  CameraTelemetry,
  SystemMetrics,
  InferStatRow,
  TrtxStatus,
  AccelRow,
  TailscaleStatus,
  DiskProjection,
  TimelineEntry,
  ChatAppReply,
} from "../types";

export const api = {
  streamFrame: (frameB64: string, camId = 0) =>
    invoke<void>("stream_frame", { frameB64, camId }),

  processFrame: (frameB64: string, camId = 0, timestampMs = 0) =>
    invoke<FrameResult>("process_frame", { frameB64, camId, timestampMs }),

  /** v9: pull the current YOLO inference state on mount so the badge resolves
   *  even if the one-shot `inference:status` event fired before CameraView
   *  attached its listener. */
  getInferenceStatus: () =>
    invoke<InferenceStatus>("get_inference_status"),

  getStorageInfo: () => invoke<StorageInfo>("get_storage_info"),
  clearNvrRecordings: () => invoke<number>("clear_nvr_recordings"),
  /** Delete all footage (NVR segments + motion events/clips) whose start time
   *  falls in [from, to] (RFC3339 UTC). DB-driven + lock-safe. */
  deleteFootageInRange: (from: string, to: string) =>
    invoke<{ segments_deleted: number; events_deleted: number }>("delete_footage_in_range", { from, to }),
  discoverCameras: () => invoke<DiscoveredCamera[]>("discover_cameras"),

  // v11: minimal real call — returns local port + token only. QR / public
  // IP / IPv6 are gone with the remote-access cut, but CameraView still
  // needs port + token to build local snapshot / NVR URLs.
  getStreamInfo: () => invoke<StreamInfo>("get_stream_info"),
  // Latest frame for a camera as base64 JPEG (over IPC). Used by face-enrollment
  // Live capture when the USB cam is owned server-side (dshow) so getUserMedia can't.
  getCameraSnapshot: (camId?: number) => invoke<string | null>("get_camera_snapshot", { camId: camId ?? 0 }),

  getSettings: () => invoke<Settings>("get_settings"),
  saveSettings: (settings: Settings) => invoke<void>("save_settings", { settings }),

  getMotionEvents: (limit?: number) =>
    invoke<MotionEvent[]>("get_motion_events", { limit: limit ?? 50 }),
  readClipFrames: (eventId: string) =>
    invoke<string[]>("read_clip_frames", { eventId }),
  saveClipBlob: (eventId: string, blobB64: string, mimeType: string) =>
    invoke<void>("save_clip_blob", { eventId, blobB64, mimeType }),
  startNvr: (camId: number, sourceUrl?: string) =>
    invoke<{ mode: string; segment_mins: number }>("start_nvr", { camId, sourceUrl: sourceUrl ?? "" }),
  stopNvr: (camId: number) => invoke<void>("stop_nvr", { camId }),
  saveNvrSegment: (camId: number, filename: string, blobB64: string, mimeType: string) =>
    invoke<void>("save_nvr_segment", { camId, filename, blobB64, mimeType }),
  getNvrSegments: (camId?: number, limit?: number) =>
    invoke<{ id: string; cam_id: number; path: string; started_at: string; ended_at: string | null; size_bytes: number }[]>("get_nvr_segments", { camId: camId ?? null, limit: limit ?? 100 }),
  startRtspRelay: (camId: number, url: string) =>
    invoke<void>("start_rtsp_relay", { camId, url }),
  stopRtspRelay: (camId: number) => invoke<void>("stop_rtsp_relay", { camId }),
  /** Test a camera URL with ffprobe before saving — codec/res/fps/audio or a precise error. */
  probeStream: (url: string, transport?: "tcp" | "udp") =>
    invoke<StreamProbe>("probe_stream", { url, transport: transport ?? null }),
  getActiveCameras: () => invoke<number[]>("get_active_cameras"),
  getCameraTelemetry: (camId: number) => invoke<CameraTelemetry>("get_camera_telemetry", { camId }),
  getCameraConfigs: () => invoke<CameraConfig[]>("get_camera_configs"),
  setCameraConfig: (config: CameraConfig) => invoke<void>("set_camera_config", { config }),
  listNvrRecordings: (camId?: number, from?: string, to?: string) =>
    invoke<{ filename: string; cam_id: number; started_at: string; size_bytes: number; duration_secs: number; path: string }[]>("list_nvr_recordings", { camId: camId ?? null, from: from ?? null, to: to ?? null }),
  /** Local days (YYYY-MM-DD) that hold recorded footage — marks them in the picker
   *  so finding footage isn't guesswork. */
  listRecordedDays: (camId?: number, from?: string, to?: string) =>
    invoke<string[]>("list_recorded_days", { camId, from, to }),
  getEventsInRange: (rangeStart: string, rangeEnd: string) =>
    invoke<MotionEvent[]>("get_events_in_range", { rangeStart, rangeEnd }),
  // Lightweight, COMPLETE event list for the NVR timeline (no thumbnails, no cap)
  // — the single source the timeline reads so it never silently drops events.
  getEventMarkers: (rangeStart: string, rangeEnd: string) =>
    invoke<EventMarker[]>("get_event_markers", { rangeStart, rangeEnd }),
  // Whole-archive keyword search over event metadata (Review search box),
  // blended with Jina-CLIP semantic results when the skill is installed.
  searchEvents: (query: string, limit?: number) =>
    invoke<MotionEvent[]>("search_events", { query, limit: limit ?? null }),
  // Image→image "Find similar" — needs an active search model; returns [] without it.
  findSimilarEvents: (eventId: string, limit?: number) =>
    invoke<MotionEvent[]>("find_similar_events", { eventId, limit: limit ?? null }),
  // Backfill embeddings for the active search model (after install / model switch).
  reindexSemanticSearch: () => invoke<number>("reindex_semantic_search"),
  // Stream a 16 kHz mono PCM window from a USB/webcam mic for YAMNet audio detection.
  analyzeAudioWindow: (camId: number, pcm: number[]) =>
    invoke<void>("analyze_audio_window", { camId, pcm }),
  // standard event lifecycle: ordered "what happened" entries for one event.
  getEventTimeline: (eventId: string) =>
    invoke<TimelineEntry[]>("get_event_timeline", { eventId }),

  // ── Review segments (server-side review items, NVR parity) ──
  /** Low-res scrub previews covering a range (one per camera per hour). */
  listPreviews: (camId: number, rangeStart: string, rangeEnd: string) =>
    invoke<Preview[]>("list_previews", { camId, rangeStart, rangeEnd }),
  getReviewSegments: (rangeStart: string, rangeEnd: string) =>
    invoke<ReviewSegment[]>("get_review_segments", { rangeStart, rangeEnd }),
  setReviewSegmentReviewed: (id: string, reviewed: boolean) =>
    invoke<void>("set_review_segment_reviewed", { id, reviewed }),

  // ── Event bookmarks (saved/favorites pattern) ──
  addBookmark: (eventId: string) => invoke<void>("add_bookmark", { eventId }),
  removeBookmark: (eventId: string) => invoke<void>("remove_bookmark", { eventId }),
  listBookmarkIds: () => invoke<string[]>("list_bookmark_ids"),
  listBookmarkedEvents: () => invoke<MotionEvent[]>("list_bookmarked_events"),
  deleteMotionEvent: (id: string) => invoke<void>("delete_motion_event", { id }),
  /** Delete events + their clip files. A Review card is a GROUP, so pass every
   *  member id in ONE call — deleting one at a time re-aggregates the group in
   *  between and the card reappears until the last member goes. */
  deleteEvents: (ids: string[]) => invoke<void>("delete_events", { ids }),
  clearAllEvents: () => invoke<void>("clear_all_events"),
  purgeOrphanedClips: () => invoke<number>("purge_orphaned_clips"),
  storeDetections: (eventId: string, detections: string) =>
    invoke<void>("store_detections", { eventId, detections }),
  keepAliveEvent: (camId = 0) =>
    invoke<void>("keep_alive_event", { camId }),

  // Structured memory files (OpenClaw-style)
  // Guardian agent
  getAgentAlerts: (limit?: number) =>
    invoke<AgentAlert[]>("get_agent_alerts", { limit: limit ?? 50 }),
  deleteAgentAlert: (id: string) => invoke<void>("delete_agent_alert", { id }),
  clearAllAgentAlerts: () => invoke<void>("clear_all_agent_alerts"),
  getAgentMemory: (key: string) =>
    invoke<string>("get_agent_memory", { key }),
  setAgentMemory: (key: string, value: string) =>
    invoke<void>("set_agent_memory", { key, value }),
  listAgentMemory: () =>
    invoke<[string, string, string][]>("list_agent_memory"),
  deleteAgentMemory: (key: string) =>
    invoke<void>("delete_agent_memory", { key }),
  exploreEvents: (filter: string) =>
    invoke<Array<{id:string;ts:string;started_at:string;duration?:string;summary?:string;risk_level:string;threat_type:string;has_clip:boolean;thumbnail?:string|null}>>("explore_events", { filter }),
  searchClips: (query: string) =>
    invoke<Array<{id:string;ts:string;started_at:string;duration?:string;summary?:string;risk_level:string;threat_type:string;has_clip:boolean;thumbnail?:string|null}>>("search_clips", { query }),
  listAlertConditions: () =>
    invoke<Array<{id:string;name:string;condition:string;channels:string;min_risk:string;enabled:boolean;trigger_count:number;created_at:string}>>("list_alert_conditions"),
  createAlertCondition: (name:string, condition:string, channels:string, min_risk:string) =>
    invoke<{id:string;name:string;condition:string;channels:string;min_risk:string;enabled:boolean;trigger_count:number;created_at:string}>("create_alert_condition", {name,condition,channels,minRisk:min_risk}),
  deleteAlertCondition: (id: string) => invoke<void>("delete_alert_condition", { id }),
  toggleAlertCondition: (id: string, enabled: boolean) => invoke<void>("toggle_alert_condition", { id, enabled }),
  readMemoryFile: (category: string) => invoke<string>("read_memory_file", { category }),
  writeMemoryFile: (category: string, content: string) => invoke<void>("write_memory_file", { category, content }),
  readAllMemoryFiles: () => invoke<string>("read_all_memory_files"),
  getAgentStatus: () =>
    invoke<AgentStatus>("get_agent_status"),
  triggerAgentNow: () =>
    invoke<void>("trigger_agent_now"),
  /** In-app Guardian chat: THE surface. Same brain and same resolved evidence
   *  as Telegram — event cards, snapshots, people, share links. */
  chatApp: (history: { role: string; content: string }[], message: string, mode?: string) =>
    invoke<ChatAppReply>("chat_app", { history, message, mode: mode ?? "ask" }),
  /** Durable conversation restore (oldest→newest) — the chat survives restarts. */
  getChatLog: (limit?: number) =>
    invoke<{ role: string; content: string; created_at: string; events?: EventCard[] }[]>("get_chat_log", { limit: limit ?? null }),
  /** Erase the durable conversation. Memory, events and footage are untouched. */
  clearChatLog: () => invoke<void>("clear_chat_log"),
  // on-device assistants-inspired intelligence
  queryEvents: (question: string, history?: { role: string; content: string }[]) =>
    invoke<string>("query_events", { question, history: history ?? null }),
  reportCrowdCount: (camId: number, count: number, eventId: string | null) =>
    invoke<void>("report_crowd_count", { camId, count, eventId: eventId ?? null }),
  analyzeSnapshot: (imageB64: string, detections: { label: string; score: number }[], sceneContext?: string) =>
    invoke<string>("analyze_snapshot", { imageB64, detections, sceneContext: sceneContext ?? null }),

  // Known persons / face recognition
  enrollPerson: (name: string, role: string, embedding: number[], thumbnail: string | null) =>
    invoke<KnownPerson>("enroll_person", { name, role, embedding, thumbnail }),
  // Guided multi-angle enrollment (ArcFace). `embedFace` returns one captured
  // angle's 512-d embedding + quality/area/crop; `enrollPersonMulti` saves them all.
  embedFace: (jpegB64: string) =>
    invoke<FaceCapture | null>("embed_face", { jpegB64 }),
  enrollPersonMulti: (name: string, role: string, embeddings: number[][], thumbnail: string | null) =>
    invoke<KnownPerson>("enroll_person_multi", { name, role, embeddings, thumbnail }),
  // Live face recognition via the unified ArcFace model (replaces face-api).
  recognizeFrame: (jpegB64: string) =>
    invoke<RecognizedFace[]>("recognize_frame", { jpegB64 }),
  // Diagnostic: does the face pipeline actually load? Resolves + try-loads the
  // active tier. Rejects with a clear reason (no model / load error).
  facePipelineStatus: () =>
    invoke<string>("face_pipeline_status"),
  // Per-stage face diagnostic (installed/loaded per model + detect/embed counts
  // on an optional frame). Drives the Enroll status line + bundled-models dots.
  faceDebug: (jpegB64?: string) =>
    invoke<FaceDebug>("face_debug", { jpegB64: jpegB64 ?? null }),
  // Hybrid matching head: is the trained classifier active, and who does it cover?
  faceClassifierStatus: () =>
    invoke<FaceClassifierStatus>("face_classifier_status"),
  // Force a retrain of the classifier (Roster "Retrain" affordance).
  retrainFaceClassifier: () =>
    invoke<FaceClassifierStatus>("retrain_face_classifier"),
  addPersonEmbedding: (id: string, embedding: number[]) =>
    invoke<void>("add_person_embedding", { id, embedding }),
  listKnownPersons: () =>
    invoke<KnownPerson[]>("list_known_persons"),
  // Unlink: the LABEL was wrong. Face crops return to the training pool so they
  // can be re-tagged. This is not erasure — see forgetPerson.
  deletePerson: (id: string) =>
    invoke<void>("delete_person", { id }),
  // Erase: the PERSON must be forgotten. Destroys every face descriptor, crop,
  // body-appearance vector and sighting, and retrains the classifier without
  // them. This is the data-subject deletion path (see PRIVACY.md); it does not
  // touch recorded footage, which retention governs. Returns rows destroyed.
  forgetPerson: (id: string) =>
    invoke<number>("forget_person", { id }),
  // Rename a person and/or change their role — sightings/history/stats follow
  // the new name (no more delete + re-enroll to fix a typo).
  renamePerson: (id: string, name: string, role?: string) =>
    invoke<void>("rename_person", { id, name, role: role ?? null }),
  markPersonSeen: (id: string) =>
    invoke<void>("mark_person_seen", { id }),

  // standard: face sightings the agent saw but couldn't identify.
  // Use these in the "Train" tab to tag unknowns into known persons.
  listRecentUnknownFaces: (limit = 60, days = 14, minQuality = 0.20) =>
    invoke<UnknownFace[]>("list_recent_unknown_faces", { limit, days, minQuality }),
  assignFaceToPerson: (faceId: string, personId: string) =>
    invoke<void>("assign_face_to_person", { faceId, personId }),
  // Distinct-individual clustering: group repeat unknowns so a whole person can
  // be named at once.
  listUnknownClusters: (days?: number, minQuality?: number) =>
    invoke<UnknownCluster[]>("list_unknown_clusters", { days: days ?? null, minQuality: minQuality ?? null }),
  getPersonSightings: (faceIds: string[]) =>
    invoke<PersonSighting[]>("get_person_sightings", { faceIds }),
  /** standard person-events: every motion event a person appears in.
   *  Identity = enrolled personId OR an unknown cluster's faceIds. Events ship
   *  the '@thumb' marker (render via eventThumbSrc); person_crop = the face
   *  crop of this person in that event (the mature NVRs object-crop preview). */
  getPersonEvents: (opts: { personId?: string; faceIds?: string[]; days?: number; limit?: number }) =>
    invoke<PersonEvent[]>("get_person_events", {
      personId: opts.personId ?? null, faceIds: opts.faceIds ?? null,
      days: opts.days ?? null, limit: opts.limit ?? null,
    }),
  listVehicles: (days?: number) =>
    invoke<Vehicle[]>("list_vehicles", { days: days ?? null }),
  /** Multi-value filters (vtype/color/plate/cams/category) are CSV strings. */
  listVehicleEvents: (opts?: { days?: number; before?: string; limit?: number; vtype?: string; color?: string; plate?: string; cams?: string; from?: string; to?: string }) =>
    invoke<VehicleEvent[]>("list_vehicle_events", {
      days: opts?.days, before: opts?.before, limit: opts?.limit,
      vtype: opts?.vtype, color: opts?.color, plate: opts?.plate,
      cams: opts?.cams, from: opts?.from, to: opts?.to,
    }),
  listAudioEvents: (opts?: { days?: number; before?: string; limit?: number; category?: string; cams?: string; from?: string; to?: string }) =>
    invoke<AudioEvent[]>("list_audio_events", {
      days: opts?.days, before: opts?.before, limit: opts?.limit, category: opts?.category,
      cams: opts?.cams, from: opts?.from, to: opts?.to,
    }),
  getPersonStats: () =>
    invoke<PersonStats[]>("get_person_stats"),
  getAudioStats: () =>
    invoke<AudioStats[]>("get_audio_stats"),
  assignFacesToPerson: (faceIds: string[], personId: string) =>
    invoke<void>("assign_faces_to_person", { faceIds, personId }),
  createPersonFromFace: (faceId: string, name: string, role: string) =>
    invoke<KnownPerson>("create_person_from_face", { faceId, name, role }),
  // Correction: fix a WRONG recognition — reassign an already-labeled face to the
  // right person (or `null` = "not them"). Records a hard negative + retrains.
  correctFace: (faceId: string, correctPersonId: string | null) =>
    invoke<void>("correct_face", { faceId, correctPersonId }),
  // Per-person face-crop gallery (detail view) + curation.
  listPersonFaces: (personId: string, limit?: number) =>
    invoke<FaceShot[]>("list_person_faces", { personId, limit: limit ?? null }),
  deleteFaceEmbedding: (id: string) =>
    invoke<void>("delete_face_embedding", { id }),
  clearUnknownFaces: () => invoke<number>("clear_unknown_faces"),
  getFaceContext: (faceId: string) => invoke<string | null>("get_face_context", { faceId }),
  // Mature NVRs "Recent Recognitions" — recent matches of enrolled people.
  listRecentRecognitions: (limit?: number, days?: number) =>
    invoke<Recognition[]>("list_recent_recognitions", { limit: limit ?? null, days: days ?? null }),
  // Body Re-ID cross-camera tracked persons (appearance-based, soft signal).
  listTrackedPersons: () =>
    invoke<TrackedPerson[]>("list_tracked_persons"),
  // Which Re-ID backbone is active ("Deep (OSNet-AIN)" / "Color histogram" …).
  reidBackendStatus: () =>
    invoke<string>("reid_backend_status"),
  // "Train" a tracked body: bind anonymous body group(s) to an enrolled person.
  assignTrackedToKnown: (bodyPersonIds: string[], knownId: string) =>
    invoke<void>("assign_tracked_to_known", { bodyPersonIds, knownId }),
  // Self-grouping: clusters of anonymous body tracks that look like the same person.
  listTrackedClusters: (days?: number) =>
    invoke<TrackedCluster[]>("list_tracked_clusters", { days: days ?? null }),
  // Name a body group with no enrolled face → a "soft" identity (no face yet).
  nameTrackedGroup: (bodyPersonIds: string[], name: string, role: string) =>
    invoke<string>("name_tracked_group", { bodyPersonIds, name, role }),
  // Correction: detach mis-grouped tracks back to anonymous so they re-cluster.
  unnameTrackedGroup: (bodyPersonIds: string[]) =>
    invoke<void>("unname_tracked_group", { bodyPersonIds }),
  // DURABLE correction: record a hard negative for the wrong person (so matching/
  // clustering won't re-attribute this appearance), then reassign to the correct person
  // or detach. wrongKnownId / correctKnownId are optional (either or both).
  correctTrack: (bodyPersonId: string, wrongKnownId: string | null, correctKnownId: string | null) =>
    invoke<void>("correct_track", { bodyPersonId, wrongKnownId, correctKnownId }),

  getLocalIp: () => invoke<string>("get_local_ip"),

  // Camera remote control
  // v11: no-op shim. Pre-v11 this notified the libp2p / WebRTC viewers
  // when the desktop's camera state changed; with remote viewing gone the
  // backend command is deleted, but call sites still fire it.
  notifyCameraState: (_active: boolean) => Promise.resolve(),

  // Revoke current token and generate a new one (disconnects all phone sessions)
  revokeToken: () => invoke<void>("revoke_token"),

  // Disconnect a single remote viewer by session ID
  disconnectClient: (id: string) => invoke<void>("disconnect_client", { id }),

  // Camera inventory
  getCameraInventory: () => invoke<CameraInventory>("get_camera_inventory"),
  reportBrowserCameras: (cameras: BrowserCameraInfo[]) =>
    invoke<void>("report_browser_cameras", { cameras }),

  // Scene object reporting for Guardian tracking
  updateSceneObjects: (camId: number, objects: { label: string; score: number }[]) =>
    invoke<void>("update_scene_objects", { camId, objects }),

  // Behavior event reporting from person tracker
  reportBehaviorEvents: (camId: number, events: { track_id: number; name: string | null; flags: string[]; duration_secs: number }[]) =>
    invoke<void>("report_behavior_events", { camId, events }),

  // Alert feedback
  setAlertFeedback: (id: string, feedback: string) => invoke<void>("set_alert_feedback", { id, feedback }),

  // Native Rust camera capture
  listNativeCameras: () => invoke<NativeCameraDevice[]>("list_native_cameras"),
  startNativeCamera: (camId: number, deviceIndex: number) =>
    invoke<void>("start_native_camera", { camId, deviceIndex }),
  // ffmpeg DirectShow USB capture (reliable Windows backend — records 24/7 server-side).
  listDshowCameras: () => invoke<string[]>("list_dshow_cameras"),
  startDshowCamera: (camId: number, deviceName: string) =>
    invoke<void>("start_dshow_camera", { camId, deviceName }),
  stopNativeCamera: (camId: number) =>
    invoke<void>("stop_native_camera", { camId }),
  stopAllNativeCameras: () => invoke<void>("stop_all_native_cameras"),

  // Notification channel tests
  sendTelegramTest:  (botToken: string, chatId: string) =>
    invoke<string>("send_telegram_test", { botToken, chatId }),
  /** One-tap connect: validate the token + auto-detect the chat ID. */
  telegramConnect:   (botToken: string) =>
    invoke<{ bot_username: string; chat_id: string; chat_name: string; needs_message: boolean;
             }>(
      "telegram_connect", { botToken }),
  /** Tailscale Funnel remote-access status (compliant live/clip sharing). */
  tailscaleStatus:   () => invoke<TailscaleStatus>("tailscale_status"),
  /** Turn on Funnel for the stream port; may return a one-time enable_url. */
  tailscaleEnable:   () => invoke<TailscaleStatus>("tailscale_enable"),

  // GPU picker
  listGpus: () => invoke<GpuInfo[]>("list_gpus"),
  setPreferredGpu: (gpuName: string) => invoke<void>("set_preferred_gpu", { gpuName }),
  /** Live host utilization (CPU/RAM/GPU) for the System Monitor. */
  getSystemMetrics: () => invoke<SystemMetrics>("get_system_metrics"),
  trtxStatus: () => invoke<TrtxStatus>("trtx_status"),
  /** Per-execution-provider state + the concrete reason each one is or isn't in
   *  use. Every row is probe-derived (adapters, pack contents, the provider
   *  DLL's own linked runtime, canary results) — never a hardcoded assumption. */
  accelReport: () => invoke<AccelRow[]>("accel_report"),
  installTrtxPack: () => invoke<TrtxStatus>("install_trtx_pack"),
  /** Import a user-downloaded TensorRT-for-RTX SDK (zip/folder path). */
  importTrtxSdk: (path: string) => invoke<TrtxStatus>("import_trtx_sdk", { path }),
  benchmarkInference: (seconds?: number) => invoke<InferStatRow[]>("benchmark_inference", { seconds }),
  /** Measured GB/day vs the disk cap — projected real retention for Storage settings. */
  nvrDiskProjection: () => invoke<DiskProjection>("nvr_disk_projection"),

  // Agent tools
  searchSimilarEvents: (threatType: string, timeOfDay?: string, limit?: number) =>
    invoke<{ event_id: string; started_at: string; ai_summary: string | null; peak_score: number }[]>("search_similar_events", { threatType, timeOfDay: timeOfDay ?? null, limit: limit ?? 10 }),
  triggerAlarm: () => invoke<void>("trigger_alarm"),

  // Multi-camera + anomaly
  recordFaceSighting: (personName: string, cameraId: number, eventId: string | null, confidence: number) =>
    invoke<void>("record_face_sighting", { personName, cameraId, eventId, confidence }),
  getCameraCorrelations: (sinceHours?: number) =>
    invoke<{ person_name: string; sightings: { camera_id: number; seen_at: string; confidence: number; event_id: string | null }[] }[]>("get_camera_correlations", { sinceHours: sinceHours ?? 24 }),
  detectAnomalies: (sinceHours?: number) =>
    invoke<{ event_id: string; started_at: string; anomaly_type: string; duration_secs: number; peak_score: number; detail: string }[]>("detect_anomalies", { sinceHours: sinceHours ?? 24 }),

  // Tunnel — v11: just lifecycle. QR + pair flow + Tailscale + libp2p went
  // with the remote-access pivot. start_tunnel/stop_tunnel still drive the
  // on-demand share infrastructure.

  // v11 share links
  generateShareLink: (kind: "live" | "clip", resourceId: string, expiryMins: number) =>
    invoke<ShareLinkResult>("generate_share_link", { kind, resourceId, expiryMins }),
  revokeAllShares:  () => invoke<void>("revoke_all_shares"),
  listActiveShares: () => invoke<ShareEntry[]>("list_active_shares"),
  setAuthPassword: (password: string) => invoke<void>("set_auth_password", { password }),
  // ── Desktop login gate (Argon2id + Telegram recovery/2FA) ──────────────────
  authStatus: () => invoke<AuthStatus>("auth_status"),
  setLoginPassword: (newPassword: string, currentPassword?: string) =>
    invoke<void>("set_login_password", { newPassword, currentPassword: currentPassword ?? null }),
  setLoginRequired: (enabled: boolean) => invoke<void>("set_login_required", { enabled }),
  set2faEnabled: (enabled: boolean) => invoke<void>("set_2fa_enabled", { enabled }),
  setRememberDevice: (enabled: boolean, days: number) => invoke<void>("set_remember_device", { enabled, days }),
  login: (password: string, remember: boolean) => invoke<LoginResult>("login", { password, remember }),
  loginVerifyOtp: (challenge: string, code: string, remember: boolean) =>
    invoke<LoginResult>("login_verify_otp", { challenge, code, remember }),
  authResume: (rememberToken: string) => invoke<boolean>("auth_resume", { rememberToken }),
  lockApp: () => invoke<void>("lock"),
  logout: (rememberToken?: string) => invoke<void>("logout", { rememberToken: rememberToken ?? null }),
  requestRecovery: () => invoke<string>("request_recovery"),
  recoveryReset: (challenge: string, code: string, newPassword: string, remember: boolean) =>
    invoke<LoginResult>("recovery_reset", { challenge, code, newPassword, remember }),
  probeMjpegUrl: (baseUrl: string, user?: string, pass?: string) =>
    invoke<string>("probe_mjpeg_url", { baseUrl, user: user ?? null, pass: pass ?? null }),
  checkForUpdate: (repo: string) => invoke<any>("check_for_update", { repo }),

  stopDirectP2P:  () => invoke<void>("stop_direct_p2p"),
  getDirectP2PStatus: () => invoke<any>("get_direct_p2p_status"),


  // AI Provider
  // `provider` (optional) lists/tests a provider the user is BROWSING without
  // committing it as the active engine; omit to use the saved provider.
  listProviderModels: (provider?: string) => invoke<Array<{id:string;name:string;category:string}>>("list_provider_models", { provider }),
  testAiProvider: (provider?: string) => invoke<{ok:boolean;models?:string[];count?:number;error?:string}>("test_ai_provider", { provider }),

  // Hardware encoder
  getHwEncoder: () => invoke<string>("get_hw_encoder"),

  // Windows Firewall — opens port 8880 via UAC-elevated PowerShell
  fixFirewall: () => invoke<string>("fix_firewall"),

  // ONVIF camera management
  discoverOnvif: (timeoutMs?: number) =>
    invoke<Array<{ device_url: string; xaddrs: string; source_ip: string }>>("discover_onvif", { timeoutMs: timeoutMs ?? null }),
  getOnvifStreams: (deviceUrl: string, username?: string, password?: string) =>
    invoke<Array<{ profile_token: string; rtsp_url: string; device_url: string }>>("get_onvif_streams", { deviceUrl, username: username ?? null, password: password ?? null }),
  getOnvifDeviceInfo: (deviceUrl: string, username?: string, password?: string) =>
    invoke<{ manufacturer: string; model: string; firmware_version: string; serial_number: string }>("get_onvif_device_info", { deviceUrl, username: username ?? null, password: password ?? null }),
  discoverAndConfigureOnvif: (username?: string, password?: string, timeoutMs?: number) =>
    invoke<Array<{ profile_token: string; rtsp_url: string; manufacturer: string; model: string; source_ip: string }>>("discover_and_configure_onvif", { username: username ?? null, password: password ?? null, timeoutMs: timeoutMs ?? null }),

  // Skills — user-downloadable AI plugins
  /** `minBytes` demands a size floor — pass a skill's `minBytes` so a model
   *  superseded by a larger one at the same path reads as needing an update
   *  rather than as installed. */
  checkSkillInstalled: (skillId: string, minBytes?: number) =>
    invoke<boolean>("check_skill_installed", { skillId, minBytes }),
  /**
   * Stream a skill download with live progress. Subscribes to the backend's
   * `skill:progress` Tauri events (filtered by `skill_id`), forwards each
   * tick to the caller's `onProgress(percent, downloaded_bytes, total_bytes)`,
   * and tears the listener down on completion. `total` may be `null` for
   * HuggingFace LFS files where `content-length` isn't present — the UI
   * should render bytes-downloaded in that case.
   */
  downloadSkill: async (
    skillId: string,
    url: string,
    onProgress?: (pct: number, downloaded?: number, total?: number | null) => void,
    filename?: string,
  ) => {
    const unlisten = await listen<{
      skill_id:    string;
      percent:     number;
      downloaded?: number;
      total?:      number | null;
    }>("skill:progress", ({ payload }) => {
      if (payload.skill_id !== skillId) return;
      onProgress?.(payload.percent, payload.downloaded, payload.total ?? null);
    });
    try {
      await invoke<void>("download_skill", { skillId, url, filename });
    } finally {
      unlisten();
    }
  },
  removeSkill: (skillId: string) => invoke<void>("remove_skill", { skillId }),

  /** Face tier suggestion for this host — still used by the Settings "Auto"
   *  button. The wider llmfit-style recommendation engine was removed. */
  recommendFaceModel: () => invoke<FaceRecommendation>("recommend_face_model"),
  listInstalledSkills: () => invoke<SkillStatus[]>("list_installed_skills"),
};

export interface HostSpecs {
  total_ram_gb:     number;
  available_ram_gb: number;
  cpu_name:         string;
  cpu_cores:        number;
  gpu_name:         string | null;
  gpu_vram_gb:      number | null;
  gpu_backend:      string;
  unified_memory:   boolean;
}

export interface FaceRecommendation {
  tier:   "off" | "small" | "large";
  reason: string;
  host:   HostSpecs;
}

export interface SkillStatus {
  id:               string;
  name:             string;
  installed:        boolean;
  size_on_disk_mb:  number;
}

// Lightweight event marker for the NVR timeline (from get_event_markers) — no
// thumbnail/ai_summary, so the whole day loads with no cap. A strict subset of
// MotionEvent, so it widens to one for the timeline components.
export interface EventMarker {
  id:              string;
  started_at:      string;
  ended_at:        string | null;
  duration_secs:   number | null;
  first_object_at: string | null;
  cam_id?:         number;
  event_category?: string | null;
  peak_score:      number;
  /** Thumbnail exists server-side (fetch via /footage/:id/thumbnail). */
  has_thumb:       boolean;
  /** A standalone cached clip exists — playable even after raw footage prunes. */
  has_clip:        boolean;
}

// Server-side review item (mature NVRs `ReviewSegment` parity) — groups overlapping
// motion_events into ONE reviewable unit. Columns + flattened ReviewSegmentData.
/** One scrub-preview file. Media time maps as `(t - start_time) / 1000`. */
export interface Preview {
  id:         string;
  cam_id:     number;
  start_time: string;
  end_time:   string;
}

export interface ReviewSegment {
  id:            string;
  cam_id:        number;
  start_time:    string;
  end_time:      string | null;
  severity:      "alert" | "detection";
  thumbnail:     string | null;
  reviewed:      boolean;
  member_ids:    string[];
  labels:        string[];
  categories:    string[];
  sub_label:     string | null;
  plate:         string | null;
  zones:         string[];
  peak:          number;
  audio:         string | null;
  fall:          string | null;
  crossing:      string | null;
  speed:         string | null;
  summary:       string | null;
  clip_event_id: string | null;
}


export interface UnknownFace {
  id:             string;
  thumbnail_b64:  string;     // 112x112 JPEG, base64
  quality:        number;     // 0..1, higher = sharper
  cam_id:         number;
  event_id:       string | null;
  seen_at:        string;     // ISO-ish
  suggested_name?:  string | null;   // closest enrolled person ("looks like X")
  suggested_score?: number | null;   // its cosine (near-miss band)
  /** The suggested person's ID — bind confirm actions to THIS, never the name. */
  suggested_person_id?: string | null;
}

// Body Re-ID tracked person (cross-camera, appearance-based).
export interface TrackedPerson {
  person_id:      string;
  label:          string;     // "Person N" (stable, by first-seen order)
  sighting_count: number;
  cameras:        number[];
  first_seen:     string;
  last_seen:      string;
  thumbnail:      string | null;  // base64 JPEG of the person crop (first sighting), if any
  known_name:     string | null;  // resolved name when auto-labelled from a face (face↔body fusion)
  /** The known person's ID — corrections bind to THIS, never the name. */
  known_person_id: string | null;
  suggested_name: string | null;  // "looks like X" — closest enrolled person by body appearance
  /** The suggested person's ID — confirm actions bind to THIS, never the name. */
  suggested_person_id: string | null;
  /** Majority-voted clothing line, e.g. "blue top · black bottom". */
  outfit:         string | null;
}

// A proposed group of anonymous body tracks that look like the same person
// (self-grouping → batch-train). Member ids are body_* person_ids.
export interface TrackedCluster {
  cluster_id:      string;
  member_ids:      string[];
  track_count:     number;
  sighting_count:  number;
  cameras:         number[];
  last_seen:       string;
  /** '@crop' marker — render via bodyCropSrc(rep_thumbnail, rep_track_id, streamInfo). */
  rep_thumbnail:   string | null;
  /** Track id whose crop represents this cluster (the crop URL key). */
  rep_track_id:    string | null;
  suggested_name:  string | null;   // "looks like X"
  suggested_score: number | null;
  /** The suggested person's ID — confirm actions bind to THIS, never the name. */
  suggested_person_id: string | null;
  samples:         TrackSample[];    // one crop per member track (expandable, deselectable)
}

export interface TrackSample {
  person_id: string;   // the body_* track id — deselect to drop it from the group
  thumbnail: string;
  sightings: number;
  last_seen: string;
}

// One stored face crop for a known person (per-person detail gallery).
export interface FaceShot {
  id:             string;
  thumbnail_b64:  string;
  quality:        number;
  cam_id:         number;
  seen_at:        string;
}

// One captured face angle during guided enrollment (ArcFace embedding + meta).
export interface FaceCapture {
  embedding:     number[];   // 512-d ArcFace
  quality:       number;     // 0..1 sharpness
  area:          number;     // face bbox area in source px
  bbox:          [number, number, number, number];
  thumbnail_b64: string;
}

// Per-stage face-pipeline diagnostic (from face_debug).
export interface FaceDebug {
  tier:               string;   // "face_small" | "face_large" | "none"
  detector_installed: boolean;
  embedder_installed: boolean;
  detector_loaded:    boolean;
  embedder_loaded:    boolean;
  img_w:              number;
  img_h:              number;
  raw_faces:          number;
  max_conf:           number;
  embedded_faces:     number;
  enrolled_persons:      number;
  enrolled_dim_mismatch: number;   // enrolled people whose embeddings don't match the active model
  note:               string;
}

// Desktop login gate status.
export interface AuthStatus {
  login_required:   boolean;
  unlocked:         boolean;
  has_password:     boolean;
  twofa_enabled:    boolean;
  telegram_ready:   boolean;
  remember_enabled: boolean;
  remember_days:    number;
}

// Result of login / OTP / recovery — discriminated by `status`.
export type LoginResult =
  | { status: "Unlocked"; remember_token: string | null }
  | { status: "Needs2fa"; challenge: string };

// Hybrid face matching head status (from face_classifier_status / retrain).
export interface FaceClassifierStatus {
  active:           boolean;   // trained classifier in use ("smart match") vs cosine-only
  trained_people:   number;    // distinct people the classifier discriminates
  has_reject_class: boolean;   // a synthetic "unknown" reject class was trained in
  trained_at:       string | null;
  person_ids:       string[];  // known-person ids the classifier covers ("Trained ✓" chip)
}

// One recognised face in a live frame (for the overlay).
export interface RecognizedFace {
  person_id: string;
  name:      string;         // "unknown" when below the recognition threshold
  score:     number;
  bbox:      [number, number, number, number];
}

// A cluster of unrecognised faces that look like the same person.
export interface UnknownCluster {
  cluster_id:    string;
  /** '@crop' marker — render via faceCropSrc(rep_thumbnail, rep_id, streamInfo). */
  rep_thumbnail: string;
  /** Face id of the representative member (the crop URL key). */
  rep_id:        string;
  count:         number;
  cameras:       number[];
  last_seen:     string;
  first_seen:    string;             // earliest sighting of this recurring stranger
  days_active:   number;             // distinct days seen — the "regular" signal
  time_pattern:  string;             // "Evenings" | "Mornings" | "Overnight" | "Any time" …
  face_ids:      string[];
  suggested_name?:  string | null;   // closest enrolled person ("looks like X")
  suggested_score?: number | null;
  /** The suggested person's ID — bind confirm actions to THIS, never the name. */
  suggested_person_id?: string | null;
  samples?: FaceSample[];            // ≤9 member faces (id+crop); expand each to its full frame
}

export interface VehicleEvent {
  id:            string;
  vtype:         string;          // car / truck / bus / motorcycle / bicycle
  cam_id:        number | null;
  started_at:    string;
  ended_at:      string | null;
  duration_secs: number | null;
  score:         number;
  thumbnail:     string | null;   // '@thumb' marker → /footage/:id/thumbnail
  plate:         string | null;
  plate_score:   number | null;
  owner:         string | null;   // known-plate friendly name
  color:         string | null;   // HSV-voted body color
  speed_kmh:     number | null;
}

export interface Vehicle {
  plate:      string;
  name:       string | null;   // friendly name from known_plates
  count:      number;
  cameras:    number[];
  first_seen: string;
  last_seen:  string;
  thumbnail:  string;
  event_ids:  string[];
}

export interface AudioEvent {
  id:            string;
  sound:         string;          // the recognised sound (dog bark, speech, alarm…)
  cam_id:        number | null;
  started_at:    string;
  ended_at:      string | null;
  duration_secs: number | null;
  score:         number;          // detection confidence 0..1
  ai_summary:    string | null;
  thumbnail:     string | null;   // '@thumb' marker → /footage/:id/thumbnail
  loudness_db:   number | null;   // dBFS of the loudest window
  high_pitch:    boolean;         // YAMNet class in the scream/alarm/glass register
  classes:       { l: string; s: number }[]; // top YAMNet classes (label + confidence)
}

export interface PersonStats {
  /** The identity this row belongs to. Null only for pre-migration sightings
   *  whose name matches no roster entry. Join on THIS, never on `name`. */
  person_id:     string | null;
  name:          string;
  sightings_30d: number;
  days_active:   number;
  peak_hour:     number | null;  // local hour-of-day (0-23) most often seen
  /** Sightings per local hour, 0..23. The backend always computed this and shipped
   *  only its argmax, so "home every evening" and "here once at 18:00" looked
   *  identical to the UI. */
  hours:         number[];
  cameras:       number[];
  last_seen:     string | null;
}

export interface AudioStats {
  sound:      string;
  count_7d:   number;
  last_heard: string;
  peak_hour:  number | null;
}

export interface PersonSighting {
  event_id:    string;
  cam_id:      number;      // last-seen location
  seen_at:     string;
  thumbnail:   string;      // bare base64 event thumbnail
  ai_summary?: string | null;
}

/** One video event a person appears in (standard person-events).
 *  `event` is a standard MotionEvent (thumbnail = '@thumb' marker — render via
 *  eventThumbSrc); `person_crop` = this person's face crop in that event. */
export interface PersonEvent {
  event:       MotionEvent;
  person_crop: string | null;
}

export interface FaceSample {
  id:        string;
  thumbnail: string;   // base64 JPEG face crop
}

// A recent recognition of an enrolled person (mature NVRs "Recent Recognitions").
export interface Recognition {
  id:             string;   // face_embeddings.id — for correcting a wrong match
  person_id:      string;
  name:           string;
  role:           string;
  thumbnail_b64:  string;
  quality:        number;
  cam_id:         number;
  seen_at:        string;
  /** Naming provenance — which recognizer, its score, runner-up margin, source
   *  event. Null on rows stored before traceability shipped. */
  match_method:   string | null;
  match_score:    number | null;
  match_margin:   number | null;
  event_id:       string | null;
}
