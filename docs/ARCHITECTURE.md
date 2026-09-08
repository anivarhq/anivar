# Architecture

This document describes how Anivar is put together. It is aimed at new
contributors who need to find their way around the codebase quickly.

## At a glance

Anivar is a desktop NVR + agentic AI security camera application. It is
delivered as a single binary via [Tauri 2](https://v2.tauri.app/):

```
┌───────────────────────────────────────────────────────────────────┐
│ Tauri shell (window, system tray, autostart, notifications)       │
│ ┌──────────────────────────┐  ┌─────────────────────────────────┐ │
│ │ Frontend (React + Vite)  │  │ Backend (Rust)                  │ │
│ │ ─ src/                   │◀▶│ ─ src-tauri/src/                │ │
│ │   features/*             │  │   lib.rs    (commands, wiring)  │ │
│ │   components/, store/    │  │   agent/    (AI: analysis+chat) │ │
│ │   workers/               │  │   nvr_*.rs  (record, HLS, keep) │ │
│ │                          │  │   local_llm (llama.cpp, in-proc)│ │
│ └──────────────────────────┘  └─────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────┘
                ▲                         ▲
                │ ffmpeg capture:         │ ONNX Runtime (detect, face,
                │ USB / RTSP / MJPEG      │   ALPR, CLIP, audio)
                │ + go2rtc WebRTC         │ llama.cpp in-process, or a
                ▼                         ▼  cloud/self-hosted provider
        ┌────────────┐            ┌─────────────────────┐
        │  Cameras   │            │  Models             │
        └────────────┘            └─────────────────────┘
```

The language model runs **inside this process** by default (`agent/local_llm.rs`);
cloud and self-hosted providers are optional and sit behind the same dispatcher.

## Backend (Rust) — `src-tauri/src/`

| Path | Lines (approx.) | Responsibility |
|---|---|---|
| `main.rs` | 6 | Tauri bootstrap — calls `nivar_lib::run()`. |
| `lib.rs` | ~295 | Module manifest, `constant_time_eq`, and the `run()` entry — Tauri builder, plugins, tray window-event, `.setup(boot::setup_app)`, and the `tauri::generate_handler!` registration. Down from 8 887 (97 % reduction). |
| `boot.rs` | ~255 | One-time `.setup(|app| …)` work extracted: system-tray menu, DB pool, settings load + decrypt, AppState construction, agent / inference / NVR / mDNS / HTTP-server task spawns. |
| `server.rs` | ~250 | HTTP server boot: token auth middleware, security headers, CORS, axum `Router` wiring, port-firewall opening, listener spawn. |
| `http_handlers.rs` | ~315 | Axum handlers: MJPEG stream, camera proxy, discovery info, password login, snapshot, mobile-viewer HTML, manifest, service worker. |
| `capture.rs` | ~425 | Native (nokhwa) camera capture — JPEG encode, RGB→gray helpers, per-camera background capture loop (motion detection, clip writer). |
| `websocket.rs` | ~260 | Bidirectional MJPEG WebSocket — desktop publisher pushes frames, mobile subscribers receive them; per-IP rate limiting, session bookkeeping, kick-by-id. |
| `state.rs` | ~530 | Central shared state: `Settings`, `AppState`, `StreamState`, `PerCamState`, `SignalRoom`, `ClientSession`, `MotionEvent`, `FrameResult`, `StreamInfo`, `CameraInventory`, all per-camera and per-event types. |
| `agent/` | ~7 000 across 16 files | Agentic AI layer — split into focused modules (see below). |
| **Utility / focused sibling modules of `lib.rs`** | | |
| `crypto.rs` | ~100 | AES-256-GCM symmetric encryption of secret settings fields. |
| `hw.rs` | ~65 | Hardware H.264 encoder detection (NVENC / QSV / AMF / VideoToolbox / V4L2). |
| `ffmpeg.rs` | ~125 | Resolve the ffmpeg binary — bundled copy, system copy, or provisioned download (memoized). |
| `onvif.rs` | ~60 | ONVIF SOAP helpers (digest auth, envelope construction). |
| `motion.rs` | ~170 | standard motion detection: grayscale diff with box blur + per-pixel mask buffer. |
| `reid.rs` | ~135 | Body Re-ID via HSV histogram appearance descriptor. |
| `inference.rs` | ~430 | ONNX Runtime inference loop with execution-provider selection (TensorRT / CUDA / DirectML / CPU, CoreML on macOS), decoder, mask filtering, NMS, per-model latency telemetry. |
| `db.rs` | ~295 | SQLite schema migrations, pool setup, settings round-trip, token storage. |
| `hls.rs` | ~45 | HLS playlist + segment file server. |
| `footage.rs` | ~230 | axum HTTP handlers for recorded-footage download endpoints and ping. |
| `nvr_pipes.rs` | ~360 | Rust-side ffmpeg pipes for continuous NVR + HLS encoding. |
| `nvr_stream.rs` | ~295 | Axum HTTP handlers for streaming the recorded NVR archive (Range, seek, concat). |
| **Tauri command grouping modules** | | |
| `rtsp.rs` | ~135 | RTSP relay commands (Feature 7). |
| `cam_config.rs` | ~65 | Multi-camera config (get/set/active). |
| `persons.rs` | ~90 | Known persons / face enrolment commands. |
| `correlation.rs` | ~140 | Multi-camera correlation + anomaly detection commands. |
| `ai_provider.rs` | ~100 | Multi-provider LLM management commands. |
| `agent_cmds.rs` | ~270 | Guardian agent Tauri commands (alerts, memory, models, chat). |
| `agent_data_cmds.rs` | ~215 | assistant-parity Tauri commands (event search, alert conditions, memory files). |
| `agent_tools.rs` | ~215 | Agent tool commands (person history, alarm trigger, channel tests). |
| `nvr_recording.rs` | ~305 | Browser-driven NVR continuous recording commands. |
| `hw_onvif.rs` | ~270 | Hardware encoder query, Windows Firewall fix, ONVIF discovery / profiles / device-info. (Skill install/remove moved to `skills.rs`.) |
| `system_cmds.rs` | ~235 | GPU picker (`list_gpus`, `set_preferred_gpu`), accelerator report, auth-token revoke, viewer kick, local IP, LAN camera discovery. |
| `frontend_cmds.rs` | ~125 | JS-frontend-facing commands: scene-object reporting, clip blob/frame round-trips, Guardian streaming chat. |
| `native_cam_cmds.rs` | ~170 | Native camera management: list/start/stop devices, browser-camera reporting, active-camera query. |
| `events_cmds.rs` | ~125 | Settings I/O, storage stats, motion-event listing/keep-alive, detection / AI-summary persistence. |
| `tunnel_cmds.rs` | ~220 | Stream info + tunnel status. The cloudflared lifecycle it used to own was removed with Cloudflare; remote access is Tailscale-only. |
| `search_cmds.rs` | ~165 | Global search across events / persons / alerts / recordings + GitHub-release update check. |
| `inference_cmds.rs` | ~235 | Frame ingest pipeline: fast-path `stream_frame`, `process_frame` motion + event state machine, shared `process_frame_inner` used by RTSP/NVR. |

| **Provisioning, models & accelerators** | | |
| `provision.rs` | ~450 | **The single contract for every managed download**: pooled client, retry/backoff, `.part`+rename, SHA-256 verification, atomic install, zip unpack, and `Requirement` — the one definition of "is it installed". Replaced eight hand-rolled downloaders. |
| `skills.rs` | ~280 | Skill (model) install / remove / installed-scan. Knows every weight format we ship, `.onnx` and `.gguf` alike. |
| `trtx_runtime.rs` | ~580 | TensorRT / TensorRT-RTX pack provisioning, plus a PE import-table reader that refuses to install a runtime the linked provider cannot load. |
| `cuda_runtime.rs` / `openvino_runtime.rs` | ~300 each | CUDA and OpenVINO runtime packs; loader-path setup per platform. |
| `face.rs` | ~1 420 | Face detection + ArcFace recognition, track-consensus naming, galleries, enrolment, self-learning corrections. |
| `face_classifier.rs` | ~200 | Hybrid classifier head over the embeddings, confirmed by cosine. |
| `alpr.rs` | ~330 | Plate detection + OCR, region models, known-plate matching. |
| `embed.rs` | ~320 | CLIP image/text encoders for semantic search (object-crop embeddings). |
| `vector_index.rs` | ~200 | usearch HNSW index over the embeddings, with a brute-force fallback. |
| `audio.rs` | ~200 | YAMNet AudioSet classification with a sustained-detection gate. |
| `depth.rs` | ~265 | Depth-map anonymisation, enforced server-side. |
| **Recording, review & platform** | | |
| `nvr_vod.rs` | ~330 | HLS VOD for recorded playback — server-side seeking over the archive. |
| `review_segments.rs` | ~315 | Review segments: grouping, severity bands, reviewed state, bookmarks. |
| `jobs.rs` | ~135 | Durable job queue (SQLite-backed) for embedding + backfill work. |
| `blobstore.rs` | ~160 | Display crops on disk instead of in SQLite (`@file:` refs). |
| `tailscale.rs` | ~275 | Tailscale Funnel remote access — the only remote-access path. |
| `auth.rs` | ~185 | Optional login gate: Argon2id, recovery, OTP. |
| `logfile.rs` | ~135 | Size-capped rolling application log (8 MB + one backup). |
| `proc.rs` | ~60 | Windows Job Object so spawned ffmpeg dies with the app — no zombie recorders. |

### Agent module layout — `src-tauri/src/agent/`

The agent code used to live in a single 6 390-line `agent.rs`. It has been
fully split into focused submodules. Each submodule is self-contained —
its public surface is re-exported from `agent/mod.rs` so external callers
(mostly `lib.rs`) keep using `crate::agent::Foo` paths unchanged.

| File | Lines | Owns |
|---|---|---|
| `agent/mod.rs` | ~60 | Module manifest — declarations + re-exports only. Top-level docs. |
| `agent/types.rs` | ~340 | Public output types (`AgentAlert`, `AgentStatus`, `EscalationState`), provider wire types, `ClipAnalysis` / `Analysis` schema, `make_summary_json`, `agent_identity`, rule-based fallback. |
| `agent/memory.rs` | ~700 | `agent_memory` reads/writes: flat KV, ACE-loop scored learned memory, OpenClaw narrative markdown files, event_subscribe alert rules, analysis-output helpers (`extract_summary_text`, `sanitize_analysis_output`), and `recall` — ranked, budgeted retrieval for the prompt. |
| `agent/llm.rs` | ~455 | Multi-provider routing (on-device / OpenAI / Anthropic / Groq / xAI / Gemini / OpenAI-compatible). `call_llm`, `call_llm_tools`, and the capability predicates `provider_supports_vision` / `provider_can_classify_risk`. |
| `agent/local_llm.rs` | ~505 | The on-device engine: llama.cpp compiled in, one worker thread owning the model, load-on-demand, idle unload, chunked prefill, and the GGUF's own chat template. |
| `agent/prompts.rs` | ~110 | Guardian system prompts for clip analysis. |
| `agent/dispatch.rs` | ~1 200 | Alert dispatch over Telegram, gated centrally by `channel_alert_allowed`. Risk → emoji/priority helpers. Telegram long-poll loop and `/menu` browser. |
| `agent/analysis.rs` | ~645 | Core `analyze_event` flow — builds prompt + context, calls the VLM, parses + sanitises the response (`extract_json_block`), applies YOLO grounding, stores v2 `ai_summary`, triggers dispatch. Risk normalisation helpers. |
| `agent/chat.rs` | ~450 | Interactive chat (`chat_with_agent`) + streaming chat (`stream_chat`, `ChatMessage`). |
| `agent/cycle.rs` | ~420 | Background loops: agent cycle, escalation timer, reflection, main loop. |
| `agent/conditions.rs` | ~430 | Semantic alert conditions — plain-English rules evaluated by the LLM. `is_quiet_hours`, `evaluate_alert_conditions`, event-search commands (`query_events_nl`, `explore_events`, `search_clips`). |
| `agent/clip.rs` | ~1 160 | Post-recording clip analysis (`analyze_event_clip`), Agies-style frame annotation, shared YOLO26 detection types, live-event monitoring loop, startup backfill, `dispatch_intelligence_alert`. |
| `agent/util.rs` | ~610 | Status helper, one-shot snapshot analysis (Test button), disk-space guard, cross-camera narrative context, heartbeat loop, proactive insights, situational awareness push. |

Visibility convention: items used by external callers (mostly `lib.rs`) are
`pub`; items shared across agent submodules but kept internal to the agent
are `pub(super)`.

### Key subsystems inside `lib.rs`

- **`AppState`** — the central `Arc<...>` shared between every Tauri command,
  worker, and background task. Holds the SQLite pool, settings, per-camera
  state, broadcast channels for live frames, the ONNX session, and more.
- **HTTP server (axum)** — binds on `127.0.0.1:<port>`. Serves MJPEG streams,
  WebRTC signalling, snapshots, recorded clip downloads, and an HTML viewer.
  All routes require a token from `AppState.auth_token`.
- **Motion detection** (`compute_motion_masked`) — standard grayscale
  frame-diff. Both `prev` and `curr` frames are box-blurred to suppress JPEG /
  sensor noise; the diff is then computed and per-pixel mask polygons drop
  masked regions from the score. See [Motion masking](MOTION.md).
- **NVR pipeline** — continuous segment recording (default 1 min per segment)
  with retention bounded by `nvr_max_gb`. Motion events store *pointers* into
  the NVR timeline rather than separate clip files (mature NVRs pattern).
- **YOLO26 inference** (`run_inference_loop`) — Rust-native ONNX Runtime
  session. Loads `<data_dir>/skills/yolo26/model.onnx` when the YOLO26 skill is
  installed. Decoder handles both `[1, 84, 8400]` and `[1, 8400, 84]` ONNX
  layouts. Output detections feed `scene_objects` and per-event detection
  buffers.

### Agent loop (`agent/`)

- **`analyze_event_clip`** — fired when a motion event closes. Extracts
  stroboscopic frames from the recorded clip, annotates them with YOLO26
  bounding boxes, sends them to the configured provider — or takes the
  rule-based path when the active engine cannot see a frame or cannot be trusted
  to classify risk — parses the JSON response (`extract_json_block`
  strips ` ```json ` fences first), and stores a v2-format `ai_summary`.
- **`chat_with_agent`** — the user-facing AI chat. Assembles context (standing
  rules, the live situation, recent events and alerts, retrieved memory, recent
  turns), calls the LLM, and parses embedded action tags (`[SNAPSHOT]`,
  `[SUBSCRIBE_ALERT:…]`, `[SEARCH_EVENTS:…]`, `[REMEMBER:…]`, etc).

  Two properties of that assembly are load-bearing:

  - **The prompt is budgeted; stored memory is not.** Memory is *retrieved*
    (`memory::recall` — keyword relevance × weekly-halving recency × confirmation
    weight, capped at `MEMORY_BUDGET_CHARS`), never dumped. History is capped at
    `HISTORY_BUDGET_CHARS` by dropping **whole turns**, which stay recoverable
    because `recall` also searches `chat_log`. Before this, every fact the agent
    learned was re-sent on every turn, and the prompt eventually outgrew what
    llama.cpp could batch — which aborted the process. `read_core_memory` still
    exists, but only as the full export/debug view.
    `local_llm`'s middle-truncation is now a last-resort guard that logs at
    `error!`: reaching it means the budgeting is broken.
  - **The agent knows the date.** Every system prompt carries the current *local*
    date, time and UTC offset, and states what "today" means. "Today" resolves to
    `date(started_at,'localtime') = date('now','localtime')` in both
    `build_situation_ctx` and `conditions.rs` — one definition — and every event,
    alert and recalled fact is rendered with its own date, so the agent can always
    say which day it is reporting on.
- **Dispatch** — when an alert fires, `dispatch_with_clip` fans out to every
  channel configured in settings: Telegram, with an inline keyboard.
  Every send passes through `channel_alert_allowed`, so a
  muted category or quiet hours cannot be bypassed by adding a new call site.

## Frontend (React + TypeScript) — `src/`

```
src/
├── api/            # Thin wrappers over Tauri commands
├── components/     # Generic UI primitives (re-usable across features)
├── features/       # Feature-scoped panels (see below)
├── lib/            # Pure helpers / shared logic
├── store/          # Zustand stores
├── workers/        # Web Workers (e.g. depth estimation)
└── App.tsx         # Top-level layout + tab routing
```

### `features/` — one folder per panel

| Folder | What it owns |
|---|---|
| `agent/` | AI agent chat panel + alerts inbox. |
| `camera/` | The single-camera live view (`CameraView.tsx`). |
| `cameras/` | Multi-camera grid, camera-onboarding flow, **mask editor** (`MaskEditor.tsx`). |
| `events/` | Motion events list + detail panel. |
| `live/` | The "Live" tab (current view + active recordings). |
| `nvr/` | The NVR scrubber timeline + segmented playback. |
| `persons/` | Face / person enrollment and known-person management. |
| `recordings/` | (placeholder — to be merged into NVR.) |
| `review/` | Event review queue (mark FP / acknowledge). |
| `search/` | Global search across events. |
| `settings/` | All settings UI. |
| `stream/` | Remote streaming / sharing UI. |

Each feature folder is self-contained — its CSS module, components, and helpers
live next to its panel. Cross-feature dependencies go through `src/api/`,
`src/store/`, or `src/lib/`.

## Mobile / remote viewing

There is no separate mobile app. Phone access is served by the desktop app
itself: `http_handlers.rs` serves a mobile viewer and `websocket.rs` streams
frames to it. Reach it over the LAN, a Tailscale Funnel share link, or a
Telegram live link — nothing to install on the phone.

The libp2p/DHT stack and the QR pairing flow that used to live here were
removed; Tailscale Funnel (`tailscale.rs`) is the only remote-access path.

## Build flow

| Mode | Entry | What runs |
|---|---|---|
| Dev | `npm run tauri dev` | Vite serves the React app, Cargo builds the Rust crate with `cargo run`, Tauri opens a window pointing at Vite's dev server. Hot reload works on both sides. |
| Release | `npm run tauri build` | Vite builds static assets to `dist/`, Cargo builds the release binary with LTO, Tauri produces a platform-native installer in `src-tauri/target/release/bundle/`. |

Convenience wrappers live in [`scripts/`](../scripts/).

## Settings, data & secrets

Per-user data is written to Tauri's `app_data_dir()`:

- Windows: `%APPDATA%\com.nivar.app\`
- macOS: `~/Library/Application Support/com.nivar.app/`
- Linux: `~/.local/share/com.nivar.app/`

This directory contains the SQLite database (`nivar.db`), NVR segment
files (`.mp4`), recorded clips, installed skills (e.g. `skills/yolo26n/model.onnx`
for the detector, `skills/local_llm/model.gguf` for the on-device language model),
and a `.master_key` used to AES-GCM-encrypt secret fields in settings
(Telegram tokens, API keys). **Nothing in the source tree should ever
contain real secrets.**

## Roadmap

- [x] ~~Split `agent.rs` (6 390 lines) into focused submodules.~~ Done — 13
      submodules now live under `agent/`.
- [x] ~~Split `lib.rs` utility sections out.~~ Done — 14 utility/helper
      modules plus 12 Tauri-command grouping modules.
- [x] ~~Extract shared state into `state.rs`.~~ Done — `Settings`, `AppState`,
      `StreamState`, `PerCamState`, `SignalRoom`, `ClientSession`, `MotionEvent`,
      `FrameResult`, `StreamInfo`, `CameraInventory` now live in a single
      ~530-line `state.rs`.
- [x] ~~Carve transport handlers out of `lib.rs`.~~ Done — `capture`,
      `webrtc`, `websocket`, `server` (boot + auth + CORS), `http_handlers`.
- [x] ~~Carve Tauri command groupings out of `lib.rs`.~~ Done — `hw_onvif`,
      `system_cmds`, `frontend_cmds`, `remote_cmds`, `native_cam_cmds`,
      `events_cmds`, `tunnel_cmds`, `search_cmds`, `inference_cmds`.
- [x] ~~Extract the `.setup(|app| …)` closure body.~~ Done — `boot::setup_app`
      now owns tray creation, DB/state construction, and background-task
      spawns. `lib.rs` is down to ~295 lines: module manifest, one
      `constant_time_eq` helper, and the `run()` Tauri builder + the
      `tauri::generate_handler!` registration. **97 % reduction from
      8 887.** This is the practical floor without a different Tauri
      handler-registration mechanism.
- [ ] Add automated integration tests for the motion + NVR pipelines.
- [ ] Document the v2 `ai_summary` JSON schema in `docs/SCHEMAS.md`.
