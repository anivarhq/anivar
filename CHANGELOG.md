# Changelog

All notable changes to Anivar are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project (will) adhere to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Privacy: anonymised cameras stopped leaking raw frames

- **Fixed: `get_camera_snapshot` ignored depth anonymisation.** Every other
  consumer of a camera frame checks `depth::is_anonymized` first — `dshow.rs`,
  `face.rs`, `go2rtc.rs`, `motion_lifecycle.rs` — and this one did not, so a
  camera the user had explicitly marked anonymised still handed back a
  recognisable JPEG. It now anonymises before returning, and **fails closed**:
  if the depth pass cannot run, it returns nothing rather than the raw frame.
- **Removed the browser-side depth worker.** It was doing anonymisation in the
  wrong place — pulling raw frames into the WebView to blur them there, which
  means the un-anonymised image had already left the server. `CameraView` now
  renders the server-anonymised image. This also deleted the
  `@huggingface/transformers` → `sharp` dependency chain and, with it, **five
  high-severity advisories**: production `npm audit` is now clean at `high`.

### Biometric data is documented, and can actually be erased

- **Added `PRIVACY.md`.** Face descriptors and body-appearance vectors are
  special-category personal data under GDPR Article 9 and equivalent laws, and
  nothing in the repository said so. It now states what is stored and where, what
  leaves the machine and only when you enable it, and where operator obligations
  begin — which is the moment a camera sees past your own household.
- **Added `forget_person` — a real erasure path.** `delete_person` answers "this
  label was wrong" by returning the crops to the training pool. That is the right
  behaviour for a mistake and the wrong one for a deletion request, because the
  descriptor still identifies the person. Erasure destroys the face descriptors,
  crops, body vectors, hard negatives and sightings, clears the name from past
  events, and retrains the classifier without them. Recorded footage is
  deliberately untouched — retention governs that.
- Regression test `erasure_leaves_nothing_behind` asserts every table is empty
  afterwards *and* that a second person survives. It immediately caught a bug in
  its own subject: `face_sightings` has no `person_id` column, so one DELETE
  matched nothing while the UI reported success.
- Documented honestly rather than papered over: **the SQLite database is not
  encrypted.** Secrets are AES-GCM at rest, the rest is plaintext, and full-disk
  encryption is the answer until that changes.

### Shipping standards

- **Added `SECURITY.md`** — private vulnerability reporting, scope, and response
  times. A security product had no disclosure policy, so the only route for a
  reporter was a public issue.
- **`.github/ISSUE_TEMPLATE/config.yml` routed vulnerabilities into public
  issues**, and pointed at `github.com/__OWNER__/...` — a placeholder that had
  never been filled in, so the one contact link was dead. Both fixed; the private
  advisory form is now the first option.
- **Added `CODE_OF_CONDUCT.md`**, including two rules specific to this project:
  do not post other people's faces or plates in an issue, and do not publish a
  working exploit for an unfixed bug.
- **Release artifacts now publish SHA-256 checksums.** Auto-updates were already
  signed, but a manual installer download had nothing to verify against, and
  Authenticode signing is still blocked on a certificate.
- **CI gained a supply-chain job** (`cargo audit` + `npm audit`, failing on high
  severity in production dependencies) and finally **runs the frontend tests** —
  vitest had been in `package.json` the whole time and CI never invoked it.

### On-device language model replaces the Ollama daemon

- **Removed Ollama entirely** — daemon *and* provider. The app no longer
  downloads, spawns, or watchdogs a language-model server. Triage that prompted
  it: a 6 GB orphaned `llama-server` that `ollama ps` did not even list, on a
  16 GB machine with 1.9 GB free.
- **Added `agent/local_llm.rs`** — llama.cpp compiled into the binary, running
  LFM2.5-350M (Q4_K_M, ~230 MB) in-process. One worker thread owns the model
  (`LlamaModel` is not `Send`), loads on demand, and drops it after 180 s idle,
  so it costs nothing at rest. Measured ~390 MB while loaded, ~0.4 s per reply
  on CPU.
- Provider id is `"local"` and it is the default; settings **migration v3**
  flips saved `"ollama"` over, and **v4** clears the now-meaningless
  `vision_model` tag it left behind.
- Sampling follows the model authors' published settings — temperature 0.1,
  top_k 50, repetition penalty 1.05. The previous greedy sampler had **no**
  repetition penalty, which is how a small model ends up in a degenerate repeat
  loop.
- Prefill is **chunked to `n_batch`**. Submitting a whole prompt in one
  `decode()` trips `GGML_ASSERT(n_tokens_all <= cparams.n_batch)`, which is an
  `abort()` — the process died instantly, taking recording with it, for any
  prompt over 512 tokens. Regression test uses a production-sized prompt.

### Capability is asked, not inferred

- Added `provider_supports_vision()` and `provider_can_classify_risk()`. Nothing
  may decide what an engine can do by checking whether `vision_model` is
  non-empty — that string outlives the engine that could use it.
- A text-only provider handed a vision job used to **invent scenes** (echoing the
  prompt's own schema placeholders into `ai_summary`). Those paths now degrade to
  detection-only summaries built from real detections, or refuse with an
  actionable message.

### Removed

- **Cloudflare tunnels** — remote access is Tailscale Funnel only.
- **Light theme** — the UI is dark-only.
- Hardware-aware model recommendation from Arsenal.

### Fixed

- Events left **open by an unclean shutdown** are now closed at startup. Stale
  `ended_at IS NULL` rows permanently occupied the live-analysis queue, so the
  same zombies were re-analysed on every launch while new events never got in.
- The on-device model can be **removed** from Arsenal. It was missing from the
  installed-skills inventory, which is the only surface that renders a Remove
  button. Removal now releases the model's mmap first — Windows cannot delete a
  memory-mapped file.
- Provider/model pairs could **cross-contaminate** in Arsenal: a stale test
  result from the previously-viewed tab could populate another provider's model
  list, committing an on-device model under LM Studio.
- Partial downloads can no longer read as installed; multi-file skills install
  all-or-nothing.
- CI fetched a `ffmpeg` asset that upstream had removed.

### Build

- `scripts/app-build.bat` is the supported way to build a shippable binary. A
  bare `cargo build --release` exits 0 but does **not** embed the frontend,
  producing an app that opens "localhost refused to connect" while the backend
  looks healthy. Size gate: ~64 MB good, ~50 MB broken.
- `scripts/cargo-env.bat` applies the three llama.cpp prerequisites (VS
  developer environment, Ninja, libclang).
- The ONNX Runtime provider DLLs are no longer committed (one is 92 MB).
  `scripts/sync-ort-dlls.ps1` stages them from the cargo build output, so their
  version always matches the linked ORT.

### Backend module structure (major refactor)

- **Agent split**: the 6 390-line `src-tauri/src/agent.rs` was replaced by a
  13-file `agent/` module — `types`, `memory`, `llm`, `prompts`, `ha`,
  `dispatch`, `analysis`, `chat`, `cycle`, `conditions`, `clip`, `util`, plus
  the now-60-line `mod.rs` as the public re-export manifest. External callers
  (`lib.rs`, etc.) keep referring to `crate::agent::Foo` unchanged.
- **lib.rs split**: the 8 887-line `lib.rs` was carved into 30 sibling
  modules + the central [`state`] module (which owns `Settings`,
  `AppState`, `StreamState`, and all per-camera / per-event types):
  - Utility helpers: `crypto`, `hw`, `qr`, `ip`, `ffmpeg`, `mqtt`, `cloudflare`,
    `onvif`, `motion`, `reid`, `inference`, `pwa`, `db`, `pairing`, `hls`,
    `footage`.
  - Tauri command groupings: `auto_reid`, `rtsp`, `cam_config`, `persons`,
    `correlation`, `direct_p2p_cmds`, `ai_provider`, `ha_cmds`, `agent_cmds`,
    `agent_data_cmds`, `agent_tools`, `nvr_recording`.
  - Transport handlers: `capture` (native nokhwa capture loop), `webrtc`
    (signalling room), `websocket` (bidirectional MJPEG WS), `server`
    (HTTP boot — auth middleware, security headers, CORS, axum Router),
    `http_handlers` (MJPEG, cam-proxy, discovery, login, snapshot, PWA
    manifest, service worker).
  - Tauri command groupings: `hw_onvif` (HW encoder, firewall, ONVIF,
    skills), `system_cmds` (GPU picker, session, LAN discovery),
    `frontend_cmds` (scene objects, clip blob I/O, agent chat),
    `remote_cmds` (Tailscale, P2P), `native_cam_cmds` (start/stop
    devices, inventory), `events_cmds` (settings, motion events,
    storage), `tunnel_cmds` (Cloudflare tunnel, pairing QR, stream
    info), `search_cmds` (global search, GitHub update check),
    `inference_cmds` (frame ingest pipeline + motion event state
    machine).
  - `boot` — extracted the `.setup(|app| …)` closure body into
    `boot::setup_app`: system-tray menu, DB pool, settings load + decrypt,
    AppState construction, agent / inference / NVR / mDNS / HTTP-server
    task spawns.
  - `lib.rs` is now ~295 lines — module declarations,
    `constant_time_eq`, and the `run()` Tauri builder + the
    `tauri::generate_handler!` registration. **97 % reduction from
    8 887.**
- `docs/ARCHITECTURE.md` updated to reflect both module trees.

### Project structure

- Repository initialised for public release: added `README.md`, `CHANGELOG.md`,
  `CONTRIBUTING.md`, `docs/ARCHITECTURE.md`, GitHub issue / PR templates and
  CI workflow stub.
- Moved `build-release.bat` / `launch-dev.bat` into `scripts/`; added
  cross-platform `.sh` equivalents.
- Removed stray files from the project root (`null`, ad-hoc fix scripts,
  `tsconfig.tsbuildinfo`).
- Extracted shared TypeScript types into `src/types/index.ts`; `src/api/`
  re-exports them so existing imports continue to work.

### Motion detection

- Switched motion detection to a standard algorithm: per-pixel grayscale
  diff with a 3×3 box blur applied to both frames first, so JPEG / sensor
  noise no longer drives `motion_score` over the sensitivity threshold.
- Polygon masks are now applied to the diff (not the input frames), so masked
  pixels can never contribute to motion regardless of when the mask was drawn.

### Mask editor

- Fixed the `MaskEditor` coordinate bug where polygon points were normalised
  against the editor's container size instead of the displayed image's
  content size. With `object-fit: contain`, this caused mask polygons to be
  shifted relative to the analysed frame whenever the container aspect ratio
  didn't match the camera aspect ratio.

### Object detection (YOLO26)

- ONNX decoder now reads the actual output tensor shape at runtime and
  supports both `[1, 84, 8400]` and `[1, 8400, 84]` exports.
- Bounding-box pixel coords are rescaled from the 640×640 model input to the
  original camera resolution.
- Threshold lowered to `0.20` with greedy NMS (IoU>0.5, capped at 50 boxes
  per frame).
- YOLO26 detections falling inside motion-mask polygons are dropped — masks
  now genuinely exclude their region from both motion *and* object detection.

### Agent / VLM analysis

- New `extract_json_block` helper strips ` ```json ` markdown fences before
  parsing the VLM response. Previously, models like Qwen2.5-VL and LLaVA
  emitted fence-wrapped JSON that our parser couldn't read, silently dropping
  the actual analysis.
- Risk levels migrated to the four-level Agies scale (`normal` / `monitor` /
  `suspicious` / `critical`) with backward-compatible normalisation for old
  events still stored under `low` / `medium` / `high`.

[Unreleased]: ./
