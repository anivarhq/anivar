# Changelog

All notable changes to Anivar are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project (will) adhere to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.4] - 2026-09-24

### Security: a shared link could leave your server on the internet

- **Sharing a clip opened a public tunnel that nothing ever closed.** It
  survived reboots, and even "Revoke all" left it open, so one shared link
  could leave the whole server reachable from the internet indefinitely.
  Revoking all shares now closes it, and so does the last share link
  expiring. A tunnel you turned on yourself is left alone.
- **A forgotten sign-in page has been removed.** It skipped the rate limiter,
  stored its password without a salt, and handed back the master token.
  Nothing in the app could ever set that password.
- **Fifteen bad requests could lock you out of your own app.** The rate
  limiter ran before the token check and counted requests by a header any
  caller can set. The token is checked first now.
- **Camera passwords no longer appear in the logs**, and the site check that
  guards the local web address no longer accepts a lookalike domain.
- **Depth Anonymization is refused on network cameras** instead of being shown
  as on. Only USB cameras can record depth-only video; an RTSP camera kept
  recording ordinary video while the screen said otherwise.
- **The firewall no longer opens ports 4002-4003.** They belonged to a feature
  that was removed, and the old rule is deleted.

### Fixed

- **Exporting a video works again.** Every export failed because the app was
  not allowed to write the file.
- **Today's recorded footage plays past the moment you opened it.** It used to
  stop at the time the panel was opened and refuse to go further.
- **Footage recovered after a database reset has its sound again.** The
  re-indexer forgot to record that a segment had audio, so it played silent.
- **ONVIF cameras that ask for a password can now be added.** The sign-in used
  the wrong hash, so it could never succeed with any camera. Network scanning
  also missed cameras whose replies use a different XML style, and a password
  containing `@ : / #` corrupted the stream address.
- **Network scanning now also asks each address directly**, so it finds
  cameras that don't answer a broadcast.
- **The update check links the installer for the system you are on.**

### Changed: less work per frame, and fewer stalls

- **Motion detection decodes only the brightness of each frame** (5.1 ms down
  to 1.1 ms), and reads camera video in chunks rather than byte by byte.
- **Whether a recording has sound is read from the file itself** instead of
  starting ffmpeg for every ten seconds of video.
- **Every ffmpeg probe now has a time limit** and is stopped if the app moves
  on, so one stuck file cannot hold up indexing, playback or clip export.
- **Several screens stopped leaking work in the background:** live view
  releases its video connection when it falls back, the USB capture loop stops
  when its screen closes, a seek is cancelled when you seek again, and a
  five-hundred-row fetch that ran every two seconds into an unused field is
  gone.

## [0.1.3] - 2026-09-16

### Changed: the People section, rebuilt around visits

- **People now answers "who was here?"** It has three tabs. **Today** lists
  visits: one continuous stay per person, across cameras. **People** is the
  roster of people you've named. **Review** asks "Who is this?" once for each
  person your cameras keep seeing but can't name. Name them once and every
  sighting is theirs. When the app thinks it already knows someone, the card
  asks "Is this Ravi?" instead.
- **Search for people by description** in the search box: "blue top with a
  backpack", "unfamiliar, no hat", or a name. Results are grouped by person, and
  any visit can find others who look like that person.
- **Less for you to do.** People no longer asks you to confirm body-appearance
  matches or narrow face matches, and no longer shows match scores. Body
  appearance still works in the background, linking a visit across cameras and
  powering "find similar". To fix a wrong name, open that person and press
  **Not them** on the shot; recognition learns from it.
- **Face pictures show the face.** They were cut from the whole person, so
  someone seated with an arm stretched out appeared as mostly wall.
  Recognition never used that picture and is unchanged. Pictures saved before
  this version are not redrawn.

### Added: alerts about what people do

- New alerts for **zone entry** (someone steps into a zone set to "Alert when
  someone enters"), **climbing** (feet going over a line marked as a fence),
  **running** and **person down**. They follow each person over time instead of
  judging single frames. Loitering now measures how long one person stays.
- **Zone entry and climbing are on by default, including after this update.** If
  you already set zones to "Alert when someone enters", you'll now get those
  alerts; the setting existed before, but nothing sent them. Running and person
  down start off. All of them are in **Settings → Alerts**.
- By default these alerts skip people whose face was recognised. Person down is
  the exception, because a resident on the floor is exactly who it's for.
- Person down and climbing need the **Body pose** model below, and climbing
  also needs a fence line drawn on the camera.

### Added: optional models in Arsenal → Enhancements

- **Deep person Re-ID** (NVIDIA ReIdentificationNet), about 92 MB, downloaded
  from NVIDIA when you install it: links a visit across cameras and powers
  "find similar".
- **Body pose** (MoveNet, Apache-2.0), about 9 MB: person down and climbing.
- **Person attributes** (PaddleClas PP-LCNet, Apache-2.0, trained on PA-100K),
  about 7 MB: clothing, bags and accessories for search. It only judges someone
  whose whole body is in frame and standing. A camera that sees people from the
  chest up, like a desk camera, gives it nothing to judge, so those searches
  won't match there.

### Fixed

- The camera filter in People listed sixteen cameras no matter how many you
  have. It lists your enabled cameras now.
- Cards in People printed raw character codes such as `\u00b7` instead of the
  characters themselves.

## [0.1.2] - 2026-09-14

### Fixed: recorded footage played in ten-second blocks

- **Playback froze about ten seconds after every load or seek, and seeks landed
  short.** Recordings reach the player as ten-second pieces, and every piece
  restarted its timestamps at zero. The player repaired the video but laid each
  piece's audio over the first ten seconds, so it could never play past them.
  Meanwhile it downloaded the rest of the hour trying to catch up, and the app
  converted every piece of it. Each piece now carries its place in the
  timeline. The fault was reproduced in Edge against real recordings with the
  app's own player settings, and the fix was confirmed there and in the
  installed app.

### Changed: detection decodes each frame once

- The object detector decoded every camera frame for the model, then decoded the
  same image a second time only to read its width and height. It now takes both
  from the first decode: same pixels, same boxes, one full-resolution decode fewer
  per detected frame.

### Changed: the app icon is repainted

- Same geometry, two differences. The bar now sits beneath the letters instead of
  cutting them in half at the waist, so a letter can be traced through the middle
  of the mark. And the strokes are painted rather than drawn.

### Changed: the update address, and what the notes say about updating

- The built-in updater was set to ask a domain that no longer exists. It now
  points at the signed `latest.json` that every GitHub release publishes.
- Nothing in the app calls that updater yet, so installs still don't update
  themselves. To update, use **Settings → Check for Updates**: it shows the new
  version and links to its installer. Earlier release notes said installs update
  automatically. That was never true, and the notes now say how updating
  actually works.

### Added

- **Permanent download links.** Every installer is also published under a name
  with no version in it (`Anivar-windows-x64-setup.exe`, `Anivar-macos-arm64.dmg`,
  `Anivar-linux-x86_64.AppImage`, `Anivar-linux-amd64.deb`,
  `Anivar-linux-x86_64.rpm`), so a `releases/latest/download/…` link keeps
  pointing at the newest release.

### Security

- **rustls 0.23.45**
  ([RUSTSEC-2026-0285](https://rustsec.org/advisories/RUSTSEC-2026-0285.html)).
  The TLS library behind the app's HTTPS downloads and the updater accepted
  TLS 1.3 handshake messages sent at the wrong encryption level. Fixed upstream in
  0.23.45.

## [0.1.1] - 2026-09-09

First release cut after installing 0.1.0 from its own installer rather than
running a dev build. Four defects surfaced in the first ten minutes, all of them
first-run problems that no dev build could have shown. Every one was traced
against the installed app's log before any code changed.

### Fixed: the assistant was unreachable on a default install

- **The chat sat on "I'm resting" no matter what you installed.**
  `agent_enabled` defaults to `false`, and `provider_ready` short-circuits on it
  before it ever checks for a model — so a present, working 697 MB LLM reported
  as not ready. The only thing that sets it true is Arsenal's **Use** button, and
  that button could never appear: `ai_provider` and `local_llm_tier` both default
  to the on-device *Balanced* row, so it showed as "✓ In use" from first launch
  and went straight from **Install** to "in use", skipping **Use** entirely. The
  chat's own instruction — *"Pick a model in Arsenal → Model (Use)"* — pointed at
  a button that could not exist. "In use" now also requires the agent to be on,
  which additionally stops the row claiming it after Guardian is switched off.

### Fixed: the audio model could never be installed

- **`audio_yamnet` failed with "server returned only 14096 bytes — not a model
  file".** The server was fine. That file is the complete AudioSet class map, 522
  lines, exactly what `audio.rs` reads. The 64 KB weights floor was being applied
  to *every* file in a skill, and YAMNet is the only one with a small sidecar; the
  all-or-nothing rule then deleted the whole skill, including the 16 MB model that
  had downloaded correctly. The floor now applies to weights only.
- **Sidecars are checked by shape instead of size**, which is the stronger test:
  it rejects an HTML error page at *any* size, including one over 64 KB that the
  floor used to wave through to fail later as an opaque ONNX parse error.
- **Downloads now send a `User-Agent`.** `reqwest` sends none unless asked, and
  several CDNs answer a UA-less request with an interstitial — HTTP 200, matching
  content-length, an HTML body — which nothing upstream would have caught.

### Fixed: "Application is not responding" on first launch

- **Boot no longer downloads anything.** The CUDA runtime — roughly 1.8 GB — was
  fetched inside a `block_on` on the thread that owns the window. Tauri shows the
  window before `setup()` runs but does not pump its message loop until setup
  returns, so that download *was* the hang: 30 seconds of it, measured. The Linux
  lane already spawned the identical call. Boot now only puts an
  already-installed runtime on the library path, which is a directory listing.
- **The orphaned-process sweep moved off the startup path.** It waited on a cold
  PowerShell + WMI query, seconds on a fresh install, on every machine. It now
  runs inside the task that binds the stream port, preserving the ordering it
  depends on.

### Changed: one consent for the NVIDIA downloads, with the real number

- The TensorRT pack was offered as "~1.85 GB" while the installed directory
  measures **4.4 GB**, and the CUDA runtime beside it was never mentioned at all.
  Both now sit behind the single existing prompt, which states the true
  footprint. Decline it and detection still runs on DirectML.

### Changed: onboarding stops implying a choice you don't have

- The welcome screen showed three chips — core count, GPU name, accelerator.
  None were clickable, but side by side, two of them labelled CPU and GPU, they
  read as a decision to make. Replaced with one sentence naming what will
  actually run detection.

### Added

- **A wordmark for the app icon** — an A and a V sharing one stroke, crossed by
  the palette's single red as a scan line. Fluffy stays exactly where it belongs,
  as the Guardian's face in the chat; the launcher, taskbar and installer now
  carry the product's mark instead.
- Tests pinning the download guard: a 14 KB CSV passes, a 14 KB HTML body does
  not, a truncated `.onnx` does not.
- A data-migration test that had never run. A stray `#[test]` had attached to its
  neighbour, leaving 43 lines asserting every legacy data-directory and database
  name migrates correctly compiled as dead code.

### Known

- `cuda/` and `trt/` overlap by ~1.76 GB across 14 files. `cublasLt` is
  byte-identical in both; the cuDNN copies are different builds, so merging them
  means reconciling two pinned versions — tracked, not guessed at.

## [0.1.0] - 2026-09-08

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
