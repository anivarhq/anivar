# Anivar

A local-first NVR with on-device AI. One cross-platform desktop app built on
[Tauri 2](https://v2.tauri.app/) with a React frontend and a Rust backend —
recording, detection, recognition, and event analysis all run on your machine.
Nothing has to leave it, including the language model.

> **Status:** alpha. Not yet released. APIs and the data-directory layout are
> still subject to change.

## What it does

**Cameras and recording**

- **Multi-camera live view** — USB/built-in webcams, RTSP, MJPEG, and ONVIF
  discovery. Capture is server-side (one `ffmpeg` per camera, one clock), so the
  browser never touches the device.
- **Sub-second live streaming** — WebRTC via a bundled [go2rtc](https://github.com/AlexxIT/go2rtc),
  degrading to HLS then MJPEG.
- **24/7 recording** — segmented, faststart-remuxed `.mp4` with audio, bounded by
  a disk budget and per-tier retention. Detection runs on a hardware-decoded
  stream while the recording itself stays `-c copy`.
- **Recorded playback** — HLS VOD with **server-side** seeking, so scrubbing a
  multi-day archive doesn't depend on the browser buffering it.

**Detection and understanding**

- **Object detection** — ONNX Runtime in-process, with an execution-provider
  chain (TensorRT → CUDA → DirectML → CPU; CoreML on macOS). Selectable YOLO
  tiers from nano to xlarge. **No detector ships with the app** — you choose one
  on first run, with its licence shown.
- **Motion detection with masks** — grayscale frame-diff with box blur, polygon
  masks for noisy regions (fans, trees, monitors), hysteresis open/close, and a
  lightning guard for whole-frame flashes.
- **Faces and people** — ArcFace recognition with track-consensus naming (a name
  needs agreement across frames, not one lucky frame), person re-ID for
  cross-event linking, and self-learning corrections.
- **Licence plates** — ALPR with known-plate matching and recurring-plate
  proposals.
- **Audio events** — YAMNet AudioSet classification (glass, alarm, shouting…)
  with a sustained-detection model so one spike isn't an alert.
- **Semantic search** — natural-language search over the archive ("red shirt",
  "person with a package") using CLIP embeddings of the detected object crop,
  indexed with usearch HNSW.
- **Privacy mode** — optional per-camera depth-map anonymisation, enforced
  server-side: recordings and streams carry depth only, while the raw frame stays
  in memory for analysis.

**The assistant**

- **Evidence-first investigator** — ask "what happened last night?" and get
  playable clips, snapshots, and person cards back, not just prose. The
  conversation is durable and shared with the Telegram channel. It knows the
  date: every answer states which day or range it covers.
- **Memory that doesn't bloat the prompt** — what the agent learns is stored
  forever but *retrieved* per question (relevance × recency × how often it's been
  confirmed) against a fixed budget, so the context sent to the model stays the
  same size whether it has learned ten facts or ten thousand.
- **On-device language model** — llama.cpp is compiled **into** the binary and
  runs LFM2.5-1.2B-Instruct (~731 MB) in-process: no daemon, no port, nothing to
  install separately. It loads on demand and is released after 180 s idle, so it
  costs nothing at rest. Optional GPU offload via Vulkan
  (`scripts\app-build.bat --gpu`) — the same build still runs on CPU where there
  is no GPU.
- **Answers are retrieved, not guessed** — a question like *"what happened
  today?"* is resolved to a database query in Rust, run, and only then handed to
  the model to phrase, with the findings in front of it. It cannot invent an
  event, and if the model returns nothing usable the findings are reported
  directly: **the answer survives the language model failing entirely.**
- **Or bring your own** — OpenAI, Anthropic, Gemini, Groq, xAI, LM Studio, or any
  OpenAI-compatible endpoint, all behind one dispatcher.
- **Alerts** — Telegram, with inline acknowledgement and a `/menu` browser for
  people, vehicles and sounds. Risk thresholds, quiet hours, and per-category
  muting.

**Access**

- **Remote viewing** — share links over your own [Tailscale](https://tailscale.com/)
  Funnel, with expiry. No third-party relay, no account with us.
- **Optional login gate** — Argon2id, with recovery through Telegram and
  optional OTP.

## Architecture

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the system overview and
code layout.

Two design rules worth knowing up front, because they explain a lot of the code:

- **Runtime artifacts are provisioned, not vendored.** Model weights, GPU
  runtimes, `ffmpeg`, and `go2rtc` all go through `provision.rs` — fetched,
  digest-verified, unpacked, and installed atomically. There is exactly one
  answer to "is this installed", so a half-finished download can never read as
  ready.
- **Capability is asked, never inferred.** Whether the active AI provider can see
  an image, or is strong enough to classify risk, is a function of the provider —
  never of whether some model-name string happens to be non-empty.

## Quick start

### Prerequisites

- **Node.js** 20+
- **Rust** 1.78+ ([rustup](https://rustup.rs/))
- **Tauri prerequisites** for your OS —
  [tauri.app/start/prerequisites](https://v2.tauri.app/start/prerequisites/)
- **A C++ toolchain for llama.cpp** — `llama-cpp-sys-2` compiles it from source,
  so you also need **cmake**, **Ninja**, and **libclang**. On Windows that means
  a Visual Studio developer environment as well.
  [`CONTRIBUTING.md`](CONTRIBUTING.md) lists each one and the misleading error it
  produces when missing.

No AI service is required. The on-device model is installed from
**Guardian → Arsenal** inside the app.

### Run in dev mode

```bash
git clone https://github.com/anivarhq/anivar nivar
cd nivar
npm install
npm run tauri dev
```

### Build a release

On Windows, use the wrapper — it applies the three llama.cpp prerequisites and,
for installers, stages the bundled resources:

```bat
scripts\app-build.bat            REM exe only, fastest
scripts\app-build.bat --bundle   REM plus the MSI/NSIS installers
```

> **Don't use a bare `cargo build --release` to produce a shippable binary.** It
> compiles and exits 0, but it does **not** embed the frontend — the window falls
> back to the Vite dev URL and opens *"localhost refused to connect"* while the
> backend looks perfectly healthy. A good binary is ~64 MB; a broken one ~50 MB,
> and that 14 MB difference is the entire UI.

Output lands in `src-tauri/target/release/` (and `bundle/` with `--bundle`).

### Large binaries are not in this repo

`ffmpeg` and model weights are **not shipped at all** — the app fetches them on
demand (see the licence note below). The four
ONNX Runtime execution-provider DLLs are staged out of the cargo build output by
`scripts/sync-ort-dlls.ps1` — `ort` already downloads them, so their version
always matches the ORT you linked against, and a 92 MB DLL stays out of git
history.

## Project layout

```
.
├── docs/                   # Architecture documentation
├── scripts/                # Build wrappers (app-build, cargo-env, sync-ort-dlls)
├── src/                    # Frontend (React + TypeScript)
│   ├── api/                # Thin wrappers over Tauri commands
│   ├── components/         # Re-usable UI primitives
│   ├── features/           # agent, auth, camera(s), events, live, nvr,
│   │                       #   onboarding, persons, review, settings
│   ├── lib/                # Pure helpers (time, clip start, camera source)
│   ├── store/              # Zustand stores
│   └── workers/            # Web Workers
├── src-tauri/src/          # Backend (Rust), ~70 modules incl.
│   ├── lib.rs              # Command registry + module wiring
│   ├── agent/              # analysis, clip, chat, llm, local_llm, memory,
│   │                       #   dispatch, tools, prompts
│   ├── capture/dshow/rtsp  # Camera ingest
│   ├── nvr_*.rs            # Recording, indexing, HLS VOD, retention
│   ├── inference.rs        # ONNX Runtime + EP selection
│   ├── face*.rs, reid.rs   # Identity
│   ├── provision.rs        # The single contract for every managed download
│   └── tailscale.rs        # Remote access
└── vite.config.ts
```

## Configuration & data

Per-user runtime data lives in Tauri's app-data directory (on Windows,
`%APPDATA%\com.nivar.app\`):

| Path | Contents |
|---|---|
| `nivar.db` | SQLite — settings, events, alerts, identities, memory |
| `nvr/` | Continuous recording segments |
| `clip_*.mp4` | Exported motion-event clips |
| `skills/<id>/` | Installed models (detector, face, ALPR, search, `local_llm`) |
| `blobs/` | Face/display crops kept off the database |
| `logs/` | Size-capped rolling application log |
| `.master_key` | Local AES-GCM key encrypting secret settings fields |

Models are installed, switched, and **removed** from Guardian → Arsenal; removing
one deletes only its weights, never your footage, events, or enrolled people.

> **No secrets, tokens, or databases should ever be committed to this
> repository.** See [`.gitignore`](.gitignore).

## Privacy & security

Everything stays on your machine: no account, no telemetry, no Anivar server.
Nothing leaves the device unless you enable a feature that sends it (a cloud AI
provider, Telegram, remote access, a share link).

Face recognition stores **biometric data**, which is special-category personal
data under GDPR and equivalent laws — [`PRIVACY.md`](PRIVACY.md) covers what is
stored, what obligations you take on when your cameras see anyone beyond your own
household, and how to erase a person's biometrics completely (People → the person
→ Remove → confirm erase).

Found a vulnerability? Report it privately — see [`SECURITY.md`](SECURITY.md).
Please don't open a public issue for one.

## Contributing

PRs welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for dev setup, the build
prerequisites and their failure modes, and commit-message style, and
[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) for the ground rules.

## License

**Apache-2.0** — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). Use it, fork it,
ship it, build a product on it. Apache-2.0 also carries an explicit patent grant,
which MIT does not: contributors licence their patent claims to you, and that grant
terminates for anyone who sues over them.

Third-party components keep their own terms, listed in
[`THIRD-PARTY-NOTICES.md`](THIRD-PARTY-NOTICES.md). The short version:

* Everything **bundled** in the installer is MIT or Apache-2.0.
* **Model weights and `ffmpeg` are not bundled.** They are downloaded on request,
  from the upstream project, at your instruction — so their licences stay between
  you and their authors rather than riding along with this application.
* **No detector is preselected.** The first run asks which one you want and shows
  each licence, because that choice has consequences: the YOLO detectors are
  Ultralytics **AGPL-3.0**, and Ultralytics require a commercial licence for
  proprietary or commercial use. Permissively licensed detectors (RF-DETR,
  RT-DETR, D-FINE, YOLOX — all Apache-2.0) are the alternative.
