<div align="center">

<img src="assets/logo.png" alt="Anivar" width="120" />

# Anivar

### Local-first NVR with on-device AI

[![Latest release](https://img.shields.io/github/v/release/anivarhq/anivar?style=for-the-badge&labelColor=0d1117)](https://github.com/anivarhq/anivar/releases/latest)
[![Downloads](https://img.shields.io/github/downloads/anivarhq/anivar/total?style=for-the-badge&labelColor=0d1117)](https://github.com/anivarhq/anivar/releases)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue?style=for-the-badge&labelColor=0d1117)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/anivarhq/anivar/ci.yml?branch=main&style=for-the-badge&labelColor=0d1117&label=CI)](https://github.com/anivarhq/anivar/actions/workflows/ci.yml)

[**Download**](#download) · [**What it does**](#what-it-does) ·
[**Architecture**](#architecture) · [**Build from source**](#build-from-source) ·
[**Privacy & security**](#privacy--security)

</div>

---

A local-first NVR with on-device AI. One cross-platform desktop app built on
[Tauri 2](https://v2.tauri.app/) with a React frontend and a Rust backend —
recording, detection, recognition, and event analysis all run on your machine,
including the language model.

> [!WARNING]
> **Alpha.** APIs and the data-directory layout are still subject to change, and
> the installers are not code-signed yet — your OS will warn you. See
> [Before you install](#before-you-install).

## Download

Latest release: **v0.1.0**. Every file below has a matching `.sha256` on the
[releases page](https://github.com/anivarhq/anivar/releases/latest), and what
changed is in [`CHANGELOG.md`](CHANGELOG.md).

| Platform | Download | |
|---|---|---|
| **Windows** 10/11, x64 | [`Anivar_0.1.0_x64-setup.exe`](https://github.com/anivarhq/anivar/releases/download/v0.1.0/Anivar_0.1.0_x64-setup.exe) | 66 MB · installs per-user, no admin |
| **macOS** 11+, Apple Silicon | [`Anivar_0.1.0_aarch64.dmg`](https://github.com/anivarhq/anivar/releases/download/v0.1.0/Anivar_0.1.0_aarch64.dmg) | 19 MB |
| **Linux** x86_64 | [`Anivar_0.1.0_amd64.AppImage`](https://github.com/anivarhq/anivar/releases/download/v0.1.0/Anivar_0.1.0_amd64.AppImage) | 97 MB · portable, `chmod +x` and run |
| | [`Anivar_0.1.0_amd64.deb`](https://github.com/anivarhq/anivar/releases/download/v0.1.0/Anivar_0.1.0_amd64.deb) | 24 MB · Debian, Ubuntu |
| | [`Anivar-0.1.0-1.x86_64.rpm`](https://github.com/anivarhq/anivar/releases/download/v0.1.0/Anivar-0.1.0-1.x86_64.rpm) | 24 MB · Fedora, RHEL, openSUSE |

One download per platform. The Windows installer carries the CUDA execution
provider and detects an NVIDIA card at runtime — there is no separate GPU
edition to choose between. Without one it uses DirectML on any DX12 GPU. macOS
uses CoreML; Linux runs detection on the CPU today.

There is no Intel-Mac or 32-bit build. Linux binaries are built against
`webkit2gtk-4.1`, so Ubuntu 22.04 / Debian 12 or newer.

Installed builds update themselves. The updater checks a manifest signed with
the project's key and refuses anything that fails that signature, so an update
cannot come from anywhere else.

### Verify your download

The installers are not code-signed yet, so a checksum is the only thing that
tells you the file arrived intact. Download the `.sha256` next to it, then:

```powershell
# Windows (PowerShell) — compare this against the .sha256 file's contents
(Get-FileHash .\Anivar_0.1.0_x64-setup.exe -Algorithm SHA256).Hash
```

```bash
# macOS
shasum -a 256 -c Anivar_0.1.0_aarch64.dmg.sha256
# Linux
sha256sum -c Anivar_0.1.0_amd64.deb.sha256
```

A SHA-256 is not a signature — anyone who could replace the installer could
replace the hash beside it. It catches a corrupted download and a mirror that
altered the file. Real code signing is on the list; it needs a purchased
certificate.

### Before you install

The Windows and macOS builds are not signed by a certificate authority yet, so
both will stop you the first time (Linux does not check):

- **Windows** shows *"Windows protected your PC"*. Click **More info** →
  **Run anyway**.
- **macOS** refuses to open an app that isn't notarised. Try to open it once,
  then go to **System Settings → Privacy & Security** and click **Open Anyway**.
  From a terminal, `xattr -dr com.apple.quarantine /Applications/Anivar.app`
  does the same thing.

If that trade is not one you want to make, [build from
source](#build-from-source) — the release workflow that produced these files is
[`release.yml`](.github/workflows/release.yml), and it builds nothing that isn't
in this repository.

## What it does

### Cameras and recording

- **Multi-camera live view** — USB/built-in webcams, RTSP, MJPEG, and ONVIF
  discovery. Capture is server-side (one `ffmpeg` per camera, one clock), so the
  browser never touches the device.
- **Sub-second live streaming** — WebRTC through [go2rtc](https://github.com/AlexxIT/go2rtc),
  degrading to HLS then MJPEG.
- **24/7 recording** — segmented, faststart-remuxed `.mp4` with audio, bounded by
  a disk budget and per-tier retention. Detection runs on a hardware-decoded
  stream while the recording itself stays `-c copy`.
- **Recorded playback** — HLS VOD over the indexed segments. The day is a fixed
  list of hourly chunks and the player holds an index into it, so a click inside
  the loaded hour is a single `currentTime` assignment against media already in
  the buffer; only a target outside it requests a new playlist. Wall clock comes
  from each fragment's `EXT-X-PROGRAM-DATE-TIME`, which stays exact across
  recording gaps.

### Detection and understanding

- **Object detection** — ONNX Runtime in-process. Windows uses DirectML on any
  DX12 GPU, with CUDA and TensorRT available to NVIDIA cards through a downloaded
  pack; macOS uses CoreML; Linux runs on the CPU today. Selectable YOLO tiers from
  nano to xlarge. You choose a detector on first run and its licence is shown
  before you commit to it.
- **Motion detection with masks** — grayscale frame-diff with box blur, polygon
  masks for noisy regions (fans, trees, monitors), hysteresis open/close, and a
  lightning guard for whole-frame flashes.
- **Faces and people** — ArcFace recognition with track-consensus naming (a name
  needs agreement across frames rather than one lucky frame), person re-ID for
  cross-event linking, and self-learning corrections.
- **Licence plates** — ALPR with known-plate matching and recurring-plate
  proposals.
- **Audio events** — YAMNet AudioSet classification (glass, alarm, shouting…)
  with a sustained-detection model, so an alert needs a pattern rather than one
  spike.
- **Semantic search** — natural-language search over the archive ("red shirt",
  "person with a package") using CLIP embeddings of the detected object crop,
  indexed with usearch HNSW.
- **Privacy mode** — optional per-camera depth-map anonymisation, enforced
  server-side: recordings and streams carry depth only, while the raw frame stays
  in memory for analysis.

### The assistant

- **Evidence-first investigator** — ask "what happened last night?" and get
  playable clips, snapshots, and person cards back alongside the prose. The
  conversation is durable and shared with the Telegram channel, and every answer
  states which day or range it covers.
- **Answers are retrieved, then phrased** — a question like *"what happened
  today?"* resolves to a database query in Rust, which runs first; the model sees
  the findings and writes them up. Every claim traces to a row, and if the model
  returns nothing usable the findings are reported directly, so the answer
  survives the language model failing entirely.
- **Memory that stays a fixed size** — what the agent learns is stored
  permanently but *retrieved* per question (relevance × recency × how often it has
  been confirmed) against a fixed budget, so the context sent to the model is the
  same size whether it has learned ten facts or ten thousand.
- **On-device language model** — llama.cpp is compiled into the binary and runs
  LFM2.5-1.2B-Instruct (Q4_K_M, ~731 MB) in-process. It loads on demand and is
  released after 180 s idle, so it costs nothing at rest. Optional Vulkan GPU
  offload via `scripts\app-build.bat --gpu`; the same build still runs on CPU.
- **Or bring your own** — OpenAI, Anthropic, Gemini, Groq, xAI, LM Studio, or any
  OpenAI-compatible endpoint, all behind one dispatcher.
- **Alerts** — Telegram, with inline acknowledgement and a `/menu` browser for
  people, vehicles and sounds. Risk thresholds, quiet hours, and per-category
  muting.

### Access

- **Remote viewing** — share links over your own [Tailscale](https://tailscale.com/)
  Funnel, with expiry. Traffic goes device to device.
- **Optional login gate** — Argon2id, with recovery through Telegram and
  optional OTP.

## Architecture

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the system overview and
code layout.

Two design rules worth knowing up front, because they explain a lot of the code:

- **Runtime artifacts are provisioned, not vendored.** Model weights, GPU
  runtimes, `ffmpeg`, and `go2rtc` all go through `provision.rs` — fetched,
  digest-verified, unpacked, and installed atomically. There is exactly one
  answer to "is this installed", so a half-finished download always reads as
  incomplete.
- **Capability is asked, never inferred.** Whether the active AI provider can see
  an image, or is strong enough to classify risk, is a function of the provider —
  never of whether some model-name string happens to be non-empty.

## Build from source

### Prerequisites

- **Node.js** 22+ (CI builds on 24; see [`.nvmrc`](.nvmrc))
- **Rust** 1.78+ ([rustup](https://rustup.rs/))
- **Tauri prerequisites** for your OS —
  [tauri.app/start/prerequisites](https://v2.tauri.app/start/prerequisites/)
- **A C++ toolchain for llama.cpp** — `llama-cpp-sys-2` compiles it from source,
  so you also need **cmake**, **Ninja**, and **libclang**. On Windows that means
  a Visual Studio developer environment as well.
  [`CONTRIBUTING.md`](CONTRIBUTING.md) lists each one and the misleading error it
  produces when missing.

An AI service is optional. The on-device model installs from **Guardian →
Arsenal** inside the app.

### Run in dev mode

```bash
git clone https://github.com/anivarhq/anivar
cd anivar
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

> **Use the wrapper, not a bare `cargo build --release`, for a shippable
> binary.** Bare cargo compiles and exits 0, but the frontend is embedded by the
> Tauri CLI's `beforeBuildCommand` — without it the window falls back to the Vite
> dev URL and opens *"localhost refused to connect"* while the backend looks
> perfectly healthy. A good binary is ~50 MB; one missing the frontend is ~35 MB.

Output lands in `src-tauri/target/release/` (and `bundle/` with `--bundle`).

### Runtime binaries are fetched on demand

`ffmpeg`, `go2rtc`, and model weights are downloaded at your instruction rather
than shipped (see the licence note below). The four ONNX Runtime
execution-provider DLLs are staged out of the cargo build output by
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
│   ├── features/           # agent, auth, camera, cameras, live, nvr,
│   │                       #   onboarding, persons, review, settings
│   ├── lib/                # Pure helpers (time, clip start, camera source)
│   └── store/              # Zustand stores
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
`%APPDATA%\com.anivar.app\`):

| Path | Contents |
|---|---|
| `anivar.db` | SQLite — settings, events, alerts, identities, memory |
| `nvr/` | Continuous recording segments |
| `clip_*.mp4` | Exported motion-event clips |
| `skills/<id>/` | Installed models (detector, face, ALPR, search, `local_llm`) |
| `blobs/` | Face/display crops kept off the database |
| `logs/` | Size-capped rolling application log |
| `.master_key` | Local AES-GCM key encrypting secret settings fields |

Models are installed, switched, and removed from Guardian → Arsenal; removing one
deletes that model's weights and leaves your footage, events, and enrolled people
intact.

> Keep secrets, tokens, and databases out of this repository. See
> [`.gitignore`](.gitignore).

## Privacy & security

Everything stays on your machine. Data leaves the device only through a feature
you turn on: a cloud AI provider, Telegram, remote access, or a share link.

Face recognition stores **biometric data**, which is special-category personal
data under GDPR and equivalent laws — [`PRIVACY.md`](PRIVACY.md) covers what is
stored, what obligations you take on when your cameras see anyone beyond your own
household, and how to erase a person's biometrics completely (People → the person
→ Remove → confirm erase).

Found a vulnerability? Report it privately through
[GitHub's advisory form](https://github.com/anivarhq/anivar/security/advisories/new)
— [`SECURITY.md`](SECURITY.md) covers scope, what helps a report land, and what to
expect back.

## Contributing

PRs welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for dev setup, the build
prerequisites and their failure modes, and commit-message style, and
[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) for the ground rules.

## License

**Apache-2.0** — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). Use it, fork it,
ship it, build a product on it. Apache-2.0 also carries an explicit patent grant,
which MIT does not: contributors licence their patent claims to you, and that
grant terminates for anyone who sues over them.

Third-party components keep their own terms, listed in
[`THIRD-PARTY-NOTICES.md`](THIRD-PARTY-NOTICES.md). The short version:

* Everything **bundled** in the installer is MIT or Apache-2.0.
* **Model weights, `ffmpeg`, and `go2rtc` are fetched on request**, from the
  upstream project, at your instruction — so their licences stay between you and
  their authors rather than riding along with this application.
* **The first run asks which detector you want and shows each licence**, because
  the choice has consequences: the YOLO detectors are Ultralytics **AGPL-3.0**,
  and Ultralytics require a commercial licence for proprietary or commercial use.
  RF-DETR, RT-DETR, D-FINE and YOLOX are Apache-2.0 alternatives.
