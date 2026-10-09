# Features, in detail

The [README](../README.md) says what Anivar does. This page says how each part
works, for anyone who wants to check before trusting it with their cameras.

## Cameras and recording

- **Multi-camera live view** — USB/built-in webcams, RTSP, MJPEG, and ONVIF
  discovery. Capture is server-side (one `ffmpeg` per camera, one clock), so the
  browser never touches the device.
- **Sub-second live streaming** — WebRTC through [go2rtc](https://github.com/AlexxIT/go2rtc)
  for RTSP cameras, degrading to HLS then MJPEG. A camera whose codec the window
  can't decode (some H.265 ones) still records natively and plays live over MJPEG.
- **24/7 recording** — segmented, faststart-remuxed `.mp4` with audio, bounded by
  a disk budget and per-tier retention. Detection runs on a hardware-decoded
  stream while the recording itself stays `-c copy`.
- **Recorded playback** — HLS VOD over the indexed segments. The day is a fixed
  list of hourly chunks and the player holds an index into it, so a click inside
  the loaded hour is a single `currentTime` assignment against media already in
  the buffer; only a target outside it requests a new playlist. Wall clock comes
  from each fragment's `EXT-X-PROGRAM-DATE-TIME`, which stays exact across
  recording gaps.

## Detection and understanding

- **Object detection** — ONNX Runtime in-process. Windows uses DirectML on any
  DX12 GPU, with CUDA and TensorRT available to NVIDIA cards through a downloaded
  pack; macOS uses CoreML; Linux runs on the CPU (CUDA for NVIDIA is
  experimental). YOLO26 tiers from nano to xlarge. None is preinstalled: you
  pick one in Arsenal, which shows its licence (Ultralytics AGPL-3.0) before
  you install it.
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
- **Privacy mode** — optional per-camera depth-map anonymisation for USB
  cameras (network cameras record raw video for now), enforced server-side: recordings and streams carry depth only, while the raw frame stays
  in memory for analysis.

## The assistant

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
  one of three Liquid AI models in-process: Fast (LFM2.5-350M, ~230 MB),
  Balanced (LFM2.5-1.2B, ~731 MB, the default) or Vision (LFM2.5-VL-1.6B,
  ~1.3 GB), which can look at a frame. It loads on demand and is
  released after 180 s idle, so it costs nothing at rest. Optional Vulkan GPU
  offload via `scripts\app-build.bat --gpu`; the same build still runs on CPU.
- **Or bring your own** — OpenAI, Anthropic, Gemini, Groq, xAI, LM Studio, or any
  OpenAI-compatible endpoint, all behind one dispatcher.
- **Alerts** — Telegram, with inline acknowledgement and a `/menu` browser for
  people, vehicles and sounds. Risk thresholds, quiet hours, and per-category
  muting.

## Access

- **Remote viewing** — signed, expiring share links over your own
  [Tailscale](https://tailscale.com/) Funnel, created from Telegram or by the
  assistant. Traffic goes device to device.
- **Optional login gate** — Argon2id password, with recovery and an optional
  six-digit code both sent over Telegram (so the gate needs Telegram set up).
  It locks the app's window, not the data on disk.

## Two design rules

- **Runtime artifacts are provisioned, not vendored.** Model weights, GPU
  runtimes, `ffmpeg`, and `go2rtc` all go through `provision.rs` — fetched,
  digest-verified, unpacked, and installed atomically. There is exactly one
  answer to "is this installed", so a half-finished download always reads as
  incomplete.
- **Capability is asked, never inferred.** Whether the active AI provider can see
  an image, or is strong enough to classify risk, is a function of the provider —
  never of whether some model-name string happens to be non-empty.
