# Roadmap

Where Anivar is heading, in rough order. This is a direction, not a set of
dates. Anivar is alpha, and the order changes when something real (a bug
report, a camera that doesn't work) outranks it.

Want to pick something up? Say so on the issue or in
[Discussions](https://github.com/anivarhq/anivar/discussions) first, so two
people don't build the same thing. Smaller starting points are labelled
[good first issue](https://github.com/anivarhq/anivar/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).

## Now

- **Share and revoke from inside the app.** Signed, expiring share links
  already exist, and so does "revoke every link". Today, though, links are
  created only from Telegram and the assistant, and revoking has no button. The
  backend is done (`share_cmds.rs`); the app needs the UI.

## Next

- **Privacy mode for network cameras.** Depth-only recording works for USB
  cameras. RTSP cameras are refused, because their stream is recorded by
  copying it, not re-encoding it.
- **H.265 live view in the app.** H.265 cameras record fine, but live view
  falls back to MJPEG where the webview can't decode H.265.
- **GPU inference on Linux, verified.** A CUDA path exists and builds. It has
  not been tested on real NVIDIA hardware, so Linux runs models on the CPU
  today.
- **Signed installers.** Windows and macOS builds aren't code-signed or
  notarised yet, so both systems ask once before opening them. Signing needs
  paid certificates.

## Later

- **Intel Macs.** macOS builds are Apple Silicon only.
- **More detectors with permissive licences.** The shipped detectors are YOLO
  (AGPL-3.0). Apache-2.0 alternatives such as RF-DETR or D-FINE would give a
  choice for commercial use.

## Under the hood

Good places to start if you'd rather work on the engine than the interface:

- **Integration tests for recording and motion.** Unit tests cover the pieces
  (`scripts\cargo-env.bat test --lib`), but nothing yet drives a real stream
  end to end through motion detection, recording and segment indexing.
- **A written schema for event summaries.** The `ai_summary` JSON that
  `agent/analysis.rs` stores on each event is documented only by the code that
  writes and reads it.

## Deliberately not planned

- **More alert channels.** Telegram is the one alert channel, and it gets the
  attention that webhooks, ntfy, Discord or Slack would split. (Home Assistant
  is covered differently: the MQTT bridge publishes motion, events and camera
  status for automations, rather than sending alerts.)
- **An Anivar cloud or account.** Everything runs on your own computer. Remote
  viewing goes over your own Tailscale network.
