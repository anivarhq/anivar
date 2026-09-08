# Third-party notices

This application is Apache-2.0 licensed (see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE)).
It builds on third-party components under their own terms, summarised here.

## Bundled with the application

These ship inside the installer, so their licences travel with the product.

| Component | Licence | Notes |
|---|---|---|
| Tauri, wry, tao | MIT / Apache-2.0 | Application shell |
| ONNX Runtime (`ort`) | MIT | Inference engine |
| ONNX Runtime execution providers | MIT | CUDA / TensorRT / DirectML provider libraries |
| llama.cpp (`llama-cpp-2`) | MIT | On-device language model runtime, compiled in |
| React, Vite, Zustand | MIT | Frontend |
| Rust crates (see `Cargo.toml`) | MIT / Apache-2.0 | Full list via `cargo tree` |

## NOT bundled — downloaded by the user, under the upstream's terms

Model weights and media binaries are **deliberately not shipped**. The application
downloads them on request, from the upstream project, at the user's instruction. They are
never redistributed by this project, and each remains governed by its own licence.

| Component | Licence | What you should know |
|---|---|---|
| **Ultralytics YOLO26** detector | **AGPL-3.0** | Ultralytics states that proprietary or commercial use requires their **Enterprise licence**, not merely AGPL compliance. Choosing this detector is a decision with licence consequences for whatever you build around it. See <https://www.ultralytics.com/license>. |
| **ffmpeg** | GPL (the build we fetch) | Invoked as a separate process for recording and playback. Not linked into the application. |
| ArcFace / face models | See each model card | Recognition + embeddings |
| CLIP / MobileCLIP / Jina-CLIP | See each model card | Semantic search |
| YAMNet | Apache-2.0 | Audio event classification |
| **LFM2.5-1.2B-Instruct** (GGUF) | **LFM Open License v1.0** | The on-device language model. Liquid AI's own licence — **not** Apache-2.0 or MIT. Fine for personal and evaluation use; read the terms before shipping a commercial product on it. See <https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct>. |
| go2rtc | MIT | WebRTC streaming, fetched on demand |

### Why models are not bundled

Shipping a model inside the installer means *distributing* it, which drags its licence onto
the product. Downloading it on the user's instruction, from the upstream, keeps that
relationship between the user and the model's author — the same approach other open-source
NVR projects take.

**No detector is preselected or installed automatically.** The first run asks which one to
use and shows each model's licence, so the choice is explicit and informed rather than a
default someone inherits without noticing.

### If you plan to sell something built on this

Two items need a decision before you do:

* **Ultralytics YOLO (AGPL-3.0)** — an Enterprise licence is required for proprietary or
  commercial use. Alternatively use a permissively licensed detector; RF-DETR, RT-DETR,
  D-FINE and YOLOX are Apache-2.0. *(Note: as of today only the YOLO detectors are wired
  into this application — adding a permissive one is tracked work, not a settings change.)*
* **ffmpeg (GPL)** — fine as a separately-downloaded, separately-executed binary. It
  becomes your obligation only if you start bundling or linking it.
