# Contributing to Anivar

Thanks for considering a contribution. Anivar is in active development and
PRs of every size are welcome — bug reports, documentation improvements,
small fixes, and feature work.

## Filing issues

Use the templates in [`.github/ISSUE_TEMPLATE/`](.github/ISSUE_TEMPLATE/).
Always include:

- Your OS + version
- The Anivar version (or commit SHA if running from source)
- Steps to reproduce
- What you expected to happen vs. what actually happened
- Relevant log output (the `tracing` subscriber prints to stderr)

**Never include real secrets.** Telegram tokens, API keys, RTSP credentials,
and similar should be redacted before posting.

## Development setup

See [`README.md`](README.md) for the prerequisite list. Quick start:

```bash
git clone <your-fork-url> nivar
cd nivar
npm install
npm run tauri dev
```

The Vite dev server and the Rust crate both hot-reload. The application data
directory (`%APPDATA%\com.nivar.app\` on Windows etc.) is shared with any
release build of Anivar already installed on your machine — if that is
undesirable, run with `--config` to point Tauri at a different bundle
identifier, or back up `nivar.db` before iterating.

## Building from source

### Native toolchain prerequisites

Anivar runs its on-device language model by compiling **llama.cpp into the
binary** (see `agent/local_llm.rs` — no Ollama, no daemon). That means the Rust
build shells out to cmake and runs bindgen, so a plain `cargo build` is not
enough. You need, on every platform:

| | |
|---|---|
| **cmake** | Windows: bundled with VS Build Tools · Linux: `apt install cmake` · macOS: `brew install cmake` |
| **ninja** | The Visual Studio generator fails on llama.cpp's `install` target — use Ninja (`CMAKE_GENERATOR=Ninja`). It ships with VS Build Tools. |
| **libclang** | bindgen needs it. Windows: install LLVM and set `LIBCLANG_PATH`. Linux: `libclang-dev`. macOS: `brew install llvm`. |
| **MSVC dev env** (Windows) | cmake cannot find `cl.exe` from a plain cargo shell — build from a **Developer Command Prompt**, or `call vcvars64.bat` first. |

On Windows the whole build therefore looks like:

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "LIBCLANG_PATH=C:\Program Files\LLVM\bin"
set "CMAKE_GENERATOR=Ninja"
npm run tauri build
```

For plain cargo work, `scripts\cargo-env.bat` applies all three and forwards its
arguments — `scripts\cargo-env.bat test --lib`, `scripts\cargo-env.bat build --release`.

### ONNX Runtime provider DLLs are not in the repo

`tauri.windows.conf.json` bundles four execution-provider DLLs as resources, and they
are **gitignored** — `onnxruntime_providers_cuda.dll` alone is 92 MB, past GitHub's
50 MB recommendation, and a blob that size is permanent once pushed.

They need no download. `ort` is built with `download-binaries` + `copy-dylibs`, so a
Rust build already places them in `src-tauri/target/<profile>/`;
`scripts\sync-ort-dlls.ps1` copies them where the bundler looks. Because they come
from the build that compiled against them, their version can never drift from the
linked ORT — which is why there is no URL and no SHA pin.

`scripts\app-build.bat --bundle` does this for you (compile → stage → bundle). You
only need the script directly if you invoke `tauri build` yourself. Skipping it makes
the bundler fail on a missing resource path; skipping it in a build that somehow
proceeds would produce an installer whose GPU lanes silently fall back to CPU.

Two traps worth knowing, both of which produce misleading errors:

* A failed cmake configure leaves a **poisoned `CMakeCache.txt`**; the next build
  reports `MSB1009: Project file does not exist` or "already configured".
  Run `cargo clean -p llama-cpp-sys-2` before retrying.
* MSVC reports **MAX_PATH overflow as `fatal error C1041`** ("cannot open program
  database"). If you build from a deeply nested directory the cmake scratch paths
  can exceed 260 characters — build from a shorter path.

### Commands

```bash
# Just the frontend (fast)
npm run build

# Just the Rust crate (needs the toolchain above)
cd src-tauri && cargo check

# Full release installer (slow)
npm run tauri build
```

The release build invokes LTO and takes several minutes the first time; the
first build in a clean checkout also compiles llama.cpp, which adds a few more.

## Code style

### Rust (`src-tauri/`)

- Format with `cargo fmt`. The default 4-space rustfmt config is fine.
- Lint with `cargo clippy --all-targets -- -D warnings` before opening a PR.
- Use `tracing::info!` / `tracing::warn!` / `tracing::debug!` for any
  diagnostic output — never `println!`.
- New SQL goes through `sqlx`. Always parameterise (`?` binds).
- Avoid panics in request-handling code; bubble errors as `anyhow::Result`
  or convert to `String` at the Tauri-command boundary.
- The motion-detection and inference hot paths run per-frame at 10–20 FPS.
  Avoid allocation in tight loops. If you must allocate, profile first.

### TypeScript (`src/`)

- Format with Prettier (the repo currently relies on each contributor's
  Prettier defaults — we will add a checked-in config soon).
- Run `npm run build` before opening a PR; it runs `tsc` and surfaces any
  type errors.
- Feature folders are self-contained — keep cross-feature dependencies going
  through `src/api/`, `src/store/`, and `src/lib/`.

### Commit messages

We use loosely-conventional commits — one of:

```
feat(camera): add ONVIF auto-discovery for IP cameras
fix(motion): correct polygon coords when canvas is letterboxed
docs(readme): document the data directory layout
refactor(agent): extract memory helpers into agent/memory.rs
chore(deps): bump tauri to 2.4
```

Squash-merge will rewrite to a single Conventional Commit on `main`.

## Tests

Automated tests are minimal today. We are happy to accept tests for any
subsystem you touch. The motion-detection and YOLO26 decoder paths are good
first candidates — they are pure functions with deterministic outputs.

When adding tests, place them next to the code under `#[cfg(test)] mod tests`
in Rust, and in `__tests__/` folders next to the feature in TypeScript.

## Reviewing your own PR

Before requesting review:

- [ ] `cargo check` is green
- [ ] `cargo clippy --all-targets -- -D warnings` is green
- [ ] `npm run build` succeeds
- [ ] No secrets / personal data in the diff (logs, screenshots, `.db` files)
- [ ] If a user-visible change: `CHANGELOG.md` updated under `[Unreleased]`
- [ ] If a Rust subsystem changed: a one-line note in `docs/ARCHITECTURE.md`

## License

This project is **Apache-2.0** ([`LICENSE`](LICENSE)). By contributing, you
agree that your contribution is licensed under those terms, including its
patent grant.
