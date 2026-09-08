@echo off
REM Build the SHIPPABLE app. Use this, not `cargo build --release`.
REM
REM Bare cargo does NOT embed the frontend here: the binary comes out ~13.8 MB
REM smaller and the window falls back to tauri.conf.json's devUrl, so it opens
REM "localhost refused to connect" unless a Vite dev server happens to be up.
REM The tauri CLI is what runs beforeBuildCommand and bakes dist/ into the exe.
REM
REM Same three prerequisites as scripts\cargo-env.bat (llama-cpp-sys-2).
REM Usage: scripts\app-build.bat              (exe only, fastest)
REM        scripts\app-build.bat --bundle     (also build the MSI/NSIS installers)
REM        scripts\app-build.bat --gpu        (Vulkan offload for the on-device LLM)
REM
REM --gpu additionally needs the Vulkan SDK, for its glslc shader compiler:
REM        winget install KhronosGroup.VulkanSDK
REM Without it the cmake step fails outright rather than degrading, which is
REM exactly why Vulkan is not in the default feature set.
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
set "CMAKE_GENERATOR=Ninja"
set "LIBCLANG_PATH=C:\Program Files\LLVM\bin"
cd /d "%~dp0.."

REM Flags may be given in any order, so parse rather than testing %1.
set "SC_FEATURES="
set "SC_BUNDLE="
:parse
if "%~1"=="" goto parsed
if /i "%~1"=="--gpu"    set "SC_FEATURES=--features vulkan"
if /i "%~1"=="--bundle" set "SC_BUNDLE=1"
shift
goto parse
:parsed

if defined SC_FEATURES (
  if not defined VULKAN_SDK (
    echo [app-build] --gpu needs the Vulkan SDK ^(for glslc^). Install it with:
    echo             winget install KhronosGroup.VulkanSDK
    echo             then open a NEW shell so VULKAN_SDK is set.
    exit /b 1
  )
  echo [app-build] Vulkan GPU offload enabled for the on-device model
)

REM --bundle only: the installer lists the ONNX provider DLLs under
REM bundle.resources, and they're gitignored (92 MB), so stage them from the cargo
REM build output first. Compile first so `ort` has downloaded and copied them; the
REM tauri build after this reuses those artifacts, so it costs one link, not a
REM second full compile. --no-bundle skips all of it (no resources are read).
if defined SC_BUNDLE (
  echo [app-build] compiling so ort materialises its provider DLLs...
  cargo build --release --manifest-path src-tauri/Cargo.toml %SC_FEATURES%
  if errorlevel 1 exit /b 1
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0sync-ort-dlls.ps1" -Profile release
  if errorlevel 1 exit /b 1
  npm run tauri build -- %SC_FEATURES%
) else (
  npm run tauri build -- --no-bundle %SC_FEATURES%
)
