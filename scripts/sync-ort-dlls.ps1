# Stage the ONNX Runtime execution-provider DLLs that tauri.windows.conf.json
# bundles as resources.
#
# These are NOT third-party artifacts we have to go and find: the `ort` crate is
# built with `download-binaries` + `copy-dylibs`, so cargo already places them in
# target/<profile>/ during the Rust build. This copies them where the bundler
# expects them, which is why there is no URL and no SHA pin here — the files come
# from the same build that compiled against them, so their version can never drift
# from the linked ORT (they did drift-check identical: all four hashes matched the
# copies that used to be committed).
#
# They are gitignored because onnxruntime_providers_cuda.dll alone is 92 MB, over
# GitHub's 50 MB recommendation, and a blob that size is permanent once pushed.
#
# Usage: pwsh scripts/sync-ort-dlls.ps1 [-Profile release]
param([string]$Profile = "release")

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$src  = Join-Path $root "src-tauri/target/$Profile"
$dst  = Join-Path $root "src-tauri/binaries/onnx"

# Exactly the set listed under `bundle.resources` in tauri.windows.conf.json.
$needed = @(
  "onnxruntime_providers_shared.dll",
  "onnxruntime_providers_cuda.dll",
  "onnxruntime_providers_tensorrt.dll",
  "onnxruntime_providers_nv_tensorrt_rtx.dll"
)

New-Item -ItemType Directory -Force -Path $dst | Out-Null

$missing = @()
foreach ($f in $needed) {
  $from = Join-Path $src $f
  if (Test-Path $from) {
    Copy-Item $from (Join-Path $dst $f) -Force
    $mb = [math]::Round((Get-Item $from).Length / 1MB, 2)
    Write-Host ("  {0,-42} {1,8} MB" -f $f, $mb)
  } else {
    $missing += $f
  }
}

if ($missing.Count -gt 0) {
  Write-Host ""
  Write-Host "MISSING from ${src}:" -ForegroundColor Yellow
  $missing | ForEach-Object { Write-Host "  $_" -ForegroundColor Yellow }
  Write-Host ""
  Write-Host "Run a Rust build first so ort can download and copy them:" -ForegroundColor Yellow
  Write-Host "  scripts\cargo-env.bat build --$Profile" -ForegroundColor Yellow
  # Fail loudly: bundling without these produces an installer whose GPU lanes
  # cannot load, which shows up as a silent fallback to CPU rather than an error.
  exit 1
}

Write-Host "ORT provider DLLs staged in src-tauri/binaries/onnx/"
