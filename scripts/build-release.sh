#!/usr/bin/env bash
# Build a release installer for Anivar (macOS .dmg, Linux .AppImage/.deb, etc.).
set -euo pipefail
cd "$(dirname "$0")/.."
echo "Building Anivar release installer..."
npm run tauri build
echo
echo "Build complete. Check src-tauri/target/release/bundle/ for the installer."
