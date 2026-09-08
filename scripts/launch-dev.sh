#!/usr/bin/env bash
# Launch Anivar in development mode (Tauri + Vite hot reload).
set -euo pipefail
cd "$(dirname "$0")/.."
echo "Starting Anivar in development mode..."
npm run tauri dev
