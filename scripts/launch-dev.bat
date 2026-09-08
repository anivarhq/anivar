@echo off
REM Launch Anivar in development mode (Tauri + Vite hot reload).
cd /d "%~dp0\.."
echo Starting Anivar in development mode...
npm run tauri dev
