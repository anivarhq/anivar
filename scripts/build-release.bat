@echo off
REM Build a release installer for Anivar.
cd /d "%~dp0\.."
echo Building Anivar release installer...
npm run tauri build
echo.
echo Build complete. Check src-tauri\target\release\bundle\ for the installer.
pause
