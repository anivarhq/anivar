@echo off
REM Build environment for llama-cpp-sys-2 (see CONTRIBUTING.md):
REM   vcvars64 - cmake cannot find cl.exe from a plain cargo shell
REM   Ninja    - the VS/MSBuild generator fails on llama.cpp's install target
REM   libclang - llama-cpp-sys-2 runs bindgen
REM Usage: scripts\cargo-env.bat build --release
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
set "CMAKE_GENERATOR=Ninja"
set "LIBCLANG_PATH=C:\Program Files\LLVM\bin"
cd /d "%~dp0..\src-tauri"
cargo %*