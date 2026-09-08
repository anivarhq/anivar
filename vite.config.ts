import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import fs from "node:fs";
import path from "node:path";

const host = process.env.TAURI_DEV_HOST;

// Serve ONNX Runtime WASM files raw — Vite's module transform pipeline breaks
// dynamic import() of .mjs loader files in web workers (adds ?import and fails).
function serveOrtWasmRaw() {
  const ORT_FILES = new Set([
    "ort-wasm-simd-threaded.asyncify.mjs",
    "ort-wasm-simd-threaded.asyncify.wasm",
    "ort-wasm-simd-threaded.mjs",
    "ort-wasm-simd-threaded.wasm",
  ]);
  return {
    name: "serve-ort-wasm-raw",
    configureServer(server: any) {
      server.middlewares.use((req: any, res: any, next: any) => {
        const url = (req.url ?? "").split("?")[0];
        const filename = path.basename(url);
        if (!ORT_FILES.has(filename)) return next();
        const filePath = path.resolve(__dirname, "public/assets", filename);
        if (!fs.existsSync(filePath)) return next();
        const ct = filename.endsWith(".mjs") ? "application/javascript" : "application/wasm";
        res.setHeader("Content-Type", ct);
        res.setHeader("Cache-Control", "no-cache");
        fs.createReadStream(filePath).pipe(res);
      });
    },
  };
}

export default defineConfig(async () => ({
  plugins: [react(), serveOrtWasmRaw()],
  clearScreen: false,
  server: {
    port: 5174,
    strictPort: true,
    host: host || false,
    hmr: host ? { protocol: "ws", host, port: 5174 } : undefined,
    watch: { ignored: ["**/src-tauri/**"] },
  },
  envPrefix: ["VITE_", "TAURI_ENV_*"],
  build: {
    target: "chrome105",
    minify: !process.env.TAURI_ENV_DEBUG ? "esbuild" : false,
    sourcemap: !!process.env.TAURI_ENV_DEBUG,
  },
  worker: {
    format: "es",
  },
}));
