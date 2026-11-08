import type { IncomingMessage } from "node:http";
import react from "@vitejs/plugin-react";
import type { Plugin } from "vite";
import { defineConfig } from "vite";
import { copyOrtAssets } from "./scripts/copy-ort.mjs";

// ORT loads the glue as "/ort/*.mjs?import". Vite treats that query as a source
// import and answers with index.html, so the wasm module never loads.
function stripOrtImportQuery(req: IncomingMessage): void {
  if (!req.url || !req.url.startsWith("/ort/") || !req.url.includes("import")) return;
  const [pathname, query = ""] = req.url.split("?");
  const kept = query.split("&").filter((part) => part && part !== "import");
  req.url = kept.length > 0 ? `${pathname}?${kept.join("&")}` : pathname;
}

function copyOrtPlugin(): Plugin {
  return {
    name: "copy-ort-assets",
    buildStart() {
      copyOrtAssets();
    },
    configureServer(server) {
      copyOrtAssets();
      server.middlewares.use((req, _res, next) => {
        stripOrtImportQuery(req);
        next();
      });
    },
    configurePreviewServer(server) {
      server.middlewares.use((req, _res, next) => {
        stripOrtImportQuery(req);
        next();
      });
    },
  };
}

const host = "127.0.0.1";
const port = 5174;
const isolation = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

export default defineConfig({
  plugins: [react(), copyOrtPlugin()],
  server: {
    host,
    port,
    strictPort: true,
    headers: isolation,
  },
  preview: {
    host,
    port,
    strictPort: true,
    headers: isolation,
  },
  resolve: {
    conditions: ["onnxruntime-web-use-extern-wasm"],
  },
  optimizeDeps: {
    exclude: ["onnxruntime-web"],
  },
});
