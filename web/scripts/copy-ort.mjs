import { cpSync, mkdirSync, readdirSync, rmSync, statSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const webRoot = join(dirname(fileURLToPath(import.meta.url)), "..");

export function copyOrtAssets() {
  const dist = join(webRoot, "node_modules/onnxruntime-web/dist");
  const dest = join(webRoot, "public/ort");
  let entries;
  try {
    entries = readdirSync(dist);
  } catch {
    throw new Error(
      `onnxruntime-web assets not found at ${dist}. Run npm install inside web/.`,
    );
  }
  rmSync(dest, { recursive: true, force: true });
  mkdirSync(dest, { recursive: true });
  let copied = 0;
  for (const name of entries) {
    if (!/^ort-wasm.*\.(wasm|mjs)$/.test(name)) continue;
    const from = join(dist, name);
    if (!statSync(from).isFile()) continue;
    cpSync(from, join(dest, name));
    copied += 1;
  }
  if (copied === 0) {
    throw new Error(`no ort-wasm wasm/mjs files in ${dist}`);
  }
  return copied;
}

const invokedDirectly =
  Boolean(process.argv[1]) && import.meta.url === pathToFileURL(resolve(process.argv[1])).href;
if (invokedDirectly) {
  const count = copyOrtAssets();
  console.log(`copied ${count} onnxruntime-web wasm/mjs files to public/ort`);
}
