export const EXPORT_COMMANDS = "dreamarcade export\ndreamarcade smoke";

export interface PanelCopy {
  title: string;
  body: string;
}

export function modelMissing(file: string): PanelCopy {
  const command = file.startsWith("breakout/") ? "dreamarcade smoke data=breakout" : EXPORT_COMMANDS;
  return {
    title: `${file} was not found`,
    body: `Nothing at /models/${file}.\nThe denoiser is not running, so this view stays blank.\n\nFrom the repo root:\n${command}`,
  };
}

export function replayMissing(file = "pong.bin"): PanelCopy {
  const command = file.startsWith("breakout") ? "dreamarcade smoke data=breakout" : EXPORT_COMMANDS;
  return {
    title: "Replay not found",
    body: `web/public/replays/${file} is missing.\n\nFrom the repo root:\n${command}`,
  };
}

export function bothEpFailed(detail: string): PanelCopy {
  return {
    title: "WebGPU and WASM both failed",
    body: `WebGPU and WASM both failed.\nUse Chromium with WebGPU, or a browser that can run the WASM fallback.\n\n${detail}\n\nFrom the repo root:\n${EXPORT_COMMANDS}`,
  };
}

export function wasmMissing(detail: string): PanelCopy {
  return {
    title: "WASM assets missing",
    body: `ONNX Runtime could not load a wasm file from /ort/.\n${detail}\n\nnpm run dev and npm run build copy those files out of node_modules/onnxruntime-web/dist.\n\nFrom the repo root:\n${EXPORT_COMMANDS}`,
  };
}
