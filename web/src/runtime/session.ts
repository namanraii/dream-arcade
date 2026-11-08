import * as ort from "onnxruntime-web/webgpu";
import type { Game } from "../store";
import { bothEpFailed, modelMissing, wasmMissing } from "../ui/messages";

function modelPath(game: Game, file: string): string {
  return game === "breakout" ? `breakout/${file}` : file;
}

ort.env.wasm.wasmPaths = "/ort/";
const cores = typeof navigator !== "undefined" ? navigator.hardwareConcurrency || 2 : 2;
ort.env.wasm.numThreads = Math.max(2, Math.min(4, cores));

export interface Denoiser {
  file: string;
  provider: string;
  run: (history: Float32Array, action: number, noisy: Float32Array, sigma: number) => Promise<Float32Array>;
}

export interface RewardHead {
  run: (history: Float32Array, action: number) => Promise<{ logits: Float32Array; doneLogit: number }>;
}

export interface PolicyHead {
  run: (history: Float32Array) => Promise<Float32Array>;
}

async function fetchModel(file: string): Promise<ArrayBuffer | null> {
  const response = await fetch(`/models/${file}`);
  if (!response.ok) return null;
  return response.arrayBuffer();
}

let ortGate: Promise<unknown> = Promise.resolve();

function enqueueOrt<T>(job: () => Promise<T>): Promise<T> {
  const run = ortGate.then(job, job);
  ortGate = run.then(
    () => undefined,
    () => undefined,
  );
  return run;
}

async function openSession(bytes: ArrayBuffer, providers: string[]): Promise<ort.InferenceSession> {
  return enqueueOrt(() =>
    ort.InferenceSession.create(bytes, {
      executionProviders: providers,
      graphOptimizationLevel: "disabled",
    }),
  );
}

export async function loadDenoiser(game: Game = "pong"): Promise<Denoiser> {
  const fp16Name = modelPath(game, "denoiser_fp16.onnx");
  const fp32Name = modelPath(game, "denoiser.onnx");
  const fp16 = await fetchModel(fp16Name);
  const fp32 = await fetchModel(fp32Name);
  if (!fp16 && !fp32) {
    const copy = modelMissing(fp32Name);
    throw new Error(`${copy.title}\n${copy.body}`);
  }
  const attempts: Array<{ file: string; bytes: ArrayBuffer; providers: string[] }> = [];
  if (fp16) attempts.push({ file: fp16Name, bytes: fp16, providers: ["webgpu"] });
  if (fp32) attempts.push({ file: fp32Name, bytes: fp32, providers: ["webgpu"] });
  if (fp32) attempts.push({ file: fp32Name, bytes: fp32, providers: ["wasm"] });
  const errors: string[] = [];
  for (const attempt of attempts) {
    try {
      const session = await openSession(attempt.bytes, attempt.providers);
      const provider = attempt.providers[0];
      return {
        file: attempt.file,
        provider,
        run: (history, action, noisy, sigma) => runDenoiser(session, history, action, noisy, sigma),
      };
    } catch (error) {
      errors.push(`${attempt.file} via ${attempt.providers[0]}: ${errorText(error)}`);
    }
  }
  const detail = errors.join("\n");
  const copy = isWasmAssetFailure(detail) ? wasmMissing(detail) : bothEpFailed(detail);
  throw new Error(`${copy.title}\n${copy.body}`);
}

export async function loadReward(game: Game = "pong"): Promise<RewardHead | null> {
  const bytes = await fetchModel(modelPath(game, "reward.onnx"));
  if (!bytes) return null;
  const session = await openSession(bytes, ["wasm"]);
  return {
    async run(history, action) {
      return enqueueOrt(async () => {
        const feeds = {
          history: tensorF32(history, [1, 12, 64, 64]),
          action: tensorAction(action),
        };
        try {
          const out = await session.run(feeds);
          return {
            logits: copyFloat(out.reward_logits),
            doneLogit: copyFloat(out.done_logit)[0] ?? 0,
          };
        } finally {
          disposeFeeds(feeds);
        }
      });
    },
  };
}

export async function loadPolicy(game: Game = "pong"): Promise<PolicyHead | null> {
  const bytes = await fetchModel(modelPath(game, "policy.onnx"));
  if (!bytes) return null;
  const session = await openSession(bytes, ["wasm"]);
  return {
    async run(history) {
      return enqueueOrt(async () => {
        const feeds = {
          history: tensorF32(history, [1, 12, 64, 64]),
        };
        try {
          const out = await session.run(feeds);
          return copyFloat(out.logits);
        } finally {
          disposeFeeds(feeds);
        }
      });
    },
  };
}

async function runDenoiser(
  session: ort.InferenceSession,
  history: Float32Array,
  action: number,
  noisy: Float32Array,
  sigma: number,
): Promise<Float32Array> {
  return enqueueOrt(async () => {
    const feeds = {
      history: tensorF32(history, [1, 12, 64, 64]),
      action: tensorAction(action),
      noisy_frame: tensorF32(noisy, [1, 3, 64, 64]),
      sigma: tensorF32(Float32Array.of(sigma), [1]),
    };
    try {
      const out = await session.run(feeds);
      return copyFloat(out.denoised);
    } finally {
      disposeFeeds(feeds);
    }
  });
}

function tensorF32(data: Float32Array, dims: number[]): ort.Tensor {
  return new ort.Tensor("float32", data, dims);
}

function tensorAction(action: number): ort.Tensor {
  return new ort.Tensor("int64", BigInt64Array.of(BigInt(action)), [1]);
}

function copyFloat(tensor: ort.Tensor | undefined): Float32Array {
  if (!tensor) throw new Error("ONNX output was missing");
  const data = tensor.data;
  if (data instanceof Float32Array) return new Float32Array(data);
  if (ArrayBuffer.isView(data)) return Float32Array.from(data as ArrayLike<number>);
  throw new Error("ONNX output was not a numeric tensor");
}

function disposeFeeds(feeds: Record<string, ort.Tensor>): void {
  for (const tensor of Object.values(feeds)) tensor.dispose();
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function isWasmAssetFailure(message: string): boolean {
  return /ort-wasm|wasm asset|Failed to fetch|dynamically imported module|WebAssembly is not|Aborted|both async and sync fetching of the wasm/i.test(
    message,
  );
}
