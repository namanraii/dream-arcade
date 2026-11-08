import { randn } from "./rng";

export const SIGMA_MIN = 0.002;
export const SIGMA_MAX = 80;
export const RHO = 7;
export const FRAME = 64;
export const FRAME_FLOATS = 3 * FRAME * FRAME;
export const HISTORY_SLOTS = 4;

/**
 * Karras sigma schedule. Length is nSteps + 1 and the last entry is 0.
 * Index 0 is sigmaMax. The entry before 0 is sigmaMin (nSteps === 1 yields [sigmaMax, 0]).
 * EDM c_skip, c_out, c_in, and c_noise stay inside the ONNX graph.
 */
export function sigmaSchedule(
  nSteps: number,
  sigmaMin = SIGMA_MIN,
  sigmaMax = SIGMA_MAX,
  rho = RHO,
): number[] {
  if (nSteps < 1 || !Number.isFinite(nSteps)) {
    throw new Error("nSteps must be >= 1");
  }
  const minInv = sigmaMin ** (1 / rho);
  const maxInv = sigmaMax ** (1 / rho);
  const sigmas: number[] = [];
  for (let i = 0; i < nSteps; i++) {
    const t = nSteps === 1 ? 0 : i / (nSteps - 1);
    if (t === 0) sigmas.push(sigmaMax);
    else if (t === 1) sigmas.push(sigmaMin);
    else sigmas.push((maxInv + t * (minInv - maxInv)) ** rho);
  }
  sigmas.push(0);
  return sigmas;
}

/** Probability-flow Euler step. Arrays are packed CHW float32. */
export function eulerStep(
  x: Float32Array,
  denoised: Float32Array,
  sigma: number,
  sigmaNext: number,
): Float32Array {
  const out = new Float32Array(x.length);
  const span = sigmaNext - sigma;
  for (let i = 0; i < x.length; i++) {
    const direction = (x[i] - denoised[i]) / sigma;
    out[i] = x[i] + span * direction;
  }
  return out;
}

/**
 * Pack up to four RGB frames into 12 channels, oldest first.
 * When `length` is under 4, the most recent `length` frames are kept and the
 * oldest of that window is repeated on the left until there are four slots.
 */
export function packHistory(frames: Float32Array[], length: number): Float32Array {
  if (frames.length === 0) {
    throw new Error("packHistory needs at least one frame");
  }
  const usable = Math.max(1, Math.min(HISTORY_SLOTS, frames.length, Math.floor(length)));
  const recent = frames.slice(frames.length - usable);
  const oldest = recent[0];
  const frameLen = oldest.length;
  const out = new Float32Array(HISTORY_SLOTS * frameLen);
  const pad = HISTORY_SLOTS - usable;
  for (let i = 0; i < HISTORY_SLOTS; i++) {
    const src = i < pad ? oldest : recent[i - pad];
    out.set(src, i * frameLen);
  }
  return out;
}

export type DenoiseFn = (
  noisy: Float32Array,
  sigma: number,
  history: Float32Array,
) => Promise<Float32Array>;

export interface DreamOptions {
  history: Float32Array;
  nSteps: number;
  churn: number;
  historyNoise: number;
  rng: () => number;
  denoise: DenoiseFn;
  noiseScale?: number;
  sigmaMin?: number;
  sigmaMax?: number;
  rho?: number;
  onStep?: (elapsedMs: number) => void;
}

/**
 * Euler sampler matching dreamarcade.models.world.sample_next_frame.
 * `churn` is the sampling-noise slider. noiseScale stays 1 unless a caller overrides it.
 * History noise is N(0, historyNoise²), applied once before the loop.
 */
export async function dreamNext(options: DreamOptions): Promise<Float32Array> {
  const nSteps = Math.min(5, Math.max(1, Math.round(options.nSteps)));
  const churn = Math.min(1, Math.max(0, options.churn));
  const historyNoise = Math.min(0.2, Math.max(0, options.historyNoise));
  const noiseScale = options.noiseScale ?? 1;
  const sigmas = sigmaSchedule(nSteps, options.sigmaMin, options.sigmaMax, options.rho);
  const history = historyNoise > 0 ? noiseCopy(options.history, historyNoise, options.rng) : options.history;

  let x = new Float32Array(FRAME_FLOATS);
  const sigma0 = sigmas[0];
  for (let i = 0; i < FRAME_FLOATS; i++) {
    x[i] = randn(options.rng) * sigma0 * noiseScale;
  }

  for (let index = 0; index < nSteps; index++) {
    const sigma = sigmas[index];
    const sigmaNext = sigmas[index + 1];
    let sigmaHat = sigma;
    if (sigma > 0 && churn > 0) {
      const gamma = Math.min(churn / nSteps, Math.SQRT2 - 1);
      sigmaHat = sigma * (1 + gamma);
      const extra = Math.sqrt(sigmaHat * sigmaHat - sigma * sigma);
      for (let i = 0; i < x.length; i++) {
        x[i] += extra * randn(options.rng);
      }
    }
    const started = options.onStep ? performance.now() : 0;
    const denoised = await options.denoise(x, sigmaHat, history);
    if (options.onStep) options.onStep(performance.now() - started);
    x = new Float32Array(eulerStep(x, denoised, sigmaHat, sigmaNext));
  }

  for (let i = 0; i < x.length; i++) {
    const value = x[i];
    if (value < -1) x[i] = -1;
    else if (value > 1) x[i] = 1;
  }
  return x;
}

function noiseCopy(history: Float32Array, std: number, rng: () => number): Float32Array {
  const out = new Float32Array(history.length);
  for (let i = 0; i < history.length; i++) {
    out[i] = history[i] + std * randn(rng);
  }
  return out;
}
