import { FRAME, FRAME_FLOATS } from "../sampler/edm";

export function rgbToChw(rgb: Uint8Array, offset = 0, width = FRAME, height = FRAME): Float32Array {
  const plane = width * height;
  const out = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    const base = offset + i * 3;
    out[i] = rgb[base] / 127.5 - 1;
    out[plane + i] = rgb[base + 1] / 127.5 - 1;
    out[2 * plane + i] = rgb[base + 2] / 127.5 - 1;
  }
  return out;
}

export function chwToRgba(chw: Float32Array, width = FRAME, height = FRAME): Uint8ClampedArray {
  const plane = width * height;
  const out = new Uint8ClampedArray(plane * 4);
  for (let i = 0; i < plane; i++) {
    out[i * 4] = toByte(chw[i]);
    out[i * 4 + 1] = toByte(chw[plane + i]);
    out[i * 4 + 2] = toByte(chw[2 * plane + i]);
    out[i * 4 + 3] = 255;
  }
  return out;
}

export function rgbToRgba(rgb: Uint8Array, offset: number, width: number, height: number): Uint8ClampedArray {
  const plane = width * height;
  const out = new Uint8ClampedArray(plane * 4);
  for (let i = 0; i < plane; i++) {
    const base = offset + i * 3;
    out[i * 4] = rgb[base];
    out[i * 4 + 1] = rgb[base + 1];
    out[i * 4 + 2] = rgb[base + 2];
    out[i * 4 + 3] = 255;
  }
  return out;
}

function toByte(value: number): number {
  const byte = Math.round((value + 1) * 127.5);
  if (byte < 0) return 0;
  if (byte > 255) return 255;
  return byte;
}

export function blackFrame(): Float32Array {
  const frame = new Float32Array(FRAME_FLOATS);
  frame.fill(-1);
  return frame;
}

export function repeatFrame(frame: Float32Array, count = 4): Float32Array[] {
  const copies: Float32Array[] = [];
  for (let i = 0; i < count; i++) copies.push(frame.slice());
  return copies;
}

/** Mean absolute error after mapping both CHW tensors from [-1, 1] into [0, 1]. */
export function mae01(left: Float32Array, right: Float32Array): number {
  const count = Math.min(left.length, right.length);
  if (count === 0) return 0;
  let sum = 0;
  for (let i = 0; i < count; i++) {
    sum += Math.abs(unit(left[i]) - unit(right[i]));
  }
  return sum / count;
}

/** MAE in [0, 1] between a CHW dream and an HWC uint8 frame. */
export function maeDreamRgb(
  dream: Float32Array,
  rgb: Uint8Array,
  offset: number,
  width: number,
  height: number,
): number {
  const plane = width * height;
  let sum = 0;
  for (let i = 0; i < plane; i++) {
    const base = offset + i * 3;
    for (let channel = 0; channel < 3; channel++) {
      const dreamed = unit(dream[channel * plane + i]);
      sum += Math.abs(dreamed - rgb[base + channel] / 255);
    }
  }
  return sum / (plane * 3);
}

function unit(value: number): number {
  const scaled = (value + 1) * 0.5;
  if (scaled < 0) return 0;
  if (scaled > 1) return 1;
  return scaled;
}

export function softmaxExpectedReward(logits: ArrayLike<number>): number {
  const values = [-1, 0, 1];
  let max = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < 3; i++) max = Math.max(max, logits[i]);
  let sum = 0;
  const weights = [0, 0, 0];
  for (let i = 0; i < 3; i++) {
    weights[i] = Math.exp(logits[i] - max);
    sum += weights[i];
  }
  let expected = 0;
  for (let i = 0; i < 3; i++) expected += (weights[i] / sum) * values[i];
  return expected;
}

export function doneFromLogit(logit: number): boolean {
  return logit > 0;
}
