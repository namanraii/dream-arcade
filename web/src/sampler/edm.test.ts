import { describe, expect, it } from "vitest";
import { dreamNext, eulerStep, packHistory, sigmaSchedule } from "./edm";
import { mulberry32 } from "./rng";

describe("sigmaSchedule", () => {
  it("uses one step as [sigmaMax, 0]", () => {
    const sigmas = sigmaSchedule(1);
    expect(sigmas[0]).toBe(80);
    expect(sigmas[sigmas.length - 1]).toBe(0);
    expect(sigmas).toHaveLength(2);
  });

  it("uses two steps as [80, 0.002, 0]", () => {
    const sigmas = sigmaSchedule(2);
    const expected = [80, 0.002, 0];
    expect(sigmas).toHaveLength(expected.length);
    for (let i = 0; i < expected.length; i++) {
      expect(Math.abs(sigmas[i] - expected[i])).toBeLessThanOrEqual(1e-9);
    }
    expect(sigmas[1]).toBeCloseTo(0.002, 12);
  });

  it("puts sigmaMin in the third slot of a three-step schedule", () => {
    const sigmas = sigmaSchedule(3);
    expect(sigmas).toHaveLength(4);
    expect(sigmas[0]).toBe(80);
    expect(sigmas[sigmas.length - 1]).toBe(0);
    expect(sigmas[2]).toBe(0.002);
  });
});

describe("eulerStep", () => {
  it("steps the documented three-element case", () => {
    const next = eulerStep(Float32Array.of(1, 2, 3), Float32Array.of(0, 0, 0), 2, 1);
    expect(Array.from(next)).toEqual([0.5, 1, 1.5]);
  });
});

describe("packHistory", () => {
  const frames = [
    Float32Array.of(1, 0, 0),
    Float32Array.of(2, 0, 0),
    Float32Array.of(3, 0, 0),
    Float32Array.of(4, 0, 0),
  ];

  it("concatenates four frames oldest-first", () => {
    expect(Array.from(packHistory(frames, 4))).toEqual([1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0]);
  });

  it("repeats the most recent frame when length is 1", () => {
    expect(Array.from(packHistory(frames, 1))).toEqual([4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0]);
  });
});

describe("dreamNext", () => {
  it("lands on the denoiser output when the schedule ends at 0 and churn is 0", async () => {
    const zeros = new Float32Array(3 * 64 * 64);
    const frame = await dreamNext({
      history: new Float32Array(12 * 64 * 64),
      nSteps: 1,
      churn: 0,
      historyNoise: 0,
      rng: mulberry32(1),
      denoise: async (_noisy, sigma) => {
        expect(sigma).toBe(80);
        return zeros;
      },
    });
    expect(frame.every((value) => value === 0)).toBe(true);
  });

  it("sends history noise to the denoiser and keeps the caller's buffer", async () => {
    const history = new Float32Array(12 * 64 * 64);
    history[0] = 0.25;
    let seen: Float32Array | null = null;
    await dreamNext({
      history,
      nSteps: 1,
      churn: 0,
      historyNoise: 0.05,
      rng: mulberry32(2),
      denoise: async (_noisy, sigma, conditioned) => {
        expect(sigma).toBe(80);
        seen = conditioned;
        return new Float32Array(3 * 64 * 64);
      },
    });
    expect(seen).not.toBeNull();
    expect(seen).not.toBe(history);
    expect(history[0]).toBe(0.25);
    expect(seen![0]).not.toBe(0.25);
  });
});

