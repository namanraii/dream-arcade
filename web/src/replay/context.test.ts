import { describe, expect, it } from "vitest";
import { contextFrames } from "./context";
import type { Replay } from "./format";

function replayWithPixels(pixels: number[]): Replay {
  const frames = new Uint8Array(pixels);
  return {
    T: pixels.length / 3,
    H: 1,
    W: 1,
    actions: new Uint8Array(pixels.length / 3),
    rewards: new Int8Array(pixels.length / 3),
    frames,
  };
}

describe("contextFrames", () => {
  const initial = Float32Array.of(-1, -1, -1);

  it("does not include the frame being predicted", () => {
    const replay = replayWithPixels([
      10, 0, 0,
      20, 0, 0,
      30, 0, 0,
      40, 0, 0,
    ]);
    const context = contextFrames(replay, 2, initial);
    expect(context).toHaveLength(4);
    expect(Array.from(context[0])).toEqual(Array.from(initial));
    expect(Array.from(context[1])).toEqual(Array.from(initial));
    expect(context[2][0]).toBeCloseTo(10 / 127.5 - 1);
    expect(context[3][0]).toBeCloseTo(20 / 127.5 - 1);
    expect(context.some((frame) => Math.abs(frame[0] - (30 / 127.5 - 1)) < 1e-6)).toBe(false);
  });
});
