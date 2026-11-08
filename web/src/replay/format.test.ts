import { describe, expect, it } from "vitest";
import { parseReplay } from "./format";

function tinyReplay(): ArrayBuffer {
  const T = 1;
  const H = 1;
  const W = 1;
  const buffer = new ArrayBuffer(16 + T + T + T * H * W * 3);
  const view = new DataView(buffer);
  view.setUint8(0, "P".charCodeAt(0));
  view.setUint8(1, "O".charCodeAt(0));
  view.setUint8(2, "N".charCodeAt(0));
  view.setUint8(3, "G".charCodeAt(0));
  view.setUint32(4, T, true);
  view.setUint32(8, H, true);
  view.setUint32(12, W, true);
  const bytes = new Uint8Array(buffer);
  bytes[16] = 2;
  new Int8Array(buffer, 17, 1)[0] = -1;
  bytes[18] = 10;
  bytes[19] = 20;
  bytes[20] = 30;
  return buffer;
}

describe("parseReplay", () => {
  it("reads a 1x1 frame, action, and signed reward", () => {
    const replay = parseReplay(tinyReplay());
    expect(replay.T).toBe(1);
    expect(replay.H).toBe(1);
    expect(replay.W).toBe(1);
    expect(Array.from(replay.actions)).toEqual([2]);
    expect(Array.from(replay.rewards)).toEqual([-1]);
    expect(Array.from(replay.frames)).toEqual([10, 20, 30]);
  });

  it("accepts a breakout replay", () => {
    const buffer = tinyReplay();
    const bytes = new Uint8Array(buffer);
    bytes[0] = "B".charCodeAt(0);
    bytes[1] = "R".charCodeAt(0);
    bytes[2] = "C".charCodeAt(0);
    bytes[3] = "K".charCodeAt(0);
    expect(parseReplay(buffer).T).toBe(1);
  });

  it("rejects a huge frame count before allocating pixels", () => {
    const buffer = tinyReplay();
    new DataView(buffer).setUint32(4, 500_000, true);
    expect(() => parseReplay(buffer)).toThrow(/out of range/);
  });

  it("rejects a bad magic", () => {
    const buffer = tinyReplay();
    new Uint8Array(buffer)[0] = "X".charCodeAt(0);
    expect(() => parseReplay(buffer)).toThrow(/magic/);
  });
});
