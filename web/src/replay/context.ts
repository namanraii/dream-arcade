import { rgbToChw } from "../render/pixels";
import { frameOffset, type Replay } from "./format";

/** Four frames strictly before `index`. Missing slots repeat the reset frame. */
export function contextFrames(replay: Replay, index: number, initial: Float32Array): Float32Array[] {
  const frames: Float32Array[] = [];
  for (let slot = 0; slot < 4; slot++) {
    const src = index - 4 + slot;
    if (src < 0) frames.push(initial.slice());
    else frames.push(rgbToChw(replay.frames, frameOffset(replay, src), replay.W, replay.H));
  }
  return frames;
}
