export interface Replay {
  T: number;
  H: number;
  W: number;
  actions: Uint8Array;
  rewards: Int8Array;
  frames: Uint8Array;
}

const MAGICS = new Set(["PONG", "BRCK"]);

/** Parse a little-endian pong.bin buffer. Frames are RGB, frame-major, row-major. */
export function parseReplay(buffer: ArrayBuffer): Replay {
  if (buffer.byteLength < 16) {
    throw new Error(`replay is shorter than the 16-byte header (${buffer.byteLength} bytes)`);
  }
  const view = new DataView(buffer);
  const magic = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
  if (!MAGICS.has(magic)) {
    throw new Error(`bad replay magic: expected PONG or BRCK, got ${JSON.stringify(magic)}`);
  }
  const T = view.getUint32(4, true);
  const H = view.getUint32(8, true);
  const W = view.getUint32(12, true);
  if (T < 1 || H < 1 || W < 1 || T > 20_000 || H > 256 || W > 256) {
    throw new Error(`replay header is out of range (T=${T}, H=${H}, W=${W})`);
  }
  const pixels = T * H * W * 3;
  const need = 16 + T + T + pixels;
  if (buffer.byteLength < need) {
    throw new Error(`replay truncated: have ${buffer.byteLength} bytes, need ${need}`);
  }
  const bytes = new Uint8Array(buffer);
  const actions = bytes.slice(16, 16 + T);
  const rewards = new Int8Array(T);
  rewards.set(new Int8Array(buffer, 16 + T, T));
  const frames = bytes.slice(16 + T + T, 16 + T + T + pixels);
  return { T, H, W, actions, rewards, frames };
}

export function frameByteLength(replay: Replay): number {
  return replay.H * replay.W * 3;
}

export function frameOffset(replay: Replay, index: number): number {
  return index * frameByteLength(replay);
}
