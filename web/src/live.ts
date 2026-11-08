import type { RgbView } from "./render/crt";

export type LiveStatus = "idle" | "loading" | "ready" | "error";

export interface LiveState {
  version: number;
  status: LiveStatus;
  errorTitle: string;
  errorBody: string;
  provider: string | null;
  modelFile: string | null;
  rewardLoaded: boolean;
  playFrame: Float32Array | null;
  dreamFrame: Float32Array | null;
  realFrame: RgbView | null;
  fps: number;
  ticksPerSec: number;
  lastTickMs: number;
  stepMs: number;
  score: number;
  predictedReward: number | null;
  done: boolean | null;
  action: number;
  maeSeries: number[];
  disagreement: number | null;
  replayIndex: number;
  replayFrames: number;
  reGround: boolean;
  restartClip: boolean;
}

export const live: LiveState = {
  version: 0,
  status: "idle",
  errorTitle: "",
  errorBody: "",
  provider: null,
  modelFile: null,
  rewardLoaded: false,
  playFrame: null,
  dreamFrame: null,
  realFrame: null,
  fps: 0,
  ticksPerSec: 0,
  lastTickMs: 0,
  stepMs: 0,
  score: 0,
  predictedReward: null,
  done: null,
  action: 0,
  maeSeries: [],
  disagreement: null,
  replayIndex: 0,
  replayFrames: 0,
  reGround: false,
  restartClip: false,
};

const listeners = new Set<() => void>();

export function publish(): void {
  live.version += 1;
  for (const listener of listeners) listener();
}

export function subscribeLive(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function resetLive(): void {
  live.status = "loading";
  live.errorTitle = "";
  live.errorBody = "";
  live.provider = null;
  live.modelFile = null;
  live.rewardLoaded = false;
  live.playFrame = null;
  live.dreamFrame = null;
  live.realFrame = null;
  live.fps = 0;
  live.ticksPerSec = 0;
  live.lastTickMs = 0;
  live.stepMs = 0;
  live.score = 0;
  live.predictedReward = null;
  live.done = null;
  live.action = 0;
  live.maeSeries = [];
  live.disagreement = null;
  live.replayIndex = 0;
  live.replayFrames = 0;
  live.reGround = false;
  live.restartClip = false;
  publish();
}

export function failLive(title: string, body: string): void {
  live.status = "error";
  live.errorTitle = title;
  live.errorBody = body;
  live.playFrame = null;
  live.dreamFrame = null;
  publish();
}
