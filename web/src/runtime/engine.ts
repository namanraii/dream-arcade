import { clearHeld, currentAction, setActionGame } from "../input/actions";
import { mae01, maeDreamRgb, repeatFrame, rgbToChw, softmaxExpectedReward, doneFromLogit, blackFrame } from "../render/pixels";
import { contextFrames } from "../replay/context";
import { frameOffset, parseReplay, type Replay } from "../replay/format";
import { dreamNext, packHistory, type DenoiseFn } from "../sampler/edm";
import { mulberry32 } from "../sampler/rng";
import { readGame, readKnobs, type Game, type Mode } from "../store";
import { failLive, live, publish, resetLive } from "../live";
import { modelMissing, replayMissing } from "../ui/messages";
import { loadDenoiser, loadPolicy, loadReward, type Denoiser, type PolicyHead, type RewardHead } from "./session";

let generation = 0;
let reGround = false;

export function requestReGround(): void {
  reGround = true;
  live.reGround = true;
  publish();
}

export function stopEngine(): void {
  generation += 1;
  clearHeld();
}

export async function startEngine(mode: Mode): Promise<void> {
  const token = ++generation;
  const dead = () => token !== generation;
  const game = readGame();
  setActionGame(game);
  resetLive();
  if (mode === "landing") {
    live.status = "idle";
    publish();
    return;
  }
  try {
    const denoiser = await loadDenoiser(game);
    if (dead()) return;
    live.provider = denoiser.provider;
    live.modelFile = denoiser.file;
    const reward = await loadReward(game).catch(() => null);
    if (dead()) return;
    live.rewardLoaded = reward !== null;
    const policy = mode === "agent" ? await loadPolicy(game) : null;
    if (dead()) return;
    if (mode === "agent" && !policy) {
      const copy = modelMissing(game === "breakout" ? "breakout/policy.onnx" : "policy.onnx");
      failLive(copy.title, copy.body);
      return;
    }
    const initial = await loadInitial(game);
    if (dead()) return;
    if (mode === "lab") {
      const replay = await loadReplay(game);
      if (dead()) return;
      live.replayFrames = replay.T;
      live.status = "ready";
      publish();
      await runLab(denoiser, reward, replay, initial, dead);
      return;
    }
    live.status = "ready";
    publish();
    await runDream(denoiser, reward, policy, initial, mode, dead);
  } catch (error) {
    if (dead()) return;
    const text = error instanceof Error ? error.message : String(error);
    const [title, ...rest] = text.split("\n");
    failLive(title || "The model failed to start", rest.join("\n"));
  }
}

async function runDream(
  denoiser: Denoiser,
  reward: RewardHead | null,
  policy: PolicyHead | null,
  initial: Float32Array,
  mode: Mode,
  dead: () => boolean,
): Promise<void> {
  const queue = repeatFrame(initial, 4);
  let rng = mulberry32(mode === "agent" ? 7 : (Date.now() & 0xfffffff));
  const ticks: number[] = [];
  while (!dead()) {
    const started = performance.now();
    const knobs = readKnobs();
    if (reGround) {
      queue.splice(0, queue.length, ...repeatFrame(initial, 4));
      reGround = false;
      live.reGround = false;
    }
    const history = packHistory(queue, knobs.historyLength);
    const action = mode === "agent" && policy ? argmax(await policy.run(history)) : currentAction();
    if (dead()) return;
    const frame = await sample(denoiser, history, action, knobs, rng);
    if (dead()) return;
    queue.push(frame);
    if (queue.length > 4) queue.shift();
    live.playFrame = frame;
    live.action = action;
    await applyReward(reward, history, action);
    const now = performance.now();
    live.lastTickMs = now - started;
    ticks.push(now);
    while (ticks.length && now - ticks[0] > 1000) ticks.shift();
    live.ticksPerSec = ticks.length;
    publish();
  }
}

async function runLab(
  denoiser: Denoiser,
  reward: RewardHead | null,
  replay: Replay,
  initial: Float32Array,
  dead: () => boolean,
): Promise<void> {
  const queue = repeatFrame(initial, 4);
  let index = 0;
  let diagnostic = 0;
  const ticks: number[] = [];
  while (!dead()) {
    const knobs = readKnobs();
    if (knobs.paused) {
      await sleep(40);
      continue;
    }
    const started = performance.now();
    const action = replay.actions[index] ?? 0;
    const realOffset = frameOffset(replay, index);
    live.realFrame = { data: replay.frames, width: replay.W, height: replay.H, offset: realOffset };
    live.replayIndex = index;
    const context = contextFrames(replay, index, initial);
    if (reGround || !knobs.freeRun) {
      queue.splice(0, queue.length, ...context.map((frame) => frame.slice()));
      reGround = false;
      live.reGround = false;
    }
    const history = packHistory(knobs.freeRun ? queue : context, knobs.historyLength);
    const rng = mulberry32(1000 + index * 17 + knobs.nSteps);
    const frame = await sample(denoiser, history, action, knobs, rng);
    if (dead()) return;
    if (knobs.freeRun) {
      queue.push(frame);
      if (queue.length > 4) queue.shift();
    }
    live.dreamFrame = frame;
    live.playFrame = frame;
    live.action = action;
    const error = maeDreamRgb(frame, replay.frames, realOffset, replay.W, replay.H);
    live.maeSeries = [...live.maeSeries, error].slice(-240);
    diagnostic += 1;
    if (diagnostic % 4 === 0) {
      const left = await sample(denoiser, history, action, knobs, mulberry32(index + 3));
      const right = await sample(denoiser, history, action, knobs, mulberry32(index + 11));
      if (dead()) return;
      live.disagreement = mae01(left, right);
    }
    await applyReward(reward, history, action);
    const now = performance.now();
    live.lastTickMs = now - started;
    ticks.push(now);
    while (ticks.length && now - ticks[0] > 1000) ticks.shift();
    live.ticksPerSec = ticks.length;
    publish();
    index = (index + 1) % replay.T;
  }
}

async function sample(
  denoiser: Denoiser,
  history: Float32Array,
  action: number,
  knobs: ReturnType<typeof readKnobs>,
  rng: () => number,
): Promise<Float32Array> {
  let stepMs = 0;
  let steps = 0;
  const denoise: DenoiseFn = async (noisy, sigma, conditioned) => {
    const t0 = performance.now();
    const out = await denoiser.run(conditioned, action, noisy, sigma);
    stepMs += performance.now() - t0;
    steps += 1;
    return out;
  };
  const frame = await dreamNext({
    history,
    nSteps: knobs.nSteps,
    churn: knobs.churn,
    historyNoise: knobs.historyNoise,
    rng,
    denoise,
  });
  live.stepMs = steps ? stepMs / steps : 0;
  return frame;
}

async function applyReward(reward: RewardHead | null, history: Float32Array, action: number): Promise<void> {
  if (!reward) {
    live.predictedReward = null;
    live.done = null;
    return;
  }
  const out = await reward.run(history, action);
  const expected = softmaxExpectedReward(out.logits);
  live.predictedReward = expected;
  live.score += expected;
  live.done = doneFromLogit(out.doneLogit);
}

function argmax(logits: Float32Array): number {
  let best = 0;
  for (let i = 1; i < logits.length; i++) {
    if (logits[i] > logits[best]) best = i;
  }
  return best;
}

async function loadInitial(game: Game): Promise<Float32Array> {
  const file = game === "breakout" ? "breakout-initial.bin" : "initial.bin";
  const response = await fetch(`/replays/${file}`);
  if (!response.ok) return blackFrame();
  const bytes = new Uint8Array(await response.arrayBuffer());
  if (bytes.byteLength !== 64 * 64 * 3) return blackFrame();
  return rgbToChw(bytes);
}

async function loadReplay(game: Game): Promise<Replay> {
  const file = game === "breakout" ? "breakout.bin" : "pong.bin";
  const response = await fetch(`/replays/${file}`);
  if (!response.ok) {
    const copy = replayMissing(file);
    throw new Error(`${copy.title}\n${copy.body}`);
  }
  return parseReplay(await response.arrayBuffer());
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
