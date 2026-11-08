import type { Game } from "../store";

const PONG_LABELS = ["NOOP", "FIRE", "UP", "DOWN", "UPFIRE", "DOWNFIRE"] as const;
const BREAKOUT_LABELS = ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"] as const;

const held = { up: false, down: false, left: false, right: false, fire: false };
let game: Game = "pong";

export function setActionGame(next: Game): void {
  game = next;
}

export function currentAction(): number {
  if (game === "breakout") {
    if (held.right && !held.left) return 2;
    if (held.left && !held.right) return 3;
    if (held.fire) return 1;
    return 0;
  }
  if (held.up && !held.down) return 2;
  if (held.down && !held.up) return 3;
  if (held.fire) return 1;
  return 0;
}

export function setHeld(key: keyof typeof held, value: boolean): void {
  held[key] = value;
}

export function clearHeld(): void {
  held.up = false;
  held.down = false;
  held.left = false;
  held.right = false;
  held.fire = false;
}

export function keyToHold(code: string): keyof typeof held | null {
  if (code === "ArrowUp") return "up";
  if (code === "ArrowDown") return "down";
  if (code === "ArrowLeft") return "left";
  if (code === "ArrowRight") return "right";
  if (code === "Space" || code === "KeyF") return "fire";
  return null;
}

export function actionLabel(action: number, which: Game = game): string {
  const labels = which === "breakout" ? BREAKOUT_LABELS : PONG_LABELS;
  return labels[action] ?? String(action);
}
