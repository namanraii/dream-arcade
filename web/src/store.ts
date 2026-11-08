import { create } from "zustand";

export type Mode = "landing" | "play" | "lab" | "agent";
export type Game = "pong" | "breakout";

interface UiState {
  mode: Mode;
  game: Game;
  nSteps: number;
  historyLength: number;
  churn: number;
  historyNoise: number;
  freeRun: boolean;
  paused: boolean;
  setMode: (mode: Mode) => void;
  setGame: (game: Game) => void;
  setNSteps: (value: number) => void;
  setHistoryLength: (value: number) => void;
  setChurn: (value: number) => void;
  setHistoryNoise: (value: number) => void;
  setFreeRun: (value: boolean) => void;
  setPaused: (value: boolean) => void;
}

function clamp(value: number, min: number, max: number): number {
  if (Number.isNaN(value)) return min;
  return Math.min(max, Math.max(min, value));
}

export const useUi = create<UiState>((set) => ({
  mode: "landing",
  game: "pong",
  nSteps: 3,
  historyLength: 4,
  churn: 0,
  historyNoise: 0.05,
  freeRun: false,
  paused: false,
  setMode: (mode) => set({ mode, paused: false }),
  setGame: (game) => set({ game, paused: false }),
  setNSteps: (value) => set({ nSteps: Math.round(clamp(value, 1, 5)) }),
  setHistoryLength: (value) => set({ historyLength: Math.round(clamp(value, 1, 4)) }),
  setChurn: (value) => set({ churn: clamp(value, 0, 1) }),
  setHistoryNoise: (value) => set({ historyNoise: clamp(value, 0, 0.2) }),
  setFreeRun: (freeRun) => set({ freeRun }),
  setPaused: (paused) => set({ paused }),
}));

export function readGame(): Game {
  return useUi.getState().game;
}

export function readKnobs(): {
  nSteps: number;
  historyLength: number;
  churn: number;
  historyNoise: number;
  freeRun: boolean;
  paused: boolean;
} {
  const state = useUi.getState();
  return {
    nSteps: state.nSteps,
    historyLength: state.historyLength,
    churn: state.churn,
    historyNoise: state.historyNoise,
    freeRun: state.freeRun,
    paused: state.paused,
  };
}
