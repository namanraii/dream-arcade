"""Collection policies.

The 40/30/30 mix is expert, uniform random, and a sloppy expert. The expert is
a policy checkpoint trained in this repo when one exists, otherwise a scripted
player that tracks the ball. Stable-Baselines3 zoo weights are not applied to
these 64x64 frames: those agents were trained on stacked 84x84 grayscale.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from dreamarcade.data.detectors import detect_balls, detect_paddle


class RandomPolicy:
    def __init__(self, n_actions: int = 6, seed: int = 0) -> None:
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray, env) -> int:
        return int(self.rng.integers(0, self.n_actions))


class EpsilonPolicy:
    def __init__(self, base, epsilon: float, seed: int = 0, n_actions: int = 6) -> None:
        self.base = base
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray, env) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))
        return int(self.base.act(obs, env))


class ScriptedPolicy:
    """Tracks the ball. Uses simulator state when the env exposes it."""

    def __init__(self, side: str = "right") -> None:
        self.side = side

    def act(self, obs: np.ndarray, env) -> int:
        semantics = getattr(env, "action_semantics", "pong_vertical")
        if semantics == "breakout_horizontal":
            return self._horizontal(obs, env)
        return self._vertical(obs, env)

    def _vertical(self, obs: np.ndarray, env) -> int:
        if hasattr(env, "agent_y") and hasattr(env, "ball_y"):
            center = env.agent_y + getattr(env, "paddle_h", 12) / 2
            ball = env.ball_y
            if getattr(env, "waiting", False):
                return 1
            if ball < center - 1.5:
                return 2
            if ball > center + 1.5:
                return 3
            return 0
        paddle = detect_paddle(obs, side=getattr(env, "paddle_side", "right"))
        balls = detect_balls(obs)
        if paddle is None or not balls:
            return 1
        ball_y = balls[0][0]
        if ball_y < paddle - 1:
            return 2
        if ball_y > paddle + 1:
            return 3
        return 0

    def _horizontal(self, obs: np.ndarray, env) -> int:
        if hasattr(env, "paddle_x") and hasattr(env, "ball_x"):
            center = env.paddle_x + getattr(env, "paddle_w", 12) / 2
            if getattr(env, "waiting", False):
                return 1
            if env.ball_x < center - 1.5:
                return 3
            if env.ball_x > center + 1.5:
                return 2
            return 0
        paddle = detect_paddle(obs, side="bottom")
        balls = detect_balls(obs, ignore_columns=(0, 64))
        if paddle is None or not balls:
            return 1
        ball_x = balls[0][1]
        if ball_x < paddle - 1:
            return 3
        if ball_x > paddle + 1:
            return 2
        return 0


class TorchPolicy:
    """Actor checkpoint produced by dreamarcade train-expert or ppo."""

    def __init__(self, path: Path, device: torch.device | None = None) -> None:
        from dreamarcade.train.actor import ActorCritic

        self.device = device or torch.device("cpu")
        self.net = ActorCritic()
        blob = torch.load(path, map_location="cpu", weights_only=False)
        state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        self.net.load_state_dict(state)
        self.net.to(self.device).eval()
        self.stack: list[torch.Tensor] = []

    def reset(self) -> None:
        self.stack = []

    @torch.no_grad()
    def act(self, obs: np.ndarray, env) -> int:
        from dreamarcade.utils import uint8_to_model

        frame = uint8_to_model(obs)
        if not self.stack:
            self.stack = [frame.clone() for _ in range(4)]
        else:
            self.stack.pop(0)
            self.stack.append(frame)
        history = torch.cat(self.stack, dim=0).unsqueeze(0).to(self.device)
        logits, _value = self.net(history)
        return int(torch.argmax(logits, dim=-1).item())


def mix_policies(seed: int, expert_path: Path | None = None, side: str = "right"):
    """Return (name, policy) factories for one episode draw.

    Probabilities: 40% expert with ε=0.1, 30% uniform, 30% sloppy ε=0.4.
    """
    rng = np.random.default_rng(seed)
    draw = float(rng.random())
    scripted = ScriptedPolicy(side=side)
    if expert_path is not None and Path(expert_path).is_file():
        base = TorchPolicy(Path(expert_path))
        kind_expert = "checkpoint"
    else:
        base = scripted
        kind_expert = "scripted"
    if draw < 0.4:
        return f"expert-{kind_expert}", EpsilonPolicy(base, epsilon=0.1, seed=seed)
    if draw < 0.7:
        return "random", RandomPolicy(seed=seed)
    return f"sloppy-{kind_expert}", EpsilonPolicy(scripted, epsilon=0.4, seed=seed + 1)
