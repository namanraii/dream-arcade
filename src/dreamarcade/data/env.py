"""Pong and Breakout environments.

Original games share the observation contract with Atari: uint8 64x64 RGB and
6 discrete actions. Atari is imported only when requested, so tests and the
original fallback do not need ROMs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

SIZE = 64
N_ACTIONS = 6


def _clip_rect(img: np.ndarray, x: float, y: float, w: int, h: int, value: int) -> None:
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(SIZE, max(0, x0 + w))
    y1 = min(SIZE, max(0, y0 + h))
    x0 = min(SIZE, max(0, x0))
    y0 = min(SIZE, max(0, y0))
    if x1 > x0 and y1 > y0:
        img[y0:y1, x0:x1] = value


class OriginalPong:
    """Single-ball Pong. The agent paddle is on the right. Action 2 is up."""

    physics = "pong"
    paddle_side = "right"
    action_semantics = "pong_vertical"
    action_meanings = ("NOOP", "FIRE", "UP", "DOWN", "UPFIRE", "DOWNFIRE")
    paddle_h = 12
    paddle_w = 2

    def __init__(
        self,
        seed: int = 0,
        max_steps: int = 300,
        size: int = 64,
        opponent_slip: float = 0.2,
    ) -> None:
        if size != 64:
            raise ValueError("original Pong renders 64x64")
        self.max_steps = max_steps
        self.opponent_slip = opponent_slip
        self.rng = np.random.default_rng(seed)
        self.agent_y = 26.0
        self.opp_y = 26.0
        self.ball_x = 31.0
        self.ball_y = 31.0
        self.vx = -1.35
        self.vy = 0.55
        self.waiting = True
        self.t = 0
        self.score_agent = 0
        self.score_opp = 0
        self.last_contact = False
        self.serve_dir = -1.0

    def reset(self) -> tuple[np.ndarray, dict[str, Any]]:
        self.agent_y = 26.0
        self.opp_y = 26.0
        self._park_ball()
        self.t = 0
        self.score_agent = 0
        self.score_opp = 0
        self.last_contact = False
        return self.render(), {"contact": False}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        action = int(action)
        self.t += 1
        self.last_contact = False
        self._move_agent(action)
        self._move_opponent()
        reward = 0.0
        if self.waiting:
            if action in (1, 4, 5):
                self.waiting = False
                self.vx = 1.35 * self.serve_dir
                self.vy = float(self.rng.uniform(-1.0, 1.0))
        else:
            reward = self._move_ball()
        done = self.t >= self.max_steps
        return self.render(), reward, done, {"contact": self.last_contact}

    def render(self) -> np.ndarray:
        img = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
        img[::4, 31:33] = (36, 36, 36)
        _clip_rect(img, 60, self.agent_y, self.paddle_w, self.paddle_h, 255)
        _clip_rect(img, 2, self.opp_y, self.paddle_w, self.paddle_h, 255)
        _clip_rect(img, self.ball_x, self.ball_y, 2, 2, 255)
        return img

    def _move_agent(self, action: int) -> None:
        if action in (2, 4):
            self.agent_y -= 2.4
        elif action in (3, 5):
            self.agent_y += 2.4
        self.agent_y = float(np.clip(self.agent_y, 1, SIZE - self.paddle_h - 1))

    def _move_opponent(self) -> None:
        if self.rng.random() < self.opponent_slip:
            return
        target = self.ball_y - self.paddle_h / 2
        delta = float(np.clip(target - self.opp_y, -1.6, 1.6))
        self.opp_y = float(np.clip(self.opp_y + delta, 1, SIZE - self.paddle_h - 1))

    def _park_ball(self) -> None:
        self.ball_x = 31.0
        self.ball_y = 31.0
        self.vx = 0.0
        self.vy = 0.0
        self.waiting = True

    def _hits(self, px: float, py: float) -> bool:
        return (
            self.ball_x < px + self.paddle_w
            and self.ball_x + 2 > px
            and self.ball_y < py + self.paddle_h
            and self.ball_y + 2 > py
        )

    def _move_ball(self) -> float:
        self.ball_x += self.vx
        self.ball_y += self.vy
        if self.ball_y < 1:
            self.ball_y = 1
            self.vy = abs(self.vy) if self.vy != 0 else 0.4
        elif self.ball_y > SIZE - 3:
            self.ball_y = SIZE - 3
            self.vy = -abs(self.vy) if self.vy != 0 else -0.4
        if self.vx > 0 and self._hits(60, self.agent_y):
            self._bounce(60, self.agent_y, direction=-1)
        elif self.vx < 0 and self._hits(2, self.opp_y):
            self._bounce(2, self.opp_y, direction=1)
        # Agent paddle is on the right: exiting left is the agent's point.
        if self.ball_x < -1:
            self.score_agent += 1
            self.serve_dir = 1.0
            self._park_ball()
            return 1.0
        if self.ball_x > SIZE:
            self.score_opp += 1
            self.serve_dir = -1.0
            self._park_ball()
            return -1.0
        return 0.0

    def _bounce(self, px: float, py: float, direction: int) -> None:
        offset = (self.ball_y + 1) - (py + self.paddle_h / 2)
        self.vx = direction * max(1.1, abs(self.vx))
        self.vy = float(np.clip(self.vy + offset * 0.12, -1.8, 1.8))
        if direction < 0:
            self.ball_x = px - 2
        else:
            self.ball_x = px + self.paddle_w
        self.last_contact = True


class OriginalBreakout:
    """Stretch-goal environment. Action 2 moves the bottom paddle right."""

    physics = "breakout"
    paddle_side = "bottom"
    action_semantics = "breakout_horizontal"
    action_meanings = ("NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE")

    def __init__(self, seed: int = 0, max_steps: int = 400, size: int = 64) -> None:
        if size != 64:
            raise ValueError("original Breakout renders 64x64")
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.paddle_x = 27.0
        self.paddle_w = 12
        self.rows = 4
        self.cols = 8
        self.bricks = np.ones((self.rows, self.cols), dtype=np.uint8)
        self.ball_x = 32.0
        self.ball_y = 40.0
        self.vx = 1.1
        self.vy = -1.1
        self.waiting = True
        self.t = 0
        self.lives = 3
        self.last_contact = False

    def reset(self) -> tuple[np.ndarray, dict[str, Any]]:
        self.paddle_x = 27.0
        self.bricks[:] = 1
        self.lives = 3
        self.t = 0
        self._park()
        return self.render(), {"contact": False}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        action = int(action)
        self.t += 1
        self.last_contact = False
        if action in (2, 4):
            self.paddle_x += 2.6
        elif action in (3, 5):
            self.paddle_x -= 2.6
        self.paddle_x = float(np.clip(self.paddle_x, 1, SIZE - self.paddle_w - 1))
        reward = 0.0
        if self.waiting:
            self.ball_x = self.paddle_x + self.paddle_w / 2
            self.ball_y = 54.0
            if action in (1, 4, 5):
                self.waiting = False
                self.vx = float(self.rng.choice([-1.15, 1.15]))
                self.vy = -1.2
        else:
            reward = self._move_ball()
        done = self.t >= self.max_steps or self.lives <= 0 or int(self.bricks.sum()) == 0
        return self.render(), reward, done, {"contact": self.last_contact}

    def render(self) -> np.ndarray:
        img = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
        colors = (
            (180, 60, 60),
            (180, 140, 40),
            (40, 160, 80),
            (50, 90, 180),
        )
        for row in range(self.rows):
            for col in range(self.cols):
                if not self.bricks[row, col]:
                    continue
                x = col * 8
                y = 8 + row * 4
                img[y : y + 3, x : x + 7] = colors[row]
        _clip_rect(img, self.paddle_x, 58, self.paddle_w, 2, 255)
        _clip_rect(img, self.ball_x, self.ball_y, 2, 2, 255)
        return img

    def _park(self) -> None:
        self.waiting = True
        self.ball_x = self.paddle_x + 4
        self.ball_y = 54.0
        self.vx = 0.0
        self.vy = 0.0

    def _move_ball(self) -> float:
        self.ball_x += self.vx
        self.ball_y += self.vy
        reward = 0.0
        if self.ball_x < 1:
            self.ball_x = 1
            self.vx = abs(self.vx)
        elif self.ball_x > SIZE - 3:
            self.ball_x = SIZE - 3
            self.vx = -abs(self.vx)
        if self.ball_y < 1:
            self.ball_y = 1
            self.vy = abs(self.vy)
        reward += self._hit_bricks()
        paddle_top = 58
        if (
            self.vy > 0
            and self.ball_y + 2 >= paddle_top
            and self.ball_y <= paddle_top + 2
            and self.ball_x + 2 >= self.paddle_x
            and self.ball_x <= self.paddle_x + self.paddle_w
        ):
            self.vy = -abs(self.vy)
            self.ball_y = paddle_top - 2
            offset = (self.ball_x + 1) - (self.paddle_x + self.paddle_w / 2)
            self.vx = float(np.clip(self.vx + offset * 0.08, -1.8, 1.8))
            self.last_contact = True
        if self.ball_y > SIZE:
            self.lives -= 1
            self._park()
            return reward - 1.0
        return reward

    def _hit_bricks(self) -> float:
        gained = 0.0
        bx = int(np.clip(self.ball_x + 1, 0, SIZE - 1))
        by = int(np.clip(self.ball_y + 1, 0, SIZE - 1))
        if by < 8 or by >= 8 + self.rows * 4:
            return 0.0
        row = (by - 8) // 4
        col = min(self.cols - 1, bx // 8)
        if 0 <= row < self.rows and self.bricks[row, col]:
            self.bricks[row, col] = 0
            self.vy = -self.vy
            gained = 1.0
            self.last_contact = True
        return gained


class AtariPixels:
    """Gymnasium + ale-py, resized to 64x64 RGB. Frameskip defaults to 4."""

    physics = "pong"
    paddle_side = "right"
    action_semantics = "pong_vertical"
    action_meanings = ("NOOP", "FIRE", "UP", "DOWN", "UPFIRE", "DOWNFIRE")

    def __init__(
        self,
        env_id: str = "ALE/Pong-v5",
        seed: int = 0,
        frameskip: int = 4,
        size: int = 64,
        repeat_action_probability: float = 0.25,
    ) -> None:
        import gymnasium as gym

        try:
            import ale_py

            gym.register_envs(ale_py)
        except Exception:
            pass
        self.size = size
        self.env = gym.make(
            env_id,
            obs_type="rgb",
            frameskip=frameskip,
            repeat_action_probability=repeat_action_probability,
            render_mode=None,
        )
        self.env.reset(seed=seed)
        if "Breakout" in env_id:
            self.physics = "breakout"
            self.paddle_side = "bottom"
            self.action_semantics = "breakout_horizontal"
            self.action_meanings = ("NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE")
        self.last_contact = False

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        image = Image.fromarray(frame)
        image = image.resize((self.size, self.size), Image.BILINEAR)
        return np.asarray(image, dtype=np.uint8)

    def reset(self) -> tuple[np.ndarray, dict[str, Any]]:
        obs, _info = self.env.reset()
        return self._resize(obs), {"contact": False}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, _info = self.env.step(int(action))
        return self._resize(obs), float(reward), bool(terminated or truncated), {"contact": False}

    def close(self) -> None:
        self.env.close()


def make_env(name: str, seed: int = 0, **kwargs: Any):
    if name in ("original_pong", "pong"):
        return OriginalPong(seed=seed, **{k: v for k, v in kwargs.items() if k in ("max_steps", "opponent_slip")})
    if name in ("original_breakout", "breakout"):
        return OriginalBreakout(seed=seed, **{k: v for k, v in kwargs.items() if k in ("max_steps",)})
    if name.startswith("ALE/"):
        return AtariPixels(env_id=name, seed=seed, **{k: v for k, v in kwargs.items() if k in ("frameskip", "repeat_action_probability")})
    raise ValueError(f"unknown environment {name}")
