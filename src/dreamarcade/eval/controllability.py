"""Does the commanded action move the paddle the way it does in the real game?"""

from __future__ import annotations

import numpy as np
import torch

from dreamarcade.data.detectors import detect_paddle
from dreamarcade.data.env import make_env
from dreamarcade.models.world import sample_next_frame
from dreamarcade.utils import model_to_uint8, uint8_to_model


def _sign(delta: float, deadzone: float = 0.5) -> int:
    if delta > deadzone:
        return 1
    if delta < -deadzone:
        return -1
    return 0


@torch.no_grad()
def controllability(
    model,
    kind: str,
    env_name: str,
    trials: int = 16,
    hold: int = 4,
    sampler_steps: int = 1,
    seed: int = 0,
    device: torch.device | None = None,
) -> dict:
    """Hold UP and DOWN from the same start. Compare the sign of paddle motion.

    For Pong, action 2 is up (y decreases) and action 3 is down (y increases).
    For Breakout, action 2 is right and action 3 is left, and the detector
    returns x.
    """
    device = device or next(model.parameters()).device
    model.eval()
    agreements = []
    real_moved = 0
    for trial in range(trials):
        for action in (2, 3):
            env = make_env(env_name, seed=seed + trial, max_steps=hold + 2)
            obs, _info = env.reset()
            side = getattr(env, "paddle_side", "right")
            real_obs = obs
            history = torch.cat([uint8_to_model(obs) for _ in range(4)], dim=0).unsqueeze(0).to(device)
            start_real = detect_paddle(real_obs, side=side)
            dream_frame = real_obs
            for _ in range(hold):
                real_obs, _r, done, _info = env.step(action)
                action_t = torch.tensor([action], device=device)
                if kind == "v1":
                    pred = model(history, action_t).clamp(-1, 1)
                else:
                    pred = sample_next_frame(model, history, action_t, steps=sampler_steps)
                dream_frame = model_to_uint8(pred[0])
                history = torch.cat([history[:, 3:], pred], dim=1)
                if done:
                    break
            if hasattr(env, "close"):
                env.close()
            end_real = detect_paddle(real_obs, side=side)
            end_dream = detect_paddle(dream_frame, side=side)
            if start_real is None or end_real is None or end_dream is None:
                continue
            # Positive delta means "moved in the commanded direction".
            # Pong y grows downward, so UP (action 2) decreases y.
            real_delta = end_real - start_real
            dream_delta = end_dream - start_real
            if side != "bottom":
                if action == 2:
                    real_delta = -real_delta
                    dream_delta = -dream_delta
            elif action == 3:
                real_delta = -real_delta
                dream_delta = -dream_delta
            if _sign(real_delta) == 0:
                continue
            real_moved += 1
            agreements.append(_sign(real_delta) == _sign(dream_delta))
    accuracy = float(np.mean(agreements)) if agreements else 0.0
    return {
        "action_effect_accuracy": accuracy,
        "compared": len(agreements),
        "real_moved": real_moved,
    }
