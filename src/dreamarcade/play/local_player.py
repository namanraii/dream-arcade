"""Keyboard-controlled dream in a local window.

The window target is 5+ frames per second on an M2 for the model that was
exported. If pygame cannot open a display, frames are written to a gif instead.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig

from dreamarcade.data.env import make_env
from dreamarcade.models.world import sample_next_frame
from dreamarcade.train.build import build_baseline, build_diffusion
from dreamarcade.utils import load_checkpoint, model_to_uint8, pick_device, set_seed, uint8_to_model


def _load(cfg: DictConfig, device: torch.device):
    kind = str(cfg.model.kind)
    if kind == "v1":
        model = build_baseline(cfg).to(device)
    else:
        model = build_diffusion(cfg).to(device)
    load_checkpoint(Path(cfg.checkpoint), model, use_ema=True)
    model.eval()
    return model, kind


@torch.no_grad()
def play(cfg: DictConfig) -> Path | None:
    set_seed(int(cfg.seed))
    device = pick_device(str(cfg.device))
    model, kind = _load(cfg, device)
    env = make_env(str(cfg.data.name), seed=int(cfg.seed), max_steps=int(cfg.steps))
    frame, _info = env.reset()
    history = torch.cat([uint8_to_model(frame) for _ in range(4)], dim=0).unsqueeze(0).to(device)
    if hasattr(env, "close"):
        env.close()
    steps = int(cfg.steps)
    sampler_steps = int(cfg.sampler_steps)
    frames = [frame]
    try:
        return _pygame_loop(model, kind, history, frames, steps, sampler_steps, device, cfg)
    except Exception as exc:
        print(f"window unavailable ({exc}); writing a gif")
        return _gif(model, kind, history, frames, steps, sampler_steps, cfg)


def _step_model(model, kind, history, action: int, sampler_steps: int, device):
    action_t = torch.tensor([action], device=device)
    if kind == "v1":
        pred = model(history, action_t).clamp(-1, 1)
    else:
        pred = sample_next_frame(
            model,
            history,
            action_t,
            steps=sampler_steps,
            history_noise=float(0.05),
        )
    nxt = torch.cat([history[:, 3:], pred], dim=1)
    return nxt, model_to_uint8(pred[0])


def _pygame_loop(model, kind, history, frames, steps, sampler_steps, device, cfg) -> None:
    import os
    import time

    import pygame

    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        # macOS can still open a window. Other headless hosts cannot.
        if not pygame.display.get_init():
            pass
    pygame.init()
    pygame.display.set_caption("Dream Arcade")
    surface = pygame.display.set_mode((512, 512))
    clock = pygame.time.Clock()
    action = 0
    ticks = 0
    started = time.perf_counter()
    running = True
    while running and ticks < steps:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
        keys = pygame.key.get_pressed()
        action = 0
        horizontal = "breakout" in str(cfg.data.name)
        if horizontal and keys[pygame.K_RIGHT]:
            action = 2
        elif horizontal and keys[pygame.K_LEFT]:
            action = 3
        elif not horizontal and keys[pygame.K_UP]:
            action = 2
        elif not horizontal and keys[pygame.K_DOWN]:
            action = 3
        elif keys[pygame.K_SPACE]:
            action = 1
        history, frame = _step_model(model, kind, history, action, sampler_steps, device)
        frames.append(frame)
        rgb = np.repeat(np.repeat(frame, 8, axis=0), 8, axis=1)
        surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
        surface.blit(surf, (0, 0))
        pygame.display.flip()
        ticks += 1
        clock.tick(60)
    elapsed = max(time.perf_counter() - started, 1e-6)
    print(f"local player {ticks / elapsed:.2f} model ticks/s over {ticks} ticks")
    pygame.quit()
    return None


def _gif(model, kind, history, frames, steps, sampler_steps, cfg) -> Path:
    import imageio.v2 as imageio

    device = history.device
    for _ in range(steps):
        history, frame = _step_model(model, kind, history, 2, sampler_steps, device)
        frames.append(frame)
    path = Path(cfg.out_dir) / "play.gif"
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, duration=0.1)
    print(f"wrote {path}")
    return path
