"""Teacher-forced and free-running rollouts along a recorded action sequence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from dreamarcade.eval.metrics import lpips_distance, mae, psnr, ssim
from dreamarcade.models.baseline import BaselineWorldModel
from dreamarcade.models.world import EDMWorldModel, sample_next_frame
from dreamarcade.utils import model_to_uint8, uint8_to_model


def _predict(model, kind: str, history: torch.Tensor, action: torch.Tensor, steps: int) -> torch.Tensor:
    if kind == "v1":
        return model(history, action).clamp(-1, 1)
    return sample_next_frame(model, history, action, steps=steps, history_noise=0.0, churn=0.0)


def rollout_episode(
    model: BaselineWorldModel | EDMWorldModel,
    kind: str,
    frames: np.ndarray,
    actions: np.ndarray,
    history: int = 4,
    sampler_steps: int = 3,
    horizons: tuple[int, ...] = (1, 5, 10, 25, 50, 100),
    device: torch.device | None = None,
) -> dict:
    """frames includes the initial observation, so len(frames) == len(actions) + 1."""
    device = device or torch.device("cpu")
    model.eval()
    steps = int(actions.shape[0])
    teacher = []
    free_frames = [uint8_to_model(frames[t]) for t in range(history)]
    records: dict[int, list] = {h: [] for h in horizons}
    with torch.no_grad():
        for t in range(history - 1, steps):
            real_hist = torch.cat(
                [uint8_to_model(frames[i]) for i in range(t - history + 1, t + 1)], dim=0
            ).unsqueeze(0).to(device)
            action = torch.tensor([int(actions[t])], device=device)
            pred = _predict(model, kind, real_hist, action, sampler_steps)
            teacher.append(model_to_uint8(pred[0]))
            if t == history - 1:
                dream_hist = real_hist.clone()
            dream = _predict(model, kind, dream_hist, action, sampler_steps)
            free_frames.append(dream[0].cpu())
            dream_hist = torch.cat([dream_hist[:, 3:], dream], dim=1)
            horizon = t - (history - 1) + 1
            if horizon in records:
                records[horizon].append(model_to_uint8(dream[0]))
    teacher_metrics = _against_real(teacher, frames[history:], device)
    drift = {}
    for horizon, preds in records.items():
        if not preds:
            continue
        target = frames[history - 1 + horizon]
        # The stored prediction is the frame at index `history - 1 + horizon`
        # which is frames[t+1] with t = history - 2 + horizon, so index
        # history - 1 + horizon. Yes.
        drift[str(horizon)] = _pair(preds[-1], target, device)
    return {"teacher_forced": teacher_metrics, "drift": drift, "teacher_frames": teacher}


def _pair(pred: np.ndarray, target: np.ndarray, device: torch.device) -> dict:
    row = {"psnr": psnr(pred, target), "ssim": ssim(pred, target), "mae": mae(pred, target)}
    distance = lpips_distance(pred, target, device)
    if distance is not None:
        row["lpips"] = distance
    return row


def _against_real(preds: list[np.ndarray], targets: np.ndarray, device: torch.device) -> dict:
    count = min(len(preds), len(targets))
    rows = [_pair(preds[i], targets[i], device) for i in range(count)]
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def load_episode(path: Path) -> tuple[np.ndarray, np.ndarray]:
    blob = np.load(path)
    return blob["frames"], blob["actions"]
