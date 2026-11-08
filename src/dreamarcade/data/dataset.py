"""Random-access windows over collected episodes.

A sample at time t (the action index) is:
  history  = frames[t-L+1 : t+1]   shape [L, 3, 64, 64] in [-1, 1]
  action   = actions[t]
  target   = frames[t+1]
  reward, done, weight align with that action
Horizon > 1 also returns the following actions and targets for scheduled sampling.
Windows never cross an episode boundary.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from dreamarcade.utils import uint8_to_model


def pixel_loss_weight(history: torch.Tensor, target: torch.Tensor, alpha: float = 4.0) -> torch.Tensor:
    """Boost pixels that moved between the last context frame and the target.

    history is [B, 12, H, W] or [12, H, W]. The ball is a few pixels, so an
    unweighted loss ignores it.
    """
    last = history[:, -3:] if history.ndim == 4 else history[-3:]
    if history.ndim == 3:
        diff = (target - last).abs().mean(dim=0, keepdim=True)
        scale = diff.mean().clamp_min(1e-3)
        return 1.0 + alpha * (diff / scale)
    diff = (target - last).abs().mean(dim=1, keepdim=True)
    scale = diff.mean(dim=(2, 3), keepdim=True).clamp_min(1e-3)
    return 1.0 + alpha * (diff / scale)


class FrameDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str = "train",
        history: int = 4,
        horizon: int = 1,
    ) -> None:
        self.root = Path(root)
        self.history = history
        self.horizon = horizon
        manifest = json.loads((self.root / "manifest.json").read_text())
        self.episodes: list[dict] = []
        self.index: list[tuple[int, int]] = []
        self.sample_weights: list[float] = []
        for episode in manifest["episodes"]:
            if split != "all" and episode["split"] != split:
                continue
            blob = np.load(self.root / "shards" / episode["file"])
            stored = {
                "frames": blob["frames"],
                "actions": blob["actions"],
                "rewards": blob["rewards"],
                "dones": blob["dones"],
                "weights": blob["weights"],
            }
            episode_index = len(self.episodes)
            self.episodes.append(stored)
            steps = int(stored["actions"].shape[0])
            # Need `history` frames ending at t inclusive, so t >= history - 1,
            # and `horizon` future frames, so t + horizon <= steps.
            first = history - 1
            last = steps - horizon
            for t in range(first, last + 1):
                if stored["dones"][t : t + horizon - 1].any():
                    continue
                self.index.append((episode_index, t))
                self.sample_weights.append(float(stored["weights"][t : t + horizon].max()))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        episode_index, t = self.index[item]
        episode = self.episodes[episode_index]
        hist_np = episode["frames"][t - self.history + 1 : t + 1]
        history = torch.stack([uint8_to_model(frame) for frame in hist_np], dim=0)
        history = history.reshape(-1, 64, 64)
        targets = []
        for step in range(self.horizon):
            targets.append(uint8_to_model(episode["frames"][t + 1 + step]))
        target = torch.stack(targets, dim=0)
        action = torch.tensor(episode["actions"][t : t + self.horizon].astype(np.int64))
        reward = torch.tensor(episode["rewards"][t : t + self.horizon].astype(np.float32))
        done = torch.tensor(episode["dones"][t : t + self.horizon].astype(np.float32))
        weight = torch.tensor(episode["weights"][t : t + self.horizon].astype(np.float32))
        return {
            "history": history,
            "action": action,
            "target": target,
            "reward": reward,
            "done": done,
            "weight": weight,
        }


def make_loader(
    root: Path,
    split: str,
    batch_size: int,
    history: int = 4,
    horizon: int = 1,
    balanced: bool = False,
    num_workers: int = 0,
    seed: int = 0,
) -> tuple[FrameDataset, torch.utils.data.DataLoader]:
    dataset = FrameDataset(root, split=split, history=history, horizon=horizon)
    sampler = None
    shuffle = True
    if balanced:
        if split != "train":
            raise ValueError("balanced sampling is for the training split")
        generator = torch.Generator()
        generator.manual_seed(seed)
        sampler = WeightedRandomSampler(
            weights=torch.tensor(dataset.sample_weights, dtype=torch.double),
            num_samples=max(len(dataset), 1),
            replacement=True,
            generator=generator,
        )
        shuffle = False
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=False,
    )
    return dataset, loader
