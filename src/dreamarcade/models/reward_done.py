"""Reward and termination head. Trained separately from the diffusion model.

Reward classes are {-1, 0, +1}. A GRU reads the four frame embeddings so the
head can see the ball moving, not only the latest pixels.
"""

from __future__ import annotations

import torch
from torch import nn

REWARD_VALUES = (-1.0, 0.0, 1.0)


def reward_to_class(reward: torch.Tensor) -> torch.Tensor:
    classes = torch.ones_like(reward, dtype=torch.long)
    classes = torch.where(reward < 0, torch.zeros_like(classes), classes)
    classes = torch.where(reward > 0, torch.full_like(classes, 2), classes)
    return classes


class RewardDoneNet(nn.Module):
    def __init__(
        self,
        history_frames: int = 4,
        num_actions: int = 6,
        hidden: int = 128,
        use_gru: bool = True,
        image_size: int = 64,
    ) -> None:
        super().__init__()
        if image_size != 64:
            raise ValueError("RewardDoneNet is sized for 64x64 frames")
        self.history_frames = history_frames
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        # 64 -> 15 -> 6, then 64 * 6 * 6.
        self.proj = nn.Linear(64 * 6 * 6, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True) if use_gru else None
        self.action_emb = nn.Embedding(num_actions, hidden)
        self.reward_head = nn.Linear(hidden, 3)
        self.done_head = nn.Linear(hidden, 1)

    def features(self, history: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        batch = history.shape[0]
        frames = history.reshape(batch, self.history_frames, 3, 64, 64)
        flat = frames.reshape(batch * self.history_frames, 3, 64, 64)
        encoded = self.proj(self.encoder(flat)).reshape(batch, self.history_frames, -1)
        if self.gru is None:
            pooled = encoded[:, -1]
        else:
            _out, hidden = self.gru(encoded)
            pooled = hidden[-1]
        return pooled + self.action_emb(action.long())

    def forward(self, history: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.features(history, action)
        return self.reward_head(feat), self.done_head(feat)

    def expected_reward(self, history: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        logits, _done = self.forward(history, action)
        probs = torch.softmax(logits, dim=-1)
        values = history.new_tensor(REWARD_VALUES)
        return (probs * values).sum(dim=-1)
