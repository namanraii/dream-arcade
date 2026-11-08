"""Shared actor-critic for the real-env expert and the dream PPO agent."""

from __future__ import annotations

import torch
from torch import nn


class ActorCritic(nn.Module):
    def __init__(self, in_channels: int = 12, n_actions: int = 6) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.actor = nn.Linear(1024, n_actions)
        self.critic = nn.Linear(1024, 1)

    def forward(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.features(history)
        return self.actor(encoded), self.critic(encoded).squeeze(-1)

    def act(self, history: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(history)
        dist = torch.distributions.Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), value
