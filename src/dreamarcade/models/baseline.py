"""One-step world model. Action conditioning uses the same adaptive group norm."""

from __future__ import annotations

import torch
from torch import nn

from dreamarcade.models.unet import ConditionalUNet


class BaselineWorldModel(nn.Module):
    def __init__(
        self,
        history_frames: int = 4,
        num_actions: int = 6,
        base_channels: int = 64,
        channel_mult: tuple[int, ...] = (1, 2, 2, 2),
        attn_resolutions: tuple[int, ...] = (16, 8),
        cond_dim: int = 256,
        image_size: int = 64,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.history_frames = history_frames
        self.num_actions = num_actions
        self.residual = residual
        self.action_emb = nn.Embedding(num_actions, cond_dim)
        self.unet = ConditionalUNet(
            in_channels=history_frames * 3,
            out_channels=3,
            base_channels=base_channels,
            channel_mult=tuple(channel_mult),
            attn_resolutions=tuple(attn_resolutions),
            cond_dim=cond_dim,
            image_size=image_size,
        )

    def forward(self, history: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        cond = self.action_emb(action.long())
        delta = self.unet(history, cond)
        if self.residual:
            last = history[:, -3:]
            return last + delta
        return delta
