"""Small conditional U-Net used by the baseline and the diffusion denoiser.

Conditioning enters through adaptive group norm. Self-attention uses explicit
matmul so the graph exports to ONNX.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def groups_for(channels: int, prefer: int = 8) -> int:
    groups = min(prefer, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return max(groups, 1)


def heads_for(channels: int, prefer: int = 4) -> int:
    heads = min(prefer, channels)
    while heads > 1 and channels % heads != 0:
        heads -= 1
    return max(heads, 1)


class AdaGN(nn.Module):
    def __init__(self, channels: int, cond_dim: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(groups_for(channels), channels, affine=False)
        self.to_scale_shift = nn.Linear(cond_dim, channels * 2)
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.to_scale_shift(cond).chunk(2, dim=-1)
        x = self.norm(x)
        return x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int) -> None:
        super().__init__()
        self.norm1 = AdaGN(in_channels, cond_dim)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = AdaGN(out_channels, cond_dim)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(x, cond)))
        hidden = self.conv2(F.silu(self.norm2(hidden, cond)))
        return hidden + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.heads = heads_for(channels)
        self.norm = nn.GroupNorm(groups_for(channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(batch, 3, self.heads, channels // self.heads, height * width)
        query, key, value = qkv.unbind(dim=1)
        query = query.permute(0, 1, 3, 2)
        key = key.permute(0, 1, 3, 2)
        value = value.permute(0, 1, 3, 2)
        scale = (channels // self.heads) ** -0.5
        weights = torch.softmax(query @ key.transpose(-1, -2) * scale, dim=-1)
        pooled = (weights @ value).permute(0, 1, 3, 2).reshape(batch, channels, height, width)
        return x + self.proj(pooled)


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ConditionalUNet(nn.Module):
    """U-Net whose input is already concatenated conditioning + image."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 64,
        channel_mult: tuple[int, ...] = (1, 2, 2, 2),
        attn_resolutions: tuple[int, ...] = (16, 8),
        cond_dim: int = 256,
        image_size: int = 64,
    ) -> None:
        super().__init__()
        self.cond_dim = cond_dim
        self.image_size = image_size
        widths = [base_channels * mult for mult in channel_mult]
        self.stem = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.encoders = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.downs = nn.ModuleList()
        channels = base_channels
        resolution = image_size
        skip_channels: list[int] = []
        for level, width in enumerate(widths):
            self.encoders.append(ResBlock(channels, width, cond_dim))
            self.attentions.append(
                SelfAttention(width) if resolution in attn_resolutions else nn.Identity()
            )
            channels = width
            skip_channels.append(channels)
            if level < len(widths) - 1:
                self.downs.append(Downsample(channels))
                resolution //= 2
            else:
                self.downs.append(nn.Identity())

        self.mid1 = ResBlock(channels, channels, cond_dim)
        self.mid_attn = (
            SelfAttention(channels) if resolution in attn_resolutions else nn.Identity()
        )
        self.mid2 = ResBlock(channels, channels, cond_dim)

        self.decoders = nn.ModuleList()
        self.dec_attentions = nn.ModuleList()
        self.ups = nn.ModuleList()
        for level, width in reversed(list(enumerate(widths))):
            skip = skip_channels.pop()
            self.decoders.append(ResBlock(channels + skip, width, cond_dim))
            self.dec_attentions.append(
                SelfAttention(width) if (image_size // (2 ** level)) in attn_resolutions else nn.Identity()
            )
            channels = width
            if level > 0:
                self.ups.append(Upsample(channels))
            else:
                self.ups.append(nn.Identity())

        self.out_norm = nn.GroupNorm(groups_for(channels), channels)
        self.out = nn.Conv2d(channels, out_channels, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        hidden = self.stem(x)
        skips: list[torch.Tensor] = []
        for encoder, attention, down in zip(self.encoders, self.attentions, self.downs, strict=True):
            hidden = attention(encoder(hidden, cond))
            skips.append(hidden)
            hidden = down(hidden)
        hidden = self.mid1(hidden, cond)
        hidden = self.mid_attn(hidden)
        hidden = self.mid2(hidden, cond)
        for decoder, attention, up in zip(self.decoders, self.dec_attentions, self.ups, strict=True):
            hidden = torch.cat([hidden, skips.pop()], dim=1)
            hidden = attention(decoder(hidden, cond))
            hidden = up(hidden)
        return self.out(F.silu(self.out_norm(hidden)))
