"""Diffusion world model: EDM denoiser conditioned on history and action."""

from __future__ import annotations

import torch
from torch import nn

from dreamarcade.models.edm import (
    euler_step,
    loss_weight,
    precondition,
    sample_sigmas,
    sigma_schedule,
    sinusoidal_embedding,
)
from dreamarcade.models.unet import ConditionalUNet


class EDMWorldModel(nn.Module):
    def __init__(
        self,
        history_frames: int = 4,
        num_actions: int = 6,
        base_channels: int = 64,
        channel_mult: tuple[int, ...] = (1, 2, 2, 2),
        attn_resolutions: tuple[int, ...] = (16, 8),
        cond_dim: int = 256,
        image_size: int = 64,
        sigma_data: float = 0.5,
        p_mean: float = -1.2,
        p_std: float = 1.2,
    ) -> None:
        super().__init__()
        self.history_frames = history_frames
        self.num_actions = num_actions
        self.sigma_data = sigma_data
        self.p_mean = p_mean
        self.p_std = p_std
        self.cond_dim = cond_dim
        self.action_emb = nn.Embedding(num_actions, cond_dim)
        self.noise_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.unet = ConditionalUNet(
            in_channels=history_frames * 3 + 3,
            out_channels=3,
            base_channels=base_channels,
            channel_mult=tuple(channel_mult),
            attn_resolutions=tuple(attn_resolutions),
            cond_dim=cond_dim,
            image_size=image_size,
        )

    def condition(self, action: torch.Tensor, c_noise: torch.Tensor) -> torch.Tensor:
        noise_emb = sinusoidal_embedding(c_noise.reshape(-1), self.cond_dim)
        return self.action_emb(action.long()) + self.noise_mlp(noise_emb)

    def denoise(
        self,
        history: torch.Tensor,
        action: torch.Tensor,
        noisy: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """EDM denoiser D_θ. sigma has shape [B] or [1]."""
        sigma = sigma.reshape(-1).to(dtype=noisy.dtype, device=noisy.device)
        sigma = sigma.expand(noisy.shape[0])
        view = sigma.view(-1, 1, 1, 1)
        c_skip, c_out, c_in, c_noise = precondition(view, self.sigma_data)
        cond = self.condition(action, c_noise.reshape(-1))
        network_in = torch.cat([history, c_in * noisy], dim=1)
        prediction = self.unet(network_in, cond)
        return c_skip * noisy + c_out * prediction

    def loss(
        self,
        history: torch.Tensor,
        action: torch.Tensor,
        target: torch.Tensor,
        pixel_weight: torch.Tensor | None = None,
        history_noise: float = 0.0,
    ) -> torch.Tensor:
        sigma = sample_sigmas(
            target.shape[0], self.p_mean, self.p_std, device=target.device
        )
        noisy = target + torch.randn_like(target) * sigma.view(-1, 1, 1, 1)
        if history_noise > 0:
            history = history + torch.randn_like(history) * history_noise
        denoised = self.denoise(history, action, noisy, sigma)
        error = (denoised - target) ** 2
        if pixel_weight is not None:
            error = error * pixel_weight
        weight = loss_weight(sigma, self.sigma_data).view(-1, 1, 1, 1)
        return (weight * error).mean()


@torch.no_grad()
def sample_next_frame(
    model: EDMWorldModel,
    history: torch.Tensor,
    action: torch.Tensor,
    steps: int = 3,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    churn: float = 0.0,
    history_noise: float = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Euler sampler. churn is the UI sampling-noise slider. 0 is deterministic given the draw of x."""
    sigmas = sigma_schedule(steps, sigma_min, sigma_max, rho, device=history.device)
    if history_noise > 0:
        noise = torch.randn(
            history.shape, device=history.device, dtype=history.dtype, generator=generator
        )
        history = history + history_noise * noise
    x = torch.randn(
        history.shape[0],
        3,
        history.shape[-2],
        history.shape[-1],
        device=history.device,
        dtype=history.dtype,
        generator=generator,
    )
    x = x * sigmas[0]
    for index in range(steps):
        sigma = sigmas[index]
        sigma_next = sigmas[index + 1]
        sigma_hat = sigma
        if churn > 0 and float(sigma) > 0:
            gamma = min(churn / steps, 2**0.5 - 1)
            sigma_hat = sigma * (1 + gamma)
            added = torch.sqrt((sigma_hat**2 - sigma**2).clamp_min(0))
            x = x + added * torch.randn(
                x.shape, device=x.device, dtype=x.dtype, generator=generator
            )
        sigma_batch = sigma_hat.expand(history.shape[0])
        denoised = model.denoise(history, action, x, sigma_batch)
        x = euler_step(x, denoised, sigma_hat, sigma_next)
    return x.clamp(-1, 1)
