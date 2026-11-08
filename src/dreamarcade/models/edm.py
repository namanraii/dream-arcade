"""EDM preconditioning, noise sampling, and the Euler sampler.

Formulas follow Karras et al., 2022, "Elucidating the Design Space of
Diffusion-Based Generative Models". The browser sampler in web/src/sampler/edm.ts
must implement the same sigma schedule and Euler step.
"""

from __future__ import annotations

import math

import torch


def precondition(
    sigma: torch.Tensor, sigma_data: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return c_skip, c_out, c_in, c_noise for each sigma.

    sigma is broadcastable. c_noise is returned with the same rank as sigma,
    equal to ln(sigma) / 4.
    """
    sigma = sigma.float()
    sd2 = sigma_data ** 2
    c_skip = sd2 / (sigma ** 2 + sd2)
    c_out = sigma * sigma_data / torch.sqrt(sigma ** 2 + sd2)
    c_in = 1.0 / torch.sqrt(sigma ** 2 + sd2)
    c_noise = torch.log(sigma.clamp_min(1e-12)) / 4.0
    return c_skip, c_out, c_in, c_noise


def loss_weight(sigma: torch.Tensor, sigma_data: float = 0.5) -> torch.Tensor:
    """λ(σ) = (σ² + σ_data²) / (σ · σ_data)²."""
    sigma = sigma.float()
    return (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2


def sample_sigmas(
    batch: int,
    p_mean: float = -1.2,
    p_std: float = 1.2,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Log-normal noise levels. ln(σ) ~ N(p_mean, p_std²)."""
    rnd = torch.randn(batch, device=device, generator=generator)
    return (rnd * p_std + p_mean).exp()


def sigma_schedule(
    n_steps: int,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Karras schedule of length n_steps + 1, ending at 0.

    Index 0 is sigma_max. Index n_steps - 1 is sigma_min. The last entry is 0.
    n_steps == 1 yields [sigma_max, 0], matching torch.linspace(0, 1, 1) == [0].
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    ramp = torch.linspace(0, 1, n_steps, device=device)
    min_inv = sigma_min ** (1.0 / rho)
    max_inv = sigma_max ** (1.0 / rho)
    sigmas = (max_inv + ramp * (min_inv - max_inv)) ** rho
    return torch.cat([sigmas, sigmas.new_zeros(1)])


def euler_step(
    x: torch.Tensor, denoised: torch.Tensor, sigma: torch.Tensor | float, sigma_next: torch.Tensor | float
) -> torch.Tensor:
    """One probability-flow Euler step from sigma to sigma_next."""
    sigma_t = torch.as_tensor(sigma, device=x.device, dtype=x.dtype)
    sigma_n = torch.as_tensor(sigma_next, device=x.device, dtype=x.dtype)
    while sigma_t.ndim < x.ndim:
        sigma_t = sigma_t.view(*sigma_t.shape, *([1] * (x.ndim - sigma_t.ndim)))
    while sigma_n.ndim < x.ndim:
        sigma_n = sigma_n.view(*sigma_n.shape, *([1] * (x.ndim - sigma_n.ndim)))
    direction = (x - denoised) / sigma_t.clamp_min(1e-12)
    return x + (sigma_n - sigma_t) * direction


def sinusoidal_embedding(values: torch.Tensor, dim: int) -> torch.Tensor:
    """Fixed sinusoidal embedding of a 1-D tensor. Output is [B, dim]."""
    if dim % 2 != 0:
        raise ValueError("embedding dim must be even")
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=values.device, dtype=torch.float32) / half
    )
    args = values.float().reshape(-1, 1) * freqs.unsqueeze(0)
    return torch.cat([args.sin(), args.cos()], dim=-1)
