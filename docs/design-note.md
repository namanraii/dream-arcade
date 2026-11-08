# Design note: EDM world model for Pong

One page, written before the long training runs. Sources: Karras et al. 2022 (EDM) and Alonso et al. 2024 (DIAMOND).

## Why this formulation

A one-step U-Net (v1) is the honest baseline and the first playable demo. It blurs under its own feedback because the training loss never sees its mistakes. Scheduled sampling and noise on the context frames are the cheap mitigations, and the drift curve says how far they get.

v2 trains a denoiser instead of a direct next frame. EDM writes the denoiser as

```
D(x; σ) = c_skip(σ) x + c_out(σ) F_θ(c_in(σ) x; c_noise(σ))
```

with

```
c_skip = σ_data² / (σ² + σ_data²)
c_out  = σ σ_data / sqrt(σ² + σ_data²)
c_in   = 1 / sqrt(σ² + σ_data²)
c_noise = ln(σ) / 4
```

σ_data is 0.5 because frames are scaled to [-1, 1]. The training loss weights each noise level by λ(σ) = (σ² + σ_data²) / (σ σ_data)², and ln(σ) is drawn from N(−1.2, 1.2²). The output convolution of F is initialised at zero, so D starts as c_skip · x, which is the EDM recommendation.

The past four frames are concatenated with the scaled noisy target. The action embedding and c_noise share an adaptive group norm. The action is an embedding, not a one-hot glued onto the pixels. A little noise is added to the history at training time and at inference, which is the DIAMOND trick against compounding error.

## Sampler

Euler, Karras schedule, ρ = 7, σ from 80 down to 0.002, then a last step to 0. Default is 3 steps. The schedule is `torch.linspace(0, 1, n)` so one step is [σ_max, 0] and two steps are [σ_max, σ_min, 0]. The browser implements this same recurrence. Changing the step count does not require a new export, because only F (wrapped as D) is in the ONNX file.

Three Euler steps from σ = 80 is a coarse integral. That is a known quality cost of running in a browser. The lab slider is how we show it. Distillation to one step is the stretch path, and it reports its own speedup and L1 gap rather than claiming the teacher quality for free.

## What we will measure

Free-running LPIPS, PSNR, SSIM, and MAE at horizons 1, 5, 10, 25, 50, 100, against v1 on the same test episodes. Paddle-direction agreement for held actions. Ball count, and bounce angle within 20 degrees of the law of reflection. Reward precision and recall per class. A PPO agent trained only on D and the reward head, scored in the real environment.

## What we expect to go wrong

The ball is tiny, so an unweighted loss paints the background and drops the sprite. Pixel weights and contact oversampling are there from the first run, and the detector fails the run if the ball count collapses. Drift inside a few dozen steps is likely on this compute budget. The re-ground control exists so a demo can reset context without pretending the horizon is longer than the curve. The dream agent will find reward the real game does not pay. We report the real-game gap and do not hide it behind the dream learning curve.
