# Model card: Dream Arcade world model

## What this is

An action-conditioned diffusion model that predicts the next 64×64 RGB frame of Pong from four past frames and one discrete action. A separate network predicts reward in {−1, 0, +1} and whether the episode ended. A third network is a PPO actor trained inside the diffusion model.

The default shipped weights, if `web/public/models/` is empty or only contains a smoke export, are **not** a trained world model. `dreamarcade smoke` writes a few-step checkpoint so the browser can execute the graph. Do not quote its frames, its loss, or its scores as results.

## How to use it

```bash
dreamarcade play checkpoint=outputs/v2/latest.pt
dreamarcade export denoiser_checkpoint=outputs/v2/latest.pt
```

The ONNX denoiser takes `history` float32 `[1,12,64,64]`, `action` int64 `[1]`, `noisy_frame` float32 `[1,3,64,64]`, and `sigma` float32 `[1]`. It returns `denoised`. Sampling is the Euler method in `src/dreamarcade/models/world.py` and `web/src/sampler/edm.ts`.

## Training data

Two environments share the interface:

- **Original Pong**, written in this repository. No third-party game assets. This is the default.
- **ALE Pong** (`ALE/Pong-v5`, frameskip 4, RGB resized to 64×64), optional. Frames come from the Arcade Learning Environment. This repository does not redistribute ROMs or those frames. If you train on ALE and publish weights, say so here and keep the raw frames private.

Collection mixes a competent policy (ε=0.1), a uniform random policy, and a sloppy policy (ε=0.4), at 40/30/30 per episode. The competent policy is a PPO expert trained in this repo, or a scripted tracker if that checkpoint is absent. It is not the Stable-Baselines3 zoo agent, whose observations do not match these frames.

Loss reweighting boosts pixels that changed since the previous frame, and the sampler oversamples paddle contacts and scoring frames. History frames are noised slightly during training.

## Metrics

Filled in by `dreamarcade eval` after a real run. Until then every cell is unmeasured.

| Metric | Value | Notes |
|---|---|---|
| LPIPS at horizon 10 / 50 / 100 | unmeasured | Needs the lpips package |
| PSNR / SSIM / MAE drift | unmeasured | See `RESULTS.md` |
| Action-effect accuracy | unmeasured | Paddle detector |
| Single-ball fraction | unmeasured | Blob detector |
| Bounce plausibility | unmeasured | 20 degree tolerance |
| Reward precision / recall | unmeasured | Per class |
| Real-game return of the dream PPO | unmeasured | Mean, std, seeds |
| Human-normalized score | unmeasured | Atari Pong only |

## Limits

- 64×64 is intentional. This is not a photorealistic generator.
- A few Euler steps cannot match a fully sampled diffusion model. The step slider is the quality/latency trade.
- Free-running rollouts drift. Noise on the history, scheduled sampling on the baseline, and the re-ground button are mitigations, not a cure.
- The agent can exploit mistakes in the dream. The real-game evaluation is the number that counts.
- The ball is a few pixels. An unweighted loss ignores it. The weighting and the detectors exist because of that.
- Browser speed depends on WebGPU. WASM is the fallback and will be slower. If neither provider loads, the page says so.

## License

Code: MIT. Weights: same repository license, with the training-data note above. Atari name and ROMs remain the property of their owners. Do not take this model card as permission to redistribute them.
