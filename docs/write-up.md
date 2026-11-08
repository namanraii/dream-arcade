# Dream Arcade: what was built, and what failed

This is the technical note that sits next to the demo. Sections that need a trained model are marked unmeasured. Fill them from `RESULTS.md` after the Kaggle runs. Do not invent the numbers.

## The claim

An action-conditioned diffusion model of Pong can be trained on free GPUs, evaluated for drift and controllability, and played in a browser because the sampler is not inside the exported graph.

## Setup

- Observations are 64×64 RGB. Four frames of history. Six discrete actions, embedded.
- v1 predicts the next frame with a U-Net and weighted L1, plus LPIPS when the weights are available. Context frames are noised. A second rollout step is trained after warmup (scheduled sampling).
- v2 is an EDM denoiser. See `docs/design-note.md` for the coefficients. Training uses mixed precision on CUDA, AdamW, cosine decay with warmup, gradient clipping, and an EMA. Checkpoints are frequent enough to survive a 12 hour session limit. Resume is `resume=<ckpt>`.
- The reward head is a CNN plus a GRU over the four frames. It is trained separately, with class-balanced sampling, because scoring events are rare.
- The browser exports only the denoiser. Euler runs in TypeScript. WebGPU first, WASM otherwise.

## What the comparison is for

v1 exists so "the diffusion model looks like Pong" is not the result. The result is the drift curve of v2 against v1 at the same horizons, on the same held-out episodes. Ablations, when they are run, are sampler steps {1, 2, 3, 5}, history length {1, 2, 4}, history noise on or off, and pixel reweighting on or off. Each of those is a config override and a row in `RESULTS.md`, not a screenshot.

## What failed, and why

Write this after the runs. The failures the plan already expects:

1. **Drift.** The model is autoregressive on its own pixels. Small errors in the ball position become a second ball, or none. History noise and scheduled sampling (v1) reduce this. They do not remove it. The coherence horizon is whatever the curve says, including a number under the 100-step target.
2. **The ball is small.** A uniform L1 loss is dominated by the black background. The pixel weight is proportional to the frame difference. If the ball still vanishes, the weighting was not enough and the sample of contact frames was too small. The detector is how we notice, early.
3. **Actions can be ignored.** A model can paint a plausible rally that does not respond to the paddle. Controllability is the sign agreement of paddle motion under a held action. A high visual score with low controllability is a failed model.
4. **Few-step sampling.** Three Euler steps from σ = 80 is a rough ODE solve. More steps look better and miss the frame budget. The slider is the experiment. Distillation buys a step back and must state the L1 gap.
5. **Reward hacking in the dream.** PPO maximises the reward head, not the game. If the head fires on a blob that is not a score, the agent farms it. The real-environment mean and std, over 20 episodes and at least three seeds, is the published score. The dream return is a diagnostic.
6. **Session limits.** A run that does not checkpoint inside 15 minutes loses the week. The resume path is part of the method, not an inconvenience.
7. **ONNX drift.** If the parity test fails, the browser is a different model. The test is `dreamarcade parity` and `tests/test_train_and_onnx.py`.

## Browser

Target is 12 model ticks per second on a recent laptop with WebGPU, with the canvas drawing held frames at 60 Hz through a nearest-neighbour upscale and a scanline shader. If the full U-Net misses that, the honest response is fewer channels or two sampler steps, stated as a quality trade, not a silent config change.

## Agent

The dream is reset from a real history window so the episode does not start from noise. The reward the agent trains on is the expected value of the three-class head. Done is the sigmoid of the termination logit. Evaluation copies the actor into the real environment.

## Reproduce

Commands, dataset sizes, and the Kaggle quota notes are in the README. One config file per reported number. Seeds are arguments.
