# Results

Numbers in the block below are written by `dreamarcade eval`. Do not type them in by hand. A smoke checkpoint is not a result: it only proves the pipeline runs.

The targets from the plan, still unmeasured until a full training run:

| Metric | Target |
|---|---|
| Coherence horizon | 100+ steps of human-style play on Pong |
| Controllability | 95%+ of action commands move the paddle as in the real game |
| Ball permanence | Exactly one ball in 98%+ of non-scoring frames |
| Bounce plausibility | 90%+ of wall and paddle bounces within tolerance |
| v2 beats v1 | Lower LPIPS at horizons 10, 50 and 100 |
| Browser | 12+ model ticks/s on a recent laptop with WebGPU |
| Load size | Under 30 MB fp16, first interactive frame under 10 s |
| Agent transfer | Mean and std over 20+ episodes and 3+ seeds, plus human-normalized score |

Human-normalized score for Atari Pong uses random −20.7 and human 14.6. Do not report that number for original Pong.

<!-- RESULTS:START -->
No evaluation has been committed yet.
<!-- RESULTS:END -->
