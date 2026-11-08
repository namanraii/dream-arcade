# Resume bullets

Leave the letters in place until `RESULTS.md` has a full run. A smoke checkpoint does not supply X, Y, N, M, or Z.

- Built and deployed an action-conditioned diffusion world model of Pong that runs in the browser on WebGPU at X frames per second from a Y MB model, trained on free-tier GPUs.
- Designed an evaluation suite (LPIPS drift curves, paddle controllability, ball permanence, bounce checks) showing the model stays coherent for N steps, against M steps for a one-step baseline.
- Trained a PPO agent entirely inside the learned model and reached a human-normalized score of Z in the real game, mean and std over 20+ episodes and 3+ seeds.

Human-normalized score uses Atari Pong constants (random −20.7, human 14.6) only if the evaluation environment was `ALE/Pong-v5`. If the published run is original Pong, replace the third bullet with the raw mean and standard deviation and say the environment is the original one.
