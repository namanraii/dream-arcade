# Checked before building

The plan said to verify these instead of trusting the draft. Status as of 8 November 2026.

## Kaggle and Colab

Public write-ups that quote Kaggle's own GPU docs, read in October and November 2026, agree on the shape of the quota:

- About 30 GPU hours a week, sometimes higher, one pool shared by the session. Confirm the remaining number on the account page. It moves.
- GPU sessions are documented at up to 12 hours. Idle limits are shorter (reports of 20 and of 60 minutes). Plan on under 9 hours and a checkpoint every few hundred steps.
- Recent accounts are offered two T4s (16 GB each). A 15 September 2026 retirement of the P100 option is reported by one write-up; older docs still mention it. Use whichever card the notebook page actually offers. This code uses a single process and one GPU.
- Colab's free GPU quota is not published. Notebooks stop by 12 hours. Use it as overflow.

Nothing here was read off an authenticated Kaggle account. Check the quota widget before the first overnight run.

## ale-py

Not required for the default path. Original Pong implements the same observation and action contract and is what the tests run. `ALE/Pong-v5` is `data=atari_pong` after `pip install -e ".[atari]"`. Confirm the import on the Kaggle image as well; ROM availability inside current `ale-py` has changed across releases. If the import fails, stay on original Pong and say so in the model card. Do not copy ROMs into the repo.

## Pretrained Pong agents

`sb3/ppo-PongNoFrameskip-v4` is on the Hugging Face Hub and is a Stable-Baselines3 zoo PPO with a self-reported mean return of 21 on that environment. The library is MIT. The agent expects the zoo's Atari wrapper: frameskip, grayscale, 84×84, stack of 4. Those weights are the wrong shape for this 64×64 RGB pipeline, so the collector does not load them. `dreamarcade train-expert` trains a small actor on the frames this project actually stores.

## WebGPU and ONNX

The page requests the WebGPU execution provider and falls back to WASM. The denoiser uses GroupNorm, SiLU, and explicit matmul attention so the graph stays in the opset-17 subset. `dreamarcade parity` compares PyTorch and ONNX Runtime on CPU for a fixed input. WebGPU numerical drift is larger; the fp16 file keeps the graph inputs and outputs in fp32. If export of fp16 fails because `onnxconverter-common` is absent, the fp32 file is still the one the WASM path loads.

## Step time

Unmeasured on a T4. The full config is base 64 channels, multipliers (1, 2, 2, 2), attention at 16×16 and 8×8. That network is about 6.1M parameters, about 12 MB in fp16, under the 30 MB budget. `tests/test_shapes.py` locks the 5–15M band. Time one hundred steps on the T4 before launching 100k, and shrink `model.base_channels` if a step is too slow for the weekly quota.
