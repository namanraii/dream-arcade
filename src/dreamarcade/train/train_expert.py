"""Short PPO on the real environment, used as the data-collection expert.

Stable-Baselines3 zoo Pong agents expect stacked 84×84 grayscale with the
Atari wrapper, so they are not dropped onto this 64×64 RGB pipeline. This
script is the "otherwise train a quick PPO" path from the plan.
"""

from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig

from dreamarcade.data.env import make_env
from dreamarcade.train.actor import ActorCritic
from dreamarcade.train.config import to_plain
from dreamarcade.train.ppo_core import gae, ppo_update
from dreamarcade.train.ppo_dream import evaluate_real
from dreamarcade.utils import HistoryStack, RunLogger, pick_device, save_checkpoint, set_seed


def train_expert(cfg: DictConfig) -> Path:
    set_seed(int(cfg.seed))
    device = pick_device(str(cfg.device))
    net = ActorCritic().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=float(cfg.lr))
    logger = RunLogger(bool(cfg.wandb), str(cfg.project), str(cfg.run_name) + "-expert", to_plain(cfg))
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    env = make_env(str(cfg.data.name), seed=int(cfg.seed), max_steps=int(cfg.episode_length))
    obs, _info = env.reset()
    stack = HistoryStack()
    history = stack.reset(obs)
    for update in range(int(cfg.updates)):
        histories = []
        actions = []
        log_probs = []
        values = []
        rewards = []
        dones = []
        for _step in range(int(cfg.rollout_steps)):
            action, logp, value = net.act(history.unsqueeze(0).to(device))
            obs, reward, done, _info = env.step(int(action.item()))
            histories.append(history)
            actions.append(action.squeeze(0))
            log_probs.append(logp.squeeze(0))
            values.append(value.squeeze(0))
            rewards.append(float(reward))
            dones.append(float(done))
            if done:
                obs, _info = env.reset()
                history = stack.reset(obs)
            else:
                history = stack.push(obs)
        with torch.no_grad():
            _a, _l, bootstrap = net.act(history.unsqueeze(0).to(device))
        advantages, returns = gae(
            torch.tensor(rewards, device=device),
            torch.stack(values).detach(),
            torch.tensor(dones, device=device),
            bootstrap.squeeze(0).detach(),
        )
        adv = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-6)
        stats = ppo_update(
            net,
            optimizer,
            torch.stack(histories).to(device).detach(),
            torch.stack(actions).detach(),
            torch.stack(log_probs).detach(),
            returns.detach(),
            adv.detach(),
            epochs=int(cfg.epochs),
            minibatch=min(int(cfg.minibatch), int(cfg.rollout_steps)),
        )
        logger.log(stats, update)
    if hasattr(env, "close"):
        env.close()
    report = evaluate_real(net, str(cfg.data.name), episodes=int(cfg.eval_episodes), seed=int(cfg.seed) + 100, device=device)
    print(f"expert real return {report['mean']:.3f} ± {report['std']:.3f}")
    path = out / "expert.pt"
    save_checkpoint(path, net, None, optimizer, int(cfg.updates), to_plain(cfg))
    logger.finish()
    return path
