"""PPO inside the world model, then a real-env evaluation.

The reward the agent sees is the reward head's expected value in {-1, 0, +1}.
Real-game scores are computed separately and are the numbers that count.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig

from dreamarcade.data.dataset import FrameDataset
from dreamarcade.data.env import make_env
from dreamarcade.models.world import sample_next_frame
from dreamarcade.train.actor import ActorCritic
from dreamarcade.train.build import build_baseline, build_diffusion, build_reward
from dreamarcade.train.config import to_plain
from dreamarcade.train.ppo_core import gae, ppo_update
from dreamarcade.utils import (
    HistoryStack,
    RunLogger,
    load_checkpoint,
    model_to_uint8,
    pick_device,
    save_checkpoint,
    set_seed,
)


class DreamRollout:
    def __init__(self, cfg: DictConfig, device: torch.device) -> None:
        self.device = device
        self.kind = str(cfg.dream_model)
        self.steps = int(cfg.sampler_steps)
        self.max_steps = int(cfg.dream_episode_steps)
        if self.kind == "v1":
            self.world = build_baseline(cfg).to(device)
            load_checkpoint(Path(cfg.v1_checkpoint), self.world, use_ema=True)
        else:
            self.world = build_diffusion(cfg).to(device)
            load_checkpoint(Path(cfg.v2_checkpoint), self.world, use_ema=True)
        self.world.eval()
        self.reward_net = build_reward(cfg).to(device)
        load_checkpoint(Path(cfg.reward_checkpoint), self.reward_net, use_ema=True)
        self.reward_net.eval()
        dataset = FrameDataset(Path(cfg.data_dir), split="train", history=int(cfg.data.history))
        self.starts = [dataset[i]["history"] for i in range(0, len(dataset), max(len(dataset) // 32, 1))]
        if not self.starts:
            raise RuntimeError("no training frames to ground dream episodes")
        self.history: torch.Tensor | None = None
        self.t = 0

    @torch.no_grad()
    def reset(self) -> torch.Tensor:
        pick = self.starts[int(torch.randint(0, len(self.starts), (1,)).item())]
        self.history = pick.to(self.device)
        self.t = 0
        return self.history

    @torch.no_grad()
    def step(self, action: int) -> tuple[torch.Tensor, float, bool]:
        assert self.history is not None
        action_t = torch.tensor([action], device=self.device)
        history = self.history.unsqueeze(0)
        if self.kind == "v1":
            frame = self.world(history, action_t).clamp(-1, 1)
        else:
            frame = sample_next_frame(self.world, history, action_t, steps=self.steps)
        expected = self.reward_net.expected_reward(history, action_t)
        _logits, done_logit = self.reward_net(history, action_t)
        done_prob = torch.sigmoid(done_logit.squeeze()).item()
        self.history = torch.cat([self.history[3:], frame.squeeze(0)], dim=0)
        self.t += 1
        done = done_prob > 0.5 or self.t >= self.max_steps
        return self.history, float(expected.item()), bool(done)


def evaluate_real(net: ActorCritic, env_name: str, episodes: int, seed: int, device: torch.device) -> dict:
    scores = []
    for episode in range(episodes):
        env = make_env(env_name, seed=seed + episode, max_steps=300)
        obs, _info = env.reset()
        stack = HistoryStack()
        history = stack.reset(obs)
        done = False
        total = 0.0
        steps = 0
        while not done and steps < 300:
            with torch.no_grad():
                action, _logp, _value = net.act(history.unsqueeze(0).to(device), deterministic=True)
            obs, reward, done, _info = env.step(int(action.item()))
            history = stack.push(obs)
            total += reward
            steps += 1
        if hasattr(env, "close"):
            env.close()
        scores.append(total)
    arr = np.asarray(scores, dtype=np.float64)
    return {
        "episodes": episodes,
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "scores": arr.tolist(),
    }


def human_normalized(score: float, random_score: float = -20.7, human_score: float = 14.6) -> float:
    return (score - random_score) / (human_score - random_score)


def train_ppo_dream(cfg: DictConfig) -> Path:
    set_seed(int(cfg.seed))
    device = pick_device(str(cfg.device))
    dream = DreamRollout(cfg, device)
    net = ActorCritic().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=float(cfg.lr))
    logger = RunLogger(bool(cfg.wandb), str(cfg.project), str(cfg.run_name), to_plain(cfg))
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    updates = int(cfg.updates)
    rollout_steps = int(cfg.rollout_steps)
    for update in range(updates):
        histories = []
        actions = []
        log_probs = []
        values = []
        rewards = []
        dones = []
        history = dream.reset()
        for _step in range(rollout_steps):
            act, logp, value = net.act(history.unsqueeze(0))
            next_history, reward, done = dream.step(int(act.item()))
            histories.append(history)
            actions.append(act.squeeze(0))
            log_probs.append(logp.squeeze(0))
            values.append(value.squeeze(0))
            rewards.append(reward)
            dones.append(float(done))
            history = dream.reset() if done else next_history
        with torch.no_grad():
            _act, _logp, bootstrap = net.act(history.unsqueeze(0))
        advantages, returns = gae(
            torch.tensor(rewards, device=device),
            torch.stack(values).detach(),
            torch.tensor(dones, device=device),
            bootstrap.squeeze(0).detach(),
            gamma=float(cfg.gamma),
            lam=float(cfg.gae_lambda),
        )
        adv = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-6)
        stats = ppo_update(
            net,
            optimizer,
            torch.stack(histories).detach(),
            torch.stack(actions).detach(),
            torch.stack(log_probs).detach(),
            returns.detach(),
            adv.detach(),
            clip=float(cfg.clip),
            ent_coef=float(cfg.ent_coef),
            vf_coef=float(cfg.vf_coef),
            epochs=int(cfg.epochs),
            minibatch=min(int(cfg.minibatch), rollout_steps),
        )
        stats["dream_return"] = float(sum(rewards))
        logger.log(stats, update)
        if (update + 1) % int(cfg.eval_every) == 0 or update == updates - 1:
            real = evaluate_real(
                net, str(cfg.data.name), episodes=int(cfg.eval_episodes), seed=int(cfg.seed), device=device
            )
            payload = {"real_mean": real["mean"], "real_std": real["std"]}
            if str(cfg.data.name).startswith("ALE/Pong"):
                payload["hns"] = human_normalized(real["mean"])
            logger.log(payload, update)
            print(f"real-env eval {real['mean']:.3f} ± {real['std']:.3f} over {real['episodes']} episodes")
    save_checkpoint(out / "policy.pt", net, None, optimizer, updates, to_plain(cfg))
    logger.finish()
    print(f"dream policy {out / 'policy.pt'}")
    return out / "policy.pt"


def dream_frame_uint8(frame: torch.Tensor) -> np.ndarray:
    return model_to_uint8(frame)
