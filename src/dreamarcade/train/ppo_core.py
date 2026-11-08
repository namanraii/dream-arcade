"""CleanRL-style PPO update, shared by the expert and the dream agent."""

from __future__ import annotations

import torch
from torch import nn


def ppo_update(
    net: nn.Module,
    optimizer: torch.optim.Optimizer,
    history: torch.Tensor,
    actions: torch.Tensor,
    old_log_prob: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    clip: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    epochs: int = 4,
    minibatch: int = 64,
) -> dict[str, float]:
    count = history.shape[0]
    last_stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    for _epoch in range(epochs):
        order = torch.randperm(count, device=history.device)
        for start in range(0, count, minibatch):
            index = order[start : start + minibatch]
            logits, values = net(history[index])
            dist = torch.distributions.Categorical(logits=logits)
            log_prob = dist.log_prob(actions[index])
            ratio = (log_prob - old_log_prob[index]).exp()
            adv = advantages[index]
            policy_loss = -torch.min(
                ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv
            ).mean()
            value_loss = 0.5 * (values - returns[index]).pow(2).mean()
            entropy = dist.entropy().mean()
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            last_stats = {
                "policy_loss": float(policy_loss.detach()),
                "value_loss": float(value_loss.detach()),
                "entropy": float(entropy.detach()),
            }
    return last_stats


def gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """rewards, values, dones are [T]. next_value is the bootstrap scalar."""
    steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last = 0.0
    for t in reversed(range(steps)):
        next_v = next_value if t == steps - 1 else values[t + 1]
        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * next_non_terminal - values[t]
        last = delta + gamma * lam * next_non_terminal * last
        advantages[t] = last
    returns = advantages + values
    return advantages, returns
