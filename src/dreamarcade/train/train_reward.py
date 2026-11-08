"""Train the reward and termination head with class-balanced sampling."""

from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig
from torch import nn

from dreamarcade.data.dataset import make_loader
from dreamarcade.models.reward_done import reward_to_class
from dreamarcade.train.build import build_reward
from dreamarcade.train.config import to_plain
from dreamarcade.train.loop import cosine_lr, cycle_batches
from dreamarcade.utils import (
    EMA,
    RunLogger,
    load_checkpoint,
    pick_device,
    save_checkpoint,
    set_seed,
)


def class_weights_from_loader(loader, device: torch.device) -> torch.Tensor:
    counts = torch.ones(3)
    seen = 0
    for batch in loader:
        classes = reward_to_class(batch["reward"][:, 0])
        for value in range(3):
            counts[value] += (classes == value).sum()
        seen += 1
        if seen >= 20:
            break
    weights = counts.sum() / counts
    return (weights / weights.mean()).to(device)


def train_reward(cfg: DictConfig) -> Path:
    set_seed(int(cfg.seed))
    device = pick_device(str(cfg.device))
    _dataset, loader = make_loader(
        Path(cfg.data_dir),
        split="train",
        batch_size=int(cfg.batch_size),
        history=int(cfg.data.history),
        horizon=1,
        balanced=True,
        seed=int(cfg.seed),
    )
    model = build_reward(cfg).to(device)
    ema = EMA(model, decay=float(cfg.ema_decay))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    start = 0
    if cfg.resume:
        start = load_checkpoint(Path(cfg.resume), model, ema, optimizer, use_ema=False)
    weights = class_weights_from_loader(loader, device)
    print(f"reward class weights {weights.tolist()}")
    logger = RunLogger(bool(cfg.wandb), str(cfg.project), str(cfg.run_name), to_plain(cfg))
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    batches = cycle_batches(loader)
    total = int(cfg.steps)
    for step in range(start, total):
        batch = next(batches)
        history = batch["history"].to(device)
        action = batch["action"][:, 0].to(device)
        target_class = reward_to_class(batch["reward"][:, 0].to(device))
        done = batch["done"][:, 0].to(device)
        lr = cosine_lr(step, total, float(cfg.lr), int(cfg.warmup))
        for group in optimizer.param_groups:
            group["lr"] = lr
        logits, done_logit = model(history, action)
        reward_loss = nn.functional.cross_entropy(logits, target_class, weight=weights)
        done_loss = nn.functional.binary_cross_entropy_with_logits(done_logit.squeeze(-1), done)
        loss = reward_loss + done_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
        optimizer.step()
        ema.update(model)
        if step % int(cfg.log_every) == 0 or step == total - 1:
            logger.log(
                {
                    "loss": float(loss.detach()),
                    "reward_loss": float(reward_loss.detach()),
                    "done_loss": float(done_loss.detach()),
                    "lr": lr,
                },
                step,
            )
        if step % int(cfg.save_every) == 0 or step == total - 1:
            save_checkpoint(out / "latest.pt", model, ema, optimizer, step + 1, to_plain(cfg))
    logger.finish()
    final = out / "latest.pt"
    print(f"reward checkpoint {final}")
    return final
