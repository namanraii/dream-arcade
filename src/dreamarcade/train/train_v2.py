"""Train the EDM world model. The exported weights are the EMA."""

from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig
from torch import nn

from dreamarcade.data.dataset import make_loader, pixel_loss_weight
from dreamarcade.train.build import build_diffusion
from dreamarcade.train.config import to_plain
from dreamarcade.train.loop import autocast_context, cosine_lr, cycle_batches
from dreamarcade.utils import (
    EMA,
    RunLogger,
    load_checkpoint,
    pick_device,
    save_checkpoint,
    set_seed,
)


def train_v2(cfg: DictConfig) -> Path:
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
    model = build_diffusion(cfg).to(device)
    ema = EMA(model, decay=float(cfg.ema_decay))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay)
    )
    start = 0
    if cfg.resume:
        start = load_checkpoint(Path(cfg.resume), model, ema, optimizer, use_ema=False)
        print(f"resumed v2 from step {start}")
    logger = RunLogger(bool(cfg.wandb), str(cfg.project), str(cfg.run_name), to_plain(cfg))
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    batches = cycle_batches(loader)
    total = int(cfg.steps)
    for step in range(start, total):
        batch = next(batches)
        history = batch["history"].to(device)
        action = batch["action"][:, 0].to(device)
        target = batch["target"][:, 0].to(device)
        sample_w = batch["weight"][:, 0].to(device).view(-1, 1, 1, 1)
        pixel_w = pixel_loss_weight(history, target, alpha=float(cfg.pixel_alpha)) * sample_w
        lr = cosine_lr(step, total, float(cfg.lr), int(cfg.warmup))
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, bool(cfg.mixed_precision)):
            loss = model.loss(
                history,
                action,
                target,
                pixel_weight=pixel_w,
                history_noise=float(cfg.history_noise),
            )
        if not torch.isfinite(loss):
            print(f"non-finite loss at step {step}; skipping the update")
            optimizer.zero_grad(set_to_none=True)
            continue
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
        optimizer.step()
        ema.update(model)
        if step % int(cfg.log_every) == 0 or step == total - 1:
            logger.log({"loss": float(loss.detach()), "lr": lr}, step)
        if step % int(cfg.save_every) == 0 or step == total - 1:
            save_checkpoint(out / "latest.pt", model, ema, optimizer, step + 1, to_plain(cfg))
    logger.finish()
    final = out / "latest.pt"
    print(f"v2 checkpoint {final}")
    return final
