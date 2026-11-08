"""Train the one-step baseline. Loss is weighted L1 plus an optional LPIPS term."""

from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig
from torch import nn

from dreamarcade.data.dataset import make_loader, pixel_loss_weight
from dreamarcade.train.build import build_baseline
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

_lpips_model = None
_lpips_failed = False


def perceptual_loss(pred: torch.Tensor, target: torch.Tensor, device: torch.device) -> torch.Tensor | None:
    global _lpips_model, _lpips_failed
    if _lpips_failed:
        return None
    try:
        import lpips
    except ImportError:
        _lpips_failed = True
        print("LPIPS package is not installed; training with weighted L1 only")
        return None
    if _lpips_model is None:
        try:
            _lpips_model = lpips.LPIPS(net="alex").to(device).eval()
            for param in _lpips_model.parameters():
                param.requires_grad_(False)
        except Exception as exc:
            _lpips_failed = True
            print(f"LPIPS weights unavailable ({exc}); training with weighted L1 only")
            return None
    return _lpips_model(pred.clamp(-1, 1), target.clamp(-1, 1)).mean()


def train_v1(cfg: DictConfig) -> Path:
    set_seed(int(cfg.seed))
    device = pick_device(str(cfg.device))
    horizon = 2 if bool(cfg.scheduled_sampling) else 1
    _dataset, loader = make_loader(
        Path(cfg.data_dir),
        split="train",
        batch_size=int(cfg.batch_size),
        history=int(cfg.data.history),
        horizon=horizon,
        balanced=True,
        seed=int(cfg.seed),
    )
    model = build_baseline(cfg).to(device)
    ema = EMA(model, decay=float(cfg.ema_decay))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay)
    )
    start = 0
    if cfg.resume:
        start = load_checkpoint(Path(cfg.resume), model, ema, optimizer, use_ema=False)
        print(f"resumed v1 from step {start}")
    logger = RunLogger(bool(cfg.wandb), str(cfg.project), str(cfg.run_name), to_plain(cfg))
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    batches = cycle_batches(loader)
    total = int(cfg.steps)
    alpha = float(cfg.pixel_alpha)
    aug = float(cfg.context_noise)
    for step in range(start, total):
        batch = next(batches)
        history = batch["history"].to(device)
        action = batch["action"][:, 0].to(device)
        target = batch["target"][:, 0].to(device)
        sample_w = batch["weight"][:, 0].to(device).view(-1, 1, 1, 1)
        if aug > 0:
            history = history + aug * torch.randn_like(history)
        lr = cosine_lr(step, total, float(cfg.lr), int(cfg.warmup))
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, bool(cfg.mixed_precision)):
            pred = model(history, action)
            weight = pixel_loss_weight(history, target, alpha=alpha) * sample_w
            loss = (weight * (pred - target).abs()).mean()
            if float(cfg.perceptual_weight) > 0:
                extra = perceptual_loss(pred, target, device)
                if extra is not None:
                    loss = loss + float(cfg.perceptual_weight) * extra
            if horizon == 2 and step >= int(cfg.ss_warmup):
                rolled = torch.cat([history[:, 3:], pred.detach()], dim=1)
                pred2 = model(rolled, batch["action"][:, 1].to(device))
                loss = loss + (pred2 - batch["target"][:, 1].to(device)).abs().mean()
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
    print(f"v1 checkpoint {final}")
    return final
