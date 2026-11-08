"""One-step student that matches a multi-step diffusion teacher.

Reports wall time per frame for the teacher and the student, and the mean
absolute gap between them. That gap is the quality cost of the speedup.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
from omegaconf import DictConfig
from torch import nn

from dreamarcade.data.dataset import make_loader
from dreamarcade.models.world import sample_next_frame
from dreamarcade.train.build import build_baseline, build_diffusion
from dreamarcade.train.config import to_plain
from dreamarcade.train.loop import cycle_batches
from dreamarcade.utils import (
    EMA,
    RunLogger,
    load_checkpoint,
    pick_device,
    save_checkpoint,
    set_seed,
)


def time_forward(fn, repeats: int = 5) -> float:
    for _ in range(1):
        fn()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats


def train_distill(cfg: DictConfig) -> Path:
    set_seed(int(cfg.seed))
    device = pick_device(str(cfg.device))
    teacher = build_diffusion(cfg).to(device).eval()
    load_checkpoint(Path(cfg.teacher_checkpoint), teacher, use_ema=True)
    for param in teacher.parameters():
        param.requires_grad_(False)
    student = build_baseline(cfg).to(device)
    ema = EMA(student, decay=float(cfg.ema_decay))
    optimizer = torch.optim.AdamW(student.parameters(), lr=float(cfg.lr))
    _dataset, loader = make_loader(
        Path(cfg.data_dir),
        split="train",
        batch_size=int(cfg.batch_size),
        history=int(cfg.data.history),
        balanced=False,
        seed=int(cfg.seed),
    )
    logger = RunLogger(bool(cfg.wandb), str(cfg.project), str(cfg.run_name) + "-distill", to_plain(cfg))
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    batches = cycle_batches(loader)
    steps = int(cfg.sampler_steps)
    for step in range(int(cfg.steps)):
        batch = next(batches)
        history = batch["history"].to(device)
        action = batch["action"][:, 0].to(device)
        with torch.no_grad():
            target = sample_next_frame(teacher, history, action, steps=steps)
        pred = student(history, action)
        loss = (pred - target).abs().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        ema.update(student)
        if step % int(cfg.log_every) == 0 or step == int(cfg.steps) - 1:
            logger.log({"distill_l1": float(loss.detach())}, step)
    batch = next(batches)
    history = batch["history"].to(device)
    action = batch["action"][:, 0].to(device)

    def teacher_call():
        with torch.no_grad():
            return sample_next_frame(teacher, history, action, steps=steps)

    def student_call():
        with torch.no_grad():
            return student(history, action)

    teacher_s = time_forward(teacher_call)
    student_s = time_forward(student_call)
    with torch.no_grad():
        gap = (teacher_call() - student_call()).abs().mean().item()
    speedup = teacher_s / max(student_s, 1e-8)
    print(
        f"distill speedup {speedup:.2f}x  teacher {teacher_s * 1000:.1f} ms  "
        f"student {student_s * 1000:.1f} ms  mean abs gap {gap:.4f}"
    )
    path = out / "student.pt"
    save_checkpoint(path, student, ema, optimizer, int(cfg.steps), to_plain(cfg))
    (out / "distill_report.txt").write_text(
        f"speedup {speedup:.4f}\nteacher_s {teacher_s:.6f}\nstudent_s {student_s:.6f}\ngap {gap:.6f}\n"
    )
    logger.finish()
    return path
