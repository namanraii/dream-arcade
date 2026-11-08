"""Seeds, devices, checkpoints, and the optional W&B logger."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


def repo_root() -> Path:
    here = Path(__file__).resolve()
    candidates = [here.parents[2], Path.cwd()]
    for path in candidates:
        if (path / "configs").is_dir() and (path / "src" / "dreamarcade").is_dir():
            return path
    return here.parents[2]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(name: str = "auto") -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def uint8_to_model(frame: np.ndarray) -> torch.Tensor:
    """HWC uint8 RGB -> CHW float in [-1, 1]."""
    tensor = torch.from_numpy(np.ascontiguousarray(frame))
    if tensor.ndim != 3 or tensor.shape[-1] != 3:
        raise ValueError(f"expected HWC RGB, got {tuple(tensor.shape)}")
    tensor = tensor.permute(2, 0, 1).float()
    return tensor / 127.5 - 1.0


def model_to_uint8(frame: torch.Tensor) -> np.ndarray:
    """CHW float in [-1, 1] -> HWC uint8."""
    image = frame.detach().float().clamp(-1, 1)
    image = ((image + 1) * 127.5).round().clamp(0, 255).byte()
    return image.permute(1, 2, 0).cpu().numpy()


class HistoryStack:
    """Four RGB frames packed as 12 channels, oldest first."""

    def __init__(self, length: int = 4) -> None:
        self.length = length
        self.frames: list[torch.Tensor] = []

    def reset(self, frame: np.ndarray) -> torch.Tensor:
        item = uint8_to_model(frame)
        self.frames = [item.clone() for _ in range(self.length)]
        return self.tensor()

    def push(self, frame: np.ndarray) -> torch.Tensor:
        self.frames.pop(0)
        self.frames.append(uint8_to_model(frame))
        return self.tensor()

    def push_tensor(self, frame: torch.Tensor) -> torch.Tensor:
        self.frames.pop(0)
        self.frames.append(frame.detach().cpu())
        return self.tensor()

    def tensor(self) -> torch.Tensor:
        return torch.cat(self.frames, dim=0)


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for key, value in model.state_dict().items():
            if not value.dtype.is_floating_point:
                self.shadow[key] = value.detach().clone()
                continue
            self.shadow[key].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)

    def copy_to(self, model: torch.nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict) -> None:
        self.decay = state["decay"]
        self.shadow = state["shadow"]


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    ema: EMA | None,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    config: dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "ema": None if ema is None else ema.state_dict(),
            "optimizer": None if optimizer is None else optimizer.state_dict(),
            "step": step,
            "config": config,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    ema: EMA | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    use_ema: bool = True,
) -> int:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if use_ema and blob.get("ema"):
        model.load_state_dict(blob["ema"]["shadow"], strict=True)
    else:
        model.load_state_dict(blob["model"], strict=True)
    if ema is not None and blob.get("ema"):
        ema.load_state_dict(blob["ema"])
    if optimizer is not None and blob.get("optimizer"):
        optimizer.load_state_dict(blob["optimizer"])
    return int(blob.get("step", 0))


class RunLogger:
    """Prints metrics. Sends them to W&B only when a key is set and tracking is on."""

    def __init__(self, enabled: bool, project: str, name: str, config: dict) -> None:
        self.enabled = bool(enabled) and bool(os.environ.get("WANDB_API_KEY"))
        self._wandb = None
        if enabled and not os.environ.get("WANDB_API_KEY"):
            print("W&B disabled: WANDB_API_KEY is not set")
        if self.enabled:
            import wandb

            self._wandb = wandb
            wandb.init(project=project, name=name, config=config)

    def log(self, metrics: dict, step: int) -> None:
        printable = " ".join(f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}" for key, value in metrics.items())
        print(f"step {step}: {printable}")
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def finish(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()
