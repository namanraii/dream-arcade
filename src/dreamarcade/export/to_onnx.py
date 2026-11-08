"""Export the denoiser, reward head, and actor. The Euler loop stays in TypeScript."""

from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig
from torch import nn

from dreamarcade.train.actor import ActorCritic
from dreamarcade.train.build import build_diffusion, build_reward
from dreamarcade.utils import load_checkpoint, pick_device


class ExportDenoiser(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        history: torch.Tensor,
        action: torch.Tensor,
        noisy_frame: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.denoise(history, action, noisy_frame, sigma)


class ExportReward(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, history: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(history, action)


class ExportPolicy(nn.Module):
    def __init__(self, model: ActorCritic) -> None:
        super().__init__()
        self.model = model

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        logits, _value = self.model(history)
        return logits


def _export(module: nn.Module, args, path: Path, names_in: list[str], names_out: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    module.eval()
    torch.onnx.export(
        module,
        args,
        str(path),
        input_names=names_in,
        output_names=names_out,
        opset_version=17,
        dynamo=False,
        do_constant_folding=True,
    )
    print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB)")


def _fp16(path: Path) -> Path | None:
    try:
        import onnx
        from onnxconverter_common import float16
    except Exception:
        try:
            import onnx
            from onnxruntime.transformers.float16 import convert_float_to_float16 as convert
        except Exception as exc:
            print(f"fp16 conversion unavailable ({exc})")
            return None
        model = onnx.load(str(path))
        converted = convert(model, keep_io_types=True)
    else:
        model = onnx.load(str(path))
        converted = float16.convert_float_to_float16(model, keep_io_types=True)
    dest = path.with_name(path.stem + "_fp16.onnx")
    onnx.save(converted, str(dest))
    print(f"wrote {dest} ({dest.stat().st_size / 1e6:.2f} MB)")
    return dest


def export_models(cfg: DictConfig) -> dict[str, Path]:
    device = torch.device("cpu")
    out = Path(cfg.out_dir)
    written: dict[str, Path] = {}
    history = torch.zeros(1, 12, 64, 64)
    action = torch.zeros(1, dtype=torch.long)
    noisy = torch.zeros(1, 3, 64, 64)
    sigma = torch.ones(1)
    if cfg.denoiser_checkpoint:
        world = build_diffusion(cfg).to(device).eval()
        load_checkpoint(Path(str(cfg.denoiser_checkpoint)), world, use_ema=True)
        path = out / "denoiser.onnx"
        _export(
            ExportDenoiser(world),
            (history, action, noisy, sigma),
            path,
            ["history", "action", "noisy_frame", "sigma"],
            ["denoised"],
        )
        written["denoiser"] = path
        fp16 = _fp16(path)
        if fp16 is not None:
            written["denoiser_fp16"] = fp16
    if cfg.reward_checkpoint:
        reward = build_reward(cfg).to(device).eval()
        load_checkpoint(Path(str(cfg.reward_checkpoint)), reward, use_ema=True)
        path = out / "reward.onnx"
        _export(ExportReward(reward), (history, action), path, ["history", "action"], ["reward_logits", "done_logit"])
        written["reward"] = path
    if cfg.policy_checkpoint:
        actor = ActorCritic().to(device).eval()
        load_checkpoint(Path(str(cfg.policy_checkpoint)), actor, use_ema=False)
        path = out / "policy.onnx"
        _export(ExportPolicy(actor), (history,), path, ["history"], ["logits"])
        written["policy"] = path
    _ = pick_device
    return written
