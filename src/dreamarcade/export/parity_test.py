"""PyTorch vs ONNX on a fixed input. fp32 must match tightly."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from dreamarcade.export.to_onnx import ExportDenoiser
from dreamarcade.train.build import build_diffusion
from dreamarcade.utils import load_checkpoint, set_seed


def compare(pytorch: np.ndarray, onnx_out: np.ndarray, atol: float, rtol: float) -> float:
    diff = np.max(np.abs(pytorch - onnx_out))
    if not np.allclose(pytorch, onnx_out, atol=atol, rtol=rtol):
        raise AssertionError(f"ONNX parity failed, max abs diff {diff}")
    return float(diff)


def run_parity(
    checkpoint: Path,
    onnx_path: Path,
    cfg,
    atol: float = 1e-4,
    rtol: float = 1e-4,
    seed: int = 0,
) -> float:
    set_seed(seed)
    device = torch.device("cpu")
    model = build_diffusion(cfg).to(device).eval()
    load_checkpoint(checkpoint, model, use_ema=True)
    history = torch.randn(1, 12, 64, 64)
    action = torch.tensor([2], dtype=torch.long)
    noisy = torch.randn(1, 3, 64, 64)
    sigma = torch.tensor([0.5])
    with torch.no_grad():
        reference = ExportDenoiser(model)(history, action, noisy, sigma).numpy()
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    outputs = session.run(
        None,
        {
            "history": history.numpy(),
            "action": action.numpy(),
            "noisy_frame": noisy.numpy(),
            "sigma": sigma.numpy(),
        },
    )
    return compare(reference, outputs[0], atol=atol, rtol=rtol)
