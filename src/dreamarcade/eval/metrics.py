"""Fidelity metrics. LPIPS is reported only when the package and weights load."""

from __future__ import annotations

import numpy as np
import torch

_lpips = None
_lpips_failed = False


def to_unit(frame: np.ndarray) -> np.ndarray:
    image = frame.astype(np.float32)
    if image.max() > 1.5:
        image = image / 255.0
    return image


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((to_unit(a) - to_unit(b)) ** 2))
    if mse < 1e-12:
        return 50.0
    return float(10.0 * np.log10(1.0 / mse))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean of per-channel SSIM with a uniform 7x7 window."""
    x = to_unit(a)
    y = to_unit(b)
    if x.ndim == 2:
        x = x[..., None]
        y = y[..., None]
    scores = [_ssim_channel(x[..., c], y[..., c]) for c in range(x.shape[-1])]
    return float(np.mean(scores))


def _box(image: np.ndarray, k: int = 7) -> np.ndarray:
    pad = k // 2
    padded = np.pad(image, pad, mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    total = integral[k:, k:] - integral[:-k, k:] - integral[k:, :-k] + integral[:-k, :-k]
    return total / float(k * k)


def _ssim_channel(x: np.ndarray, y: np.ndarray) -> float:
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = _box(x)
    mu_y = _box(y)
    sigma_x = _box(x * x) - mu_x**2
    sigma_y = _box(y * y) - mu_y**2
    sigma_xy = _box(x * y) - mu_x * mu_y
    num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    den = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    return float((num / den).mean())


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(to_unit(a) - to_unit(b))))


def lpips_distance(pred_uint8: np.ndarray, target_uint8: np.ndarray, device: torch.device) -> float | None:
    global _lpips, _lpips_failed
    if _lpips_failed:
        return None
    try:
        import lpips
    except ImportError:
        _lpips_failed = True
        return None
    if _lpips is None:
        try:
            _lpips = lpips.LPIPS(net="alex").to(device).eval()
        except Exception:
            _lpips_failed = True
            return None
    def tensor(image: np.ndarray) -> torch.Tensor:
        arr = to_unit(image).transpose(2, 0, 1)
        return torch.from_numpy(arr).unsqueeze(0).to(device) * 2 - 1

    with torch.no_grad():
        value = _lpips(tensor(pred_uint8), tensor(target_uint8))
    return float(value.item())


def precision_recall(pred: np.ndarray, target: np.ndarray, n_classes: int = 3) -> dict:
    report = {}
    for cls in range(n_classes):
        tp = int(np.sum((pred == cls) & (target == cls)))
        fp = int(np.sum((pred == cls) & (target != cls)))
        fn = int(np.sum((pred != cls) & (target == cls)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        report[str(cls)] = {"precision": precision, "recall": recall, "support": int(np.sum(target == cls))}
    return report
