"""Pixel detectors for the paddle and the ball.

They are deliberately simple: Pong is a bright sprite on a dark field. The
same functions run on original Pong and on resized Atari frames.
"""

from __future__ import annotations

import numpy as np

PADDLE_COLUMNS = {
    "right": (58, 64),
    "left": (0, 6),
    "bottom": None,
}


def luminance(frame: np.ndarray) -> np.ndarray:
    image = frame.astype(np.float32)
    if image.max() > 1.5:
        image = image / 255.0
    return image.mean(axis=-1)


def detect_paddle(frame: np.ndarray, side: str = "right", threshold: float = 0.5) -> float | None:
    """Return the paddle center in pixels. y grows downward, except side='bottom' which returns x."""
    lum = luminance(frame)
    if side == "bottom":
        band = lum[56:64, :]
        column = band.max(axis=0)
        mask = column >= threshold
        if not mask.any():
            return None
        index = np.flatnonzero(mask)
        return float((index[0] + index[-1]) / 2)
    start, end = PADDLE_COLUMNS[side]
    band = lum[:, start:end]
    row = band.max(axis=1)
    mask = row >= threshold
    if not mask.any():
        return None
    index = np.flatnonzero(mask)
    return float((index[0] + index[-1]) / 2)


def _components(mask: np.ndarray) -> list[tuple[float, float, int]]:
    """4-connected components. Returns (center_y, center_x, area)."""
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    found: list[tuple[float, float, int]] = []
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or seen[y, x]:
                continue
            stack = [(y, x)]
            seen[y, x] = True
            points: list[tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                points.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if ny < 0 or nx < 0 or ny >= height or nx >= width:
                        continue
                    if seen[ny, nx] or not mask[ny, nx]:
                        continue
                    seen[ny, nx] = True
                    stack.append((ny, nx))
            ys = [p[0] for p in points]
            xs = [p[1] for p in points]
            found.append((float(np.mean(ys)), float(np.mean(xs)), len(points)))
    return found


def detect_balls(
    frame: np.ndarray,
    threshold: float = 0.6,
    min_area: int = 1,
    max_area: int = 12,
    ignore_columns: tuple[int, int] = (6, 58),
) -> list[tuple[float, float, int]]:
    """Bright blobs that are not the paddles. ignore_columns is the open interval of playfield."""
    lum = luminance(frame)
    mask = lum >= threshold
    left, right = ignore_columns
    keep = np.zeros_like(mask)
    keep[:, left:right] = True
    # Bottom paddle band for Breakout is handled by the caller via ignore rows if needed.
    blobs = _components(mask & keep)
    return [blob for blob in blobs if min_area <= blob[2] <= max_area]


def ball_count(frame: np.ndarray) -> int:
    return len(detect_balls(frame))
