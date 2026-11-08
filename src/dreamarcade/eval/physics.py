"""Ball permanence and a reflection check against the law of reflection.

A bounce counts when the ball is near a wall or the agent paddle and the
velocity component that should flip does flip. The outgoing angle must lie
within `tolerance_deg` of the reflected incoming angle.
"""

from __future__ import annotations

import math

import numpy as np

from dreamarcade.data.detectors import detect_balls


def _angle(dx: float, dy: float) -> float:
    return math.atan2(dy, dx)


def _track(frames: list[np.ndarray]) -> list[tuple[float, float] | None]:
    centers: list[tuple[float, float] | None] = []
    previous: tuple[float, float] | None = None
    for frame in frames:
        blobs = detect_balls(frame)
        if len(blobs) != 1:
            centers.append(None)
            previous = None
            continue
        cy, cx = blobs[0][0], blobs[0][1]
        if previous is not None and abs(cy - previous[0]) + abs(cx - previous[1]) > 20:
            centers.append(None)
            previous = None
            continue
        centers.append((cy, cx))
        previous = (cy, cx)
    return centers


def physics_report(frames: list[np.ndarray], tolerance_deg: float = 20.0) -> dict:
    counts = [len(detect_balls(frame)) for frame in frames]
    single = sum(count == 1 for count in counts)
    centers = _track(frames)
    checked = 0
    ok = 0
    for index in range(2, len(centers) - 1):
        c0, c1, c2 = centers[index - 1], centers[index], centers[index + 1]
        if c0 is None or c1 is None or c2 is None:
            continue
        vin = (c1[1] - c0[1], c1[0] - c0[0])
        vout = (c2[1] - c1[1], c2[0] - c1[0])
        speed = math.hypot(*vin)
        if speed < 0.4 or math.hypot(*vout) < 0.4:
            continue
        near_wall = c1[0] < 4 or c1[0] > 59
        near_paddle = c1[1] > 55
        if not near_wall and not near_paddle:
            continue
        if near_wall:
            expected = (vin[0], -vin[1])
        else:
            expected = (-vin[0], vin[1])
        err = abs(_angle(*expected) - _angle(*vout))
        err = min(err, 2 * math.pi - err)
        checked += 1
        if math.degrees(err) <= tolerance_deg:
            ok += 1
    return {
        "frames": len(frames),
        "single_ball_fraction": single / max(len(frames), 1),
        "bounces_checked": checked,
        "bounce_plausible_fraction": (ok / checked) if checked else None,
        "tolerance_deg": tolerance_deg,
    }
