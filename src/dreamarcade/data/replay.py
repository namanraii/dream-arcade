"""Browser replay format shared with web/src/replay/format.ts.

Layout, little-endian:
  4 bytes magic ASCII "PONG" or "BRCK"
  uint32 T, uint32 H, uint32 W
  T uint8 actions
  T int8 rewards
  T*H*W*3 uint8 RGB frames, frame-major, row-major

initial.bin is a raw 64*64*3 uint8 frame with no header.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def write_replay(
    path: Path,
    frames: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    magic: bytes = b"PONG",
) -> None:
    frames = np.asarray(frames, dtype=np.uint8)
    actions = np.asarray(actions, dtype=np.uint8)
    rewards = np.clip(np.rint(rewards), -128, 127).astype(np.int8)
    if magic not in (b"PONG", b"BRCK"):
        raise ValueError("replay magic must be PONG or BRCK")
    if frames.ndim != 4:
        raise ValueError("frames must be [T,H,W,3]")
    count = frames.shape[0]
    if actions.shape[0] != count or rewards.shape[0] != count:
        raise ValueError("actions and rewards must match the frame count")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames.shape[1], frames.shape[2]
    with path.open("wb") as handle:
        handle.write(magic)
        handle.write(struct.pack("<III", count, height, width))
        handle.write(actions.tobytes())
        handle.write(rewards.tobytes())
        handle.write(np.ascontiguousarray(frames).tobytes())


def write_initial_frame(path: Path, frame: np.ndarray) -> None:
    frame = np.asarray(frame, dtype=np.uint8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(np.ascontiguousarray(frame).tobytes())


def read_replay(path: Path) -> dict:
    raw = Path(path).read_bytes()
    if raw[:4] not in (b"PONG", b"BRCK"):
        raise ValueError("bad replay magic")
    count, height, width = struct.unpack_from("<III", raw, 4)
    if count < 1 or count > 20_000 or not 1 <= height <= 256 or not 1 <= width <= 256:
        raise ValueError("replay dimensions are out of range")
    need = 16 + count + count + count * height * width * 3
    if len(raw) < need:
        raise ValueError("replay truncated")
    cursor = 16
    actions = np.frombuffer(raw, dtype=np.uint8, count=count, offset=cursor).copy()
    cursor += count
    rewards = np.frombuffer(raw, dtype=np.int8, count=count, offset=cursor).copy()
    cursor += count
    frames = np.frombuffer(raw, dtype=np.uint8, count=count * height * width * 3, offset=cursor)
    frames = frames.reshape(count, height, width, 3).copy()
    return {"frames": frames, "actions": actions, "rewards": rewards}
