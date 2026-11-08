"""Write one compressed episode per file, plus a manifest.

Each episode file stores frames[T+1], actions[T], rewards[T], dones[T],
weights[T]. Frame 0 is the reset observation. Action t produces frame t+1
and reward t. Splits are by episode.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dreamarcade.data.env import make_env
from dreamarcade.data.policies import mix_policies
from dreamarcade.data.replay import write_initial_frame, write_replay


def split_for(episode_id: int) -> str:
    bucket = episode_id % 10
    if bucket == 0:
        return "test"
    if bucket == 1:
        return "val"
    return "train"


def frame_weight(reward: float, contact: bool, score_boost: float, contact_boost: float) -> float:
    weight = 1.0
    if reward != 0:
        weight *= score_boost
    if contact:
        weight *= contact_boost
    return weight


def collect_episodes(
    out_dir: Path,
    env_name: str = "original_pong",
    episodes: int = 8,
    episode_length: int = 128,
    seed: int = 0,
    score_boost: float = 8.0,
    contact_boost: float = 4.0,
    expert_path: Path | None = None,
) -> dict:
    out_dir = Path(out_dir)
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    manifest_episodes: list[dict] = []
    action_hist = np.zeros(6, dtype=np.int64)
    reward_counts = {"-1": 0, "0": 0, "1": 0}
    contact_frames = 0
    total_frames = 0
    returns: list[float] = []
    preview: dict | None = None

    for episode_id in range(episodes):
        env = make_env(env_name, seed=seed + episode_id, max_steps=episode_length)
        policy_name, policy = mix_policies(
            seed + episode_id,
            expert_path=expert_path,
            side=getattr(env, "paddle_side", "right"),
        )
        if hasattr(policy, "reset"):
            policy.reset()
        obs, _info = env.reset()
        frames = [obs]
        actions: list[int] = []
        rewards: list[float] = []
        dones: list[bool] = []
        weights: list[float] = []
        total_return = 0.0
        done = False
        for _step in range(episode_length):
            if done:
                break
            action = int(policy.act(obs, env))
            obs, reward, done, info = env.step(action)
            contact = bool(info.get("contact", False))
            frames.append(obs)
            actions.append(action)
            rewards.append(float(reward))
            dones.append(bool(done))
            weights.append(frame_weight(float(reward), contact, score_boost, contact_boost))
            action_hist[action] += 1
            key = "0" if reward == 0 else ("1" if reward > 0 else "-1")
            reward_counts[key] += 1
            contact_frames += int(contact)
            total_return += float(reward)
            total_frames += 1
        if hasattr(env, "close"):
            env.close()
        returns.append(total_return)
        frame_arr = np.stack(frames).astype(np.uint8)
        action_arr = np.asarray(actions, dtype=np.uint8)
        reward_arr = np.asarray(rewards, dtype=np.float32)
        name = f"episode_{episode_id:05d}.npz"
        np.savez_compressed(
            shard_dir / name,
            frames=frame_arr,
            actions=action_arr,
            rewards=reward_arr,
            dones=np.asarray(dones, dtype=np.uint8),
            weights=np.asarray(weights, dtype=np.float32),
        )
        record = {
            "id": episode_id,
            "file": name,
            "length": int(action_arr.shape[0]),
            "split": split_for(episode_id),
            "policy": policy_name,
            "return": total_return,
        }
        manifest_episodes.append(record)
        if preview is None and record["split"] == "test":
            preview = {
                "frames": frame_arr[1:],
                "actions": action_arr,
                "rewards": reward_arr,
                "initial": frame_arr[0],
            }
        elif preview is None and episode_id == episodes - 1:
            preview = {
                "frames": frame_arr[1:],
                "actions": action_arr,
                "rewards": reward_arr,
                "initial": frame_arr[0],
            }

    if preview is not None:
        magic = b"BRCK" if "breakout" in env_name else b"PONG"
        write_replay(
            out_dir / "replay.bin",
            preview["frames"],
            preview["actions"],
            preview["rewards"],
            magic=magic,
        )
        write_initial_frame(out_dir / "initial.bin", preview["initial"])

    stats = {
        "env": env_name,
        "episodes": episodes,
        "frames": int(total_frames),
        "action_hist": action_hist.tolist(),
        "reward_counts": reward_counts,
        "contact_frames": int(contact_frames),
        "contact_rate": float(contact_frames / max(total_frames, 1)),
        "return_mean": float(np.mean(returns) if returns else 0.0),
        "return_std": float(np.std(returns) if returns else 0.0),
        "expert": "checkpoint" if expert_path and Path(expert_path).is_file() else "scripted",
    }
    manifest = {"image_size": 64, "env": env_name, "episodes": manifest_episodes, "stats": stats}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats
