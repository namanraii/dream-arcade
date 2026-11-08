from pathlib import Path

import numpy as np
import torch

from dreamarcade.data.collector import collect_episodes, split_for
from dreamarcade.data.dataset import FrameDataset, pixel_loss_weight
from dreamarcade.data.replay import read_replay


def test_split_is_by_episode():
    assert split_for(0) == "test"
    assert split_for(1) == "val"
    assert split_for(2) == "train"
    assert split_for(10) == "test"


def test_collect_and_dataset_roundtrip(tmp_path: Path):
    stats = collect_episodes(
        tmp_path,
        env_name="original_pong",
        episodes=4,
        episode_length=16,
        seed=0,
    )
    assert stats["frames"] > 0
    assert sum(stats["action_hist"]) == stats["frames"]
    dataset = FrameDataset(tmp_path, split="train", history=4, horizon=1)
    assert len(dataset) > 0
    sample = dataset[0]
    assert sample["history"].shape == (12, 64, 64)
    assert sample["target"].shape == (1, 3, 64, 64)
    assert sample["action"].shape == (1,)
    assert float(sample["history"].min()) >= -1.01
    assert float(sample["history"].max()) <= 1.01
    replay = read_replay(tmp_path / "replay.bin")
    assert replay["frames"].shape[0] == replay["actions"].shape[0]
    assert replay["frames"].shape[1:] == (64, 64, 3)
    initial = np.frombuffer((tmp_path / "initial.bin").read_bytes(), dtype=np.uint8)
    assert initial.size == 64 * 64 * 3


def test_pixel_weight_boosts_the_moving_object():
    history = torch.zeros(1, 12, 16, 16)
    target = torch.zeros(1, 3, 16, 16)
    target[0, :, 4, 5] = 1
    weight = pixel_loss_weight(history, target, alpha=4)
    assert weight.shape == (1, 1, 16, 16)
    assert weight[0, 0, 4, 5] > weight[0, 0, 0, 0]


def test_score_frames_are_oversampled(tmp_path: Path):
    collect_episodes(tmp_path, episodes=6, episode_length=48, seed=1)
    dataset = FrameDataset(tmp_path, split="all", history=4)
    weights = np.asarray(dataset.sample_weights)
    assert weights.max() > 1
