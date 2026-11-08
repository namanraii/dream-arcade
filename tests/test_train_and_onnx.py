from pathlib import Path

import torch

from dreamarcade.data.collector import collect_episodes
from dreamarcade.export.parity_test import run_parity
from dreamarcade.export.to_onnx import export_models
from dreamarcade.train.config import load_config
from dreamarcade.train.train_v1 import train_v1
from dreamarcade.train.train_v2 import train_v2


def test_one_step_and_onnx_parity(tmp_path: Path):
    data = tmp_path / "data"
    collect_episodes(data, episodes=3, episode_length=20, seed=0)
    overrides = [
        f"data_dir={data}",
        "steps=1",
        "batch_size=2",
        "log_every=1",
        "save_every=1",
        "device=cpu",
        "model=tiny",
        "reward=tiny",
        "warmup=0",
        "perceptual_weight=0",
        "scheduled_sampling=false",
        "wandb=false",
    ]
    v1_dir = tmp_path / "v1"
    v2_dir = tmp_path / "v2"
    train_v1(load_config("train_v1", overrides + [f"out_dir={v1_dir}"]))
    assert (v1_dir / "latest.pt").is_file()
    train_v2(load_config("train_v2", overrides + [f"out_dir={v2_dir}"]))
    assert (v2_dir / "latest.pt").is_file()
    export_cfg = load_config(
        "export",
        [
            f"denoiser_checkpoint={v2_dir / 'latest.pt'}",
            "reward_checkpoint=null",
            "policy_checkpoint=null",
            f"out_dir={tmp_path / 'onnx'}",
            "model=tiny",
            "reward=tiny",
        ],
    )
    written = export_models(export_cfg)
    diff = run_parity(v2_dir / "latest.pt", written["denoiser"], export_cfg, atol=1e-4, rtol=1e-4)
    assert diff < 1e-4
    blob = torch.load(v2_dir / "latest.pt", map_location="cpu", weights_only=False)
    assert "ema" in blob and blob["step"] == 1
