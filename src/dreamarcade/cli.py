"""One command per stage. Run from the repository root.

    dreamarcade collect
    dreamarcade train-v1
    dreamarcade train-v2
    dreamarcade train-reward
    dreamarcade eval
    dreamarcade play
    dreamarcade export
    dreamarcade parity
    dreamarcade ppo
    dreamarcade train-expert
    dreamarcade distill
    dreamarcade smoke
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from dreamarcade.train.config import load_config
from dreamarcade.utils import repo_root


def _overrides(args: argparse.Namespace) -> list[str]:
    return list(args.override or [])


def cmd_collect(args: argparse.Namespace) -> None:
    from dreamarcade.data.collector import collect_episodes

    cfg = load_config(args.config, _overrides(args))
    stats = collect_episodes(
        out_dir=Path(cfg.out_dir),
        env_name=str(cfg.data.name),
        episodes=int(cfg.episodes),
        episode_length=int(cfg.episode_length),
        seed=int(cfg.seed),
        score_boost=float(cfg.data.score_boost),
        contact_boost=float(cfg.data.contact_boost),
        expert_path=Path(cfg.expert_path) if cfg.expert_path else None,
    )
    print(stats)


def cmd_train_v1(args: argparse.Namespace) -> None:
    from dreamarcade.train.train_v1 import train_v1

    train_v1(load_config(args.config, _overrides(args)))


def cmd_train_v2(args: argparse.Namespace) -> None:
    from dreamarcade.train.train_v2 import train_v2

    train_v2(load_config(args.config, _overrides(args)))


def cmd_train_reward(args: argparse.Namespace) -> None:
    from dreamarcade.train.train_reward import train_reward

    train_reward(load_config(args.config, _overrides(args)))


def cmd_eval(args: argparse.Namespace) -> None:
    from dreamarcade.eval.run_eval import evaluate

    evaluate(load_config(args.config, _overrides(args)))


def cmd_play(args: argparse.Namespace) -> None:
    from dreamarcade.play.local_player import play

    play(load_config(args.config, _overrides(args)))


def cmd_export(args: argparse.Namespace) -> None:
    from dreamarcade.export.to_onnx import export_models

    export_models(load_config(args.config, _overrides(args)))


def cmd_parity(args: argparse.Namespace) -> None:
    from dreamarcade.export.parity_test import run_parity

    cfg = load_config(args.config, _overrides(args))
    diff = run_parity(Path(cfg.denoiser_checkpoint), Path(cfg.onnx_path), cfg)
    print(f"parity max abs diff {diff:.3e}")


def cmd_ppo(args: argparse.Namespace) -> None:
    from dreamarcade.train.ppo_dream import train_ppo_dream

    train_ppo_dream(load_config(args.config, _overrides(args)))


def cmd_expert(args: argparse.Namespace) -> None:
    from dreamarcade.train.train_expert import train_expert

    train_expert(load_config(args.config, _overrides(args)))


def cmd_distill(args: argparse.Namespace) -> None:
    from dreamarcade.train.distill import train_distill

    train_distill(load_config(args.config, _overrides(args)))


def _retarget(cfg, out_dir: str, run_name: str, data_dir: Path):
    from omegaconf import OmegaConf

    cloned = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cloned.out_dir = str(out_dir)
    cloned.run_name = run_name
    cloned.data_dir = str(data_dir)
    return cloned


def cmd_smoke(args: argparse.Namespace) -> None:
    """Tiny CPU/MPS pass through every stage so the browser has a model to load."""
    from dreamarcade.data.collector import collect_episodes
    from dreamarcade.export.parity_test import run_parity
    from dreamarcade.export.to_onnx import export_models
    from dreamarcade.train.train_expert import train_expert
    from dreamarcade.train.train_reward import train_reward
    from dreamarcade.train.train_v1 import train_v1
    from dreamarcade.train.train_v2 import train_v2

    root = repo_root()
    cfg = load_config("smoke", _overrides(args))
    breakout = "breakout" in str(cfg.data.name)
    if breakout:
        data_dir = root / "outputs" / "smoke" / "breakout" / "data"
        v1_out = root / "outputs" / "smoke" / "breakout" / "v1"
        v2_out = root / "outputs" / "smoke" / "breakout" / "v2"
        reward_out = root / "outputs" / "smoke" / "breakout" / "reward"
        expert_out = root / "outputs" / "smoke" / "breakout" / "expert"
        model_dir = root / "web" / "public" / "models" / "breakout"
        replay_name = "breakout.bin"
        initial_name = "breakout-initial.bin"
    else:
        data_dir = root / "outputs" / "smoke" / "data"
        v1_out = Path(cfg.v1_out)
        v2_out = Path(cfg.v2_out)
        reward_out = Path(cfg.reward_out)
        expert_out = Path(cfg.expert_out)
        model_dir = root / "web" / "public" / "models"
        replay_name = "pong.bin"
        initial_name = "initial.bin"
    print("smoke: collect")
    collect_episodes(
        out_dir=data_dir,
        env_name=str(cfg.data.name),
        episodes=int(cfg.episodes),
        episode_length=int(cfg.episode_length),
        seed=int(cfg.seed),
        score_boost=float(cfg.data.score_boost),
        contact_boost=float(cfg.data.contact_boost),
    )
    print("smoke: train v1")
    train_v1(_retarget(cfg, v1_out, "smoke-v1", data_dir))
    print("smoke: train v2")
    train_v2(_retarget(cfg, v2_out, "smoke-v2", data_dir))
    print("smoke: train reward")
    train_reward(_retarget(cfg, reward_out, "smoke-reward", data_dir))
    print("smoke: train expert for a few updates")
    train_expert(_retarget(cfg, expert_out, "smoke-expert", data_dir))
    print("smoke: export")
    export_cfg = load_config(
        "export",
        [
            f"denoiser_checkpoint={v2_out}/latest.pt",
            f"reward_checkpoint={reward_out}/latest.pt",
            f"policy_checkpoint={expert_out}/expert.pt",
            f"out_dir={model_dir.relative_to(root)}",
            "model=tiny",
            "reward=tiny",
        ],
    )
    export_models(export_cfg)
    diff = run_parity(
        Path(v2_out) / "latest.pt",
        model_dir / "denoiser.onnx",
        export_cfg,
    )
    print(f"smoke parity max abs diff {diff:.3e}")
    replay_src = data_dir / "replay.bin"
    initial_src = data_dir / "initial.bin"
    if replay_src.is_file():
        dest = root / "web" / "public" / "replays"
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy(replay_src, dest / replay_name)
        shutil.copy(initial_src, dest / initial_name)
        print(f"copied replay to {dest}")
    print("smoke done")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dreamarcade")
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, fn, default_config: str) -> None:
        cmd = sub.add_parser(name)
        cmd.add_argument("--config", default=default_config)
        cmd.add_argument("override", nargs="*", help="Hydra overrides, for example seed=1")
        cmd.set_defaults(func=fn)

    add("collect", cmd_collect, "collect")
    add("train-v1", cmd_train_v1, "train_v1")
    add("train-v2", cmd_train_v2, "train_v2")
    add("train-reward", cmd_train_reward, "train_reward")
    add("eval", cmd_eval, "eval")
    add("play", cmd_play, "play")
    add("export", cmd_export, "export")
    add("parity", cmd_parity, "export")
    add("ppo", cmd_ppo, "ppo")
    add("train-expert", cmd_expert, "expert")
    add("distill", cmd_distill, "distill")
    add("smoke", cmd_smoke, "smoke")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
