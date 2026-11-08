"""One command: fidelity, drift, controllability, physics, and a RESULTS.md row."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig

from dreamarcade.data.dataset import FrameDataset
from dreamarcade.eval.controllability import controllability
from dreamarcade.eval.physics import physics_report
from dreamarcade.eval.rollouts import load_episode, rollout_episode
from dreamarcade.models.reward_done import reward_to_class
from dreamarcade.train.build import build_baseline, build_diffusion, build_reward
from dreamarcade.utils import load_checkpoint, pick_device, repo_root, set_seed


def _load_world(cfg: DictConfig, device: torch.device):
    kind = str(cfg.model.kind)
    if kind == "v1":
        model = build_baseline(cfg).to(device)
    else:
        model = build_diffusion(cfg).to(device)
    load_checkpoint(Path(cfg.checkpoint), model, use_ema=True)
    model.eval()
    return model, kind


def _episode_paths(data_dir: Path, split: str, limit: int) -> list[Path]:
    manifest = json.loads((data_dir / "manifest.json").read_text())
    paths = []
    for episode in manifest["episodes"]:
        if episode["split"] != split:
            continue
        paths.append(data_dir / "shards" / episode["file"])
        if len(paths) >= limit:
            break
    return paths


def reward_scores(cfg: DictConfig, device: torch.device) -> dict | None:
    if not cfg.reward_checkpoint:
        return None
    path = Path(str(cfg.reward_checkpoint))
    if not path.is_file():
        return None
    net = build_reward(cfg).to(device)
    load_checkpoint(path, net, use_ema=True)
    net.eval()
    dataset = FrameDataset(Path(cfg.data_dir), split="test", history=int(cfg.data.history))
    preds = []
    targets = []
    with torch.no_grad():
        for index in range(min(len(dataset), int(cfg.reward_batches) * int(cfg.batch_size))):
            sample = dataset[index]
            logits, _done = net(sample["history"].unsqueeze(0).to(device), sample["action"][:1].to(device))
            preds.append(int(logits.argmax(dim=-1).item()))
            targets.append(int(reward_to_class(sample["reward"][:1]).item()))
    if not targets:
        return None
    from dreamarcade.eval.metrics import precision_recall

    return precision_recall(np.asarray(preds), np.asarray(targets))


def evaluate(cfg: DictConfig) -> dict:
    set_seed(int(cfg.seed))
    device = pick_device(str(cfg.device))
    model, kind = _load_world(cfg, device)
    data_dir = Path(cfg.data_dir)
    horizons = tuple(int(h) for h in cfg.horizons)
    per_episode = []
    dreamed_frames: list[np.ndarray] = []
    for path in _episode_paths(data_dir, "test", int(cfg.episodes)):
        frames, actions = load_episode(path)
        report = rollout_episode(
            model,
            kind,
            frames,
            actions,
            history=int(cfg.data.history),
            sampler_steps=int(cfg.sampler_steps),
            horizons=horizons,
            device=device,
        )
        per_episode.append({"teacher_forced": report["teacher_forced"], "drift": report["drift"]})
        dreamed_frames.extend(report["teacher_frames"][:40])
    summary = _aggregate(per_episode)
    summary["controllability"] = controllability(
        model,
        kind,
        env_name=str(cfg.data.name),
        trials=int(cfg.control_trials),
        sampler_steps=int(cfg.sampler_steps),
        seed=int(cfg.seed),
        device=device,
    )
    if dreamed_frames and str(cfg.data.get("physics", "pong")) == "pong":
        summary["physics"] = physics_report(dreamed_frames)
    else:
        summary["physics"] = {"note": "reflection check is implemented for Pong"}
    summary["reward"] = reward_scores(cfg, device)
    summary["kind"] = kind
    summary["checkpoint"] = str(cfg.checkpoint)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    _plot_drift(summary, out / "drift.png")
    _update_results(summary, cfg)
    print(json.dumps({k: summary[k] for k in ("kind", "checkpoint") if k in summary}, indent=2))
    _print_table(summary)
    return summary


def _aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {"teacher_forced": {}, "drift": {}}
    teacher_keys = rows[0]["teacher_forced"].keys()
    teacher = {
        key: float(np.mean([row["teacher_forced"][key] for row in rows if key in row["teacher_forced"]]))
        for key in teacher_keys
    }
    horizons = sorted({h for row in rows for h in row["drift"]}, key=int)
    drift = {}
    for horizon in horizons:
        present = [row["drift"][horizon] for row in rows if horizon in row["drift"]]
        if not present:
            continue
        keys = present[0].keys()
        drift[horizon] = {}
        for key in keys:
            values = [item[key] for item in present]
            drift[horizon][key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
    return {"teacher_forced": teacher, "drift": drift, "episodes": len(rows)}


def _plot_drift(summary: dict, path: Path) -> None:
    drift = summary.get("drift") or {}
    if not drift:
        return
    horizons = [int(h) for h in drift]
    fig, ax = plt.subplots(figsize=(6, 4))
    for metric in ("mae", "lpips", "psnr"):
        if metric not in next(iter(drift.values())):
            continue
        means = [drift[str(h)][metric]["mean"] for h in horizons]
        stds = [drift[str(h)][metric]["std"] for h in horizons]
        ax.plot(horizons, means, marker="o", label=metric)
        ax.fill_between(
            horizons,
            np.array(means) - np.array(stds),
            np.array(means) + np.array(stds),
            alpha=0.15,
        )
    ax.set_xlabel("horizon (frames)")
    ax.set_ylabel("metric")
    ax.set_title("Free-running drift")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _print_table(summary: dict) -> None:
    print("teacher-forced", summary.get("teacher_forced"))
    for horizon, metrics in (summary.get("drift") or {}).items():
        bits = " ".join(
            f"{name}={values['mean']:.3f}±{values['std']:.3f}" for name, values in metrics.items()
        )
        print(f"h={horizon} {bits}")
    print("controllability", summary.get("controllability"))
    print("physics", summary.get("physics"))
    print("reward", summary.get("reward"))


def _update_results(summary: dict, cfg: DictConfig) -> None:
    path = repo_root() / "RESULTS.md"
    if not path.is_file():
        path.write_text("# Results\n\nNumbers below are produced by `dreamarcade eval`. Do not fill them in by hand.\n\n")
    text = path.read_text()
    start = "<!-- RESULTS:START -->"
    end = "<!-- RESULTS:END -->"
    drift = summary.get("drift") or {}
    lines = [
        start,
        "",
        f"Checkpoint `{summary.get('checkpoint')}` ({summary.get('kind')}), {summary.get('episodes', 0)} test episodes.",
        "",
        "| Horizon | " + " | ".join(sorted({k for v in drift.values() for k in v})) + " |",
    ]
    metric_names = sorted({k for v in drift.values() for k in v})
    if metric_names:
        lines.append("|---|" + "|".join(["---"] * len(metric_names)) + "|")
        for horizon in sorted(drift, key=int):
            cells = []
            for name in metric_names:
                values = drift[horizon][name]
                cells.append(f"{values['mean']:.3f} ± {values['std']:.3f}")
            lines.append(f"| {horizon} | " + " | ".join(cells) + " |")
    else:
        lines.append("No drift points. The test episodes were shorter than the requested horizons.")
    lines.append("")
    lines.append(f"Controllability: `{json.dumps(summary.get('controllability'))}`")
    lines.append("")
    lines.append(f"Physics: `{json.dumps(summary.get('physics'))}`")
    lines.append("")
    lines.append(f"Reward head: `{json.dumps(summary.get('reward'))}`")
    lines.append("")
    lines.append(end)
    block = "\n".join(lines) + "\n"
    if start in text and end in text:
        pre, _rest = text.split(start, 1)
        _mid, post = text.split(end, 1)
        text = pre + block + post.lstrip("\n")
    else:
        text = text.rstrip() + "\n\n" + block
    path.write_text(text)
    print(f"updated {path}")
