"""Load a Hydra config from the repo configs directory."""

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf

from dreamarcade.utils import repo_root


def load_config(name: str, overrides: list[str] | None = None) -> DictConfig:
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    config_dir = repo_root() / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=name, overrides=list(overrides or []))
    return cfg


def to_plain(cfg: DictConfig) -> dict:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
