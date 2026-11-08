"""Construct models from a Hydra config."""

from __future__ import annotations

from omegaconf import DictConfig

from dreamarcade.models.baseline import BaselineWorldModel
from dreamarcade.models.reward_done import RewardDoneNet
from dreamarcade.models.world import EDMWorldModel
from dreamarcade.train.actor import ActorCritic


def _mult(cfg_model) -> tuple[int, ...]:
    return tuple(int(v) for v in cfg_model.channel_mult)


def _attn(cfg_model) -> tuple[int, ...]:
    return tuple(int(v) for v in cfg_model.attn_resolutions)


def build_baseline(cfg: DictConfig) -> BaselineWorldModel:
    model = cfg.model
    return BaselineWorldModel(
        history_frames=int(cfg.data.history),
        base_channels=int(model.base_channels),
        channel_mult=_mult(model),
        attn_resolutions=_attn(model),
        cond_dim=int(model.cond_dim),
        residual=bool(model.get("residual", True)),
    )


def build_diffusion(cfg: DictConfig) -> EDMWorldModel:
    model = cfg.model
    return EDMWorldModel(
        history_frames=int(cfg.data.history),
        base_channels=int(model.base_channels),
        channel_mult=_mult(model),
        attn_resolutions=_attn(model),
        cond_dim=int(model.cond_dim),
        sigma_data=float(model.sigma_data),
        p_mean=float(model.p_mean),
        p_std=float(model.p_std),
    )


def build_reward(cfg: DictConfig) -> RewardDoneNet:
    reward = cfg.reward
    return RewardDoneNet(
        history_frames=int(cfg.data.history),
        hidden=int(reward.hidden),
        use_gru=bool(reward.use_gru),
    )


def build_actor(cfg: DictConfig | None = None) -> ActorCritic:
    return ActorCritic()
