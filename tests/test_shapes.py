import torch

from dreamarcade.models.baseline import BaselineWorldModel
from dreamarcade.models.edm import precondition
from dreamarcade.models.reward_done import RewardDoneNet
from dreamarcade.models.unet import ConditionalUNet
from dreamarcade.models.world import EDMWorldModel, sample_next_frame
from dreamarcade.train.actor import ActorCritic


def test_unet_and_models_match_the_onnx_contract():
    torch.manual_seed(0)
    history = torch.randn(2, 12, 64, 64)
    action = torch.tensor([2, 3])
    baseline = BaselineWorldModel(
        base_channels=8, channel_mult=(1, 2), attn_resolutions=(32,), cond_dim=32
    )
    pred = baseline(history, action)
    assert pred.shape == (2, 3, 64, 64)

    world = EDMWorldModel(
        base_channels=8, channel_mult=(1, 2), attn_resolutions=(32,), cond_dim=32
    )
    noisy = torch.randn(2, 3, 64, 64)
    sigma = torch.tensor([0.8, 1.4])
    denoised = world.denoise(history, action, noisy, sigma)
    assert denoised.shape == (2, 3, 64, 64)
    loss = world.loss(history, action, torch.randn(2, 3, 64, 64))
    assert torch.isfinite(loss)

    frame = sample_next_frame(world, history[:1], action[:1], steps=1, sigma_max=1.0)
    assert frame.shape == (1, 3, 64, 64)
    assert float(frame.min()) >= -1
    assert float(frame.max()) <= 1

    reward = RewardDoneNet(hidden=32)
    logits, done = reward(history, action)
    assert logits.shape == (2, 3)
    assert done.shape == (2, 1)

    actor = ActorCritic()
    logits, value = actor(history)
    assert logits.shape == (2, 6)
    assert value.shape == (2,)


def test_zero_network_matches_skip_connection():
    """With F = 0, D = c_skip * noisy. The U-Net output conv is zero-initialised."""
    world = EDMWorldModel(
        base_channels=8, channel_mult=(1,), attn_resolutions=(), cond_dim=16
    )
    history = torch.zeros(1, 12, 64, 64)
    action = torch.tensor([0])
    noisy = torch.randn(1, 3, 64, 64)
    sigma = torch.tensor([1.0])
    denoised = world.denoise(history, action, noisy, sigma)
    c_skip, _, _, _ = precondition(sigma.view(1, 1, 1, 1), world.sigma_data)
    assert torch.allclose(denoised, c_skip * noisy, atol=1e-5)


def test_full_config_fits_the_fp16_budget():
    model = EDMWorldModel(
        base_channels=64,
        channel_mult=(1, 2, 2, 2),
        attn_resolutions=(16, 8),
        cond_dim=256,
    )
    count = sum(p.numel() for p in model.parameters())
    # fp16 is 2 bytes per parameter. The plan asks for about 5-15M and under 30 MB.
    assert 5_000_000 < count < 15_000_000, count
    assert count * 2 < 30 * 1024 * 1024


def test_attention_levels_exist_on_the_full_unet():
    unet = ConditionalUNet(
        in_channels=15,
        out_channels=3,
        base_channels=64,
        channel_mult=(1, 2, 2, 2),
        attn_resolutions=(16, 8),
    )
    from dreamarcade.models.unet import SelfAttention

    attentive = [m for m in unet.modules() if isinstance(m, SelfAttention)]
    assert len(attentive) >= 2
