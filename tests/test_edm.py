import math

import torch

from dreamarcade.models.edm import euler_step, loss_weight, precondition, sigma_schedule


def test_precondition_at_unit_sigma():
    sigma = torch.tensor([1.0])
    c_skip, c_out, c_in, c_noise = precondition(sigma, sigma_data=0.5)
    assert torch.allclose(c_skip, torch.tensor([0.2]))
    assert torch.allclose(c_out, torch.tensor([0.5 / math.sqrt(1.25)]))
    assert torch.allclose(c_in, torch.tensor([1.0 / math.sqrt(1.25)]))
    assert torch.allclose(c_noise, torch.tensor([0.0]))


def test_loss_weight_matches_edm():
    sigma = torch.tensor([1.0])
    # (1 + 0.25) / (0.5 ** 2) = 5
    assert torch.allclose(loss_weight(sigma, 0.5), torch.tensor([5.0]))


def test_sigma_schedule_endpoints():
    one = sigma_schedule(1)
    assert one.shape == (2,)
    assert torch.allclose(one[0], torch.tensor(80.0))
    assert one[-1].item() == 0.0
    two = sigma_schedule(2, sigma_min=0.002, sigma_max=80.0, rho=7.0)
    assert torch.allclose(two, torch.tensor([80.0, 0.002, 0.0]), atol=1e-6)
    three = sigma_schedule(3)
    assert three.shape == (4,)
    assert torch.allclose(three[0], torch.tensor(80.0))
    assert torch.allclose(three[2], torch.tensor(0.002), atol=1e-6)
    assert three[-1].item() == 0.0


def test_euler_step_known_case():
    x = torch.tensor([1.0, 2.0, 3.0])
    denoised = torch.zeros(3)
    out = euler_step(x, denoised, 2.0, 1.0)
    assert torch.allclose(out, torch.tensor([0.5, 1.0, 1.5]))
