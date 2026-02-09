"""Optimizer and training loop integration tests.

Tests SGD/Adam optimizer steps, zero_grad, multi-step training,
train/eval mode, gradient clipping, parameter freezing, and model.to(device).
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Optimizer step tests
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_sgd_optimizer_step(device):
    """SGD optimizer step on a simple linear model updates parameters."""
    torch.manual_seed(42)

    # CPU reference
    model_cpu = nn.Linear(4, 2)
    opt_cpu = torch.optim.SGD(model_cpu.parameters(), lr=0.1)
    x_cpu = torch.randn(3, 4)

    # Sky
    model_sky = nn.Linear(4, 2)
    model_sky.load_state_dict(model_cpu.state_dict())
    model_sky = model_sky.to(device)
    opt_sky = torch.optim.SGD(model_sky.parameters(), lr=0.1)
    x_sky = x_cpu.to(device)

    # Save initial parameters
    w_before = model_sky.weight.cpu().clone()

    # Forward + backward + step
    loss_cpu = model_cpu(x_cpu).sum()
    loss_cpu.backward()
    opt_cpu.step()

    loss_sky = model_sky(x_sky).sum()
    loss_sky.backward()
    opt_sky.step()

    # Parameters should have changed
    w_after = model_sky.weight.cpu()
    assert not torch.allclose(w_before, w_after), "Parameters did not change after step"

    # Compare with CPU reference
    torch.testing.assert_close(
        w_after, model_cpu.weight, atol=1e-4, rtol=1e-4, check_device=False
    )
    torch.testing.assert_close(
        model_sky.bias.cpu(),
        model_cpu.bias,
        atol=1e-4,
        rtol=1e-4,
        check_device=False,
    )


@pytest.mark.it
@pytest.mark.asyncio
async def test_adam_optimizer_step(device):
    """Adam optimizer step updates parameters and creates moment buffers."""
    torch.manual_seed(42)

    model_cpu = nn.Linear(4, 2)
    opt_cpu = torch.optim.Adam(model_cpu.parameters(), lr=0.01)
    x_cpu = torch.randn(3, 4)

    model_sky = nn.Linear(4, 2)
    model_sky.load_state_dict(model_cpu.state_dict())
    model_sky = model_sky.to(device)
    opt_sky = torch.optim.Adam(model_sky.parameters(), lr=0.01)
    x_sky = x_cpu.to(device)

    w_before = model_sky.weight.cpu().clone()

    loss_cpu = model_cpu(x_cpu).sum()
    loss_cpu.backward()
    opt_cpu.step()

    loss_sky = model_sky(x_sky).sum()
    loss_sky.backward()
    opt_sky.step()

    w_after = model_sky.weight.cpu()
    assert not torch.allclose(w_before, w_after), "Parameters did not change after step"

    torch.testing.assert_close(
        w_after, model_cpu.weight, atol=1e-4, rtol=1e-4, check_device=False
    )


# =============================================================================
# Zero grad
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_optimizer_zero_grad(device):
    """optimizer.zero_grad() zeros all gradients."""
    model = nn.Linear(4, 2).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.randn(3, 4).to(device)

    # Create gradients
    loss = model(x).sum()
    loss.backward()

    # Gradients should be non-zero
    assert model.weight.grad is not None
    assert model.weight.grad.abs().sum().item() > 0

    # Zero gradients (set_to_none=False to keep zero tensors instead of None)
    opt.zero_grad(set_to_none=False)

    # Gradients should be zero
    assert model.weight.grad is not None
    torch.testing.assert_close(
        model.weight.grad.cpu(),
        torch.zeros_like(model.weight.cpu()),
        check_device=False,
    )


# =============================================================================
# Gradient clipping
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_clip_grad_norm(device):
    """torch.nn.utils.clip_grad_norm_() clips gradients."""
    model_cpu = nn.Linear(4, 2)
    model_sky = nn.Linear(4, 2)
    model_sky.load_state_dict(model_cpu.state_dict())
    model_sky = model_sky.to(device)

    x_cpu = torch.randn(3, 4)
    x_sky = x_cpu.to(device)

    # Create large gradients
    loss_cpu = (model_cpu(x_cpu) * 100).sum()
    loss_cpu.backward()

    loss_sky = (model_sky(x_sky) * 100).sum()
    loss_sky.backward()

    # Clip
    max_norm = 1.0
    norm_cpu = nn.utils.clip_grad_norm_(model_cpu.parameters(), max_norm)
    norm_sky = nn.utils.clip_grad_norm_(model_sky.parameters(), max_norm)

    # Norms should match (move to CPU for comparison since norm may be on sky)
    if isinstance(norm_sky, torch.Tensor):
        norm_sky_val = norm_sky.cpu().item()
    else:
        norm_sky_val = float(norm_sky)
    if isinstance(norm_cpu, torch.Tensor):
        norm_cpu_val = norm_cpu.cpu().item()
    else:
        norm_cpu_val = float(norm_cpu)
    assert abs(norm_sky_val - norm_cpu_val) < 1e-3

    # Clipped gradients should match
    torch.testing.assert_close(
        model_sky.weight.grad.cpu(),
        model_cpu.weight.grad,
        atol=1e-4,
        rtol=1e-4,
        check_device=False,
    )


# =============================================================================
# Parameter freezing
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_parameter_freezing(device):
    """Setting requires_grad=False excludes parameter from optimization."""
    model = nn.Linear(4, 2).to(device)
    model.weight.requires_grad_(False)  # Freeze weight

    x = torch.randn(3, 4).to(device)
    loss = model(x).sum()
    loss.backward()

    # Weight gradient should be None (frozen)
    assert model.weight.grad is None
    # Bias gradient should exist
    assert model.bias.grad is not None


# =============================================================================
# model.to(device)
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_model_to_device(device):
    """model.to(device) moves all parameters to the device."""
    model = nn.Sequential(
        nn.Linear(8, 4),
        nn.ReLU(),
        nn.Linear(4, 2),
    )

    model = model.to(device)

    for param in model.parameters():
        assert param.device.type == "sky", f"Parameter still on {param.device}"


# =============================================================================
# Train/eval mode
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_dropout_train_vs_eval(device):
    """Dropout is active in train mode, inactive in eval mode."""
    x = torch.ones(100, 100).to(device)

    # Train mode: dropout zeros some elements
    result_train = F.dropout(x, p=0.5, training=True)
    n_zeros_train = (result_train.cpu() == 0).sum().item()
    assert n_zeros_train > 0, "Dropout should zero some elements in train mode"

    # Eval mode: dropout is identity
    result_eval = F.dropout(x, p=0.5, training=False)
    torch.testing.assert_close(result_eval.cpu(), x.cpu(), check_device=False)
