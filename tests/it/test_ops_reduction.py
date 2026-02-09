"""Reduction operation correctness tests.

Tests sum, mean, argmax, max, min, prod, any, all (global + dim variants).
"""

import pytest
import torch


# =============================================================================
# Reduction forward tests
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "op_name,fn",
    [
        ("sum_global", lambda x: x.sum()),
        ("sum_dim0", lambda x: x.sum(dim=0)),
        ("sum_dim1", lambda x: x.sum(dim=1)),
        ("sum_dim1_keepdim", lambda x: x.sum(dim=1, keepdim=True)),
        ("mean_global", lambda x: x.mean()),
        ("mean_dim0", lambda x: x.mean(dim=0)),
        ("mean_dim1", lambda x: x.mean(dim=1)),
        ("mean_dim1_keepdim", lambda x: x.mean(dim=1, keepdim=True)),
    ],
)
async def test_reduction_forward(device, op_name, fn):
    x_cpu = torch.randn(4, 6)
    cpu_result = fn(x_cpu)
    sky_result = fn(x_cpu.to(device))
    torch.testing.assert_close(sky_result.cpu(), cpu_result, check_device=False)


@pytest.mark.it
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "op_name,fn",
    [
        ("argmax_global", lambda x: x.argmax()),
        ("argmax_dim0", lambda x: x.argmax(dim=0)),
        ("argmax_dim1", lambda x: x.argmax(dim=1)),
    ],
)
async def test_argmax_forward(device, op_name, fn):
    x_cpu = torch.randn(4, 6)
    cpu_result = fn(x_cpu)
    sky_result = fn(x_cpu.to(device))
    torch.testing.assert_close(sky_result.cpu(), cpu_result, check_device=False)


@pytest.mark.it
@pytest.mark.asyncio
async def test_max_global_forward(device):
    x_cpu = torch.randn(4, 6)
    cpu_result = x_cpu.max()
    sky_result = x_cpu.to(device).max()
    torch.testing.assert_close(sky_result.cpu(), cpu_result, check_device=False)


@pytest.mark.it
@pytest.mark.asyncio
@pytest.mark.parametrize("dim", [0, 1])
async def test_max_dim_forward(device, dim):
    x_cpu = torch.randn(4, 6)
    cpu_values, cpu_indices = x_cpu.max(dim=dim)
    sky_values, sky_indices = x_cpu.to(device).max(dim=dim)
    torch.testing.assert_close(sky_values.cpu(), cpu_values, check_device=False)
    torch.testing.assert_close(sky_indices.cpu(), cpu_indices, check_device=False)


@pytest.mark.it
@pytest.mark.asyncio
@pytest.mark.parametrize("dim", [0, 1])
async def test_min_dim_forward(device, dim):
    x_cpu = torch.randn(4, 6)
    cpu_values, cpu_indices = x_cpu.min(dim=dim)
    sky_values, sky_indices = x_cpu.to(device).min(dim=dim)
    torch.testing.assert_close(sky_values.cpu(), cpu_values, check_device=False)
    torch.testing.assert_close(sky_indices.cpu(), cpu_indices, check_device=False)


# =============================================================================
# Reduction gradient tests
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "op_name,fn",
    [
        ("sum_global", lambda x: x.sum()),
        ("sum_dim0", lambda x: x.sum(dim=0).sum()),
        ("sum_dim1", lambda x: x.sum(dim=1).sum()),
        ("mean_global", lambda x: x.mean()),
        ("mean_dim0", lambda x: x.mean(dim=0).sum()),
        ("mean_dim1", lambda x: x.mean(dim=1).sum()),
    ],
)
async def test_reduction_grad(device, op_name, fn):
    x_cpu = torch.randn(4, 6, requires_grad=True)
    x_sky = x_cpu.clone().to(device).detach().requires_grad_(True)

    loss_cpu = fn(x_cpu)
    loss_cpu.backward()

    loss_sky = fn(x_sky)
    loss_sky.backward()

    assert x_sky.grad is not None, f"{op_name} gradient is None"
    torch.testing.assert_close(x_sky.grad.cpu(), x_cpu.grad, check_device=False)


@pytest.mark.it
@pytest.mark.asyncio
async def test_max_dim_grad(device):
    """max.dim returns (values, indices) — only values are differentiable."""
    x_cpu = torch.randn(4, 6, requires_grad=True)
    x_sky = x_cpu.clone().to(device).detach().requires_grad_(True)

    values_cpu, _ = x_cpu.max(dim=1)
    loss_cpu = values_cpu.sum()
    loss_cpu.backward()

    values_sky, _ = x_sky.max(dim=1)
    loss_sky = values_sky.sum()
    loss_sky.backward()

    assert x_sky.grad is not None, "max.dim gradient is None"
    torch.testing.assert_close(x_sky.grad.cpu(), x_cpu.grad, check_device=False)


# =============================================================================
# Additional reduction tests
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_prod_forward(device):
    x_cpu = torch.randn(3, 4)
    cpu_result = x_cpu.prod()
    sky_result = x_cpu.to(device).prod()
    torch.testing.assert_close(
        sky_result.cpu(), cpu_result, atol=1e-4, rtol=1e-4, check_device=False
    )


@pytest.mark.it
@pytest.mark.asyncio
async def test_any_all_forward(device):
    x_cpu = torch.tensor([True, False, True, True])

    cpu_any = x_cpu.any()
    sky_any = x_cpu.to(device).any()
    assert sky_any.cpu().item() == cpu_any.item()

    cpu_all = x_cpu.all()
    sky_all = x_cpu.to(device).all()
    assert sky_all.cpu().item() == cpu_all.item()
