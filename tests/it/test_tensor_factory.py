"""On-device tensor creation via factory functions.

Tests torch.zeros, torch.ones, torch.full, torch.empty, torch.randn, torch.rand,
torch.arange with device= parameter, plus _like variants and dtype/requires_grad options.
"""

import pytest
import torch


# =============================================================================
# Basic factory functions with device=
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_zeros_on_device(device):
    """torch.zeros(shape, device=device) creates all-zeros tensor."""
    result = torch.zeros(3, 4, device=device)

    assert result.shape == (3, 4)
    assert result.dtype == torch.float32
    torch.testing.assert_close(
        result.cpu(), torch.zeros(3, 4), check_device=False
    )


@pytest.mark.it
@pytest.mark.asyncio
async def test_ones_on_device(device):
    """torch.ones(shape, device=device) creates all-ones tensor."""
    result = torch.ones(3, 4, device=device)

    assert result.shape == (3, 4)
    assert result.dtype == torch.float32
    torch.testing.assert_close(
        result.cpu(), torch.ones(3, 4), check_device=False
    )


@pytest.mark.it
@pytest.mark.asyncio
async def test_full_on_device(device):
    """torch.full(shape, fill_value, device=device) creates correctly filled tensor."""
    result = torch.full((3, 4), 7.5, device=device)

    assert result.shape == (3, 4)
    torch.testing.assert_close(
        result.cpu(), torch.full((3, 4), 7.5), check_device=False
    )


@pytest.mark.it
@pytest.mark.asyncio
async def test_empty_on_device(device):
    """torch.empty(shape, device=device) creates tensor with correct shape/dtype."""
    result = torch.empty(3, 4, device=device)

    assert result.shape == (3, 4)
    assert result.dtype == torch.float32
    assert result.device.type == "sky"


@pytest.mark.it
@pytest.mark.asyncio
async def test_randn_on_device(device):
    """torch.randn(shape, device=device) creates tensor with correct shape, finite values."""
    result = torch.randn(5, 5, device=device)

    assert result.shape == (5, 5)
    assert result.dtype == torch.float32
    # Values should be finite
    result_cpu = result.cpu()
    assert torch.isfinite(result_cpu).all()


@pytest.mark.it
@pytest.mark.asyncio
async def test_rand_on_device(device):
    """torch.rand(shape, device=device) creates values in [0, 1)."""
    result = torch.rand(100, device=device)

    result_cpu = result.cpu()
    assert result_cpu.min() >= 0.0
    assert result_cpu.max() < 1.0


@pytest.mark.it
@pytest.mark.asyncio
async def test_tensor_from_list(device):
    """torch.tensor([values], device=device) creates tensor directly on sky."""
    x = torch.tensor([1.0, 2.0, 3.0], device=device)
    assert x.device.type == "sky"
    assert x.shape == (3,)
    torch.testing.assert_close(
        x.cpu(), torch.tensor([1.0, 2.0, 3.0]), check_device=False
    )


@pytest.mark.xfail(reason="Cannot mix cpu tensors with sky tensors")
@pytest.mark.it
@pytest.mark.asyncio
async def test_arange_on_device(device):
    """torch.arange(n, device=device) creates correct integer sequence."""
    result = torch.arange(10, device=device)

    assert result.shape == (10,)
    assert result.device.type == "sky"

    result_cpu = result.cpu()
    for i in range(10):
        assert result_cpu[i].item() == i


# =============================================================================
# _like variants
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_zeros_like(device):
    """torch.zeros_like(sky_tensor) matches shape/dtype, all zeros."""
    x = torch.randn(3, 4, dtype=torch.float64, device=device)
    result = torch.zeros_like(x)

    assert result.shape == (3, 4)
    assert result.dtype == torch.float64
    assert result.device.type == "sky"
    torch.testing.assert_close(
        result.cpu(), torch.zeros(3, 4, dtype=torch.float64), check_device=False
    )


@pytest.mark.it
@pytest.mark.asyncio
async def test_ones_like(device):
    """torch.ones_like(sky_tensor) matches shape/dtype, all ones."""
    x = torch.randn(3, 4, dtype=torch.float64, device=device)
    result = torch.ones_like(x)

    assert result.shape == (3, 4)
    assert result.dtype == torch.float64
    assert result.device.type == "sky"
    torch.testing.assert_close(
        result.cpu(), torch.ones(3, 4, dtype=torch.float64), check_device=False
    )


@pytest.mark.it
@pytest.mark.asyncio
async def test_full_like(device):
    """torch.full_like(sky_tensor, fill_value) matches shape/dtype."""
    x = torch.randn(3, 4, device=device)
    result = torch.full_like(x, 3.14)

    assert result.shape == (3, 4)
    assert result.device.type == "sky"
    torch.testing.assert_close(
        result.cpu(), torch.full((3, 4), 3.14), check_device=False
    )


@pytest.mark.it
@pytest.mark.asyncio
async def test_empty_like(device):
    """torch.empty_like(sky_tensor) matches shape/dtype."""
    x = torch.randn(3, 4, dtype=torch.float64, device=device)
    result = torch.empty_like(x)

    assert result.shape == (3, 4)
    assert result.dtype == torch.float64
    assert result.device.type == "sky"


# =============================================================================
# Factory with explicit dtype
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_zeros_with_explicit_dtype(device):
    """Factory with explicit dtype: torch.zeros(..., dtype=torch.float64)."""
    result = torch.zeros(3, 3, dtype=torch.float64, device=device)

    assert result.dtype == torch.float64
    torch.testing.assert_close(
        result.cpu(), torch.zeros(3, 3, dtype=torch.float64), check_device=False
    )


@pytest.mark.it
@pytest.mark.asyncio
async def test_ones_with_explicit_dtype(device):
    """Factory with explicit dtype: torch.ones(..., dtype=torch.int64)."""
    result = torch.ones(3, 3, dtype=torch.int64, device=device)

    assert result.dtype == torch.int64
    torch.testing.assert_close(
        result.cpu(), torch.ones(3, 3, dtype=torch.int64), check_device=False
    )


# =============================================================================
# Factory with requires_grad
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_factory_with_requires_grad(device):
    """Factory with requires_grad=True."""
    result = torch.zeros(3, 3, device=device, requires_grad=True)

    assert result.requires_grad
    assert result.device.type == "sky"


@pytest.mark.it
@pytest.mark.asyncio
async def test_randn_with_requires_grad(device):
    """torch.randn with requires_grad=True, verify backward works."""
    x = torch.randn(3, 3, device=device, requires_grad=True)

    loss = (x ** 2).sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == (3, 3)
