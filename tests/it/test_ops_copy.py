"""Copy and clone semantics tests.

Tests clone(), copy_() between sky tensors and CPU tensors, clone in autograd
graphs, contiguity handling, and storage independence.
"""

import pytest
import torch


# =============================================================================
# Clone tests
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_clone_produces_independent_copy(device):
    """clone() produces an independent tensor with same values."""
    x_cpu = torch.randn(4, 4)
    x_sky = x_cpu.to(device)

    cloned = x_sky.clone()

    torch.testing.assert_close(cloned.cpu(), x_cpu, check_device=False)
    assert cloned.device.type == "sky"


@pytest.mark.it
@pytest.mark.asyncio
async def test_clone_independent_storage(device):
    """Mutating a clone doesn't affect the original."""
    x_cpu = torch.randn(4, 4)
    x_sky = x_cpu.to(device)

    cloned = x_sky.clone()
    cloned.add_(1.0)

    # Original should be unchanged
    torch.testing.assert_close(x_sky.cpu(), x_cpu, check_device=False)
    # Clone should be modified
    torch.testing.assert_close(
        cloned.cpu(), x_cpu + 1.0, check_device=False
    )


@pytest.mark.it
@pytest.mark.asyncio
async def test_clone_with_requires_grad(device):
    """clone() on tensor with requires_grad: gradient flows through clone."""
    x_cpu = torch.randn(3, 3, requires_grad=True)
    x_sky = x_cpu.clone().to(device).detach().requires_grad_(True)

    # CPU
    y_cpu = x_cpu.clone()
    loss_cpu = (y_cpu ** 2).sum()
    loss_cpu.backward()

    # Sky
    y_sky = x_sky.clone()
    loss_sky = (y_sky ** 2).sum()
    loss_sky.backward()

    assert x_sky.grad is not None, "Gradient through clone is None"
    torch.testing.assert_close(x_sky.grad.cpu(), x_cpu.grad, check_device=False)


# =============================================================================
# copy_() tests
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_copy_sky_to_sky(device):
    """copy_() between two sky tensors (server-side copy)."""
    src_cpu = torch.randn(4, 4)
    src_sky = src_cpu.to(device)
    dst_sky = torch.empty(4, 4, device=device)

    dst_sky.copy_(src_sky)

    torch.testing.assert_close(dst_sky.cpu(), src_cpu, check_device=False)


@pytest.mark.it
@pytest.mark.asyncio
async def test_copy_cpu_to_sky(device):
    """copy_() from CPU tensor to sky tensor."""
    src_cpu = torch.randn(4, 4)
    dst_sky = torch.empty(4, 4, device=device)

    dst_sky.copy_(src_cpu)

    torch.testing.assert_close(dst_sky.cpu(), src_cpu, check_device=False)


@pytest.mark.it
@pytest.mark.asyncio
async def test_copy_sky_to_cpu(device):
    """copy_() from sky tensor to CPU tensor."""
    src_cpu = torch.randn(4, 4)
    src_sky = src_cpu.to(device)
    dst_cpu = torch.empty(4, 4)

    dst_cpu.copy_(src_sky)

    torch.testing.assert_close(dst_cpu, src_cpu)


@pytest.mark.it
@pytest.mark.asyncio
async def test_copy_with_broadcast(device):
    """copy_() with broadcasting (scalar to matrix)."""
    src_cpu = torch.tensor(3.14)
    src_sky = src_cpu.to(device)
    dst_sky = torch.empty(3, 3, device=device)

    dst_sky.copy_(src_sky)

    expected = torch.full((3, 3), 3.14)
    torch.testing.assert_close(dst_sky.cpu(), expected, check_device=False)


# =============================================================================
# Contiguity tests
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_contiguous_on_contiguous_noop(device):
    """.contiguous() on already-contiguous tensor is a no-op."""
    x_cpu = torch.randn(3, 4)
    x_sky = x_cpu.to(device)

    assert x_sky.is_contiguous()

    result = x_sky.contiguous()
    torch.testing.assert_close(result.cpu(), x_cpu, check_device=False)


@pytest.mark.it
@pytest.mark.asyncio
async def test_contiguous_on_transposed(device):
    """.contiguous() on a transposed (non-contiguous) sky tensor."""
    x_cpu = torch.randn(3, 4)
    x_sky = x_cpu.to(device)

    transposed_cpu = x_cpu.t()
    transposed_sky = x_sky.t()

    result_cpu = transposed_cpu.contiguous()
    result_sky = transposed_sky.contiguous()

    assert result_sky.is_contiguous()
    torch.testing.assert_close(result_sky.cpu(), result_cpu, check_device=False)


# =============================================================================
# Clone vs copy_ comparison with CPU
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_clone_matches_cpu(device):
    """clone() on sky tensor produces same result as clone() on CPU."""
    x_cpu = torch.randn(5, 5)
    x_sky = x_cpu.to(device)

    cloned_cpu = x_cpu.clone()
    cloned_sky = x_sky.clone()

    torch.testing.assert_close(cloned_sky.cpu(), cloned_cpu, check_device=False)


@pytest.mark.it
@pytest.mark.asyncio
async def test_copy_preserves_dtype(device):
    """copy_() preserves the destination tensor's dtype."""
    src_cpu = torch.randn(3, 3, dtype=torch.float64)
    dst_sky = torch.empty(3, 3, dtype=torch.float32, device=device)

    dst_sky.copy_(src_cpu)

    assert dst_sky.dtype == torch.float32
    expected = src_cpu.float()
    torch.testing.assert_close(dst_sky.cpu(), expected, check_device=False)
