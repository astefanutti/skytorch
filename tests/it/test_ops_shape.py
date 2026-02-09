"""Shape operation correctness tests.

These operations are most likely to have the CompositeImplicitAutograd bug
where an explicit PrivateUse1 registration overrides PyTorch's built-in
autograd decomposition, breaking gradient tracking.
"""

import pytest
import torch

from .op_test_utils import assert_forward_correct, assert_grad_correct


# =============================================================================
# Forward correctness tests
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "op_name,fn,shape",
    [
        ("flatten", lambda x: torch.flatten(x, 1), (2, 3, 4)),
        ("flatten_all", lambda x: torch.flatten(x), (2, 3, 4)),
        ("squeeze_dim", lambda x: x.squeeze(1), (2, 1, 3)),
        ("squeeze_dims", lambda x: x.squeeze((1, 3)), (2, 1, 3, 1)),
        ("unsqueeze_0", lambda x: x.unsqueeze(0), (3, 4)),
        ("unsqueeze_1", lambda x: x.unsqueeze(1), (3, 4)),
        ("unsqueeze_-1", lambda x: x.unsqueeze(-1), (3, 4)),
        ("permute", lambda x: x.permute(2, 0, 1), (2, 3, 4)),
        ("transpose", lambda x: x.transpose(0, 1), (3, 4)),
        ("t", lambda x: x.t(), (3, 4)),
        ("contiguous", lambda x: x.transpose(0, 1).contiguous(), (3, 4)),
        ("expand", lambda x: x.expand(3, 4), (1, 4)),
        ("expand_3d", lambda x: x.expand(2, 3, 4), (1, 3, 4)),
        ("view", lambda x: x.view(6, 4), (2, 3, 4)),
        ("reshape", lambda x: x.reshape(6, 4), (2, 3, 4)),
    ],
    ids=lambda x: x if isinstance(x, str) else "",
)
async def test_shape_op_forward(device, op_name, fn, shape):
    x_cpu = torch.randn(shape)
    assert_forward_correct(fn, [x_cpu], device)


@pytest.mark.it
@pytest.mark.asyncio
async def test_select_forward(device):
    x_cpu = torch.randn(3, 4, 5)
    cpu_result = x_cpu.select(1, 2)
    sky_result = x_cpu.to(device).select(1, 2)
    torch.testing.assert_close(sky_result.cpu(), cpu_result, check_device=False)


@pytest.mark.it
@pytest.mark.asyncio
async def test_slice_forward(device):
    x_cpu = torch.randn(4, 6)
    cpu_result = x_cpu[1:3, 2:5]
    sky_result = x_cpu.to(device)[1:3, 2:5]
    torch.testing.assert_close(sky_result.cpu(), cpu_result, check_device=False)


@pytest.mark.it
@pytest.mark.asyncio
async def test_index_tensor_forward(device):
    x_cpu = torch.randn(5, 4)
    idx = torch.tensor([0, 2, 4])
    cpu_result = x_cpu[idx]
    sky_result = x_cpu.to(device)[idx.to(device)]
    torch.testing.assert_close(sky_result.cpu(), cpu_result, check_device=False)


# =============================================================================
# Gradient correctness tests
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "op_name,fn,shape",
    [
        ("flatten", lambda x: torch.flatten(x, 1).sum(), (2, 3, 4)),
        ("flatten_all", lambda x: torch.flatten(x).sum(), (2, 3, 4)),
        ("squeeze_dim", lambda x: x.squeeze(1).sum(), (2, 1, 3)),
        ("squeeze_dims", lambda x: x.squeeze((1, 3)).sum(), (2, 1, 3, 1)),
        ("unsqueeze", lambda x: x.unsqueeze(0).sum(), (3, 4)),
        ("permute", lambda x: x.permute(2, 0, 1).sum(), (2, 3, 4)),
        ("transpose", lambda x: x.transpose(0, 1).sum(), (3, 4)),
        ("t", lambda x: x.t().sum(), (3, 4)),
        ("expand", lambda x: x.expand(3, 4).sum(), (1, 4)),
        ("view", lambda x: x.view(6, 4).sum(), (2, 3, 4)),
        ("reshape", lambda x: x.reshape(6, 4).sum(), (2, 3, 4)),
    ],
    ids=lambda x: x if isinstance(x, str) else "",
)
async def test_shape_op_grad(device, op_name, fn, shape):
    x_cpu = torch.randn(shape, requires_grad=True)
    assert_grad_correct(fn, [x_cpu], device)


@pytest.mark.it
@pytest.mark.asyncio
async def test_select_grad(device):
    x_cpu = torch.randn(3, 4, 5, requires_grad=True)
    x_sky = x_cpu.clone().to(device).detach().requires_grad_(True)

    loss_cpu = x_cpu.select(1, 2).sum()
    loss_cpu.backward()

    loss_sky = x_sky.select(1, 2).sum()
    loss_sky.backward()

    assert x_sky.grad is not None, (
        "select gradient is None — likely CompositeImplicitAutograd issue"
    )
    torch.testing.assert_close(x_sky.grad.cpu(), x_cpu.grad, check_device=False)


@pytest.mark.it
@pytest.mark.asyncio
async def test_slice_grad(device):
    x_cpu = torch.randn(4, 6, requires_grad=True)
    x_sky = x_cpu.clone().to(device).detach().requires_grad_(True)

    loss_cpu = x_cpu[1:3, 2:5].sum()
    loss_cpu.backward()

    loss_sky = x_sky[1:3, 2:5].sum()
    loss_sky.backward()

    assert x_sky.grad is not None, (
        "slice gradient is None — likely CompositeImplicitAutograd issue"
    )
    torch.testing.assert_close(x_sky.grad.cpu(), x_cpu.grad, check_device=False)


@pytest.mark.it
@pytest.mark.asyncio
async def test_index_tensor_grad(device):
    x_cpu = torch.randn(5, 4, requires_grad=True)
    x_sky = x_cpu.clone().to(device).detach().requires_grad_(True)
    idx = torch.tensor([0, 2, 4])

    loss_cpu = x_cpu[idx].sum()
    loss_cpu.backward()

    loss_sky = x_sky[idx.to(device)].sum()
    loss_sky.backward()

    assert x_sky.grad is not None, (
        "index gradient is None — likely CompositeImplicitAutograd issue"
    )
    torch.testing.assert_close(x_sky.grad.cpu(), x_cpu.grad, check_device=False)


# =============================================================================
# Chained gradient tests (MNIST pattern)
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_flatten_mm_sum_grad(device):
    """Reproduce MNIST pattern: conv output -> flatten -> linear.

    This chains flatten with mm and sum, the exact pattern that broke
    in MNIST training when flatten had an explicit PrivateUse1 registration.
    """
    # Simulate conv output: (batch=2, channels=3, h=4, w=4)
    x_cpu = torch.randn(2, 3, 4, 4, requires_grad=True)
    w_cpu = torch.randn(48, 10, requires_grad=True)

    x_sky = x_cpu.clone().to(device).detach().requires_grad_(True)
    w_sky = w_cpu.clone().to(device).detach().requires_grad_(True)

    # CPU: flatten -> mm -> sum
    flat_cpu = torch.flatten(x_cpu, 1)  # (2, 48)
    out_cpu = flat_cpu.mm(w_cpu)  # (2, 10)
    loss_cpu = out_cpu.sum()
    loss_cpu.backward()

    # Sky: flatten -> mm -> sum
    flat_sky = torch.flatten(x_sky, 1)
    out_sky = flat_sky.mm(w_sky)
    loss_sky = out_sky.sum()
    loss_sky.backward()

    assert x_sky.grad is not None, (
        "Input gradient is None after flatten->mm->sum chain. "
        "This is the MNIST pattern — flatten likely has an explicit "
        "PrivateUse1 registration breaking CompositeImplicitAutograd."
    )
    assert w_sky.grad is not None, "Weight gradient is None"
    torch.testing.assert_close(
        x_sky.grad.cpu(), x_cpu.grad, atol=1e-4, rtol=1e-4, check_device=False
    )
    torch.testing.assert_close(
        w_sky.grad.cpu(), w_cpu.grad, atol=1e-4, rtol=1e-4, check_device=False
    )


@pytest.mark.it
@pytest.mark.asyncio
async def test_reshape_chain_grad(device):
    """Test gradient flow through chained reshape operations."""
    x_cpu = torch.randn(2, 3, 4, requires_grad=True)
    x_sky = x_cpu.clone().to(device).detach().requires_grad_(True)

    # CPU
    y_cpu = x_cpu.reshape(6, 4).reshape(2, 12).reshape(24)
    loss_cpu = y_cpu.sum()
    loss_cpu.backward()

    # Sky
    y_sky = x_sky.reshape(6, 4).reshape(2, 12).reshape(24)
    loss_sky = y_sky.sum()
    loss_sky.backward()

    assert x_sky.grad is not None, "Chained reshape gradient is None"
    torch.testing.assert_close(x_sky.grad.cpu(), x_cpu.grad, check_device=False)
