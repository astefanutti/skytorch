"""Activation function correctness tests.

Tests relu, gelu, sigmoid, tanh, softmax, log_softmax (forward + gradient),
and inplace variants.
"""

import pytest
import torch
import torch.nn.functional as F


# =============================================================================
# Forward correctness tests
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name,fn",
    [
        ("relu", lambda x: F.relu(x)),
        ("gelu", lambda x: F.gelu(x)),
        ("sigmoid", lambda x: torch.sigmoid(x)),
        ("tanh", lambda x: torch.tanh(x)),
        ("softmax", lambda x: F.softmax(x, dim=-1)),
        ("log_softmax", lambda x: F.log_softmax(x, dim=-1)),
        ("silu", lambda x: F.silu(x)),
        ("leaky_relu", lambda x: F.leaky_relu(x)),
    ],
)
async def test_activation_forward(device, name, fn):
    x_cpu = torch.randn(4, 6)
    cpu_result = fn(x_cpu)
    sky_result = fn(x_cpu.to(device))
    torch.testing.assert_close(sky_result.cpu(), cpu_result, check_device=False)


# =============================================================================
# Gradient correctness tests
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name,fn",
    [
        ("relu", lambda x: F.relu(x).sum()),
        ("gelu", lambda x: F.gelu(x).sum()),
        ("sigmoid", lambda x: torch.sigmoid(x).sum()),
        ("tanh", lambda x: torch.tanh(x).sum()),
        ("softmax", lambda x: F.softmax(x, dim=-1).sum()),
        ("log_softmax", lambda x: F.log_softmax(x, dim=-1).sum()),
        ("silu", lambda x: F.silu(x).sum()),
        ("leaky_relu", lambda x: F.leaky_relu(x).sum()),
    ],
)
async def test_activation_grad(device, name, fn):
    x_cpu = torch.randn(4, 6, requires_grad=True)
    x_sky = x_cpu.clone().to(device).detach().requires_grad_(True)

    loss_cpu = fn(x_cpu)
    loss_cpu.backward()

    loss_sky = fn(x_sky)
    loss_sky.backward()

    assert x_sky.grad is not None, (
        f"{name} gradient is None — likely CompositeImplicitAutograd issue"
    )
    torch.testing.assert_close(
        x_sky.grad.cpu(), x_cpu.grad, check_device=False
    )


# =============================================================================
# Inplace forward tests
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name,fn",
    [
        ("relu_", lambda x: F.relu(x, inplace=True)),
        ("gelu_", lambda x: torch.ops.aten.gelu_(x)),
        ("sigmoid_", lambda x: x.sigmoid_()),
        ("tanh_", lambda x: x.tanh_()),
    ],
)
async def test_activation_inplace_forward(device, name, fn):
    x_cpu = torch.randn(4, 6)
    x_sky_data = x_cpu.clone()

    cpu_result = fn(x_cpu.clone())
    if cpu_result is None:
        pytest.skip(f"{name} not available")

    sky_input = x_sky_data.to(device)
    sky_result = fn(sky_input)

    torch.testing.assert_close(sky_result.cpu(), cpu_result, check_device=False)
