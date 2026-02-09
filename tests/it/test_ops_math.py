"""Element-wise math and initialization operation correctness tests.

Tests clone, neg, abs, sqrt, rsqrt, exp, log, pow (forward + gradient + inplace),
and initialization ops (zeros, ones, full, fill_, zero_).
"""

import pytest
import torch


# =============================================================================
# Forward correctness tests
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "op_name,fn,input_fn",
    [
        ("clone", lambda x: x.clone(), lambda: torch.randn(4, 4)),
        ("neg", lambda x: torch.neg(x), lambda: torch.randn(4, 4)),
        ("abs", lambda x: torch.abs(x), lambda: torch.randn(4, 4)),
        ("sqrt", lambda x: torch.sqrt(x), lambda: torch.rand(4, 4) + 0.1),
        ("rsqrt", lambda x: torch.rsqrt(x), lambda: torch.rand(4, 4) + 0.1),
        ("exp", lambda x: torch.exp(x), lambda: torch.randn(4, 4) * 0.5),
        ("log", lambda x: torch.log(x), lambda: torch.rand(4, 4) + 0.1),
        ("pow2", lambda x: torch.pow(x, 2), lambda: torch.randn(4, 4)),
        ("pow0.5", lambda x: torch.pow(x, 0.5), lambda: torch.rand(4, 4) + 0.1),
        ("reciprocal", lambda x: torch.reciprocal(x), lambda: torch.randn(4, 4).abs() + 0.1),
        ("sign", lambda x: torch.sign(x), lambda: torch.randn(4, 4)),
        ("floor", lambda x: torch.floor(x), lambda: torch.randn(4, 4) * 10),
        ("ceil", lambda x: torch.ceil(x), lambda: torch.randn(4, 4) * 10),
        ("round", lambda x: torch.round(x), lambda: torch.randn(4, 4) * 10),
        ("clamp", lambda x: torch.clamp(x, -0.5, 0.5), lambda: torch.randn(4, 4)),
    ],
    ids=lambda x: x if isinstance(x, str) else "",
)
async def test_math_forward(device, op_name, fn, input_fn):
    x_cpu = input_fn()
    cpu_result = fn(x_cpu)
    sky_result = fn(x_cpu.to(device))
    torch.testing.assert_close(sky_result.cpu(), cpu_result, check_device=False)


# =============================================================================
# Gradient correctness tests
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "op_name,fn,input_fn",
    [
        ("neg", lambda x: torch.neg(x).sum(), lambda: torch.randn(4, 4)),
        ("abs", lambda x: torch.abs(x).sum(), lambda: torch.randn(4, 4) + 1.0),
        ("sqrt", lambda x: torch.sqrt(x).sum(), lambda: torch.rand(4, 4) + 0.1),
        ("rsqrt", lambda x: torch.rsqrt(x).sum(), lambda: torch.rand(4, 4) + 0.1),
        ("exp", lambda x: torch.exp(x).sum(), lambda: torch.randn(4, 4) * 0.5),
        ("log", lambda x: torch.log(x).sum(), lambda: torch.rand(4, 4) + 0.1),
        ("pow2", lambda x: torch.pow(x, 2).sum(), lambda: torch.randn(4, 4)),
        (
            "reciprocal",
            lambda x: torch.reciprocal(x).sum(),
            lambda: torch.randn(4, 4).abs() + 0.5,
        ),
        ("clamp", lambda x: torch.clamp(x, -0.5, 0.5).sum(), lambda: torch.randn(4, 4)),
    ],
    ids=lambda x: x if isinstance(x, str) else "",
)
async def test_math_grad(device, op_name, fn, input_fn):
    x_cpu = input_fn().requires_grad_(True)
    x_sky = x_cpu.clone().to(device).detach().requires_grad_(True)

    loss_cpu = fn(x_cpu)
    loss_cpu.backward()

    loss_sky = fn(x_sky)
    loss_sky.backward()

    assert x_sky.grad is not None, (
        f"{op_name} gradient is None — likely CompositeImplicitAutograd issue"
    )
    torch.testing.assert_close(
        x_sky.grad.cpu(), x_cpu.grad, check_device=False
    )


# =============================================================================
# Inplace operations
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "op_name,fn,input_fn",
    [
        ("neg_", lambda x: x.neg_(), lambda: torch.randn(4, 4)),
        ("abs_", lambda x: x.abs_(), lambda: torch.randn(4, 4)),
        ("sqrt_", lambda x: x.sqrt_(), lambda: torch.rand(4, 4) + 0.1),
        ("exp_", lambda x: x.exp_(), lambda: torch.randn(4, 4) * 0.5),
        ("log_", lambda x: x.log_(), lambda: torch.rand(4, 4) + 0.1),
        ("clamp_", lambda x: x.clamp_(-0.5, 0.5), lambda: torch.randn(4, 4)),
        ("floor_", lambda x: x.floor_(), lambda: torch.randn(4, 4) * 10),
        ("ceil_", lambda x: x.ceil_(), lambda: torch.randn(4, 4) * 10),
    ],
    ids=lambda x: x if isinstance(x, str) else "",
)
async def test_math_inplace(device, op_name, fn, input_fn):
    x_cpu = input_fn()
    x_cpu_clone = x_cpu.clone()
    fn(x_cpu_clone)

    x_sky = x_cpu.clone().to(device)
    fn(x_sky)

    torch.testing.assert_close(x_sky.cpu(), x_cpu_clone, check_device=False)


# =============================================================================
# Initialization operations
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_zeros(device):
    result = torch.zeros(3, 4, device=device)
    expected = torch.zeros(3, 4)
    torch.testing.assert_close(result.cpu(), expected, check_device=False)


@pytest.mark.it
@pytest.mark.asyncio
async def test_ones(device):
    result = torch.ones(3, 4, device=device)
    expected = torch.ones(3, 4)
    torch.testing.assert_close(result.cpu(), expected, check_device=False)


@pytest.mark.it
@pytest.mark.asyncio
async def test_full(device):
    result = torch.full((3, 4), 7.5, device=device)
    expected = torch.full((3, 4), 7.5)
    torch.testing.assert_close(result.cpu(), expected, check_device=False)


@pytest.mark.it
@pytest.mark.asyncio
async def test_zeros_like(device):
    x = torch.randn(3, 4, device=device)
    result = torch.zeros_like(x)
    expected = torch.zeros(3, 4)
    torch.testing.assert_close(result.cpu(), expected, check_device=False)


@pytest.mark.it
@pytest.mark.asyncio
async def test_ones_like(device):
    x = torch.randn(3, 4, device=device)
    result = torch.ones_like(x)
    expected = torch.ones(3, 4)
    torch.testing.assert_close(result.cpu(), expected, check_device=False)


@pytest.mark.it
@pytest.mark.asyncio
async def test_fill_(device):
    x = torch.empty(3, 4, device=device)
    x.fill_(3.14)
    expected = torch.full((3, 4), 3.14)
    torch.testing.assert_close(x.cpu(), expected, check_device=False)


@pytest.mark.it
@pytest.mark.asyncio
async def test_zero_(device):
    x = torch.randn(3, 4).to(device)
    x.zero_()
    expected = torch.zeros(3, 4)
    torch.testing.assert_close(x.cpu(), expected, check_device=False)


@pytest.mark.it
@pytest.mark.asyncio
async def test_empty(device):
    """empty should create a tensor of the right shape and device."""
    result = torch.empty(3, 4, device=device)
    assert result.shape == (3, 4)


@pytest.mark.xfail(reason="Cannot mix cpu tensors with sky tensors")
@pytest.mark.it
@pytest.mark.asyncio
async def test_arange(device):
    result = torch.arange(10, device=device)
    expected = torch.arange(10)
    torch.testing.assert_close(result.cpu(), expected, check_device=False)


# =============================================================================
# Pow with tensor exponent
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_pow_tensor_forward(device):
    base = torch.rand(4, 4) + 0.1
    exp = torch.rand(4, 4) + 0.5

    cpu_result = torch.pow(base, exp)
    sky_result = torch.pow(base.to(device), exp.to(device))

    torch.testing.assert_close(
        sky_result.cpu(), cpu_result, atol=1e-4, rtol=1e-4, check_device=False
    )


@pytest.mark.it
@pytest.mark.asyncio
async def test_pow_tensor_grad(device):
    base = (torch.rand(4, 4) + 0.1).requires_grad_(True)
    exp = (torch.rand(4, 4) + 0.5).requires_grad_(True)

    base_sky = base.clone().to(device).detach().requires_grad_(True)
    exp_sky = exp.clone().to(device).detach().requires_grad_(True)

    loss_cpu = torch.pow(base, exp).sum()
    loss_cpu.backward()

    loss_sky = torch.pow(base_sky, exp_sky).sum()
    loss_sky.backward()

    assert base_sky.grad is not None, "pow base gradient is None"
    assert exp_sky.grad is not None, "pow exponent gradient is None"
    torch.testing.assert_close(
        base_sky.grad.cpu(), base.grad, atol=1e-4, rtol=1e-4, check_device=False
    )
    torch.testing.assert_close(
        exp_sky.grad.cpu(), exp.grad, atol=1e-4, rtol=1e-4, check_device=False
    )
