"""Basic arithmetic operation correctness tests.

Tests add, sub, mul, div with Tensor, Scalar, out, and inplace variants.
"""

import pytest
import torch


# =============================================================================
# Forward correctness tests (Tensor variants)
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "op_name,fn",
    [
        ("add", lambda x, y: torch.add(x, y)),
        ("sub", lambda x, y: torch.sub(x, y)),
        ("mul", lambda x, y: torch.mul(x, y)),
        ("div", lambda x, y: torch.div(x, y)),
    ],
)
async def test_arithmetic_tensor_forward(device, op_name, fn):
    x_cpu = torch.randn(4, 4)
    y_cpu = torch.randn(4, 4)
    if op_name == "div":
        y_cpu = y_cpu.abs() + 0.1  # avoid division by zero

    cpu_result = fn(x_cpu, y_cpu)
    sky_result = fn(x_cpu.to(device), y_cpu.to(device))
    torch.testing.assert_close(sky_result.cpu(), cpu_result, check_device=False)


# =============================================================================
# Forward correctness tests (Scalar variants)
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "op_name,fn",
    [
        ("add_scalar", lambda x: x + 2.5),
        ("sub_scalar", lambda x: x - 1.5),
        ("mul_scalar", lambda x: x * 3.0),
        ("div_scalar", lambda x: x / 2.0),
        ("radd_scalar", lambda x: 2.5 + x),
        ("rsub_scalar", lambda x: 1.5 - x),
        ("rmul_scalar", lambda x: 3.0 * x),
    ],
)
async def test_arithmetic_scalar_forward(device, op_name, fn):
    x_cpu = torch.randn(4, 4)
    cpu_result = fn(x_cpu)
    sky_result = fn(x_cpu.to(device))
    torch.testing.assert_close(sky_result.cpu(), cpu_result, check_device=False)


# =============================================================================
# Div rounding modes
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rounding_mode",
    [None, "trunc", "floor"],
)
async def test_div_rounding_mode_forward(device, rounding_mode):
    x_cpu = torch.randn(4, 4)
    y_cpu = torch.randn(4, 4).abs() + 0.1

    cpu_result = torch.div(x_cpu, y_cpu, rounding_mode=rounding_mode)
    sky_result = torch.div(
        x_cpu.to(device), y_cpu.to(device), rounding_mode=rounding_mode
    )
    torch.testing.assert_close(sky_result.cpu(), cpu_result, check_device=False)


# =============================================================================
# Gradient correctness tests (Tensor variants)
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "op_name,fn",
    [
        ("add", lambda x, y: (x + y).sum()),
        ("sub", lambda x, y: (x - y).sum()),
        ("mul", lambda x, y: (x * y).sum()),
        ("div", lambda x, y: (x / y).sum()),
    ],
)
async def test_arithmetic_tensor_grad(device, op_name, fn):
    x_cpu = torch.randn(4, 4, requires_grad=True)
    y_cpu = torch.randn(4, 4, requires_grad=True)
    if op_name == "div":
        y_cpu = (torch.randn(4, 4).abs() + 0.1).requires_grad_(True)

    x_sky = x_cpu.clone().to(device).detach().requires_grad_(True)
    y_sky = y_cpu.clone().to(device).detach().requires_grad_(True)

    loss_cpu = fn(x_cpu, y_cpu)
    loss_cpu.backward()

    loss_sky = fn(x_sky, y_sky)
    loss_sky.backward()

    assert x_sky.grad is not None, f"{op_name} x gradient is None"
    assert y_sky.grad is not None, f"{op_name} y gradient is None"
    torch.testing.assert_close(x_sky.grad.cpu(), x_cpu.grad, check_device=False)
    torch.testing.assert_close(y_sky.grad.cpu(), y_cpu.grad, check_device=False)


# =============================================================================
# Gradient correctness tests (Scalar variants)
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "op_name,fn",
    [
        ("add_scalar", lambda x: (x + 2.5).sum()),
        ("sub_scalar", lambda x: (x - 1.5).sum()),
        ("mul_scalar", lambda x: (x * 3.0).sum()),
        ("div_scalar", lambda x: (x / 2.0).sum()),
    ],
)
async def test_arithmetic_scalar_grad(device, op_name, fn):
    x_cpu = torch.randn(4, 4, requires_grad=True)
    x_sky = x_cpu.clone().to(device).detach().requires_grad_(True)

    loss_cpu = fn(x_cpu)
    loss_cpu.backward()

    loss_sky = fn(x_sky)
    loss_sky.backward()

    assert x_sky.grad is not None, f"{op_name} gradient is None"
    torch.testing.assert_close(x_sky.grad.cpu(), x_cpu.grad, check_device=False)


# =============================================================================
# Inplace operations
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "op_name,fn",
    [
        ("add_", lambda x, y: x.add_(y)),
        ("sub_", lambda x, y: x.sub_(y)),
        ("mul_", lambda x, y: x.mul_(y)),
        ("div_", lambda x, y: x.div_(y)),
    ],
)
async def test_arithmetic_inplace_tensor(device, op_name, fn):
    x_cpu = torch.randn(4, 4)
    y_cpu = torch.randn(4, 4)
    if "div" in op_name:
        y_cpu = y_cpu.abs() + 0.1

    x_cpu_clone = x_cpu.clone()
    fn(x_cpu_clone, y_cpu)

    x_sky = x_cpu.clone().to(device)
    fn(x_sky, y_cpu.to(device))

    torch.testing.assert_close(x_sky.cpu(), x_cpu_clone, check_device=False)


@pytest.mark.it
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "op_name,fn",
    [
        ("add_scalar", lambda x: x.add_(2.5)),
        ("sub_scalar", lambda x: x.sub_(1.5)),
        ("mul_scalar", lambda x: x.mul_(3.0)),
        ("div_scalar", lambda x: x.div_(2.0)),
    ],
)
async def test_arithmetic_inplace_scalar(device, op_name, fn):
    x_cpu = torch.randn(4, 4)
    x_cpu_clone = x_cpu.clone()
    fn(x_cpu_clone)

    x_sky = x_cpu.clone().to(device)
    fn(x_sky)

    torch.testing.assert_close(x_sky.cpu(), x_cpu_clone, check_device=False)


# =============================================================================
# Out variant operations
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "op_name,fn",
    [
        ("add.out", lambda x, y, o: torch.add(x, y, out=o)),
        ("sub.out", lambda x, y, o: torch.sub(x, y, out=o)),
        ("mul.out", lambda x, y, o: torch.mul(x, y, out=o)),
        ("div.out", lambda x, y, o: torch.div(x, y, out=o)),
    ],
)
async def test_arithmetic_out_variant(device, op_name, fn):
    x_cpu = torch.randn(4, 4)
    y_cpu = torch.randn(4, 4)
    if "div" in op_name:
        y_cpu = y_cpu.abs() + 0.1

    out_cpu = torch.empty(4, 4)
    fn(x_cpu, y_cpu, out_cpu)

    out_sky = torch.empty(4, 4, device=device)
    fn(x_cpu.to(device), y_cpu.to(device), out_sky)

    torch.testing.assert_close(out_sky.cpu(), out_cpu, check_device=False)


# =============================================================================
# Broadcasting
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "op_name,fn",
    [
        ("add", lambda x, y: x + y),
        ("mul", lambda x, y: x * y),
    ],
)
async def test_arithmetic_broadcast(device, op_name, fn):
    x_cpu = torch.randn(4, 3)
    y_cpu = torch.randn(1, 3)  # broadcast along dim 0

    cpu_result = fn(x_cpu, y_cpu)
    sky_result = fn(x_cpu.to(device), y_cpu.to(device))

    torch.testing.assert_close(sky_result.cpu(), cpu_result, check_device=False)
