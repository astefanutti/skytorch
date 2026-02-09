"""Matrix multiplication operation correctness tests.

Tests mm, bmm, addmm (forward + gradient + out/inplace variants).
"""

import pytest
import torch


# =============================================================================
# Forward correctness tests
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_mm_forward(device):
    a_cpu = torch.randn(3, 4)
    b_cpu = torch.randn(4, 5)

    cpu_result = torch.mm(a_cpu, b_cpu)
    sky_result = torch.mm(a_cpu.to(device), b_cpu.to(device))

    torch.testing.assert_close(
        sky_result.cpu(), cpu_result, atol=1e-4, rtol=1e-4, check_device=False
    )


@pytest.mark.it
@pytest.mark.asyncio
async def test_bmm_forward(device):
    a_cpu = torch.randn(2, 3, 4)
    b_cpu = torch.randn(2, 4, 5)

    cpu_result = torch.bmm(a_cpu, b_cpu)
    sky_result = torch.bmm(a_cpu.to(device), b_cpu.to(device))

    torch.testing.assert_close(
        sky_result.cpu(), cpu_result, atol=1e-4, rtol=1e-4, check_device=False
    )


@pytest.mark.it
@pytest.mark.asyncio
async def test_addmm_forward(device):
    bias_cpu = torch.randn(5)
    a_cpu = torch.randn(3, 4)
    b_cpu = torch.randn(4, 5)

    cpu_result = torch.addmm(bias_cpu, a_cpu, b_cpu)
    sky_result = torch.addmm(
        bias_cpu.to(device), a_cpu.to(device), b_cpu.to(device)
    )

    torch.testing.assert_close(
        sky_result.cpu(), cpu_result, atol=1e-4, rtol=1e-4, check_device=False
    )


@pytest.mark.it
@pytest.mark.asyncio
async def test_matmul_forward(device):
    """Test torch.matmul (general matrix multiply)."""
    a_cpu = torch.randn(3, 4)
    b_cpu = torch.randn(4, 5)

    cpu_result = torch.matmul(a_cpu, b_cpu)
    sky_result = torch.matmul(a_cpu.to(device), b_cpu.to(device))

    torch.testing.assert_close(
        sky_result.cpu(), cpu_result, atol=1e-4, rtol=1e-4, check_device=False
    )


# =============================================================================
# Gradient correctness tests
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_mm_grad(device):
    a_cpu = torch.randn(3, 4, requires_grad=True)
    b_cpu = torch.randn(4, 5, requires_grad=True)

    a_sky = a_cpu.clone().to(device).detach().requires_grad_(True)
    b_sky = b_cpu.clone().to(device).detach().requires_grad_(True)

    loss_cpu = torch.mm(a_cpu, b_cpu).sum()
    loss_cpu.backward()

    loss_sky = torch.mm(a_sky, b_sky).sum()
    loss_sky.backward()

    assert a_sky.grad is not None, "mm: a gradient is None"
    assert b_sky.grad is not None, "mm: b gradient is None"
    torch.testing.assert_close(
        a_sky.grad.cpu(), a_cpu.grad, atol=1e-4, rtol=1e-4, check_device=False
    )
    torch.testing.assert_close(
        b_sky.grad.cpu(), b_cpu.grad, atol=1e-4, rtol=1e-4, check_device=False
    )


@pytest.mark.it
@pytest.mark.asyncio
async def test_bmm_grad(device):
    a_cpu = torch.randn(2, 3, 4, requires_grad=True)
    b_cpu = torch.randn(2, 4, 5, requires_grad=True)

    a_sky = a_cpu.clone().to(device).detach().requires_grad_(True)
    b_sky = b_cpu.clone().to(device).detach().requires_grad_(True)

    loss_cpu = torch.bmm(a_cpu, b_cpu).sum()
    loss_cpu.backward()

    loss_sky = torch.bmm(a_sky, b_sky).sum()
    loss_sky.backward()

    assert a_sky.grad is not None, "bmm: a gradient is None"
    assert b_sky.grad is not None, "bmm: b gradient is None"
    torch.testing.assert_close(
        a_sky.grad.cpu(), a_cpu.grad, atol=1e-4, rtol=1e-4, check_device=False
    )
    torch.testing.assert_close(
        b_sky.grad.cpu(), b_cpu.grad, atol=1e-4, rtol=1e-4, check_device=False
    )


@pytest.mark.it
@pytest.mark.asyncio
async def test_addmm_grad(device):
    bias_cpu = torch.randn(5, requires_grad=True)
    a_cpu = torch.randn(3, 4, requires_grad=True)
    b_cpu = torch.randn(4, 5, requires_grad=True)

    bias_sky = bias_cpu.clone().to(device).detach().requires_grad_(True)
    a_sky = a_cpu.clone().to(device).detach().requires_grad_(True)
    b_sky = b_cpu.clone().to(device).detach().requires_grad_(True)

    loss_cpu = torch.addmm(bias_cpu, a_cpu, b_cpu).sum()
    loss_cpu.backward()

    loss_sky = torch.addmm(bias_sky, a_sky, b_sky).sum()
    loss_sky.backward()

    for name, sky_t, cpu_t in [
        ("bias", bias_sky, bias_cpu),
        ("a", a_sky, a_cpu),
        ("b", b_sky, b_cpu),
    ]:
        assert sky_t.grad is not None, f"addmm: {name} gradient is None"
        torch.testing.assert_close(
            sky_t.grad.cpu(),
            cpu_t.grad,
            atol=1e-4,
            rtol=1e-4,
            check_device=False,
        )


# =============================================================================
# Out variants
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_mm_out(device):
    a_cpu = torch.randn(3, 4)
    b_cpu = torch.randn(4, 5)

    out_cpu = torch.empty(3, 5)
    torch.mm(a_cpu, b_cpu, out=out_cpu)

    out_sky = torch.empty(3, 5, device=device)
    torch.mm(a_cpu.to(device), b_cpu.to(device), out=out_sky)

    torch.testing.assert_close(
        out_sky.cpu(), out_cpu, atol=1e-4, rtol=1e-4, check_device=False
    )


@pytest.mark.it
@pytest.mark.asyncio
async def test_bmm_out(device):
    a_cpu = torch.randn(2, 3, 4)
    b_cpu = torch.randn(2, 4, 5)

    out_cpu = torch.empty(2, 3, 5)
    torch.bmm(a_cpu, b_cpu, out=out_cpu)

    out_sky = torch.empty(2, 3, 5, device=device)
    torch.bmm(a_cpu.to(device), b_cpu.to(device), out=out_sky)

    torch.testing.assert_close(
        out_sky.cpu(), out_cpu, atol=1e-4, rtol=1e-4, check_device=False
    )


@pytest.mark.it
@pytest.mark.asyncio
async def test_addmm_out(device):
    bias_cpu = torch.randn(5)
    a_cpu = torch.randn(3, 4)
    b_cpu = torch.randn(4, 5)

    out_cpu = torch.empty(3, 5)
    torch.addmm(bias_cpu, a_cpu, b_cpu, out=out_cpu)

    out_sky = torch.empty(3, 5, device=device)
    torch.addmm(
        bias_cpu.to(device), a_cpu.to(device), b_cpu.to(device), out=out_sky
    )

    torch.testing.assert_close(
        out_sky.cpu(), out_cpu, atol=1e-4, rtol=1e-4, check_device=False
    )
