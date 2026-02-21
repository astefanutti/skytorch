"""Matrix multiplication operation correctness tests.

Tests mm, bmm, addmm, _grouped_mm (forward + gradient + out/inplace variants).
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


# =============================================================================
# _grouped_mm (CEA override — requires PrivateUse1 registration to bypass
# CompositeExplicitAutograd decomposition which contains CUDA-specific checks)
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_grouped_mm_3d_3d(device):
    """_grouped_mm with 3D mat1 and 3D mat2 (batched experts).

    Uses bfloat16 and dimensions that are multiples of 8 to satisfy
    _grouped_mm's stride alignment requirements (16-byte aligned).
    """
    num_experts = 4
    m, k, n = 8, 16, 8

    mat1_cpu = torch.randn(num_experts, m, k, dtype=torch.bfloat16)
    mat2_cpu = torch.randn(num_experts, k, n, dtype=torch.bfloat16)

    cpu_result = torch._grouped_mm(mat1_cpu, mat2_cpu)
    sky_result = torch._grouped_mm(mat1_cpu.to(device), mat2_cpu.to(device))

    torch.testing.assert_close(
        sky_result.cpu(), cpu_result, atol=1e-2, rtol=1e-2, check_device=False
    )


@pytest.mark.it
@pytest.mark.asyncio
async def test_grouped_mm_2d_3d_with_offsets(device):
    """_grouped_mm with 2D mat1, 3D mat2, and offsets tensor.

    Uses bfloat16 and dimensions that are multiples of 8 to satisfy
    _grouped_mm's stride alignment requirements (16-byte aligned).
    """
    num_experts = 3
    k, n = 16, 8
    tokens_per_expert = [8, 16, 8]
    total_tokens = sum(tokens_per_expert)

    mat1_cpu = torch.randn(total_tokens, k, dtype=torch.bfloat16)
    mat2_cpu = torch.randn(num_experts, k, n, dtype=torch.bfloat16)
    offs_cpu = torch.tensor(tokens_per_expert, dtype=torch.int32).cumsum(dim=0, dtype=torch.int32)

    cpu_result = torch._grouped_mm(mat1_cpu, mat2_cpu, offs=offs_cpu)
    sky_result = torch._grouped_mm(
        mat1_cpu.to(device), mat2_cpu.to(device), offs=offs_cpu.to(device)
    )

    torch.testing.assert_close(
        sky_result.cpu(), cpu_result, atol=1e-2, rtol=1e-2, check_device=False
    )
