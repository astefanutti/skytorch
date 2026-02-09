"""Type casting and dtype promotion tests.

Tests .to(dtype=...), .float(), .double(), .half(), .bfloat16(), .int(), .long(),
mixed-dtype arithmetic type promotion, and dtype preservation through operations.
"""

import pytest
import torch


# =============================================================================
# Explicit dtype casting
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_to_float64(device):
    """Cast float32 -> float64 via .to(dtype=torch.float64)."""
    x_cpu = torch.randn(3, 4, dtype=torch.float32)
    x_sky = x_cpu.to(device)

    result_cpu = x_cpu.to(dtype=torch.float64)
    result_sky = x_sky.to(dtype=torch.float64)

    assert result_sky.dtype == torch.float64
    torch.testing.assert_close(result_sky.cpu(), result_cpu, check_device=False)


@pytest.mark.it
@pytest.mark.asyncio
async def test_to_float32_via_float(device):
    """Cast float64 -> float32 via .float()."""
    x_cpu = torch.randn(3, 4, dtype=torch.float64)
    x_sky = x_cpu.to(device)

    result_cpu = x_cpu.float()
    result_sky = x_sky.float()

    assert result_sky.dtype == torch.float32
    torch.testing.assert_close(result_sky.cpu(), result_cpu, check_device=False)


@pytest.mark.it
@pytest.mark.asyncio
async def test_to_float64_via_double(device):
    """Cast float32 -> float64 via .double()."""
    x_cpu = torch.randn(3, 4, dtype=torch.float32)
    x_sky = x_cpu.to(device)

    result_cpu = x_cpu.double()
    result_sky = x_sky.double()

    assert result_sky.dtype == torch.float64
    torch.testing.assert_close(result_sky.cpu(), result_cpu, check_device=False)


@pytest.mark.it
@pytest.mark.asyncio
async def test_to_float16_via_half(device):
    """Cast float32 -> float16 via .half()."""
    x_cpu = torch.randn(3, 4, dtype=torch.float32)
    x_sky = x_cpu.to(device)

    result_cpu = x_cpu.half()
    result_sky = x_sky.half()

    assert result_sky.dtype == torch.float16
    torch.testing.assert_close(
        result_sky.cpu(), result_cpu, check_device=False, atol=1e-3, rtol=1e-3
    )


@pytest.mark.it
@pytest.mark.asyncio
async def test_to_bfloat16(device):
    """Cast float32 -> bfloat16 via .bfloat16()."""
    x_cpu = torch.randn(3, 4, dtype=torch.float32)
    x_sky = x_cpu.to(device)

    result_cpu = x_cpu.bfloat16()
    result_sky = x_sky.bfloat16()

    assert result_sky.dtype == torch.bfloat16
    torch.testing.assert_close(
        result_sky.cpu(), result_cpu, check_device=False, atol=1e-2, rtol=1e-2
    )


# =============================================================================
# Int/float conversions
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_float_to_int(device):
    """Cast float -> int32 via .int()."""
    x_cpu = torch.tensor([1.7, 2.3, -0.9, 4.5])
    x_sky = x_cpu.to(device)

    result_cpu = x_cpu.int()
    result_sky = x_sky.int()

    assert result_sky.dtype == torch.int32
    torch.testing.assert_close(result_sky.cpu(), result_cpu, check_device=False)


@pytest.mark.it
@pytest.mark.asyncio
async def test_float_to_long(device):
    """Cast float -> int64 via .long()."""
    x_cpu = torch.tensor([1.7, 2.3, -0.9, 4.5])
    x_sky = x_cpu.to(device)

    result_cpu = x_cpu.long()
    result_sky = x_sky.long()

    assert result_sky.dtype == torch.int64
    torch.testing.assert_close(result_sky.cpu(), result_cpu, check_device=False)


@pytest.mark.it
@pytest.mark.asyncio
async def test_int_to_float(device):
    """Cast int -> float32."""
    x_cpu = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    x_sky = x_cpu.to(device)

    result_cpu = x_cpu.float()
    result_sky = x_sky.float()

    assert result_sky.dtype == torch.float32
    torch.testing.assert_close(result_sky.cpu(), result_cpu, check_device=False)


# =============================================================================
# Mixed-dtype arithmetic (type promotion)
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_mixed_dtype_add_promotes(device):
    """float32 + float64 -> float64 result (type promotion)."""
    a_cpu = torch.randn(3, 3, dtype=torch.float32)
    b_cpu = torch.randn(3, 3, dtype=torch.float64)

    result_cpu = a_cpu + b_cpu
    result_sky = a_cpu.to(device) + b_cpu.to(device)

    assert result_cpu.dtype == torch.float64
    assert result_sky.dtype == torch.float64
    torch.testing.assert_close(result_sky.cpu(), result_cpu, check_device=False)


@pytest.mark.it
@pytest.mark.asyncio
async def test_mixed_dtype_mul_promotes(device):
    """int64 * float32 -> float32 result."""
    a_cpu = torch.tensor([1, 2, 3], dtype=torch.int64)
    b_cpu = torch.tensor([1.5, 2.5, 3.5], dtype=torch.float32)

    result_cpu = a_cpu * b_cpu
    result_sky = a_cpu.to(device) * b_cpu.to(device)

    assert result_cpu.dtype == torch.float32
    assert result_sky.dtype == torch.float32
    torch.testing.assert_close(result_sky.cpu(), result_cpu, check_device=False)


# =============================================================================
# Dtype preservation through operations
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_dtype_preserved_through_mul(device):
    """float64 input -> float64 output after multiplication."""
    x_cpu = torch.randn(3, 3, dtype=torch.float64)
    x_sky = x_cpu.to(device)

    result_cpu = x_cpu * 2.0
    result_sky = x_sky * 2.0

    assert result_sky.dtype == torch.float64
    torch.testing.assert_close(result_sky.cpu(), result_cpu, check_device=False)


@pytest.mark.it
@pytest.mark.asyncio
async def test_dtype_preserved_through_sum(device):
    """float64 input -> float64 output after sum."""
    x_cpu = torch.randn(3, 3, dtype=torch.float64)
    x_sky = x_cpu.to(device)

    result_cpu = x_cpu.sum()
    result_sky = x_sky.sum()

    assert result_sky.dtype == torch.float64
    torch.testing.assert_close(result_sky.cpu(), result_cpu, check_device=False)


# =============================================================================
# .to(dtype=...) on sky tensor (no device transfer)
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_to_dtype_only_no_device_change(device):
    """.to(dtype=...) on a sky tensor should change dtype but stay on sky."""
    x_cpu = torch.randn(3, 3, dtype=torch.float32)
    x_sky = x_cpu.to(device)

    result = x_sky.to(dtype=torch.float64)

    assert result.device.type == "sky"
    assert result.dtype == torch.float64

    expected = x_cpu.to(dtype=torch.float64)
    torch.testing.assert_close(result.cpu(), expected, check_device=False)


# =============================================================================
# Cast and requires_grad interaction
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_float_cast_preserves_requires_grad(device):
    """Casting between float types should preserve requires_grad."""
    x_cpu = torch.randn(3, 3, requires_grad=True)
    x_sky = x_cpu.to(device)

    result_sky = x_sky.double()
    assert result_sky.requires_grad


@pytest.mark.it
@pytest.mark.asyncio
async def test_int_cast_detaches(device):
    """Casting to int should detach (int tensors can't require grad)."""
    x_cpu = torch.randn(3, 3, requires_grad=True)
    x_sky = x_cpu.to(device)

    result_sky = x_sky.int()
    assert not result_sky.requires_grad
