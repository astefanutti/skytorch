"""Comparison and logical operation correctness tests.

Tests eq, ne, gt, lt, ge, le with tensor and scalar variants, bool result dtype,
out variants, broadcasting, and chained accuracy-computation patterns.
"""

import pytest
import torch


# =============================================================================
# Comparison forward tests (tensor variants) — moved from test_ops_reduction.py
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "op_name,fn",
    [
        ("eq", lambda x, y: torch.eq(x, y)),
        ("ne", lambda x, y: torch.ne(x, y)),
        ("gt", lambda x, y: torch.gt(x, y)),
        ("lt", lambda x, y: torch.lt(x, y)),
        ("ge", lambda x, y: torch.ge(x, y)),
        ("le", lambda x, y: torch.le(x, y)),
    ],
)
async def test_comparison_tensor_forward(device, op_name, fn):
    x_cpu = torch.randn(4, 4)
    y_cpu = torch.randn(4, 4)

    cpu_result = fn(x_cpu, y_cpu)
    sky_result = fn(x_cpu.to(device), y_cpu.to(device))

    torch.testing.assert_close(sky_result.cpu(), cpu_result, check_device=False)


# =============================================================================
# Comparison forward tests (scalar variants) — moved from test_ops_reduction.py
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "op_name,fn",
    [
        ("eq_scalar", lambda x: x.eq(0.0)),
        ("ne_scalar", lambda x: x.ne(0.0)),
        ("gt_scalar", lambda x: x.gt(0.0)),
        ("lt_scalar", lambda x: x.lt(0.0)),
        ("ge_scalar", lambda x: x.ge(0.0)),
        ("le_scalar", lambda x: x.le(0.0)),
    ],
)
async def test_comparison_scalar_forward(device, op_name, fn):
    x_cpu = torch.randn(4, 4)

    cpu_result = fn(x_cpu)
    sky_result = fn(x_cpu.to(device))

    torch.testing.assert_close(sky_result.cpu(), cpu_result, check_device=False)


# =============================================================================
# Bool result dtype verification
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_comparison_returns_bool_dtype(device):
    """Comparison ops should return bool tensors."""
    x = torch.randn(3, 3).to(device)
    y = torch.randn(3, 3).to(device)

    result = torch.eq(x, y)
    assert result.dtype == torch.bool

    result = x.gt(0.0)
    assert result.dtype == torch.bool


# =============================================================================
# Out variants
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "op_name,fn",
    [
        ("eq.Tensor_out", lambda x, y, o: torch.eq(x, y, out=o)),
        ("ne.Tensor_out", lambda x, y, o: torch.ne(x, y, out=o)),
        ("gt.Tensor_out", lambda x, y, o: torch.gt(x, y, out=o)),
        ("lt.Tensor_out", lambda x, y, o: torch.lt(x, y, out=o)),
    ],
)
async def test_comparison_out_variant(device, op_name, fn):
    x_cpu = torch.randn(4, 4)
    y_cpu = torch.randn(4, 4)

    out_cpu = torch.empty(4, 4, dtype=torch.bool)
    fn(x_cpu, y_cpu, out_cpu)

    out_sky = torch.empty(4, 4, dtype=torch.bool, device=device)
    fn(x_cpu.to(device), y_cpu.to(device), out_sky)

    torch.testing.assert_close(out_sky.cpu(), out_cpu, check_device=False)


@pytest.mark.it
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "op_name,fn",
    [
        ("eq.Scalar_out", lambda x, o: torch.eq(x, 0.0, out=o)),
        ("ne.Scalar_out", lambda x, o: torch.ne(x, 0.0, out=o)),
    ],
)
async def test_comparison_scalar_out_variant(device, op_name, fn):
    x_cpu = torch.randn(4, 4)

    out_cpu = torch.empty(4, 4, dtype=torch.bool)
    fn(x_cpu, out_cpu)

    out_sky = torch.empty(4, 4, dtype=torch.bool, device=device)
    fn(x_cpu.to(device), out_sky)

    torch.testing.assert_close(out_sky.cpu(), out_cpu, check_device=False)


# =============================================================================
# Comparison with broadcasting
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_comparison_broadcast(device):
    """Comparison with broadcasting (row vector vs matrix)."""
    x_cpu = torch.randn(4, 3)
    y_cpu = torch.randn(1, 3)  # broadcast along dim 0

    cpu_result = torch.gt(x_cpu, y_cpu)
    sky_result = torch.gt(x_cpu.to(device), y_cpu.to(device))

    torch.testing.assert_close(sky_result.cpu(), cpu_result, check_device=False)


# =============================================================================
# Chained accuracy pattern
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_accuracy_pattern(device):
    """Common accuracy pattern: (predictions == labels).float().sum()."""
    torch.manual_seed(42)

    # Simulate classification output
    logits_cpu = torch.randn(10, 5)
    labels_cpu = torch.randint(0, 5, (10,))

    predictions_cpu = logits_cpu.argmax(dim=1)
    accuracy_cpu = (predictions_cpu == labels_cpu).float().sum()

    logits_sky = logits_cpu.to(device)
    labels_sky = labels_cpu.to(device)

    predictions_sky = logits_sky.argmax(dim=1)
    accuracy_sky = (predictions_sky == labels_sky).float().sum()

    torch.testing.assert_close(
        accuracy_sky.cpu(), accuracy_cpu, check_device=False
    )


@pytest.mark.it
@pytest.mark.asyncio
async def test_accuracy_mean_pattern(device):
    """Accuracy as mean: (pred == target).float().mean()."""
    torch.manual_seed(42)

    logits_cpu = torch.randn(20, 3)
    labels_cpu = torch.randint(0, 3, (20,))

    acc_cpu = (logits_cpu.argmax(1) == labels_cpu).float().mean()
    acc_sky = (
        logits_cpu.to(device).argmax(1) == labels_cpu.to(device)
    ).float().mean()

    torch.testing.assert_close(acc_sky.cpu(), acc_cpu, check_device=False)
