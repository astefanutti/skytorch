import pytest
import torch


@pytest.mark.it
@pytest.mark.asyncio
async def test_tensor_creation(device):
    """Test creating a tensor on KPU device."""
    x = torch.tensor([1.0, 2.0, 3.0], device=device)
    assert x.device.type == "kpu"


@pytest.mark.it
@pytest.mark.asyncio
async def test_tensor_add(device):
    """Test tensor addition."""
    x = torch.tensor([1.0, 2.0, 3.0], device=device)
    y = torch.tensor([4.0, 5.0, 6.0], device=device)
    z = x + y
    expected = torch.tensor([5.0, 7.0, 9.0])
    assert torch.allclose(z.cpu(), expected)


@pytest.mark.it
@pytest.mark.asyncio
async def test_tensor_mul(device):
    """Test tensor multiplication."""
    x = torch.tensor([1.0, 2.0, 3.0], device=device)
    y = torch.tensor([2.0, 3.0, 4.0], device=device)
    z = x * y
    expected = torch.tensor([2.0, 6.0, 12.0])
    assert torch.allclose(z.cpu(), expected)


@pytest.mark.it
@pytest.mark.asyncio
async def test_tensor_matmul(device):
    """Test matrix multiplication."""
    x = torch.randn(3, 4, device=device)
    y = torch.randn(4, 5, device=device)
    z = torch.matmul(x, y)

    # Verify shape
    assert z.shape == (3, 5)

    # Verify result matches CPU computation
    x_cpu = x.cpu()
    y_cpu = y.cpu()
    expected = torch.matmul(x_cpu, y_cpu)
    assert torch.allclose(z.cpu(), expected, rtol=1e-4, atol=1e-4)


@pytest.mark.it
@pytest.mark.asyncio
async def test_tensor_sum(device):
    """Test tensor reduction (sum)."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
    result = x.sum()
    assert torch.allclose(result.cpu(), torch.tensor(10.0))


@pytest.mark.it
@pytest.mark.asyncio
async def test_scalar_multiply(device):
    """Test scalar multiplication."""
    x = torch.tensor([1.0, 2.0, 3.0], device=device)
    z = x * 2.0
    expected = torch.tensor([2.0, 4.0, 6.0])
    assert torch.allclose(z.cpu(), expected)
