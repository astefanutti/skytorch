import pytest
import torch


@pytest.mark.it
@pytest.mark.asyncio
async def test_cpu_to_kpu(device):
    """Test transferring tensor from CPU to KPU."""
    x_cpu = torch.randn(10, 10)
    x_kpu = x_cpu.to(device)

    assert x_kpu.device.type == "kpu"
    assert x_kpu.shape == x_cpu.shape


@pytest.mark.it
@pytest.mark.asyncio
async def test_kpu_to_cpu(device):
    """Test transferring tensor from KPU to CPU."""
    x = torch.randn(10, 10, device=device)
    x_cpu = x.cpu()

    assert x_cpu.device.type == "cpu"
    assert x_cpu.shape == x.shape


@pytest.mark.it
@pytest.mark.asyncio
async def test_data_integrity_roundtrip(device):
    """Test data integrity after CPU -> KPU -> CPU roundtrip."""
    original = torch.randn(5, 5)

    # Transfer to KPU and back
    on_kpu = original.to(device)
    back_to_cpu = on_kpu.cpu()

    assert torch.allclose(original, back_to_cpu)


@pytest.mark.it
@pytest.mark.asyncio
async def test_various_dtypes(device):
    """Test transfer with various tensor dtypes."""
    dtypes = [torch.float32, torch.float64, torch.int32, torch.int64]

    for dtype in dtypes:
        x = torch.tensor([1, 2, 3], dtype=dtype)
        x_kpu = x.to(device)
        x_back = x_kpu.cpu()
        assert torch.equal(x, x_back), f"Failed for dtype {dtype}"


@pytest.mark.it
@pytest.mark.asyncio
async def test_large_tensor(device):
    """Test transfer of larger tensor."""
    x = torch.randn(100, 100)
    x_kpu = x.to(device)
    x_back = x_kpu.cpu()

    assert torch.allclose(x, x_back)
