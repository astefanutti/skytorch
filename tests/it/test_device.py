import pytest

from kpu.torch.server import Compute


@pytest.mark.it
@pytest.mark.asyncio
async def test_device_creation(compute: Compute):
    """Test creating a KPU device from Compute."""
    device = compute.device("cpu")
    assert device.type == "kpu"
    assert device.index is not None


@pytest.mark.it
@pytest.mark.asyncio
async def test_device_string_parsing(compute: Compute):
    """Test device string with index parsing."""
    device = compute.device("cpu:0")
    assert device.type == "kpu"


@pytest.mark.it
@pytest.mark.asyncio
async def test_device_explicit_index(compute: Compute):
    """Test device with explicit index argument."""
    device = compute.device("cpu", 0)
    assert device.type == "kpu"


@pytest.mark.it
@pytest.mark.asyncio
async def test_device_invalid_format(compute: Compute):
    """Test error on invalid device format."""
    with pytest.raises(RuntimeError, match="Invalid device string"):
        compute.device("invalid:device:format")


@pytest.mark.it
@pytest.mark.asyncio
async def test_device_conflicting_index(compute: Compute):
    """Test error when index specified both in string and argument."""
    with pytest.raises(RuntimeError, match="must not include an index"):
        compute.device("cpu:0", 1)
