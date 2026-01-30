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


@pytest.mark.it
@pytest.mark.asyncio
async def test_scalar_tensor_roundtrip(device):
    """Test 0-dimensional (scalar) tensor CPU -> KPU -> CPU roundtrip."""
    scalar = torch.tensor(42.0)

    # Transfer to KPU and back
    on_kpu = scalar.to(device)
    back_to_cpu = on_kpu.cpu()

    assert scalar.dim() == 0
    assert on_kpu.dim() == 0
    assert back_to_cpu.dim() == 0
    assert torch.allclose(scalar, back_to_cpu)


@pytest.mark.it
@pytest.mark.asyncio
async def test_scalar_tensor_item(device):
    """Test .item() on scalar tensor - requires copying to CPU."""
    scalar = torch.tensor(3.14159)
    on_kpu = scalar.to(device)

    # .item() internally copies the tensor to CPU
    value = on_kpu.item()

    assert abs(value - 3.14159) < 1e-5


@pytest.mark.it
@pytest.mark.asyncio
async def test_scalar_tensor_various_dtypes(device):
    """Test scalar tensor transfer with various dtypes."""
    test_cases = [
        (torch.float32, 42.5),
        (torch.float64, 42.5),
        (torch.int32, 42),
        (torch.int64, 42),
    ]

    for dtype, value in test_cases:
        scalar = torch.tensor(value, dtype=dtype)
        on_kpu = scalar.to(device)
        back_to_cpu = on_kpu.cpu()

        assert scalar.dim() == 0
        assert torch.equal(scalar, back_to_cpu), f"Failed for dtype {dtype}"


@pytest.mark.it
@pytest.mark.asyncio
async def test_item_on_intermediate_operation_result(device):
    """Test .item() on scalar resulting from chained KPU operations.

    This reproduces the bug from the MNIST example where:
    accuracy = (correct.float() / total).item() * 100

    The intermediate scalar tensor from the division is created on the server
    but may not be properly registered for later retrieval.
    """
    # Create tensors on CPU and move to device to ensure data is uploaded
    # (torch.tensor(..., device=device) creates empty tensors via fallback)
    correct = torch.tensor(85.0).to(device)
    total = torch.tensor(100.0).to(device)

    # Division result
    division_result = correct / total
    result = division_result.item()

    assert abs(result - 0.85) < 1e-5


@pytest.mark.it
@pytest.mark.asyncio
async def test_nn_parameter_transfer(device):
    """Test nn.Parameter transfer to KPU device.

    This verifies the GIL-free allocator works correctly with nn.Parameter,
    which is critical for PyTorch 2.10 compatibility (kHasPyObject flag).
    """
    import torch.nn as nn

    # Create parameter on CPU and transfer to KPU
    param = nn.Parameter(torch.randn(3, 3))
    param_kpu = param.to(device)

    assert param_kpu.device.type == "kpu"
    assert param_kpu.requires_grad

    # Verify data integrity
    back_to_cpu = param_kpu.cpu()
    assert torch.allclose(param.data, back_to_cpu)


@pytest.mark.it
@pytest.mark.asyncio
async def test_kpu_to_kpu_copy(device):
    """Test server-side copy between KPU tensors."""
    src = torch.randn(5, 5).to(device)
    dst = torch.empty(5, 5, device=device)

    dst.copy_(src)

    assert torch.allclose(src.cpu(), dst.cpu())


@pytest.mark.it
@pytest.mark.asyncio
async def test_tensor_view_transfer(device):
    """Test transferring tensor views to/from KPU.

    Note: Views are transferred as contiguous data to KPU, so the
    storage_offset is not preserved. The key test is data integrity.
    """
    base = torch.randn(4, 4)
    view = base[1:3, 1:3]  # 2x2 view with storage_offset

    assert view.storage_offset() > 0  # Verify source has offset

    view_kpu = view.to(device)

    assert view_kpu.shape == (2, 2)

    back = view_kpu.cpu()
    assert torch.allclose(view, back)


@pytest.mark.it
@pytest.mark.asyncio
async def test_non_contiguous_tensor_transfer(device):
    """Test transfer of non-contiguous tensors."""
    base = torch.randn(4, 4)
    transposed = base.t()  # Non-contiguous

    assert not transposed.is_contiguous()

    kpu_tensor = transposed.to(device)
    back = kpu_tensor.cpu()

    assert torch.allclose(transposed, back)


@pytest.mark.it
@pytest.mark.asyncio
async def test_sliced_tensor_roundtrip(device):
    """Test sliced tensor CPU -> KPU -> CPU roundtrip."""
    original = torch.randn(10, 10)
    sliced = original[2:7, 3:8]  # 5x5 slice

    on_kpu = sliced.to(device)
    back = on_kpu.cpu()

    assert sliced.shape == back.shape
    assert torch.allclose(sliced, back)


@pytest.mark.it
@pytest.mark.asyncio
async def test_half_precision_transfer(device):
    """Test transfer with half precision dtypes."""
    for dtype in [torch.float16, torch.bfloat16]:
        x = torch.randn(3, 3).to(dtype)
        x_kpu = x.to(device)
        x_back = x_kpu.cpu()
        assert torch.allclose(x, x_back, rtol=1e-2, atol=1e-2)


@pytest.mark.it
@pytest.mark.asyncio
async def test_boolean_tensor_transfer(device):
    """Test transfer of boolean tensors."""
    x = torch.tensor([[True, False], [False, True]])
    x_kpu = x.to(device)
    x_back = x_kpu.cpu()
    assert torch.equal(x, x_back)


@pytest.mark.it
@pytest.mark.asyncio
async def test_empty_tensor_transfer(device):
    """Test transfer of empty tensors (0 elements)."""
    x = torch.empty(0, 3)
    x_kpu = x.to(device)

    assert x_kpu.shape == (0, 3)
    assert x_kpu.numel() == 0

    x_back = x_kpu.cpu()
    assert x_back.shape == (0, 3)


@pytest.mark.it
@pytest.mark.asyncio
async def test_complex_dtype_transfer(device):
    """Test transfer of complex tensors."""
    x = torch.randn(3, 3, dtype=torch.complex64)
    x_kpu = x.to(device)
    x_back = x_kpu.cpu()
    assert torch.allclose(x, x_back)


@pytest.mark.it
@pytest.mark.asyncio
async def test_lazy_storage_registration(device):
    """Test that storage registration is deferred until first use.

    This verifies the GIL-free allocator pattern where storage IDs
    are generated atomically in C++ and only registered in Python
    when the tensor is first used in an operation.
    """
    # Create tensor on KPU with known values
    x = torch.zeros(3, 3, device=device)

    # First operation triggers lazy registration
    y = x + 1

    # Verify tensor works correctly with numerical check
    result = y.cpu()
    expected = torch.ones(3, 3)
    assert torch.allclose(result, expected)


@pytest.mark.it
@pytest.mark.asyncio
async def test_nn_module_transfer(device):
    """Test nn.Module transfer to KPU device.

    This verifies the GIL-free allocator works correctly with nn.Module,
    which is critical for PyTorch 2.10 compatibility (kHasPyObject flag).
    nn.Linear specifically reproduced the issue.
    """
    import torch.nn as nn

    # Create module on CPU and transfer to KPU
    linear = nn.Linear(4, 2)
    linear_kpu = linear.to(device)

    # Verify parameters are on KPU
    assert linear_kpu.weight.device.type == "kpu"
    assert linear_kpu.bias.device.type == "kpu"

    # Verify forward pass works
    x = torch.randn(3, 4).to(device)
    output = linear_kpu(x)

    assert output.device.type == "kpu"
    assert output.shape == (3, 2)

    # Verify data integrity by comparing CPU forward pass
    x_cpu = x.cpu()
    linear_cpu = nn.Linear(4, 2)
    linear_cpu.weight.data = linear_kpu.weight.cpu()
    linear_cpu.bias.data = linear_kpu.bias.cpu()
    expected = linear_cpu(x_cpu)

    assert torch.allclose(output.cpu(), expected)


@pytest.mark.it
@pytest.mark.asyncio
async def test_nn_sequential_transfer(device):
    """Test nn.Sequential with multiple layers transfers to KPU."""
    import torch.nn as nn

    model = nn.Sequential(
        nn.Linear(8, 4),
        nn.ReLU(),
        nn.Linear(4, 2),
    )
    model_kpu = model.to(device)

    # Verify all parameters are on KPU
    for param in model_kpu.parameters():
        assert param.device.type == "kpu"

    # Verify forward pass works
    x = torch.randn(2, 8).to(device)
    output = model_kpu(x)

    assert output.device.type == "kpu"
    assert output.shape == (2, 2)

    # Verify numerical correctness by comparing with CPU forward pass
    x_cpu = x.cpu()
    model_cpu = nn.Sequential(
        nn.Linear(8, 4),
        nn.ReLU(),
        nn.Linear(4, 2),
    )
    # Copy weights from KPU model
    model_cpu[0].weight.data = model_kpu[0].weight.cpu()
    model_cpu[0].bias.data = model_kpu[0].bias.cpu()
    model_cpu[2].weight.data = model_kpu[2].weight.cpu()
    model_cpu[2].bias.data = model_kpu[2].bias.cpu()
    expected = model_cpu(x_cpu)

    assert torch.allclose(output.cpu(), expected)
