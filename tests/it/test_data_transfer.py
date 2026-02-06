import pytest
import torch


@pytest.mark.it
@pytest.mark.asyncio
async def test_cpu_to_st(device):
    """Test transferring tensor from cpu to sky."""
    x = torch.randn(10, 10)
    x_st = x.to(device)

    assert x_st.device.type == "sky"
    assert x_st.shape == x.shape


@pytest.mark.it
@pytest.mark.asyncio
async def test_st_to_cpu(device):
    """Test transferring tensor from sky to cpu."""
    x = torch.randn(10, 10, device=device)
    result = x.cpu()

    assert result.device.type == "cpu"
    assert result.shape == x.shape


@pytest.mark.it
@pytest.mark.asyncio
async def test_data_integrity_roundtrip(device):
    """Test data integrity after cpu -> sky -> cpu roundtrip."""
    original = torch.randn(5, 5)

    # Transfer to sky device and back
    on_st = original.to(device)
    result = on_st.cpu()

    assert torch.allclose(original, result)


@pytest.mark.it
@pytest.mark.asyncio
async def test_various_dtypes(device):
    """Test transfer with various tensor dtypes."""
    dtypes = [torch.float32, torch.float64, torch.int32, torch.int64]

    for dtype in dtypes:
        x = torch.tensor([1, 2, 3], dtype=dtype)
        x_st = x.to(device)
        x_back = x_st.cpu()
        assert torch.equal(x, x_back), f"Failed for dtype {dtype}"


@pytest.mark.it
@pytest.mark.asyncio
async def test_large_tensor(device):
    """Test transfer of larger tensor."""
    x = torch.randn(100, 100)
    x_st = x.to(device)
    x_back = x_st.cpu()

    assert torch.allclose(x, x_back)


@pytest.mark.it
@pytest.mark.asyncio
async def test_scalar_tensor_roundtrip(device):
    """Test 0-dimensional (scalar) tensor cpu -> sky -> cpu roundtrip."""
    scalar = torch.tensor(42.0)

    # Transfer to sky device and back
    on_st = scalar.to(device)
    result = on_st.cpu()

    assert scalar.dim() == 0
    assert on_st.dim() == 0
    assert result.dim() == 0
    assert torch.allclose(scalar, result)


@pytest.mark.it
@pytest.mark.asyncio
async def test_scalar_tensor_item(device):
    """Test .item() on scalar tensor - requires copying to cpu."""
    scalar = torch.tensor(3.14159)
    on_st = scalar.to(device)

    # .item() internally copies the tensor to cpu
    value = on_st.item()

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
        on_st = scalar.to(device)
        result = on_st.cpu()

        assert scalar.dim() == 0
        assert torch.equal(scalar, result), f"Failed for dtype {dtype}"


@pytest.mark.it
@pytest.mark.asyncio
async def test_item_on_intermediate_operation_result(device):
    """Test .item() on scalar resulting from chained sky operations.

    This reproduces the bug from the MNIST example where:
    accuracy = (correct.float() / total).item() * 100

    The intermediate scalar tensor from the division is created on the server
    but may not be properly registered for later retrieval.
    """
    # Create tensors on cpu and move to device to ensure data is uploaded
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
    """Test nn.Parameter transfer to sky device.

    This verifies the GIL-free allocator works correctly with nn.Parameter,
    which is critical for PyTorch 2.10 compatibility (kHasPyObject flag).
    """
    import torch.nn as nn

    # Create parameter on cpu and transfer to sky device
    param = nn.Parameter(torch.randn(3, 3))
    param_st = param.to(device)

    assert param_st.device.type == "sky"
    assert param_st.requires_grad

    # Verify data integrity
    result = param_st.cpu()
    assert torch.allclose(param.data, result)


@pytest.mark.it
@pytest.mark.asyncio
async def test_st_to_st_copy(device):
    """Test server-side copy between sky tensors."""
    src = torch.randn(5, 5).to(device)
    dst = torch.empty(5, 5, device=device)

    dst.copy_(src)

    assert torch.allclose(src.cpu(), dst.cpu())


@pytest.mark.it
@pytest.mark.asyncio
async def test_tensor_view_transfer(device):
    """Test transferring tensor views to/from sky device.

    Note: Views are transferred as contiguous data to sky device, so the
    storage_offset is not preserved. The key test is data integrity.
    """
    base = torch.randn(4, 4)
    view = base[1:3, 1:3]  # 2x2 view with storage_offset

    assert view.storage_offset() > 0  # Verify source has offset

    view_st = view.to(device)

    assert view_st.shape == (2, 2)

    back = view_st.cpu()
    assert torch.allclose(view, back)


@pytest.mark.it
@pytest.mark.asyncio
async def test_non_contiguous_tensor_transfer(device):
    """Test transfer of non-contiguous tensors."""
    base = torch.randn(4, 4)
    transposed = base.t()  # Non-contiguous

    assert not transposed.is_contiguous()

    sky_tensor = transposed.to(device)
    back = sky_tensor.cpu()

    assert torch.allclose(transposed, back)


@pytest.mark.it
@pytest.mark.asyncio
async def test_sliced_tensor_roundtrip(device):
    """Test sliced tensor cpu -> sky -> cpu roundtrip."""
    original = torch.randn(10, 10)
    sliced = original[2:7, 3:8]  # 5x5 slice

    on_st = sliced.to(device)
    back = on_st.cpu()

    assert sliced.shape == back.shape
    assert torch.allclose(sliced, back)


@pytest.mark.it
@pytest.mark.asyncio
async def test_half_precision_transfer(device):
    """Test transfer with half precision dtypes."""
    for dtype in [torch.float16, torch.bfloat16]:
        x = torch.randn(3, 3).to(dtype)
        x_st = x.to(device)
        x_back = x_st.cpu()
        assert torch.allclose(x, x_back, rtol=1e-2, atol=1e-2)


@pytest.mark.it
@pytest.mark.asyncio
async def test_boolean_tensor_transfer(device):
    """Test transfer of boolean tensors."""
    x = torch.tensor([[True, False], [False, True]])
    x_st = x.to(device)
    x_back = x_st.cpu()
    assert torch.equal(x, x_back)


@pytest.mark.it
@pytest.mark.asyncio
async def test_empty_tensor_transfer(device):
    """Test transfer of empty tensors (0 elements)."""
    x = torch.empty(0, 3)
    x_st = x.to(device)

    assert x_st.shape == (0, 3)
    assert x_st.numel() == 0

    x_back = x_st.cpu()
    assert x_back.shape == (0, 3)


@pytest.mark.it
@pytest.mark.asyncio
async def test_complex_dtype_transfer(device):
    """Test transfer of complex tensors."""
    x = torch.randn(3, 3, dtype=torch.complex64)
    x_st = x.to(device)
    x_back = x_st.cpu()
    assert torch.allclose(x, x_back)


@pytest.mark.it
@pytest.mark.asyncio
async def test_lazy_storage_registration(device):
    """Test that storage registration is deferred until first use.

    This verifies the GIL-free allocator pattern where storage IDs
    are generated atomically in C++ and only registered in Python
    when the tensor is first used in an operation.
    """
    # Create tensor on sky device with known values
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
    """Test nn.Module transfer to sky device.

    This verifies the GIL-free allocator works correctly with nn.Module,
    which is critical for PyTorch 2.10 compatibility (kHasPyObject flag).
    nn.Linear specifically reproduced the issue.
    """
    import torch.nn as nn

    # Create module on cpu and transfer to sky device
    linear = nn.Linear(4, 2)
    linear_st = linear.to(device)

    # Verify parameters are on sky device
    assert linear_st.weight.device.type == "sky"
    assert linear_st.bias.device.type == "sky"

    # Verify forward pass works
    x = torch.randn(3, 4).to(device)
    output = linear_st(x)

    assert output.device.type == "sky"
    assert output.shape == (3, 2)

    # Verify data integrity by comparing cpu forward pass
    ref_linear = nn.Linear(4, 2)
    ref_linear.weight.data = linear_st.weight.cpu()
    ref_linear.bias.data = linear_st.bias.cpu()
    expected = ref_linear(x.cpu())

    assert torch.allclose(output.cpu(), expected)


@pytest.mark.it
@pytest.mark.asyncio
async def test_nn_sequential_transfer(device):
    """Test nn.Sequential with multiple layers transfers to sky device."""
    import torch.nn as nn

    model = nn.Sequential(
        nn.Linear(8, 4),
        nn.ReLU(),
        nn.Linear(4, 2),
    )
    model_st = model.to(device)

    # Verify all parameters are on sky device
    for param in model_st.parameters():
        assert param.device.type == "sky"

    # Verify forward pass works
    x = torch.randn(2, 8).to(device)
    output = model_st(x)

    assert output.device.type == "sky"
    assert output.shape == (2, 2)

    # Verify numerical correctness by comparing with cpu forward pass
    ref_model = nn.Sequential(
        nn.Linear(8, 4),
        nn.ReLU(),
        nn.Linear(4, 2),
    )
    # Copy weights from SkyTorch model
    ref_model[0].weight.data = model_st[0].weight.cpu()
    ref_model[0].bias.data = model_st[0].bias.cpu()
    ref_model[2].weight.data = model_st[2].weight.cpu()
    ref_model[2].bias.data = model_st[2].bias.cpu()
    expected = ref_model(x.cpu())

    assert torch.allclose(output.cpu(), expected)
