import pytest
import torch


# =============================================================================
# Basic Operations Tests
# =============================================================================


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


# =============================================================================
# Tensor Creation Tests
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_tensor_creation_various_shapes(device):
    """Test tensor creation with various shapes."""
    test_shapes = [(2, 2), (3, 4), (1, 5, 6), (2, 3, 4)]

    for shape in test_shapes:
        cpu_tensor = torch.randn(*shape)
        kpu_tensor = cpu_tensor.to(device)
        assert kpu_tensor.shape == shape
        assert cpu_tensor.shape == kpu_tensor.shape


@pytest.mark.it
@pytest.mark.asyncio
async def test_tensor_creation_with_gradients(device):
    """Test creating tensors with gradients enabled."""
    cpu_tensor = torch.randn(3, 3, requires_grad=True)
    kpu_tensor = cpu_tensor.to(device)
    assert cpu_tensor.requires_grad
    assert kpu_tensor.requires_grad


@pytest.mark.it
@pytest.mark.asyncio
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
async def test_tensor_creation_different_dtypes(device, dtype):
    """Test tensor creation with different data types."""
    cpu_tensor = torch.randn(2, 2, dtype=dtype)
    kpu_tensor = cpu_tensor.to(device)
    assert cpu_tensor.dtype == dtype
    assert kpu_tensor.dtype == dtype


# =============================================================================
# Arithmetic Operations Tests
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_tensor_addition(device):
    """Test tensor addition on KPU devices."""
    x = torch.randn(2, 2)
    y = torch.randn(2, 2)

    x_kpu = x.to(device)
    y_kpu = y.to(device)

    z_kpu = x_kpu + y_kpu
    z_expected = x + y

    assert torch.allclose(z_kpu.cpu(), z_expected)


@pytest.mark.it
@pytest.mark.asyncio
async def test_tensor_subtraction(device):
    """Test tensor subtraction on KPU devices."""
    x = torch.randn(3, 3)
    y = torch.randn(3, 3)

    x_kpu = x.to(device)
    y_kpu = y.to(device)

    z_kpu = x_kpu - y_kpu
    z_expected = x - y

    assert torch.allclose(z_kpu.cpu(), z_expected)


@pytest.mark.it
@pytest.mark.asyncio
async def test_tensor_multiplication(device):
    """Test element-wise tensor multiplication."""
    x = torch.randn(2, 3)
    y = torch.randn(2, 3)

    x_kpu = x.to(device)
    y_kpu = y.to(device)

    z_kpu = x_kpu * y_kpu
    z_expected = x * y

    assert torch.allclose(z_kpu.cpu(), z_expected)


@pytest.mark.it
@pytest.mark.asyncio
async def test_tensor_division(device):
    """Test tensor division on KPU devices."""
    x = torch.randn(2, 2) + 1.0  # Add 1 to avoid division by zero
    y = torch.randn(2, 2) + 1.0

    x_kpu = x.to(device)
    y_kpu = y.to(device)

    z_kpu = x_kpu / y_kpu
    z_expected = x / y

    assert torch.allclose(z_kpu.cpu(), z_expected)


@pytest.mark.it
@pytest.mark.asyncio
async def test_scalar_operations(device):
    """Test operations with scalars."""
    x = torch.randn(2, 2)
    x_kpu = x.to(device)
    scalar = 2.5

    # Addition with scalar
    result_add = x_kpu + scalar
    expected_add = x + scalar
    assert torch.allclose(result_add.cpu(), expected_add)

    # Multiplication with scalar
    result_mul = x_kpu * scalar
    expected_mul = x * scalar
    assert torch.allclose(result_mul.cpu(), expected_mul)


# =============================================================================
# Matrix Operations Tests
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_matrix_multiplication(device):
    """Test matrix multiplication on KPU devices."""
    x = torch.randn(2, 2)
    y = torch.randn(2, 2)

    x_kpu = x.to(device)
    y_kpu = y.to(device)

    w_kpu = x_kpu.mm(y_kpu)
    w_expected = x.mm(y)

    assert w_kpu.shape == (2, 2)
    assert torch.allclose(w_kpu.cpu(), w_expected)


@pytest.mark.it
@pytest.mark.asyncio
async def test_matrix_multiplication_rectangular(device):
    """Test matrix multiplication with rectangular matrices."""
    x = torch.randn(3, 4)
    y = torch.randn(4, 5)

    x_kpu = x.to(device)
    y_kpu = y.to(device)

    result_kpu = x_kpu.mm(y_kpu)
    result_expected = x.mm(y)

    assert result_kpu.shape == (3, 5)
    assert torch.allclose(result_kpu.cpu(), result_expected)


@pytest.mark.it
@pytest.mark.asyncio
async def test_batch_matrix_multiplication(device):
    """Test batch matrix multiplication on KPU devices."""
    x = torch.randn(2, 3, 4)
    y = torch.randn(2, 4, 5)

    x_kpu = x.to(device)
    y_kpu = y.to(device)

    result_kpu = torch.bmm(x_kpu, y_kpu)
    result_expected = torch.bmm(x, y)

    assert result_kpu.shape == (2, 3, 5)
    assert torch.allclose(result_kpu.cpu(), result_expected)


# =============================================================================
# Conversion Tests
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_cpu_to_kpu_conversion(device):
    """Test converting CPU tensors to KPU devices."""
    cpu_tensor = torch.randn(3, 3)
    kpu_tensor = cpu_tensor.to(device)

    assert kpu_tensor.device.type == "kpu"
    assert cpu_tensor.shape == kpu_tensor.shape
    assert cpu_tensor.dtype == kpu_tensor.dtype


@pytest.mark.it
@pytest.mark.asyncio
async def test_kpu_to_cpu_conversion(device):
    """Test converting KPU tensors back to CPU."""
    original_cpu = torch.randn(2, 2)
    kpu_tensor = original_cpu.to(device)
    back_to_cpu = kpu_tensor.cpu()

    assert back_to_cpu.device.type == "cpu"
    assert torch.allclose(back_to_cpu, original_cpu)


# =============================================================================
# Property Tests
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_tensor_shape_access(device):
    """Test accessing tensor shape on KPU devices."""
    shapes_to_test = [(2, 2), (3, 4), (1, 5, 6)]

    for shape in shapes_to_test:
        kpu_tensor = torch.randn(*shape, device=device)
        assert kpu_tensor.shape == shape


@pytest.mark.it
@pytest.mark.asyncio
async def test_tensor_dtype_access(device):
    """Test accessing tensor dtype on KPU devices."""
    dtypes_to_test = [torch.float32, torch.float64]

    for dtype in dtypes_to_test:
        kpu_tensor = torch.randn(2, 2, dtype=dtype, device=device)
        assert kpu_tensor.dtype == dtype


@pytest.mark.it
@pytest.mark.asyncio
async def test_tensor_device_access(device):
    """Test accessing tensor device information."""
    kpu_tensor = torch.randn(2, 2, device=device)

    assert kpu_tensor.device.type == "kpu"
    assert isinstance(kpu_tensor.device.index, int)
    assert kpu_tensor.device.index >= 0


@pytest.mark.it
@pytest.mark.asyncio
async def test_tensor_requires_grad_access(device):
    """Test accessing requires_grad property."""
    # Tensor without gradients
    tensor_no_grad = torch.randn(2, 2, device=device)
    assert not tensor_no_grad.requires_grad

    # Tensor with gradients
    tensor_with_grad = torch.randn(2, 2, device=device, requires_grad=True)
    assert tensor_with_grad.requires_grad


# =============================================================================
# Comparison Tests
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_tensor_equality(device):
    """Test tensor equality operations."""
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    z = torch.tensor([[1.0, 2.0], [3.0, 5.0]])

    x_kpu = x.to(device)
    y_kpu = y.to(device)
    z_kpu = z.to(device)

    # Test equality
    eq_result = torch.eq(x_kpu, y_kpu)
    eq_expected = torch.eq(x, y)
    assert torch.equal(eq_result.cpu(), eq_expected)

    # Test inequality
    neq_result = torch.eq(x_kpu, z_kpu)
    neq_expected = torch.eq(x, z)
    assert torch.equal(neq_result.cpu(), neq_expected)


# =============================================================================
# Concatenation Tests
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_cat_dim0_basic(device):
    """Test basic concatenation along dimension 0."""
    x = torch.randn(2, 3)
    y = torch.randn(3, 3)

    x_kpu = x.to(device)
    y_kpu = y.to(device)

    result_kpu = torch.cat([x_kpu, y_kpu], dim=0)
    result_expected = torch.cat([x, y], dim=0)

    assert result_kpu.shape == (5, 3)
    assert torch.allclose(result_kpu.cpu(), result_expected)


@pytest.mark.it
@pytest.mark.asyncio
async def test_cat_dim1_basic(device):
    """Test basic concatenation along dimension 1."""
    x = torch.randn(2, 2)
    y = torch.randn(2, 3)

    x_kpu = x.to(device)
    y_kpu = y.to(device)

    result_kpu = torch.cat([x_kpu, y_kpu], dim=1)
    result_expected = torch.cat([x, y], dim=1)

    assert result_kpu.shape == (2, 5)
    assert torch.allclose(result_kpu.cpu(), result_expected)


@pytest.mark.it
@pytest.mark.asyncio
async def test_cat_multiple_tensors(device):
    """Test concatenation with multiple tensors."""
    x = torch.randn(2, 2)
    y = torch.randn(2, 2)
    z = torch.randn(2, 2)

    x_kpu = x.to(device)
    y_kpu = y.to(device)
    z_kpu = z.to(device)

    result_kpu = torch.cat([x_kpu, y_kpu, z_kpu], dim=0)
    result_expected = torch.cat([x, y, z], dim=0)

    assert result_kpu.shape == (6, 2)
    assert torch.allclose(result_kpu.cpu(), result_expected)


@pytest.mark.it
@pytest.mark.asyncio
async def test_cat_single_tensor(device):
    """Test concatenation with a single tensor."""
    x = torch.randn(2, 3)
    x_kpu = x.to(device)

    result_kpu = torch.cat([x_kpu], dim=0)
    result_expected = torch.cat([x], dim=0)

    assert result_kpu.shape == (2, 3)
    assert torch.allclose(result_kpu.cpu(), result_expected)


@pytest.mark.it
@pytest.mark.asyncio
@pytest.mark.parametrize("dim", [0, 1])
async def test_cat_parametrized_dimensions(device, dim):
    """Test concatenation along different dimensions."""
    if dim == 0:
        x = torch.randn(2, 3)
        y = torch.randn(4, 3)
        expected_shape = (6, 3)
    else:  # dim == 1
        x = torch.randn(2, 3)
        y = torch.randn(2, 4)
        expected_shape = (2, 7)

    x_kpu = x.to(device)
    y_kpu = y.to(device)

    result_kpu = torch.cat([x_kpu, y_kpu], dim=dim)
    result_expected = torch.cat([x, y], dim=dim)

    assert result_kpu.shape == expected_shape
    assert torch.allclose(result_kpu.cpu(), result_expected)


# =============================================================================
# Parametrized Operations Test
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation,expected_shape",
    [
        (lambda x, y: x + y, (2, 2)),
        (lambda x, y: x - y, (2, 2)),
        (lambda x, y: x * y, (2, 2)),
        (lambda x, y: x.mm(y), (2, 2)),
    ],
    ids=["add", "sub", "mul", "mm"],
)
async def test_parametrized_operations(device, operation, expected_shape):
    """Test various operations with parametrized inputs."""
    x = torch.randn(2, 2)
    y = torch.randn(2, 2)

    x_kpu = x.to(device)
    y_kpu = y.to(device)

    result_kpu = operation(x_kpu, y_kpu)
    result_expected = operation(x, y)

    assert result_kpu.shape == expected_shape
    assert torch.allclose(result_kpu.cpu(), result_expected)


# =============================================================================
# Multiple Operations Test
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_inplace_add_multiple_iterations(device):
    """Test in-place addition across multiple iterations.

    This tests the pattern used by optimizer.step() which updates
    weights in-place across training iterations.
    """
    # Simulates a weight tensor that persists across iterations
    weight = torch.tensor([1.0, 2.0, 3.0]).to(device)

    for i in range(3):
        # Simulate gradient (new tensor each iteration)
        grad = torch.tensor([0.1, 0.1, 0.1]).to(device)

        # In-place update (like optimizer.step())
        weight.add_(grad)

        # Verify the weight is correctly updated
        expected = torch.tensor([1.0 + 0.1 * (i + 1),
                                  2.0 + 0.1 * (i + 1),
                                  3.0 + 0.1 * (i + 1)])
        assert torch.allclose(weight.cpu(), expected), f"Iteration {i}: mismatch"


@pytest.mark.it
@pytest.mark.asyncio
async def test_multiple_operations_sequence(device):
    """Test a sequence of operations similar to training loop.

    This reproduces the pattern: forward -> loss -> verify results are valid.
    """
    for iteration in range(3):
        # New input data each iteration (like DataLoader)
        # Create on CPU and move to device to ensure data is uploaded
        x = torch.randn(10, 5).to(device)
        w = torch.randn(5, 3).to(device)

        # Forward pass operations
        y = x @ w
        z = y.relu()
        loss = z.sum()

        # Verify loss is not NaN
        loss_value = loss.item()
        assert loss_value == loss_value, \
            f"Iteration {iteration}: loss is NaN"
