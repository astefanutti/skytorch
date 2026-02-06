import warnings

import pytest
import torch


# =============================================================================
# Basic Gradient Computation Tests
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_simple_backward_pass(device):
    """Test simple backward pass on KPU tensors."""
    x = torch.randn(2, 2, requires_grad=True)
    x_kpu = x.to(device).detach().requires_grad_()

    y_kpu = x_kpu.sum()
    y_kpu.backward()

    assert x_kpu.grad is not None
    expected_grad = torch.ones_like(x)
    assert torch.allclose(x_kpu.grad.cpu(), expected_grad)


@pytest.mark.it
@pytest.mark.asyncio
async def test_scalar_multiplication_grad(device):
    """Test gradients for scalar multiplication."""
    x = torch.randn(3, 3, requires_grad=True)
    x_kpu = x.to(device).detach().requires_grad_()
    scalar = 2.5

    y_kpu = x_kpu * scalar
    loss_kpu = y_kpu.sum()
    loss_kpu.backward()

    assert x_kpu.grad is not None
    expected_grad = torch.full_like(x, scalar)
    assert torch.allclose(x_kpu.grad.cpu(), expected_grad)


@pytest.mark.it
@pytest.mark.asyncio
async def test_addition_gradients(device):
    """Test gradients for tensor addition."""
    x = torch.randn(2, 2, requires_grad=True)
    y = torch.randn(2, 2, requires_grad=True)

    x_kpu = x.to(device).detach().requires_grad_()
    y_kpu = y.to(device).detach().requires_grad_()

    z_kpu = x_kpu + y_kpu
    loss_kpu = z_kpu.sum()
    loss_kpu.backward()

    assert x_kpu.grad is not None
    assert y_kpu.grad is not None

    expected_grad = torch.ones_like(x)
    assert torch.allclose(x_kpu.grad.cpu(), expected_grad)
    assert torch.allclose(y_kpu.grad.cpu(), expected_grad)


@pytest.mark.it
@pytest.mark.asyncio
async def test_element_wise_multiplication_grad(device):
    """Test gradients for element-wise multiplication."""
    x = torch.randn(2, 2, requires_grad=True)
    y = torch.randn(2, 2, requires_grad=True)

    x_kpu = x.to(device).detach().requires_grad_()
    y_kpu = y.to(device).detach().requires_grad_()

    z_kpu = x_kpu * y_kpu
    loss_kpu = z_kpu.sum()
    loss_kpu.backward()

    assert x_kpu.grad is not None
    assert y_kpu.grad is not None

    # x's gradient should be y, y's gradient should be x
    assert torch.allclose(x_kpu.grad.cpu(), y)
    assert torch.allclose(y_kpu.grad.cpu(), x)


# =============================================================================
# Matrix Operation Gradient Tests
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_matrix_multiplication_gradients(device):
    """Test gradients for matrix multiplication."""
    x = torch.randn(2, 3, requires_grad=True)
    y = torch.randn(3, 2, requires_grad=True)

    x_kpu = x.to(device).detach().requires_grad_()
    y_kpu = y.to(device).detach().requires_grad_()

    z_kpu = x_kpu.mm(y_kpu)
    loss_kpu = z_kpu.sum()
    loss_kpu.backward()

    assert x_kpu.grad is not None
    assert y_kpu.grad is not None
    assert x_kpu.grad.shape == x.shape
    assert y_kpu.grad.shape == y.shape


@pytest.mark.it
@pytest.mark.asyncio
async def test_batch_matrix_multiplication_gradients(device):
    """Test gradients for batch matrix multiplication."""
    x = torch.randn(2, 3, 4, requires_grad=True)
    y = torch.randn(2, 4, 5, requires_grad=True)

    x_kpu = x.to(device).detach().requires_grad_()
    y_kpu = y.to(device).detach().requires_grad_()

    try:
        z_kpu = torch.bmm(x_kpu, y_kpu)
        loss_kpu = z_kpu.sum()
        loss_kpu.backward()

        assert x_kpu.grad is not None
        assert y_kpu.grad is not None
        assert x_kpu.grad.shape == x.shape
        assert y_kpu.grad.shape == y.shape
    except (RuntimeError, NotImplementedError):
        pytest.skip("Batch matrix multiplication gradients not supported")


# =============================================================================
# View Operation Gradient Tests
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_view_operation_gradients(device):
    """Test that gradients flow through view operations."""
    x = torch.randn(2, 6, requires_grad=True)
    x_kpu = x.to(device).detach().requires_grad_()

    y_kpu = x_kpu.view(3, 4)
    loss_kpu = y_kpu.sum()
    loss_kpu.backward()

    assert x_kpu.grad is not None
    assert x_kpu.grad.shape == x.shape
    expected_grad = torch.ones_like(x)
    assert torch.allclose(x_kpu.grad.cpu(), expected_grad)


@pytest.mark.it
@pytest.mark.asyncio
async def test_transpose_operation_gradients(device):
    """Test that gradients flow through transpose operations."""
    x = torch.randn(3, 4, requires_grad=True)
    x_kpu = x.to(device).detach().requires_grad_()

    y_kpu = x_kpu.transpose(0, 1)
    loss_kpu = y_kpu.sum()
    loss_kpu.backward()

    assert x_kpu.grad is not None
    assert x_kpu.grad.shape == x.shape
    expected_grad = torch.ones_like(x)
    assert torch.allclose(x_kpu.grad.cpu(), expected_grad)


# @pytest.mark.it
# @pytest.mark.asyncio
# async def test_reshape_operation_gradients(device):
#     """Test that gradients flow through reshape operations."""
#     x = torch.randn(2, 3, 4, requires_grad=True)
#     x_kpu = x.to(device).detach().requires_grad_()
#
#     y_kpu = x_kpu.reshape(6, 4)
#     loss_kpu = y_kpu.sum()
#     loss_kpu.backward()
#
#     assert x_kpu.grad is not None
#     assert x_kpu.grad.shape == x.shape


# =============================================================================
# Retain Grad Tests
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_retain_grad_basic(device):
    """Test basic retain_grad() functionality with KPU tensors."""
    x = torch.randn(3, 3, requires_grad=True)
    y = torch.randn(3, 3, requires_grad=True)

    x_kpu = x.to(device).detach().requires_grad_()
    y_kpu = y.to(device).detach().requires_grad_()

    # Intermediate computation - normally gradients wouldn't be retained
    z_kpu = x_kpu * y_kpu
    z_kpu.retain_grad()

    # Final computation
    loss_kpu = z_kpu.sum()
    loss_kpu.backward()

    # Check that gradients are available
    assert x_kpu.grad is not None
    assert y_kpu.grad is not None
    assert z_kpu.grad is not None, "z should have gradients (retained)"

    # Verify gradient values are correct
    expected_z_grad = torch.ones_like(z_kpu.cpu())
    expected_x_grad = y  # d/dx(x*y) = y
    expected_y_grad = x  # d/dy(x*y) = x

    assert torch.allclose(z_kpu.grad.cpu(), expected_z_grad)
    assert torch.allclose(x_kpu.grad.cpu(), expected_x_grad)
    assert torch.allclose(y_kpu.grad.cpu(), expected_y_grad)


@pytest.mark.it
@pytest.mark.asyncio
async def test_retain_grad_multiple_intermediates(device):
    """Test retain_grad() with multiple intermediate tensors."""
    x = torch.randn(2, 2, requires_grad=True)
    y = torch.randn(2, 2, requires_grad=True)

    x_kpu = x.to(device).detach().requires_grad_()
    y_kpu = y.to(device).detach().requires_grad_()

    # Multiple intermediate computations
    z1_kpu = x_kpu + y_kpu
    z1_kpu.retain_grad()

    z2_kpu = z1_kpu * 2
    z2_kpu.retain_grad()

    # Final loss
    loss_kpu = z2_kpu.mean()
    loss_kpu.backward()

    # All should have gradients
    assert x_kpu.grad is not None
    assert y_kpu.grad is not None
    assert z1_kpu.grad is not None, "z1 should have gradients (retained)"
    assert z2_kpu.grad is not None, "z2 should have gradients (retained)"

    # Verify gradient shapes
    assert z1_kpu.grad.shape == z1_kpu.shape
    assert z2_kpu.grad.shape == z2_kpu.shape


@pytest.mark.it
@pytest.mark.asyncio
async def test_retain_grad_comparison_with_cpu(device):
    """Test that retain_grad() on KPU tensors matches CPU behavior."""
    x = torch.randn(2, 3, requires_grad=True)
    y = torch.randn(2, 3, requires_grad=True)

    # CPU computation
    z_cpu = x * y
    z_cpu.retain_grad()
    loss_cpu = z_cpu.sum()
    loss_cpu.backward()

    # KPU computation
    x_kpu = x.clone().to(device).detach().requires_grad_()
    y_kpu = y.clone().to(device).detach().requires_grad_()
    z_kpu = x_kpu * y_kpu
    z_kpu.retain_grad()
    loss_kpu = z_kpu.sum()
    loss_kpu.backward()

    # Compare gradients
    assert torch.allclose(x_kpu.grad.cpu(), x.grad)
    assert torch.allclose(y_kpu.grad.cpu(), y.grad)
    assert torch.allclose(z_kpu.grad.cpu(), z_cpu.grad)


@pytest.mark.it
@pytest.mark.asyncio
async def test_without_retain_grad_no_intermediate_gradients(device):
    """Test that without retain_grad(), intermediate tensors don't have gradients."""
    x = torch.randn(2, 2, requires_grad=True)
    y = torch.randn(2, 2, requires_grad=True)

    x_kpu = x.to(device).detach().requires_grad_()
    y_kpu = y.to(device).detach().requires_grad_()

    # Intermediate computation WITHOUT retain_grad()
    z_kpu = x_kpu * y_kpu
    # z_kpu.retain_grad()  # <-- NOT called

    loss_kpu = z_kpu.sum()
    loss_kpu.backward()

    # Check gradients
    assert x_kpu.grad is not None
    assert y_kpu.grad is not None

    # Accessing .grad on non-leaf tensor without retain_grad() should be None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        assert z_kpu.grad is None, "z should NOT have gradients (not retained)"


# =============================================================================
# Gradient Accumulation Tests
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_gradient_accumulation_basic(device):
    """Test basic gradient accumulation."""
    x = torch.randn(2, 2, requires_grad=True)
    x_kpu = x.to(device).detach().requires_grad_()

    # First backward pass
    y1_kpu = x_kpu.sum()
    y1_kpu.backward(retain_graph=True)
    first_grad = x_kpu.grad.clone()

    # Second backward pass (should accumulate)
    y2_kpu = (x_kpu * 2).sum()
    y2_kpu.backward()

    # Verify gradient accumulation
    expected_grad = first_grad + 2 * torch.ones_like(first_grad)
    t = x_kpu.grad.cpu()
    u = expected_grad.cpu()
    assert torch.allclose(t, u)


@pytest.mark.it
@pytest.mark.asyncio
async def test_gradient_zero_behavior(device):
    """Test gradient zeroing behavior."""
    x = torch.randn(2, 2, requires_grad=True)
    x_kpu = x.to(device).detach().requires_grad_()

    # First backward pass
    y_kpu = x_kpu.sum()
    y_kpu.backward()

    # Verify gradients exist
    assert x_kpu.grad is not None

    # Zero gradients
    x_kpu.grad.zero_()

    # Verify gradients are zero
    expected_zero = torch.zeros_like(x)
    assert torch.allclose(x_kpu.grad.cpu(), expected_zero)


# =============================================================================
# Autograd Function Tests
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_requires_grad_propagation(device):
    """Test that requires_grad propagates correctly."""
    x = torch.randn(2, 2, requires_grad=True)
    y = torch.randn(2, 2, requires_grad=False)

    x_kpu = x.to(device).detach().requires_grad_()
    y_kpu = y.to(device).detach()

    # Operations with requires_grad=True tensor
    z_kpu = x_kpu + y_kpu
    assert z_kpu.requires_grad

    # Operations with only requires_grad=False tensors
    w_kpu = y_kpu * 2
    assert not w_kpu.requires_grad


@pytest.mark.it
@pytest.mark.asyncio
async def test_detach_behavior(device):
    """Test tensor detach behavior."""
    x = torch.randn(2, 2, requires_grad=True)
    x_kpu = x.to(device).detach().requires_grad_()

    # Detach tensor
    x_detached = x_kpu.detach()
    assert not x_detached.requires_grad

    # Operations on detached tensor shouldn't require grad
    y_kpu = x_detached * 2
    assert not y_kpu.requires_grad


@pytest.mark.it
@pytest.mark.asyncio
async def test_no_grad_context(device):
    """Test torch.no_grad() context behavior."""
    x = torch.randn(2, 2, requires_grad=True)
    x_kpu = x.to(device).detach().requires_grad_()

    with torch.no_grad():
        y_kpu = x_kpu * 2
        assert not y_kpu.requires_grad


# =============================================================================
# Numerical Verification Tests
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_gradient_numerical_simple(device):
    """Test gradient computation against CPU reference."""
    x = torch.randn(2, 2, requires_grad=True)
    x_kpu = x.clone().to(device).detach().requires_grad_()

    # CPU computation
    y_cpu = (x**2).sum()
    y_cpu.backward()
    cpu_grad = x.grad.clone()

    # KPU computation
    y_kpu = (x_kpu**2).sum()
    y_kpu.backward()

    # Compare gradients
    assert torch.allclose(x_kpu.grad.cpu(), cpu_grad)


@pytest.mark.it
@pytest.mark.asyncio
async def test_chain_rule_verification(device):
    """Test chain rule implementation."""
    x = torch.randn(2, 2, requires_grad=True)
    x_kpu = x.clone().to(device).detach().requires_grad_()

    # CPU computation: y = (x^2 + 1) * 3
    y_cpu = ((x**2) + 1) * 3
    loss_cpu = y_cpu.sum()
    loss_cpu.backward()
    cpu_grad = x.grad.clone()

    # KPU computation
    y_kpu = ((x_kpu**2) + 1) * 3
    loss_kpu = y_kpu.sum()
    loss_kpu.backward()

    # Compare gradients
    assert torch.allclose(x_kpu.grad.cpu(), cpu_grad)


# =============================================================================
# Parametrized Tests
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
@pytest.mark.parametrize("shape", [(2, 2), (3, 3), (2, 3)])
async def test_parametrized_gradient_shapes(device, shape):
    """Test gradient computation with various tensor shapes."""
    x = torch.randn(shape, requires_grad=True)
    x_kpu = x.to(device).detach().requires_grad_()

    y_kpu = x_kpu.sum()
    y_kpu.backward()

    assert x_kpu.grad is not None
    assert x_kpu.grad.shape == shape
    expected_grad = torch.ones(shape)
    assert torch.allclose(x_kpu.grad.cpu(), expected_grad)


@pytest.mark.it
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "op_name,operation",
    [
        ("sum", lambda x: x.sum()),
        ("x**2", lambda x: (x**2).sum()),
        ("x*2", lambda x: (x * 2).sum()),
        ("mean", lambda x: x.mean()),
    ],
)
async def test_parametrized_operations_gradients(device, op_name, operation):
    """Test gradients for various operations."""
    x = torch.randn(3, 3, requires_grad=True)
    x_kpu = x.clone().to(device).detach().requires_grad_()

    try:
        # CPU computation
        y_cpu = operation(x)
        y_cpu.backward()
        cpu_grad = x.grad.clone()

        # KPU computation
        y_kpu = operation(x_kpu)
        y_kpu.backward()

        # Compare gradients
        assert torch.allclose(x_kpu.grad.cpu(), cpu_grad)
    except (RuntimeError, NotImplementedError):
        pytest.skip(f"Operation {op_name} gradients not supported")
