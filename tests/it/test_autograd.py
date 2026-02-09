import warnings

import pytest
import torch
from torch.autograd import gradcheck, gradgradcheck


# =============================================================================
# Tensor Hook Tests
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_register_hook_basic(device):
    """Test that register_hook fires and receives the correct gradient."""
    x_cpu = torch.randn(3, 3, requires_grad=True)
    x_sky = x_cpu.to(device).detach().requires_grad_()

    hook_grads_cpu = []
    hook_grads_sky = []

    x_cpu.register_hook(lambda g: hook_grads_cpu.append(g.clone()))
    x_sky.register_hook(lambda g: hook_grads_sky.append(g.clone()))

    loss_cpu = (x_cpu * 2).sum()
    loss_cpu.backward()

    loss_sky = (x_sky * 2).sum()
    loss_sky.backward()

    assert len(hook_grads_cpu) == 1
    assert len(hook_grads_sky) == 1
    torch.testing.assert_close(hook_grads_sky[0].cpu(), hook_grads_cpu[0], check_device=False)


@pytest.mark.it
@pytest.mark.asyncio
async def test_register_hook_modify_grad(device):
    """Test that a hook can modify the gradient by returning a new value."""
    x_cpu = torch.randn(3, 3, requires_grad=True)
    x_sky = x_cpu.to(device).detach().requires_grad_()

    # Hook that scales gradient by 0.5
    x_cpu.register_hook(lambda g: g * 0.5)
    x_sky.register_hook(lambda g: g * 0.5)

    loss_cpu = (x_cpu * 3).sum()
    loss_cpu.backward()

    loss_sky = (x_sky * 3).sum()
    loss_sky.backward()

    # The stored .grad should reflect the modified gradient
    torch.testing.assert_close(x_sky.grad.cpu(), x_cpu.grad, check_device=False)
    # Verify the hook actually modified it (unmodified grad would be 3.0)
    expected = torch.full_like(x_cpu, 1.5)
    torch.testing.assert_close(x_cpu.grad, expected)


@pytest.mark.it
@pytest.mark.asyncio
async def test_register_hook_multiple(device):
    """Test that multiple hooks on the same tensor all fire."""
    x_cpu = torch.randn(2, 2, requires_grad=True)
    x_sky = x_cpu.to(device).detach().requires_grad_()

    call_order_cpu = []
    call_order_sky = []

    x_cpu.register_hook(lambda g: call_order_cpu.append(1) or g * 2)
    x_cpu.register_hook(lambda g: call_order_cpu.append(2) or g * 3)

    x_sky.register_hook(lambda g: call_order_sky.append(1) or g * 2)
    x_sky.register_hook(lambda g: call_order_sky.append(2) or g * 3)

    loss_cpu = x_cpu.sum()
    loss_cpu.backward()

    loss_sky = x_sky.sum()
    loss_sky.backward()

    assert call_order_cpu == [1, 2]
    assert call_order_sky == [1, 2]
    torch.testing.assert_close(x_sky.grad.cpu(), x_cpu.grad, check_device=False)


# =============================================================================
# Functional Gradient API and Graph Retention Tests
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_autograd_grad_basic(device):
    """Test torch.autograd.grad() returns correct gradients."""
    x_cpu = torch.randn(3, 3, requires_grad=True)
    x_sky = x_cpu.to(device).detach().requires_grad_()

    y_cpu = (x_cpu ** 2).sum()
    y_sky = (x_sky ** 2).sum()

    (grad_cpu,) = torch.autograd.grad(y_cpu, x_cpu)
    (grad_sky,) = torch.autograd.grad(y_sky, x_sky)

    torch.testing.assert_close(grad_sky.cpu(), grad_cpu, check_device=False)
    # .grad should NOT be populated by torch.autograd.grad
    assert x_sky.grad is None


@pytest.mark.it
@pytest.mark.asyncio
async def test_autograd_grad_multiple_inputs(device):
    """Test torch.autograd.grad() with multiple inputs."""
    x_cpu = torch.randn(2, 3, requires_grad=True)
    y_cpu = torch.randn(2, 3, requires_grad=True)

    x_sky = x_cpu.to(device).detach().requires_grad_()
    y_sky = y_cpu.to(device).detach().requires_grad_()

    loss_cpu = (x_cpu * y_cpu + x_cpu ** 2).sum()
    loss_sky = (x_sky * y_sky + x_sky ** 2).sum()

    grads_cpu = torch.autograd.grad(loss_cpu, (x_cpu, y_cpu))
    grads_sky = torch.autograd.grad(loss_sky, (x_sky, y_sky))

    for g_sky, g_cpu in zip(grads_sky, grads_cpu):
        torch.testing.assert_close(g_sky.cpu(), g_cpu, check_device=False)


@pytest.mark.it
@pytest.mark.asyncio
async def test_autograd_grad_retain_graph(device):
    """Test multiple torch.autograd.grad() calls with retain_graph=True."""
    x_cpu = torch.randn(2, 2, requires_grad=True)
    x_sky = x_cpu.to(device).detach().requires_grad_()

    y_cpu = (x_cpu ** 2).sum()
    y_sky = (x_sky ** 2).sum()

    # First call with retain_graph
    (g1_cpu,) = torch.autograd.grad(y_cpu, x_cpu, retain_graph=True)
    (g1_sky,) = torch.autograd.grad(y_sky, x_sky, retain_graph=True)

    # Second call on same graph
    (g2_cpu,) = torch.autograd.grad(y_cpu, x_cpu)
    (g2_sky,) = torch.autograd.grad(y_sky, x_sky)

    torch.testing.assert_close(g1_sky.cpu(), g1_cpu, check_device=False)
    torch.testing.assert_close(g2_sky.cpu(), g2_cpu, check_device=False)
    # Both calls should return the same gradient
    torch.testing.assert_close(g1_sky.cpu(), g2_sky.cpu())


@pytest.mark.it
@pytest.mark.asyncio
async def test_multiple_backward_retain_graph(device):
    """Test multiple backward passes on the same graph with retain_graph."""
    x_cpu = torch.randn(2, 3, requires_grad=True)
    x_sky = x_cpu.to(device).detach().requires_grad_()

    # Shared computation
    hidden_cpu = x_cpu ** 2
    hidden_sky = x_sky ** 2

    # Two different losses from the same graph
    loss1_cpu = hidden_cpu.sum()
    loss2_cpu = (hidden_cpu * 2).sum()

    loss1_sky = hidden_sky.sum()
    loss2_sky = (hidden_sky * 2).sum()

    # First backward with retain
    (g1_cpu,) = torch.autograd.grad(loss1_cpu, x_cpu, retain_graph=True)
    (g1_sky,) = torch.autograd.grad(loss1_sky, x_sky, retain_graph=True)

    # Second backward on same graph
    (g2_cpu,) = torch.autograd.grad(loss2_cpu, x_cpu)
    (g2_sky,) = torch.autograd.grad(loss2_sky, x_sky)

    torch.testing.assert_close(g1_sky.cpu(), g1_cpu, check_device=False)
    torch.testing.assert_close(g2_sky.cpu(), g2_cpu, check_device=False)

    # loss2 has 2x the gradient of loss1
    torch.testing.assert_close(g2_cpu, g1_cpu * 2)


@pytest.mark.it
@pytest.mark.asyncio
async def test_autograd_grad_allow_unused(device):
    """Test torch.autograd.grad() with allow_unused=True for unused inputs."""
    x_cpu = torch.randn(2, 2, requires_grad=True)
    y_cpu = torch.randn(2, 2, requires_grad=True)

    x_sky = x_cpu.to(device).detach().requires_grad_()
    y_sky = y_cpu.to(device).detach().requires_grad_()

    # Loss only depends on x, not y
    loss_cpu = (x_cpu ** 2).sum()
    loss_sky = (x_sky ** 2).sum()

    gx_cpu, gy_cpu = torch.autograd.grad(loss_cpu, (x_cpu, y_cpu), allow_unused=True)
    gx_sky, gy_sky = torch.autograd.grad(loss_sky, (x_sky, y_sky), allow_unused=True)

    torch.testing.assert_close(gx_sky.cpu(), gx_cpu, check_device=False)
    assert gy_cpu is None
    assert gy_sky is None


@pytest.mark.it
@pytest.mark.asyncio
async def test_backward_with_grad_tensors(device):
    """Test backward() with explicit gradient argument for non-scalar loss."""
    x_cpu = torch.randn(3, 3, requires_grad=True)
    x_sky = x_cpu.to(device).detach().requires_grad_()

    y_cpu = x_cpu ** 2
    y_sky = x_sky ** 2

    # Non-scalar output requires explicit grad_tensors
    grad_output_cpu = torch.randn(3, 3)
    grad_output_sky = grad_output_cpu.to(device)

    y_cpu.backward(grad_output_cpu)
    y_sky.backward(grad_output_sky)

    torch.testing.assert_close(x_sky.grad.cpu(), x_cpu.grad, check_device=False)


@pytest.mark.it
@pytest.mark.asyncio
async def test_backward_on_freed_graph_errors(device):
    """Test that calling backward() twice without retain_graph raises RuntimeError."""
    x = torch.randn(2, 2, device=device, requires_grad=True)
    y = (x ** 2).sum()
    y.backward()
    with pytest.raises(RuntimeError):
        y.backward()


# =============================================================================
# Higher-Order Gradient Tests
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_double_backward(device):
    """Test backward with create_graph=True followed by a second backward."""
    x_cpu = torch.randn(3, requires_grad=True)
    x_sky = x_cpu.to(device).detach().requires_grad_()

    # y = x^3, dy/dx = 3x^2, d2y/dx2 = 6x
    y_cpu = (x_cpu ** 3).sum()
    y_sky = (x_sky ** 3).sum()

    # First backward with create_graph to keep the graph for second derivative
    (g1_cpu,) = torch.autograd.grad(y_cpu, x_cpu, create_graph=True)
    (g1_sky,) = torch.autograd.grad(y_sky, x_sky, create_graph=True)

    torch.testing.assert_close(g1_sky.cpu(), g1_cpu, check_device=False)

    # Second backward: gradient of the gradient
    (g2_cpu,) = torch.autograd.grad(g1_cpu.sum(), x_cpu)
    (g2_sky,) = torch.autograd.grad(g1_sky.sum(), x_sky)

    torch.testing.assert_close(g2_sky.cpu(), g2_cpu, check_device=False)
    # d2/dx2(x^3) = 6x
    expected = 6 * x_cpu
    torch.testing.assert_close(g2_cpu, expected)


@pytest.mark.it
@pytest.mark.asyncio
async def test_autograd_grad_create_graph(device):
    """Test torch.autograd.grad(create_graph=True) for gradient-based regularization."""
    x_cpu = torch.randn(4, requires_grad=True)
    x_sky = x_cpu.to(device).detach().requires_grad_()

    # Compute loss and gradient
    loss_cpu = (x_cpu ** 2).sum()
    loss_sky = (x_sky ** 2).sum()

    (g_cpu,) = torch.autograd.grad(loss_cpu, x_cpu, create_graph=True)
    (g_sky,) = torch.autograd.grad(loss_sky, x_sky, create_graph=True)

    # Gradient penalty: penalize gradient magnitude
    penalty_cpu = (g_cpu ** 2).sum()
    penalty_sky = (g_sky ** 2).sum()

    # Backward through penalty to get second-order gradient
    (g2_cpu,) = torch.autograd.grad(penalty_cpu, x_cpu)
    (g2_sky,) = torch.autograd.grad(penalty_sky, x_sky)

    torch.testing.assert_close(g2_sky.cpu(), g2_cpu, check_device=False)


# =============================================================================
# Numerical Gradient Verification Tests
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_gradcheck_elementwise(device):
    """Test gradcheck on elementwise ops (x^2 + x).sum() with numerical Jacobian verification."""

    def func(x):
        return (x ** 2 + x).sum()

    x = torch.randn(3, 3, dtype=torch.double, device=device, requires_grad=True)
    assert gradcheck(func, (x,), raise_exception=True)


@pytest.mark.it
@pytest.mark.asyncio
async def test_gradcheck_matmul(device):
    """Test gradcheck on matmul (a @ b).sum() with numerical Jacobian verification."""

    def func(a, b):
        return (a @ b).sum()

    a = torch.randn(3, 4, dtype=torch.double, device=device, requires_grad=True)
    b = torch.randn(4, 5, dtype=torch.double, device=device, requires_grad=True)
    assert gradcheck(func, (a, b), raise_exception=True)


@pytest.mark.it
@pytest.mark.asyncio
async def test_gradgradcheck_double_backward(device):
    """Test gradgradcheck on x^3 for second-order numerical gradient verification."""

    def func(x):
        return (x ** 3).sum()

    x = torch.randn(3, dtype=torch.double, device=device, requires_grad=True)
    assert gradgradcheck(func, (x,), raise_exception=True)


# =============================================================================
# In-Place Operations with Autograd Tests
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_inplace_on_leaf_requires_grad_errors(device):
    """Test that in-place operations on a leaf tensor with requires_grad raise an error."""
    x_sky = torch.randn(2, 2, device=device, requires_grad=True)

    with pytest.raises(RuntimeError):
        x_sky.add_(1)


@pytest.mark.it
@pytest.mark.asyncio
async def test_inplace_after_backward_safe(device):
    """Test that in-place operations on .grad are allowed after backward."""
    x_cpu = torch.randn(2, 2, requires_grad=True)
    x_sky = x_cpu.to(device).detach().requires_grad_()

    loss_cpu = (x_cpu * 2).sum()
    loss_cpu.backward()

    loss_sky = (x_sky * 2).sum()
    loss_sky.backward()

    # In-place on .grad is allowed (this is how optimizers zero grads)
    x_cpu.grad.mul_(0.5)
    x_sky.grad.mul_(0.5)

    torch.testing.assert_close(x_sky.grad.cpu(), x_cpu.grad, check_device=False)


# =============================================================================
# Nested Gradient Context Tests
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_no_grad_context(device):
    """Test torch.no_grad() context behavior."""
    x_cpu = torch.randn(2, 2, requires_grad=True)
    x_sky = x_cpu.to(device).detach().requires_grad_()

    with torch.no_grad():
        y_sky = x_sky * 2
        assert not y_sky.requires_grad


@pytest.mark.it
@pytest.mark.asyncio
async def test_enable_grad_inside_no_grad(device):
    """Test that enable_grad() re-enables tracking inside no_grad()."""
    x_cpu = torch.randn(2, 2, requires_grad=True)
    x_sky = x_cpu.to(device).detach().requires_grad_()

    with torch.no_grad():
        # Outer no_grad: no tracking
        y_sky = x_sky * 2
        assert not y_sky.requires_grad

        with torch.enable_grad():
            # Inner enable_grad: tracking re-enabled
            z_sky = x_sky * 3
            assert z_sky.requires_grad

        # Back to no_grad
        w_sky = x_sky * 4
        assert not w_sky.requires_grad

    # Outside both: tracking enabled
    v_sky = x_sky * 5
    assert v_sky.requires_grad


@pytest.mark.it
@pytest.mark.asyncio
async def test_inference_mode(device):
    """Test that torch.inference_mode() disables grad tracking."""
    x_cpu = torch.randn(2, 2, requires_grad=True)
    x_sky = x_cpu.to(device).detach().requires_grad_()

    with torch.inference_mode():
        y_sky = x_sky * 2
        assert not y_sky.requires_grad

    # Outside inference mode: tracking enabled again
    z_sky = x_sky * 2
    assert z_sky.requires_grad


# =============================================================================
# retain_grad Tests
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_retain_grad(device):
    """Test retain_grad() on an intermediate (non-leaf) tensor."""
    x_cpu = torch.randn(3, 3, requires_grad=True)
    x_sky = x_cpu.to(device).detach().requires_grad_()

    y_cpu = x_cpu * 2
    y_cpu.retain_grad()

    y_sky = x_sky * 2
    y_sky.retain_grad()

    y_cpu.sum().backward()
    y_sky.sum().backward()

    assert y_cpu.grad is not None
    assert y_sky.grad is not None
    torch.testing.assert_close(y_sky.grad.cpu(), y_cpu.grad, check_device=False)
    torch.testing.assert_close(x_sky.grad.cpu(), x_cpu.grad, check_device=False)


@pytest.mark.it
@pytest.mark.asyncio
async def test_retain_grad_multiple_intermediates(device):
    """Test retain_grad() on multiple intermediate tensors in a chain."""
    x_cpu = torch.randn(3, 3, requires_grad=True)
    x_sky = x_cpu.to(device).detach().requires_grad_()

    y_cpu = x_cpu * 2
    z_cpu = y_cpu + 1
    y_cpu.retain_grad()
    z_cpu.retain_grad()

    y_sky = x_sky * 2
    z_sky = y_sky + 1
    y_sky.retain_grad()
    z_sky.retain_grad()

    z_cpu.sum().backward()
    z_sky.sum().backward()

    assert y_cpu.grad is not None and z_cpu.grad is not None
    assert y_sky.grad is not None and z_sky.grad is not None
    torch.testing.assert_close(y_sky.grad.cpu(), y_cpu.grad, check_device=False)
    torch.testing.assert_close(z_sky.grad.cpu(), z_cpu.grad, check_device=False)


@pytest.mark.it
@pytest.mark.asyncio
async def test_without_retain_grad_no_intermediate_gradients(device):
    """Test that intermediate grads are discarded without retain_grad()."""
    x_cpu = torch.randn(3, 3, requires_grad=True)
    x_sky = x_cpu.to(device).detach().requires_grad_()

    y_cpu = x_cpu * 2
    y_sky = x_sky * 2

    y_cpu.sum().backward()
    y_sky.sum().backward()

    # Intermediate grads discarded by default
    assert y_cpu.grad is None
    assert y_sky.grad is None
    # Leaf grads still present
    assert x_cpu.grad is not None
    assert x_sky.grad is not None


@pytest.mark.it
@pytest.mark.asyncio
async def test_retain_grad_with_inplace_on_non_leaf(device):
    """Test retain_grad with in-place op on non-leaf tensor."""
    x_cpu = torch.randn(3, 3, requires_grad=True)
    x_sky = x_cpu.to(device).detach().requires_grad_()

    y_cpu = x_cpu * 2
    y_cpu.retain_grad()
    y_cpu.mul_(2)

    y_sky = x_sky * 2
    y_sky.retain_grad()
    y_sky.mul_(2)

    y_cpu.sum().backward()
    y_sky.sum().backward()

    assert y_sky.grad is not None
    torch.testing.assert_close(y_sky.grad.cpu(), y_cpu.grad, check_device=False)
    torch.testing.assert_close(x_sky.grad.cpu(), x_cpu.grad, check_device=False)


# =============================================================================
# Gradient Accumulation Tests
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_gradient_accumulation(device):
    """Test that calling backward() twice accumulates gradients."""
    x_cpu = torch.randn(3, 3, requires_grad=True)
    x_sky = x_cpu.to(device).detach().requires_grad_()

    loss_cpu = (x_cpu * 2).sum()
    loss_sky = (x_sky * 2).sum()

    # First backward (retain_graph so we can backward again)
    loss_cpu.backward(retain_graph=True)
    loss_sky.backward(retain_graph=True)

    # Second backward — gradients accumulate
    loss_cpu.backward()
    loss_sky.backward()

    # After two backwards, grad should be 2x a single backward
    torch.testing.assert_close(x_sky.grad.cpu(), x_cpu.grad, check_device=False)
    expected = torch.full_like(x_cpu, 4.0)  # 2 * 2 (grad=2 each time, accumulated)
    torch.testing.assert_close(x_cpu.grad, expected)


@pytest.mark.it
@pytest.mark.asyncio
async def test_gradient_accumulation_different_losses(device):
    """Test gradient accumulation from two different losses on a shared graph."""
    x_cpu = torch.randn(3, 3, requires_grad=True)
    x_sky = x_cpu.to(device).detach().requires_grad_()

    h_cpu = x_cpu ** 2
    h_sky = x_sky ** 2

    loss1_cpu = h_cpu.sum()
    loss2_cpu = (h_cpu * 3).sum()

    loss1_sky = h_sky.sum()
    loss2_sky = (h_sky * 3).sum()

    # First backward with retain_graph
    loss1_cpu.backward(retain_graph=True)
    loss1_sky.backward(retain_graph=True)

    # Second backward accumulates
    loss2_cpu.backward()
    loss2_sky.backward()

    # Accumulated grad = grad(loss1) + grad(loss2) = 2x + 6x = 8x
    torch.testing.assert_close(x_sky.grad.cpu(), x_cpu.grad, check_device=False)


# =============================================================================
# grad.zero_ and grad = None Tests
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_grad_zero_(device):
    """Test that grad.zero_() zeros the gradient and subsequent backward accumulates."""
    x_cpu = torch.randn(3, 3, requires_grad=True)
    x_sky = x_cpu.to(device).detach().requires_grad_()

    loss_cpu = (x_cpu * 2).sum()
    loss_sky = (x_sky * 2).sum()

    loss_cpu.backward()
    loss_sky.backward()

    # Zero out gradients
    x_cpu.grad.zero_()
    x_sky.grad.zero_()

    # Verify grad is zero (not None)
    assert x_cpu.grad is not None
    assert x_sky.grad is not None
    torch.testing.assert_close(
        x_sky.grad.cpu(), torch.zeros(3, 3), check_device=False
    )

    # Backward again — should accumulate on top of zeros (same as fresh)
    loss_cpu2 = (x_cpu * 3).sum()
    loss_sky2 = (x_sky * 3).sum()
    loss_cpu2.backward()
    loss_sky2.backward()

    torch.testing.assert_close(x_sky.grad.cpu(), x_cpu.grad, check_device=False)
    expected = torch.full_like(x_cpu, 3.0)
    torch.testing.assert_close(x_cpu.grad, expected)


@pytest.mark.it
@pytest.mark.asyncio
async def test_grad_set_to_none(device):
    """Test that setting grad = None clears it and next backward gives fresh grad."""
    x_cpu = torch.randn(3, 3, requires_grad=True)
    x_sky = x_cpu.to(device).detach().requires_grad_()

    loss_cpu = (x_cpu * 2).sum()
    loss_sky = (x_sky * 2).sum()

    loss_cpu.backward()
    loss_sky.backward()

    # Set grad to None
    x_cpu.grad = None
    x_sky.grad = None

    assert x_cpu.grad is None
    assert x_sky.grad is None

    # Backward again — fresh gradient (not accumulated)
    loss_cpu2 = (x_cpu * 5).sum()
    loss_sky2 = (x_sky * 5).sum()
    loss_cpu2.backward()
    loss_sky2.backward()

    torch.testing.assert_close(x_sky.grad.cpu(), x_cpu.grad, check_device=False)
    expected = torch.full_like(x_cpu, 5.0)
    torch.testing.assert_close(x_cpu.grad, expected)


@pytest.mark.it
@pytest.mark.asyncio
async def test_grad_zero_vs_none_accumulation(device):
    """Test difference between grad.zero_() and grad = None on accumulation."""
    a_cpu = torch.randn(3, 3, requires_grad=True)
    b_cpu = torch.randn(3, 3, requires_grad=True)
    a_sky = a_cpu.to(device).detach().requires_grad_()
    b_sky = b_cpu.to(device).detach().requires_grad_()

    # First backward
    loss_cpu = (a_cpu * 2 + b_cpu * 3).sum()
    loss_sky = (a_sky * 2 + b_sky * 3).sum()
    loss_cpu.backward()
    loss_sky.backward()

    # Zero a's grad, set b's grad to None
    a_cpu.grad.zero_()
    a_sky.grad.zero_()
    b_cpu.grad = None
    b_sky.grad = None

    # Second backward
    loss2_cpu = (a_cpu * 7 + b_cpu * 11).sum()
    loss2_sky = (a_sky * 7 + b_sky * 11).sum()
    loss2_cpu.backward()
    loss2_sky.backward()

    # a: zero + 7 = 7 (accumulated on zeros)
    torch.testing.assert_close(a_sky.grad.cpu(), a_cpu.grad, check_device=False)
    torch.testing.assert_close(a_cpu.grad, torch.full_like(a_cpu, 7.0))

    # b: None → fresh 11 (no accumulation)
    torch.testing.assert_close(b_sky.grad.cpu(), b_cpu.grad, check_device=False)
    torch.testing.assert_close(b_cpu.grad, torch.full_like(b_cpu, 11.0))


# =============================================================================
# requires_grad Tests
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_requires_grad_kwarg_constructor(device):
    """Test creating tensors with requires_grad=True on sky device."""
    x = torch.randn(3, 3, device=device, requires_grad=True)
    assert x.requires_grad
    assert x.is_leaf

    # Operations produce tensors with requires_grad=True
    y = x * 2
    assert y.requires_grad


@pytest.mark.it
@pytest.mark.asyncio
async def test_requires_grad_method(device):
    """Test requires_grad_() method toggles gradient tracking."""
    x = torch.randn(3, 3, device=device)
    assert not x.requires_grad

    x.requires_grad_()
    assert x.requires_grad
    assert x.is_leaf

    x.requires_grad_(False)
    assert not x.requires_grad


@pytest.mark.it
@pytest.mark.asyncio
async def test_requires_grad_propagation(device):
    """Test that requires_grad propagates through operations."""
    x_grad = torch.randn(3, 3, device=device, requires_grad=True)
    x_no_grad = torch.randn(3, 3, device=device)

    # Ops on grad-enabled tensor produce grad-enabled output
    y = x_grad * 2 + 1
    assert y.requires_grad

    # Ops on non-grad tensor produce non-grad output
    z = x_no_grad * 2 + 1
    assert not z.requires_grad


@pytest.mark.it
@pytest.mark.asyncio
async def test_requires_grad_false_no_graph(device):
    """Test that only grad-enabled tensors receive gradients."""
    a_cpu = torch.randn(3, 3, requires_grad=True)
    b_cpu = torch.randn(3, 3)
    a_sky = a_cpu.to(device).detach().requires_grad_()
    b_sky = b_cpu.to(device)

    loss_cpu = (a_cpu + b_cpu).sum()
    loss_sky = (a_sky + b_sky).sum()

    loss_cpu.backward()
    loss_sky.backward()

    assert a_cpu.grad is not None
    assert a_sky.grad is not None
    torch.testing.assert_close(a_sky.grad.cpu(), a_cpu.grad, check_device=False)

    assert b_cpu.grad is None
    assert b_sky.grad is None


# =============================================================================
# grad on Detached Tensors Tests
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_detach(device):
    """Test that detach() creates a tensor that doesn't track gradients."""
    x_cpu = torch.randn(3, 3, requires_grad=True)
    x_sky = x_cpu.to(device).detach().requires_grad_()

    y_cpu = x_cpu.detach()
    y_sky = x_sky.detach()

    assert not y_cpu.requires_grad
    assert not y_sky.requires_grad
    assert y_cpu.grad is None
    assert y_sky.grad is None

    # Backward through x still works
    loss_cpu = (x_cpu ** 2).sum()
    loss_sky = (x_sky ** 2).sum()
    loss_cpu.backward()
    loss_sky.backward()

    assert x_cpu.grad is not None
    assert x_sky.grad is not None
    torch.testing.assert_close(x_sky.grad.cpu(), x_cpu.grad, check_device=False)


@pytest.mark.it
@pytest.mark.asyncio
async def test_detach_stops_gradient_flow(device):
    """Test that detach() severs the computation graph and stops gradient flow."""
    x_cpu = torch.randn(3, 3, requires_grad=True)
    x_sky = x_cpu.to(device).detach().requires_grad_()

    y_cpu = x_cpu * 2
    z_cpu = y_cpu.detach() * 3
    loss_cpu = z_cpu.sum()

    y_sky = x_sky * 2
    z_sky = y_sky.detach() * 3
    loss_sky = z_sky.sum()

    # z doesn't require grad because detach severs the graph
    assert not z_cpu.requires_grad
    assert not z_sky.requires_grad

    # backward won't reach x because detach severs the connection
    # loss has no grad-enabled inputs, so backward raises or grad stays None
    assert x_cpu.grad is None
    assert x_sky.grad is None


@pytest.mark.it
@pytest.mark.asyncio
async def test_detach_requires_grad_new_leaf(device):
    """Test detach().requires_grad_() creates a new independent leaf."""
    x_cpu = torch.randn(3, 3, requires_grad=True)
    x_sky = x_cpu.to(device).detach().requires_grad_()

    # Create new leaf via detach + requires_grad_
    y_cpu = x_cpu.detach().requires_grad_()
    y_sky = x_sky.detach().requires_grad_()

    assert y_cpu.is_leaf
    assert y_sky.is_leaf

    # Build graph from y and backward
    loss_cpu = (y_cpu ** 2).sum()
    loss_sky = (y_sky ** 2).sum()
    loss_cpu.backward()
    loss_sky.backward()

    # y has grad, x does not (disconnected)
    assert y_cpu.grad is not None
    assert y_sky.grad is not None
    torch.testing.assert_close(y_sky.grad.cpu(), y_cpu.grad, check_device=False)

    assert x_cpu.grad is None
    assert x_sky.grad is None


# =============================================================================
# torch.no_grad Additional Tests
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_no_grad_no_graph_built(device):
    """Test that no_grad() prevents graph construction."""
    x_cpu = torch.randn(3, 3, requires_grad=True)
    x_sky = x_cpu.to(device).detach().requires_grad_()

    # Compute something before no_grad
    y_cpu = (x_cpu ** 2).sum()
    y_sky = (x_sky ** 2).sum()

    with torch.no_grad():
        z_cpu = x_cpu * 2 + x_cpu
        z_sky = x_sky * 2 + x_sky
        assert not z_cpu.requires_grad
        assert not z_sky.requires_grad

    # Backward on graph built before no_grad still works
    y_cpu.backward()
    y_sky.backward()
    torch.testing.assert_close(x_sky.grad.cpu(), x_cpu.grad, check_device=False)


@pytest.mark.it
@pytest.mark.asyncio
async def test_no_grad_as_decorator(device):
    """Test torch.no_grad() used as a function decorator."""

    @torch.no_grad()
    def compute(x):
        return x * 2 + x ** 2

    x_cpu = torch.randn(3, 3, requires_grad=True)
    x_sky = x_cpu.to(device).detach().requires_grad_()

    y_cpu = compute(x_cpu)
    y_sky = compute(x_sky)

    assert not y_cpu.requires_grad
    assert not y_sky.requires_grad

    torch.testing.assert_close(y_sky.cpu(), y_cpu, check_device=False)


@pytest.mark.it
@pytest.mark.asyncio
async def test_no_grad_inplace_on_leaf_allowed(device):
    """Test that in-place ops on leaves with requires_grad are allowed under no_grad()."""
    x_cpu = torch.randn(3, 3, requires_grad=True)
    x_sky = x_cpu.to(device).detach().requires_grad_()

    # Normally, in-place on leaf with requires_grad raises
    with pytest.raises(RuntimeError):
        x_sky.mul_(2)

    # Under no_grad, it should succeed
    with torch.no_grad():
        x_cpu.mul_(2)
        x_sky.mul_(2)

    torch.testing.assert_close(x_sky.cpu(), x_cpu, check_device=False)
