"""Real-world workflow integration tests.

Tests complete training loops, multi-layer models, gradient accumulation,
train/eval mode switching, loss.item(), and CNN-like pipeline chains.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Simple linear regression training loop
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_linear_regression_training(device):
    """Full train loop: forward -> loss -> backward -> step, loss decreases."""
    torch.manual_seed(42)

    # Target: y = 3*x + 1
    x_data = torch.linspace(-1, 1, 20).unsqueeze(1).to(device)
    y_data = (3 * x_data + 1)

    model = nn.Linear(1, 1).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    losses = []
    for _ in range(50):
        optimizer.zero_grad()
        pred = model(x_data)
        loss = F.mse_loss(pred, y_data)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] * 0.1, (
        f"Loss did not decrease enough: first={losses[0]:.4f}, last={losses[-1]:.4f}"
    )


# =============================================================================
# Multi-layer perceptron forward + backward
# =============================================================================


@pytest.mark.xfail(
    reason="Tensor does not exist — view/alias not tracked on server"
)
@pytest.mark.it
@pytest.mark.asyncio
async def test_mlp_forward_backward(device):
    """Forward + backward through multiple linear+relu layers, compare with CPU."""
    torch.manual_seed(42)

    # Use raw tensor parameters (same pattern as MNIST test in test_ops_nn.py)
    w1 = torch.randn(4, 8, requires_grad=True)
    b1 = torch.randn(8, requires_grad=True)
    w2 = torch.randn(8, 4, requires_grad=True)
    b2 = torch.randn(4, requires_grad=True)

    cpu_params = [w1, b1, w2, b2]
    sky_params = [p.clone().to(device).detach().requires_grad_(True) for p in cpu_params]
    param_names = ["w1", "b1", "w2", "b2"]

    def forward(x, params):
        w1, b1, w2, b2 = params
        x = F.relu(x.mm(w1) + b1)
        x = x.mm(w2) + b2
        return x

    x_cpu = torch.randn(2, 4)

    # Forward
    out_cpu = forward(x_cpu, cpu_params)
    out_sky = forward(x_cpu.to(device), sky_params)

    torch.testing.assert_close(
        out_sky.cpu(), out_cpu, atol=1e-4, rtol=1e-4, check_device=False
    )

    # Backward
    loss_cpu = out_cpu.sum()
    loss_cpu.backward()

    loss_sky = out_sky.sum()
    loss_sky.backward()

    for name, sky_p, cpu_p in zip(param_names, sky_params, cpu_params):
        assert sky_p.grad is not None, f"MLP: {name} gradient is None"
        torch.testing.assert_close(
            sky_p.grad.cpu(),
            cpu_p.grad,
            atol=1e-3,
            rtol=1e-3,
            check_device=False,
        )


# =============================================================================
# Gradient accumulation across mini-batches
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_gradient_accumulation_training(device):
    """Accumulate gradients over K mini-batches before stepping."""
    torch.manual_seed(42)

    model_cpu = nn.Linear(4, 2)
    model_sky = nn.Linear(4, 2)
    model_sky.load_state_dict(model_cpu.state_dict())
    model_sky = model_sky.to(device)

    opt_cpu = torch.optim.SGD(model_cpu.parameters(), lr=0.1)
    opt_sky = torch.optim.SGD(model_sky.parameters(), lr=0.1)

    accumulation_steps = 3
    batches = [torch.randn(2, 4) for _ in range(accumulation_steps)]

    opt_cpu.zero_grad()
    opt_sky.zero_grad()

    for batch in batches:
        # CPU
        loss_cpu = model_cpu(batch).sum() / accumulation_steps
        loss_cpu.backward()

        # Sky
        loss_sky = model_sky(batch.to(device)).sum() / accumulation_steps
        loss_sky.backward()

    # Step after accumulation
    opt_cpu.step()
    opt_sky.step()

    torch.testing.assert_close(
        model_sky.weight.cpu(),
        model_cpu.weight,
        atol=1e-4,
        rtol=1e-4,
        check_device=False,
    )
    torch.testing.assert_close(
        model_sky.bias.cpu(),
        model_cpu.bias,
        atol=1e-4,
        rtol=1e-4,
        check_device=False,
    )


# =============================================================================
# .item() on loss scalar during training
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_loss_item_during_training(device):
    """.item() on loss scalar works during training (the MNIST pattern)."""
    model = nn.Linear(4, 2).to(device)
    x = torch.randn(3, 4).to(device)

    loss = model(x).sum()
    loss_value = loss.item()

    assert isinstance(loss_value, float)
    assert loss_value == loss_value  # not NaN


# =============================================================================
# Mixed operations chain (CNN-like pipeline)
# =============================================================================


@pytest.mark.xfail(
    reason="PrivateUse1 dispatch overrides CompositeImplicitAutograd, breaking gradient"
)
@pytest.mark.it
@pytest.mark.asyncio
async def test_cnn_pipeline_chain(device):
    """Chain: conv -> relu -> pool -> flatten, compared with CPU.

    This replicates the MNIST test pattern from test_ops_nn.py but with a
    simpler architecture to test the pipeline chain.
    """
    torch.manual_seed(99)

    conv1_w = torch.randn(4, 1, 3, 3, requires_grad=True)
    conv1_b = torch.randn(4, requires_grad=True)

    cpu_params = [conv1_w, conv1_b]
    sky_params = [p.clone().to(device).detach().requires_grad_(True) for p in cpu_params]
    param_names = ["conv1_w", "conv1_b"]

    def forward(x, params):
        c1w, c1b = params
        x = F.relu(F.conv2d(x, c1w, c1b))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        return x

    x_cpu = torch.randn(2, 1, 8, 8)

    # Forward on both (matching MNIST test pattern)
    out_cpu = forward(x_cpu, cpu_params)
    out_sky = forward(x_cpu.to(device), sky_params)

    torch.testing.assert_close(
        out_sky.cpu(), out_cpu, atol=1e-4, rtol=1e-4, check_device=False
    )

    # Backward on both
    loss_cpu = out_cpu.sum()
    loss_cpu.backward()

    loss_sky = out_sky.sum()
    loss_sky.backward()

    for name, sky_p, cpu_p in zip(param_names, sky_params, cpu_params):
        assert sky_p.grad is not None, (
            f"CNN pipeline: {name} gradient is None. "
            f"This operation likely has an explicit PrivateUse1 registration "
            f"that breaks CompositeImplicitAutograd gradient tracking."
        )
        torch.testing.assert_close(
            sky_p.grad.cpu(),
            cpu_p.grad,
            atol=1e-3,
            rtol=1e-3,
            check_device=False,
        )


# =============================================================================
# Multi-step training with CPU comparison
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_multi_step_training_matches_cpu(device):
    """Multiple training steps produce same parameters as CPU reference."""
    torch.manual_seed(42)

    model_cpu = nn.Linear(4, 2)
    model_sky = nn.Linear(4, 2)
    model_sky.load_state_dict(model_cpu.state_dict())
    model_sky = model_sky.to(device)

    opt_cpu = torch.optim.SGD(model_cpu.parameters(), lr=0.01)
    opt_sky = torch.optim.SGD(model_sky.parameters(), lr=0.01)

    for i in range(5):
        torch.manual_seed(i)
        x = torch.randn(3, 4)

        # CPU step
        opt_cpu.zero_grad()
        loss_cpu = model_cpu(x).sum()
        loss_cpu.backward()
        opt_cpu.step()

        # Sky step
        opt_sky.zero_grad()
        loss_sky = model_sky(x.to(device)).sum()
        loss_sky.backward()
        opt_sky.step()

    torch.testing.assert_close(
        model_sky.weight.cpu(),
        model_cpu.weight,
        atol=1e-3,
        rtol=1e-3,
        check_device=False,
    )
    torch.testing.assert_close(
        model_sky.bias.cpu(),
        model_cpu.bias,
        atol=1e-3,
        rtol=1e-3,
        check_device=False,
    )
