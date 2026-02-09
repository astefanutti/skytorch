"""Neural network operation correctness tests.

Tests conv2d, batch_norm, pooling, dropout, nll_loss, embedding, cat,
and a small MNIST-like CNN end-to-end.
"""

import pytest
import torch
import torch.nn.functional as F


# =============================================================================
# Conv2d
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_conv2d_forward(device):
    x_cpu = torch.randn(1, 1, 8, 8)
    w_cpu = torch.randn(2, 1, 3, 3)
    b_cpu = torch.randn(2)

    cpu_result = F.conv2d(x_cpu, w_cpu, b_cpu)
    sky_result = F.conv2d(x_cpu.to(device), w_cpu.to(device), b_cpu.to(device))

    torch.testing.assert_close(
        sky_result.cpu(), cpu_result, atol=1e-4, rtol=1e-4, check_device=False
    )


@pytest.mark.it
@pytest.mark.asyncio
async def test_conv2d_grad(device):
    x_cpu = torch.randn(1, 1, 8, 8, requires_grad=True)
    w_cpu = torch.randn(2, 1, 3, 3, requires_grad=True)
    b_cpu = torch.randn(2, requires_grad=True)

    x_sky = x_cpu.clone().to(device).detach().requires_grad_(True)
    w_sky = w_cpu.clone().to(device).detach().requires_grad_(True)
    b_sky = b_cpu.clone().to(device).detach().requires_grad_(True)

    loss_cpu = F.conv2d(x_cpu, w_cpu, b_cpu).sum()
    loss_cpu.backward()

    loss_sky = F.conv2d(x_sky, w_sky, b_sky).sum()
    loss_sky.backward()

    for name, sky_t, cpu_t in [
        ("input", x_sky, x_cpu),
        ("weight", w_sky, w_cpu),
        ("bias", b_sky, b_cpu),
    ]:
        assert sky_t.grad is not None, f"conv2d {name} gradient is None"
        torch.testing.assert_close(
            sky_t.grad.cpu(),
            cpu_t.grad,
            atol=1e-4,
            rtol=1e-4,
            check_device=False,
        )


# =============================================================================
# BatchNorm
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_batch_norm_train_forward(device):
    x_cpu = torch.randn(4, 3, 8, 8)
    w_cpu = torch.randn(3)
    b_cpu = torch.randn(3)
    rm = torch.zeros(3)
    rv = torch.ones(3)

    cpu_result = F.batch_norm(x_cpu, rm.clone(), rv.clone(), w_cpu, b_cpu, training=True)
    sky_result = F.batch_norm(
        x_cpu.to(device),
        rm.clone().to(device),
        rv.clone().to(device),
        w_cpu.to(device),
        b_cpu.to(device),
        training=True,
    )

    torch.testing.assert_close(
        sky_result.cpu(), cpu_result, atol=1e-4, rtol=1e-4, check_device=False
    )


@pytest.mark.it
@pytest.mark.asyncio
async def test_batch_norm_train_grad(device):
    x_cpu = torch.randn(4, 3, 8, 8, requires_grad=True)
    w_cpu = torch.randn(3, requires_grad=True)
    b_cpu = torch.randn(3, requires_grad=True)
    rm = torch.zeros(3)
    rv = torch.ones(3)

    x_sky = x_cpu.clone().to(device).detach().requires_grad_(True)
    w_sky = w_cpu.clone().to(device).detach().requires_grad_(True)
    b_sky = b_cpu.clone().to(device).detach().requires_grad_(True)

    loss_cpu = F.batch_norm(
        x_cpu, rm.clone(), rv.clone(), w_cpu, b_cpu, training=True
    ).sum()
    loss_cpu.backward()

    loss_sky = F.batch_norm(
        x_sky,
        rm.clone().to(device),
        rv.clone().to(device),
        w_sky,
        b_sky,
        training=True,
    ).sum()
    loss_sky.backward()

    for name, sky_t, cpu_t in [
        ("input", x_sky, x_cpu),
        ("weight", w_sky, w_cpu),
        ("bias", b_sky, b_cpu),
    ]:
        assert sky_t.grad is not None, f"batch_norm {name} gradient is None"
        torch.testing.assert_close(
            sky_t.grad.cpu(),
            cpu_t.grad,
            atol=1e-4,
            rtol=1e-4,
            check_device=False,
        )


@pytest.mark.it
@pytest.mark.asyncio
async def test_batch_norm_eval_forward(device):
    x_cpu = torch.randn(4, 3, 8, 8)
    w_cpu = torch.randn(3)
    b_cpu = torch.randn(3)
    rm = torch.randn(3)
    rv = torch.abs(torch.randn(3)) + 0.1

    cpu_result = F.batch_norm(
        x_cpu, rm.clone(), rv.clone(), w_cpu, b_cpu, training=False
    )
    sky_result = F.batch_norm(
        x_cpu.to(device),
        rm.clone().to(device),
        rv.clone().to(device),
        w_cpu.to(device),
        b_cpu.to(device),
        training=False,
    )

    torch.testing.assert_close(
        sky_result.cpu(), cpu_result, atol=1e-4, rtol=1e-4, check_device=False
    )


# =============================================================================
# Pooling
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pool_name,pool_fn",
    [
        ("max_pool2d", lambda x: F.max_pool2d(x, 2)),
        ("avg_pool2d", lambda x: F.avg_pool2d(x, 2)),
        ("adaptive_avg_pool2d", lambda x: F.adaptive_avg_pool2d(x, (2, 2))),
    ],
)
async def test_pool_forward(device, pool_name, pool_fn):
    x_cpu = torch.randn(1, 2, 8, 8)
    cpu_result = pool_fn(x_cpu)
    sky_result = pool_fn(x_cpu.to(device))
    torch.testing.assert_close(sky_result.cpu(), cpu_result, check_device=False)


@pytest.mark.it
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pool_name,pool_fn",
    [
        ("max_pool2d", lambda x: F.max_pool2d(x, 2)),
        ("avg_pool2d", lambda x: F.avg_pool2d(x, 2)),
        ("adaptive_avg_pool2d", lambda x: F.adaptive_avg_pool2d(x, (2, 2))),
    ],
)
async def test_pool_grad(device, pool_name, pool_fn):
    x_cpu = torch.randn(1, 2, 8, 8, requires_grad=True)
    x_sky = x_cpu.clone().to(device).detach().requires_grad_(True)

    loss_cpu = pool_fn(x_cpu).sum()
    loss_cpu.backward()

    loss_sky = pool_fn(x_sky).sum()
    loss_sky.backward()

    assert x_sky.grad is not None, f"{pool_name} gradient is None"
    torch.testing.assert_close(
        x_sky.grad.cpu(), x_cpu.grad, atol=1e-4, rtol=1e-4, check_device=False
    )


# =============================================================================
# Dropout
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_dropout_p0_identity(device):
    """Dropout with p=0 should be identity."""
    x_cpu = torch.randn(4, 4)
    result = F.dropout(x_cpu.to(device), p=0.0, training=True)
    torch.testing.assert_close(result.cpu(), x_cpu, check_device=False)


@pytest.mark.it
@pytest.mark.asyncio
async def test_dropout_p1_zero(device):
    """Dropout with p=1 should zero everything."""
    x_cpu = torch.randn(4, 4)
    result = F.dropout(x_cpu.to(device), p=1.0, training=True)
    torch.testing.assert_close(
        result.cpu(), torch.zeros(4, 4), check_device=False
    )


@pytest.mark.it
@pytest.mark.asyncio
async def test_dropout_eval_identity(device):
    """Dropout in eval mode should be identity regardless of p."""
    x_cpu = torch.randn(4, 4)
    result = F.dropout(x_cpu.to(device), p=0.5, training=False)
    torch.testing.assert_close(result.cpu(), x_cpu, check_device=False)


@pytest.mark.it
@pytest.mark.asyncio
async def test_dropout_p0_grad(device):
    """Gradient through dropout with p=0 should be identity."""
    x_cpu = torch.randn(4, 4, requires_grad=True)
    x_sky = x_cpu.clone().to(device).detach().requires_grad_(True)

    loss_cpu = F.dropout(x_cpu, p=0.0, training=True).sum()
    loss_cpu.backward()

    loss_sky = F.dropout(x_sky, p=0.0, training=True).sum()
    loss_sky.backward()

    assert x_sky.grad is not None, "dropout gradient is None"
    torch.testing.assert_close(x_sky.grad.cpu(), x_cpu.grad, check_device=False)


# =============================================================================
# NLL Loss
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_nll_loss_forward(device):
    logits_cpu = torch.randn(4, 10)
    target_cpu = torch.tensor([3, 5, 7, 1])

    log_probs_cpu = F.log_softmax(logits_cpu, dim=1)
    cpu_result = F.nll_loss(log_probs_cpu, target_cpu)

    log_probs_sky = F.log_softmax(logits_cpu.to(device), dim=1)
    sky_result = F.nll_loss(log_probs_sky, target_cpu.to(device))

    torch.testing.assert_close(
        sky_result.cpu(), cpu_result, atol=1e-4, rtol=1e-4, check_device=False
    )


@pytest.mark.it
@pytest.mark.asyncio
async def test_nll_loss_grad(device):
    logits_cpu = torch.randn(4, 10, requires_grad=True)
    target_cpu = torch.tensor([3, 5, 7, 1])

    logits_sky = logits_cpu.clone().to(device).detach().requires_grad_(True)

    log_probs_cpu = F.log_softmax(logits_cpu, dim=1)
    loss_cpu = F.nll_loss(log_probs_cpu, target_cpu)
    loss_cpu.backward()

    log_probs_sky = F.log_softmax(logits_sky, dim=1)
    loss_sky = F.nll_loss(log_probs_sky, target_cpu.to(device))
    loss_sky.backward()

    assert logits_sky.grad is not None, "nll_loss gradient is None"
    torch.testing.assert_close(
        logits_sky.grad.cpu(),
        logits_cpu.grad,
        atol=1e-4,
        rtol=1e-4,
        check_device=False,
    )


# =============================================================================
# Embedding
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_embedding_forward(device):
    weight_cpu = torch.randn(10, 4)
    indices = torch.tensor([1, 3, 5, 7])

    cpu_result = F.embedding(indices, weight_cpu)
    sky_result = F.embedding(indices.to(device), weight_cpu.to(device))

    torch.testing.assert_close(sky_result.cpu(), cpu_result, check_device=False)


@pytest.mark.it
@pytest.mark.asyncio
async def test_embedding_grad(device):
    weight_cpu = torch.randn(10, 4, requires_grad=True)
    weight_sky = weight_cpu.clone().to(device).detach().requires_grad_(True)
    indices = torch.tensor([1, 3, 5, 7])

    loss_cpu = F.embedding(indices, weight_cpu).sum()
    loss_cpu.backward()

    loss_sky = F.embedding(indices.to(device), weight_sky).sum()
    loss_sky.backward()

    assert weight_sky.grad is not None, "embedding weight gradient is None"
    torch.testing.assert_close(
        weight_sky.grad.cpu(), weight_cpu.grad, check_device=False
    )


# =============================================================================
# Cat
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_cat_forward(device):
    a_cpu = torch.randn(2, 3)
    b_cpu = torch.randn(2, 3)
    c_cpu = torch.randn(2, 3)

    cpu_result = torch.cat([a_cpu, b_cpu, c_cpu], dim=0)
    sky_result = torch.cat(
        [a_cpu.to(device), b_cpu.to(device), c_cpu.to(device)], dim=0
    )

    torch.testing.assert_close(sky_result.cpu(), cpu_result, check_device=False)


@pytest.mark.it
@pytest.mark.asyncio
async def test_cat_dim1_forward(device):
    a_cpu = torch.randn(2, 3)
    b_cpu = torch.randn(2, 4)

    cpu_result = torch.cat([a_cpu, b_cpu], dim=1)
    sky_result = torch.cat([a_cpu.to(device), b_cpu.to(device)], dim=1)

    torch.testing.assert_close(sky_result.cpu(), cpu_result, check_device=False)


@pytest.mark.it
@pytest.mark.asyncio
async def test_cat_grad(device):
    a_cpu = torch.randn(2, 3, requires_grad=True)
    b_cpu = torch.randn(2, 3, requires_grad=True)

    a_sky = a_cpu.clone().to(device).detach().requires_grad_(True)
    b_sky = b_cpu.clone().to(device).detach().requires_grad_(True)

    loss_cpu = torch.cat([a_cpu, b_cpu], dim=0).sum()
    loss_cpu.backward()

    loss_sky = torch.cat([a_sky, b_sky], dim=0).sum()
    loss_sky.backward()

    assert a_sky.grad is not None, "cat gradient for first tensor is None"
    assert b_sky.grad is not None, "cat gradient for second tensor is None"
    torch.testing.assert_close(a_sky.grad.cpu(), a_cpu.grad, check_device=False)
    torch.testing.assert_close(b_sky.grad.cpu(), b_cpu.grad, check_device=False)


# =============================================================================
# MNIST-like CNN end-to-end
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_mnist_cnn_forward_backward(device):
    """Small CNN reproducing the MNIST architecture: forward + backward.

    Architecture: conv -> relu -> conv -> relu -> max_pool -> flatten -> relu -> linear
    Compare all parameter gradients against CPU reference.
    """
    torch.manual_seed(42)

    # Model parameters
    conv1_w = torch.randn(4, 1, 3, 3, requires_grad=True)
    conv1_b = torch.randn(4, requires_grad=True)
    conv2_w = torch.randn(8, 4, 3, 3, requires_grad=True)
    conv2_b = torch.randn(8, requires_grad=True)
    # After conv1(8x8)->6x6, conv2->4x4, pool(2)->2x2: 8*2*2=32
    fc_w = torch.randn(32, 10, requires_grad=True)
    fc_b = torch.randn(10, requires_grad=True)

    cpu_params = [conv1_w, conv1_b, conv2_w, conv2_b, fc_w, fc_b]
    sky_params = [p.clone().to(device).detach().requires_grad_(True) for p in cpu_params]
    param_names = ["conv1_w", "conv1_b", "conv2_w", "conv2_b", "fc_w", "fc_b"]

    def forward(x, params):
        c1w, c1b, c2w, c2b, fw, fb = params
        x = F.relu(F.conv2d(x, c1w, c1b))
        x = F.relu(F.conv2d(x, c2w, c2b))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(x.mm(fw) + fb)
        return x

    # Input
    x_cpu = torch.randn(2, 1, 8, 8)
    target = torch.tensor([3, 7])

    # CPU forward + backward
    out_cpu = forward(x_cpu, cpu_params)
    loss_cpu = F.nll_loss(F.log_softmax(out_cpu, dim=1), target)
    loss_cpu.backward()

    # Sky forward + backward
    out_sky = forward(x_cpu.to(device), sky_params)
    loss_sky = F.nll_loss(
        F.log_softmax(out_sky, dim=1), target.to(device)
    )
    loss_sky.backward()

    # Compare forward output
    torch.testing.assert_close(
        out_sky.cpu(), out_cpu, atol=1e-4, rtol=1e-4, check_device=False
    )

    # Compare all parameter gradients
    for name, sky_p, cpu_p in zip(param_names, sky_params, cpu_params):
        assert sky_p.grad is not None, (
            f"MNIST CNN: {name} gradient is None. "
            f"This operation likely has an explicit PrivateUse1 registration "
            f"that breaks CompositeImplicitAutograd gradient tracking."
        )
        torch.testing.assert_close(
            sky_p.grad.cpu(),
            cpu_p.grad,
            atol=1e-3,
            rtol=1e-3,
            check_device=False,
            msg=f"MNIST CNN: {name} gradient mismatch",
        )
