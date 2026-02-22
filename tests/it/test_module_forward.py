"""Integration tests for module-level forward proxying (ExecuteModuleForward RPC)."""

import pytest
import torch
import torch.nn as nn


class InnerModule(nn.Module):
    """A simple module to proxy."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)


class OuterModel(nn.Module):
    """Model containing a sub-module that we'll proxy."""

    def __init__(self):
        super().__init__()
        self.pre = nn.Linear(4, 8)
        self.inner = InnerModule(8, 4)
        self.post = nn.Linear(4, 2)

    def forward(self, x):
        x = torch.relu(self.pre(x))
        x = self.inner(x)
        x = self.post(x)
        return x


class MultiInnerModel(nn.Module):
    """Model with multiple sub-modules to proxy via wildcard."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([InnerModule(4, 4) for _ in range(3)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MultiOutputModule(nn.Module):
    """Module that returns multiple tensors."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 4)
        self.linear2 = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear1(x), self.linear2(x)


class WrapperModel(nn.Module):
    """Model wrapping a multi-output module."""

    def __init__(self):
        super().__init__()
        self.multi = MultiOutputModule()

    def forward(self, x):
        a, b = self.multi(x)
        return a.sum() + b.sum()


# Module-level factory functions (no closures, safe for remote execution)


def _create_outer_model():
    torch.manual_seed(42)
    return OuterModel()


def _create_multi_inner_model():
    torch.manual_seed(42)
    return MultiInnerModel()


def _create_wrapper_model():
    torch.manual_seed(42)
    return WrapperModel()


@pytest.mark.it
@pytest.mark.asyncio
async def test_execute_with_retain_model(compute):
    """Execute a function that returns a model, retaining it on the server."""
    state_dict = await compute.execute(_create_outer_model, retain_model=True)

    # State dict should have a model_id
    assert state_dict.model_id > 0
    assert state_dict._compute is compute

    # Should contain all model parameters
    assert "pre.weight" in state_dict
    assert "pre.bias" in state_dict
    assert "inner.linear.weight" in state_dict
    assert "inner.linear.bias" in state_dict
    assert "post.weight" in state_dict
    assert "post.bias" in state_dict


@pytest.mark.it
@pytest.mark.asyncio
async def test_execute_module_forward_basic(compute):
    """Test proxying a sub-module's forward through ExecuteModuleForward RPC."""
    state_dict = await compute.execute(_create_outer_model, retain_model=True)

    # Load into a client-side model and proxy the inner module
    with torch.device("meta"):
        model = OuterModel()
    state_dict.load_into(model, triton_modules=["inner"])

    # Create input tensor on sky device
    device = compute.device("cpu")
    x = torch.randn(2, 4, device=device)

    # Run through pre (ATen dispatch) -> inner (proxied forward) -> post (ATen dispatch)
    h = torch.relu(model.pre(x))
    h = model.inner(h)
    result = model.post(h)

    # Verify output shape and that it's a sky tensor
    assert result.shape == (2, 2)
    assert result.device.type == "sky"

    # Verify result is finite
    result_cpu = result.cpu()
    assert torch.isfinite(result_cpu).all()


@pytest.mark.it
@pytest.mark.asyncio
async def test_execute_module_forward_wildcard(compute):
    """Test proxying modules using wildcard patterns."""
    state_dict = await compute.execute(_create_multi_inner_model, retain_model=True)

    with torch.device("meta"):
        model = MultiInnerModel()
    state_dict.load_into(model, triton_modules=["layers.*"])

    device = compute.device("cpu")
    x = torch.randn(2, 4, device=device)

    result = model(x)
    assert result.shape == (2, 4)
    assert result.device.type == "sky"

    # Verify result is finite
    result_cpu = result.cpu()
    assert torch.isfinite(result_cpu).all()


@pytest.mark.it
@pytest.mark.asyncio
async def test_execute_module_forward_multiple_outputs(compute):
    """Test proxying a module that returns multiple tensors (tuple)."""
    state_dict = await compute.execute(_create_wrapper_model, retain_model=True)

    with torch.device("meta"):
        model = WrapperModel()
    state_dict.load_into(model, triton_modules=["multi"])

    device = compute.device("cpu")
    x = torch.randn(2, 4, device=device)

    # The proxied multi module returns a tuple
    a, b = model.multi(x)
    assert a.shape == (2, 4)
    assert b.shape == (2, 2)


@pytest.mark.it
@pytest.mark.asyncio
async def test_execute_module_forward_cached(compute):
    """Test that repeated calls with same input shapes use fire-and-forget (shape cache)."""
    state_dict = await compute.execute(_create_outer_model, retain_model=True)

    with torch.device("meta"):
        model = OuterModel()
    state_dict.load_into(model, triton_modules=["inner"])

    device = compute.device("cpu")

    # First call: sync (cache miss)
    x1 = torch.randn(2, 8, device=device)
    result1 = model.inner(x1)
    assert result1.shape == (2, 4)
    assert result1.device.type == "sky"

    # Second call with same shapes: fire-and-forget (cache hit)
    x2 = torch.randn(2, 8, device=device)
    result2 = model.inner(x2)
    assert result2.shape == (2, 4)
    assert result2.device.type == "sky"

    # Third call: verify result is correct by transferring to CPU
    result2_cpu = result2.cpu()
    assert torch.isfinite(result2_cpu).all()

    # Results should be different (different inputs)
    result1_cpu = result1.cpu()
    assert not torch.equal(result1_cpu, result2_cpu)


@pytest.mark.it
@pytest.mark.asyncio
async def test_triton_modules_without_retain_model_fails(compute):
    """Using triton_modules without retain_model should raise an error."""
    state_dict = await compute.execute(_create_outer_model)

    with torch.device("meta"):
        model = OuterModel()

    with pytest.raises(RuntimeError, match="retain_model=True"):
        state_dict.load_into(model, triton_modules=["inner"])
