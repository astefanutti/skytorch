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


class SharedStorageModule(nn.Module):
    """Module that returns two tensors sharing the same GPU storage."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 8)

    def forward(self, x):
        combined = self.linear(x)  # (batch, 8)
        return combined[:, :4], combined[:, 4:]


class SharedStorageWrapper(nn.Module):
    """Wrapper for SharedStorageModule."""

    def __init__(self):
        super().__init__()
        self.shared = SharedStorageModule()

    def forward(self, x):
        a, b = self.shared(x)
        return a + b


class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Linear(dim, dim * 2)
        self.down = nn.Linear(dim * 2, dim)

    def forward(self, x):
        return self.down(torch.relu(self.up(x)))


class MultiOutputMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Linear(dim, dim * 2)
        self.down = nn.Linear(dim * 2, dim)

    def forward(self, x):
        h = self.down(torch.relu(self.up(x)))
        return h, h.sum().unsqueeze(0)


class DecoderLayerModel(nn.Module):
    def __init__(self, num_layers=4, dim=8):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "attn": nn.Linear(dim, dim),
                        "mlp": MLP(dim),
                    }
                )
            )

    def forward(self, x):
        for layer in self.layers:
            residual = x
            x = layer["attn"](x)
            x = x + residual
            residual = x
            x = layer["mlp"](x)
            x = x + residual
        return x


class MoEDecoderModel(nn.Module):
    def __init__(self, num_layers=4, dim=8):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "attn": nn.Linear(dim, dim),
                        "mlp": MultiOutputMLP(dim),
                    }
                )
            )

    def forward(self, x):
        for layer in self.layers:
            residual = x
            x = layer["attn"](x)
            x = x + residual
            residual = x
            x, _ = layer["mlp"](x)  # discard auxiliary output
            x = x + residual
        return x


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


def _create_shared_storage_model():
    torch.manual_seed(42)
    return SharedStorageWrapper()


def _create_decoder_layer_model():
    torch.manual_seed(42)
    return DecoderLayerModel(num_layers=4, dim=8)


def _create_moe_decoder_model():
    torch.manual_seed(42)
    return MoEDecoderModel(num_layers=4, dim=8)


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


@pytest.mark.it
@pytest.mark.asyncio
async def test_multi_output_discard_one(compute):
    """Discarding one output of a multi-output module must not corrupt the other.

    Covers the id(object()) storage_id collision fix: when one output is deleted,
    the kept output must remain usable in subsequent ATen ops.
    """
    state_dict = await compute.execute(_create_wrapper_model, retain_model=True)

    with torch.device("meta"):
        model = WrapperModel()
    state_dict.load_into(model, triton_modules=["multi"])

    device = compute.device("cpu")
    x = torch.randn(2, 4, device=device)

    a, b = model.multi(x)
    del b

    # The kept output must survive deletion of the other and work in ATen ops
    result = a.sum()
    result_cpu = result.cpu()
    assert torch.isfinite(result_cpu).all()


@pytest.mark.it
@pytest.mark.asyncio
async def test_multi_output_shared_storage(compute):
    """Outputs sharing GPU storage must both be registered and transferable.

    Covers the _pending_tensors list fix and the as_strided registration ordering.
    """
    state_dict = await compute.execute(_create_shared_storage_model, retain_model=True)

    with torch.device("meta"):
        model = SharedStorageWrapper()
    state_dict.load_into(model, triton_modules=["shared"])

    device = compute.device("cpu")
    x = torch.randn(2, 4, device=device)

    left, right = model.shared(x)
    assert left.shape == (2, 4)
    assert right.shape == (2, 4)

    left_cpu = left.cpu()
    right_cpu = right.cpu()
    assert torch.isfinite(left_cpu).all()
    assert torch.isfinite(right_cpu).all()

    # The two halves should be numerically different
    assert not torch.equal(left_cpu, right_cpu)


@pytest.mark.it
@pytest.mark.asyncio
async def test_multi_output_cached_lifecycle(compute):
    """Fire-and-forget (cache hit) path must handle output lifecycle correctly.

    Covers the _storage_sentinels persistence fix across repeated calls where
    some outputs are discarded.
    """
    state_dict = await compute.execute(_create_wrapper_model, retain_model=True)

    with torch.device("meta"):
        model = WrapperModel()
    state_dict.load_into(model, triton_modules=["multi"])

    device = compute.device("cpu")

    # Call 1 (cache miss): keep both outputs
    x1 = torch.randn(2, 4, device=device)
    a1, b1 = model.multi(x1)

    # Call 2 (cache hit): discard second output
    x2 = torch.randn(2, 4, device=device)
    a2, _ = model.multi(x2)

    # Call 3 (cache hit): keep both outputs again
    x3 = torch.randn(2, 4, device=device)
    a3, b3 = model.multi(x3)

    # All kept outputs must transfer to CPU as finite values
    a2_cpu = a2.cpu()
    assert torch.isfinite(a2_cpu).all()

    a3_cpu = a3.cpu()
    assert torch.isfinite(a3_cpu).all()

    b3_cpu = b3.cpu()
    assert torch.isfinite(b3_cpu).all()

    # Clean up call-1 outputs (verifies they survived through later calls)
    del a1, b1


@pytest.mark.it
@pytest.mark.asyncio
async def test_sequential_proxied_layers_with_aten_ops(compute):
    """Multi-layer chaining with proxied modules interleaved with ATen ops.

    Covers the stream deferred-delete ordering fix: intermediate tensors go out
    of scope between layers, and GC-triggered deletes must not corrupt the stream.
    """
    state_dict = await compute.execute(_create_decoder_layer_model, retain_model=True)

    with torch.device("meta"):
        model = DecoderLayerModel(num_layers=4, dim=8)
    state_dict.load_into(model, triton_modules=["layers.*.mlp"])

    device = compute.device("cpu")
    x = torch.randn(2, 8, device=device)

    # First pass (cache miss, sync): establishes shape cache
    result1 = model(x)
    assert result1.shape == (2, 8)

    result1_cpu = result1.cpu()
    assert torch.isfinite(result1_cpu).all()

    # Second pass (cache hit, fire-and-forget): exercises deferred-delete ordering
    x2 = torch.randn(2, 8, device=device)
    result2 = model(x2)
    assert result2.shape == (2, 8)
    assert result2.device.type == "sky"

    result2_cpu = result2.cpu()
    assert torch.isfinite(result2_cpu).all()


@pytest.mark.it
@pytest.mark.asyncio
async def test_moe_multi_pass_discard(compute):
    """Multi-pass MoE model where discarded auxiliary outputs trigger deferred deletes.

    Regression test for the flush ordering bug: when multi-output proxied modules
    discard one output (like MoE router scores), GC-triggered deletes of those
    intermediate tensors were flushed to the server before the ATen ops that still
    referenced them, causing 'Tensor X does not exist' errors on cache-hit passes.
    """
    state_dict = await compute.execute(_create_moe_decoder_model, retain_model=True)

    with torch.device("meta"):
        model = MoEDecoderModel(num_layers=4, dim=8)
    state_dict.load_into(model, triton_modules=["layers.*.mlp"])

    device = compute.device("cpu")

    # First pass (cache miss, sync): establishes shape cache
    x1 = torch.randn(2, 8, device=device)
    result1 = model(x1)
    assert result1.shape == (2, 8)

    result1_cpu = result1.cpu()
    assert torch.isfinite(result1_cpu).all()

    # Second pass (cache hit, fire-and-forget): triggers the bug
    # - FF module forward requests enqueue via _enqueue_with_flush
    # - Discarded auxiliary outputs generate deferred deletes
    # - ATen ops between layers buffer in _raw_batch_buffer
    # Before fix: deletes flushed before raw batch -> "Tensor X does not exist"
    x2 = torch.randn(2, 8, device=device)
    result2 = model(x2)
    assert result2.shape == (2, 8)
    assert result2.device.type == "sky"

    result2_cpu = result2.cpu()
    assert torch.isfinite(result2_cpu).all()
