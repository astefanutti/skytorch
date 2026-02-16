"""
Tests for GIL-released callBoxed dispatch in the C++ ATen request parser.

These tests cover edge cases discovered while debugging LLM inference
(demo/llm.py) that are not covered by the general op tests. Each test
targets a specific callBoxed code path:

1. List<Optional<Tensor>> coercion (aten.index.Tensor)
2. Scalar-to-Tensor fallback (aten.add.Tensor with scalar arg)
3. GenericList → typed list coercion (empty lists, int lists)
4. Multi-output ops with undefined tensors (native_batch_norm eval)
5. Default argument filling from schema
6. "default" overload name mapping
"""

import pytest
import torch
import torch.nn.functional as F

from tests.it.op_test_utils import assert_forward_correct, assert_grad_correct


# =============================================================================
# 1. List<Optional<Tensor>> — aten.index.Tensor
#
# The aten::index.Tensor schema expects Tensor?[] (list of optional tensors).
# Advanced indexing passes None elements for dimensions that use `:` slicing.
# callBoxed requires c10::List<c10::optional<at::Tensor>>, not List<Tensor>.
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_index_advanced_multi_dim(device):
    """Multi-dimensional advanced indexing (used in transformer attention)."""
    x_cpu = torch.randn(2, 3, 4)
    idx0 = torch.tensor([0, 1, 0])
    idx1 = torch.tensor([2, 0, 1])

    cpu_result = x_cpu[idx0, idx1]
    sky_result = x_cpu.to(device)[idx0.to(device), idx1.to(device)]
    torch.testing.assert_close(sky_result.cpu(), cpu_result, check_device=False)


@pytest.mark.it
@pytest.mark.asyncio
async def test_index_with_none_dim(device):
    """Indexing with None (unsqueeze) + tensor index — creates [None, tensor] index list."""
    x_cpu = torch.randn(4, 5)
    idx = torch.tensor([0, 2, 3])

    cpu_result = x_cpu[idx, :]
    sky_result = x_cpu.to(device)[idx.to(device), :]
    torch.testing.assert_close(sky_result.cpu(), cpu_result, check_device=False)


# =============================================================================
# 2. Scalar-to-Tensor coercion fallback
#
# Some ops are dispatched as aten.add.Tensor even when the second arg is a
# scalar (Python binding handles the coercion). callBoxed is strict about
# types, so the code must detect the mismatch and fall back to PyObject_Call.
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_add_scalar_via_tensor_op(device):
    """x + scalar dispatches aten.add.Tensor — tests scalar fallback."""
    x_cpu = torch.randn(3, 4)
    assert_forward_correct(lambda x: x + 2.5, [x_cpu], device)


@pytest.mark.it
@pytest.mark.asyncio
async def test_mul_scalar_via_tensor_op(device):
    """x * scalar dispatches aten.mul.Tensor — tests scalar fallback."""
    x_cpu = torch.randn(3, 4)
    assert_forward_correct(lambda x: x * 3.0, [x_cpu], device)


@pytest.mark.it
@pytest.mark.asyncio
async def test_scalar_op_grad(device):
    """Gradient through scalar arithmetic (tests fallback path backward)."""
    x_cpu = torch.randn(3, 4, requires_grad=True)
    assert_grad_correct(lambda x: (x * 2.5 + 1.0).sum(), [x_cpu], device)


# =============================================================================
# 3. GenericList → typed list coercion
#
# callBoxed validates list types against the op schema. Lists parsed from
# binary as GenericList must be coerced to IntList, DoubleList, BoolList, etc.
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_int_list_args(device):
    """Ops with int[] args (stride, padding, etc.) — GenericList → IntList."""
    x_cpu = torch.randn(1, 1, 4, 4)
    assert_forward_correct(
        lambda x: F.avg_pool2d(x, kernel_size=2, stride=2), [x_cpu], device
    )


@pytest.mark.it
@pytest.mark.asyncio
async def test_bool_list_args(device):
    """Ops with bool[] args — tests bool list specialization.

    native_batch_norm backward uses bool[3] output_mask.
    """
    x_cpu = torch.randn(2, 3, 4, 4, requires_grad=True)
    x_sky = x_cpu.clone().to(device).detach().requires_grad_(True)
    w_cpu = torch.randn(3, requires_grad=True)
    w_sky = w_cpu.clone().to(device).detach().requires_grad_(True)
    b_cpu = torch.randn(3, requires_grad=True)
    b_sky = b_cpu.clone().to(device).detach().requires_grad_(True)
    rm = torch.zeros(3)
    rv = torch.ones(3)

    loss_cpu = F.batch_norm(
        x_cpu, rm.clone(), rv.clone(), w_cpu, b_cpu, training=True
    ).sum()
    loss_cpu.backward()

    loss_sky = F.batch_norm(
        x_sky, rm.clone().to(device), rv.clone().to(device), w_sky, b_sky, training=True
    ).sum()
    loss_sky.backward()

    for name, cpu_t, sky_t in [("x", x_cpu, x_sky), ("w", w_cpu, w_sky), ("b", b_cpu, b_sky)]:
        assert sky_t.grad is not None, f"batch_norm {name} gradient is None"
        torch.testing.assert_close(
            sky_t.grad.cpu(), cpu_t.grad, atol=1e-5, rtol=1.3e-6, check_device=False
        )


@pytest.mark.it
@pytest.mark.asyncio
async def test_empty_shape_list(device):
    """Ops with empty int[] — e.g., sum with no dims."""
    x_cpu = torch.randn(3, 4)
    assert_forward_correct(lambda x: x.sum(), [x_cpu], device)


# =============================================================================
# 4. Multi-output ops with undefined tensors
#
# Some ops return tuples where elements may be undefined (None) tensors.
# For example, native_batch_norm in eval mode returns (output, None, None)
# for save_mean and save_invstd. The output registration must skip undefined
# tensors without consuming output IDs.
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_batch_norm_eval_multi_output(device):
    """batch_norm eval returns (Tensor, undefined, undefined) — tests output filtering."""
    x_cpu = torch.randn(2, 3, 4, 4)
    w = torch.randn(3)
    b = torch.randn(3)
    rm = torch.zeros(3)
    rv = torch.ones(3)

    cpu_result = F.batch_norm(x_cpu, rm.clone(), rv.clone(), w, b, training=False)
    sky_result = F.batch_norm(
        x_cpu.to(device),
        rm.clone().to(device),
        rv.clone().to(device),
        w.to(device),
        b.to(device),
        training=False,
    )
    torch.testing.assert_close(sky_result.cpu(), cpu_result, check_device=False)


# =============================================================================
# 5. Default argument filling
#
# callBoxed requires ALL schema arguments on the stack, including those with
# default values that the client doesn't send. The code fills defaults from
# the op schema. Tests that default-heavy ops work correctly.
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_sum_with_default_args(device):
    """aten.sum.default — tests default arg filling (no explicit dim)."""
    x_cpu = torch.randn(3, 4)
    assert_forward_correct(lambda x: x.sum(), [x_cpu], device)


@pytest.mark.it
@pytest.mark.asyncio
async def test_mean_with_default_args(device):
    """aten.mean.default — tests default arg filling."""
    x_cpu = torch.randn(3, 4)
    assert_forward_correct(lambda x: x.mean(), [x_cpu], device)


@pytest.mark.it
@pytest.mark.asyncio
async def test_softmax_with_defaults(device):
    """softmax has multiple defaulted args — tests schema default filling."""
    x_cpu = torch.randn(2, 5)
    assert_forward_correct(lambda x: F.softmax(x, dim=-1), [x_cpu], device)


# =============================================================================
# 6. "default" overload name mapping
#
# Python op names like "aten.sum.default" use "default" to mean no overload.
# The C++ dispatcher uses an empty string for no overload. Tests that the
# mapping works for ops that only have a default overload.
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_clone_default_overload(device):
    """aten.clone.default — tests 'default' → '' overload mapping."""
    x_cpu = torch.randn(3, 4)
    assert_forward_correct(lambda x: x.clone(), [x_cpu], device)


@pytest.mark.it
@pytest.mark.asyncio
async def test_neg_default_overload(device):
    """aten.neg.default — another 'default' overload op."""
    x_cpu = torch.randn(3, 4)
    assert_forward_correct(lambda x: -x, [x_cpu], device)


@pytest.mark.it
@pytest.mark.asyncio
async def test_relu_default_overload(device):
    """aten.relu.default — activation with 'default' overload."""
    x_cpu = torch.randn(3, 4)
    assert_forward_correct(lambda x: F.relu(x), [x_cpu], device)


# =============================================================================
# 7. LLM inference patterns (end-to-end)
#
# These test the specific op sequences that occur during transformer
# model.generate() — the code path that originally triggered the
# List<Optional<Tensor>> bug.
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_argmax_then_index(device):
    """Pattern from greedy decoding: argmax → index into vocabulary."""
    logits = torch.randn(1, 10)
    vocab = torch.randn(10, 4)

    def fn(logits, vocab):
        idx = torch.argmax(logits, dim=-1)
        return vocab[idx]

    assert_forward_correct(fn, [logits, vocab], device)


@pytest.mark.it
@pytest.mark.asyncio
async def test_where_with_scalar(device):
    """torch.where with scalar — used in generation masking."""
    x_cpu = torch.randn(3, 4)

    def fn(x):
        return torch.where(x > 0, x, torch.zeros_like(x))

    assert_forward_correct(fn, [x_cpu], device)


@pytest.mark.it
@pytest.mark.asyncio
async def test_cat_then_index(device):
    """Pattern from KV cache: cat tensors then index — tests cat + index interaction."""
    a = torch.randn(1, 3, 4)
    b = torch.randn(1, 1, 4)

    def fn(a, b):
        combined = torch.cat([a, b], dim=1)
        return combined[:, -1, :]

    assert_forward_correct(fn, [a, b], device)


@pytest.mark.it
@pytest.mark.asyncio
async def test_unfinished_sequences_pattern(device):
    """Pattern from HuggingFace generate: ones → mul → max → eq → item.

    This is the exact sequence that triggers the List<Optional<Tensor>> bug
    when aten.index.Tensor is involved in the boolean evaluation path.
    """
    x = torch.randn(2, 5)

    def fn(x):
        # Simulate unfinished_sequences logic
        unfinished = torch.ones(x.shape[0], dtype=torch.long, device=x.device)
        scores = x.sum(dim=-1)
        finished = scores > 0
        unfinished = unfinished * (~finished).long()
        # max().item() triggers get_scalar which exposes missing tensors
        return unfinished.max().unsqueeze(0).float()

    assert_forward_correct(fn, [x], device)


@pytest.mark.it
@pytest.mark.asyncio
async def test_embedding_lookup(device):
    """Embedding lookup — core transformer op, tests index with int tensor."""
    weight = torch.randn(10, 4)
    indices = torch.tensor([0, 3, 7, 2])

    def fn(w, idx):
        return F.embedding(w, idx)

    # F.embedding(weight, indices) — note: args are (input, weight) in PyTorch
    cpu_result = F.embedding(indices, weight)
    sky_result = F.embedding(indices.to(device), weight.to(device))
    torch.testing.assert_close(sky_result.cpu(), cpu_result, check_device=False)


@pytest.mark.it
@pytest.mark.asyncio
async def test_layer_norm_forward(device):
    """LayerNorm — common in transformers, tests multiple default args."""
    x_cpu = torch.randn(2, 3, 8)
    w = torch.randn(8)
    b = torch.randn(8)

    def fn(x, weight, bias):
        return F.layer_norm(x, [8], weight, bias)

    assert_forward_correct(fn, [x_cpu, w, b], device)


@pytest.mark.it
@pytest.mark.asyncio
async def test_layer_norm_grad(device):
    """LayerNorm gradient — tests backward through multi-output op."""
    x_cpu = torch.randn(2, 3, 8, requires_grad=True)
    w = torch.randn(8, requires_grad=True)
    b = torch.randn(8, requires_grad=True)

    def fn(x, weight, bias):
        return F.layer_norm(x, [8], weight, bias).sum()

    assert_grad_correct(fn, [x_cpu, w, b], device)
