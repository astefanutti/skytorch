"""Tests for dispatch ordering correctness at sync points.

These tests exercise the interaction between fire-and-forget ATen operations
and synchronous operations like .item() and .cpu(). The key invariant is that
all prior fire-and-forget ops must complete successfully before a sync point
returns.

Root cause of the MNIST "Tensor not found" bug:
    When a CPU tensor larger than STREAM_CHUNK_SIZE (1 MB) is transferred to
    the sky device via .to(device), the data is sent as a chunked
    UpdateTensor request. The old code submitted these chunks via a
    call_soon_threadsafe callback that flushed _mt_ops first — sending
    execute_aten operations that reference the uploaded tensor BEFORE the
    chunks that create it on the server. The fix routes chunks through
    _mt_ops to preserve FIFO ordering.

Additional ordering bugs fixed:
    - _drain_cpp_raw race: C++ fast-path ops went through a separate event
      loop callback (_drain_cpp_raw) that drained directly into
      _raw_batch_buffer. A stale callback could drain ops submitted AFTER
      an _mt_ops "req" entry (update_tensor, delete_tensors), then
      _enqueue_with_flush would flush those ops ahead of the "req".
      Fix: _drain_cpp_raw now delegates to _drain_mt_ops.
    - Delete ordering: submit_delete_tensors bypassed _mt_ops via
      call_soon_threadsafe, so GC-triggered deletes could reach the server
      before the ATen op that creates the tensor. The server stored the
      error as _deferred_error, surfaced on the next get_scalar.
      Fix: deletes now route through _mt_ops via _submit_request.

The race conditions require a separate server process (network latency
allows the event loop to interleave with the main thread); the in-process
test server is too fast to trigger the race deterministically, but the
tests guard the code paths against regression.
"""

import gc

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Minimum number of float32 elements to exceed the 1 MB stream chunk size
# and trigger the chunked upload path in _submit_chunked_update_tensor.
# 1 MB / 4 bytes = 262_144; use a round number above that.
_CHUNK_THRESHOLD_NUMEL = 300_000


# =============================================================================
# Chunked tensor upload ordering (root cause)
#
# These tests transfer CPU tensors >1 MB to the sky device, which triggers
# _submit_chunked_update_tensor. Before the fix, the chunks were enqueued
# AFTER execute_aten ops that reference them, causing "Tensor not found".
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_large_tensor_transfer_then_operation(device):
    """Transfer a >1 MB tensor to sky device, then immediately use it.

    This is the minimal reproduction of the MNIST bug. The large tensor
    triggers _submit_chunked_update_tensor. The subsequent add operation
    dispatches an execute_aten that references the uploaded tensor. If
    chunks are not ordered before the execute_aten in the stream, the
    server receives the operation before it has the tensor.
    """
    x = torch.randn(_CHUNK_THRESHOLD_NUMEL).to(device)
    y = x + 1.0
    result = y.sum().item()
    assert isinstance(result, float)


@pytest.mark.it
@pytest.mark.asyncio
async def test_large_tensor_transfer_then_matmul(device):
    """Transfer a large matrix and immediately multiply it.

    Matrix multiply (mm) references both the uploaded tensor and creates
    a new output. If the upload chunks arrive after the mm request, the
    server reports "Tensor not found" for the input.
    """
    a = torch.randn(600, 600).to(device)
    b = torch.randn(600, 600).to(device)
    c = a @ b
    result = c[0, 0].item()
    assert isinstance(result, float)


@pytest.mark.it
@pytest.mark.asyncio
async def test_multiple_large_transfers_then_operations(device):
    """Transfer multiple large tensors, then use them together.

    This matches the MNIST pattern where both data and target are
    transferred before the forward pass. Multiple chunked uploads
    followed by operations referencing all of them.
    """
    data = torch.randn(500, 600).to(device)
    weights = torch.randn(600, 10).to(device)
    output = data @ weights
    loss = output.sum()
    result = loss.item()
    assert isinstance(result, float)


@pytest.mark.it
@pytest.mark.asyncio
async def test_large_transfer_repeated_use(device):
    """Transfer a large tensor and use it in a sequence of operations.

    After the chunked upload, several operations reference the tensor.
    All must arrive at the server after the chunks.
    """
    x = torch.randn(_CHUNK_THRESHOLD_NUMEL).to(device)
    y = x * 2
    z = y + x
    w = z - 1
    result = w.sum().item()
    assert isinstance(result, float)


@pytest.mark.it
@pytest.mark.asyncio
async def test_large_transfer_in_loop_with_sync(device):
    """Repeatedly transfer large tensors and sync, mimicking data loading.

    Each iteration transfers a new batch (>1 MB), runs operations, and
    syncs with .item(). This is the exact MNIST training loop pattern
    where data.to(device) is called every iteration.
    """
    model = nn.Linear(600, 10).to(device)
    model.eval()

    for _ in range(5):
        x = torch.randn(500, 600).to(device)
        with torch.no_grad():
            output = model(x)
        val = output.sum().item()
        assert isinstance(val, float)


@pytest.mark.it
@pytest.mark.asyncio
async def test_mnist_data_loading_pattern(device):
    """Reproduce the exact MNIST eval pattern with chunked transfers.

    Transfers large data batches, runs a CNN forward pass, accumulates
    predictions, and syncs with .item(). This is the pattern that
    triggered the original bug.
    """
    torch.manual_seed(42)
    num_classes = 4

    model = nn.Sequential(
        nn.Linear(600, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes),
    ).to(device)
    model.eval()

    correct = torch.tensor(0, device=device)
    total = torch.tensor(0, device=device)

    with torch.no_grad():
        for _ in range(3):
            # >1 MB transfer triggers chunked upload
            x = torch.randn(500, 600).to(device)
            target = torch.randint(0, num_classes, (500,)).to(device)

            output = model(x)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum()
            total += 500

    accuracy = (correct.float() / total).item() * 100
    assert isinstance(accuracy, float)
    assert 0.0 <= accuracy <= 100.0


# =============================================================================
# CNN model for train/eval cycle tests
# =============================================================================


class SmallCNN(nn.Module):
    """Simplified CNN that generates many ATen ops per forward pass."""

    def __init__(self, num_classes=4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, 1)
        self.conv2 = nn.Conv2d(8, 16, 3, 1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16 * 5 * 5, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# =============================================================================
# MNIST-like train/eval cycle with CNN
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_cnn_train_then_eval_item(device):
    """Train CNN, then eval with accumulation + .item().

    Training fills dispatch caches. Eval uses different op patterns
    with accumulation (correct += ...) and a sync at the end.
    """
    torch.manual_seed(42)
    num_classes = 4
    model = SmallCNN(num_classes=num_classes).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    model.train()
    for _ in range(3):
        x = torch.randn(4, 1, 14, 14).to(device)
        target = torch.randint(0, num_classes, (4,)).to(device)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), target)
        loss.backward()
        optimizer.step()

    model.eval()
    correct = torch.tensor(0, device=device)
    total = torch.tensor(0, device=device)
    with torch.no_grad():
        for _ in range(3):
            x = torch.randn(4, 1, 14, 14).to(device)
            target = torch.randint(0, num_classes, (4,)).to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == target).sum()
            total += 4

    accuracy = (correct.float() / total).item() * 100
    assert isinstance(accuracy, float)
    assert 0.0 <= accuracy <= 100.0


@pytest.mark.it
@pytest.mark.asyncio
async def test_cnn_multi_epoch_with_eval(device):
    """Multiple train/eval epochs, each ending with .item()."""
    torch.manual_seed(42)
    num_classes = 4
    model = SmallCNN(num_classes=num_classes).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(3):
        model.train()
        for _ in range(3):
            x = torch.randn(4, 1, 14, 14).to(device)
            target = torch.randint(0, num_classes, (4,)).to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), target)
            loss.backward()
            optimizer.step()

        train_loss = loss.item()
        assert isinstance(train_loss, float), f"Epoch {epoch}: not float"

        model.eval()
        correct = torch.tensor(0, device=device)
        total = torch.tensor(0, device=device)
        with torch.no_grad():
            for _ in range(3):
                x = torch.randn(4, 1, 14, 14).to(device)
                target = torch.randint(0, num_classes, (4,)).to(device)
                pred = model(x).argmax(dim=1)
                correct += (pred == target).sum()
                total += 4

        accuracy = (correct.float() / total).item() * 100
        assert 0.0 <= accuracy <= 100.0, f"Epoch {epoch}: {accuracy} out of range"


# =============================================================================
# Other ordering scenarios
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_many_ops_exceed_batch_threshold(device):
    """Dispatch >64 fire-and-forget ops (batch flush threshold), then sync."""
    x = torch.randn(8, 8).to(device)
    for _ in range(120):
        x = x * 0.999 + 0.001
    result = x.sum().item()
    assert isinstance(result, float)
    assert result == result  # not NaN


@pytest.mark.it
@pytest.mark.asyncio
async def test_interleaved_compute_and_sync_at_scale(device):
    """Interleave large bursts of fire-and-forget ops with sync points."""
    x = torch.randn(8, 8).to(device)
    for _ in range(5):
        y = x
        for _ in range(80):
            y = y * 0.99 + 0.01
        val = y.sum().item()
        assert isinstance(val, float)


@pytest.mark.it
@pytest.mark.asyncio
async def test_training_loop_with_loss_item(device):
    """Multi-step training with .item() on loss every iteration."""
    torch.manual_seed(42)
    model = nn.Linear(8, 2).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for _ in range(20):
        x = torch.randn(4, 8).to(device)
        target = torch.randint(0, 2, (4,)).to(device)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), target)
        loss.backward()
        optimizer.step()
        loss_val = loss.item()
        assert isinstance(loss_val, float)
        assert loss_val == loss_val  # not NaN


# =============================================================================
# GC-triggered delete ordering
#
# When Python garbage-collects a sky tensor, submit_delete_tensors is called.
# If the delete bypasses _mt_ops (the old code used call_soon_threadsafe
# directly), it can reach the server before the ATen op that creates the
# tensor, causing a deferred "Tensor does not exist" error surfaced on the
# next sync point.
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_gc_during_training_loop(device):
    """Force GC every iteration to trigger delete ordering race.

    backward() releases autograd-saved intermediates. gc.collect() forces
    their __del__ → submit_delete_tensors. If deletes bypass _mt_ops, they
    can overtake the ATen ops that created those tensors.
    """
    torch.manual_seed(42)
    model = nn.Linear(16, 4).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for _ in range(30):
        x = torch.randn(8, 16).to(device)
        target = torch.randint(0, 4, (8,)).to(device)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), target)
        loss.backward()
        optimizer.step()
        gc.collect()
        loss_val = loss.item()
        assert isinstance(loss_val, float)
        assert loss_val == loss_val


@pytest.mark.it
@pytest.mark.asyncio
async def test_gc_during_cnn_training(device):
    """CNN training with forced GC creates many intermediate tensors.

    CNNs generate more intermediates (conv, batchnorm, relu, pool) than
    linear models. Forced GC between backward and .item() maximizes the
    chance of delete/create races.
    """
    torch.manual_seed(42)
    num_classes = 4
    model = SmallCNN(num_classes=num_classes).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for _ in range(10):
        x = torch.randn(4, 1, 14, 14).to(device)
        target = torch.randint(0, num_classes, (4,)).to(device)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), target)
        loss.backward()
        optimizer.step()
        gc.collect()
        loss_val = loss.item()
        assert isinstance(loss_val, float)


@pytest.mark.it
@pytest.mark.asyncio
async def test_rapid_tensor_create_delete_cycles(device):
    """Create and immediately drop sky tensors in a tight loop.

    Each iteration creates tensors via .to(device) and drops the previous
    ones (triggering delete). The rapid create/delete interleaving stresses
    the ordering between ATen ops (from .to() and arithmetic) and deletes.
    """
    prev = None
    for i in range(50):
        x = torch.randn(32, 32).to(device)
        y = x + float(i)
        prev = y  # drop previous y, triggering delete of old tensors
    result = prev.sum().item()
    assert isinstance(result, float)


@pytest.mark.it
@pytest.mark.asyncio
async def test_delete_between_ops_and_sync(device):
    """Delete tensors between fire-and-forget ops and the sync point.

    Creates intermediates, explicitly deletes them, then syncs.
    The delete must not corrupt the stream ordering.
    """
    x = torch.randn(16, 16).to(device)
    # Create intermediates
    a = x * 2
    b = a + 1
    c = b - 0.5
    result_tensor = c.sum()
    # Drop intermediates before syncing
    del a, b, c
    gc.collect()
    # Sync — must not see a deferred "Tensor does not exist"
    result = result_tensor.item()
    assert isinstance(result, float)


# =============================================================================
# _drain_cpp_raw ordering (C++ fast-path vs _mt_ops)
#
# The C++ fast path (dispatch_cached_aten cache hits) submits ops via
# cpp_submit_raw → C++ buffer → _drain_cpp_raw callback. If _drain_cpp_raw
# drains directly into _raw_batch_buffer instead of going through _mt_ops,
# a C++ op submitted after an update_tensor can be flushed before it.
#
# These tests exercise patterns where small .to(device) transfers (which
# go through _mt_ops as "req" entries) are immediately followed by many
# ATen ops (which go through the C++ fast path after cache warm-up).
# =============================================================================


@pytest.mark.it
@pytest.mark.asyncio
async def test_small_transfer_interleaved_with_ops(device):
    """Small tensor upload followed by ops, repeated many times.

    Small .to(device) uses submit_update_tensor → _submit_request → "req"
    in _mt_ops. Subsequent ATen ops go through the C++ fast path after
    cache warm-up. If _drain_cpp_raw puts C++ ops in _raw_batch_buffer
    while the "req" is still in _mt_ops, _enqueue_with_flush flushes them
    out of order.
    """
    model = nn.Linear(32, 8).to(device)
    model.eval()

    for _ in range(30):
        x = torch.randn(16, 32).to(device)
        with torch.no_grad():
            out = model(x)
        val = out.sum().item()
        assert isinstance(val, float)


@pytest.mark.it
@pytest.mark.asyncio
async def test_multi_tensor_upload_then_ops(device):
    """Upload multiple small tensors, then use all of them.

    Each .to(device) creates a "req" entry in _mt_ops. The subsequent
    ops reference all uploaded tensors. If any upload is reordered after
    an op that references it, the server errors.
    """
    for _ in range(20):
        a = torch.randn(8, 8).to(device)
        b = torch.randn(8, 8).to(device)
        c = torch.randn(8, 8).to(device)
        result = (a @ b + c).sum().item()
        assert isinstance(result, float)


@pytest.mark.it
@pytest.mark.asyncio
async def test_upload_ops_sync_tight_loop(device):
    """Tight loop: upload → single op → sync, many iterations.

    Minimal work between upload and sync maximizes the chance of the
    _drain_cpp_raw callback interleaving with _drain_mt_ops.
    """
    for _ in range(50):
        x = torch.randn(4, 4).to(device)
        val = (x + 1).sum().item()
        assert isinstance(val, float)


@pytest.mark.it
@pytest.mark.asyncio
async def test_training_with_gc_and_scheduler(device):
    """Full MNIST-like training pattern: upload, forward, backward, step, gc, sync.

    Combines all the fixed ordering scenarios:
    - Small tensor uploads ("req" in _mt_ops)
    - Many ATen ops (C++ fast path after cache warm-up)
    - Backward frees intermediates (GC-triggered deletes)
    - Scheduler step (additional ops)
    - .item() sync point
    """
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 4),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    for _ in range(30):
        x = torch.randn(16, 32).to(device)
        target = torch.randint(0, 4, (16,)).to(device)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        gc.collect()
        loss_val = loss.item()
        assert isinstance(loss_val, float)
        assert loss_val == loss_val
