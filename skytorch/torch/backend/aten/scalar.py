"""
SkyTorch ATen Scalar Operations.

This module implements scalar operations that require fetching
values from sky tensors to the local host.
"""

import os
import time

import torch

from skytorch.torch.backend._async import run_async
from skytorch.torch.backend._client import (
    ENABLE_STREAMING,
    _get_tensor_metadata_if_new,
    _register_tensor_locally,
    _require_compute,
    get_scalar,
)
from skytorch.torch.client.request import tensor_metadata_to_proto
from skytorch.torch.client.tensor import get_tensor_id
from skytorch.torch.profiler import PROFILING_ENABLED

try:
    from skytorch.torch.backend._C import (
        _get_ops_counter,
        _increment_ops_counter,
        _reset_ops_counter,
    )
except ImportError:
    _get_ops_counter = None
    _increment_ops_counter = None
    _reset_ops_counter = None

# Scalar speculation: predict the next .item() result based on the last-seen value.
# This eliminates blocking at sync points for predictable scalars (e.g., stopping criteria
# in model.generate() where __bool__() on a bool tensor returns False every token).
#
# Safety constraints:
# - Only applied to bool-dtype tensors (control flow checks, not metrics/loss)
# - Only activates when there's significant computation between .item() calls
#   (>= _MIN_OPS_BETWEEN_SYNCS fire-and-forget ops), distinguishing generate() loops
#   (200+ ops/token) from gradcheck (few ops between allclose checks)
# - Incidental syncs (0 ops between) with matching values confirm immediately,
#   so speculation kicks in from token 1 in generate() loops
# - Qualifying syncs always validate the previous prediction (mandatory back-pressure)
#   to prevent the client from racing ahead of the server and overwhelming the stream
# - Non-qualifying confirmed syncs skip get_scalar (no server round-trip) when the
#   tensor is already registered, returning the predicted value immediately
# - On misprediction: falls back to blocking, generates at most one extra token
_SPECULATION_ENABLED = os.environ.get("SKYTORCH_SPECULATIVE_SCALAR", "1") == "1"

# Minimum fire-and-forget ops between .item() calls to consider speculation.
# generate() dispatches 200+ ops per token for the main forward pass, plus
# 25-40 ops between the stopping criterion and secondary cache checks.
# gradcheck dispatches ~10-20 ops between allclose checks. A threshold of 25
# separates the two patterns while keeping secondary generate checks above
# the threshold (preventing unwanted speculation reset).
_MIN_OPS_BETWEEN_SYNCS = 25

# Counter incremented by the dispatch path for each fire-and-forget op.
# Read and reset by _local_scalar_dense to detect the generate() pattern.
_ops_since_last_sync: int = 0

# Speculation state machine:
#   None → first call, no history
#   {"value": v, "future": f|None, "confirmed": bool}
#     confirmed=False: seeded, waiting for a matching value to confirm
#     confirmed=True:  pattern confirmed, actively speculating
_speculation_state: dict | None = None

# Flag to bypass speculation for specific call sites. Set by _equal to avoid
# returning speculated values for equality checks — the result depends on
# the specific tensors being compared, not on a repeating pattern.
_bypass_speculation: bool = False


def _reset_speculation() -> None:
    """Reset speculation state. Called on device reset / stream close."""
    global _speculation_state, _ops_since_last_sync
    _speculation_state = None
    _ops_since_last_sync = 0
    if _reset_ops_counter is not None:
        _reset_ops_counter()


def _local_scalar_dense(self: torch.Tensor):
    """Get the scalar value from a single-element sky tensor.

    This operation fetches the scalar value from the remote server.
    When streaming is enabled, uses the dedicated GetScalar RPC which
    avoids full tensor serialization overhead.

    When speculation is enabled, returns a predicted value immediately
    and validates the prediction asynchronously. This eliminates blocking
    at sync points for predictable scalars (e.g., stopping criteria in
    model.generate() where .item() returns 1 for all tokens except the last).

    Args:
        self: A sky tensor with exactly one element

    Returns:
        Python scalar value (int, float, bool, etc.)

    Raises:
        RuntimeError: If tensor has more than one element
    """
    global _speculation_state, _ops_since_last_sync

    if self.numel() != 1:
        raise RuntimeError(
            f"a Tensor with {self.numel()} elements cannot be converted to Scalar"
        )

    if PROFILING_ENABLED:
        _t0 = time.perf_counter_ns()

    if ENABLE_STREAMING:
        # Speculation: only for bool tensors with significant computation between calls.
        # This targets generate() loops (200+ ops/token) and excludes gradcheck
        # (which calls allclose → bool .item() with few ops in between).
        #
        # HuggingFace generate() produces TWO bool .item() calls per token:
        # 1. Stopping criterion: unfinished_sequences.max() == 0 (~200 ops since last)
        # 2. Cache check: cache_position[-1] >= input_ids.shape[1] (few ops since last)
        # #1 is a "qualifying" sync that seeds/confirms/speculates with back-pressure.
        # #2 is a non-qualifying sync that returns the predicted value without any
        # server round-trip (the tensor is already registered by the dispatch path).
        if _get_ops_counter is not None:
            _ops_between = _get_ops_counter()
        else:
            _ops_between = _ops_since_last_sync
        _can_speculate = (
            _SPECULATION_ENABLED
            and self.dtype == torch.bool
            and _ops_between >= _MIN_OPS_BETWEEN_SYNCS
        )
        # Only reset counter on qualifying syncs — non-qualifying ones are transparent
        if _can_speculate:
            if _reset_ops_counter is not None:
                _reset_ops_counter()
            else:
                _ops_since_last_sync = 0

        # Non-qualifying confirmed sync: return predicted value without a server
        # round-trip. The tensor was already registered by the dispatch path that
        # created it, so no metadata needs to be sent. This check must come BEFORE
        # submitting get_scalar to avoid overwhelming the gRPC stream.
        if (
            not _can_speculate
            and _speculation_state is not None
            and _speculation_state["confirmed"]
            and self.dtype == torch.bool
            and not _bypass_speculation
            and _get_tensor_metadata_if_new(self) is None
        ):
            if PROFILING_ENABLED:
                from skytorch.torch.profiler import ClientProfiler

                _prof = ClientProfiler.get()
                _prof.scalar_speculative_hits += 1
                _t1 = time.perf_counter_ns()
                _prof.sync_total.add(_t1 - _t0)
            return _speculation_state["value"]

        compute = _require_compute(self)
        tensor_id = get_tensor_id(self)

        meta = _get_tensor_metadata_if_new(self)
        metadata_proto = None
        if meta is not None:
            metadata_proto = tensor_metadata_to_proto(meta)

        # Submit async scalar request
        future = run_async(get_scalar(compute, tensor_id, metadata_proto))

        if meta is not None:
            _register_tensor_locally(self)

        if _can_speculate and _speculation_state is not None:
            state = _speculation_state

            if state["confirmed"]:
                # Pattern confirmed — actively speculating
                predicted = state["value"]
                prev_future = state["future"]

                # Mandatory validation: always block on prev_future to provide
                # back-pressure. Without this, the client races ahead of the
                # server, overwhelming the gRPC stream and causing deadlocks
                # (especially on remote GPUs or cold-server first runs).
                # On warm servers, prev_future is typically already done by this
                # point (the server had a full forward pass to process it), so
                # this block is essentially free.
                if prev_future is not None:
                    actual = prev_future.result()
                    if actual != predicted:
                        # Pattern broke — stop speculating, block on current
                        _speculation_state = None
                        if PROFILING_ENABLED:
                            from skytorch.torch.profiler import ClientProfiler

                            ClientProfiler.get().scalar_speculative_misses += 1
                        result = future.result()
                        if PROFILING_ENABLED:
                            _t1 = time.perf_counter_ns()
                            ClientProfiler.get().sync_total.add(_t1 - _t0)
                        return result

                # Speculate: return predicted, save future for next validation
                state["future"] = future
                if PROFILING_ENABLED:
                    from skytorch.torch.profiler import ClientProfiler

                    _prof = ClientProfiler.get()
                    _prof.scalar_speculative_hits += 1
                    _t1 = time.perf_counter_ns()
                    _prof.sync_total.add(_t1 - _t0)
                return predicted

            else:
                # Not confirmed yet — block and check if value repeats
                result = future.result()
                if result == state["value"]:
                    # Two consecutive identical values → confirm pattern
                    state["confirmed"] = True
                    state["future"] = None
                else:
                    # Different value — update seed, stay unconfirmed
                    state["value"] = result

                if PROFILING_ENABLED:
                    from skytorch.torch.profiler import ClientProfiler

                    _t1 = time.perf_counter_ns()
                    ClientProfiler.get().sync_total.add(_t1 - _t0)
                return result

        # No speculation path: first call, non-bool tensor, few ops, or disabled.
        # Don't reset unconfirmed state — let it survive to the next qualifying
        # sync for confirmation. Resetting would prevent confirmation from ever
        # succeeding when the secondary generate check doesn't qualify.

        result = future.result()

        if _can_speculate and _speculation_state is None:
            # Seed speculation state (unconfirmed)
            _speculation_state = {"value": result, "future": None, "confirmed": False}
        elif (
            not _can_speculate
            and _speculation_state is not None
            and not _speculation_state["confirmed"]
            and self.dtype == torch.bool
            and _ops_between == 0
            and result == _speculation_state["value"]
        ):
            # Incidental sync (0 ops) with matching value confirms immediately.
            # This targets the generate() pattern where a secondary bool check
            # follows the stopping criterion with zero ops in between. Confirms
            # speculation at token 1 instead of waiting for token 2, eliminating
            # one extra blocking .item() call — critical for the first run after
            # server start when each blocking call is expensive (server backlog).
            # The _ops_between == 0 guard prevents early confirmation during
            # gradcheck (where allclose checks have 10-20+ ops between them).
            _speculation_state["confirmed"] = True

        if PROFILING_ENABLED:
            from skytorch.torch.profiler import ClientProfiler

            _t1 = time.perf_counter_ns()
            ClientProfiler.get().sync_total.add(_t1 - _t0)

        return result
    else:
        from .copy import _copy_from_device

        cpu_tensor = _copy_from_device(self)
        return cpu_tensor.item()


def _equal(self: torch.Tensor, other: torch.Tensor) -> bool:
    """Compare two sky tensors for equality.

    Performs element-wise comparison on the sky device, then reduces
    to a single boolean result.

    Args:
        self: First sky tensor
        other: Second sky tensor

    Returns:
        True if all elements are equal, False otherwise
    """
    global _bypass_speculation

    # Check basic compatibility
    if self.shape != other.shape:
        return False
    if self.dtype != other.dtype:
        return False

    # Perform element-wise comparison on sky device
    eq_tensor = torch.eq(self, other)

    # Reduce to single boolean
    all_equal_tensor = torch.all(eq_tensor)

    # Bypass speculation — equality results depend on the specific tensors,
    # not on a repeating pattern like generate()'s stopping criterion.
    _bypass_speculation = True
    try:
        return all_equal_tensor.item()
    finally:
        _bypass_speculation = False
