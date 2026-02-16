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

# Scalar speculation: predict the next .item() result based on the last-seen value.
# This eliminates blocking at sync points for predictable scalars (e.g., stopping criteria
# in model.generate() where __bool__() on a bool tensor returns False every token).
#
# Safety constraints:
# - Only applied to bool-dtype tensors (control flow checks, not metrics/loss)
# - Requires two consecutive identical values before speculating (confirmation)
# - Only activates when there's significant computation between .item() calls
#   (>= _MIN_OPS_BETWEEN_SYNCS fire-and-forget ops), distinguishing generate() loops
#   (200+ ops/token) from gradcheck (few ops between allclose checks)
# - On misprediction: falls back to blocking, generates at most one extra token
_SPECULATION_ENABLED = os.environ.get("SKYTORCH_SPECULATIVE_SCALAR", "1") == "1"

# Minimum fire-and-forget ops between .item() calls to consider speculation.
# generate() dispatches 200-300 ops per token; gradcheck dispatches ~10-20 ops
# between allclose checks. A threshold of 50 cleanly separates the two patterns.
_MIN_OPS_BETWEEN_SYNCS = 50

# Counter incremented by the dispatch path for each fire-and-forget op.
# Read and reset by _local_scalar_dense to detect the generate() pattern.
_ops_since_last_sync: int = 0

# Speculation state machine:
#   None → first call, no history
#   {"value": v, "future": f|None, "confirmed": bool}
#     confirmed=False: seen one value, need a matching second before speculating
#     confirmed=True:  pattern confirmed, actively speculating
_speculation_state: dict | None = None


def _reset_speculation() -> None:
    """Reset speculation state. Called on device reset / stream close."""
    global _speculation_state, _ops_since_last_sync
    _speculation_state = None
    _ops_since_last_sync = 0


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
        # 1. Stopping criterion: unfinished_sequences.max() == 0 (~1800 ops since last)
        # 2. Cache check: cache_position[-1] >= input_ids.shape[1] (0 ops since last)
        # Only #1 is the "qualifying" sync. #2 is an incidental sync that must not
        # reset the speculation state or ops counter.
        _ops_between = _ops_since_last_sync
        _can_speculate = (
            _SPECULATION_ENABLED
            and self.dtype == torch.bool
            and _ops_between >= _MIN_OPS_BETWEEN_SYNCS
        )
        # Only reset counter on qualifying syncs — non-qualifying ones are transparent
        if _can_speculate:
            _ops_since_last_sync = 0

        # Incidental bool sync during active speculation (e.g., cache_position[-1] >= shape
        # in HuggingFace's _cache_dependant_input_preparation). These have 0 ops between
        # syncs but their value is also False during generation. Return speculated value
        # immediately without submitting a server request — avoids event loop contention
        # from unnecessary round-trips.
        if (
            not _can_speculate
            and _speculation_state is not None
            and _speculation_state["confirmed"]
            and self.dtype == torch.bool
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

                if prev_future is not None and prev_future.done():
                    # Previous result available — validate for free (no blocking)
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
        # Only reset state when there's positive evidence we've left a generate loop:
        # a bool tensor with too few ops (e.g., gradcheck pattern). Non-qualifying
        # calls within the same token (0 ops) and non-bool dtype calls are transparent.
        if (
            _speculation_state is not None
            and not _can_speculate
            and self.dtype == torch.bool
            and _ops_between > 0
            and _ops_between < _MIN_OPS_BETWEEN_SYNCS
        ):
            _speculation_state = None

        result = future.result()

        if _can_speculate and _speculation_state is None:
            # Seed speculation state (unconfirmed)
            _speculation_state = {"value": result, "future": None, "confirmed": False}

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
    # Check basic compatibility
    if self.shape != other.shape:
        return False
    if self.dtype != other.dtype:
        return False

    # Perform element-wise comparison on sky device
    eq_tensor = torch.eq(self, other)

    # Reduce to single boolean
    all_equal_tensor = torch.all(eq_tensor)

    # Get scalar result (copies single value to cpu)
    return all_equal_tensor.item()
