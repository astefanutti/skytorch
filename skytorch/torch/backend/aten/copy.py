"""
SkyTorch ATen Copy Operations.

This module implements copy operations between sky and other devices.
Copy operations need explicit implementation because they involve
data transfer between devices via gRPC.

The actual transfer logic is delegated to the manager module.
"""

from __future__ import annotations

import os

import torch

from skytorch.torch.backend import _client

# Async scalar copy: when enabled, single-element sky→cpu copies return a
# DeferredScalarTensor immediately instead of blocking on the gRPC round-trip.
# The actual value is resolved lazily on tolist()/item(). This eliminates the
# server idle gap during LLM generation (the .cpu() call in HuggingFace's
# generate loop is only used by the streamer).
_ASYNC_COPY_ENABLED = os.environ.get("SKYTORCH_ASYNC_COPY", "0") == "1"

# Back-pressure: the previous async future. Validated before issuing the
# next async copy to prevent the client from racing ahead of the server.
_async_prev_future = None


def _reset_async_copy() -> None:
    """Reset async copy state. Called on device reset / stream close."""
    global _async_prev_future
    _async_prev_future = None


def _copy_from_device(tensor: torch.Tensor) -> torch.Tensor:
    """Copy data from sky tensor to cpu tensor.

    Args:
        tensor: Source sky tensor

    Returns:
        cpu tensor with copied data
    """
    return _client.copy_sky_to_cpu(tensor)


def _copy_to_device(src: torch.Tensor, dst: torch.Tensor) -> None:
    """Copy data from cpu tensor to sky tensor.

    When streaming is enabled, the update_tensor goes through the stream
    ensuring proper ordering with other operations.

    Args:
        src: Source cpu tensor
        dst: Destination sky tensor

    Returns:
        Destination tensor (same as dst)
    """
    _client.copy_cpu_to_sky(src, dst)


def _copy_sky_to_sky(src: torch.Tensor, dst: torch.Tensor) -> None:
    """Copy data between sky tensors.

    Args:
        src: Source sky tensor
        dst: Destination sky tensor
    """
    _client.copy_sky_to_sky(src, dst)


def _copy_from_device_async(from_: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
    """Non-blocking sky→cpu copy for single-element tensors.

    Returns a DeferredScalarTensor immediately. The actual value is resolved
    lazily when tolist() or item() is called. Back-pressure is enforced by
    validating the previous async future before issuing a new one.

    Args:
        from_: Source sky tensor (numel() == 1)
        target_dtype: Data type for the result tensor

    Returns:
        DeferredScalarTensor wrapping the backing tensor
    """
    global _async_prev_future

    from skytorch.torch.backend.aten.deferred import DeferredScalarTensor

    # Back-pressure: validate previous async copy completed.
    # On warm servers, the previous future is already resolved (server had a
    # full forward pass to process it), so this is essentially free.
    if _async_prev_future is not None:
        _async_prev_future.result()

    future = _client.copy_sky_to_cpu_async(from_)
    _async_prev_future = future

    return DeferredScalarTensor.create(from_.shape, target_dtype, future)


def _copy_from(
    from_: torch.Tensor,
    to_: torch.Tensor,
    non_blocking: bool = False,
) -> torch.Tensor:
    """Copy data from one tensor to another, handling sky device transfers.

    This function implements the core copy operation for sky tensors,
    supporting cpu<->sky transfers and sky<->sky copies.

    Args:
        from_: Source tensor to copy from
        to_: Target tensor to copy to
        non_blocking: Whether to perform the copy asynchronously (currently ignored)

    Returns:
        Target tensor with copied data

    Raises:
        RuntimeError: If attempting unsupported copy operations
    """
    if from_.device.type == "sky" and to_.device.type == "cpu":
        host_mem = _copy_from_device(from_)
        return to_.copy_(host_mem)

    elif from_.device.type == "cpu" and to_.device.type == "sky":
        # cpu to sky
        _copy_to_device(from_, to_)
        return to_

    elif from_.device.type == "sky" and to_.device.type == "sky":
        # sky to sky
        _copy_sky_to_sky(from_, to_)
        return to_

    else:
        raise RuntimeError(
            f"Copy operation from {from_.device.type} to {to_.device.type} "
            f"is not supported. Only cpu<->sky transfers and sky<->sky copies "
            f"are allowed."
        )


def _sky_to_copy(
    self,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    non_blocking=False,
    memory_format=None,
):
    """Override aten::_to_copy for sky tensors.

    This intercepts .cpu() / .to('cpu') at the level where PyTorch uses the
    return value, unlike _copy_from whose return value is discarded by copy_().
    For single-element sky→cpu copies with async copy enabled, returns a
    DeferredScalarTensor directly. For all other cases, replicates the
    CompositeExplicitAutograd decomposition (empty_strided + _copy_from).
    """
    target_device = device if device is not None else self.device
    target_dtype = dtype if dtype is not None else self.dtype

    # Async path: sky → cpu for single-element tensors
    if (
        target_device.type == "cpu"
        and _ASYNC_COPY_ENABLED
        and self.numel() == 1
        and _client.ENABLE_STREAMING
    ):
        return _copy_from_device_async(self, target_dtype)

    # Default path: replicate _to_copy CEA decomposition.
    # PyTorch's decomposition checks is_non_overlapping_and_dense() before
    # using empty_strided — expanded tensors (stride 0) have overlapping
    # memory and must use contiguous layout instead.
    has_overlapping = any(s == 0 and d > 1 for s, d in zip(self.stride(), self.shape))
    if memory_format is None or memory_format == torch.preserve_format:
        if not has_overlapping:
            result = torch.empty_strided(
                self.shape,
                self.stride(),
                dtype=target_dtype,
                device=target_device,
            )
        else:
            result = torch.empty(
                self.shape,
                dtype=target_dtype,
                device=target_device,
                memory_format=torch.contiguous_format,
            )
    else:
        result = torch.empty(
            self.shape,
            dtype=target_dtype,
            device=target_device,
            memory_format=memory_format,
        )
    _copy_from(self, result, non_blocking)
    return result
