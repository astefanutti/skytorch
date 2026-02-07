"""
SkyTorch ATen Copy Operations.

This module implements copy operations between sky and other devices.
Copy operations need explicit implementation because they involve
data transfer between devices via gRPC.

The actual transfer logic is delegated to the manager module.
"""

from __future__ import annotations

import torch

from skytorch.torch.backend import _client


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
        # sky to cpu
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
