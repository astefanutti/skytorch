"""
KPU ATen Copy Operations.

This module implements copy operations between KPU and other devices.
Copy operations need explicit implementation because they involve
data transfer between devices via gRPC.

The actual transfer logic is delegated to the manager module.
"""

from __future__ import annotations

import torch

from kpu.torch.backend._async import run_async
from kpu.torch.backend import _client


def _copy_from_device(tensor: torch.Tensor) -> torch.Tensor:
    """Copy data from KPU tensor to CPU tensor.

    Args:
        tensor: Source KPU tensor

    Returns:
        CPU tensor with copied data
    """
    return run_async(_client.copy_kpu_to_cpu(tensor))


def _copy_to_device(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Copy data from CPU tensor to KPU tensor.

    Args:
        src: Source CPU tensor
        dst: Destination KPU tensor

    Returns:
        Destination tensor (same as dst)
    """
    return run_async(_client.copy_cpu_to_kpu(src, dst))


def _copy_kpu_to_kpu(src: torch.Tensor, dst: torch.Tensor) -> None:
    """Copy data between KPU tensors.

    Args:
        src: Source KPU tensor
        dst: Destination KPU tensor
    """
    run_async(_client.copy_kpu_to_kpu(src, dst))


def _copy_from(
    from_: torch.Tensor,
    to_: torch.Tensor,
    non_blocking: bool = False,
) -> torch.Tensor:
    """Copy data from one tensor to another, handling KPU device transfers.

    This function implements the core copy operation for KPU tensors,
    supporting CPU<->KPU transfers and KPU<->KPU copies.

    Args:
        from_: Source tensor to copy from
        to_: Target tensor to copy to
        non_blocking: Whether to perform the copy asynchronously (currently ignored)

    Returns:
        Target tensor with copied data

    Raises:
        RuntimeError: If attempting unsupported copy operations
    """
    if from_.device.type == "kpu" and to_.device.type == "cpu":
        # KPU to CPU
        host_mem = _copy_from_device(from_)
        return to_.copy_(host_mem)

    elif from_.device.type == "cpu" and to_.device.type == "kpu":
        # CPU to KPU
        return _copy_to_device(from_, to_)

    elif from_.device.type == "kpu" and to_.device.type == "kpu":
        # KPU to KPU
        _copy_kpu_to_kpu(from_, to_)
        return to_

    else:
        raise RuntimeError(
            f"Copy operation from {from_.device.type} to {to_.device.type} "
            f"is not supported. Only CPU<->KPU transfers and KPU<->KPU copies "
            f"are allowed."
        )
