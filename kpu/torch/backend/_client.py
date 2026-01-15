"""
KPU Client operations - Async tensor transfer and remote execution.

This module provides async functions for tensor data transfer and
remote ATen operation execution via gRPC.
"""

from __future__ import annotations

import torch

from kpu.client import Compute
from kpu.torch.backend._device import device_manager
from kpu.torch.client.tensor import (
    get_tensor_id,
    require_compute,
    resolve_compute,
)
from kpu.torch.client import TensorClient
from kpu.torch.client.metadata import TensorMetadata


async def copy_kpu_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    """
    Copy data from a KPU tensor to a CPU tensor.

    Uses get_storage_data to download tensor data from the server.

    Args:
        tensor: Source KPU tensor

    Returns:
        CPU tensor with copied data
    """
    if tensor.device.type != "kpu":
        raise ValueError("copy_kpu_to_cpu requires a KPU tensor")

    compute = require_compute(tensor)
    client = _require_client(compute)

    cpu_tensor = await client.get_storage_data(
        tensor_id=get_tensor_id(tensor),
        shape=tuple(tensor.shape),
        dtype=tensor.dtype,
        stride=tuple(tensor.stride()),
        storage_offset=tensor.storage_offset(),
    )

    # Ensure the received tensor has the correct shape
    if cpu_tensor.shape != tensor.shape:
        cpu_tensor = cpu_tensor.view(tensor.shape)

    return cpu_tensor


async def copy_cpu_to_kpu(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    Copy data from a CPU tensor to a KPU tensor.

    Uses update_tensor to upload tensor data to the server.

    Args:
        src: Source CPU tensor
        dst: Destination KPU tensor

    Returns:
        Destination tensor (same as dst)
    """
    if dst.device.type != "kpu":
        raise ValueError("copy_cpu_to_kpu requires a KPU target tensor")
    if src.device.type != "cpu":
        raise ValueError("copy_cpu_to_kpu requires a CPU source tensor")

    compute = require_compute(dst)
    client = _require_client(compute)

    await client.update_tensor(
        tensor=src.contiguous(),
        tensor_id=get_tensor_id(dst),
        storage_offset=dst.storage_offset(),
    )

    return dst


async def copy_kpu_to_kpu(src: torch.Tensor, dst: torch.Tensor) -> None:
    """
    Copy data between KPU tensors on the same Compute.

    Uses server-side copy_tensor for efficiency (no data round-trip).

    Args:
        src: Source KPU tensor
        dst: Destination KPU tensor

    Raises:
        ValueError: If tensors are not KPU tensors
        RuntimeError: If no Compute context is available or tensors
            are on different Computes
    """
    if src.device.type != "kpu" or dst.device.type != "kpu":
        raise ValueError("copy_kpu_to_kpu requires KPU tensors")

    src_compute = resolve_compute(src)
    dst_compute = resolve_compute(dst)

    if src_compute is None or dst_compute is None:
        raise RuntimeError(
            "Cannot copy between KPU tensors without Compute context. "
            "Ensure you are within an 'async with Compute(...):' block."
        )

    if src_compute is not dst_compute:
        raise RuntimeError(
            "Cross-Compute tensor copy is not supported. "
            "Both tensors must be on the same Compute resource."
        )

    client = _require_client(src_compute)

    # Use server-side copy for efficiency
    await client.copy_tensor(
        src_tensor_id=get_tensor_id(src),
        dst_tensor_id=get_tensor_id(dst),
        src_offset=src.storage_offset() * src.element_size(),
        dst_offset=dst.storage_offset() * dst.element_size(),
        num_bytes=src.numel() * src.element_size(),
    )


async def execute_aten_operation(
    kpu_device: torch.device,
    op_name: str,
    args: tuple,
    kwargs: dict,
    output_tensors: list[torch.Tensor] | None,
) -> list[TensorMetadata] | None:
    """
    Execute an ATen operation on the remote Compute.

    Supports two modes:
    - Pre-allocated outputs: output_tensors provided, writes to them, returns None
    - Server-created outputs: output_tensors is None, returns list[TensorMetadata]

    Args:
        kpu_device: KPU device to execute on
        op_name: ATen operation name (e.g., "aten::add.Tensor")
        args: Positional arguments (may contain KPU tensors)
        kwargs: Keyword arguments (may contain KPU tensors)
        output_tensors: Pre-allocated output tensors, or None for server-created

    Returns:
        None if output_tensors provided, list[TensorMetadata] if server created outputs

    Raises:
        RuntimeError: If no Compute registered for the device
    """
    compute = device_manager.get_compute(kpu_device.index)
    if compute is None:
        raise RuntimeError(
            "No Compute context available for ATen operation. "
            "Ensure you are within an 'async with Compute(...):' block."
        )

    client = _require_client(compute)

    return await client.execute_aten_operation(
        op_name=op_name,
        args=args,
        kwargs=kwargs,
        output_tensors=output_tensors,
    )


def _require_client(compute: Compute) -> TensorClient:
    """
    Get the TensorClient from a Compute instance.

    Args:
        compute: The Compute instance

    Returns:
        TensorClient for gRPC operations

    Raises:
        RuntimeError: If the Compute is not ready
    """
    if compute._grpc_client is None:
        raise RuntimeError(
            f"Compute '{compute.name}' is not ready. "
            "The gRPC client has not been initialized. "
            "Ensure the Compute is ready before performing tensor operations."
        )
    return compute._grpc_client.torch
