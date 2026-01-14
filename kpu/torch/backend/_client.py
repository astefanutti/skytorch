"""
KPU Client operations - Async tensor transfer and remote execution.

This module provides async functions for tensor data transfer and
remote ATen operation execution via gRPC.
"""

from __future__ import annotations

from typing import Optional

import torch

from kpu.torch.backend._tensor import (
    get_tensor_id,
    get_tensor_metadata,
    get_tensor_client,
    require_compute,
    resolve_compute,
)


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
    client = get_tensor_client(compute)

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
    client = get_tensor_client(compute)

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

    client = get_tensor_client(src_compute)

    # Use server-side copy for efficiency
    await client.copy_tensor(
        src_tensor_id=get_tensor_id(src),
        dst_tensor_id=get_tensor_id(dst),
        src_offset=src.storage_offset() * src.element_size(),
        dst_offset=dst.storage_offset() * dst.element_size(),
        num_bytes=src.numel() * src.element_size(),
    )


async def execute_aten_operation(
    op_name: str,
    input_tensors: list[torch.Tensor],
    output_tensors: list[torch.Tensor],
    kwargs: Optional[dict] = None,
) -> None:
    """
    Execute an ATen operation on the remote Compute.

    Resolves the Compute from the input tensors. All tensors must be on
    the same Compute.

    Args:
        op_name: ATen operation name (e.g., "aten::add.Tensor")
        input_tensors: List of input KPU tensors
        output_tensors: List of output KPU tensors
        kwargs: Optional keyword arguments for the operation

    Raises:
        RuntimeError: If no Compute context is available
    """
    if not input_tensors:
        raise ValueError("execute_aten_operation requires at least one input tensor")

    compute = require_compute(input_tensors[0])
    client = get_tensor_client(compute)

    input_refs = [get_tensor_metadata(t) for t in input_tensors]
    output_refs = [get_tensor_metadata(t) for t in output_tensors]

    await client.execute_aten_operation(
        op_name=op_name,
        input_refs=input_refs,
        output_refs=output_refs,
        kwargs=kwargs,
    )
