"""
KPU Client operations - Async tensor transfer and remote execution.

This module provides async functions for tensor data transfer,
remote ATen operation execution via gRPC, and Compute resolution.
"""

from __future__ import annotations

from typing import Optional

import torch

from kpu.client import Compute
from kpu.torch.backend._device import device_manager
from kpu.torch.backend._storage import storage_manager
from kpu.torch.client.tensor import get_storage_id, get_tensor_id, get_tensor_metadata
from kpu.torch.client.service import TensorClient
from kpu.torch.client.utils import async_map_args_kwargs


async def copy_kpu_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    """
    Copy data from a KPU tensor to a CPU tensor.

    Uses get_tensor to download tensor data from the server.

    Args:
        tensor: Source KPU tensor

    Returns:
        CPU tensor with copied data
    """
    if tensor.device.type != "kpu":
        raise ValueError("copy_kpu_to_cpu requires a KPU tensor")

    compute = _require_compute(tensor)
    client = _require_client(compute)

    cpu_tensor = await client.get_tensor(
        tensor_id=get_tensor_id(tensor),
        shape=tuple(tensor.shape),
        dtype=tensor.dtype,
        stride=tuple(tensor.stride()),
        storage_offset=tensor.storage_offset(),
    )

    return cpu_tensor


async def copy_cpu_to_kpu(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    Copy data from a CPU tensor to a KPU tensor.

    Uses update_tensor to upload tensor to the server.
    If the destination tensor is not registered,
    creates it first before updating.

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

    compute = _require_compute(dst)
    client = _require_client(compute)

    await _ensure_tensor_created(dst, client)
    await client.update_tensor(src=src, tensor_id=get_tensor_id(dst))

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

    src_compute = _resolve_compute(src)
    dst_compute = _resolve_compute(dst)

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

    # Ensure both tensors are created on the server
    await _ensure_tensor_created(src, client)
    await _ensure_tensor_created(dst, client)

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
) -> list[int] | None:
    """
    Execute an ATen operation on the remote Compute.

    Supports two modes:
    - Pre-allocated outputs: output_tensors provided, writes to them, returns None
    - Server-created outputs: output_tensors is None, returns list[int] (tensor_ids)

    Args:
        kpu_device: KPU device to execute on
        op_name: ATen operation name (e.g., "aten::add.Tensor")
        args: Positional arguments (may contain KPU tensors)
        kwargs: Keyword arguments (may contain KPU tensors)
        output_tensors: Pre-allocated output tensors, or None for server-created

    Returns:
        None if output_tensors provided, list[int] of tensor_ids if server created outputs

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

    async def process_arg(obj):
        """Process an argument: register tensors and map devices."""
        if isinstance(obj, torch.Tensor):
            if obj.device.type == "kpu":
                await _ensure_tensor_created(obj, client)
                return obj
            elif obj.device.type == "cpu" and obj.dim() == 0:
                return obj  # CPU scalar tensors are valid
            else:
                raise ValueError(
                    f"Unsupported tensor: {obj.device.type} with dim {obj.dim()}. "
                    f"Only KPU tensors and 0-dim CPU scalar tensors are allowed."
                )
        elif isinstance(obj, torch.device):
            if obj.type == "kpu":
                # Map KPU device to remote device
                info = device_manager.get_remote_device_info(obj.index or 0)
                return torch.device(info.device_type, info.device_index)
            return obj
        return obj

    # Process args/kwargs: register tensors and map devices
    processed_args, processed_kwargs = await async_map_args_kwargs(
        process_arg, args, kwargs
    )

    result = await client.execute_aten_operation(
        op_name=op_name,
        args=processed_args,
        kwargs=processed_kwargs,
        output_tensors=output_tensors,
    )

    # Register output tensors
    if output_tensors:
        for tensor in output_tensors:
            if tensor is not None:
                storage_manager.register(tensor)

    return result


async def _ensure_tensor_created(
        tensor: torch.Tensor,
        client: TensorClient,
) -> None:
    """
    Ensure a KPU tensor is created on the remote server.

    If the tensor is not already registered, creates it on the server
    with the appropriate remote device mapping and registers it locally.
    If the tensor is a view of an already-registered tensor, creates
    a view on the server referencing the base tensor's storage.

    Args:
        tensor: KPU tensor to create
        client: TensorClient for gRPC operations
    """
    tensor_id = get_tensor_id(tensor)
    ref = storage_manager.reference(tensor)

    if ref == tensor_id:
        # Tensor already registered
        return

    if ref is not None:
        # ref is different tensor_id → this tensor is a view of base tensor
        # Create view on server referencing the base tensor
        metadata = get_tensor_metadata(tensor)
        remote_info = device_manager.get_remote_device_info(tensor.device.index)
        metadata.device_type = remote_info.device_type
        metadata.device_index = remote_info.device_index
        await client.create_tensor(metadata, tensor_ref=ref)
        storage_manager.register(tensor)
        return

    # No reference found → create new tensor with fresh storage
    metadata = get_tensor_metadata(tensor)
    remote_info = device_manager.get_remote_device_info(tensor.device.index)
    metadata.device_type = remote_info.device_type
    metadata.device_index = remote_info.device_index
    await client.create_tensor(metadata)
    storage_manager.register(tensor)


def _resolve_compute(tensor: torch.Tensor) -> Optional[Compute]:
    """
    Resolve the Compute associated with a KPU tensor.

    Resolution order:
    1. Check if the tensor's storage has an associated Compute
    2. Fall back to the current context (compute_ctx)

    Args:
        tensor: A KPU tensor

    Returns:
        The associated Compute, or None if not found
    """
    storage_id = get_storage_id(tensor)
    storage_info = storage_manager.get(storage_id)

    # First, try storage-associated Compute
    if storage_info is not None and storage_info.compute is not None:
        return storage_info.compute

    # Fall back to context
    from kpu.client.context import compute_ctx

    return compute_ctx.get(None)


def _require_compute(tensor: torch.Tensor) -> Compute:
    """
    Resolve and require a Compute for a tensor.

    Like resolve_compute but raises if no Compute is available.

    Args:
        tensor: A KPU tensor

    Returns:
        The associated Compute

    Raises:
        RuntimeError: If no Compute context is available
    """
    compute = _resolve_compute(tensor)
    if compute is None:
        raise RuntimeError(
            "No Compute context available for KPU tensor operation. "
            "Ensure you are within an 'async with Compute(...):' block."
        )
    return compute


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
