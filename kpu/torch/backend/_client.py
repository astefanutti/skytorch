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
from kpu.torch.client.metadata import TensorMetadata
from kpu.torch.client.service import TensorClient
from kpu.torch.client.utils import map_args_kwargs


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

    Uses a single update_tensor call with metadata for auto-creation.
    The server will create the tensor if it doesn't exist.

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

    # Get metadata for auto-creation if tensor is not registered
    metadata = _get_tensor_metadata_if_new(dst)

    # Single call - server will auto-create tensor from metadata
    await client.update_tensor(
        src=src,
        tensor_id=get_tensor_id(dst),
        tensor_metadata=metadata,
    )

    # Register locally after RPC
    if metadata is not None:
        _register_tensor_locally(dst)

    return dst


async def copy_kpu_to_kpu(src: torch.Tensor, dst: torch.Tensor) -> None:
    """
    Copy data between KPU tensors on the same Compute.

    Uses a single copy_tensor call with metadata for auto-creation.
    The server will create tensors if they don't exist.

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

    # Get metadata for auto-creation if tensors are not registered
    src_meta = _get_tensor_metadata_if_new(src)
    dst_meta = _get_tensor_metadata_if_new(dst)

    # Single call - server will auto-create tensors from metadata
    await client.copy_tensor(
        src_tensor_id=get_tensor_id(src),
        dst_tensor_id=get_tensor_id(dst),
        src_offset=src.storage_offset() * src.element_size(),
        dst_offset=dst.storage_offset() * dst.element_size(),
        num_bytes=src.numel() * src.element_size(),
        src_metadata=src_meta,
        dst_metadata=dst_meta,
    )

    # Register locally after RPC
    if src_meta is not None:
        _register_tensor_locally(src)
    if dst_meta is not None:
        _register_tensor_locally(dst)


async def delete_tensors(compute: Compute, tensor_ids: list[int]) -> None:
    """
    Delete tensors on the remote server.

    Args:
        compute: The Compute instance
        tensor_ids: List of tensor IDs to delete
    """
    client = _require_client(compute)
    await client.delete_tensors(tensor_ids)


async def execute_aten_operation(
    kpu_device: torch.device,
    op_name: str,
    args: tuple,
    kwargs: dict,
    output_tensors: list[torch.Tensor] | None,
) -> list[int] | None:
    """
    Execute an ATen operation on the remote Compute.

    Uses a single execute_aten_operation call with metadata for auto-creation.
    The server will create tensors if they don't exist.

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

    # Collect metadata for all unregistered input tensors
    tensor_metadata_list: list[TensorMetadata] = []
    tensors_to_register: list[torch.Tensor] = []

    def process_arg(obj):
        """Process an argument: collect metadata and map devices."""
        if isinstance(obj, torch.Tensor):
            if obj.device.type == "kpu":
                # Collect metadata for auto-creation if not registered
                meta = _get_tensor_metadata_if_new(obj)
                if meta is not None:
                    tensor_metadata_list.append(meta)
                    tensors_to_register.append(obj)
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

    # Process args/kwargs: collect metadata and map devices (synchronously)
    processed_args, processed_kwargs = map_args_kwargs(process_arg, args, kwargs)

    # Collect metadata for output tensors
    output_metadata_list: list[TensorMetadata] = []
    output_tensors_to_register: list[torch.Tensor] = []
    if output_tensors:
        for tensor in output_tensors:
            if tensor is not None:
                meta = _get_tensor_metadata_if_new(tensor)
                if meta is not None:
                    output_metadata_list.append(meta)
                    output_tensors_to_register.append(tensor)

    # Single call - server will auto-create tensors from metadata
    result = await client.execute_aten_operation(
        op_name=op_name,
        args=processed_args,
        kwargs=processed_kwargs,
        output_tensors=output_tensors,
        tensor_metadata=tensor_metadata_list if tensor_metadata_list else None,
        output_metadata=output_metadata_list if output_metadata_list else None,
    )

    # Register tensors locally after RPC
    for tensor in tensors_to_register:
        _register_tensor_locally(tensor)
    for tensor in output_tensors_to_register:
        _register_tensor_locally(tensor)

    return result


def _get_tensor_metadata_if_new(tensor: torch.Tensor) -> Optional[TensorMetadata]:
    """
    Return metadata for a tensor if it's not yet registered, else None.

    Checks if the tensor is already registered on the server. If not,
    creates metadata with the appropriate remote device mapping and
    tensor_ref if it's a view.

    Args:
        tensor: KPU tensor to check

    Returns:
        TensorMetadata if tensor needs to be created, None if already registered
    """
    tensor_id = get_tensor_id(tensor)
    ref = storage_manager.tensor_ref(tensor)

    if ref == tensor_id:
        # Tensor already registered with this exact tensor_id
        return None

    # Create metadata for the tensor
    metadata = get_tensor_metadata(tensor)
    remote_info = device_manager.get_remote_device_info(tensor.device.index)
    metadata.device_type = remote_info.device_type
    metadata.device_index = remote_info.device_index

    if ref is not None:
        # ref is different tensor_id → this tensor is a view of base tensor
        metadata.tensor_ref = ref

    return metadata


def _register_tensor_locally(tensor: torch.Tensor) -> None:
    """
    Register a tensor locally after it has been created on the server.

    Handles lazy storage registration (storage IDs are generated by
    the C++ allocator) and tensor registration.

    Args:
        tensor: KPU tensor to register
    """
    storage_id = get_storage_id(tensor)
    storage_manager.register_storage(
        storage_id=storage_id,
        nbytes=tensor.untyped_storage().nbytes(),
        device_index=tensor.device.index or 0,
    )
    storage_manager.register_tensor(tensor)


def _resolve_compute(tensor: torch.Tensor) -> Optional[Compute]:
    """
    Resolve the Compute associated with a KPU tensor.

    Resolution order:
    1. Check if the tensor's storage has an associated Compute
    2. Check the device_manager for the device index (for lazy-allocated storage)
    3. Fall back to the current context (compute_ctx)

    Args:
        tensor: A KPU tensor

    Returns:
        The associated Compute, or None if not found
    """
    storage_id = get_storage_id(tensor)
    storage_info = storage_manager.get_storage(storage_id)

    # First, try storage-associated Compute
    if storage_info is not None and storage_info.compute is not None:
        return storage_info.compute

    # Second, try device_manager using device index
    # This handles lazy-allocated storage that hasn't been registered yet
    device_index = tensor.device.index or 0
    compute = device_manager.get_compute(device_index)
    if compute is not None:
        return compute

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

    The GRPCClient handles thread-local channel management internally,
    returning the appropriate client for the current thread.

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
            "The gRPC client has not been initialized."
        )
    return compute._grpc_client.torch
