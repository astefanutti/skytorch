"""
KPU Client operations - Async tensor transfer and remote execution.

This module provides async functions for tensor data transfer,
remote ATen operation execution via gRPC, and Compute resolution.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional

import torch

# Feature flags
ENABLE_STREAMING = os.environ.get("KPU_ENABLE_STREAMING", "1") == "1"

from kpu.torch.server.serialization import (
    tensor_from_bytes,
    tensor_to_bytes,
)

from kpu.client import Compute
from kpu.torch.backend._device import device_manager
from kpu.torch.backend._storage import storage_manager
from kpu.torch.client.tensor import get_storage_id, get_tensor_id, get_tensor_metadata
from kpu.torch.client.metadata import TensorMetadata
from kpu.torch.client.service import TensorClient
from kpu.torch.client.request import (
    tensor_metadata_to_proto,
    build_copy_tensor_request,
    build_execute_aten_request,
)


async def copy_kpu_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    """
    Copy data from a KPU tensor to a CPU tensor.

    Uses streaming get_tensor to download tensor data from the server.

    Args:
        tensor: Source KPU tensor

    Returns:
        CPU tensor with copied data
    """
    from kpu.torch.server import service_pb2

    if tensor.device.type != "kpu":
        raise ValueError("copy_kpu_to_cpu requires a KPU tensor")

    compute = _require_compute(tensor)

    if ENABLE_STREAMING:
        stream_manager = compute._grpc_client.stream

        tensor_id = get_tensor_id(tensor)

        request = service_pb2.GetTensorRequest(
            tensor_id=tensor_id,
            shape=list(tensor.shape),
            dtype=str(tensor.dtype),
            stride=list(tensor.stride()),
            storage_offset=tensor.storage_offset(),
        )

        future = stream_manager.submit_get_tensor(request)

        # Await the future - it's on the global loop which we're running on
        response = await future

        if not response.success:
            raise RuntimeError(f"Failed to get tensor: {response.error_message}")

        # Deserialize tensor data
        dtype = eval(response.get_tensor.dtype)
        shape = list(response.get_tensor.shape)
        data = response.get_tensor.data

        return tensor_from_bytes(data, dtype, shape)

    else:
        # Use unary RPC
        client = _require_client(compute)
        cpu_tensor = await client.get_tensor(
            tensor_id=get_tensor_id(tensor),
            shape=tuple(tensor.shape),
            dtype=tensor.dtype,
            stride=tuple(tensor.stride()),
            storage_offset=tensor.storage_offset(),
        )
        return cpu_tensor


async def copy_cpu_to_kpu(src: torch.Tensor, dst: torch.Tensor):
    """
    Copy data from a CPU tensor to a KPU tensor.

    Uses streaming update_tensor call with metadata for auto-creation.
    The server will create the tensor if it doesn't exist.

    Args:
        src: Source CPU tensor
        dst: Destination KPU tensor

    Returns:
        Destination tensor (same as dst)
    """
    from kpu.torch.server import service_pb2

    if dst.device.type != "kpu":
        raise ValueError("copy_cpu_to_kpu requires a KPU target tensor")
    if src.device.type != "cpu":
        raise ValueError("copy_cpu_to_kpu requires a CPU source tensor")

    compute = _require_compute(dst)

    # Get metadata for auto-creation if tensor is not registered
    meta = _get_tensor_metadata_if_new(dst)

    if ENABLE_STREAMING:
        # Use streaming channel for proper ordering
        stream_manager = compute._grpc_client.stream

        # Serialize tensor data (detach in case it requires grad)
        data = tensor_to_bytes(src)

        # Build metadata proto if needed
        metadata_proto = None
        if meta is not None:
            metadata_proto = tensor_metadata_to_proto(meta)

        request = service_pb2.UpdateTensorRequest(
            tensor_id=get_tensor_id(dst),
            data=data,
            shape=list(dst.shape),
            dtype=str(dst.dtype),
            stride=list(dst.stride()),
            storage_offset=dst.storage_offset(),
        )
        if metadata_proto is not None:
            request.metadata.CopyFrom(metadata_proto)

        # FIXME: error handling
        future = stream_manager.submit_update_tensor(request)
    else:
        # Use unary RPC
        client = _require_client(compute)
        await client.update_tensor(
            src=src,
            tensor_id=get_tensor_id(dst),
            tensor_metadata=meta,
        )

    # Register locally after RPC
    if meta is not None:
        _register_tensor_locally(dst)


async def copy_kpu_to_kpu(src: torch.Tensor, dst: torch.Tensor) -> None:
    """
    Copy data between KPU tensors on the same Compute.

    Uses streaming mode (fire-and-forget) when ENABLE_STREAMING is True,
    otherwise uses synchronous unary RPC.

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

    # Get metadata for auto-creation if tensors are not registered
    src_meta = _get_tensor_metadata_if_new(src)
    dst_meta = _get_tensor_metadata_if_new(dst)

    if ENABLE_STREAMING:
        stream_manager = src_compute._grpc_client.stream

        # Build request
        request = build_copy_tensor_request(
            src_tensor_id=get_tensor_id(src),
            dst_tensor_id=get_tensor_id(dst),
            src_offset=src.storage_offset() * src.element_size(),
            dst_offset=dst.storage_offset() * dst.element_size(),
            num_bytes=src.numel() * src.element_size(),
            src_metadata=src_meta,
            dst_metadata=dst_meta,
        )

        # FIXME: error handling
        future = stream_manager.submit_copy_tensor(request)
        # Fire-and-forget: don't await in streaming mode
    else:
        # Unary mode: wait for operation to complete
        client = _require_client(src_compute)
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

    When streaming is enabled, routes through the stream to maintain
    ordering with other operations. Otherwise uses unary RPC.

    Args:
        compute: The Compute instance
        tensor_ids: List of tensor IDs to delete
    """
    from kpu.torch.server import service_pb2

    if ENABLE_STREAMING:
        # Use streaming channel to maintain ordering with other ops
        stream_manager = compute._grpc_client.stream
        request = service_pb2.DeleteTensorsRequest(tensor_ids=tensor_ids)
        # FIXME: error handling
        future = stream_manager.submit_delete_tensors(request)
        # Fire-and-forget: don't wait for response
        # The stream ordering ensures deletes happen after prior ops complete
    else:
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

    Uses streaming (fire-and-forget) when ENABLE_STREAMING is True,
    otherwise uses unary RPC and waits for completion.

    Args:
        kpu_device: KPU device to execute on
        op_name: ATen operation name (e.g., "aten::add.Tensor")
        args: Positional arguments (may contain KPU tensors)
        kwargs: Keyword arguments (may contain KPU tensors)
        output_tensors: Pre-allocated output tensors, or None for server-created

    Returns:
        None in streaming mode, list[int] of tensor_ids in unary mode if server created outputs
    """
    compute = device_manager.get_compute(kpu_device.index)
    if compute is None:
        raise RuntimeError(
            "No Compute context available for ATen operation. "
            "Ensure you are within an 'async with Compute(...):' block."
        )

    if compute._grpc_client is None:
        raise RuntimeError(
            f"Compute '{compute.name}' is not ready. "
            "The gRPC client has not been initialized."
        )

    # Collect metadata for unregistered input tensors
    tensor_metadata_list: list[TensorMetadata] = []
    tensors_to_register: list[torch.Tensor] = []

    def process_arg(obj):
        """Process an argument: collect metadata and map devices."""
        if isinstance(obj, torch.Tensor):
            if obj.device.type == "kpu":
                meta = _get_tensor_metadata_if_new(obj)
                if meta is not None:
                    tensor_metadata_list.append(meta)
                    tensors_to_register.append(obj)
                return obj
            elif obj.device.type == "cpu" and obj.dim() == 0:
                return obj
            else:
                raise ValueError(
                    f"Unsupported tensor: {obj.device.type} with dim {obj.dim()}. "
                    f"Only KPU tensors and 0-dim CPU scalar tensors are allowed."
                )
        elif isinstance(obj, torch.device):
            if obj.type == "kpu":
                info = device_manager.get_remote_device_info(obj.index or 0)
                return torch.device(info.device_type, info.device_index)
            return obj
        return obj

    processed_args, processed_kwargs = map_args_kwargs(process_arg, args, kwargs)

    # Collect metadata for unregistered output tensors
    output_metadata_list: list[TensorMetadata] = []
    output_tensors_to_register: list[torch.Tensor] = []
    if output_tensors:
        for tensor in output_tensors:
            if tensor is not None:
                meta = _get_tensor_metadata_if_new(tensor)
                if meta is not None:
                    output_metadata_list.append(meta)
                    output_tensors_to_register.append(tensor)

    if ENABLE_STREAMING:
        stream_manager = compute._grpc_client.stream

        request = build_execute_aten_request(
            op_name=op_name,
            args=processed_args,
            kwargs=processed_kwargs,
            output_tensors=output_tensors,
            tensor_metadata=tensor_metadata_list if tensor_metadata_list else None,
            output_metadata=output_metadata_list if output_metadata_list else None,
        )

        # FIXME: error handling
        future = stream_manager.submit_execute_aten(request)

        # Register tensors locally immediately (optimistic)
        for tensor in tensors_to_register:
            _register_tensor_locally(tensor)
        for tensor in output_tensors_to_register:
            _register_tensor_locally(tensor)

        return None  # Fire-and-forget
    else:
        client = compute._grpc_client.torch

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


def map_args_kwargs(
        func: Callable[[Any], Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """
    Apply func to all elements in args/kwargs, recursing into lists/tuples.

    The func should handle leaf values (non-containers). Recursion into
    lists and tuples is handled by this function.

    Args:
        func: Transformer function for leaf values
        args: Positional arguments to transform
        kwargs: Keyword arguments to transform

    Returns:
        Transformed (args, kwargs) tuple
    """

    def transform(obj: Any) -> Any:
        if isinstance(obj, (list, tuple)):
            return type(obj)(transform(item) for item in obj)
        return func(obj)

    return (
        tuple(transform(arg) for arg in args),
        {k: transform(v) for k, v in kwargs.items()},
    )


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
