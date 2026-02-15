"""
SkyTorch Client operations - Tensor transfer and remote execution.

This module provides functions for tensor data transfer,
remote ATen operation execution via gRPC, and Compute resolution.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Optional

import torch

# Feature flags
ENABLE_STREAMING = os.environ.get("SKYTORCH_ENABLE_STREAMING", "1") == "1"
ENABLE_CPP_REQUEST_BUILDER = os.environ.get("SKYTORCH_CPP_REQUEST_BUILDER", "1") == "1"

from skytorch.torch.server.serialization import (
    tensor_from_bytes,
    tensor_to_bytes,
)

from skytorch.client import Compute
from skytorch.torch.backend._device import device_manager
from skytorch.torch.backend._storage import storage_manager
from skytorch.torch.client.tensor import get_storage_id, get_tensor_id, get_tensor_metadata
from skytorch.torch.client.metadata import TensorMetadata
from skytorch.torch.client.service import TensorClient
from skytorch.torch.backend._async import run_async
from skytorch.torch.client.request import (
    tensor_metadata_to_proto,
    build_copy_tensor_request,
    build_execute_aten_request,
)
from skytorch.torch.profiler import PROFILING_ENABLED


def copy_sky_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    """
    Copy data from a sky tensor to a cpu tensor.

    Uses streaming get_tensor or unary RPC depending on ENABLE_STREAMING.

    Args:
        tensor: Source sky tensor

    Returns:
        cpu tensor with copied data
    """
    return run_async(_copy_sky_to_cpu_async(tensor)).result()


async def _copy_sky_to_cpu_async(tensor: torch.Tensor) -> torch.Tensor:
    """Async implementation of copy_sky_to_cpu."""
    from skytorch.torch.server import service_pb2

    if tensor.device.type != "sky":
        raise ValueError("copy_sky_to_cpu requires a sky tensor")

    compute = _require_compute(tensor)

    # Get metadata for auto-creation if tensor is not registered
    meta = _get_tensor_metadata_if_new(tensor)

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
        if meta is not None:
            request.metadata.CopyFrom(tensor_metadata_to_proto(meta))

        future = stream_manager.submit_get_tensor(request)
        response = await future

        if not response.success:
            raise RuntimeError(f"Failed to get tensor: {response.error_message}")

        # Register locally after successful response
        if meta is not None:
            _register_tensor_locally(tensor)

        # Deserialize tensor data
        dtype = eval(response.get_tensor.dtype)
        shape = list(response.get_tensor.shape)
        data = response.get_tensor.data

        return tensor_from_bytes(data, dtype, shape)

    else:
        client = _require_client(compute)
        cpu_tensor = await client.get_tensor(
            tensor_id=get_tensor_id(tensor),
            shape=tuple(tensor.shape),
            dtype=tensor.dtype,
            stride=tuple(tensor.stride()),
            storage_offset=tensor.storage_offset(),
            tensor_metadata=meta,
        )

        # Register locally after successful response
        if meta is not None:
            _register_tensor_locally(tensor)

        return cpu_tensor


def copy_cpu_to_sky(src: torch.Tensor, dst: torch.Tensor) -> None:
    """
    Copy data from a cpu tensor to a sky tensor.

    When streaming is enabled, the update_tensor goes through the stream
    ensuring proper ordering with other operations.

    Args:
        src: Source cpu tensor
        dst: Destination sky tensor
    """
    from skytorch.torch.server import service_pb2

    if dst.device.type != "sky":
        raise ValueError("copy_cpu_to_sky requires a sky target tensor")
    if src.device.type != "cpu":
        raise ValueError("copy_cpu_to_sky requires a cpu source tensor")

    compute = _require_compute(dst)

    # Get metadata for auto-creation if tensor is not registered
    meta = _get_tensor_metadata_if_new(dst)

    # Cast to destination dtype if needed (copy_ semantics require dtype casting)
    src = src.to(dst.dtype) if src.dtype != dst.dtype else src

    if ENABLE_STREAMING:
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

        stream_manager.submit_update_tensor(request)

    else:
        run_async(_copy_cpu_to_sky_unary(compute, src, dst, meta)).result()

    # Register locally after RPC
    if meta is not None:
        _register_tensor_locally(dst)


async def _copy_cpu_to_sky_unary(
    compute: Compute,
    src: torch.Tensor,
    dst: torch.Tensor,
    meta: Optional[TensorMetadata],
) -> None:
    """Async helper for unary copy_cpu_to_sky RPC."""
    client = _require_client(compute)
    await client.update_tensor(
        src=src,
        tensor_id=get_tensor_id(dst),
        tensor_metadata=meta,
    )


def copy_sky_to_sky(src: torch.Tensor, dst: torch.Tensor) -> None:
    """
    Copy data between sky tensors on the same Compute.

    Args:
        src: Source sky tensor
        dst: Destination sky tensor

    Raises:
        ValueError: If tensors are not sky tensors
        RuntimeError: If no Compute context is available or tensors
            are on different Computes
    """
    if src.device.type != "sky" or dst.device.type != "sky":
        raise ValueError("copy_sky_to_sky requires sky tensors")

    src_compute = _resolve_compute(src)
    dst_compute = _resolve_compute(dst)

    if src_compute is None or dst_compute is None:
        raise RuntimeError(
            "Cannot copy between sky tensors without Compute context. "
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
        request = build_copy_tensor_request(
            src_tensor_id=get_tensor_id(src),
            dst_tensor_id=get_tensor_id(dst),
            src_offset=src.storage_offset() * src.element_size(),
            dst_offset=dst.storage_offset() * dst.element_size(),
            num_bytes=src.numel() * src.element_size(),
            src_metadata=src_meta,
            dst_metadata=dst_meta,
        )
        stream_manager = src_compute._grpc_client.stream
        stream_manager.submit_copy_tensor(request)
    else:
        run_async(
            _copy_sky_to_sky_unary(
                src_compute,
                src,
                dst,
                src_meta,
                dst_meta,
            )
        ).result()

    # Register locally after RPC
    if src_meta is not None:
        _register_tensor_locally(src)
    if dst_meta is not None:
        _register_tensor_locally(dst)


async def _copy_sky_to_sky_unary(
    compute: Compute,
    src: torch.Tensor,
    dst: torch.Tensor,
    src_meta: Optional[TensorMetadata],
    dst_meta: Optional[TensorMetadata],
) -> None:
    """Async helper for unary copy_sky_to_sky RPC."""
    client = _require_client(compute)
    await client.copy_tensor(
        src_tensor_id=get_tensor_id(src),
        dst_tensor_id=get_tensor_id(dst),
        src_offset=src.storage_offset() * src.element_size(),
        dst_offset=dst.storage_offset() * dst.element_size(),
        num_bytes=src.numel() * src.element_size(),
        src_metadata=src_meta,
        dst_metadata=dst_meta,
    )


async def get_scalar(compute: Compute, tensor_id: int, metadata=None):
    """
    Get a scalar value from a remote tensor via streaming GetScalar RPC.

    Args:
        compute: The Compute instance
        tensor_id: ID of the tensor to get scalar from
        metadata: Optional TensorMetadata proto for auto-creation

    Returns:
        Python scalar value (int, float, or bool)
    """
    from skytorch.torch.server import service_pb2

    stream_manager = compute._grpc_client.stream

    request = service_pb2.GetScalarRequest(tensor_id=tensor_id)
    if metadata is not None:
        request.tensor_metadata.append(metadata)

    response = await stream_manager.submit_get_scalar(request)

    if not response.success:
        raise RuntimeError(f"Failed to get scalar: {response.error_message}")

    scalar_response = response.get_scalar
    scalar_type = scalar_response.WhichOneof("value")
    if scalar_type == "float_value":
        return scalar_response.float_value
    elif scalar_type == "int_value":
        return scalar_response.int_value
    elif scalar_type == "bool_value":
        return scalar_response.bool_value
    else:
        raise RuntimeError("GetScalar response has no value set")


async def delete_tensors(compute: Compute, tensor_ids: list[int]) -> None:
    """
    Delete tensors on the remote server.

    When streaming is enabled, routes through the stream to maintain
    ordering with other operations.

    Args:
        compute: The Compute instance
        tensor_ids: List of tensor IDs to delete
    """
    from skytorch.torch.server import service_pb2

    if ENABLE_STREAMING:
        stream_manager = compute._grpc_client.stream
        request = service_pb2.DeleteTensorsRequest(tensor_ids=tensor_ids)
        stream_manager.submit_delete_tensors(request)
    else:
        client = _require_client(compute)
        await client.delete_tensors(tensor_ids)


def execute_aten_operation(
    sky_device: torch.device,
    op_name: str,
    args: tuple,
    kwargs: dict,
    output_tensors: list[torch.Tensor] | None,
) -> None:
    """
    Execute an ATen operation on the remote Compute.

    Uses streaming (fire-and-forget) when ENABLE_STREAMING is True,
    otherwise uses unary RPC.

    When ENABLE_CPP_REQUEST_BUILDER is True and streaming is enabled,
    uses the C++ binary request builder for faster serialization.

    Args:
        sky_device: sky device to execute on
        op_name: ATen operation name (e.g., "aten::add.Tensor")
        args: Positional arguments (may contain sky tensors)
        kwargs: Keyword arguments (may contain sky tensors)
        output_tensors: Pre-allocated output tensors, or None for server-created
    """
    device_index = sky_device.index or 0
    remote_info = device_manager._local_to_remote.get(device_index)
    if remote_info is None:
        raise RuntimeError(
            "No Compute context available for ATen operation. "
            "Ensure you are within an 'async with Compute(...):' block."
        )
    compute = remote_info.compute

    if compute._grpc_client is None:
        raise RuntimeError(
            f"Compute '{compute.name}' is not ready. " "The gRPC client has not been initialized."
        )

    if ENABLE_STREAMING and ENABLE_CPP_REQUEST_BUILDER:
        _execute_aten_cpp_fast_path(
            compute, sky_device, op_name, args, kwargs, output_tensors, remote_info
        )
    elif ENABLE_STREAMING:
        _execute_aten_python_path(compute, sky_device, op_name, args, kwargs, output_tensors)
    else:
        _execute_aten_unary_path(compute, sky_device, op_name, args, kwargs, output_tensors)


def _execute_aten_cpp_fast_path(
    compute,
    sky_device: torch.device,
    op_name: str,
    args: tuple,
    kwargs: dict,
    output_tensors: list[torch.Tensor] | None,
    remote_info=None,
) -> None:
    """Fast path: C++ binary serialization + raw bytes submission."""
    from skytorch.torch.backend._C import _build_execute_aten_request

    if remote_info is None:
        remote_info = device_manager.get_remote_device_info(sky_device.index or 0)

    if PROFILING_ENABLED:
        _t0 = time.perf_counter_ns()

    # C++ builds binary request + identifies new tensors
    raw_bytes, new_tensor_ids, new_storage_ids = _build_execute_aten_request(
        op_name,
        args,
        kwargs,
        output_tensors,
        sky_device.index or 0,
        remote_info.device_type,
        remote_info.device_index,
    )

    if PROFILING_ENABLED:
        _t1 = time.perf_counter_ns()

    stream_manager = compute._grpc_client.stream
    stream_manager.submit_execute_aten_bytes(raw_bytes)

    if PROFILING_ENABLED:
        from skytorch.torch.profiler import ClientProfiler

        _t2 = time.perf_counter_ns()
        _prof = ClientProfiler.get()
        _prof.cpp_serialization.add(_t1 - _t0)
        _prof.event_loop_submit.add(_t2 - _t1)

    # Register new tensors locally (C++ already updated its tracking set)
    for tensor_id, storage_id in zip(new_tensor_ids, new_storage_ids):
        storage_manager.register_storage(
            storage_id=storage_id,
            nbytes=0,  # Actual nbytes tracked by server
            device_index=sky_device.index or 0,
        )
        # Register tensor_id → storage mapping
        storage_manager._storage_to_tensors[storage_id].add(tensor_id)


def _execute_aten_python_path(
    compute,
    sky_device: torch.device,
    op_name: str,
    args: tuple,
    kwargs: dict,
    output_tensors: list[torch.Tensor] | None,
) -> None:
    """Python path: protobuf serialization via build_execute_aten_request."""
    # Collect metadata for unregistered input tensors
    tensor_metadata_list: list[TensorMetadata] = []
    tensors_to_register: list[torch.Tensor] = []

    def process_arg(obj):
        """Process an argument: collect metadata and map devices."""
        if isinstance(obj, torch.Tensor):
            if obj.device.type == "sky":
                meta = _get_tensor_metadata_if_new(obj)
                if meta is not None:
                    tensor_metadata_list.append(meta)
                    tensors_to_register.append(obj)
                return obj
            elif obj.device.type == "cpu" and obj.dim() == 0:
                return obj
            elif obj.device.type == "cpu" and obj.numel() == 0:
                promoted = torch.empty(obj.shape, dtype=obj.dtype, device=sky_device)
                meta = _get_tensor_metadata_if_new(promoted)
                if meta is not None:
                    tensor_metadata_list.append(meta)
                    tensors_to_register.append(promoted)
                return promoted
            else:
                raise ValueError(
                    f"Unsupported tensor: {obj.device.type} with dim {obj.dim()}. "
                    f"Only sky tensors and 0-dim cpu scalar tensors are allowed."
                )
        elif isinstance(obj, torch.device):
            if obj.type == "sky":
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

    request = build_execute_aten_request(
        op_name=op_name,
        args=processed_args,
        kwargs=processed_kwargs,
        output_tensors=output_tensors,
        tensor_metadata=tensor_metadata_list if tensor_metadata_list else None,
        output_metadata=output_metadata_list if output_metadata_list else None,
    )
    stream_manager = compute._grpc_client.stream
    stream_manager.submit_execute_aten(request)

    # Register tensors locally after RPC
    for tensor in tensors_to_register:
        _register_tensor_locally(tensor)
    for tensor in output_tensors_to_register:
        _register_tensor_locally(tensor)


def _execute_aten_unary_path(
    compute,
    sky_device: torch.device,
    op_name: str,
    args: tuple,
    kwargs: dict,
    output_tensors: list[torch.Tensor] | None,
) -> None:
    """Unary RPC path (non-streaming fallback)."""
    # Collect metadata for unregistered input tensors
    tensor_metadata_list: list[TensorMetadata] = []
    tensors_to_register: list[torch.Tensor] = []

    def process_arg(obj):
        """Process an argument: collect metadata and map devices."""
        if isinstance(obj, torch.Tensor):
            if obj.device.type == "sky":
                meta = _get_tensor_metadata_if_new(obj)
                if meta is not None:
                    tensor_metadata_list.append(meta)
                    tensors_to_register.append(obj)
                return obj
            elif obj.device.type == "cpu" and obj.dim() == 0:
                return obj
            elif obj.device.type == "cpu" and obj.numel() == 0:
                promoted = torch.empty(obj.shape, dtype=obj.dtype, device=sky_device)
                meta = _get_tensor_metadata_if_new(promoted)
                if meta is not None:
                    tensor_metadata_list.append(meta)
                    tensors_to_register.append(promoted)
                return promoted
            else:
                raise ValueError(
                    f"Unsupported tensor: {obj.device.type} with dim {obj.dim()}. "
                    f"Only sky tensors and 0-dim cpu scalar tensors are allowed."
                )
        elif isinstance(obj, torch.device):
            if obj.type == "sky":
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

    run_async(
        _execute_aten_operation_unary(
            compute,
            op_name,
            processed_args,
            processed_kwargs,
            output_tensors,
            tensor_metadata_list,
            output_metadata_list,
        )
    ).result()

    # Register tensors locally after RPC
    for tensor in tensors_to_register:
        _register_tensor_locally(tensor)
    for tensor in output_tensors_to_register:
        _register_tensor_locally(tensor)


async def _execute_aten_operation_unary(
    compute: Compute,
    op_name: str,
    args: tuple,
    kwargs: dict,
    output_tensors: list[torch.Tensor] | None,
    tensor_metadata: list[TensorMetadata],
    output_metadata: list[TensorMetadata],
) -> None:
    """Async helper for unary execute_aten_operation RPC."""
    client = compute._grpc_client.torch
    await client.execute_aten_operation(
        op_name=op_name,
        args=args,
        kwargs=kwargs,
        output_tensors=output_tensors,
        tensor_metadata=tensor_metadata if tensor_metadata else None,
        output_metadata=output_metadata if output_metadata else None,
    )


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
        tensor: sky tensor to check

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
        tensor: sky tensor to register
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
    Resolve the Compute associated with a sky tensor.

    Resolution order:
    1. Check if the tensor's storage has an associated Compute
    2. Check the device_manager for the device index (for lazy-allocated storage)
    3. Fall back to the current context (compute_ctx)

    Args:
        tensor: A sky tensor

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
    from skytorch.client.context import compute_ctx

    return compute_ctx.get(None)


def _require_compute(tensor: torch.Tensor) -> Compute:
    """
    Resolve and require a Compute for a tensor.

    Like resolve_compute but raises if no Compute is available.

    Args:
        tensor: A sky tensor

    Returns:
        The associated Compute

    Raises:
        RuntimeError: If no Compute context is available
    """
    compute = _resolve_compute(tensor)
    if compute is None:
        raise RuntimeError(
            "No Compute context available for sky tensor operation. "
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
            f"Compute '{compute.name}' is not ready. " "The gRPC client has not been initialized."
        )
    return compute._grpc_client.torch
