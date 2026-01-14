"""
KPU ATen Dispatch - Meta tensor execution for shape inference.

This module provides the fallback mechanism for ATen operations on KPU devices.
It uses meta tensors to infer output shapes without moving data, then creates
output tensors on the KPU device and executes operations remotely.
"""

from typing import Any, Callable

import torch

from kpu.torch.backend._async import run_async
from kpu.torch.backend import _client


def _map_args_kwargs(
    func: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Apply func to all elements in args/kwargs, recursing into lists/tuples."""

    def map_container(container):
        if isinstance(container, (list, tuple)):
            return type(container)(func(item) for item in container)
        return func(container)

    return (
        tuple(map_container(arg) for arg in args),
        {k: map_container(v) for k, v in kwargs.items()},
    )


def _create_meta_tensor_from_kpu(
    kpu_tensor: torch.Tensor,
    meta_storage_cache: dict[torch.UntypedStorage, torch.UntypedStorage],
) -> torch.Tensor:
    """Create a meta tensor that mirrors a KPU tensor, including storage sharing."""
    original_storage = kpu_tensor.untyped_storage()

    # Create or reuse meta storage to preserve storage sharing relationships
    if original_storage not in meta_storage_cache:
        nbytes = original_storage.nbytes()
        meta_storage_cache[original_storage] = torch.UntypedStorage(
            nbytes, device="meta"
        )

    meta_storage = meta_storage_cache[original_storage]

    # Create meta tensor with same metadata as KPU tensor
    meta_tensor = torch.empty(0, dtype=kpu_tensor.dtype, device="meta")
    meta_tensor.set_(
        meta_storage,
        kpu_tensor.storage_offset(),
        kpu_tensor.shape,
        kpu_tensor.stride(),
    )

    return meta_tensor


def _execute_meta_operation(
    op: torch._ops.OpOverload,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    device_container: list[torch.device],
) -> tuple[Any, dict]:
    """Execute operation on meta tensors for shape inference and device resolution."""
    original_tensors: dict[torch.UntypedStorage, torch.Tensor] = {}
    meta_storage_cache: dict[torch.UntypedStorage, torch.UntypedStorage] = {}

    if "device" in kwargs:
        device_container.append(kwargs["device"])

    def to_meta_tensor(obj):
        # Check tensor device if container still empty
        if not device_container and isinstance(obj, torch.Tensor):
            if obj.device.type == "kpu":
                device_container.append(obj.device)

        # Convert tensor to meta for shape inference
        if isinstance(obj, torch.Tensor):
            # Validate device type: must be KPU or CPU scalar (0-dim)
            if obj.device.type != "kpu":
                if obj.device.type == "cpu" and obj.dim() == 0:
                    # CPU scalar tensors are allowed - pass through as-is
                    return obj
                else:
                    raise RuntimeError(
                        f"Cannot mix {obj.device.type} tensors with kpu tensors. "
                        f"Only 0-dimensional CPU scalar tensors are allowed. "
                        f"Please move your tensor to the kpu device first."
                    )

            meta_tensor = _create_meta_tensor_from_kpu(obj, meta_storage_cache)
            original_tensors[meta_tensor.untyped_storage()] = obj
            return meta_tensor

        # Convert device arguments to meta device
        if isinstance(obj, torch.device):
            return torch.device("meta")

        return obj

    meta_args, meta_kwargs = _map_args_kwargs(to_meta_tensor, args, kwargs)
    meta_result = op(*meta_args, **meta_kwargs)

    return meta_result, original_tensors


def _create_output_tensors(
    meta_outputs: list,
    original_tensors: dict[torch.UntypedStorage, torch.Tensor],
    kpu_device: torch.device,
) -> list[torch.Tensor | None]:
    """Create output tensors based on meta execution results with proper alias detection."""
    output_tensors: list[torch.Tensor | None] = []

    for meta_output in meta_outputs:
        # Handle None outputs (common in backward operations)
        if meta_output is None:
            output_tensors.append(None)
            continue

        meta_storage = meta_output.untyped_storage()

        if meta_storage in original_tensors:
            # This output uses storage from an existing tensor (view/alias)
            original_tensor = original_tensors[meta_storage]

            # Resize if the original tensor is uninitialized and output has data
            if original_tensor.numel() == 0 and meta_output.numel() > 0:
                original_tensor.resize_(meta_output.shape)

            tensor = original_tensor.as_strided(
                meta_output.shape,
                meta_output.stride(),
                meta_output.storage_offset(),
            )
            output_tensors.append(tensor)
        else:
            # Create new tensor with new storage
            tensor = torch.empty_strided(
                meta_output.shape,
                meta_output.stride(),
                dtype=meta_output.dtype,
                device=kpu_device,
            )
            # Record the storage mapping for future outputs that might alias
            original_tensors[meta_storage] = tensor
            output_tensors.append(tensor)

    return output_tensors


def _execute_with_static_outputs(
    op: torch._ops.OpOverload,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    kpu_device: torch.device,
    meta_result: Any,
    original_tensors: dict[torch.UntypedStorage, torch.Tensor],
) -> Any:
    """Execute operation using meta tensors for shape inference."""
    # Normalize meta_result to list
    if isinstance(meta_result, torch.Tensor):
        meta_outputs = [meta_result]
    elif isinstance(meta_result, (tuple, list)):
        meta_outputs = list(meta_result)
    else:
        meta_outputs = []

    # Create output tensors based on meta shapes
    output_tensors = (
        _create_output_tensors(meta_outputs, original_tensors, kpu_device)
        if meta_outputs
        else []
    )

    # Execute operation remotely via gRPC
    non_none_outputs = [t for t in output_tensors if t is not None]
    if non_none_outputs:
        # Collect input KPU tensors from original_tensors
        input_tensors = [
            t for t in original_tensors.values() if t.device.type == "kpu"
        ]

        if input_tensors:
            run_async(
                _client.execute_aten_operation(
                    op_name=str(op),
                    input_tensors=input_tensors,
                    output_tensors=non_none_outputs,
                    kwargs=kwargs if kwargs else None,
                )
            )

    # Return results
    if len(output_tensors) > 1:
        return tuple(output_tensors)
    elif output_tensors:
        return output_tensors[0]
    else:
        return None


def _kpu_kernel_fallback(
    op: torch._ops.OpOverload,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Execute PyTorch operations on KPU devices using meta tensor dispatch."""
    device_container: list[torch.device] = []

    try:
        meta_result, original_tensors = _execute_meta_operation(
            op, args, kwargs, device_container
        )

        if not device_container:
            # No KPU device found in args, likely a factory function
            # Try to get device from first tensor output
            if isinstance(meta_result, torch.Tensor):
                device_container.append(torch.device("kpu", 0))
            else:
                raise RuntimeError(
                    f"Could not determine KPU device for operation {op}"
                )

        return _execute_with_static_outputs(
            op, args, kwargs, device_container[0], meta_result, original_tensors
        )
    except NotImplementedError:
        # Meta execution not implemented for this operation
        raise NotImplementedError(
            f"Operation {op} is not supported on KPU device. "
            f"Meta tensor execution failed."
        )
