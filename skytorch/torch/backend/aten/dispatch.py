"""
SkyTorch ATen Dispatch - Meta tensor execution for shape inference.

This module provides the fallback mechanism for ATen operations on SkyTorch devices.
It uses meta tensors to infer output shapes without moving data, then creates
output tensors on the SkyTorch device and executes operations remotely.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch._dynamo  # Pre-import to avoid circular import issues during debugging

from skytorch.torch.backend import _client
from skytorch.torch.backend._client import map_args_kwargs
from skytorch.torch.profiler import PROFILING_ENABLED

logger = logging.getLogger(__name__)

# C++ _compute_dispatch_context is ~12x faster at computing cache keys than
# the Python _build_cache_key_with_context, but it fills the gRPC streaming
# pipeline faster. This increases latency at synchronous scalar fetch points
# (_local_scalar_dense), resulting in a net regression for workloads with
# frequent sync points (e.g., LLM inference with model.generate).
# Disabled until the streaming client has proper flow control/backpressure.
# See: https://github.com/astefanutti/skytorch/issues/XXX

# try:
#     from skytorch.torch.backend._C import _compute_dispatch_context
# except ImportError:
#     _compute_dispatch_context = None

_compute_dispatch_context = None

_SHAPE_CACHE_MAX_SIZE = 4096
_sky_device_cache: dict[int, torch.device] = {}


@dataclass(frozen=True, slots=True)
class _OutputMeta:
    """Cached metadata for a single output tensor."""

    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    storage_offset: int
    # Index into input tensor list if this output aliases an input, else -1
    alias_input: int


# Cache: op signature -> list of output metadata
# Key is int (C++ 64-bit hash) when C++ extension available, else tuple (Python)
_shape_cache: dict = {}


_UNCACHEABLE = object()

# Cache for str(op) conversion — avoids ~1μs per call (called twice on cache hit)
_op_name_cache: dict = {}


def _get_op_name(op: torch._ops.OpOverload) -> str:
    try:
        return _op_name_cache[op]
    except KeyError:
        name = str(op)
        _op_name_cache[op] = name
        return name


def _build_cache_key_with_context(
    op: torch._ops.OpOverload,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[tuple | None, torch.device | None, list[torch.Tensor]]:
    """Build cache key, resolve sky device, and collect input tensors in one pass.

    Returns (cache_key, sky_device, input_tensors).
    cache_key is None if args contain unhashable types.
    sky_device is the first sky device found in args/kwargs.
    input_tensors is a list of unique sky tensors (by storage ptr).
    """
    parts: list = [_get_op_name(op)]
    sky_device: torch.device | None = None
    input_tensors: list[torch.Tensor] = []
    seen_ptrs: set[int] = set()

    # Track storage sharing: data_ptr -> group ID
    storage_groups: dict[int, int] = {}
    next_group = 0
    uncacheable = False

    def _process(obj: Any) -> Any:
        nonlocal sky_device, next_group, uncacheable
        if isinstance(obj, torch.Tensor):
            if obj.device.type == "sky":
                if sky_device is None:
                    sky_device = obj.device
                storage = obj.untyped_storage()
                ptr = storage.data_ptr()
                if ptr not in seen_ptrs:
                    seen_ptrs.add(ptr)
                    input_tensors.append(obj)
                if ptr not in storage_groups:
                    storage_groups[ptr] = next_group
                    next_group += 1
                return (
                    "T",
                    tuple(obj.shape),
                    obj.dtype,
                    tuple(obj.stride()),
                    obj.storage_offset(),
                    storage.nbytes(),
                    storage_groups[ptr],
                )
            # Non-sky tensor (e.g., cpu scalar) — still build key
            storage = obj.untyped_storage()
            ptr = storage.data_ptr()
            if ptr not in storage_groups:
                storage_groups[ptr] = next_group
                next_group += 1
            return (
                "T",
                tuple(obj.shape),
                obj.dtype,
                tuple(obj.stride()),
                obj.storage_offset(),
                storage.nbytes(),
                storage_groups[ptr],
            )
        if isinstance(obj, (list, tuple)):
            return tuple(_process(v) for v in obj)
        if isinstance(obj, torch.device):
            if obj.type == "sky" and sky_device is None:
                sky_device = obj
            return ("D", obj.type, obj.index)
        if obj is None:
            return None
        if isinstance(obj, (int, float, bool, str, torch.dtype, torch.memory_format, torch.layout)):
            return obj
        uncacheable = True
        return _UNCACHEABLE

    # Check kwargs device first (like _resolve_device did)
    if "device" in kwargs and isinstance(kwargs["device"], torch.device):
        if kwargs["device"].type == "sky":
            sky_device = kwargs["device"]

    for arg in args:
        k = _process(arg)
        if uncacheable:
            return None, sky_device, input_tensors
        parts.append(k)

    if kwargs:
        for key in sorted(kwargs):
            k = _process(kwargs[key])
            if uncacheable:
                return None, sky_device, input_tensors
            parts.append((key, k))

    return tuple(parts), sky_device, input_tensors


def _create_meta_tensor_from_sky(
    sky_tensor: torch.Tensor,
    meta_storage_cache: dict[torch.UntypedStorage, torch.UntypedStorage],
) -> torch.Tensor:
    """Create a meta tensor that mirrors a sky tensor, including storage sharing."""
    original_storage = sky_tensor.untyped_storage()

    # Create or reuse meta storage to preserve storage sharing relationships
    if original_storage not in meta_storage_cache:
        nbytes = original_storage.nbytes()
        meta_storage_cache[original_storage] = torch.UntypedStorage(
            nbytes, device="meta"
        )

    meta_storage = meta_storage_cache[original_storage]

    # Create meta tensor with same metadata as sky tensor
    meta_tensor = torch.empty(0, dtype=sky_tensor.dtype, device="meta")
    meta_tensor.set_(
        meta_storage,
        sky_tensor.storage_offset(),
        sky_tensor.shape,
        sky_tensor.stride(),
    )

    return meta_tensor


def _execute_meta_operation(
    op: torch._ops.OpOverload,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    devices: list[torch.device],
) -> tuple[Any, dict, bool]:
    """Execute operation on meta tensors for shape inference and device resolution."""
    original_tensors: dict[torch.UntypedStorage, torch.Tensor] = {}
    meta_storage_cache: dict[torch.UntypedStorage, torch.UntypedStorage] = {}
    has_empty_cpu = False

    if "device" in kwargs:
        devices.append(kwargs["device"])

    def to_meta_tensor(obj):
        nonlocal has_empty_cpu

        if isinstance(obj, torch.Tensor) and obj.device.type == "sky":
            devices.append(obj.device)

        # Convert tensor to meta for shape inference
        if isinstance(obj, torch.Tensor):
            # Validate device type: must be sky or cpu scalar (0-dim)
            if obj.device.type != "sky":
                if obj.device.type == "cpu" and obj.dim() == 0:
                    # cpu scalar tensors are allowed - pass through as-is
                    return obj
                elif obj.device.type == "cpu" and obj.numel() == 0:
                    # empty cpu tensors are allowed - promote to meta for shape inference
                    has_empty_cpu = True
                    return torch.empty(obj.shape, dtype=obj.dtype, device="meta")
                else:
                    raise RuntimeError(
                        f"Cannot mix {obj.device.type} tensors with sky tensors "
                        f"in {op}. "
                        f"Got {obj.device.type} tensor with shape={list(obj.shape)}, "
                        f"dtype={obj.dtype}. "
                        f"Only 0-dimensional cpu scalar tensors are allowed. "
                        f"Please move your tensor to the sky device first."
                    )

            meta_tensor = _create_meta_tensor_from_sky(obj, meta_storage_cache)
            original_tensors[meta_tensor.untyped_storage()] = obj
            return meta_tensor

        # Convert device arguments to meta device
        if isinstance(obj, torch.device):
            return torch.device("meta")

        return obj

    meta_args, meta_kwargs = map_args_kwargs(to_meta_tensor, args, kwargs)
    meta_result = op(*meta_args, **meta_kwargs)

    return meta_result, original_tensors, has_empty_cpu


def _create_output_tensors(
    meta_outputs: list,
    original_tensors: dict[torch.UntypedStorage, torch.Tensor],
    sky_device: torch.device,
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
                device=sky_device,
            )
            # Record the storage mapping for future outputs that might alias
            original_tensors[meta_storage] = tensor
            output_tensors.append(tensor)

    return output_tensors


def _execute_with_static_outputs(
    op: torch._ops.OpOverload,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    sky_device: torch.device,
    meta_result: Any,
    original_tensors: dict[torch.UntypedStorage, torch.Tensor],
    cache_key: Any = None,
    input_tensors: list[torch.Tensor] | None = None,
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
        _create_output_tensors(meta_outputs, original_tensors, sky_device)
        if meta_outputs
        else []
    )

    # Populate the shape cache
    if cache_key is not None and input_tensors is not None:
        _populate_cache(cache_key, meta_outputs, original_tensors, input_tensors)

    # Execute operation remotely via gRPC
    _client.execute_aten_operation(
        sky_device=sky_device,
        op_name=_get_op_name(op),
        args=args,
        kwargs=kwargs,
        output_tensors=output_tensors,
    )

    # Return results
    if len(output_tensors) > 1:
        return tuple(output_tensors)
    elif output_tensors:
        return output_tensors[0]
    else:
        return None


def _populate_cache(
    cache_key: Any,
    meta_outputs: list,
    original_tensors: dict[torch.UntypedStorage, torch.Tensor],
    input_tensors: list[torch.Tensor],
) -> None:
    """Store shape inference results in cache."""
    if len(_shape_cache) >= _SHAPE_CACHE_MAX_SIZE:
        return

    # Build storage -> input index mapping for alias detection
    storage_to_input: dict[int, int] = {}
    for i, t in enumerate(input_tensors):
        ptr = t.untyped_storage().data_ptr()
        if ptr not in storage_to_input:
            storage_to_input[ptr] = i

    cached_outputs: list[_OutputMeta | None] = []
    for meta_output in meta_outputs:
        if meta_output is None:
            cached_outputs.append(None)
            continue

        meta_storage = meta_output.untyped_storage()
        alias_input = -1

        if meta_storage in original_tensors:
            original = original_tensors[meta_storage]
            ptr = original.untyped_storage().data_ptr()
            alias_input = storage_to_input.get(ptr, -1)

        cached_outputs.append(
            _OutputMeta(
                shape=tuple(meta_output.shape),
                stride=tuple(meta_output.stride()),
                dtype=meta_output.dtype,
                storage_offset=meta_output.storage_offset(),
                alias_input=alias_input,
            )
        )

    _shape_cache[cache_key] = cached_outputs


def _create_outputs_from_cache(
    cached_outputs: list[_OutputMeta | None],
    input_tensors: list[torch.Tensor],
    sky_device: torch.device,
) -> list[torch.Tensor | None]:
    """Create output tensors from cached shape metadata."""
    output_tensors: list[torch.Tensor | None] = []

    for meta in cached_outputs:
        if meta is None:
            output_tensors.append(None)
            continue

        if meta.alias_input >= 0:
            # Output aliases an input tensor
            original = input_tensors[meta.alias_input]

            if original.numel() == 0 and len(meta.shape) > 0 and all(s > 0 for s in meta.shape):
                original.resize_(meta.shape)

            tensor = original.as_strided(meta.shape, meta.stride, meta.storage_offset)
            output_tensors.append(tensor)
        else:
            # New storage
            tensor = torch.empty_strided(
                meta.shape,
                meta.stride,
                dtype=meta.dtype,
                device=sky_device,
            )
            output_tensors.append(tensor)

    return output_tensors


def _find_offending_arg(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> str | None:
    """Find the argument location of a non-sky, non-scalar tensor.

    Scans args and kwargs (recursing into lists/tuples) and returns a
    human-readable path like ``args[1]`` or ``kwargs['mask'][0]`` for
    the first tensor that is not on the sky device and not a 0-dim cpu
    scalar.  Returns ``None`` if no offending tensor is found.

    Only called on the error path so performance is not a concern.
    """

    def _scan(obj: Any, path: str) -> str | None:
        if isinstance(obj, torch.Tensor):
            if obj.device.type != "sky" and not (obj.device.type == "cpu" and obj.dim() == 0):
                return path
        elif isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                result = _scan(item, f"{path}[{i}]")
                if result is not None:
                    return result
        return None

    for i, arg in enumerate(args):
        result = _scan(arg, f"args[{i}]")
        if result is not None:
            return result

    for key, val in kwargs.items():
        result = _scan(val, f"kwargs['{key}']")
        if result is not None:
            return result

    return None


def _collect_input_tensors(args: tuple[Any, ...], kwargs: dict[str, Any]) -> list[torch.Tensor]:
    """Collect all sky tensors from args/kwargs in order (for cache alias tracking)."""
    tensors: list[torch.Tensor] = []
    seen: set[int] = set()

    def _collect(obj: Any) -> None:
        if isinstance(obj, torch.Tensor) and obj.device.type == "sky":
            ptr = obj.untyped_storage().data_ptr()
            if ptr not in seen:
                seen.add(ptr)
                tensors.append(obj)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                _collect(v)

    for arg in args:
        _collect(arg)
    for val in kwargs.values():
        _collect(val)
    return tensors


def _sky_kernel_fallback(
    op: torch._ops.OpOverload,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Execute PyTorch operations on SkyTorch devices using meta tensor dispatch."""
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Dispatching {op}")

    try:
        op_name = _get_op_name(op)

        if PROFILING_ENABLED:
            from skytorch.torch.profiler import ClientProfiler

            _prof = ClientProfiler.get()
            _t0 = time.perf_counter_ns()
            if _prof.last_dispatch_end > 0:
                _prof.inter_op_gap.add(_t0 - _prof.last_dispatch_end)
            if _prof.first_dispatch_ns == 0:
                _prof.first_dispatch_ns = _t0
            _prof.total_ops += 1

        # Compute cache key, resolve device, and collect tensors
        if _compute_dispatch_context is not None:
            # C++ fast path: single walk of args/kwargs
            cache_hash, input_tensors, sky_device_idx = _compute_dispatch_context(
                op_name, args, kwargs
            )
            cache_key = cache_hash if cache_hash != 0 else None
            if sky_device_idx >= 0:
                sky_device = _sky_device_cache.get(sky_device_idx)
                if sky_device is None:
                    sky_device = torch.device("sky", sky_device_idx)
                    _sky_device_cache[sky_device_idx] = sky_device
            else:
                sky_device = None
        else:
            # Python fallback
            cache_key, sky_device, input_tensors = _build_cache_key_with_context(
                op, args, kwargs
            )

        if PROFILING_ENABLED:
            _t1 = time.perf_counter_ns()
            _prof.cache_key_build.add(_t1 - _t0)

        # Try shape cache first
        if cache_key is not None:
            cached = _shape_cache.get(cache_key)
            if cached is not None and sky_device is not None:
                output_tensors = _create_outputs_from_cache(
                    cached, input_tensors, sky_device
                )

                if PROFILING_ENABLED:
                    _t2 = time.perf_counter_ns()
                    _prof.output_creation.add(_t2 - _t1)
                    _prof.cache_hits += 1

                _client.execute_aten_operation(
                    sky_device=sky_device,
                    op_name=op_name,
                    args=args,
                    kwargs=kwargs,
                    output_tensors=output_tensors,
                )

                if PROFILING_ENABLED:
                    _t3 = time.perf_counter_ns()
                    _prof.execute_dispatch.add(_t3 - _t2)
                    _prof.last_dispatch_end = _t3
                    _prof.last_dispatch_ns = _t3

                if len(output_tensors) > 1:
                    return tuple(output_tensors)
                elif output_tensors:
                    return output_tensors[0]
                else:
                    return None

        if PROFILING_ENABLED:
            _prof.cache_misses += 1

        # Cache miss — full meta execution
        devices: list[torch.device] = []

        meta_result, original_tensors, has_empty_cpu = _execute_meta_operation(
            op, args, kwargs, devices
        )

        if not devices:
            raise RuntimeError(f"Could not determine sky device for operation {op}")

        sky_device = devices[0]

        if has_empty_cpu:

            def _promote(obj):
                if (
                    isinstance(obj, torch.Tensor)
                    and obj.device.type == "cpu"
                    and obj.numel() == 0
                    and obj.dim() > 0
                ):
                    return torch.empty(obj.shape, dtype=obj.dtype, device=sky_device)
                return obj

            args, kwargs = map_args_kwargs(_promote, args, kwargs)
            # Re-collect input tensors after promotion
            input_tensors = (
                _collect_input_tensors(args, kwargs) if cache_key is not None else None
            )
        else:
            input_tensors = input_tensors if cache_key is not None else None

        result = _execute_with_static_outputs(
            op, args, kwargs, sky_device, meta_result, original_tensors,
            cache_key=cache_key, input_tensors=input_tensors,
        )

        if PROFILING_ENABLED:
            _tend = time.perf_counter_ns()
            _prof.last_dispatch_end = _tend
            _prof.last_dispatch_ns = _tend

        return result
    except RuntimeError as e:
        if "Cannot mix" in str(e):
            location = _find_offending_arg(args, kwargs)
            if location:
                raise RuntimeError(f"{e} ({location})") from None
        raise
    except NotImplementedError:
        # Meta execution not implemented for this operation
        raise NotImplementedError(
            f"Operation {op} is not supported on sky device. " f"Meta tensor execution failed."
        )
