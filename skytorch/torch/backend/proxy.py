"""Module-level forward proxying for Triton kernel modules.

When a model contains modules that use Triton kernels (e.g., MXFP4 quantized
MoE experts), those kernels bypass the ATen dispatcher and cannot be intercepted
by SkyTorch's PrivateUse1 dispatch. This module replaces the forward() method
of such modules with an RPC proxy that executes the forward on the server where
the real GPU tensors and Triton kernels live.

The first call for each unique input shape signature is synchronous (to learn
output shapes). Subsequent calls with the same input shapes use pre-allocated
output tensors and fire-and-forget dispatch, matching the ATen op fast path.

Usage:
    proxy_triton_modules(model, compute, model_id, ["model.layers.*.experts"])
"""

import fnmatch
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from skytorch.torch.backend._C import _create_remote_tensor
from skytorch.torch.backend._storage import storage_manager
from skytorch.torch.client.tensor import get_tensor_id
from skytorch.torch.server import service_pb2

if TYPE_CHECKING:
    from skytorch.torch.client.stream import StreamManager

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _OutputSpec:
    """Cached output tensor specification from a module forward call."""

    shape: tuple[int, ...]
    dtype: str
    stride: tuple[int, ...]
    storage_offset: int
    storage_nbytes: int
    device_type: str
    device_index: int


def _resolve_module_paths(
    model: torch.nn.Module,
    patterns: list[str],
) -> list[str]:
    """Resolve glob patterns to concrete module paths.

    Args:
        model: The model to search for matching modules.
        patterns: List of module path patterns, possibly containing wildcards
                  (e.g., "model.layers.*.experts").

    Returns:
        Sorted list of unique concrete module paths that match any pattern.
    """
    all_paths = {name for name, _ in model.named_modules() if name}
    matched = set()
    for pattern in patterns:
        for path in all_paths:
            if fnmatch.fnmatch(path, pattern):
                matched.add(path)
    return sorted(matched)


def _input_cache_key(args: tuple) -> tuple:
    """Compute a cache key from input tensor shapes and dtypes."""
    return tuple((arg.shape, arg.dtype) for arg in args)


def _make_forward_proxy(
    module_path: str,
    model_id: int,
    stream_manager: "StreamManager",
    sky_device_index: int,
):
    """Create a proxy forward function for a module.

    The proxy uses a shape cache:
    - First call (cache miss): sync RPC, learns output shapes, caches them.
    - Subsequent calls (cache hit): pre-allocates output tensors, fire-and-forget.

    Args:
        module_path: Dotted path to the module (e.g., "layers.0.experts").
        model_id: Server-assigned model ID.
        stream_manager: StreamManager for the bidirectional stream.
        sky_device_index: Local sky device index for output tensors.

    Returns:
        A callable that replaces the module's forward().
    """
    from skytorch.torch.backend._async import run_async
    from skytorch.torch.client.request import tensor_metadata_to_proto
    from skytorch.torch.client.tensor import get_tensor_metadata

    # Shape cache: input_cache_key -> list[_OutputSpec]
    shape_cache: dict[tuple, list[_OutputSpec]] = {}

    # Persistent sentinel objects whose id() values serve as storage IDs.
    # Kept alive for the proxy's lifetime to prevent CPython address reuse
    # across calls, which would cause storage_id collisions in the storage
    # manager (multiple tensors sharing one storage → premature deletion).
    _storage_sentinels: list[object] = []

    def _collect_inputs(args, kwargs):
        if kwargs:
            raise TypeError(
                f"Proxied module '{module_path}' received keyword arguments "
                f"{list(kwargs.keys())}. Only positional sky tensor arguments "
                f"are supported. Consider proxying at a higher module level "
                f"where forward() takes only tensor inputs."
            )

        input_tensor_ids = []
        input_metadata = []
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor) and arg.device.type == "sky":
                tid = get_tensor_id(arg)
                input_tensor_ids.append(tid)
                meta = get_tensor_metadata(arg)
                input_metadata.append(tensor_metadata_to_proto(meta))
            else:
                if isinstance(arg, torch.Tensor):
                    detail = f"tensor on {arg.device}"
                else:
                    detail = type(arg).__name__
                raise TypeError(
                    f"Proxied module '{module_path}' received a non-sky-tensor "
                    f"argument at position {i}: {detail}. Only positional sky "
                    f"tensor arguments are supported. Consider proxying at a "
                    f"higher module level where forward() takes only tensor "
                    f"inputs (e.g., proxy the MLP module instead of the experts "
                    f"sub-module)."
                )
        return input_tensor_ids, input_metadata

    def _create_outputs_from_response(fwd_response):
        """Create sky tensors from sync RPC response and cache the output specs."""
        output_tensors = []
        registrations = []
        output_specs = []

        for info in fwd_response.output_tensors:
            # Always create each output tensor independently with a unique
            # client-side storage_id. Using as_strided for shared-storage
            # outputs would dispatch an ATen op to the server BEFORE the
            # RegisterTensorsRequest, causing "Tensor does not exist" errors.
            sentinel = object()
            _storage_sentinels.append(sentinel)
            client_storage_id = id(sentinel)
            sky_tensor = _create_remote_tensor(
                client_storage_id,
                list(info.shape),
                info.dtype,
                list(info.stride),
                info.storage_offset,
                info.storage_nbytes,
                sky_device_index,
            )

            tensor_id = get_tensor_id(sky_tensor)
            storage_manager.register_storage(
                client_storage_id, info.storage_nbytes, sky_device_index
            )
            storage_manager.register_tensor(sky_tensor)
            registrations.append(
                service_pb2.TensorRegistration(storage_id=info.storage_id, tensor_id=tensor_id)
            )
            output_tensors.append(sky_tensor)
            output_specs.append(
                _OutputSpec(
                    shape=tuple(info.shape),
                    dtype=info.dtype,
                    stride=tuple(info.stride),
                    storage_offset=info.storage_offset,
                    storage_nbytes=info.storage_nbytes,
                    device_type=info.device_type,
                    device_index=info.device_index,
                )
            )

        if registrations:
            stream_manager.submit_register_tensors(
                service_pb2.RegisterTensorsRequest(registrations=registrations)
            )

        return output_tensors, output_specs

    def _create_outputs_from_cache(specs: list[_OutputSpec]):
        """Pre-allocate output sky tensors from cached specs."""
        output_tensors = []
        output_tensor_ids = []
        output_metadata = []

        for spec in specs:
            sentinel = object()
            _storage_sentinels.append(sentinel)
            storage_id = id(sentinel)
            sky_tensor = _create_remote_tensor(
                storage_id,
                list(spec.shape),
                spec.dtype,
                list(spec.stride),
                spec.storage_offset,
                spec.storage_nbytes,
                sky_device_index,
            )
            storage_manager.register_storage(storage_id, spec.storage_nbytes, sky_device_index)
            storage_manager.register_tensor(sky_tensor)

            tid = get_tensor_id(sky_tensor)
            output_tensor_ids.append(tid)
            output_tensors.append(sky_tensor)

            meta = get_tensor_metadata(sky_tensor)
            output_metadata.append(tensor_metadata_to_proto(meta))

        return output_tensors, output_tensor_ids, output_metadata

    def proxy_forward(*args, **kwargs):
        input_tensor_ids, input_metadata = _collect_inputs(args, kwargs)
        cache_key = _input_cache_key(args)

        cached_specs = shape_cache.get(cache_key)
        if cached_specs is not None:
            # Cache hit: pre-allocate outputs and fire-and-forget
            output_tensors, output_tensor_ids, output_meta = _create_outputs_from_cache(
                cached_specs
            )

            request = service_pb2.ExecuteModuleForwardRequest(
                model_id=model_id,
                module_path=module_path,
                input_tensor_ids=input_tensor_ids,
                input_metadata=input_metadata,
                output_tensor_ids=output_tensor_ids,
                output_metadata=output_meta,
            )
            stream_manager.submit_execute_module_forward_ff(request)

            if len(output_tensors) == 1:
                return output_tensors[0]
            return tuple(output_tensors)

        # Cache miss: sync RPC to learn output shapes
        request = service_pb2.ExecuteModuleForwardRequest(
            model_id=model_id,
            module_path=module_path,
            input_tensor_ids=input_tensor_ids,
            input_metadata=input_metadata,
        )

        async def _submit_and_await():
            return await stream_manager.submit_execute_module_forward(request)

        response = run_async(_submit_and_await()).result()

        if not response.success:
            raise RuntimeError(
                f"ExecuteModuleForward failed for {module_path}: " f"{response.error_message}"
            )

        fwd_response = response.execute_module_forward
        output_tensors, output_specs = _create_outputs_from_response(fwd_response)

        # Cache the output specs for future calls with this input shape
        shape_cache[cache_key] = output_specs

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Cached output specs for {module_path} "
                f"(key={cache_key}, {len(output_specs)} outputs)"
            )

        if len(output_tensors) == 1:
            return output_tensors[0]
        return tuple(output_tensors)

    return proxy_forward


def proxy_triton_modules(
    model: torch.nn.Module,
    compute,
    model_id: int,
    module_paths: list[str],
) -> list[str]:
    """Replace forward() on specified modules with an RPC proxy.

    Args:
        model: The client-side model (on meta/sky device).
        compute: The Compute instance (has _grpc_client.stream).
        model_id: Server-assigned model ID from ExecuteFunction.
        module_paths: List of module path patterns (supports wildcards).

    Returns:
        List of concrete module paths that were proxied.
    """
    resolved = _resolve_module_paths(model, module_paths)
    if not resolved:
        logger.warning(
            f"No modules matched patterns {module_paths}. "
            f"Available modules: {[n for n, _ in model.named_modules() if n][:20]}"
        )
        return []

    stream_manager = compute._grpc_client.stream

    # Determine sky device index from the compute
    from skytorch.torch.backend._device import device_manager

    sky_device_index = device_manager.get_sky_device(compute, "cuda", 0).index

    proxied = []
    for path in resolved:
        # Resolve the module
        parts = path.split(".")
        parent = model
        for p in parts[:-1]:
            if p.isdigit():
                parent = parent[int(p)]
            else:
                parent = getattr(parent, p)
        module = getattr(parent, parts[-1]) if not parts[-1].isdigit() else parent[int(parts[-1])]

        # Replace forward
        proxy_fn = _make_forward_proxy(path, model_id, stream_manager, sky_device_index)
        module.forward = proxy_fn
        proxied.append(path)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Proxied module forward: {path}")

    logger.info(f"Proxied {len(proxied)} module(s) for Triton kernel forwarding")
    return proxied
