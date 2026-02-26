"""Module-level forward proxying for Triton kernel modules.

When a model contains modules that use Triton kernels (e.g., MXFP4 quantized
MoE experts), those kernels bypass the ATen dispatcher and cannot be intercepted
by SkyTorch's PrivateUse1 dispatch. This module replaces the forward() method
of such modules with an RPC proxy that executes the forward on the server where
the real GPU tensors and Triton kernels live.

Output shapes are predicted locally by running the module's original forward on
meta tensors. All calls use pre-allocated output tensors and fire-and-forget
dispatch, matching the ATen op fast path. A shape cache avoids repeated meta
inference for inputs with the same shapes and dtypes.

Usage:
    proxy_triton_modules(model, compute, model_id, ["model.layers.*.experts"])
"""

import fnmatch
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from skytorch.torch.backend._C import _allocate_storage_id, _create_remote_tensor
from skytorch.torch.backend._device import device_manager
from skytorch.torch.backend._storage import storage_manager
from skytorch.torch.client.tensor import get_storage_id, get_tensor_id
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

    # Filter out children of already-matched paths: if "layers.0" is proxied,
    # "layers.0.linear" is redundant (the server runs it as part of the parent's forward).
    result = []
    for path in sorted(matched):
        if not any(path.startswith(p + ".") for p in result):
            result.append(path)
    return result


def _input_cache_key(args: tuple) -> tuple:
    """Compute a cache key from input tensor shapes and dtypes."""
    return tuple((arg.shape, arg.dtype) for arg in args)


def _make_forward_proxy(
    module_path: str,
    model_id: int,
    stream_manager: "StreamManager",
    sky_device_index: int,
    module: torch.nn.Module,
    compute_dtype: torch.dtype | None = None,
):
    """Create a proxy forward function for a module.

    Output shapes are predicted locally via meta tensor execution on the
    module's original forward. All calls use fire-and-forget dispatch.

    Args:
        module_path: Dotted path to the module (e.g., "layers.0.experts").
        model_id: Server-assigned model ID.
        stream_manager: StreamManager for the bidirectional stream.
        sky_device_index: Local sky device index for output tensors.
        module: The nn.Module whose forward will be proxied.

    Returns:
        A callable that replaces the module's forward().
    """
    # Shape cache: input_cache_key -> list[_OutputSpec]
    shape_cache: dict[tuple, list[_OutputSpec]] = {}

    # Save original forward for meta shape prediction before it gets replaced
    original_forward = module.forward

    # Pre-compute meta-device copies of parameters and buffers (once at setup).
    # When compute_dtype is provided, override float dtypes — proxied module
    # parameters may still be float32 meta tensors (never loaded by load_into),
    # while the actual server-side weights use the model's compute dtype.
    _meta_state: dict[str, torch.Tensor] = {}
    for name, param in module.named_parameters():
        dtype = compute_dtype if compute_dtype and param.is_floating_point() else param.dtype
        _meta_state[name] = torch.empty(param.shape, dtype=dtype, device="meta")
    for name, buf in module.named_buffers():
        dtype = compute_dtype if compute_dtype and buf.is_floating_point() else buf.dtype
        _meta_state[name] = torch.empty(buf.shape, dtype=dtype, device="meta")

    def _collect_inputs(args, kwargs):
        if kwargs:
            raise TypeError(
                f"Proxied module '{module_path}' received keyword arguments "
                f"{list(kwargs.keys())}. Only positional sky tensor arguments "
                f"are supported. Consider proxying at a higher module level "
                f"where forward() takes only tensor inputs."
            )

        input_tensor_ids = []
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor) and arg.device.type == "sky":
                tid = get_tensor_id(arg)
                input_tensor_ids.append(tid)
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
        return input_tensor_ids

    def _predict_output_specs(args):
        """Predict output tensor specs by running the original forward on meta tensors."""
        meta_args = tuple(
            torch.empty(
                arg.shape,
                dtype=compute_dtype if compute_dtype and arg.is_floating_point() else arg.dtype,
                device="meta",
            )
            for arg in args
        )

        # Use functional_call to run original forward with meta parameters,
        # avoiding direct param.data assignment which rejects cross-backend swaps.
        current_fwd = module.forward
        module.forward = original_forward
        try:
            with torch.no_grad():
                meta_result = torch.func.functional_call(module, _meta_state, meta_args)
        finally:
            module.forward = current_fwd

        # Normalize to flat list of tensors
        if isinstance(meta_result, torch.Tensor):
            meta_outputs = [meta_result]
        elif isinstance(meta_result, (tuple, list)):
            meta_outputs = [t for t in meta_result if isinstance(t, torch.Tensor)]
        else:
            meta_outputs = []

        specs = []
        for t in meta_outputs:
            nbytes = t.untyped_storage().nbytes()
            if nbytes == 0 and t.numel() > 0:
                # Meta storage may report 0; compute from tensor metadata
                max_idx = t.storage_offset()
                for s, st in zip(t.shape, t.stride()):
                    if s > 0:
                        max_idx += (s - 1) * st
                nbytes = (max_idx + 1) * t.element_size()
            specs.append(
                _OutputSpec(
                    shape=tuple(t.shape),
                    dtype=str(t.dtype),
                    stride=tuple(t.stride()),
                    storage_offset=t.storage_offset(),
                    storage_nbytes=nbytes,
                    device_type="cuda",
                    device_index=0,
                )
            )

        return specs

    def _create_outputs(specs: list[_OutputSpec]):
        """Pre-allocate output sky tensors from specs."""
        output_tensors = []
        output_tensor_ids = []

        for spec in specs:
            storage_id = _allocate_storage_id()
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
            storage_manager.register_tensor(sky_tensor, skip_cpp=True)

            tid = get_tensor_id(sky_tensor)
            output_tensor_ids.append(tid)
            output_tensors.append(sky_tensor)

        return output_tensors, output_tensor_ids

    def proxy_forward(*args, **kwargs):
        input_tensor_ids = _collect_inputs(args, kwargs)
        cache_key = _input_cache_key(args)

        cached_specs = shape_cache.get(cache_key)
        if cached_specs is None:
            # Predict output shapes via meta tensor execution
            cached_specs = _predict_output_specs(args)
            shape_cache[cache_key] = cached_specs

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Predicted output specs for {module_path} "
                    f"(key={cache_key}, {len(cached_specs)} outputs)"
                )

        output_tensors, output_tensor_ids = _create_outputs(cached_specs)

        request = service_pb2.ExecuteModuleForwardRequest(
            model_id=model_id,
            module_path=module_path,
            input_tensor_ids=input_tensor_ids,
            output_tensor_ids=output_tensor_ids,
        )
        stream_manager.submit_execute_module_forward_ff(request)

        # Register output tensors in C++ tracking set AFTER submission,
        # so they are not marked as "known" before the server has them.
        # This prevents a cascade of "Tensor does not exist" errors if the
        # module forward fails server-side.
        from skytorch.torch.backend import _C

        for sky_tensor in output_tensors:
            tid = get_tensor_id(sky_tensor)
            sid = get_storage_id(sky_tensor)
            _C._register_tensor_id(tid)
            _C._register_storage_tensor_mapping(sid, tid)

        if len(output_tensors) == 1:
            return output_tensors[0]
        return tuple(output_tensors)

    return proxy_forward


def proxy_triton_modules(
    model: torch.nn.Module,
    compute,
    model_id: int,
    module_paths: list[str],
    compute_dtype: torch.dtype | None = None,
) -> list[str]:
    """Replace forward() on specified modules with an RPC proxy.

    Args:
        model: The client-side model (on meta/sky device).
        compute: The Compute instance (has _grpc_client.stream).
        model_id: Server-assigned model ID from ExecuteFunction.
        module_paths: List of module path patterns (supports wildcards).
        compute_dtype: Optional dtype override for floating-point parameters
            whose meta tensors may not reflect the server-side compute dtype.

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

    # Determine sky device index from the compute's existing device registration
    sky_device_index = device_manager.get_compute_sky_device(compute).index

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
        proxy_fn = _make_forward_proxy(
            path, model_id, stream_manager, sky_device_index, module, compute_dtype
        )
        module.forward = proxy_fn
        proxied.append(path)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Proxied module forward: {path}")

    logger.info(f"Proxied {len(proxied)} module(s) for Triton kernel forwarding")
    return proxied
