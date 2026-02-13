"""Direct Compute for SkyTorch gRPC server."""

import asyncio
import functools
import inspect
import logging
import os
import re
import textwrap
from typing import Callable, Optional

import grpc
import torch

from skytorch.client.grpc import GRPCClient
from skytorch.client.state_dict import SkyStateDict

logger = logging.getLogger(__name__)


class Compute:
    """
    Compute for direct connection to SkyTorch gRPC server without Kubernetes.

    Provides the same interface as skytorch.client.Compute for device management
    and gRPC communication, but connects directly to a gRPC server URL.

    Usage:
        from skytorch.torch.server.testing import Compute

        async with Compute("localhost:50051") as compute:
            device = compute.device("cpu")
            x = torch.tensor([1, 2, 3], device=device)
            y = x + 1
            print(y.cpu())
    """

    # Device string parsing pattern
    _DEVICE_PATTERN = re.compile(r"^([a-zA-Z_]+)(?::(\d+))?$")

    def __init__(
        self,
        url: str = "",
        name: str = "compute",
        on_metrics: Optional[Callable[[object], None]] = None,
    ):
        """
        Initialize Compute.

        Args:
            url: gRPC server URL (host:port). Defaults to SKYTORCH_SERVER_URL
                 environment variable or "localhost:50051".
            name: Name for this compute (used in error messages).
            on_metrics: Optional callback to receive metrics from this Compute resource.
        """
        self.url = url or os.environ.get("SKYTORCH_SERVER_URL", "localhost:50051")
        self.name = name
        self._on_metrics = on_metrics
        self._grpc_client: Optional[GRPCClient] = None
        self._metrics_stream_task: Optional[asyncio.Task] = None

    def _parse_url(self) -> tuple[str, int]:
        """Parse URL into host and port."""
        if ":" in self.url:
            host, port_str = self.url.rsplit(":", 1)
            return host, int(port_str)
        return self.url, 50051

    async def _stream_metrics(self):
        """Stream metrics from this Compute resource and call the callback."""
        try:
            async for snapshot in self._grpc_client.metrics.stream_metrics(
                metric_names=[
                    "gpu.utilization.compute",
                    "gpu.utilization.memory",
                    "gpu.memory.used",
                    "gpu.power.usage",
                ],
                interval_seconds=1.0,
            ):
                self._on_metrics(snapshot)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error streaming metrics: {e}")

    async def __aenter__(self) -> "Compute":
        """Connect to the gRPC server."""
        host, port = self._parse_url()
        self._grpc_client = GRPCClient(host=host, port=port)
        await self._grpc_client.__aenter__()

        # Start metrics streaming if callback is provided
        if self._on_metrics is not None:
            self._metrics_stream_task = asyncio.create_task(self._stream_metrics())

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Disconnect from the gRPC server."""
        # Cancel metrics stream task if running
        if self._metrics_stream_task is not None:
            self._metrics_stream_task.cancel()
            try:
                await self._metrics_stream_task
            except asyncio.CancelledError:
                pass
            self._metrics_stream_task = None

        if self._grpc_client:
            await self._grpc_client.__aexit__(exc_type, exc_val, exc_tb)
            self._grpc_client = None

    def device(self, type: str = "cpu", index: Optional[int] = None) -> torch.device:
        """
        Get a sky device mapped to this Compute.

        Args:
            type: Remote device type, optionally with index (e.g., "cuda", "cuda:0", "cpu")
            index: Remote device index (default: 0). Cannot be specified if type
                   already contains an index.

        Returns:
            torch.device with type "sky" and mapped local index

        Raises:
            RuntimeError: If type contains an index and index is also passed explicitly,
                          or if the device string format is invalid

        Example:
            >>> compute = Compute("localhost:50051")
            >>> device = compute.device("cuda")      # Same as cuda:0
            >>> device = compute.device("cuda:1")    # Uses index 1
            >>> device = compute.device("cuda", 1)   # Same as cuda:1
        """
        from skytorch.torch.backend._device import device_manager

        # Validate and parse device string
        match = self._DEVICE_PATTERN.match(type)
        if not match:
            raise RuntimeError(
                f"Invalid device string: {type!r}. "
                f"Expected format: 'device_type' or 'device_type:index'"
            )

        device_type = match.group(1)
        parsed_index = match.group(2)

        if parsed_index is not None:
            if index is not None:
                raise RuntimeError(
                    f"type (string) must not include an index because index was "
                    f"passed explicitly: {type}"
                )
            device_index = int(parsed_index)
        else:
            device_index = index if index is not None else 0

        return device_manager.get_sky_device(self, device_type, device_index)

    @staticmethod
    def _generate_imports(fn) -> str:
        """Generate import statements for global references used by a function."""
        import types

        lines = []
        seen = set()
        for name in fn.__code__.co_names:
            if name in seen or name not in fn.__globals__:
                continue
            seen.add(name)
            obj = fn.__globals__[name]
            if isinstance(obj, types.ModuleType):
                if obj.__name__ != name:
                    lines.append(f"import {obj.__name__} as {name}")
                else:
                    lines.append(f"import {obj.__name__}")
            elif hasattr(obj, "__module__") and hasattr(obj, "__qualname__"):
                module = getattr(obj, "__module__", None)
                if module and module != "builtins":
                    lines.append(f"from {module} import {name}")
        return "\n".join(lines)

    @staticmethod
    def _get_callable_source(fn) -> tuple[str, str] | None:
        """Extract source code and name from a callable, or None if not possible."""
        if getattr(fn, "__name__", "") == "<lambda>":
            return None

        if fn.__code__.co_freevars:
            return None  # Has closure variables — can't represent as source

        try:
            source = inspect.getsource(fn)
            source = textwrap.dedent(source)
        except (OSError, TypeError):
            return None

        imports = Compute._generate_imports(fn)
        if imports:
            source = imports + "\n\n" + source

        return source, fn.__name__

    async def execute(self, fn, *args, **kwargs):
        """
        Execute a function on the remote server and return sky tensors.

        For regular functions, the source code is sent as text (version-independent).
        Falls back to cloudpickle for lambdas, closures, or when source is unavailable.
        Any tensors in the result (e.g., model state_dict) stay on the GPU server;
        only metadata is returned. Client-side sky tensors are created to reference
        the remote storage.

        Args:
            fn: Callable to execute on the server
            *args: Positional arguments for the callable
            **kwargs: Keyword arguments for the callable

        Returns:
            dict[str, torch.Tensor]: Dictionary of sky tensors referencing
                remote GPU storage

        Raises:
            RuntimeError: If execution fails or gRPC client is not connected
        """
        import pickle

        import cloudpickle

        from skytorch.torch.backend._C import _create_remote_tensor
        from skytorch.torch.backend._device import device_manager
        from skytorch.torch.backend._storage import storage_manager
        from skytorch.torch.client.tensor import get_tensor_id
        from skytorch.torch.server import service_pb2

        if self._grpc_client is None:
            raise RuntimeError(
                "gRPC client is not connected. " "Use 'async with compute:' or call ready() first."
            )

        # 1. Serialize and call unary RPC
        source_info = self._get_callable_source(fn)
        try:
            if source_info:
                source, name = source_info
                response = await self._grpc_client.torch.execute_function(
                    b"",
                    pickle.dumps(args),
                    pickle.dumps(kwargs),
                    callable_source=source,
                    callable_name=name,
                )
            else:
                response = await self._grpc_client.torch.execute_function(
                    cloudpickle.dumps(fn),
                    pickle.dumps(args),
                    pickle.dumps(kwargs),
                )
        except grpc.aio.AioRpcError as e:
            raise RuntimeError(
                f"Failed to execute function: {e.code().name}: {e.details()}"
            ) from e

        if not response.success:
            raise RuntimeError(f"Remote execution failed: {response.error_message}")

        # 2. Create sky tensors from metadata
        sky_state_dict = {}
        registrations = []
        seen_storage: dict[int, torch.Tensor] = {}

        for info in response.tensors:
            sky_device_index = device_manager.get_sky_device(
                self, info.device_type, info.device_index
            ).index

            if info.storage_id in seen_storage:
                # Shared storage (weight tying) — create view from existing sky tensor
                base = seen_storage[info.storage_id]
                sky_tensor = base.as_strided(
                    list(info.shape), list(info.stride), info.storage_offset
                )
            else:
                sky_tensor = _create_remote_tensor(
                    info.storage_id,
                    list(info.shape),
                    info.dtype,
                    list(info.stride),
                    info.storage_offset,
                    info.storage_nbytes,
                    sky_device_index,
                )
                seen_storage[info.storage_id] = sky_tensor

            tensor_id = get_tensor_id(sky_tensor)
            storage_manager.register_storage(
                info.storage_id, info.storage_nbytes, sky_device_index
            )
            storage_manager.register_tensor(sky_tensor)
            registrations.append(
                service_pb2.TensorRegistration(storage_id=info.storage_id, tensor_id=tensor_id)
            )
            sky_state_dict[info.name] = sky_tensor

        # 4. Send mapping to server via stream (fire-and-forget)
        self._grpc_client.stream.submit_register_tensors(
            service_pb2.RegisterTensorsRequest(registrations=registrations)
        )

        logger.info(
            f"Received {len(sky_state_dict)} tensors from remote execution "
            f"({len(seen_storage)} unique storages)"
        )

        return SkyStateDict(sky_state_dict)


def compute(
    url: str = "localhost:50051",
    *,
    name: str = "compute",
    on_metrics: Optional[Callable[[object], None]] = None,
):
    """
    Decorator that automatically manages a Compute instance lifecycle.

    Creates a Compute instance, connects to the gRPC server, and passes
    the compute instance to the decorated function.

    Args:
        url: gRPC server URL (host:port). Defaults to SKYTORCH_SERVER_URL
             environment variable or "localhost:50051".
        name: Name for this compute (used in error messages).
        on_metrics: Optional callback to receive metrics from this Compute resource.

    Returns:
        Decorator function that wraps async functions.

    Example:
        >>> from skytorch.torch.server import compute, Compute
        >>>
        >>> @compute("localhost:50051")
        ... async def test_addition(compute: Compute):
        ...     device = compute.device()
        ...     x = torch.tensor([1, 2, 3], device=device)
        ...     y = torch.tensor([4, 5, 6], device=device)
        ...     z = x + y
        ...     print(z.cpu())
        >>>
        >>> await test_addition()
    """

    def decorator(func):
        # Inspect the function signature to find a Compute parameter
        sig = inspect.signature(func)
        compute_param_name = None

        for param_name, param in sig.parameters.items():
            if param.annotation is not inspect.Parameter.empty:
                # Check if the annotation is Compute
                if param.annotation is Compute or (
                    hasattr(param.annotation, "__name__") and param.annotation.__name__ == "Compute"
                ):
                    compute_param_name = param_name
                    break

        @functools.wraps(func)
        async def wrapper(*func_args, **func_kwargs):
            compute_instance = Compute(url=url, name=name, on_metrics=on_metrics)

            async with compute_instance as c:
                if compute_param_name:
                    # Pass as keyword argument to the typed parameter
                    func_kwargs[compute_param_name] = c
                    return await func(*func_args, **func_kwargs)
                else:
                    # Fallback to positional argument (first position)
                    return await func(c, *func_args, **func_kwargs)

        return wrapper

    return decorator
