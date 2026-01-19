"""Direct Compute for KPU gRPC server."""

import functools
import inspect
import os
from typing import Optional

import torch

from kpu.client.grpc import GRPCClient


class Compute:
    """
    Compute for direct connection to KPU gRPC server without Kubernetes.

    Provides the same interface as kpu.client.Compute for device management
    and gRPC communication, but connects directly to a gRPC server URL.

    Usage:
        from kpu.torch.server.testing import Compute

        async with Compute("localhost:50051") as compute:
            device = compute.device("cpu")
            x = torch.tensor([1, 2, 3], device=device)
            y = x + 1
            print(y.cpu())
    """

    def __init__(
        self,
        url: str = "",
        name: str = "compute",
    ):
        """
        Initialize Compute.

        Args:
            url: gRPC server URL (host:port). Defaults to KPU_SERVER_URL
                 environment variable or "localhost:50051".
            name: Name for this compute (used in error messages).
        """
        self.url = url or os.environ.get("KPU_SERVER_URL", "localhost:50051")
        self.name = name
        self._grpc_client: Optional[GRPCClient] = None

    def _parse_url(self) -> tuple[str, int]:
        """Parse URL into host and port."""
        if ":" in self.url:
            host, port_str = self.url.rsplit(":", 1)
            return host, int(port_str)
        return self.url, 50051

    async def __aenter__(self) -> "Compute":
        """Connect to the gRPC server."""
        host, port = self._parse_url()
        self._grpc_client = GRPCClient(host=host, port=port)
        await self._grpc_client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Disconnect from the gRPC server."""
        if self._grpc_client:
            await self._grpc_client.__aexit__(exc_type, exc_val, exc_tb)
            self._grpc_client = None

    def device(self, type: str = "cpu", index: int = 0) -> torch.device:
        """
        Get a KPU device mapped to this Compute.

        Args:
            type: Remote device type (e.g., "cpu", "cuda")
            index: Remote device index

        Returns:
            torch.device with type "kpu" and mapped local index
        """
        from kpu.torch.backend._device import device_manager

        return device_manager.get_kpu_device(self, type, index)


def compute(
    url: str = "localhost:50051",
    *,
    name: str = "compute",
):
    """
    Decorator that automatically manages a Compute instance lifecycle.

    Creates a Compute instance, connects to the gRPC server, and passes
    the compute instance to the decorated function.

    Args:
        url: gRPC server URL (host:port). Defaults to KPU_SERVER_URL
             environment variable or "localhost:50051".
        name: Name for this compute (used in error messages).

    Returns:
        Decorator function that wraps async functions.

    Example:
        >>> from kpu.torch.server import compute, Compute
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
                    hasattr(param.annotation, "__name__")
                    and param.annotation.__name__ == "Compute"
                ):
                    compute_param_name = param_name
                    break

        @functools.wraps(func)
        async def wrapper(*func_args, **func_kwargs):
            compute_instance = Compute(url=url, name=name)

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
