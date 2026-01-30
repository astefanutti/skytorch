"""
Unified gRPC client that manages a single channel shared by multiple service clients.

This module provides a GRPCClient class that encapsulates all gRPC service clients
(TensorClient, MetricsClient, etc.) and manages a single shared channel for efficiency.
"""

import asyncio
import logging
import os
import threading
from typing import Optional

# Suppress gRPC fork warning when using threads
# Must be set before importing grpc
# See: https://github.com/grpc/grpc/issues/38703
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

try:
    import grpc
except ImportError as e:
    raise ImportError(f"grpcio package is required: {e}\nInstall with: pip install grpcio")

from grpc.aio._typing import MetadataType

logger = logging.getLogger(__name__)


class LoopAwareClientPool:
    """
    Manages gRPC channels across multiple event loops/threads.

    gRPC async channels are bound to the event loop they were created on.
    When accessed from a different loop (e.g., DataLoader worker threads),
    we need separate channels for each loop.
    """

    def __init__(self, address: str, metadata: Optional[MetadataType] = None):
        self._address = address
        self._metadata = metadata
        self._primary_channel: Optional[grpc.aio.Channel] = None
        self._primary_loop: Optional[asyncio.AbstractEventLoop] = None
        self._secondary_channels: dict[int, grpc.aio.Channel] = {}
        self._lock = threading.Lock()

    async def connect(self) -> None:
        """Initialize the primary channel on the current event loop."""
        self._primary_channel = grpc.aio.insecure_channel(self._address)
        self._primary_loop = asyncio.get_running_loop()

    def get_channel(self) -> grpc.aio.Channel:
        """Get a channel for the current event loop."""
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            return self._primary_channel

        if current_loop is self._primary_loop:
            return self._primary_channel

        loop_id = id(current_loop)
        with self._lock:
            if loop_id not in self._secondary_channels:
                self._secondary_channels[loop_id] = grpc.aio.insecure_channel(
                    self._address
                )
            return self._secondary_channels[loop_id]

    async def close(self) -> None:
        """Close all channels (primary and secondary)."""
        if self._primary_channel is not None:
            await self._primary_channel.close()
            self._primary_channel = None
            self._primary_loop = None

        with self._lock:
            for channel in self._secondary_channels.values():
                await channel.close()
            self._secondary_channels.clear()


class GRPCClient:
    """
    Unified gRPC client that manages a single channel shared by all service clients.

    This class provides access to different gRPC service clients (tensor, metrics, etc.)
    while reusing the same underlying gRPC channel for efficiency. Handles multi-threaded
    access by maintaining separate channels per event loop.

    Example:
        >>> async with GRPCClient(host="localhost", port=50051) as client:
        ...     # Use PyTorch tensor service
        ...     tensors = await client.torch.receive_tensors(count=1)
        ...
        ...     # Use metrics service
        ...     metrics = await client.metrics.get_metrics()
    """

    def __init__(
        self,
        host: str = 'localhost',
        port: int = 50051,
        metadata: Optional[MetadataType] = None
    ):
        """
        Initialize the gRPC client.

        Args:
            host: Server host address
            port: Server port
            metadata: Optional metadata to include in all requests
        """
        self.address = f'{host}:{port}'
        self.metadata = metadata
        self._pool: Optional[LoopAwareClientPool] = None
        self._tensor_clients: dict[int, object] = {}
        self._metrics_clients: dict[int, object] = {}
        self._lock = threading.Lock()

    def _get_loop_id(self) -> int:
        """Get the current event loop ID, or 0 if no loop is running."""
        try:
            return id(asyncio.get_running_loop())
        except RuntimeError:
            return 0

    @property
    def channel(self) -> grpc.aio.Channel:
        """
        Get the gRPC channel for the current event loop.

        Returns:
            The gRPC channel

        Raises:
            RuntimeError: If the client is not connected
        """
        if self._pool is None:
            raise RuntimeError(
                "GRPCClient is not connected. Use 'async with GRPCClient(...)' "
                "or call __aenter__() first."
            )
        return self._pool.get_channel()

    @property
    def torch(self):
        """
        Get the PyTorch tensor service client for the current event loop.

        Returns:
            TensorClient instance using the loop-appropriate channel

        Raises:
            RuntimeError: If the client is not connected
        """
        loop_id = self._get_loop_id()
        with self._lock:
            if loop_id not in self._tensor_clients:
                # Lazy import to avoid circular dependencies
                from kpu.torch.client.service import TensorClient

                self._tensor_clients[loop_id] = TensorClient(
                    channel=self.channel,
                    metadata=self.metadata
                )
            return self._tensor_clients[loop_id]

    @property
    def metrics(self):
        """
        Get the metrics service client for the current event loop.

        Returns:
            MetricsClient instance using the loop-appropriate channel

        Raises:
            RuntimeError: If the client is not connected
        """
        loop_id = self._get_loop_id()
        with self._lock:
            if loop_id not in self._metrics_clients:
                # Lazy import to avoid circular dependencies
                from kpu.client.metrics import MetricsClient

                self._metrics_clients[loop_id] = MetricsClient(
                    channel=self.channel,
                    metadata=self.metadata
                )
            return self._metrics_clients[loop_id]

    async def __aenter__(self):
        """
        Async context manager entry: create the gRPC channel pool.

        Returns:
            Self
        """
        logger.debug(f"Connecting to gRPC server at {self.address}")
        self._pool = LoopAwareClientPool(self.address, self.metadata)
        await self._pool.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Async context manager exit: close all gRPC channels.
        """
        if self._pool is not None:
            logger.debug(f"Closing gRPC connection to {self.address}")
            await self._pool.close()
            self._pool = None
            self._tensor_clients.clear()
            self._metrics_clients.clear()
