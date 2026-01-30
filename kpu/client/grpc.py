"""
Unified gRPC client that manages a single channel shared by multiple service clients.

This module provides a GRPCClient class that encapsulates all gRPC service clients
(TensorClient, MetricsClient, etc.) and manages channels with thread-safe isolation.

gRPC async channels have thread affinity - they must be used from the thread where
they were created. This module handles multi-threaded access (e.g., PyTorch autograd,
DataLoader workers, debuggers) by maintaining thread-local channels.
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

# Suppress asyncio errors from gRPC poller during debugging
# (BlockingIOError when debugger pauses the event loop)
logging.getLogger('asyncio').setLevel(logging.CRITICAL)


class GRPCClient:
    """
    Thread-safe gRPC client with automatic thread-local channel isolation.

    This class provides access to different gRPC service clients (tensor, metrics, etc.)
    while handling multi-threaded access safely. When accessed from the same event loop
    where the client was created, it uses the primary channel. When accessed from other
    threads (e.g., PyTorch autograd, DataLoader workers), it automatically creates
    thread-local channels to avoid contention.

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

        # Primary channel (created in __aenter__, used by main thread/loop)
        self._primary_channel: Optional[grpc.aio.Channel] = None
        self._primary_loop: Optional[asyncio.AbstractEventLoop] = None
        self._primary_tensor_client = None
        self._primary_metrics_client = None

        # Thread-local storage for secondary channels (worker threads)
        self._thread_local = threading.local()

    def _is_primary_loop(self) -> bool:
        """Check if we're on the primary event loop where the client was created."""
        if self._primary_loop is None:
            return False
        try:
            return asyncio.get_running_loop() is self._primary_loop
        except RuntimeError:
            return False

    def _get_thread_local_channel(self) -> grpc.aio.Channel:
        """Get or create a thread-local channel for the current thread."""
        channel = getattr(self._thread_local, 'channel', None)
        if channel is None:
            channel = grpc.aio.insecure_channel(self.address)
            self._thread_local.channel = channel
        return channel

    @property
    def channel(self) -> grpc.aio.Channel:
        """
        Get a gRPC channel appropriate for the current thread.

        Returns the primary channel when on the main event loop, or a
        thread-local channel when called from other threads.

        Returns:
            The gRPC channel

        Raises:
            RuntimeError: If the client is not connected
        """
        if self._primary_channel is None:
            raise RuntimeError(
                "GRPCClient is not connected. Use 'async with GRPCClient(...)' "
                "or call __aenter__() first."
            )

        if self._is_primary_loop():
            return self._primary_channel

        return self._get_thread_local_channel()

    @property
    def torch(self):
        """
        Get the PyTorch tensor service client for the current thread.

        Returns:
            TensorClient instance using a thread-appropriate channel

        Raises:
            RuntimeError: If the client is not connected
        """
        if self._primary_channel is None:
            raise RuntimeError(
                "GRPCClient is not connected. Use 'async with GRPCClient(...)' "
                "or call __aenter__() first."
            )

        if self._is_primary_loop():
            if self._primary_tensor_client is None:
                from kpu.torch.client.service import TensorClient
                self._primary_tensor_client = TensorClient(
                    channel=self._primary_channel,
                    metadata=self.metadata
                )
            return self._primary_tensor_client

        # Thread-local client
        client = getattr(self._thread_local, 'tensor_client', None)
        if client is None:
            from kpu.torch.client.service import TensorClient
            client = TensorClient(
                channel=self._get_thread_local_channel(),
                metadata=self.metadata
            )
            self._thread_local.tensor_client = client
        return client

    @property
    def metrics(self):
        """
        Get the metrics service client for the current thread.

        Returns:
            MetricsClient instance using a thread-appropriate channel

        Raises:
            RuntimeError: If the client is not connected
        """
        if self._primary_channel is None:
            raise RuntimeError(
                "GRPCClient is not connected. Use 'async with GRPCClient(...)' "
                "or call __aenter__() first."
            )

        if self._is_primary_loop():
            if self._primary_metrics_client is None:
                from kpu.client.metrics import MetricsClient
                self._primary_metrics_client = MetricsClient(
                    channel=self._primary_channel,
                    metadata=self.metadata
                )
            return self._primary_metrics_client

        # Thread-local client
        client = getattr(self._thread_local, 'metrics_client', None)
        if client is None:
            from kpu.client.metrics import MetricsClient
            client = MetricsClient(
                channel=self._get_thread_local_channel(),
                metadata=self.metadata
            )
            self._thread_local.metrics_client = client
        return client

    async def __aenter__(self):
        """
        Async context manager entry: create the primary gRPC channel.

        Returns:
            Self
        """
        logger.debug(f"Connecting to gRPC server at {self.address}")
        self._primary_channel = grpc.aio.insecure_channel(self.address)
        self._primary_loop = asyncio.get_running_loop()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Async context manager exit: close the primary gRPC channel.

        Note: Thread-local channels are not explicitly closed as they are
        managed by their respective threads and will be cleaned up when
        the threads terminate.
        """
        if self._primary_channel is not None:
            logger.debug(f"Closing gRPC connection to {self.address}")
            await self._primary_channel.close()
            self._primary_channel = None
            self._primary_loop = None
            self._primary_tensor_client = None
            self._primary_metrics_client = None
