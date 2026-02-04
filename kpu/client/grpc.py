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
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from kpu.torch.client.stream import StreamManager

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
        self._stream_manager: Optional["StreamManager"] = None

        # Thread-local storage for secondary channels (worker threads)
        self._thread_local = threading.local()

    def _is_primary_loop(self) -> bool:
        """Check if we're on the primary event loop (global background loop)."""
        if self._primary_loop is None:
            return False
        try:
            running_loop = asyncio.get_running_loop()
            return running_loop is self._primary_loop
        except RuntimeError:
            # No running loop - we're not on the primary loop
            # but run_async will submit to the global loop anyway
            return False

    def _get_thread_local_channel(self) -> grpc.aio.Channel:
        """Get or create a thread-local channel for the current thread."""
        channel = getattr(self._thread_local, 'channel', None)
        if channel is None:
            channel = grpc.aio.insecure_channel(self.address)
            self._thread_local.channel = channel
        return channel

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

    @property
    def stream(self) -> "StreamManager":
        """
        Get the StreamManager for pipelined operations.

        All threads share a single StreamManager to ensure ordering across
        all operations. The StreamManager handles thread-safe submission.

        Returns:
            StreamManager instance (shared across all threads)

        Raises:
            RuntimeError: If the client is not connected or stream not started
        """
        if self._stream_manager is None:
            raise RuntimeError(
                "StreamManager is not available. Use 'async with GRPCClient(...)' "
                "or call __aenter__() first."
            )
        return self._stream_manager

    async def __aenter__(self):
        """
        Async context manager entry: create the primary gRPC channel.

        Creates the channel and StreamManager on the global background loop
        to ensure proper ordering of operations from all threads.

        Returns:
            Self
        """
        from kpu.torch.backend._async import get_event_loop

        logger.debug(f"Connecting to gRPC server at {self.address}")

        # Get the background loop - this ensures it's running
        global_loop = get_event_loop()

        # Create the channel on the global loop
        async def create_channel():
            return grpc.aio.insecure_channel(self.address)

        # Run channel creation on global loop
        future = asyncio.run_coroutine_threadsafe(create_channel(), global_loop)
        self._primary_channel = future.result()
        self._primary_loop = global_loop

        # Initialize StreamManager on the global loop
        from kpu.torch.client.stream import StreamManager
        from kpu.torch.server import service_pb2_grpc

        stub = service_pb2_grpc.ServiceStub(self._primary_channel)
        self._stream_manager = StreamManager(stub, self.metadata)

        # Start the stream manager on the global loop
        async def start_stream():
            await self._stream_manager.start(global_loop)

        future = asyncio.run_coroutine_threadsafe(start_stream(), global_loop)
        future.result()

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Async context manager exit: close the primary gRPC channel.

        Note: Thread-local channels are not explicitly closed as they are
        managed by their respective threads and will be cleaned up when
        the threads terminate.
        """
        from kpu.torch.backend._async import get_event_loop

        global_loop = get_event_loop()

        # Close StreamManager first
        if self._stream_manager is not None:
            async def close_stream():
                await self._stream_manager.close()

            future = asyncio.run_coroutine_threadsafe(close_stream(), global_loop)
            future.result()
            self._stream_manager = None

        if self._primary_channel is not None:
            logger.debug(f"Closing gRPC connection to {self.address}")

            async def close_channel():
                await self._primary_channel.close()

            future = asyncio.run_coroutine_threadsafe(close_channel(), global_loop)
            future.result()
            self._primary_channel = None
            self._primary_loop = None
            self._primary_tensor_client = None
            self._primary_metrics_client = None
