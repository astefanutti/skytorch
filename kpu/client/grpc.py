"""
Unified gRPC client that manages a single channel shared by multiple service clients.

This module provides a GRPCClient class that encapsulates all gRPC service clients
(TensorClient, MetricsClient, etc.) and manages a single shared channel for efficiency.
"""

import logging
from typing import Optional

try:
    import grpc
except ImportError as e:
    raise ImportError(f"grpcio package is required: {e}\nInstall with: pip install grpcio")

from grpc.aio._typing import MetadataType

logger = logging.getLogger(__name__)


class GRPCClient:
    """
    Unified gRPC client that manages a single channel shared by all service clients.

    This class provides access to different gRPC service clients (tensor, metrics, etc.)
    while reusing the same underlying gRPC channel for efficiency.

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
        self._channel: Optional[grpc.aio.Channel] = None
        self._tensor_client = None
        self._metrics_client = None

    @property
    def channel(self) -> grpc.aio.Channel:
        """
        Get the gRPC channel.

        Returns:
            The gRPC channel

        Raises:
            RuntimeError: If the client is not connected
        """
        if self._channel is None:
            raise RuntimeError(
                "GRPCClient is not connected. Use 'async with GRPCClient(...)' "
                "or call __aenter__() first."
            )
        return self._channel

    @property
    def torch(self):
        """
        Get the PyTorch tensor service client.

        Returns:
            TensorClient instance using the shared channel

        Raises:
            RuntimeError: If the client is not connected
        """
        if self._tensor_client is None:
            # Lazy import to avoid circular dependencies
            from kpu.torch.client.service import TensorClient

            self._tensor_client = TensorClient(
                channel=self.channel,
                metadata=self.metadata
            )

        return self._tensor_client

    @property
    def metrics(self):
        """
        Get the metrics service client.

        Returns:
            MetricsClient instance using the shared channel

        Raises:
            RuntimeError: If the client is not connected
        """
        if self._metrics_client is None:
            # Lazy import to avoid circular dependencies
            from kpu.client.metrics import MetricsClient

            self._metrics_client = MetricsClient(
                channel=self.channel,
                metadata=self.metadata
            )

        return self._metrics_client

    async def __aenter__(self):
        """
        Async context manager entry: create the gRPC channel.

        Returns:
            Self
        """
        logger.debug(f"Connecting to gRPC server at {self.address}")
        self._channel = grpc.aio.insecure_channel(self.address)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Async context manager exit: close the gRPC channel.
        """
        if self._channel is not None:
            logger.debug(f"Closing gRPC connection to {self.address}")
            await self._channel.close()
            self._channel = None
            self._tensor_client = None
            self._metrics_client = None
