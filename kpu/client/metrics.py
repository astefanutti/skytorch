"""
gRPC client for the KPU metrics service.

This module provides a MetricsClient class for polling and streaming metrics
from KPU compute resources.
"""

import logging
from typing import Optional

try:
    import grpc
except ImportError as e:
    raise ImportError(f"grpcio package is required: {e}\nInstall with: pip install grpcio")

try:
    from kpu.server.metrics import metrics_pb2
    from kpu.server.metrics import metrics_pb2_grpc
except ImportError:
    raise ImportError(
        "Generated gRPC code not found. Run hack/gen-grpc-proto.sh first."
    )

from grpc.aio._typing import MetadataType

logger = logging.getLogger(__name__)


class MetricsClient:
    """
    Async gRPC client for the metrics service.

    Uses a shared gRPC channel provided by the caller (typically GRPCClient).
    Provides methods for both polling and streaming metrics.
    """

    def __init__(self, channel: grpc.aio.Channel, metadata: Optional[MetadataType] = None):
        """
        Initialize the metrics client.

        Args:
            channel: gRPC channel to use for communication
            metadata: Optional metadata to include in requests
        """
        self.channel = channel
        self.metadata = metadata
        self.stub = metrics_pb2_grpc.MetricsStub(self.channel)

    async def get_metrics(
        self,
        metric_names: Optional[list[str]] = None,
        sources: Optional[list[str]] = None,
        label_filters: Optional[dict[str, str]] = None
    ) -> list[metrics_pb2.MetricsSnapshot]:
        """
        Get current metrics (polling mode).

        Args:
            metric_names: Optional list of specific metric names to fetch
            sources: Optional list of metrics source names to query
            label_filters: Optional label filters to apply

        Returns:
            List of metrics snapshots from different sources
        """
        request = metrics_pb2.GetMetricsRequest(
            metric_names=metric_names or [],
            sources=sources or [],
        )

        if label_filters:
            request.label_filters.update(label_filters)

        logger.debug(
            f"Fetching metrics (sources: {sources or 'all'}, "
            f"metric_names: {metric_names or 'all'})"
        )

        response = await self.stub.GetMetrics(request, metadata=self.metadata)
        logger.info(f"Received {len(response.snapshots)} metrics snapshots")

        return list(response.snapshots)

    async def stream_metrics(
        self,
        interval_seconds: float = 1.0,
        metric_names: Optional[list[str]] = None,
        sources: Optional[list[str]] = None,
        label_filters: Optional[dict[str, str]] = None
    ):
        """
        Stream metrics continuously (push mode).

        Args:
            interval_seconds: How often to send metrics updates
            metric_names: Optional list of specific metric names to fetch
            sources: Optional list of metrics source names to query
            label_filters: Optional label filters to apply

        Yields:
            MetricsSnapshot objects as they arrive from the server
        """
        request = metrics_pb2.StreamMetricsRequest(
            interval_seconds=interval_seconds,
            metric_names=metric_names or [],
            sources=sources or [],
        )

        if label_filters:
            request.label_filters.update(label_filters)

        logger.info(
            f"Starting metrics stream (interval: {interval_seconds}s, "
            f"sources: {sources or 'all'}, metric_names: {metric_names or 'all'})"
        )

        async for snapshot in self.stub.StreamMetrics(request, metadata=self.metadata):
            logger.debug(
                f"Received metrics snapshot from {snapshot.source} "
                f"with {len(snapshot.metrics)} metrics"
            )
            yield snapshot
