"""
Metrics gRPC service implementation with support for multiple metrics sources.
"""
import asyncio
import logging
import time
from typing import AsyncIterator, Dict, List

try:
    import grpc
except ImportError as e:
    raise ImportError(f"grpcio is required: {e}. Install with: pip install grpcio")

# These imports will work after running hack/gen-grpc-proto.sh
try:
    from kpu.server.metrics import metrics_pb2
    from kpu.server.metrics import metrics_pb2_grpc
except ImportError:
    raise ImportError(
        "Generated gRPC code not found. Run hack/gen-grpc-proto.sh first.\n"
        "Make sure to install grpcio-tools: pip install grpcio-tools"
    )

from kpu.server.metrics.source import MetricsSource, MetricData

logger = logging.getLogger(__name__)


class MetricsServicer(metrics_pb2_grpc.MetricsServicer):
    """
    gRPC servicer for metrics collection with support for multiple sources.
    """

    def __init__(self, *sources: MetricsSource):
        """
        Initialize the metrics servicer with metrics sources.

        Args:
            *sources: Metrics sources to register
        """
        self._sources: Dict[str, MetricsSource] = {}

        # Register provided sources
        for source in sources:
            self._register_source(source)

    def _register_source(self, source: MetricsSource) -> None:
        """
        Register a metrics source.

        Args:
            source: Metrics source to register
        """
        source_name = source.get_source_name()

        if not source.is_available():
            logger.warning(f"Metrics source '{source_name}' is not available, skipping registration")
            return

        self._sources[source_name] = source
        logger.info(f"Registered metrics source: {source_name}")

        # Log source metadata
        metadata = source.get_metadata()
        if metadata:
            logger.info(f"  Metadata: {metadata}")

    def cleanup(self) -> None:
        """Cleanup all registered sources."""
        for source_name, source in list(self._sources.items()):
            try:
                source.cleanup()
                logger.info(f"Cleaned up metrics source: {source_name}")
            except Exception as e:
                logger.error(f"Error cleaning up source '{source_name}': {e}")

    async def GetMetrics(
        self,
        request: metrics_pb2.GetMetricsRequest,
        context: grpc.aio.ServicerContext
    ) -> metrics_pb2.GetMetricsResponse:
        """
        Get current metrics from all or specific sources.

        Args:
            request: Request with optional filters
            context: gRPC context

        Returns:
            Response containing metrics snapshots
        """
        try:
            snapshots = []

            # Filter sources if requested
            sources_to_query = self._get_filtered_sources(request.sources)

            # Collect metrics from each source
            for source_name, source in sources_to_query.items():
                try:
                    # Collect metrics with filters
                    metric_names = list(request.metric_names) if request.metric_names else None
                    label_filters = dict(request.label_filters) if request.label_filters else None

                    metrics_data = source.collect(
                        metric_names=metric_names,
                        label_filters=label_filters
                    )

                    # Convert to protobuf metrics
                    proto_metrics = self._convert_metrics_to_proto(metrics_data)

                    # Create snapshot
                    snapshot = metrics_pb2.MetricsSnapshot(
                        metrics=proto_metrics,
                        source=source_name,
                        timestamp=int(time.time())
                    )

                    snapshots.append(snapshot)

                except Exception as e:
                    logger.error(f"Error collecting metrics from source '{source_name}': {e}")

            return metrics_pb2.GetMetricsResponse(snapshots=snapshots)

        except Exception as e:
            logger.error(f"Error in GetMetrics: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Error getting metrics: {e}")

    async def StreamMetrics(
        self,
        request: metrics_pb2.StreamMetricsRequest,
        context: grpc.aio.ServicerContext
    ) -> AsyncIterator[metrics_pb2.MetricsSnapshot]:
        """
        Stream metrics continuously at a specified interval.

        Args:
            request: Request with interval and filters
            context: gRPC context

        Yields:
            Metrics snapshots
        """
        try:
            # Get interval (default: 1.0 seconds)
            interval = request.interval_seconds if request.interval_seconds > 0 else 1.0

            # Filter sources if requested
            sources_to_query = self._get_filtered_sources(request.sources)

            logger.info(
                f"Starting metrics stream with interval={interval}s, "
                f"sources={list(sources_to_query.keys())}"
            )

            # Stream metrics until client disconnects
            while not context.cancelled():
                for source_name, source in sources_to_query.items():
                    try:
                        # Collect metrics with filters
                        metric_names = list(request.metric_names) if request.metric_names else None
                        label_filters = dict(request.label_filters) if request.label_filters else None

                        metrics_data = source.collect(
                            metric_names=metric_names,
                            label_filters=label_filters
                        )

                        # Convert to protobuf metrics
                        proto_metrics = self._convert_metrics_to_proto(metrics_data)

                        # Create and yield snapshot
                        snapshot = metrics_pb2.MetricsSnapshot(
                            metrics=proto_metrics,
                            source=source_name,
                            timestamp=int(time.time())
                        )

                        yield snapshot

                    except Exception as e:
                        logger.error(f"Error collecting metrics from source '{source_name}': {e}")

                # Wait for next interval
                await asyncio.sleep(interval)

        except asyncio.CancelledError:
            logger.info("Metrics stream cancelled by client")
        except Exception as e:
            logger.error(f"Error in StreamMetrics: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Error streaming metrics: {e}")

    def _get_filtered_sources(self, source_filter: List[str]) -> Dict[str, MetricsSource]:
        """
        Get sources filtered by name.

        Args:
            source_filter: List of source names to include (empty = all)

        Returns:
            Dictionary of filtered sources
        """
        if not source_filter:
            return self._sources

        filtered = {}
        for source_name in source_filter:
            if source_name in self._sources:
                filtered[source_name] = self._sources[source_name]
            else:
                logger.warning(f"Requested source '{source_name}' is not registered")

        return filtered

    def _convert_metrics_to_proto(self, metrics: List[MetricData]) -> List[metrics_pb2.Metric]:
        """
        Convert MetricData objects to protobuf Metric messages.

        Args:
            metrics: List of MetricData objects

        Returns:
            List of protobuf Metric messages
        """
        proto_metrics = []

        for metric in metrics:
            # Map metric type string to enum
            metric_type = metrics_pb2.UNKNOWN
            if metric.metric_type == "GAUGE":
                metric_type = metrics_pb2.GAUGE
            elif metric.metric_type == "COUNTER":
                metric_type = metrics_pb2.COUNTER
            elif metric.metric_type == "HISTOGRAM":
                metric_type = metrics_pb2.HISTOGRAM

            proto_metric = metrics_pb2.Metric(
                name=metric.name,
                type=metric_type,
                value=metric.value,
                unit=metric.unit,
                labels=metric.labels,
                timestamp=metric.timestamp,
                help=metric.help_text
            )

            proto_metrics.append(proto_metric)

        return proto_metrics
