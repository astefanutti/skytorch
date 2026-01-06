"""
Async gRPC server implementation for streaming PyTorch tensors.
"""
import logging
from typing import Optional

try:
    import grpc
except ImportError as e:
    raise ImportError(f"Required dependency not found: {e}. Install with: pip install grpcio")

# These imports will work after running hack/gen-grpc-proto.sh
try:
    from kpu.torch.server import service_pb2
    from kpu.torch.server import service_pb2_grpc
except ImportError:
    raise ImportError(
        "Generated gRPC code not found. Run hack/gen-grpc-proto.sh first.\n"
        "Make sure to install grpcio-tools: pip install grpcio-tools"
    )

from kpu.torch.server.serialization import DEFAULT_CHUNK_SIZE
from kpu.torch.server.service import TensorServicer
from kpu.server.health import HealthServicer
from kpu.server.health import health_pb2
from kpu.server.health import health_pb2_grpc
from kpu.server.metrics import MetricsServicer, MetricsSource
from kpu.server.metrics import metrics_pb2_grpc


logger = logging.getLogger(__name__)


async def serve(
    server: grpc.aio.Server,
    *,
    host: str = "[::]",
    port: int = 50051,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    metrics_sources: Optional[list[MetricsSource]] = None
) -> None:
    """
    Start the async gRPC server.

    Args:
        server: Server to start
        host: Host to listen on
        port: Port to listen on
        chunk_size: Size of chunks for streaming tensors
        metrics_sources: Optional list of metrics sources
    """

    # Add health service
    # FIXME: grpcio-health-checking should be used instead
    # Note enter_graceful_shutdown should be called, and also look at xDS.
    health_servicer = HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

    # Add tensor service
    servicer = TensorServicer(chunk_size=chunk_size)
    service_pb2_grpc.add_ServiceServicer_to_server(servicer, server)
    health_servicer.set_service_status(
        "kpu.torch.Service",
        health_pb2.HealthCheckResponse.SERVING
    )

    # Add metrics service with provided sources
    if metrics_sources:
        metrics_servicer = MetricsServicer(*metrics_sources)
        metrics_pb2_grpc.add_MetricsServicer_to_server(metrics_servicer, server)
        health_servicer.set_service_status(
            "kpu.server.Metrics",
            health_pb2.HealthCheckResponse.SERVING
        )

    # Set overall server health
    health_servicer.set_service_status(
        "",
        health_pb2.HealthCheckResponse.SERVING
    )

    listen_addr = f'{host}:{port}'
    server.add_insecure_port(listen_addr)

    logger.info(f"Starting server on {listen_addr}")
    await server.start()

    logger.info(f"Server listening on port {port}")
    logger.info(f"Chunk size: {chunk_size} bytes")

    await server.wait_for_termination()


async def graceful_shutdown(
    server: grpc.aio.Server,
    metrics_sources: list[MetricsSource],
    grace: Optional[float],
) -> None:
    logging.info("Graceful shutdown...")

    # Shuts down the server. During the grace period,
    # the server won't accept new connections and allow
    # existing RPCs to continue within the grace period.
    await server.stop(grace=grace)

    # Cleanup metrics sources
    for source in metrics_sources:
        source.cleanup()

    logging.info("Graceful shutdown complete")
