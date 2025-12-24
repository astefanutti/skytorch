"""
gRPC Health Checking service implementation.

Implements the gRPC Health Checking Protocol as specified in:
https://github.com/grpc/grpc/blob/master/doc/health-checking.md
"""

import asyncio
import logging
from typing import AsyncIterator, Dict

try:
    import grpc
except ImportError as e:
    raise ImportError(f"Required dependency not found: {e}. Install with: pip install grpcio")

try:
    from kpu.server.health import health_pb2
    from kpu.server.health import health_pb2_grpc
except ImportError:
    raise ImportError(
        "Generated gRPC code not found. Run hack/gen-grpc-proto.sh first.\n"
        "Make sure to install grpcio-tools: pip install grpcio-tools"
    )


logger = logging.getLogger(__name__)


class HealthServicer(health_pb2_grpc.HealthServicer):
    """
    gRPC Health Checking service implementation.

    This service allows clients to query the health status of the server
    and its individual services.
    """

    def __init__(self):
        """Initialize the health servicer with default statuses."""
        self._status: Dict[str, health_pb2.HealthCheckResponse.ServingStatus] = {}
        self._watchers: Dict[str, list] = {}

    def set_service_status(
        self,
        service: str,
        status: health_pb2.HealthCheckResponse.ServingStatus
    ) -> None:
        """
        Set the health status for a service.

        Args:
            service: Service name (empty string for overall server health)
            status: Health status (SERVING, NOT_SERVING, etc.)
        """
        self._status[service] = status
        logger.info(f"Health status for service '{service}' set to {status}")

        # Notify watchers of the status change
        if service in self._watchers:
            for queue in self._watchers[service]:
                try:
                    queue.put_nowait(status)
                except asyncio.QueueFull:
                    logger.warning(f"Watcher queue full for service '{service}'")

    def get_service_status(
        self,
        service: str
    ) -> health_pb2.HealthCheckResponse.ServingStatus:
        """
        Get the health status for a service.

        Args:
            service: Service name (empty string for overall server health)

        Returns:
            Health status (SERVING, NOT_SERVING, SERVICE_UNKNOWN, etc.)
        """
        if service in self._status:
            return self._status[service]
        else:
            return health_pb2.HealthCheckResponse.SERVICE_UNKNOWN

    async def Check(
        self,
        request: health_pb2.HealthCheckRequest,
        context: grpc.aio.ServicerContext
    ) -> health_pb2.HealthCheckResponse:
        """
        Unary RPC to check the health status of a service.

        Args:
            request: Health check request with service name
            context: gRPC context

        Returns:
            Health check response with serving status
        """
        service = request.service
        status = self.get_service_status(service)

        logger.debug(f"Health check for service '{service}': {status}")

        return health_pb2.HealthCheckResponse(status=status)

    async def Watch(
        self,
        request: health_pb2.HealthCheckRequest,
        context: grpc.aio.ServicerContext
    ) -> AsyncIterator[health_pb2.HealthCheckResponse]:
        """
        Streaming RPC to watch the health status of a service.

        The server will immediately send the current status, then stream
        updates whenever the status changes.

        Args:
            request: Health check request with service name
            context: gRPC context

        Yields:
            Health check responses with serving status updates
        """
        service = request.service

        # Create a queue for this watcher
        queue = asyncio.Queue(maxsize=10)

        # Register the watcher
        if service not in self._watchers:
            self._watchers[service] = []
        self._watchers[service].append(queue)

        try:
            # Send initial status
            initial_status = self.get_service_status(service)
            logger.debug(f"Starting health watch for service '{service}': {initial_status}")
            yield health_pb2.HealthCheckResponse(status=initial_status)

            # Stream updates
            while not context.cancelled():
                try:
                    # Wait for status updates with timeout
                    status = await asyncio.wait_for(queue.get(), timeout=1.0)
                    logger.debug(f"Health status update for service '{service}': {status}")
                    yield health_pb2.HealthCheckResponse(status=status)
                except asyncio.TimeoutError:
                    # No update, continue waiting
                    continue

        except asyncio.CancelledError:
            logger.debug(f"Health watch cancelled for service '{service}'")
        finally:
            # Unregister the watcher
            if service in self._watchers:
                try:
                    self._watchers[service].remove(queue)
                except ValueError:
                    pass
                # Clean up empty watcher lists
                if not self._watchers[service]:
                    del self._watchers[service]
