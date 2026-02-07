"""
StreamManager for bidirectional gRPC streaming.

This module provides thread-safe bidirectional stream management for
low-latency tensor operations, enabling pipelining of requests without
waiting for individual responses.

All operations (execute_aten, copy_tensor, update_tensor, get_tensor, etc.)
go through a single ordered stream to ensure proper sequencing.
"""

from __future__ import annotations

import asyncio
import collections
import logging
from dataclasses import dataclass
from typing import Optional

try:
    import grpc
except ImportError as e:
    raise ImportError(f"grpcio package is required: {e}")

try:
    from skytorch.torch.server import service_pb2
    from skytorch.torch.server import service_pb2_grpc
except ImportError:
    raise ImportError(
        "Generated gRPC code not found. Run hack/gen-grpc-proto.sh first."
    )

from grpc.aio._typing import MetadataType

logger = logging.getLogger(__name__)

# Chunk size for large tensor payloads (1MB, matching serialization module)
STREAM_CHUNK_SIZE = 1024 * 1024


@dataclass
class PendingRequest:
    """Tracks a pending request awaiting response."""
    future: Optional[asyncio.Future]
    request_type: str


class StreamManager:
    """
    Bidirectional stream manager for pipelined operations.

    All tensor operations go through a single ordered stream to ensure
    proper sequencing. Responses are matched to requests by order (FIFO).

    Key features:
    - FIFO ordering of requests and responses
    - Efficient async queue (no busy-wait polling)
    - Automatic stream lifecycle management

    Note: All operations must be called from the asyncio event loop thread.
    """

    def __init__(
        self,
        stub: service_pb2_grpc.ServiceStub,
        metadata: Optional[MetadataType] = None,
    ):
        """
        Initialize the stream manager.

        Args:
            stub: gRPC service stub
            metadata: Optional metadata for requests
        """
        self._stub = stub
        self._metadata = metadata

        # Pending requests - FIFO queue of futures matched by order
        self._pending: collections.deque[PendingRequest] = collections.deque()

        # Stream state
        self._request_queue: Optional[asyncio.Queue] = None
        self._sender_task: Optional[asyncio.Task] = None
        self._receiver_task: Optional[asyncio.Task] = None
        self._stream_call: Optional[grpc.aio.StreamStreamCall] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._started = False
        self._closing = False

        # Deferred error from fire-and-forget operations
        self._last_error: Optional[Exception] = None

        # Shutdown signaling
        self._shutdown_event: Optional[asyncio.Event] = None

    async def start(self, loop: asyncio.AbstractEventLoop) -> None:
        """
        Start the bidirectional stream.

        Args:
            loop: Event loop to run stream tasks on
        """
        if self._started:
            return

        self._loop = loop
        self._request_queue = asyncio.Queue()
        self._shutdown_event = asyncio.Event()

        # Start sender and receiver tasks
        self._sender_task = asyncio.create_task(self._sender_loop())
        self._receiver_task = asyncio.create_task(self._receiver_loop())
        self._started = True

    def _enqueue(self, pending, stream_request):
        """Enqueue a single request. Must run on the event loop thread."""
        self._pending.append(pending)
        self._request_queue.put_nowait(stream_request)

    def _enqueue_batch(self, pending, stream_requests):
        """Enqueue multiple requests with one pending entry. Must run on the event loop thread."""
        for request in stream_requests:
            self._request_queue.put_nowait(request)
        self._pending.append(pending)

    async def _sender_loop(self) -> None:
        """Send requests from queue to the stream."""
        try:
            async def request_generator():
                while not self._closing:
                    request = await self._request_queue.get()
                    if request is None:
                        break
                    yield request
                    # Drain ready items without re-awaiting
                    while not self._request_queue.empty():
                        request = self._request_queue.get_nowait()
                        if request is None:
                            return
                        yield request

            # Start the bidirectional stream
            self._stream_call = self._stub.StreamOperations(
                request_generator(), metadata=self._metadata
            )

            # Wait for shutdown signal
            await self._shutdown_event.wait()

        except grpc.RpcError as e:
            logger.error(f"Stream sender error: {e}")
            self._fail_all_pending(str(e))
        except Exception as e:
            logger.error(f"Unexpected stream sender error: {e}")
            self._fail_all_pending(str(e))

    async def _receiver_loop(self) -> None:
        """Receive responses from the stream and resolve futures in order."""
        try:
            # Wait for stream to be established
            while self._stream_call is None and not self._closing:
                await asyncio.sleep(0.01)

            if self._stream_call is None:
                return

            async for response in self._stream_call:
                if not self._pending:
                    logger.warning("Received response but no pending requests")
                    continue
                pending = self._pending.popleft()

                if pending.future is None:
                    # Fire-and-forget: log errors but don't raise
                    if not response.success:
                        logger.error(
                            f"Fire-and-forget {pending.request_type} failed: "
                            f"{response.error_message}"
                        )
                        self._last_error = RuntimeError(response.error_message)
                elif response.success:
                    pending.future.set_result(response)
                else:
                    error = RuntimeError(
                        f"Stream operation failed: {response.error_message}"
                    )
                    pending.future.set_exception(error)
                    self.record_error(error)

        except grpc.RpcError as e:
            if not self._closing:
                logger.error(f"Stream receiver error: {e}")
                self._fail_all_pending(str(e))
        except Exception as e:
            if not self._closing:
                logger.error(f"Unexpected stream receiver error: {e}")
                self._fail_all_pending(str(e))

    def record_error(self, error: Exception) -> None:
        """Record error for deferred raising. First-error-wins semantics."""
        if self._last_error is None:
            self._last_error = error

    def check_error(self) -> None:
        """Raise deferred error if any, then clear it."""
        error = self._last_error
        if error is not None:
            self._last_error = None
            raise error

    def _fail_all_pending(self, error_message: str) -> None:
        """Fail all pending requests with an error."""
        for pending in self._pending:
            if pending.future is not None and not pending.future.done():
                pending.future.set_exception(
                    RuntimeError(f"Stream error: {error_message}")
                )
        self._pending.clear()

    def _submit_request(
        self, stream_request: service_pb2.StreamRequest, request_type: str
    ) -> None:
        """
        Submit a fire-and-forget request to the stream (callable from any thread).

        No future is created — the request is enqueued and the response is
        discarded (errors are logged by the receiver loop).

        Args:
            stream_request: StreamRequest to submit
            request_type: Type of request for tracking
        """
        pending = PendingRequest(future=None, request_type=request_type)
        self._loop.call_soon_threadsafe(self._enqueue, pending, stream_request)

    def submit_execute_aten(
        self, request: service_pb2.ExecuteAtenRequest
    ) -> None:
        """Submit a fire-and-forget execute_aten request (callable from any thread)."""
        stream_request = service_pb2.StreamRequest(execute_aten=request)
        self._submit_request(stream_request, "execute_aten")

    def submit_copy_tensor(
        self, request: service_pb2.CopyTensorRequest
    ) -> None:
        """Submit a fire-and-forget copy_tensor request (callable from any thread)."""
        stream_request = service_pb2.StreamRequest(copy_tensor=request)
        self._submit_request(stream_request, "copy_tensor")

    def submit_delete_tensors(
        self, request: service_pb2.DeleteTensorsRequest
    ) -> None:
        """Submit a fire-and-forget delete_tensors request (callable from any thread)."""
        stream_request = service_pb2.StreamRequest(delete_tensors=request)
        self._submit_request(stream_request, "delete_tensors")

    def submit_update_tensor(
        self, request: service_pb2.UpdateTensorRequest
    ) -> None:
        """Submit a fire-and-forget update_tensor request (callable from any thread)."""
        data_size = len(request.data)

        if data_size <= STREAM_CHUNK_SIZE:
            stream_request = service_pb2.StreamRequest(update_tensor=request)
            self._submit_request(stream_request, "update_tensor")
        else:
            self._submit_chunked_update_tensor(request)

    def _submit_chunked_update_tensor(
        self, request: service_pb2.UpdateTensorRequest
    ) -> None:
        """Submit a large update_tensor request as multiple fire-and-forget chunks."""
        data = request.data
        total_size = len(data)
        total_chunks = (total_size + STREAM_CHUNK_SIZE - 1) // STREAM_CHUNK_SIZE

        mv = memoryview(data)
        stream_requests = []

        for chunk_number in range(total_chunks):
            start = chunk_number * STREAM_CHUNK_SIZE
            end = min(start + STREAM_CHUNK_SIZE, total_size)
            chunk_data = bytes(mv[start:end])

            chunk_request = service_pb2.UpdateTensorRequest(
                tensor_id=request.tensor_id,
                data=chunk_data,
            )
            if chunk_number == 0:
                chunk_request.shape.extend(request.shape)
                chunk_request.dtype = request.dtype
                chunk_request.stride.extend(request.stride)
                chunk_request.storage_offset = request.storage_offset
                if request.HasField("metadata"):
                    chunk_request.metadata.CopyFrom(request.metadata)

            stream_requests.append(service_pb2.StreamRequest(
                update_tensor=chunk_request,
                chunk_number=chunk_number,
                total_chunks=total_chunks,
                total_bytes=total_size if chunk_number == 0 else 0,
            ))

        pending = PendingRequest(future=None, request_type="update_tensor")
        self._loop.call_soon_threadsafe(
            self._enqueue_batch, pending, stream_requests
        )

    def submit_get_tensor(
        self, request: service_pb2.GetTensorRequest
    ) -> asyncio.Future:
        """
        Submit a get_tensor request.

        Must be called from the event loop thread.

        Args:
            request: GetTensorRequest to submit

        Returns:
            Awaitable that resolves to StreamResponse with tensor data,
            after checking for deferred errors from prior operations
        """
        if self._loop is None:
            raise RuntimeError("StreamManager not started")

        future = self._loop.create_future()
        stream_request = service_pb2.StreamRequest(get_tensor=request)

        self._pending.append(PendingRequest(
            future=future,
            request_type="get_tensor",
        ))
        self._request_queue.put_nowait(stream_request)

        async def _with_error_check():
            response = await future
            # FIFO ordering guarantees all prior fire-and-forget errors are recorded
            self.check_error()
            return response

        return _with_error_check()

    async def close(self) -> None:
        """Gracefully close the stream."""
        if not self._started:
            return

        self._closing = True

        # Wake up sender loop
        if self._shutdown_event is not None:
            self._shutdown_event.set()

        # Signal sender to stop
        if self._request_queue is not None:
            self._request_queue.put_nowait(None)

        # Cancel stream call
        if self._stream_call is not None:
            self._stream_call.cancel()

        # Wait for tasks to complete
        if self._sender_task is not None:
            self._sender_task.cancel()
            try:
                await self._sender_task
            except asyncio.CancelledError:
                pass

        if self._receiver_task is not None:
            self._receiver_task.cancel()
            try:
                await self._receiver_task
            except asyncio.CancelledError:
                pass

        # Fail any remaining pending requests
        self._fail_all_pending("Stream closed")

        self._started = False
