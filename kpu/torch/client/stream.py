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
import threading
from dataclasses import dataclass
from typing import Optional

try:
    import grpc
except ImportError as e:
    raise ImportError(f"grpcio package is required: {e}")

try:
    from kpu.torch.server import service_pb2
    from kpu.torch.server import service_pb2_grpc
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
    future: asyncio.Future
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
        self._pending_lock = threading.Lock()

        # Stream state
        self._request_queue: Optional[asyncio.Queue] = None
        self._sender_task: Optional[asyncio.Task] = None
        self._receiver_task: Optional[asyncio.Task] = None
        self._stream_call: Optional[grpc.aio.StreamStreamCall] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._started = False
        self._closing = False

        # Sync support
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

    async def _sender_loop(self) -> None:
        """Send requests from queue to the stream."""
        try:
            async def request_generator():
                while not self._closing:
                    # Efficient await - no busy-wait polling
                    request = await self._request_queue.get()
                    if request is None:
                        # Shutdown signal
                        break
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
                with self._pending_lock:
                    if not self._pending:
                        logger.warning("Received response but no pending requests")
                        continue
                    pending = self._pending.popleft()

                if response.success:
                    pending.future.set_result(response)
                else:
                    error = RuntimeError(
                        f"Stream operation failed: {response.error_message}"
                    )
                    pending.future.set_exception(error)
                    # Track error for sync
                    self._last_error = error

        except grpc.RpcError as e:
            if not self._closing:
                logger.error(f"Stream receiver error: {e}")
                self._fail_all_pending(str(e))
        except Exception as e:
            if not self._closing:
                logger.error(f"Unexpected stream receiver error: {e}")
                self._fail_all_pending(str(e))

    def _fail_all_pending(self, error_message: str) -> None:
        """Fail all pending requests with an error."""
        with self._pending_lock:
            for pending in self._pending:
                if not pending.future.done():
                    pending.future.set_exception(
                        RuntimeError(f"Stream error: {error_message}")
                    )
            self._pending.clear()

    def _submit_request(
        self, stream_request: service_pb2.StreamRequest, request_type: str
    ) -> asyncio.Future:
        """
        Submit a request to the stream (internal helper).

        Must be called from the event loop thread.

        Args:
            stream_request: StreamRequest to submit
            request_type: Type of request for tracking

        Returns:
            Future that resolves to StreamResponse
        """
        if self._loop is None:
            raise RuntimeError("StreamManager not started")

        future = self._loop.create_future()

        with self._pending_lock:
            self._pending.append(PendingRequest(
                future=future,
                request_type=request_type,
            ))

        self._request_queue.put_nowait(stream_request)

        return future

    def submit_execute_aten(
        self, request: service_pb2.ExecuteAtenRequest
    ) -> asyncio.Future:
        """
        Submit an execute_aten request.

        Args:
            request: ExecuteAtenRequest to submit

        Returns:
            Future that resolves to StreamResponse
        """
        stream_request = service_pb2.StreamRequest(execute_aten=request)
        return self._submit_request(stream_request, "execute_aten")

    def submit_copy_tensor(
        self, request: service_pb2.CopyTensorRequest
    ) -> asyncio.Future:
        """
        Submit a copy_tensor request.

        Args:
            request: CopyTensorRequest to submit

        Returns:
            Future that resolves to StreamResponse
        """
        stream_request = service_pb2.StreamRequest(copy_tensor=request)
        return self._submit_request(stream_request, "copy_tensor")

    def submit_delete_tensors(
        self, request: service_pb2.DeleteTensorsRequest
    ) -> asyncio.Future:
        """
        Submit a delete_tensors request.

        Args:
            request: DeleteTensorsRequest to submit

        Returns:
            Future that resolves to StreamResponse
        """
        stream_request = service_pb2.StreamRequest(delete_tensors=request)
        return self._submit_request(stream_request, "delete_tensors")

    def submit_update_tensor(
        self, request: service_pb2.UpdateTensorRequest
    ) -> asyncio.Future:
        """
        Submit an update_tensor request.

        For large tensors, automatically splits into multiple
        chunked messages to avoid gRPC message size limits.

        Args:
            request: UpdateTensorRequest to submit

        Returns:
            Future that resolves to StreamResponse
        """
        data_size = len(request.data)

        if data_size <= STREAM_CHUNK_SIZE:
            # Small tensor: single message
            stream_request = service_pb2.StreamRequest(update_tensor=request)
            return self._submit_request(stream_request, "update_tensor")

        # Large tensor: split into chunks
        return self._submit_chunked_update_tensor(request)

    def _submit_chunked_update_tensor(
        self, request: service_pb2.UpdateTensorRequest
    ) -> asyncio.Future:
        """
        Submit a large update_tensor request as multiple chunks.

        Only the last chunk is tracked for response; intermediate chunks
        are enqueued without tracking to preserve FIFO ordering.

        Args:
            request: UpdateTensorRequest with large data payload

        Returns:
            Future that resolves to StreamResponse (for final chunk)
        """
        data = request.data
        total_size = len(data)
        total_chunks = (total_size + STREAM_CHUNK_SIZE - 1) // STREAM_CHUNK_SIZE

        mv = memoryview(data)

        for chunk_number in range(total_chunks):
            start = chunk_number * STREAM_CHUNK_SIZE
            end = min(start + STREAM_CHUNK_SIZE, total_size)
            chunk_data = bytes(mv[start:end])

            # Build chunk request (metadata only on first chunk)
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

            stream_request = service_pb2.StreamRequest(
                update_tensor=chunk_request,
                chunk_number=chunk_number,
                total_chunks=total_chunks,
                total_bytes=total_size if chunk_number == 0 else 0,
            )

            if chunk_number == total_chunks - 1:
                # Last chunk: track for response
                return self._submit_request(stream_request, "update_tensor")
            else:
                # Intermediate: enqueue without tracking
                self._request_queue.put_nowait(stream_request)

    def submit_get_tensor(
        self, request: service_pb2.GetTensorRequest
    ) -> asyncio.Future:
        """
        Submit a get_tensor request.

        Args:
            request: GetTensorRequest to submit

        Returns:
            Future that resolves to StreamResponse with tensor data
        """
        stream_request = service_pb2.StreamRequest(get_tensor=request)
        return self._submit_request(stream_request, "get_tensor")

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
