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
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional

from skytorch.torch.profiler import PROFILING_ENABLED

try:
    import grpc
except ImportError as e:
    raise ImportError(f"grpcio package is required: {e}")

try:
    from skytorch.torch.server import service_pb2
    from skytorch.torch.server import service_pb2_grpc
except ImportError:
    raise ImportError("Generated gRPC code not found. Run hack/gen-grpc-proto.sh first.")

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

        # Batch buffer for execute_aten operations
        self._batch_buffer: list[service_pb2.ExecuteAtenRequest] = []
        self._flush_scheduled: bool = False
        self._batch_flush_timer = None

        # Batch buffer for raw binary execute_aten operations (C++ fast path)
        self._raw_batch_buffer: list[bytes] = []
        self._raw_flush_scheduled: bool = False
        self._raw_flush_timer = None

        # Main-thread submission buffer: collects ops from all submit methods
        # under a single lock, preserving FIFO ordering while reducing
        # call_soon_threadsafe calls by ~98%. Each entry is either:
        #   ("raw", bytes)    — raw binary execute_aten
        #   ("req", StreamRequest) — non-batched request (copy, update, register)
        self._mt_ops: list[tuple] = []
        self._mt_lock = threading.Lock()
        self._mt_wake_pending: bool = False

        # Buffer for batched delete tensor IDs
        self._delete_buffer: list[int] = []
        self._delete_flush_scheduled: bool = False

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
        if pending is not None:
            self._pending.append(pending)
        self._request_queue.put_nowait(stream_request)

    def _enqueue_batch(self, pending, stream_requests):
        """Enqueue multiple requests with one pending entry. Must run on the event loop thread."""
        for request in stream_requests:
            self._request_queue.put_nowait(request)
        if pending is not None:
            self._pending.append(pending)

    # Flush threshold: when this many ops are buffered, flush immediately.
    # Override with SKYTORCH_BATCH_THRESHOLD env var.
    _BATCH_FLUSH_THRESHOLD = int(os.environ.get("SKYTORCH_BATCH_THRESHOLD", "64"))

    # Coalescing delay: defer partial-batch flush by this many seconds,
    # allowing more ops to accumulate and amortize per-message gRPC overhead.
    # At C++ dispatch rate (~17 us/op), 1 ms ≈ 59 ops per batch.
    # Sync points force an immediate flush, so this never adds sync latency.
    # Override with SKYTORCH_BATCH_COALESCE_MS env var (in milliseconds).
    _BATCH_COALESCE_DELAY = float(os.environ.get("SKYTORCH_BATCH_COALESCE_MS", "2")) / 1000.0

    def _enqueue_execute_aten(self, request: service_pb2.ExecuteAtenRequest) -> None:
        """Buffer an execute_aten request for batching. Must run on the event loop thread."""
        self._batch_buffer.append(request)
        if len(self._batch_buffer) >= self._BATCH_FLUSH_THRESHOLD:
            self._flush_batch()
        elif not self._flush_scheduled:
            self._flush_scheduled = True
            self._batch_flush_timer = self._loop.call_later(
                self._BATCH_COALESCE_DELAY, self._flush_batch
            )

    def _enqueue_execute_aten_bytes(self, raw_bytes: bytes) -> None:
        """Buffer a raw binary execute_aten request for batching. Must run on the event loop thread."""
        self._raw_batch_buffer.append(raw_bytes)
        if len(self._raw_batch_buffer) >= self._BATCH_FLUSH_THRESHOLD:
            self._flush_raw_batch()
        elif not self._raw_flush_scheduled:
            self._raw_flush_scheduled = True
            self._raw_flush_timer = self._loop.call_later(
                self._BATCH_COALESCE_DELAY, self._flush_raw_batch
            )

    def _enqueue_delete_ids(self, tensor_ids: list[int]) -> None:
        """Buffer tensor IDs for batched deletion. Must run on event loop thread."""
        self._delete_buffer.extend(tensor_ids)
        if not self._delete_flush_scheduled:
            self._delete_flush_scheduled = True
            self._loop.call_soon(self._flush_deletes)

    def _flush_deletes(self) -> None:
        """Flush buffered delete tensor IDs. Must run on event loop thread."""
        self._delete_flush_scheduled = False
        if not self._delete_buffer:
            return
        # If a batch flush is pending, defer: the pending _flush_raw_batch or
        # _flush_batch will call _flush_deletes after sending the batch,
        # ensuring deletes arrive at the server AFTER ops that still reference
        # those tensors.
        if self._raw_flush_scheduled or self._flush_scheduled:
            return
        request = service_pb2.DeleteTensorsRequest(tensor_ids=self._delete_buffer)
        stream_request = service_pb2.StreamRequest(delete_tensors=request)
        self._request_queue.put_nowait(stream_request)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Flushing batched delete of {len(self._delete_buffer)} tensors")
        self._delete_buffer = []

    def _flush_batch(self) -> None:
        """Flush buffered execute_aten requests as a batch. Must run on the event loop thread."""
        self._flush_scheduled = False
        if self._batch_flush_timer is not None:
            self._batch_flush_timer.cancel()
            self._batch_flush_timer = None
        if not self._batch_buffer:
            self._flush_deletes()
            return

        if len(self._batch_buffer) == 1:
            # Single request: send as regular execute_aten to avoid wrapping overhead
            stream_request = service_pb2.StreamRequest(execute_aten=self._batch_buffer[0])
        else:
            # Multiple requests: send as batch
            batch = service_pb2.BatchedExecuteAtenRequest(operations=self._batch_buffer)
            stream_request = service_pb2.StreamRequest(batched_execute_aten=batch)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Flushing batch of {len(self._batch_buffer)} execute_aten ops")

        self._batch_buffer = []
        self._request_queue.put_nowait(stream_request)
        self._flush_deletes()

    def _flush_raw_batch(self) -> None:
        """Flush buffered raw binary execute_aten requests. Must run on the event loop thread."""
        self._raw_flush_scheduled = False
        if self._raw_flush_timer is not None:
            self._raw_flush_timer.cancel()
            self._raw_flush_timer = None
        if not self._raw_batch_buffer:
            self._flush_deletes()
            return

        if PROFILING_ENABLED:
            from skytorch.torch.profiler import ClientProfiler

            _prof = ClientProfiler.get()
            _batch_size = len(self._raw_batch_buffer)
            _prof.batch_count.add_count()
            _prof.batch_size_total += _batch_size
            if _batch_size > _prof.batch_size_max:
                _prof.batch_size_max = _batch_size

        if len(self._raw_batch_buffer) == 1:
            # Single request: send as raw_execute_aten
            stream_request = service_pb2.StreamRequest(raw_execute_aten=self._raw_batch_buffer[0])
        else:
            # Multiple requests: concatenate into raw_batched_execute_aten
            # Each request is prefixed with its length as uint32 LE
            import struct

            parts = []
            for raw in self._raw_batch_buffer:
                parts.append(struct.pack("<I", len(raw)))
                parts.append(raw)
            stream_request = service_pb2.StreamRequest(raw_batched_execute_aten=b"".join(parts))
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Flushing raw batch of {len(self._raw_batch_buffer)} execute_aten ops"
                )

        self._raw_batch_buffer = []
        self._request_queue.put_nowait(stream_request)
        self._flush_deletes()

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

                # Check for deferred error from fire-and-forget operations
                if response.error_message and not response.success:
                    error = RuntimeError(f"Stream operation failed: {response.error_message}")
                    if pending.future is not None:
                        pending.future.set_exception(error)
                    self.record_error(error)
                elif pending.future is not None:
                    pending.future.set_result(response)

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
                pending.future.set_exception(RuntimeError(f"Stream error: {error_message}"))
        self._pending.clear()

    def _drain_mt_ops(self) -> None:
        """Drain the main-thread ops buffer. Must run on the event loop thread."""
        with self._mt_lock:
            if not self._mt_ops:
                self._mt_wake_pending = False
                return
            batch = self._mt_ops
            self._mt_ops = []
            self._mt_wake_pending = False
        self._process_mt_ops(batch)

    def _process_mt_ops(self, ops: list[tuple]) -> None:
        """Process a list of (type, data) ops in order. Must run on the event loop thread."""
        for op_type, data in ops:
            if op_type == "raw":
                self._enqueue_execute_aten_bytes(data)
            elif op_type == "req":
                self._enqueue_with_flush(data)

    def _flush_mt_ops(self) -> None:
        """Drain main-thread ops buffer at sync points. Must run on the event loop thread."""
        with self._mt_lock:
            if not self._mt_ops:
                self._mt_wake_pending = False
                return
            batch = self._mt_ops
            self._mt_ops = []
            self._mt_wake_pending = False
        self._process_mt_ops(batch)

    def _enqueue_with_flush(self, stream_request):
        """Flush any pending batch, then enqueue a request. Must run on the event loop thread."""
        self._flush_batch()
        self._flush_raw_batch()
        self._request_queue.put_nowait(stream_request)

    def _submit_request(self, stream_request: service_pb2.StreamRequest, request_type: str) -> None:
        """
        Submit a fire-and-forget request to the stream (callable from any thread).

        Routes through the unified main-thread ops buffer to preserve FIFO
        ordering with submit_execute_aten_bytes calls.

        Args:
            stream_request: StreamRequest to submit
            request_type: Type of request for tracking
        """
        with self._mt_lock:
            self._mt_ops.append(("req", stream_request))
            if not self._mt_wake_pending:
                self._mt_wake_pending = True
                self._loop.call_soon_threadsafe(self._drain_mt_ops)

    def submit_execute_aten(self, request: service_pb2.ExecuteAtenRequest) -> None:
        """Submit a fire-and-forget execute_aten request (callable from any thread).

        Requests are buffered and flushed as a batch when the event loop
        processes pending callbacks, reducing per-operation gRPC overhead.
        """
        if logger.isEnabledFor(logging.DEBUG):
            input_tensor_ids = [
                arg.tensor.tensor_id for arg in request.args if arg.WhichOneof("value") == "tensor"
            ]
            output_tensor_ids = [ref.tensor_id for ref in request.outputs]
            logger.debug(
                f"Executing {request.op_name} | "
                f"inputs={input_tensor_ids} | outputs={output_tensor_ids}"
            )
        self._loop.call_soon_threadsafe(self._enqueue_execute_aten, request)

    def submit_execute_aten_bytes(self, raw_bytes: bytes) -> None:
        """Submit a fire-and-forget binary-serialized execute_aten request.

        Like submit_execute_aten but takes pre-serialized bytes from
        the C++ request builder, avoiding Python protobuf overhead entirely.
        Callable from any thread.

        Ops are collected in a unified main-thread buffer along with other
        submit calls, preserving FIFO ordering while reducing
        call_soon_threadsafe wake-ups by ~98%.
        """
        with self._mt_lock:
            self._mt_ops.append(("raw", raw_bytes))
            if not self._mt_wake_pending:
                self._mt_wake_pending = True
                self._loop.call_soon_threadsafe(self._drain_mt_ops)

    def submit_copy_tensor(self, request: service_pb2.CopyTensorRequest) -> None:
        """Submit a fire-and-forget copy_tensor request (callable from any thread)."""
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Copying tensor {request.src_tensor_id} " f"to tensor {request.dst_tensor_id}"
            )
        stream_request = service_pb2.StreamRequest(copy_tensor=request)
        self._submit_request(stream_request, "copy_tensor")

    def submit_register_tensors(self, request: service_pb2.RegisterTensorsRequest) -> None:
        """Submit a fire-and-forget register_tensors request (callable from any thread)."""
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Registering {len(request.registrations)} tensors")
        stream_request = service_pb2.StreamRequest(register_tensors=request)
        self._submit_request(stream_request, "register_tensors")

    def submit_delete_tensors(self, request: service_pb2.DeleteTensorsRequest) -> None:
        """Buffer delete_tensors for batched submission (callable from any thread)."""
        tensor_ids = list(request.tensor_ids)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Buffering delete for tensors {tensor_ids}")
        self._loop.call_soon_threadsafe(self._enqueue_delete_ids, tensor_ids)

    def submit_update_tensor(self, request: service_pb2.UpdateTensorRequest) -> None:
        """Submit a fire-and-forget update_tensor request (callable from any thread)."""
        data_size = len(request.data)

        if logger.isEnabledFor(logging.DEBUG):
            chunked = data_size > STREAM_CHUNK_SIZE
            msg = (
                f"Updating tensor {request.tensor_id} | "
                f"shape={list(request.shape)} | "
                f"size={data_size} bytes"
            )
            if chunked:
                num_chunks = (data_size + STREAM_CHUNK_SIZE - 1) // STREAM_CHUNK_SIZE
                msg += f" | chunks={num_chunks}"
            logger.debug(msg)

        if data_size <= STREAM_CHUNK_SIZE:
            stream_request = service_pb2.StreamRequest(update_tensor=request)
            self._submit_request(stream_request, "update_tensor")
        else:
            self._submit_chunked_update_tensor(request)

    def _submit_chunked_update_tensor(self, request: service_pb2.UpdateTensorRequest) -> None:
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

            stream_requests.append(
                service_pb2.StreamRequest(
                    update_tensor=chunk_request,
                    chunk_number=chunk_number,
                    total_chunks=total_chunks,
                    total_bytes=total_size if chunk_number == 0 else 0,
                )
            )

        def _flush_and_enqueue_batch():
            self._flush_mt_ops()
            self._flush_batch()
            self._flush_raw_batch()
            for request in stream_requests:
                self._request_queue.put_nowait(request)

        self._loop.call_soon_threadsafe(_flush_and_enqueue_batch)

    def submit_get_tensor(self, request: service_pb2.GetTensorRequest) -> asyncio.Future:
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

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Getting tensor {request.tensor_id}")

        if PROFILING_ENABLED:
            _t_flush_start = time.perf_counter_ns()
            _mt_ops_count = len(self._mt_ops)

        # Flush any pending batches before sync operation
        self._flush_mt_ops()
        self._flush_batch()
        self._flush_raw_batch()
        self._flush_deletes()

        if PROFILING_ENABLED:
            from skytorch.torch.profiler import ClientProfiler

            _t_flush_end = time.perf_counter_ns()
            _prof = ClientProfiler.get()
            _prof.sync_flush.add(_t_flush_end - _t_flush_start)
            _prof.sync_mt_ops_total += _mt_ops_count
            _qdepth = self._request_queue.qsize()
            _prof.sync_queue_depth_total += _qdepth
            if _qdepth > _prof.sync_queue_depth_max:
                _prof.sync_queue_depth_max = _qdepth

        future = self._loop.create_future()
        stream_request = service_pb2.StreamRequest(get_tensor=request)

        self._pending.append(
            PendingRequest(
                future=future,
                request_type="get_tensor",
            )
        )
        self._request_queue.put_nowait(stream_request)

        if PROFILING_ENABLED:
            _t_enqueue = time.perf_counter_ns()

        async def _with_error_check():
            response = await future
            if PROFILING_ENABLED:
                _t_resolved = time.perf_counter_ns()
                _wait_ns = _t_resolved - _t_enqueue
                _prof.sync_wait.add(_wait_ns)
                # Decompose sync_wait using server-provided timing
                _server_total = response.server_backlog_ns + response.server_handle_ns
                _network_rtt = _wait_ns - _server_total
                _prof.sync_network_rtt.add(max(0, _network_rtt))
                _prof.sync_server_backlog.add(response.server_backlog_ns)
                _prof.sync_server_handle.add(response.server_handle_ns)
            # FIFO ordering guarantees all prior fire-and-forget errors are recorded
            self.check_error()
            return response

        return _with_error_check()

    def submit_get_scalar(self, request: service_pb2.GetScalarRequest) -> asyncio.Future:
        """
        Submit a get_scalar request.

        Must be called from the event loop thread.

        Args:
            request: GetScalarRequest to submit

        Returns:
            Awaitable that resolves to StreamResponse with scalar value,
            after checking for deferred errors from prior operations
        """
        if self._loop is None:
            raise RuntimeError("StreamManager not started")

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Getting scalar for tensor {request.tensor_id}")

        if PROFILING_ENABLED:
            _t_flush_start = time.perf_counter_ns()
            _mt_ops_count = len(self._mt_ops)

        # Flush any pending batches before sync operation
        self._flush_mt_ops()
        self._flush_batch()
        self._flush_raw_batch()
        self._flush_deletes()

        if PROFILING_ENABLED:
            from skytorch.torch.profiler import ClientProfiler

            _t_flush_end = time.perf_counter_ns()
            _prof = ClientProfiler.get()
            _prof.sync_flush.add(_t_flush_end - _t_flush_start)
            _prof.sync_mt_ops_total += _mt_ops_count
            _qdepth = self._request_queue.qsize()
            _prof.sync_queue_depth_total += _qdepth
            if _qdepth > _prof.sync_queue_depth_max:
                _prof.sync_queue_depth_max = _qdepth

        future = self._loop.create_future()
        stream_request = service_pb2.StreamRequest(get_scalar=request)

        self._pending.append(
            PendingRequest(
                future=future,
                request_type="get_scalar",
            )
        )
        self._request_queue.put_nowait(stream_request)

        if PROFILING_ENABLED:
            _t_enqueue = time.perf_counter_ns()

        async def _with_error_check():
            response = await future
            if PROFILING_ENABLED:
                _t_resolved = time.perf_counter_ns()
                _wait_ns = _t_resolved - _t_enqueue
                _prof.sync_wait.add(_wait_ns)
                # Decompose sync_wait using server-provided timing
                _server_total = response.server_backlog_ns + response.server_handle_ns
                _network_rtt = _wait_ns - _server_total
                _prof.sync_network_rtt.add(max(0, _network_rtt))
                _prof.sync_server_backlog.add(response.server_backlog_ns)
                _prof.sync_server_handle.add(response.server_handle_ns)
            # FIFO ordering guarantees all prior fire-and-forget errors are recorded
            self.check_error()
            return response

        return _with_error_check()

    async def close(self) -> None:
        """Gracefully close the stream."""
        if not self._started:
            return

        # Flush pending batches before closing
        self._flush_mt_ops()
        self._flush_batch()
        self._flush_raw_batch()
        self._flush_deletes()

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

        if PROFILING_ENABLED:
            from skytorch.torch.profiler import ClientProfiler

            ClientProfiler.get().print_summary()
            ClientProfiler.reset()
