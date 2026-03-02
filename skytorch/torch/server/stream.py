"""Worker thread for StreamOperations."""

import logging
import struct
import time

logger = logging.getLogger(__name__)

try:
    from skytorch.torch.server._C import (
        execute_raw_aten_inline as _cpp_execute_raw_aten_inline,
        execute_raw_batched_aten_inline as _cpp_execute_raw_batched_aten_inline,
        batch_has_special_op as _cpp_batch_has_special_op,
        set_server_profiling_enabled as _cpp_set_server_profiling_enabled,
    )

    _USE_CPP_PARSER = True
except ImportError:
    _USE_CPP_PARSER = False

_STRUCT_I = struct.Struct("<I")  # uint32

# Work item type tags for the per-stream worker queue
RAW = 0
RAW_BATCH = 1
SYNC = 2
FF_REQUEST = 3
CHUNK = 4
SHUTDOWN = 5

# Marker bytes for non-ATen requests embedded in the raw binary ATen stream.
# Safe because the first byte of an ATen binary request is num_args (uint8)
# and no ATen op has 254+ arguments.
_MODULE_FORWARD_MARKER = 0xFF
_COPY_TENSOR_MARKER = 0xFE


def _batch_has_special_op(raw_data: bytes) -> bool:
    """Scan a raw batched payload for module_forward marker (0xFF).

    The batch format is [uint32_len][op_data]... — we check the first byte
    of each op_data segment. Only 0xFF (module forward) triggers mixed batch;
    0xFE (copy_tensor) is handled inline in the C++ batch path.
    """
    pos = 0
    n = len(raw_data)
    while pos < n:
        op_len = _STRUCT_I.unpack_from(raw_data, pos)[0]
        pos += 4
        if raw_data[pos] == _MODULE_FORWARD_MARKER:
            return True
        pos += op_len
    return False


def _execute_mixed_batch(raw_data: bytes, servicer) -> int:
    """Execute a raw batch containing module_forward ops mixed with ATen/copy ops.

    Splits the batch into contiguous ATen segments (processed via C++ batch path
    with batch-level GIL release) and module_forward ops (handled individually).
    Copy_tensor ops (0xFE) are included in ATen segments since the C++ batch
    path handles them inline.

    Returns the total number of ops executed.
    """
    from skytorch.torch.server import service_pb2

    pos = 0
    n = len(raw_data)
    n_ops = 0
    segment_start = -1  # start offset of current ATen segment, -1 = no segment
    store = servicer.tensor_manager.store if _USE_CPP_PARSER else None

    while pos < n:
        op_len = _STRUCT_I.unpack_from(raw_data, pos)[0]
        marker = raw_data[pos + 4]  # first byte of op_data

        if marker == _MODULE_FORWARD_MARKER:
            # Flush accumulated ATen segment via C++ batch path
            if segment_start >= 0:
                if _USE_CPP_PARSER:
                    n_ops += _cpp_execute_raw_batched_aten_inline(
                        raw_data[segment_start:pos], store
                    )
                else:
                    n_ops += _execute_segment_python(
                        raw_data, segment_start, pos, servicer
                    )
                segment_start = -1

            # Handle module_forward individually
            op_data = raw_data[pos + 4 : pos + 4 + op_len]
            fwd_request = service_pb2.ExecuteModuleForwardRequest()
            fwd_request.ParseFromString(op_data[1:])
            servicer._handle_execute_module_forward_ff(fwd_request)
            n_ops += 1
            pos += 4 + op_len
        else:
            # ATen op or copy_tensor — accumulate into segment
            if segment_start < 0:
                segment_start = pos
            pos += 4 + op_len

    # Flush final segment
    if segment_start >= 0:
        if _USE_CPP_PARSER:
            n_ops += _cpp_execute_raw_batched_aten_inline(
                raw_data[segment_start:n], store
            )
        else:
            n_ops += _execute_segment_python(
                raw_data, segment_start, n, servicer
            )

    return n_ops


def _execute_segment_python(
    raw_data: bytes, start: int, end: int, servicer
) -> int:
    """Fallback: execute an ATen segment op-by-op via Python (no C++ parser)."""
    pos = start
    n_ops = 0
    while pos < end:
        op_len = _STRUCT_I.unpack_from(raw_data, pos)[0]
        pos += 4
        op_data = raw_data[pos : pos + op_len]
        pos += op_len
        if op_data[0] == _COPY_TENSOR_MARKER:
            from skytorch.torch.server import service_pb2

            copy_request = service_pb2.CopyTensorRequest()
            copy_request.ParseFromString(op_data[1:])
            servicer._copy_tensor_sync(copy_request)
        else:
            servicer._execute_raw_aten_inline(op_data)
        n_ops += 1
    return n_ops


def stream_worker(work_queue, servicer, loop, server_profiler):
    """Worker thread for StreamOperations — processes ops from the queue.

    Runs in a dedicated thread per stream, allowing gRPC I/O on the event loop
    to overlap with op execution (C++ code holding the GIL).
    """
    chunk_state = [None]

    # Enable C++ per-phase profiling when server profiler is active
    if _USE_CPP_PARSER and server_profiler is not None:
        _cpp_set_server_profiling_enabled(True)

    if server_profiler is not None:
        _cycle_backlog_ops = 0
        _cycle_backlog_time_ns = 0
        _cycle_first_idle_ns = 0
        _cycle_started = False
        _last_was_exec = False

    while True:
        if server_profiler is not None:
            _t_wait = time.perf_counter_ns()

        item = work_queue.get()

        if server_profiler is not None:
            _t_recv = time.perf_counter_ns()
            _idle_ns = _t_recv - _t_wait
            server_profiler.idle_time.add(_idle_ns)

        tag = item[0]

        if tag == SHUTDOWN:
            break
        elif tag == RAW:
            if servicer._deferred_error is not None:
                continue
            try:
                if server_profiler is not None:
                    _t0 = time.perf_counter_ns()
                    if not _cycle_started:
                        _cycle_first_idle_ns = _t_recv - _t_wait
                        _cycle_started = True
                    if _last_was_exec:
                        server_profiler.hot_idle.add(_idle_ns)

                data = item[1]
                if data[0] == _MODULE_FORWARD_MARKER:
                    from skytorch.torch.server import service_pb2

                    fwd_request = service_pb2.ExecuteModuleForwardRequest()
                    fwd_request.ParseFromString(data[1:])
                    servicer._handle_execute_module_forward_ff(fwd_request)
                elif data[0] == _COPY_TENSOR_MARKER:
                    from skytorch.torch.server import service_pb2

                    copy_request = service_pb2.CopyTensorRequest()
                    copy_request.ParseFromString(data[1:])
                    servicer._copy_tensor_sync(copy_request)
                elif _USE_CPP_PARSER:
                    _cpp_execute_raw_aten_inline(data, servicer.tensor_manager.store)
                else:
                    servicer._execute_raw_aten_inline(data)

                if server_profiler is not None:
                    _t1 = time.perf_counter_ns()
                    server_profiler.raw_execute.add(_t1 - _t0)
                    server_profiler.total_ops += 1
                    _cycle_backlog_ops += 1
                    _cycle_backlog_time_ns += _t1 - _t0
                    _last_was_exec = True
            except Exception as e:
                logger.error(f"Error in raw operation: {e}")
                if servicer._deferred_error is None:
                    servicer._deferred_error = str(e)
        elif tag == RAW_BATCH:
            if servicer._deferred_error is not None:
                continue
            try:
                if server_profiler is not None:
                    _t0 = time.perf_counter_ns()
                    if not _cycle_started:
                        _cycle_first_idle_ns = _t_recv - _t_wait
                        _cycle_started = True
                    if _last_was_exec:
                        server_profiler.hot_idle.add(_idle_ns)

                raw_data = item[1]
                _n_ops = 0
                _is_mixed = False
                if _USE_CPP_PARSER and not _cpp_batch_has_special_op(raw_data):
                    _n_ops = _cpp_execute_raw_batched_aten_inline(
                        raw_data, servicer.tensor_manager.store
                    )
                else:
                    _is_mixed = True
                    _n_ops = _execute_mixed_batch(raw_data, servicer)

                if server_profiler is not None:
                    _t1 = time.perf_counter_ns()
                    server_profiler.raw_batched_execute.add(_t1 - _t0)
                    if _n_ops == 0:
                        # Fallback: count ops by scanning (shouldn't happen now)
                        raw_data = item[1]
                        _pos = 0
                        while _pos < len(raw_data):
                            _op_len = _STRUCT_I.unpack_from(raw_data, _pos)[0]
                            _pos += 4 + _op_len
                            _n_ops += 1
                    server_profiler.total_ops += _n_ops
                    if _is_mixed:
                        server_profiler.mixed_batch_calls += 1
                        server_profiler.mixed_batch_ops += _n_ops
                    _cycle_backlog_ops += _n_ops
                    _cycle_backlog_time_ns += _t1 - _t0
                    _last_was_exec = True
            except Exception as e:
                logger.error(f"Error in raw batch operation: {e}")
                if servicer._deferred_error is None:
                    servicer._deferred_error = str(e)
        elif tag == FF_REQUEST:
            if servicer._deferred_error is not None:
                # Still process delete_tensors to free GPU memory
                request_type = item[1].WhichOneof("request")
                if request_type != "delete_tensors":
                    continue
            servicer._handle_fire_and_forget_sync(item[1])
            if server_profiler is not None:
                _last_was_exec = False
        elif tag == CHUNK:
            servicer._handle_chunked_sync(item[1], chunk_state)
            if server_profiler is not None:
                _last_was_exec = False
        elif tag == SYNC:
            request, future = item[1], item[2]

            if server_profiler is not None:
                server_profiler.sync_cycle_count += 1
                server_profiler.sync_backlog_ops.add_count(_cycle_backlog_ops)
                if _cycle_backlog_time_ns > 0:
                    server_profiler.sync_backlog_time.add(_cycle_backlog_time_ns)
                if _cycle_started:
                    server_profiler.sync_idle_before.add(_cycle_first_idle_ns)
                # Save backlog time before resetting for response embedding
                _response_backlog_ns = _cycle_backlog_time_ns
                _cycle_backlog_ops = 0
                _cycle_backlog_time_ns = 0
                _cycle_first_idle_ns = 0
                _cycle_started = False

                _t0 = time.perf_counter_ns()

            response = servicer._handle_single_request_sync(request, server_profiler)

            if server_profiler is not None:
                _t1 = time.perf_counter_ns()
                _handle_ns = _t1 - _t0
                server_profiler.sync_handle.add(_handle_ns)
                # Embed server-side timing in response for client decomposition
                response.server_backlog_ns = _response_backlog_ns
                response.server_handle_ns = _handle_ns

            loop.call_soon_threadsafe(future.set_result, response)
            if server_profiler is not None:
                _last_was_exec = False
