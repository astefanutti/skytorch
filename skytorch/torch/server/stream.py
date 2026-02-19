"""Worker thread for StreamOperations."""

import struct
import time

try:
    from skytorch.torch.server._C import (
        execute_raw_aten_inline as _cpp_execute_raw_aten_inline,
        execute_raw_batched_aten_inline as _cpp_execute_raw_batched_aten_inline,
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


def stream_worker(work_queue, servicer, loop, server_profiler):
    """Worker thread for StreamOperations — processes ops from the queue.

    Runs in a dedicated thread per stream, allowing gRPC I/O on the event loop
    to overlap with op execution (C++ code holding the GIL).
    """
    chunk_state = [None]

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
            try:
                if server_profiler is not None:
                    _t0 = time.perf_counter_ns()
                    if not _cycle_started:
                        _cycle_first_idle_ns = _t_recv - _t_wait
                        _cycle_started = True
                    if _last_was_exec:
                        server_profiler.hot_idle.add(_idle_ns)

                if _USE_CPP_PARSER:
                    _cpp_execute_raw_aten_inline(item[1], servicer.tensor_manager.store)
                else:
                    servicer._execute_raw_aten_inline(item[1])

                if server_profiler is not None:
                    _t1 = time.perf_counter_ns()
                    server_profiler.raw_execute.add(_t1 - _t0)
                    server_profiler.total_ops += 1
                    _cycle_backlog_ops += 1
                    _cycle_backlog_time_ns += _t1 - _t0
                    _last_was_exec = True
            except Exception as e:
                servicer._deferred_error = str(e)
        elif tag == RAW_BATCH:
            try:
                if server_profiler is not None:
                    _t0 = time.perf_counter_ns()
                    if not _cycle_started:
                        _cycle_first_idle_ns = _t_recv - _t_wait
                        _cycle_started = True
                    if _last_was_exec:
                        server_profiler.hot_idle.add(_idle_ns)

                if _USE_CPP_PARSER:
                    _cpp_execute_raw_batched_aten_inline(
                        item[1], servicer.tensor_manager.store
                    )
                else:
                    raw_data = item[1]
                    pos = 0
                    while pos < len(raw_data):
                        op_len = _STRUCT_I.unpack_from(raw_data, pos)[0]
                        pos += 4
                        servicer._execute_raw_aten_inline(raw_data[pos : pos + op_len])
                        pos += op_len

                if server_profiler is not None:
                    _t1 = time.perf_counter_ns()
                    server_profiler.raw_batched_execute.add(_t1 - _t0)
                    raw_data = item[1]
                    _n_ops = 0
                    _pos = 0
                    while _pos < len(raw_data):
                        _op_len = _STRUCT_I.unpack_from(raw_data, _pos)[0]
                        _pos += 4 + _op_len
                        _n_ops += 1
                    server_profiler.total_ops += _n_ops
                    _cycle_backlog_ops += _n_ops
                    _cycle_backlog_time_ns += _t1 - _t0
                    _last_was_exec = True
            except Exception as e:
                servicer._deferred_error = str(e)
        elif tag == FF_REQUEST:
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
