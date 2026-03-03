"""Worker thread for StreamOperations."""

import logging
import struct
import time

import torch

logger = logging.getLogger(__name__)

try:
    from skytorch.torch.server._C import (
        execute_raw_aten_inline as _cpp_execute_raw_aten_inline,
        execute_raw_batched_aten_inline as _cpp_execute_raw_batched_aten_inline,
        execute_raw_mixed_batch_inline as _cpp_execute_raw_mixed_batch_inline,
        batch_has_special_op as _cpp_batch_has_special_op,
        set_server_profiling_enabled as _cpp_set_server_profiling_enabled,
    )

    _USE_CPP_PARSER = True
except ImportError:
    _USE_CPP_PARSER = False

_STRUCT_I = struct.Struct("<I")  # uint32
_STRUCT_Q = struct.Struct("<Q")  # uint64
_STRUCT_H = struct.Struct("<H")  # uint16

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


def _parse_and_execute_module_forward_binary(data: bytes, pos: int, end: int, servicer) -> None:
    """Parse binary module_forward format and execute directly.

    Binary format (after 0xFF marker byte):
    [model_id:u64][path_len:u16][path:utf8][n_in:u8][in_ids:u64...][n_out:u8][out_ids:u64...]

    Avoids protobuf object creation and ParseFromString overhead.
    """
    # Skip marker byte
    p = pos + 1

    model_id = _STRUCT_Q.unpack_from(data, p)[0]
    p += 8

    path_len = _STRUCT_H.unpack_from(data, p)[0]
    p += 2
    module_path = data[p : p + path_len].decode("utf-8")
    p += path_len

    n_in = data[p]
    p += 1
    input_ids = struct.unpack_from(f"<{n_in}Q", data, p)
    p += n_in * 8

    n_out = data[p]
    p += 1
    output_ids = struct.unpack_from(f"<{n_out}Q", data, p)

    # Resolve module (cached)
    key = (model_id, module_path)
    module = servicer._module_cache.get(key)
    if module is None:
        model = servicer._retained_models.get(model_id)
        if model is None:
            raise RuntimeError(f"Model {model_id} not found")
        module = model
        if module_path:
            for part in module_path.split("."):
                if part.isdigit():
                    module = module[int(part)]
                else:
                    module = getattr(module, part)
        servicer._module_cache[key] = module

    # Get inputs from TensorStore directly
    store = servicer.tensor_manager.store
    if store is not None:
        inputs = [store.get(tid) for tid in input_ids]
    else:
        inputs = [servicer.tensor_manager.get(tid) for tid in input_ids]

    # Execute
    result = module(*inputs)

    # Normalize result
    if isinstance(result, torch.Tensor):
        result_tensors = [result]
    elif isinstance(result, (tuple, list)):
        result_tensors = [t for t in result if isinstance(t, torch.Tensor)]
    else:
        result_tensors = []

    # Validate and register outputs
    if len(result_tensors) != len(output_ids):
        raise RuntimeError(
            f"Module forward output count mismatch for {module_path}: "
            f"expected {len(output_ids)}, got {len(result_tensors)}"
        )
    if store is not None:
        for tid, tensor in zip(output_ids, result_tensors):
            if tensor is not None:
                store.set(tid, tensor)
    else:
        for tid, tensor in zip(output_ids, result_tensors):
            if tensor is not None:
                servicer.tensor_manager.register(tid, tensor)


def _parse_and_execute_copy_tensor_binary(data: bytes, servicer) -> None:
    """Parse binary copy_tensor and execute dst.copy_(src).

    Binary format (starting at 0xFE marker):
    [0xFE][src_id:u64][dst_id:u64][has_src:u8][src_meta...][has_dst:u8][dst_meta...]

    Metadata (if present) uses ATen binary tensor metadata format.
    For individual RAW ops, data starts at the 0xFE marker.
    For C++ batch path, copy_tensor is handled entirely in C++ (GIL-free).
    This function is only used for individual RAW ops and Python fallback.
    """
    p = 1  # skip 0xFE marker
    src_id = _STRUCT_Q.unpack_from(data, p)[0]
    p += 8
    dst_id = _STRUCT_Q.unpack_from(data, p)[0]
    p += 8

    # Skip metadata sections — for individual RAW ops, tensors should
    # already exist in the store (metadata was handled during batch processing).
    # The binary metadata format is the same as ATen ops, so we skip it
    # by reading through the fields.
    for _ in range(2):  # src_metadata, dst_metadata
        has_meta = data[p]
        p += 1
        if has_meta:
            # Skip: tensor_id(8) + ndim(1) + shape+stride(ndim*16) +
            #        dtype_len(1) + dtype + offset(8) + nbytes(8) +
            #        dt_len(1) + device_type + device_index(4) +
            #        has_ref(1) + ref(8 if has_ref)
            p += 8  # tensor_id
            ndim = data[p]
            p += 1 + ndim * 16  # ndim + shape + stride
            dtype_len = data[p]
            p += 1 + dtype_len  # dtype_len + dtype
            p += 16  # storage_offset + nbytes
            dt_len = data[p]
            p += 1 + dt_len + 4  # dt_len + device_type + device_index
            has_ref = data[p]
            p += 1
            if has_ref:
                p += 8  # tensor_ref

    # For individual RAW copy ops, use _copy_tensor_sync logic
    # (tensors must already exist)
    store = servicer.tensor_manager.store
    if store is not None:
        src = store.get(src_id)
        dst = store.get(dst_id)
    else:
        src = servicer.tensor_manager.get(src_id)
        dst = servicer.tensor_manager.get(dst_id)
    dst.copy_(src)


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
    if _USE_CPP_PARSER:
        # C++ handles scanning, segment splitting, and ATen batch execution.
        # Only module_forward ops call back to Python (no bytes copy for ATen segments).
        store = servicer.tensor_manager.store

        def _module_forward_cb(data, start, end):
            """Callback from C++ for module_forward ops — receives raw buffer offsets."""
            with torch.no_grad():
                _parse_and_execute_module_forward_binary(data, start, end, servicer)

        return _cpp_execute_raw_mixed_batch_inline(raw_data, store, _module_forward_cb)

    # Python fallback: scan and split manually
    pos = 0
    n = len(raw_data)
    n_ops = 0
    segment_start = -1

    with torch.no_grad():
        while pos < n:
            op_len = _STRUCT_I.unpack_from(raw_data, pos)[0]
            marker = raw_data[pos + 4]

            if marker == _MODULE_FORWARD_MARKER:
                if segment_start >= 0:
                    n_ops += _execute_segment_python(
                        raw_data, segment_start, pos, servicer
                    )
                    segment_start = -1

                _parse_and_execute_module_forward_binary(
                    raw_data, pos + 4, pos + 4 + op_len, servicer
                )
                n_ops += 1
                pos += 4 + op_len
            else:
                if segment_start < 0:
                    segment_start = pos
                pos += 4 + op_len

        if segment_start >= 0:
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
            _parse_and_execute_copy_tensor_binary(op_data, servicer)
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
                    with torch.no_grad():
                        _parse_and_execute_module_forward_binary(
                            data, 0, len(data), servicer
                        )
                elif data[0] == _COPY_TENSOR_MARKER:
                    if _USE_CPP_PARSER:
                        # Route through C++ batch path which handles 0xFE inline
                        # (including metadata creation via parse_and_create_tensor_gilfree)
                        batch = _STRUCT_I.pack(len(data)) + data
                        _cpp_execute_raw_batched_aten_inline(
                            batch, servicer.tensor_manager.store
                        )
                    else:
                        _parse_and_execute_copy_tensor_binary(data, servicer)
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
