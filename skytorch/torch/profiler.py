"""
SkyTorch profiling instrumentation.

Lightweight profiling gated by SKYTORCH_PROFILE=1 environment variable.
Overhead: ~200ns/op when enabled (~12ms total for 58K ops), zero-cost when disabled.
"""

import os
import sys

PROFILING_ENABLED = os.environ.get("SKYTORCH_PROFILE", "0") == "1"


class Counter:
    """Accumulates timing and count data."""

    __slots__ = ("total_ns", "count", "max_ns")

    def __init__(self):
        self.total_ns: int = 0
        self.count: int = 0
        self.max_ns: int = 0

    def add(self, ns: int) -> None:
        self.total_ns += ns
        self.count += 1
        if ns > self.max_ns:
            self.max_ns = ns

    def add_count(self, n: int = 1) -> None:
        self.count += n

    @property
    def avg_us(self) -> float:
        return (self.total_ns / self.count / 1000) if self.count else 0.0

    @property
    def total_ms(self) -> float:
        return self.total_ns / 1_000_000

    @property
    def max_ms(self) -> float:
        return self.max_ns / 1_000_000

    @property
    def avg_ms(self) -> float:
        return (self.total_ns / self.count / 1_000_000) if self.count else 0.0


class ClientProfiler:
    """Singleton profiler for client-side dispatch breakdown."""

    _instance = None

    def __init__(self):
        # Dispatch counters
        self.cache_key_build = Counter()
        self.output_creation = Counter()
        self.execute_dispatch = Counter()
        self.cpp_serialization = Counter()
        self.event_loop_submit = Counter()

        # Sync counters
        self.sync_total = Counter()

        # Sync phase breakdown
        self.sync_flush = Counter()
        self.sync_wait = Counter()

        # Sync wait decomposition (server-provided timing)
        self.sync_network_rtt = Counter()
        self.sync_server_backlog = Counter()
        self.sync_server_handle = Counter()

        # Scalar speculation stats
        self.scalar_speculative_hits: int = 0
        self.scalar_speculative_misses: int = 0

        # Sync buffer state (non-timing accumulators)
        self.sync_mt_ops_total: int = 0
        self.sync_queue_depth_total: int = 0
        self.sync_queue_depth_max: int = 0

        # Batch counters
        self.batch_count = Counter()
        self.batch_size_total: int = 0
        self.batch_size_max: int = 0

        # Inter-op gap
        self.inter_op_gap = Counter()
        self.last_dispatch_end: int = 0

        # Cache stats
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.total_ops: int = 0

        # Wall time
        self.first_dispatch_ns: int = 0
        self.last_dispatch_ns: int = 0

    @classmethod
    def get(cls) -> "ClientProfiler":
        if cls._instance is None:
            cls._instance = ClientProfiler()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    def print_summary(self) -> None:
        # Fetch C++ fast path counters
        try:
            from skytorch.torch.backend._C import _get_cpp_profile_counters

            cpp = _get_cpp_profile_counters()
        except (ImportError, AttributeError):
            cpp = None

        cpp_ops = cpp["fast_path_count"][1] if cpp else 0
        total = self.total_ops + cpp_ops

        if total == 0:
            return

        wall_ms = (self.last_dispatch_ns - self.first_dispatch_ns) / 1_000_000
        sync_pct = (self.sync_total.total_ms / wall_ms * 100) if wall_ms > 0 else 0

        lines = [
            "",
            "=== SkyTorch Client Profile ===",
            f"Ops: {total:,} ({self.cache_hits:,} cache hits, "
            f"{self.cache_misses:,} misses"
            + (f", {cpp_ops:,} C++ fast path" if cpp_ops > 0 else "")
            + ")",
            f"Syncs: {self.sync_total.count:,}",
            "",
            "Dispatch (per-op avg / total):",
            f"  Cache key build:    {self.cache_key_build.avg_us:6.1f} us  |  "
            f"{self.cache_key_build.total_ms:,.0f} ms",
            f"  Output creation:    {self.output_creation.avg_us:6.1f} us  |  "
            f"{self.output_creation.total_ms:,.0f} ms",
            f"  Execute dispatch:   {self.execute_dispatch.avg_us:6.1f} us  |  "
            f"{self.execute_dispatch.total_ms:,.0f} ms",
            f"    C++ serialize:    {self.cpp_serialization.avg_us:6.1f} us  |  "
            f"{self.cpp_serialization.total_ms:,.0f} ms",
            f"    Event loop sub:   {self.event_loop_submit.avg_us:6.1f} us  |  "
            f"{self.event_loop_submit.total_ms:,.0f} ms",
            f"  Inter-op gap:       {self.inter_op_gap.avg_us:6.1f} us  |  "
            f"{self.inter_op_gap.total_ms:,.0f} ms",
            "",
        ]

        if cpp and cpp_ops > 0:
            _iv_ns, _ = cpp["ivalue_to_py_ns"]
            _dc_ns, _ = cpp["dispatch_cached_ns"]
            _sm_ns, _ = cpp["submit_ns"]
            _rw_ns, _ = cpp["rewrite_stack_ns"]
            _total_ns = _iv_ns + _dc_ns + _rw_ns
            _pct = cpp_ops / total * 100 if total > 0 else 0
            lines.extend(
                [
                    "C++ fast path:",
                    f"  Ops handled:        {cpp_ops:>8,} / {total:,} ({_pct:.1f}%)",
                    f"  IValue -> py args:  {_iv_ns / cpp_ops / 1000:6.2f} us  |  "
                    f"{_iv_ns / 1_000_000:,.0f} ms",
                    f"  dispatch_cached:    {_dc_ns / cpp_ops / 1000:6.2f} us  |  "
                    f"{_dc_ns / 1_000_000:,.0f} ms",
                    f"    submit:           {_sm_ns / cpp_ops / 1000:6.2f} us  |  "
                    f"{_sm_ns / 1_000_000:,.0f} ms",
                    f"  Rewrite stack:      {_rw_ns / cpp_ops / 1000:6.2f} us  |  "
                    f"{_rw_ns / 1_000_000:,.0f} ms",
                    f"  Total per-op:       {_total_ns / cpp_ops / 1000:6.2f} us  |  "
                    f"{_total_ns / 1_000_000:,.0f} ms",
                    "",
                ]
            )

        # Python fallback path and inter-op gap (from C++ counters)
        if cpp:
            _, _py_count = cpp.get("python_fallback_count", (0, 0))
            _py_ns, _ = cpp.get("python_fallback_ns", (0, 0))
            _gap_ns, _gap_count = cpp.get("inter_op_gap_ns", (0, 0))
            if _py_count > 0:
                lines.extend(
                    [
                        "Python fallback path:",
                        f"  Ops handled:        {_py_count:>8,} / {total:,} "
                        f"({_py_count / total * 100:.1f}%)",
                        f"  Total time:         {_py_ns / _py_count / 1000:6.1f} us  |  "
                        f"{_py_ns / 1_000_000:,.0f} ms",
                        "",
                    ]
                )
            if _gap_count > 0:
                _gap_ms = _gap_ns / 1_000_000
                _gap_pct = (_gap_ms / wall_ms * 100) if wall_ms > 0 else 0
                lines.extend(
                    [
                        "Inter-op gap (C++):",
                        f"  Total:              {_gap_ms:,.0f} ms ({_gap_pct:.0f}% of wall time)",
                        f"  Per-gap avg:        {_gap_ns / _gap_count / 1000:.1f} us  "
                        f"({_gap_count:,} gaps)",
                    ]
                )
                # Gap decomposition
                _gil_wait_ns, _ = cpp.get("gil_wait_ns", (0, 0))
                _ag_ns, _ = cpp.get("autograd_overhead_ns", (0, 0))
                _outside_ag_ns, _outside_ag_count = cpp.get(
                    "outside_autograd_ns", (0, 0)
                )
                _total_ops = cpp_ops + _py_count
                if _total_ops > 0:
                    # autograd_overhead includes fallback_kernel time inside it.
                    # "inside autograd overhead" = autograd_total - fallback_kernel_inner
                    # This is: guard setup + callBoxed dispatch + device guard + guard teardown
                    _fast_ns, _ = cpp.get("dispatch_cached_ns", (0, 0))
                    _iv_ns, _ = cpp.get("ivalue_to_py_ns", (0, 0))
                    _rw_ns, _ = cpp.get("rewrite_stack_ns", (0, 0))
                    _fallback_ns = _fast_ns + _iv_ns + _rw_ns + _py_ns
                    _ag_inner_ns = max(0, _ag_ns - _fallback_ns)
                    # "outside autograd" = Python model code + PyTorch outer
                    # dispatcher (Python binding boxing/unboxing, dispatch key
                    # selection). Measured directly between autograd exit → entry.
                    # "inside autograd overhead" = gap - outside - GIL
                    # This captures: boxed dispatch infra inside autograd
                    # (guard, callBoxed, device guard)
                    lines.extend(
                        [
                            f"  Decomposition:",
                            f"    Outside autograd: {_outside_ag_ns / 1_000_000:,.0f} ms  "
                            f"({_outside_ag_ns / max(1, _outside_ag_count) / 1000:.1f} us/op)"
                            f"  [Python model + outer dispatch]",
                            f"    Inside autograd:  {_ag_inner_ns / 1_000_000:,.0f} ms  "
                            f"({_ag_inner_ns / _total_ops / 1000:.1f} us/op)"
                            f"  [guard + callBoxed + device guard]",
                            f"    GIL acquisition:  {_gil_wait_ns / 1_000_000:,.0f} ms  "
                            f"({_gil_wait_ns / _total_ops / 1000:.1f} us/op)",
                        ]
                    )
                # Large gap decomposition (>1ms)
                _lg_gil_ns, _lg_count = cpp.get("large_gap_gil_ns", (0, 0))
                _lg_other_ns, _ = cpp.get("large_gap_other_ns", (0, 0))
                if _lg_count > 0:
                    _lg_total = _lg_gil_ns + _lg_other_ns
                    lines.extend(
                        [
                            f"  Large gaps (>1ms):  {_lg_count:,} gaps, "
                            f"{_lg_total / 1_000_000:,.0f} ms total",
                            f"    GIL contention:   {_lg_gil_ns / 1_000_000:,.0f} ms  "
                            f"({_lg_gil_ns / _lg_count / 1_000_000:.1f} ms avg)"
                            f"  [event loop holding GIL]",
                            f"    Other (pre-GIL):  {_lg_other_ns / 1_000_000:,.0f} ms  "
                            f"({_lg_other_ns / _lg_count / 1_000_000:.1f} ms avg)"
                            f"  [sync wait, Python, dispatch]",
                        ]
                    )
                # Gap histogram
                _hist = cpp.get("gap_histogram", None)
                if _hist and any(v > 0 for v in _hist):
                    _labels = ["<1us", "1-10us", "10-100us", "100us-1ms", "1-10ms", ">10ms"]
                    _parts = []
                    for lbl, cnt in zip(_labels, _hist):
                        if cnt > 0:
                            _parts.append(f"{lbl}:{cnt:,}")
                    lines.append(f"  Histogram:          {' | '.join(_parts)}")
                lines.append("")
            _, _gil_count = cpp.get("gil_release_count", (0, 0))
            if _gil_count > 0:
                lines.extend(
                    [
                        "GIL releases:",
                        f"  Count:              {_gil_count:,}",
                        "",
                    ]
                )

        lines.extend(
            [
                "Sync points:",
                f"  Total wait:         {self.sync_total.total_ms:,.0f} ms "
                f"({sync_pct:.0f}% of wall time)",
                f"  Avg / Max:          {self.sync_total.avg_ms:.1f} ms / "
                f"{self.sync_total.max_ms:.1f} ms",
                "",
            ]
        )

        if self.sync_flush.count > 0:
            _sync_count = self.sync_flush.count
            _avg_ops = self.sync_mt_ops_total / _sync_count
            _avg_qdepth = self.sync_queue_depth_total / _sync_count
            lines.extend(
                [
                    "Sync phases (per-sync avg / total):",
                    f"  Flush (buffers->queue): {self.sync_flush.avg_ms:.2f} ms  |  "
                    f"{self.sync_flush.total_ms:,.0f} ms",
                    f"  Wait (enqueue->result): {self.sync_wait.avg_ms:.2f} ms  |  "
                    f"{self.sync_wait.total_ms:,.0f} ms",
                    f"  Ops drained at sync:   {_avg_ops:.0f} avg",
                    f"  Queue depth at sync:   {_avg_qdepth:.1f} avg  |  "
                    f"max {self.sync_queue_depth_max}",
                    "",
                ]
            )

        if self.sync_network_rtt.count > 0:
            lines.extend(
                [
                    "Sync wait decomposition (per-sync avg / total):",
                    f"  Network RTT:          {self.sync_network_rtt.avg_ms:.2f} ms  |  "
                    f"{self.sync_network_rtt.total_ms:,.0f} ms",
                    f"  Server backlog:       {self.sync_server_backlog.avg_ms:.2f} ms  |  "
                    f"{self.sync_server_backlog.total_ms:,.0f} ms",
                    f"  Server handle (GPU):  {self.sync_server_handle.avg_ms:.2f} ms  |  "
                    f"{self.sync_server_handle.total_ms:,.0f} ms",
                    "",
                ]
            )

        if self.scalar_speculative_hits > 0 or self.scalar_speculative_misses > 0:
            _total_spec = self.scalar_speculative_hits + self.scalar_speculative_misses
            _hit_pct = (self.scalar_speculative_hits / _total_spec * 100) if _total_spec else 0
            lines.extend(
                [
                    "Scalar speculation:",
                    f"  Hits: {self.scalar_speculative_hits:,}  |  "
                    f"Misses: {self.scalar_speculative_misses:,}  |  "
                    f"Rate: {_hit_pct:.0f}%",
                    "",
                ]
            )

        lines.extend(
            [
                "Batching:",
                (
                    f"  Batches: {self.batch_count.count:,}  |  "
                    f"Avg size: {(self.batch_size_total / self.batch_count.count):.1f}"
                    if self.batch_count.count > 0
                    else "  Batches: 0"
                ),
                f"  Max: {self.batch_size_max}" if self.batch_count.count > 0 else "",
                "",
                f"Wall time: {wall_ms:,.0f} ms",
                "",
            ]
        )
        sys.stderr.write("\n".join(lines))
        sys.stderr.flush()


class ServerProfiler:
    """Per-stream profiler for server-side processing."""

    def __init__(self):
        # Execution counters
        self.raw_execute = Counter()
        self.raw_batched_execute = Counter()
        self.total_ops: int = 0

        # Mixed batch tracking
        self.mixed_batch_calls: int = 0
        self.mixed_batch_ops: int = 0

        # Sync counters
        self.sync_handle = Counter()
        self.scalar_gpu_sync = Counter()
        self.scalar_lookup = Counter()

        # Idle time
        self.idle_time = Counter()
        # Hot idle: idle time only between consecutive execution ops (not model loading gaps)
        self.hot_idle = Counter()

        # Per-sync-cycle breakdown
        self.sync_backlog_ops = Counter()
        self.sync_backlog_time = Counter()
        self.sync_idle_before = Counter()
        self.sync_cycle_count: int = 0

        # Wall time
        self.stream_start_ns: int = 0
        self.stream_end_ns: int = 0

    def print_summary(self) -> None:
        if self.stream_start_ns == 0:
            return

        wall_ms = (self.stream_end_ns - self.stream_start_ns) / 1_000_000
        if wall_ms <= 0:
            return

        fire_and_forget = self.raw_execute.count + self.raw_batched_execute.count
        sync_count = self.sync_handle.count

        exec_ms = self.raw_execute.total_ms + self.raw_batched_execute.total_ms
        exec_pct = (exec_ms / wall_ms * 100) if wall_ms > 0 else 0
        sync_pct = (self.sync_handle.total_ms / wall_ms * 100) if wall_ms > 0 else 0
        idle_pct = (self.idle_time.total_ms / wall_ms * 100) if wall_ms > 0 else 0

        lines = [
            "",
            "=== SkyTorch Server Profile ===",
            f"Requests: {fire_and_forget + sync_count:,} "
            f"({fire_and_forget:,} fire-and-forget, {sync_count:,} sync)",
            f"Ops: {self.total_ops:,}",
            "",
            "Execution:",
            f"  Raw execute:        {self.raw_execute.count:,} calls | "
            f"{self.raw_execute.total_ms:,.0f} ms",
            f"  Batched execute:    {self.raw_batched_execute.count:,} calls | "
            f"{self.raw_batched_execute.total_ms:,.0f} ms"
            + (
                f"  ({self.mixed_batch_calls:,} mixed, "
                f"{self.mixed_batch_ops:,} ops)"
                if self.mixed_batch_calls > 0
                else ""
            ),
            f"  Total execution:    {exec_ms:,.0f} ms ({exec_pct:.0f}%)",
            f"  Sync handle:        {sync_count:,} calls | "
            f"{self.sync_handle.total_ms:,.0f} ms ({sync_pct:.0f}%)",
        ]

        if self.scalar_gpu_sync.count > 0:
            lines.append(
                f"    GPU sync (item):  {self.scalar_gpu_sync.avg_ms:.1f} ms avg | "
                f"{self.scalar_gpu_sync.total_ms:,.0f} ms total"
            )
        if self.scalar_lookup.count > 0:
            lines.append(
                f"    Tensor lookup:    {self.scalar_lookup.avg_us:.1f} us avg | "
                f"{self.scalar_lookup.total_ms:,.0f} ms total"
            )

        if self.sync_cycle_count > 0:
            _sc = self.sync_cycle_count
            lines.extend(
                [
                    "",
                    f"Sync cycles: {_sc}",
                    f"  Backlog ops/cycle:  " f"{(self.sync_backlog_ops.count / _sc):.0f} avg",
                    f"  Backlog exec time:  "
                    f"{self.sync_backlog_time.avg_ms:.2f} ms avg  |  "
                    f"{self.sync_backlog_time.total_ms:,.0f} ms total",
                    f"  Server idle before: "
                    f"{self.sync_idle_before.avg_ms:.2f} ms avg  |  "
                    f"{self.sync_idle_before.total_ms:,.0f} ms total",
                    f"  Scalar handle:      "
                    f"{self.sync_handle.avg_ms:.2f} ms avg  |  "
                    f"{self.sync_handle.total_ms:,.0f} ms total",
                ]
            )

        # C++ per-phase execution breakdown (Step 1)
        try:
            from skytorch.torch.server._C import (
                get_server_profile_counters,
                reset_server_profile_counters,
            )

            cpp_prof = get_server_profile_counters()
            _cpp_op_count = cpp_prof["op_count"]
            if _cpp_op_count > 0:
                _parse_ns = cpp_prof["parse_ns"]
                _callboxed_ns = cpp_prof["callboxed_ns"]
                _output_ns = cpp_prof["output_ns"]
                _total_profiled_ns = _parse_ns + _callboxed_ns + _output_ns
                lines.extend(
                    [
                        "",
                        f"C++ per-phase breakdown ({_cpp_op_count:,} ops):",
                        f"  Parse (header+args): "
                        f"{_parse_ns / _cpp_op_count / 1000:6.1f} us/op  |  "
                        f"{_parse_ns / 1_000_000:,.0f} ms",
                        f"  callBoxed dispatch:  "
                        f"{_callboxed_ns / _cpp_op_count / 1000:6.1f} us/op  |  "
                        f"{_callboxed_ns / 1_000_000:,.0f} ms",
                        f"  Output extraction:   "
                        f"{_output_ns / _cpp_op_count / 1000:6.1f} us/op  |  "
                        f"{_output_ns / 1_000_000:,.0f} ms",
                        f"  Total per-op:        "
                        f"{_total_profiled_ns / _cpp_op_count / 1000:6.1f} us/op  |  "
                        f"{_total_profiled_ns / 1_000_000:,.0f} ms",
                    ]
                )
                # Sub-phase parse breakdown
                _parse_meta_ns = cpp_prof.get("parse_meta_ns", 0)
                _parse_args_ns = cpp_prof.get("parse_args_ns", 0)
                _tensor_arg_count = cpp_prof.get("tensor_arg_count", 0)
                _metadata_create_count = cpp_prof.get("metadata_create_count", 0)
                if _parse_meta_ns > 0 or _parse_args_ns > 0:
                    lines.extend(
                        [
                            f"  Parse sub-phases:",
                            f"    Meta (hdr+hash+meta+OpInfo): "
                            f"{_parse_meta_ns / _cpp_op_count / 1000:6.1f} us/op  |  "
                            f"{_parse_meta_ns / 1_000_000:,.0f} ms",
                            f"    Args (parse+defaults+coerce):"
                            f" {_parse_args_ns / _cpp_op_count / 1000:6.1f} us/op  |  "
                            f"{_parse_args_ns / 1_000_000:,.0f} ms",
                        ]
                    )
                    if _tensor_arg_count > 0:
                        _per_tensor_arg_ns = _parse_args_ns / _tensor_arg_count
                        lines.append(
                            f"    Tensor args: {_tensor_arg_count:,} "
                            f"({_tensor_arg_count / _cpp_op_count:.1f}/op, "
                            f"{_per_tensor_arg_ns / 1000:.2f} us/tensor-arg)"
                        )
                    if _metadata_create_count > 0:
                        lines.append(
                            f"    Metadata creates: {_metadata_create_count:,} "
                            f"(direct TensorImpl)"
                        )
                # Python fallback instrumentation
                _kwargs_fb = cpp_prof.get("kwargs_fallback_count", 0)
                _blocked_fb = cpp_prof.get("blocked_fallback_count", 0)
                _coercion_fb = cpp_prof.get("coercion_fallback_count", 0)
                _fallback_exec_ns = cpp_prof.get("fallback_exec_ns", 0)
                _fb_total = _kwargs_fb + _blocked_fb + _coercion_fb
                if _fb_total > 0:
                    _fb_exec_us = (
                        _fallback_exec_ns / _fb_total / 1000 if _fb_total > 0 else 0
                    )
                    lines.extend(
                        [
                            f"  Python fallback ops: {_fb_total:,}",
                            f"    kwargs:            {_kwargs_fb:,}",
                            f"    callboxed_blocked: {_blocked_fb:,}",
                            f"    coercion failure:  {_coercion_fb:,}",
                            f"    Execution time:    {_fb_exec_us:6.1f} us/op  |  "
                            f"{_fallback_exec_ns / 1_000_000:,.0f} ms",
                        ]
                    )
                # Copy_tensor inline profiling
                _ct_count = cpp_prof.get("copy_tensor_count", 0)
                if _ct_count > 0:
                    _ct_total = cpp_prof.get("copy_tensor_total_ns", 0)
                    _ct_gil = cpp_prof.get("copy_tensor_gil_ns", 0)
                    _ct_proto = cpp_prof.get("copy_tensor_proto_ns", 0)
                    _ct_meta = cpp_prof.get("copy_tensor_meta_ns", 0)
                    _ct_copy = cpp_prof.get("copy_tensor_copy_ns", 0)
                    _ct_meta_count = cpp_prof.get("copy_tensor_meta_count", 0)
                    lines.extend(
                        [
                            f"  Copy_tensor ops:     {_ct_count:,}  "
                            f"({_ct_total / _ct_count / 1000:.1f} us/op  |  "
                            f"{_ct_total / 1_000_000:,.0f} ms)",
                            f"    GIL acquire:       "
                            f"{_ct_gil / _ct_count / 1000:6.1f} us/op  |  "
                            f"{_ct_gil / 1_000_000:,.0f} ms",
                            f"    Protobuf parse:    "
                            f"{_ct_proto / _ct_count / 1000:6.1f} us/op  |  "
                            f"{_ct_proto / 1_000_000:,.0f} ms",
                            f"    Metadata create:   "
                            f"{_ct_meta / _ct_count / 1000:6.1f} us/op  |  "
                            f"{_ct_meta / 1_000_000:,.0f} ms"
                            + (
                                f"  ({_ct_meta_count:,} tensors)"
                                if _ct_meta_count > 0
                                else ""
                            ),
                            f"    copy_() exec:      "
                            f"{_ct_copy / _ct_count / 1000:6.1f} us/op  |  "
                            f"{_ct_copy / 1_000_000:,.0f} ms",
                        ]
                    )
                # Batch-level decomposition
                _bt_count = cpp_prof.get("batch_count", 0)
                _bt_loop = cpp_prof.get("batch_loop_ns", 0)
                _bt_boundary = cpp_prof.get("batch_boundary_ns", 0)
                _bt_op_wall = cpp_prof.get("batch_op_wall_ns", 0)
                _bt_interop = cpp_prof.get("batch_interop_ns", 0)
                if _bt_count > 0 and _bt_boundary > 0:
                    _bt_boundary_only = _bt_boundary - _bt_loop
                    _bt_op_internal = (
                        _parse_ns + _callboxed_ns + _output_ns + _fallback_exec_ns
                    )
                    _bt_op_overhead = _bt_op_wall - _bt_op_internal
                    _batched_exec_ms = self.raw_batched_execute.total_ms
                    _py_vs_cpp = _batched_exec_ms - _bt_boundary / 1_000_000
                    lines.extend(
                        [
                            "",
                            f"  Batch decomposition ({_bt_count:,} C++ batch calls):",
                            f"    C++ func wall:     "
                            f"{_bt_boundary / 1_000_000:,.0f} ms  "
                            f"(loop: {_bt_loop / 1_000_000:,.0f} ms  |  "
                            f"boundary: {_bt_boundary_only / 1_000_000:,.0f} ms)",
                            f"    Op wall (outside):  "
                            f"{_bt_op_wall / 1_000_000:,.0f} ms  "
                            f"(internal: {_bt_op_internal / 1_000_000:,.0f} ms  |  "
                            f"overhead: {_bt_op_overhead / 1_000_000:,.0f} ms)",
                            f"    Inter-op gap:       "
                            f"{_bt_interop / 1_000_000:,.0f} ms  "
                            f"({_bt_interop / _cpp_op_count / 1000:.2f} us/op)",
                            f"    Python↔C++ gap:     "
                            f"{_py_vs_cpp:,.0f} ms  "
                            f"(Python wall - C++ func wall)",
                        ]
                    )
            reset_server_profile_counters()
        except (ImportError, AttributeError):
            pass

        lines.extend(
            [
                "",
                f"  Idle (between req): {self.idle_time.total_ms:,.0f} ms ({idle_pct:.0f}%)",
                f"  Hot idle (exec→exec): {self.hot_idle.total_ms:,.0f} ms "
                f"({self.hot_idle.count:,} gaps, "
                f"{self.hot_idle.avg_us:.0f} us avg, "
                f"{self.hot_idle.max_ms:.1f} ms max)",
                "",
                f"Wall time: {wall_ms:,.0f} ms",
                "",
            ]
        )
        sys.stderr.write("\n".join(lines))
        sys.stderr.flush()
