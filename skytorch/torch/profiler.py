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
        if self.total_ops == 0:
            return

        wall_ms = (self.last_dispatch_ns - self.first_dispatch_ns) / 1_000_000
        sync_pct = (self.sync_total.total_ms / wall_ms * 100) if wall_ms > 0 else 0

        lines = [
            "",
            "=== SkyTorch Client Profile ===",
            f"Ops: {self.total_ops:,} ({self.cache_hits:,} cache hits, "
            f"{self.cache_misses:,} misses)",
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
            "Sync points:",
            f"  Total wait:         {self.sync_total.total_ms:,.0f} ms "
            f"({sync_pct:.0f}% of wall time)",
            f"  Avg / Max:          {self.sync_total.avg_ms:.1f} ms / "
            f"{self.sync_total.max_ms:.1f} ms",
            "",
        ]

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
            f"{self.raw_batched_execute.total_ms:,.0f} ms",
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
