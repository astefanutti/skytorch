"""
KPU Runtime Manager - Manages device and stream state.

This module provides the RuntimeManager for managing device state,
streams, and events for the KPU backend.
"""

from collections import defaultdict


class RuntimeManager:
    """
    Runtime manager for device and stream state.

    Manages the current device, streams, and events for the KPU backend.
    """

    def __init__(self):
        self._current_device: int = 0
        self._device_count: int = 1
        self._current_streams: dict[int, int] = defaultdict(lambda: 0)
        self._stream_registry: dict[int, list[int]] = defaultdict(lambda: [0])
        self._next_stream_id: int = 1
        self._next_event_id: int = 1

    def get_device_count(self) -> int:
        """Get the number of available devices."""
        return self._device_count

    def set_device_count(self, count: int) -> None:
        """Set the number of available devices."""
        self._device_count = count

    def get_device(self) -> int:
        """Get the current device index."""
        return self._current_device

    def set_device(self, device_index: int) -> None:
        """Set the current device."""
        if 0 <= device_index < self._device_count:
            self._current_device = device_index

    def exchange_device(self, device_index: int) -> int:
        """Exchange the current device, returning the previous one."""
        old_device = self._current_device
        self.set_device(device_index)
        return old_device

    def get_stream(self, device_index: int) -> int:
        """Get the current stream for a device."""
        return self._current_streams[device_index]

    def get_new_stream(self, device_index: int, priority: int = 0) -> int:
        """Create a new stream for a device."""
        stream_id = self._next_stream_id
        self._next_stream_id += 1
        self._stream_registry[device_index].append(stream_id)
        return stream_id

    def exchange_stream(self, stream_id: int, device_index: int) -> int:
        """Exchange the current stream, returning the previous one."""
        old_stream = self._current_streams[device_index]
        self._current_streams[device_index] = stream_id
        return old_stream

    def synchronize_stream(self, stream_id: int, device_index: int) -> None:
        """Synchronize a stream (no-op for now)."""
        pass

    def create_event(self, device_index: int, flag: int) -> int:
        """Create a new event."""
        event_id = self._next_event_id
        self._next_event_id += 1
        return event_id

    def has_primary_context(self, device_index: int) -> bool:
        """Check if a device has a primary context."""
        return 0 <= device_index < self._device_count


# Global runtime manager singleton
runtime_manager = RuntimeManager()
