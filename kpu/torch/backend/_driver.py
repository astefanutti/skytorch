"""
KPU Driver - Backend driver for C++ extension callbacks.

This module provides the driver that handles callbacks from the C++ extension.
It manages device state, storage allocation, and stream operations.
"""

from collections import defaultdict
from typing import Any, Callable

from kpu.torch.backend._storage import StorageManager


def register(registry: dict[str, Callable]):
    """Decorator to register a method in the driver registry."""

    def decorator(func: Callable) -> Callable:
        registry[func.__name__] = func
        return func

    return decorator


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


class Driver:
    """
    Driver that handles C++ extension callbacks.

    Uses a registry pattern to dispatch method calls from C++.
    """

    registry: dict[str, Callable] = {}

    def __init__(self):
        self.runtime_manager = RuntimeManager()
        self.storage_manager = StorageManager()

    def get_method(self, name: str) -> Callable:
        """
        Get a method by name for C++ callbacks.

        Args:
            name: Method name

        Returns:
            Callable that can be invoked from C++
        """
        if name in Driver.registry:
            return lambda *args: Driver.registry[name](self, *args)
        raise RuntimeError(f"Unknown driver method: {name}")

    # Storage operations

    @register(registry)
    def create_storage(self, nbytes: int, device_index: int) -> int:
        """Create a new storage allocation."""
        return self.storage_manager.create(nbytes, device_index)

    @register(registry)
    def free_storage(self, storage_id: int) -> None:
        """Free a storage allocation."""
        self.storage_manager.free(storage_id)

    @register(registry)
    def resize_storage(self, storage_id: int, new_nbytes: int) -> None:
        """Resize a storage allocation."""
        self.storage_manager.resize(storage_id, new_nbytes)

    @register(registry)
    def get_storage_nbytes(self, storage_id: int) -> int:
        """Get the size of a storage in bytes."""
        return self.storage_manager.get_nbytes(storage_id)

    # Device operations

    @register(registry)
    def device_count(self) -> int:
        """Get the number of devices."""
        return self.runtime_manager.get_device_count()

    @register(registry)
    def get_device(self) -> int:
        """Get the current device index."""
        return self.runtime_manager.get_device()

    @register(registry)
    def current_device(self) -> int:
        """Get the current device index (alias for get_device)."""
        return self.runtime_manager.get_device()

    @register(registry)
    def set_device(self, device_index: int) -> None:
        """Set the current device."""
        self.runtime_manager.set_device(device_index)

    @register(registry)
    def exchange_device(self, device_index: int) -> int:
        """Exchange the current device."""
        return self.runtime_manager.exchange_device(device_index)

    # Stream operations

    @register(registry)
    def get_stream(self, device_index: int) -> int:
        """Get the current stream for a device."""
        return self.runtime_manager.get_stream(device_index)

    @register(registry)
    def get_new_stream(self, device_index: int, priority: int = 0) -> int:
        """Create a new stream for a device."""
        return self.runtime_manager.get_new_stream(device_index, priority)

    @register(registry)
    def exchange_stream(self, stream_id: int, device_index: int) -> int:
        """Exchange the current stream."""
        return self.runtime_manager.exchange_stream(stream_id, device_index)

    @register(registry)
    def synchronize_stream(self, stream_id: int, device_index: int) -> None:
        """Synchronize a stream."""
        self.runtime_manager.synchronize_stream(stream_id, device_index)

    # Event operations

    @register(registry)
    def create_event(self, device_index: int, flag: int) -> int:
        """Create a new event."""
        return self.runtime_manager.create_event(device_index, flag)

    @register(registry)
    def has_primary_context(self, device_index: int) -> bool:
        """Check if a device has a primary context."""
        return self.runtime_manager.has_primary_context(device_index)

    @register(registry)
    def synchronize(self, device_index: int) -> None:
        """Synchronize the device (no-op for remote execution)."""
        pass


# Global driver instance
driver = Driver()
