"""
KPU Driver - Backend driver for C++ extension callbacks.

This module provides the driver that handles callbacks from the C++ extension.
It delegates to the singleton runtime_manager and storage_manager instances.
"""

from typing import Callable

from kpu.torch.backend._runtime import runtime_manager
from kpu.torch.backend._storage import storage_manager


def register(registry: dict[str, Callable]):
    """Decorator to register a method in the driver registry."""

    def decorator(func: Callable) -> Callable:
        registry[func.__name__] = func
        return func

    return decorator


class Driver:
    """
    Driver that handles C++ extension callbacks.

    Uses a registry pattern to dispatch method calls from C++.
    """

    registry: dict[str, Callable] = {}

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
        return storage_manager.create(nbytes, device_index)

    @register(registry)
    def free_storage(self, storage_id: int) -> None:
        """Free a storage allocation."""
        storage_manager.free(storage_id)

    @register(registry)
    def resize_storage(self, storage_id: int, new_nbytes: int) -> None:
        """Resize a storage allocation."""
        storage_manager.resize(storage_id, new_nbytes)

    # Device operations

    @register(registry)
    def device_count(self) -> int:
        """Get the number of devices."""
        return runtime_manager.get_device_count()

    @register(registry)
    def get_device(self) -> int:
        """Get the current device index."""
        return runtime_manager.get_device()

    @register(registry)
    def current_device(self) -> int:
        """Get the current device index (alias for get_device)."""
        return runtime_manager.get_device()

    @register(registry)
    def set_device(self, device_index: int) -> None:
        """Set the current device."""
        runtime_manager.set_device(device_index)

    @register(registry)
    def exchange_device(self, device_index: int) -> int:
        """Exchange the current device."""
        return runtime_manager.exchange_device(device_index)

    # Stream operations

    @register(registry)
    def get_stream(self, device_index: int) -> int:
        """Get the current stream for a device."""
        return runtime_manager.get_stream(device_index)

    @register(registry)
    def get_new_stream(self, device_index: int, priority: int = 0) -> int:
        """Create a new stream for a device."""
        return runtime_manager.get_new_stream(device_index, priority)

    @register(registry)
    def exchange_stream(self, stream_id: int, device_index: int) -> int:
        """Exchange the current stream."""
        return runtime_manager.exchange_stream(stream_id, device_index)

    @register(registry)
    def synchronize_stream(self, stream_id: int, device_index: int) -> None:
        """Synchronize a stream."""
        runtime_manager.synchronize_stream(stream_id, device_index)

    # Event operations

    @register(registry)
    def create_event(self, device_index: int, flag: int) -> int:
        """Create a new event."""
        return runtime_manager.create_event(device_index, flag)

    @register(registry)
    def has_primary_context(self, device_index: int) -> bool:
        """Check if a device has a primary context."""
        return runtime_manager.has_primary_context(device_index)

    @register(registry)
    def synchronize(self, device_index: int) -> None:
        """Synchronize the device (no-op for remote execution)."""
        pass


# Global driver instance
driver = Driver()
