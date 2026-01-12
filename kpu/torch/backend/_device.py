"""
KPU device module - Provides torch.kpu.* interface.

This module implements the device management interface for the KPU backend,
following the pattern established by torch.cuda and other backends.
"""

import torch
from typing import Optional, Union

# Module-level state
_current_device_index: int = 0
_initialized: bool = False


def _lazy_init():
    """Initialize the KPU backend on first use."""
    global _initialized
    if not _initialized:
        _initialized = True


def is_available() -> bool:
    """
    Check if KPU backend is available.

    The KPU backend is available when the backend is registered with PyTorch.
    Actual remote Compute resources are checked separately via device_count().

    Returns:
        True if the KPU backend is available.
    """
    return True


def device_count() -> int:
    """
    Return the number of available KPU devices.

    The device count is determined by the number of Compute resources
    available in the current context. Without a Compute context, returns 1
    to allow tensor creation.

    Returns:
        Number of available KPU devices (at least 1 for compatibility).
    """
    try:
        from kpu.client.context import compute_ctx

        compute = compute_ctx.get(None)
        if compute is None:
            # No context, but return 1 to allow basic tensor creation
            return 1

        # Check if it's a Cluster with multiple Computes
        if hasattr(compute, "_computes"):
            return len(compute._computes)

        # Single Compute
        return 1
    except ImportError:
        # kpu.client not available
        return 1


def current_device() -> int:
    """
    Return the index of the currently selected KPU device.

    Returns:
        The current device index.
    """
    _lazy_init()
    return _current_device_index


def set_device(device: Union[int, torch.device, str]) -> None:
    """
    Set the current KPU device.

    Args:
        device: Device index, torch.device, or device string (e.g., "kpu:0").

    Raises:
        RuntimeError: If the device index is invalid.
    """
    global _current_device_index
    _lazy_init()

    if isinstance(device, str):
        device = torch.device(device)
    if isinstance(device, torch.device):
        if device.index is not None:
            device = device.index
        else:
            device = 0

    count = device_count()
    if device >= count:
        raise RuntimeError(
            f"Invalid device index {device}, only {count} KPU device(s) available"
        )

    _current_device_index = device


def synchronize(device: Optional[Union[int, torch.device]] = None) -> None:
    """
    Synchronize with the KPU device.

    For KPU, this ensures all pending gRPC operations have completed.
    In the current implementation, operations are synchronous so this is a no-op.

    Args:
        device: Device to synchronize with (default: current device).
    """
    _lazy_init()
    # gRPC operations are awaited, so no explicit sync needed
    pass


def get_device_name(device: Optional[Union[int, torch.device]] = None) -> str:
    """
    Get the name of the specified KPU device.

    Args:
        device: Device index (default: current device).

    Returns:
        Device name string.
    """
    _lazy_init()
    idx = device if isinstance(device, int) else (device.index if device else current_device())
    return f"KPU Remote Compute:{idx}"


def get_device_capability(
    device: Optional[Union[int, torch.device]] = None
) -> tuple[int, int]:
    """
    Get device compute capability.

    This returns a reasonable default for CUDA compatibility.

    Args:
        device: Device index (default: current device).

    Returns:
        Tuple of (major, minor) version.
    """
    _lazy_init()
    # Return SM 8.0 (Ampere) compatibility level
    return (8, 0)


class _DeviceProperties:
    """Device properties container for KPU devices."""

    def __init__(self, device_index: int = 0):
        self.name = f"KPU Remote Compute:{device_index}"
        self.major = 8
        self.minor = 0
        self.total_memory = 0  # Unknown for remote device
        self.multi_processor_count = 0


def get_device_properties(
    device: Optional[Union[int, torch.device]] = None
) -> _DeviceProperties:
    """
    Get device properties.

    Args:
        device: Device index (default: current device).

    Returns:
        Device properties object.
    """
    _lazy_init()
    idx = device if isinstance(device, int) else (device.index if device else current_device())
    return _DeviceProperties(idx)


# Stream support (minimal implementation for compatibility)
class Stream:
    """
    KPU Stream - minimal implementation for compatibility.

    KPU operations are currently synchronous over gRPC, so streams
    provide minimal functionality for API compatibility.
    """

    def __init__(
        self,
        device: Optional[Union[int, torch.device]] = None,
        priority: int = 0,
    ):
        """
        Create a new stream.

        Args:
            device: Device for this stream (default: current device).
            priority: Stream priority (not used for KPU).
        """
        _lazy_init()
        if device is None:
            self.device_index = current_device()
        elif isinstance(device, int):
            self.device_index = device
        elif isinstance(device, torch.device):
            self.device_index = device.index if device.index is not None else 0
        else:
            self.device_index = 0
        self.priority = priority

    @property
    def device(self) -> torch.device:
        """Get the device for this stream."""
        return torch.device("kpu", self.device_index)

    def synchronize(self) -> None:
        """Synchronize the stream."""
        synchronize(self.device_index)

    def wait_event(self, event: "Event") -> None:
        """Wait for an event."""
        pass  # No-op for KPU

    def wait_stream(self, stream: "Stream") -> None:
        """Wait for another stream."""
        pass  # No-op for KPU

    def record_event(self, event: Optional["Event"] = None) -> "Event":
        """Record an event on this stream."""
        if event is None:
            event = Event()
        event.record(self)
        return event

    def query(self) -> bool:
        """Check if all operations have completed."""
        return True  # Always complete for sync operations

    def __enter__(self) -> "Stream":
        """Enter stream context."""
        return self

    def __exit__(self, *args) -> None:
        """Exit stream context."""
        pass


def stream(stream_obj: Optional[Stream] = None):
    """
    Context manager for stream.

    Args:
        stream_obj: Stream to use (default: creates new stream).

    Returns:
        Stream context manager.
    """
    return stream_obj or Stream()


def current_stream(device: Optional[Union[int, torch.device]] = None) -> Stream:
    """
    Get the current stream for a device.

    Args:
        device: Device index (default: current device).

    Returns:
        Current stream for the device.
    """
    _lazy_init()
    idx = device if isinstance(device, int) else (device.index if device else current_device())
    return Stream(idx)


def default_stream(device: Optional[Union[int, torch.device]] = None) -> Stream:
    """
    Get the default stream for a device.

    Args:
        device: Device index (default: current device).

    Returns:
        Default stream for the device.
    """
    _lazy_init()
    idx = device if isinstance(device, int) else (device.index if device else current_device())
    return Stream(idx)


# Event support (minimal implementation for compatibility)
class Event:
    """
    KPU Event - minimal implementation for compatibility.

    Events provide synchronization primitives. For KPU's synchronous
    operations, they function as simple markers.
    """

    def __init__(self, enable_timing: bool = False, blocking: bool = False):
        """
        Create a new event.

        Args:
            enable_timing: Whether to enable timing (not fully supported).
            blocking: Whether to use blocking synchronization.
        """
        _lazy_init()
        self.enable_timing = enable_timing
        self.blocking = blocking
        self._recorded = False

    def record(self, stream: Optional[Stream] = None) -> None:
        """
        Record the event on a stream.

        Args:
            stream: Stream to record on (default: current stream).
        """
        self._recorded = True

    def wait(self, stream: Optional[Stream] = None) -> None:
        """
        Make a stream wait for this event.

        Args:
            stream: Stream to wait (default: current stream).
        """
        pass  # No-op for KPU

    def synchronize(self) -> None:
        """Block until the event is complete."""
        pass  # No-op for sync operations

    def elapsed_time(self, end_event: "Event") -> float:
        """
        Get elapsed time between events.

        Args:
            end_event: End event for timing.

        Returns:
            Elapsed time in milliseconds.
        """
        return 0.0  # Timing not supported

    def query(self) -> bool:
        """
        Check if the event has completed.

        Returns:
            True if the event is complete.
        """
        return True  # Always complete for sync operations


# Device context manager
class device:
    """
    Context manager for switching KPU devices.

    Example:
        with torch.kpu.device(1):
            # Operations use device 1
            t = torch.randn(3, 3, device="kpu")
    """

    def __init__(self, device_arg: Union[int, torch.device, str]):
        """
        Create a device context.

        Args:
            device_arg: Device to switch to.
        """
        _lazy_init()
        if isinstance(device_arg, str):
            device_arg = torch.device(device_arg)
        if isinstance(device_arg, torch.device):
            self._target = device_arg.index if device_arg.index is not None else 0
        else:
            self._target = device_arg
        self._prev = None

    def __enter__(self) -> None:
        """Enter device context."""
        self._prev = current_device()
        set_device(self._target)

    def __exit__(self, *args) -> None:
        """Exit device context, restoring previous device."""
        if self._prev is not None:
            set_device(self._prev)
