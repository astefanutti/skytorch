"""
KPU PyTorch Backend - Remote execution backend for Kubernetes Processing Units.

This module provides a PyTorch backend that enables running tensor operations
on remote Kubernetes Compute resources via gRPC. It follows the PyTorch
PrivateUse1/open_registration pattern.

Usage:
    import kpu.torch.backend  # Registers the 'kpu' device

    # Create tensors on KPU device
    t = torch.empty(3, 3, device="kpu")

    # Move tensors to KPU
    cpu_tensor = torch.randn(3, 3)
    kpu_tensor = cpu_tensor.to("kpu")

    # Use with Compute context
    from kpu.client import Compute
    async with Compute(name="my-compute") as compute:
        device = torch.device("kpu")
        model = MyModel().to(device)
        output = model(input.to(device))
"""

from __future__ import annotations

import types
from typing import Optional, Union

import torch

# Backend name constant
BACKEND_NAME = "kpu"

# Track registration state
_is_registered = False
_has_cpp_extension = False

# Import driver FIRST - must be available before C++ extension loads
# The C++ extension will call driver methods via get_method()
from kpu.torch.backend._driver import driver  # noqa: E402


def _autoload():
    """
    Entry point for automatic backend loading via setup.py entry_points.

    This function is called by PyTorch's backend discovery mechanism when
    the 'kpu' backend is registered in setup.py entry_points.
    """
    pass  # Registration happens on import


def _load_cpp_extension() -> bool:
    """Load the C++ extension module if available."""
    global _has_cpp_extension
    try:
        from kpu.torch.backend import _C  # noqa: F401

        _has_cpp_extension = True
        return True
    except ImportError:
        # Extension not built - this is OK for basic functionality
        # Some features may be limited without the C++ extension
        _has_cpp_extension = False
        return False


def _create_device_module() -> types.ModuleType:
    """
    Create the KPU device module for PyTorch backend registration.

    This function creates a module that implements the PyTorch accelerator
    backend interface for KPU devices. It provides device context management,
    RNG state handling, and other core device operations.

    Returns:
        Module implementing the KPU device backend interface
    """
    module = types.ModuleType("_KpuMod")

    def device_count() -> int:
        """Get the number of available KPU devices.

        Returns:
            Number of KPU devices available
        """
        return driver.device_count()

    def is_available() -> bool:
        """Check if KPU device support is available.

        Returns:
            True if KPU devices are available, False otherwise
        """
        return True

    def current_device() -> int:
        """Get the current KPU device index.

        Returns:
            Current device index
        """
        return driver.current_device()

    def set_device(device: Union[int, torch.device]) -> None:
        """Set the current KPU device.

        Args:
            device: Device index or torch.device to set as current
        """
        if isinstance(device, torch.device):
            idx = device.index if device.index is not None else 0
        else:
            idx = int(device)
        driver.set_device(idx)

    def get_rng_state(device: Union[int, torch.device] = 0) -> torch.Tensor:
        """Get the random number generator state for a KPU device.

        Args:
            device: KPU device index or torch.device to get RNG state from

        Returns:
            Tensor containing the RNG state
        """
        if isinstance(device, torch.device):
            idx = device.index if device.index is not None else 0
        else:
            idx = int(device)

        if _has_cpp_extension:
            from kpu.torch.backend import _C
            default_generator = _C._get_default_generator(idx)
            return default_generator.get_state()
        else:
            # Fallback: return empty tensor
            return torch.empty(0, dtype=torch.uint8)

    def set_rng_state(
        new_state: torch.Tensor, device: Union[int, torch.device] = 0
    ) -> None:
        """Set the random number generator state for a KPU device.

        Args:
            new_state: Tensor containing the new RNG state
            device: KPU device index or torch.device to set RNG state for
        """
        if isinstance(device, torch.device):
            idx = device.index if device.index is not None else 0
        else:
            idx = int(device)

        if _has_cpp_extension:
            from kpu.torch.backend import _C
            default_generator = _C._get_default_generator(idx)
            default_generator.set_state(new_state)

    def manual_seed(seed: int) -> None:
        """Set the random seed for the current KPU device.

        Args:
            seed: Random seed value
        """
        seed = int(seed)
        idx = current_device()

        if _has_cpp_extension:
            from kpu.torch.backend import _C
            default_generator = _C._get_default_generator(idx)
            default_generator.manual_seed(seed)

    def manual_seed_all(seed: int) -> None:
        """Set the random seed for all KPU devices.

        Args:
            seed: Random seed value
        """
        seed = int(seed)

        for idx in range(device_count()):
            if _has_cpp_extension:
                from kpu.torch.backend import _C
                default_generator = _C._get_default_generator(idx)
                default_generator.manual_seed(seed)

    def is_initialized() -> bool:
        """Check if the KPU backend is initialized."""
        return module._initialized

    def _lazy_init() -> None:
        """Lazily initialize the KPU backend."""
        if is_initialized():
            return
        if _has_cpp_extension:
            from kpu.torch.backend import _C
            _C._init()
        module._initialized = True

    def _is_in_bad_fork() -> bool:
        """Check if we're in a bad fork state for multiprocessing.

        Returns:
            False - KPU doesn't have fork issues
        """
        return False

    def get_amp_supported_dtype():
        """Get the list of supported dtypes for AMP (Automatic Mixed Precision).

        Returns:
            List of torch.dtype objects supported for AMP operations
        """
        return [torch.float16, torch.bfloat16]

    def synchronize(device: Union[int, torch.device, None] = None) -> None:
        """Synchronize the KPU device.

        Args:
            device: Device to synchronize (default: current device)
        """
        if device is None:
            idx = current_device()
        elif isinstance(device, torch.device):
            idx = device.index if device.index is not None else 0
        else:
            idx = int(device)
        driver.synchronize(idx)

    def get_device_name(device: Optional[Union[int, torch.device]] = None) -> str:
        """Get the name of the specified KPU device.

        Args:
            device: Device index (default: current device)

        Returns:
            Device name string
        """
        _lazy_init()
        if device is None:
            idx = current_device()
        elif isinstance(device, torch.device):
            idx = device.index if device.index is not None else 0
        else:
            idx = int(device)
        return f"KPU Remote Compute:{idx}"

    def get_device_capability(
        device: Optional[Union[int, torch.device]] = None,
    ) -> tuple[int, int]:
        """Get device compute capability.

        Returns a reasonable default for CUDA compatibility.

        Args:
            device: Device index (default: current device)

        Returns:
            Tuple of (major, minor) version
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
        device: Optional[Union[int, torch.device]] = None,
    ) -> _DeviceProperties:
        """Get device properties.

        Args:
            device: Device index (default: current device)

        Returns:
            Device properties object
        """
        _lazy_init()
        if device is None:
            idx = current_device()
        elif isinstance(device, torch.device):
            idx = device.index if device.index is not None else 0
        else:
            idx = int(device)
        return _DeviceProperties(idx)

    # Stream class for compatibility
    class Stream:
        """KPU Stream - minimal implementation for compatibility.

        KPU operations are currently synchronous over gRPC, so streams
        provide minimal functionality for API compatibility.
        """

        def __init__(
            self,
            device: Optional[Union[int, torch.device]] = None,
            priority: int = 0,
        ):
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
            return torch.device("kpu", self.device_index)

        def synchronize(self) -> None:
            synchronize(self.device_index)

        def wait_event(self, event: Event) -> None:
            pass  # No-op for KPU

        def wait_stream(self, stream: Stream) -> None:
            pass  # No-op for KPU

        def record_event(self, event: Optional[Event] = None) -> Event:
            if event is None:
                event = Event()
            event.record(self)
            return event

        def query(self) -> bool:
            return True  # Always complete for sync operations

        def __enter__(self) -> Stream:
            return self

        def __exit__(self, *args) -> None:
            pass

    # Event class for compatibility
    class Event:
        """KPU Event - minimal implementation for compatibility."""

        def __init__(self, enable_timing: bool = False, blocking: bool = False):
            _lazy_init()
            self.enable_timing = enable_timing
            self.blocking = blocking
            self._recorded = False

        def record(self, stream: Optional[Stream] = None) -> None:
            self._recorded = True

        def wait(self, stream: Optional[Stream] = None) -> None:
            pass  # No-op for KPU

        def synchronize(self) -> None:
            pass  # No-op for sync operations

        def elapsed_time(self, end_event: Event) -> float:
            return 0.0  # Timing not supported

        def query(self) -> bool:
            return True  # Always complete for sync operations

    def stream(stream_obj: Optional[Stream] = None) -> Stream:
        """Context manager for stream.

        Args:
            stream_obj: Stream to use (default: creates new stream)

        Returns:
            Stream context manager
        """
        return stream_obj or Stream()

    def current_stream(device: Optional[Union[int, torch.device]] = None) -> Stream:
        """Get the current stream for a device.

        Args:
            device: Device index (default: current device)

        Returns:
            Current stream for the device
        """
        _lazy_init()
        if device is None:
            idx = current_device()
        elif isinstance(device, torch.device):
            idx = device.index if device.index is not None else 0
        else:
            idx = int(device)
        return Stream(idx)

    def default_stream(device: Optional[Union[int, torch.device]] = None) -> Stream:
        """Get the default stream for a device.

        Args:
            device: Device index (default: current device)

        Returns:
            Default stream for the device
        """
        _lazy_init()
        if device is None:
            idx = current_device()
        elif isinstance(device, torch.device):
            idx = device.index if device.index is not None else 0
        else:
            idx = int(device)
        return Stream(idx)

    # Device context manager
    class device:
        """Context manager for switching KPU devices.

        Example:
            with torch.kpu.device(1):
                # Operations use device 1
                t = torch.randn(3, 3, device="kpu")
        """

        def __init__(self, device_arg: Union[int, torch.device, str]):
            _lazy_init()
            if isinstance(device_arg, str):
                device_arg = torch.device(device_arg)
            if isinstance(device_arg, torch.device):
                self._target = device_arg.index if device_arg.index is not None else 0
            else:
                self._target = device_arg
            self._prev: Optional[int] = None

        def __enter__(self) -> None:
            self._prev = current_device()
            set_device(self._target)

        def __exit__(self, *args) -> None:
            if self._prev is not None:
                set_device(self._prev)

    # Attach all methods to the module
    module._initialized = False
    module._lazy_init = _lazy_init
    module.is_initialized = is_initialized

    module.is_available = is_available
    module.device_count = device_count
    module.current_device = current_device
    module.set_device = set_device
    module.synchronize = synchronize

    module.get_rng_state = get_rng_state
    module.set_rng_state = set_rng_state
    module.manual_seed = manual_seed
    module.manual_seed_all = manual_seed_all

    module._is_in_bad_fork = _is_in_bad_fork
    module.get_amp_supported_dtype = get_amp_supported_dtype

    # Device info functions
    module.get_device_name = get_device_name
    module.get_device_capability = get_device_capability
    module.get_device_properties = get_device_properties

    # Stream and Event classes
    module.Stream = Stream
    module.Event = Event
    module.stream = stream
    module.current_stream = current_stream
    module.default_stream = default_stream

    # Device context manager
    module.device = device

    return module


def _register_backend():
    """Register the KPU backend with PyTorch."""
    global _is_registered

    if _is_registered:
        return

    # Rename PrivateUse1 to "kpu"
    torch.utils.rename_privateuse1_backend(BACKEND_NAME)

    # Create and register the device module
    device_module = _create_device_module()
    torch._register_device_module(BACKEND_NAME, device_module)

    # Generate convenience methods on Tensor, Module, and Storage
    # This creates methods like tensor.kpu(), tensor.is_kpu, module.kpu(), etc.
    torch.utils.generate_methods_for_privateuse1_backend(
        for_tensor=True,
        for_module=True,
        for_storage=True,
    )

    _is_registered = True


def is_available() -> bool:
    """
    Check if the KPU backend is available.

    Returns:
        True if the KPU backend is registered and can be used.
    """
    return _is_registered


# Load C++ extension FIRST (registers allocator, guard, hooks, etc.)
# This must happen before backend registration so PyTorch knows about our types
_load_cpp_extension()

# Register backend with PyTorch
_register_backend()

# Import ATen operations module to register Python-based operations
# This must happen AFTER backend registration
from kpu.torch.backend import aten  # noqa: F401, E402

__all__ = [
    "BACKEND_NAME",
    "is_available",
    "driver",
]
