"""
Device manager for SkyTorch PyTorch backend.

This module provides the DeviceManager for managing the mapping between
remote Compute resources and local SkyTorch device indices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from skytorch.client.compute import Compute


@dataclass
class RemoteDeviceInfo:
    """Information about a remote device."""

    # TODO: This should ideally be a weak-reference but it can be garbage collected in some cases
    #       before the fist tensor is created if a strong reference isn't kept until then.
    compute: Compute  # Strong reference to Compute
    device_type: str  # Remote device type (e.g., "cuda", "cpu")
    device_index: int  # Remote device index


class DeviceManager:
    """
    Manager for mapping between local SkyTorch device indices and remote devices.

    Provides bidirectional lookup between:
    - Local SkyTorch device index -> (Compute, remote_device_type, remote_device_index)
    - (Compute, remote_device_type, remote_device_index) -> Local SkyTorch device index
    """

    def __init__(self) -> None:
        # local_index -> RemoteDeviceInfo
        self._local_to_remote: dict[int, RemoteDeviceInfo] = {}
        # (compute_id, device_type, device_index) -> local_index
        # We use id(compute) as the key since Compute objects are unique
        self._remote_to_local: dict[tuple[int, str, int], int] = {}
        self._next_index = 0

    def get_sky_device(
        self,
        compute: Compute,
        device_type: str,
        device_index: int = 0,
    ) -> torch.device:
        """
        Get a torch.device for the given Compute and remote device.

        Creates the mapping if it doesn't exist, otherwise returns the existing one.
        Also updates the RuntimeManager's device count when new devices are added.

        Args:
            compute: The Compute resource
            device_type: The remote device type (e.g., "cuda", "cpu")
            device_index: The remote device index

        Returns:
            torch.device object with type "sky" and the mapped local index
        """
        # Ensure backend is initialized before first device use
        # This is lazy to avoid interfering with MPS/CUDA (PyTorch bug #161129)
        from skytorch.torch.backend import _ensure_initialized

        _ensure_initialized()

        # Use id(compute) as part of the key for object identity
        remote_key = (id(compute), device_type, device_index)

        # Check if device mapping already exists
        local_index = self._remote_to_local.get(remote_key)
        if local_index is not None:
            return torch.device("sky", local_index)

        # Create new mapping
        local_index = self._next_index
        self._next_index += 1

        # Store bidirectional mapping
        remote_info = RemoteDeviceInfo(
            compute=compute,
            device_type=device_type,
            device_index=device_index,
        )
        self._local_to_remote[local_index] = remote_info
        self._remote_to_local[remote_key] = local_index

        # Update RuntimeManager device count
        from skytorch.torch.backend._runtime import runtime_manager

        if self._next_index > runtime_manager.get_device_count():
            runtime_manager.set_device_count(self._next_index)

        # Register device mapping in C++ for fused dispatch
        try:
            from skytorch.torch.backend._C import _register_device_mapping

            _register_device_mapping(local_index, device_type, device_index)
        except (ImportError, AttributeError):
            pass

        return torch.device("sky", local_index)

    def get_remote_device_info(self, device_index: int) -> RemoteDeviceInfo:
        """
        Get remote device info for a given SkyTorch device index.

        Args:
            device_index: Local SkyTorch device index

        Returns:
            RemoteDeviceInfo with compute, device_type, and device_index

        Raises:
            KeyError: If no mapping exists for the device index
        """
        return self._local_to_remote[device_index]

    def get_compute(self, device_index: int) -> Optional[Compute]:
        """
        Get the Compute for a given SkyTorch device index.

        Args:
            device_index: Local SkyTorch device index

        Returns:
            The Compute, or None if no mapping exists or Compute was garbage collected
        """
        info = self._local_to_remote.get(device_index)
        return info.compute if info else None

    def has_device(self, device_index: int) -> bool:
        """Check if a mapping exists for the given device index."""
        return device_index in self._local_to_remote

    def device_count(self) -> int:
        """Get the number of registered devices."""
        return len(self._local_to_remote)

    def remove_compute_devices(self, compute: Compute) -> list[int]:
        """
        Remove all device mappings for a Compute.

        Args:
            compute: The Compute resource to remove mappings for

        Returns:
            List of local device indices that were removed
        """
        compute_id = id(compute)
        removed_indices = []

        # Find all mappings for this Compute
        keys_to_remove = [key for key in self._remote_to_local if key[0] == compute_id]

        for key in keys_to_remove:
            local_index = self._remote_to_local.pop(key)
            self._local_to_remote.pop(local_index, None)
            removed_indices.append(local_index)

        return removed_indices

    def reset(self) -> None:
        """
        Reset all device state.

        This clears all device mappings and resets the index counter.
        Useful for testing to ensure fresh state between tests.
        """
        self._local_to_remote.clear()
        self._remote_to_local.clear()
        self._next_index = 0

        # Clear C++ device mappings and shape cache
        try:
            from skytorch.torch.backend._C import _clear_device_mappings, _clear_shape_cache

            _clear_device_mappings()
            _clear_shape_cache()
        except (ImportError, AttributeError):
            pass

        # Reset scalar speculation state
        from skytorch.torch.backend.aten.scalar import _reset_speculation

        _reset_speculation()

        # Clear cached stream managers and submit callback
        from skytorch.torch.backend.aten import dispatch as _dispatch_mod

        _dispatch_mod._cached_stream_managers.clear()
        _dispatch_mod._submit_callback_registered = False
        try:
            from skytorch.torch.backend._C import _clear_submit_callback

            _clear_submit_callback()
        except (ImportError, AttributeError):
            pass


# Global device manager instance
device_manager = DeviceManager()
