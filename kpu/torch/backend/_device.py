"""
Device manager for KPU PyTorch backend.

This module provides the DeviceManager for managing the mapping between
remote Compute resources and local KPU device indices.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from kpu.client.compute import Compute


@dataclass
class RemoteDeviceInfo:
    """Information about a remote device."""

    compute_ref: weakref.ref[Compute]  # Weak reference to Compute
    device_type: str  # Remote device type (e.g., "cuda", "cpu")
    device_index: int  # Remote device index

    @property
    def compute(self) -> Optional[Compute]:
        """Get the Compute, or None if it was garbage collected."""
        return self.compute_ref()


class DeviceManager:
    """
    Manager for mapping between local KPU device indices and remote devices.

    Provides bidirectional lookup between:
    - Local KPU device index -> (Compute, remote_device_type, remote_device_index)
    - (Compute, remote_device_type, remote_device_index) -> Local KPU device index
    """

    def __init__(self) -> None:
        # local_index -> RemoteDeviceInfo
        self._local_to_remote: dict[int, RemoteDeviceInfo] = {}
        # (compute_id, device_type, device_index) -> local_index
        # We use id(compute) as the key since Compute objects are unique
        self._remote_to_local: dict[tuple[int, str, int], int] = {}
        self._next_index = 0

    def get_kpu_device(
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
            torch.device object with type "kpu" and the mapped local index
        """
        # Use id(compute) as part of the key for object identity
        remote_key = (id(compute), device_type, device_index)

        # Check if device mapping already exists
        local_index = self._remote_to_local.get(remote_key)
        if local_index is not None:
            return torch.device("kpu", local_index)

        # Create new mapping
        local_index = self._next_index
        self._next_index += 1

        # Store bidirectional mapping
        remote_info = RemoteDeviceInfo(
            compute_ref=weakref.ref(compute),
            device_type=device_type,
            device_index=device_index,
        )
        self._local_to_remote[local_index] = remote_info
        self._remote_to_local[remote_key] = local_index

        # Update RuntimeManager device count
        from kpu.torch.backend._runtime import runtime_manager

        if self._next_index > runtime_manager.get_device_count():
            runtime_manager.set_device_count(self._next_index)

        return torch.device("kpu", local_index)

    def get_remote_device_info(self, device_index: int) -> RemoteDeviceInfo:
        """
        Get remote device info for a given KPU device index.

        Args:
            device_index: Local KPU device index

        Returns:
            RemoteDeviceInfo with compute, device_type, and device_index

        Raises:
            KeyError: If no mapping exists for the device index
        """
        return self._local_to_remote[device_index]

    def get_compute(self, device_index: int) -> Optional[Compute]:
        """
        Get the Compute for a given KPU device index.

        Args:
            device_index: Local KPU device index

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
        keys_to_remove = [
            key for key in self._remote_to_local if key[0] == compute_id
        ]

        for key in keys_to_remove:
            local_index = self._remote_to_local.pop(key)
            self._local_to_remote.pop(local_index, None)
            removed_indices.append(local_index)

        return removed_indices


# Global device manager instance
device_manager = DeviceManager()
