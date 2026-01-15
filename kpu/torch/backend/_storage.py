"""
KPU Storage Manager - Tracks remote storage allocations.

This module manages storage allocations for KPU tensors, tracking storage IDs
and their metadata. Storage IDs are used as proxy data pointers in the
allocator, avoiding actual memory allocation on the client side.
"""

from __future__ import annotations

import weakref
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional
from weakref import WeakValueDictionary

import torch

from kpu.torch.client.tensor import get_storage_id, get_tensor_id

if TYPE_CHECKING:
    from kpu.client.compute import Compute


@dataclass
class StorageInfo:
    """Information about a storage allocation."""

    nbytes: int
    device_index: int
    _compute_ref: weakref.ref[Compute] = field(repr=False)

    @property
    def compute(self) -> Optional[Compute]:
        """Get the associated Compute, or None if garbage collected."""
        return self._compute_ref()


class StorageManager:
    """
    Manages storage allocations for KPU tensors.

    Storage IDs are simple integers that serve as references to remote storage.
    The data pointer in PyTorch tensors is set to the storage ID cast to void*,
    allowing efficient tracking without actual memory allocation.
    """

    def __init__(self):
        self._storages: dict[int, StorageInfo] = {}
        self._tensor_to_storage: WeakValueDictionary[
            int, torch.UntypedStorage
        ] = WeakValueDictionary()
        self._storage_to_tensors: dict[int, set[int]] = defaultdict(set)
        self._next_id: int = 1

    def create(
        self,
        nbytes: int,
        device_index: int,
    ) -> int:
        """
        Create a new storage allocation.

        Retrieves the associated Compute from the DeviceManager using device_index.

        Args:
            nbytes: Size of the storage in bytes
            device_index: Device index for the storage

        Returns:
            Storage ID (unique identifier for this storage)
        """
        from kpu.torch.backend._device import device_manager

        # Get Compute from DeviceManager using device_index
        compute = device_manager.get_compute(device_index)
        if compute is None:
            raise RuntimeError(
                f"No Compute registered for device index {device_index}. "
                "Ensure you have called compute.device() to register the device."
            )

        storage_id = self._next_id
        self._next_id += 1
        info = StorageInfo(nbytes, device_index, _compute_ref=weakref.ref(compute))
        self._storages[storage_id] = info
        return storage_id

    def free(self, storage_id: int) -> None:
        """
        Free a storage allocation.

        Args:
            storage_id: ID of the storage to free
        """
        if storage_id in self._storages:
            del self._storages[storage_id]

    def resize(self, storage_id: int, new_nbytes: int) -> None:
        """
        Resize a storage allocation.

        Args:
            storage_id: ID of the storage to resize
            new_nbytes: New size in bytes
        """
        if storage_id in self._storages:
            self._storages[storage_id].nbytes = new_nbytes

    def get(self, storage_id: int) -> Optional[StorageInfo]:
        """
        Get storage info by ID.

        Args:
            storage_id: ID of the storage

        Returns:
            StorageInfo or None if not found
        """
        return self._storages.get(storage_id)

    def register_tensor(self, tensor: torch.Tensor) -> int:
        """
        Register a tensor with its associated storage.

        Args:
            tensor: The KPU tensor to register

        Returns:
            The tensor ID
        """
        tensor_id = get_tensor_id(tensor)
        storage_id = get_storage_id(tensor)
        self._tensor_to_storage[tensor_id] = tensor.untyped_storage()
        self._storage_to_tensors[storage_id].add(tensor_id)
        return tensor_id

    def get_storage_for_tensor(self, tensor_id: int) -> Optional[torch.UntypedStorage]:
        """
        Get the storage associated with a tensor.

        Args:
            tensor_id: The tensor ID

        Returns:
            The UntypedStorage, or None if not found or garbage collected
        """
        return self._tensor_to_storage.get(tensor_id)

    def get_storage_id_for_tensor(self, tensor_id: int) -> Optional[int]:
        """
        Get the storage ID associated with a tensor.

        Args:
            tensor_id: The tensor ID

        Returns:
            The storage ID, or None if not found
        """
        storage = self._tensor_to_storage.get(tensor_id)
        return storage.data_ptr() if storage is not None else None

    def get_tensors_for_storage(self, storage_id: int) -> set[int]:
        """
        Get all tensor IDs associated with a storage.

        Args:
            storage_id: The storage ID

        Returns:
            Set of tensor IDs (may be empty)
        """
        return self._storage_to_tensors.get(storage_id, set())


# Global storage manager singleton
storage_manager = StorageManager()
