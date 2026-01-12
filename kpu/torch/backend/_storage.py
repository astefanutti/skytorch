"""
KPU Storage Manager - Tracks remote storage allocations.

This module manages storage allocations for KPU tensors, tracking storage IDs
and their metadata. Storage IDs are used as proxy data pointers in the
allocator, avoiding actual memory allocation on the client side.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class StorageInfo:
    """Information about a storage allocation."""

    nbytes: int
    device_index: int
    # Future: Add remote storage reference when integrated with gRPC


class StorageManager:
    """
    Manages storage allocations for KPU tensors.

    Storage IDs are simple integers that serve as references to remote storage.
    The data pointer in PyTorch tensors is set to the storage ID cast to void*,
    allowing efficient tracking without actual memory allocation.
    """

    def __init__(self):
        self._storages: dict[int, StorageInfo] = {}
        self._next_id: int = 1

    def create(self, nbytes: int, device_index: int) -> int:
        """
        Create a new storage allocation.

        Args:
            nbytes: Size of the storage in bytes
            device_index: Device index for the storage

        Returns:
            Storage ID (unique identifier for this storage)
        """
        storage_id = self._next_id
        self._next_id += 1
        self._storages[storage_id] = StorageInfo(nbytes, device_index)
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

    def get_nbytes(self, storage_id: int) -> int:
        """
        Get the size of a storage in bytes.

        Args:
            storage_id: ID of the storage

        Returns:
            Size in bytes, or 0 if not found
        """
        info = self._storages.get(storage_id)
        return info.nbytes if info else 0

    def get_device_index(self, storage_id: int) -> int:
        """
        Get the device index for a storage.

        Args:
            storage_id: ID of the storage

        Returns:
            Device index, or 0 if not found
        """
        info = self._storages.get(storage_id)
        return info.device_index if info else 0

    @property
    def count(self) -> int:
        """Number of active storage allocations."""
        return len(self._storages)

    @property
    def total_bytes(self) -> int:
        """Total bytes allocated across all storages."""
        return sum(info.nbytes for info in self._storages.values())
