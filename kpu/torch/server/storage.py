"""
Server-Side Storage Manager for KPU Tensor Data.

This module manages tensor storage on the server side. It is separate from
the client-side storage manager in kpu/torch/backend/_storage.py.
"""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ServerStorageInfo:
    """Server-side storage information."""

    tensor_id: int
    nbytes: int
    dtype: torch.dtype
    device_type: str
    device_index: int
    tensor: torch.Tensor  # Actual tensor data


class ServerStorageManager:
    """Server-side storage manager for tensor data."""

    def __init__(self):
        self._storages: dict[int, ServerStorageInfo] = {}

    def create(
        self,
        tensor_id: int,
        nbytes: int,
        dtype: torch.dtype,
        device_type: str = "cpu",
        device_index: int = 0,
    ) -> ServerStorageInfo:
        """Create storage with given tensor_id.

        Args:
            tensor_id: Unique identifier for the tensor
            nbytes: Total size in bytes
            dtype: PyTorch dtype for the storage
            device_type: Device type (e.g., "cpu", "cuda")
            device_index: Device index

        Returns:
            ServerStorageInfo for the created storage
        """
        # Allocate actual tensor storage
        elem_size = torch.empty(0, dtype=dtype).element_size()
        numel = nbytes // elem_size
        tensor = torch.empty(numel, dtype=dtype)

        info = ServerStorageInfo(
            tensor_id=tensor_id,
            nbytes=nbytes,
            dtype=dtype,
            device_type=device_type,
            device_index=device_index,
            tensor=tensor,
        )
        self._storages[tensor_id] = info
        return info

    def get(self, tensor_id: int) -> Optional[ServerStorageInfo]:
        """Get storage by tensor_id.

        Args:
            tensor_id: Unique identifier for the tensor

        Returns:
            ServerStorageInfo if found, None otherwise
        """
        return self._storages.get(tensor_id)

    def get_or_create(
        self,
        tensor_id: int,
        nbytes: int,
        dtype: torch.dtype,
        device_type: str = "cpu",
        device_index: int = 0,
    ) -> ServerStorageInfo:
        """Get existing storage or create new one.

        Args:
            tensor_id: Unique identifier for the tensor
            nbytes: Total size in bytes (used if creating)
            dtype: PyTorch dtype for the storage (used if creating)
            device_type: Device type (used if creating)
            device_index: Device index (used if creating)

        Returns:
            ServerStorageInfo for the existing or created storage
        """
        if tensor_id in self._storages:
            return self._storages[tensor_id]
        return self.create(tensor_id, nbytes, dtype, device_type, device_index)

    def register_tensor(self, tensor: torch.Tensor) -> int:
        """Register an existing tensor and return its storage ID.

        Uses the tensor's data_ptr as the storage ID. If already registered,
        returns the existing ID.

        Args:
            tensor: The tensor to register

        Returns:
            The storage ID (data_ptr of the underlying storage)
        """
        storage = tensor.untyped_storage()
        storage_id = storage.data_ptr()

        if storage_id not in self._storages:
            info = ServerStorageInfo(
                tensor_id=storage_id,
                nbytes=storage.nbytes(),
                dtype=tensor.dtype,
                device_type=tensor.device.type,
                device_index=tensor.device.index or 0,
                tensor=tensor.view(-1),  # Store as 1D view
            )
            self._storages[storage_id] = info

        return storage_id

    def delete(self, tensor_id: int) -> None:
        """Delete storage by tensor_id.

        Args:
            tensor_id: Unique identifier for the tensor to delete
        """
        if tensor_id in self._storages:
            del self._storages[tensor_id]

    def __contains__(self, tensor_id: int) -> bool:
        """Check if storage exists."""
        return tensor_id in self._storages

    def __len__(self) -> int:
        """Return number of stored tensors."""
        return len(self._storages)
