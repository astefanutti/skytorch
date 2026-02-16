"""
Server-Side Tensor Manager.

This module manages tensors on the server side.
"""

import logging

import torch

logger = logging.getLogger(__name__)

try:
    from skytorch.torch.server._C import TensorStore as _TensorStore

    _HAS_CPP_STORE = True
except ImportError:
    _HAS_CPP_STORE = False


class TensorManager:
    """Server-side tensor manager."""

    def __init__(self):
        if _HAS_CPP_STORE:
            self._store = _TensorStore()
        else:
            self._store = None
            self._tensors: dict[int, torch.Tensor] = {}

    @property
    def store(self):
        """Return the underlying C++ TensorStore for passing to C++ functions.

        Falls back to None if C++ extension is unavailable.
        """
        return self._store

    def get(self, tensor_id: int) -> torch.Tensor:
        """Get tensor by tensor_id.

        Args:
            tensor_id: Unique identifier for the tensor

        Returns:
            The tensor

        Raises:
            ValueError: If tensor does not exist
        """
        try:
            if self._store is not None:
                return self._store.get(tensor_id)
            return self._tensors[tensor_id]
        except (KeyError, RuntimeError):
            raise ValueError(f"Tensor {tensor_id} does not exist")

    def register(self, tensor_id: int, tensor: torch.Tensor) -> None:
        """Register an existing tensor with a specific ID.

        Args:
            tensor_id: Unique identifier for the tensor
            tensor: The tensor to register
        """
        if self._store is not None:
            self._store.set(tensor_id, tensor)
        else:
            self._tensors[tensor_id] = tensor

    def delete(self, tensor_id: int) -> bool:
        """Delete tensor by tensor_id.

        Args:
            tensor_id: Unique identifier for the tensor to delete

        Returns:
            True if tensor was deleted, False if it didn't exist
        """
        if self._store is not None:
            return self._store.erase(tensor_id)
        if tensor_id in self._tensors:
            del self._tensors[tensor_id]
            return True
        return False

    def clear(self) -> None:
        """Remove all tensors from the manager."""
        if self._store is not None:
            self._store.clear()
        else:
            self._tensors.clear()

    def __contains__(self, tensor_id: int) -> bool:
        """Check if tensor exists."""
        if self._store is not None:
            return tensor_id in self._store
        return tensor_id in self._tensors

    def __len__(self) -> int:
        """Return number of stored tensors."""
        if self._store is not None:
            return len(self._store)
        return len(self._tensors)
