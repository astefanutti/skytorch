"""
Server-Side Tensor Manager.

This module manages tensors on the server side.
"""

import logging

import torch

logger = logging.getLogger(__name__)


class TensorManager:
    """Server-side tensor manager."""

    def __init__(self):
        self._tensors: dict[int, torch.Tensor] = {}

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
            return self._tensors[tensor_id]
        except KeyError:
            raise ValueError(f"Tensor {tensor_id} does not exist")

    def register(self, tensor_id: int, tensor: torch.Tensor) -> None:
        """Register an existing tensor with a specific ID.

        Args:
            tensor_id: Unique identifier for the tensor
            tensor: The tensor to register
        """
        if logger.isEnabledFor(logging.DEBUG):
            if tensor_id in self._tensors:
                existing = self._tensors[tensor_id]
                if existing is tensor:
                    # Same tensor object - this is fine (e.g., in-place op)
                    logger.debug(f"Tensor {tensor_id} re-registered (same object)")
                elif existing.data_ptr() == tensor.data_ptr():
                    # Same storage - likely a view, this is fine
                    logger.debug(f"Tensor {tensor_id} re-registered (same storage)")
                else:
                    # Different tensor - this is a collision!
                    logger.debug(
                        f"Tensor ID collision! ID={tensor_id} "
                        f"existing: shape={existing.shape}, data_ptr={existing.data_ptr()}, "
                        f"new: shape={tensor.shape}, data_ptr={tensor.data_ptr()}"
                    )
        self._tensors[tensor_id] = tensor

    def delete(self, tensor_id: int) -> bool:
        """Delete tensor by tensor_id.

        Args:
            tensor_id: Unique identifier for the tensor to delete

        Returns:
            True if tensor was deleted, False if it didn't exist
        """
        if tensor_id in self._tensors:
            del self._tensors[tensor_id]
            return True
        return False

    def clear(self) -> None:
        """Remove all tensors from the manager."""
        self._tensors.clear()

    def __contains__(self, tensor_id: int) -> bool:
        """Check if tensor exists."""
        return tensor_id in self._tensors

    def __len__(self) -> int:
        """Return number of stored tensors."""
        return len(self._tensors)
