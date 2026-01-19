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
        if tensor_id not in self._tensors:
            raise ValueError(f"Tensor {tensor_id} does not exist")
        return self._tensors[tensor_id]

    def register(self, tensor_id: int, tensor: torch.Tensor) -> None:
        """Register an existing tensor with a specific ID.

        Args:
            tensor_id: Unique identifier for the tensor
            tensor: The tensor to register
        """
        if tensor_id in self._tensors:
            # FIXME: decide how to sync client state on re-runs
            logger.warning(f"Tensor {tensor_id} already exists!")
        self._tensors[tensor_id] = tensor

    def delete(self, tensor_id: int) -> None:
        """Delete tensor by tensor_id.

        Args:
            tensor_id: Unique identifier for the tensor to delete

        Raises:
            ValueError: If tensor does not exist
        """
        if tensor_id not in self._tensors:
            raise ValueError(f"Tensor {tensor_id} does not exist")
        del self._tensors[tensor_id]

    def __contains__(self, tensor_id: int) -> bool:
        """Check if tensor exists."""
        return tensor_id in self._tensors

    def __len__(self) -> int:
        """Return number of stored tensors."""
        return len(self._tensors)
