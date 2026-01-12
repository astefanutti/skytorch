"""
KPU ATen Scalar Operations.

This module implements scalar operations that require fetching
values from KPU tensors to the local host.
"""

import torch

from .copy import _copy_from_device


def _local_scalar_dense(self: torch.Tensor):
    """Get the scalar value from a single-element KPU tensor.

    This operation copies the single element from the KPU device
    to the CPU and returns it as a Python scalar.

    Args:
        self: A KPU tensor with exactly one element

    Returns:
        Python scalar value (int, float, bool, etc.)

    Raises:
        RuntimeError: If tensor has more than one element
    """
    if self.numel() != 1:
        raise RuntimeError(
            f"a Tensor with {self.numel()} elements cannot be converted to Scalar"
        )

    # Copy scalar to CPU
    cpu_tensor = _copy_from_device(self)

    # Extract Python scalar
    return cpu_tensor.item()


def _equal(self: torch.Tensor, other: torch.Tensor) -> bool:
    """Compare two KPU tensors for equality.

    Performs element-wise comparison on the KPU device, then reduces
    to a single boolean result.

    Args:
        self: First KPU tensor
        other: Second KPU tensor

    Returns:
        True if all elements are equal, False otherwise
    """
    # Check basic compatibility
    if self.shape != other.shape:
        return False
    if self.dtype != other.dtype:
        return False

    # Perform element-wise comparison on KPU device
    eq_tensor = torch.eq(self, other)

    # Reduce to single boolean
    all_equal_tensor = torch.all(eq_tensor)

    # Get scalar result (copies single value to CPU)
    return all_equal_tensor.item()
