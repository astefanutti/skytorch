"""
SkyTorch ATen Scalar Operations.

This module implements scalar operations that require fetching
values from sky tensors to the local host.
"""

import torch

from .copy import _copy_from_device


def _local_scalar_dense(self: torch.Tensor):
    """Get the scalar value from a single-element sky tensor.

    This operation copies the single element from the SkyTorch device
    to the cpu and returns it as a python scalar.

    Args:
        self: A sky tensor with exactly one element

    Returns:
        Python scalar value (int, float, bool, etc.)

    Raises:
        RuntimeError: If tensor has more than one element
    """
    if self.numel() != 1:
        raise RuntimeError(
            f"a Tensor with {self.numel()} elements cannot be converted to Scalar"
        )

    # Copy scalar to cpu
    cpu_tensor = _copy_from_device(self)

    # Extract Python scalar
    return cpu_tensor.item()


def _equal(self: torch.Tensor, other: torch.Tensor) -> bool:
    """Compare two sky tensors for equality.

    Performs element-wise comparison on the sky device, then reduces
    to a single boolean result.

    Args:
        self: First sky tensor
        other: Second sky tensor

    Returns:
        True if all elements are equal, False otherwise
    """
    # Check basic compatibility
    if self.shape != other.shape:
        return False
    if self.dtype != other.dtype:
        return False

    # Perform element-wise comparison on sky device
    eq_tensor = torch.eq(self, other)

    # Reduce to single boolean
    all_equal_tensor = torch.all(eq_tensor)

    # Get scalar result (copies single value to cpu)
    return all_equal_tensor.item()
