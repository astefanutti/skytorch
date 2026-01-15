"""
KPU Tensor utilities - Tensor ID, metadata, and Compute resolution.

This module provides utilities for working with KPU tensors including
computing tensor IDs, extracting metadata, and resolving associated Compute.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from kpu.torch.client.metadata import TensorMetadata

if TYPE_CHECKING:
    from kpu.client.compute import Compute


def get_storage_id(tensor: torch.Tensor) -> int:
    """Get the storage ID from a KPU tensor.

    The storage ID is stored as the data pointer in KPU tensors.

    Args:
        tensor: A KPU tensor

    Returns:
        The storage ID
    """
    return tensor.data_ptr()


def get_tensor_id(tensor: torch.Tensor) -> int:
    """Get the tensor ID from a KPU tensor using metadata hash.

    Uses the C++ _get_metadata_hash function which computes a hash
    including shape, strides, dtype, offset, and storage ID.

    Args:
        tensor: A KPU tensor

    Returns:
        The tensor ID (metadata hash)

    Raises:
        ValueError: If tensor is not a KPU tensor
    """
    if tensor.device.type != "kpu":
        raise ValueError(
            f"get_tensor_id requires a KPU tensor, got {tensor.device.type}"
        )

    from kpu.torch.backend._C import _get_metadata_hash

    return _get_metadata_hash(tensor)


def get_tensor_metadata(tensor: torch.Tensor) -> TensorMetadata:
    """Create TensorMetadata from a KPU tensor.

    Args:
        tensor: A KPU tensor

    Returns:
        TensorMetadata with all tensor properties
    """
    return TensorMetadata(
        tensor_id=get_tensor_id(tensor),
        shape=tuple(tensor.shape),
        dtype=tensor.dtype,
        nbytes=tensor.untyped_storage().nbytes(),
        device_type=tensor.device.type,
        stride=tuple(tensor.stride()),
        storage_offset=tensor.storage_offset(),
        device_index=tensor.device.index if tensor.device.index is not None else 0,
    )


def resolve_compute(tensor: torch.Tensor) -> Optional[Compute]:
    """
    Resolve the Compute associated with a KPU tensor.

    Resolution order:
    1. Check if the tensor's storage has an associated Compute
    2. Fall back to the current context (compute_ctx)

    Args:
        tensor: A KPU tensor

    Returns:
        The associated Compute, or None if not found
    """
    from kpu.torch.backend._storage import storage_manager

    storage_id = get_storage_id(tensor)
    storage_info = storage_manager.get(storage_id)

    # First, try storage-associated Compute
    if storage_info is not None and storage_info.compute is not None:
        return storage_info.compute

    # Fall back to context
    from kpu.client.context import compute_ctx

    return compute_ctx.get(None)


def require_compute(tensor: torch.Tensor) -> Compute:
    """
    Resolve and require a Compute for a tensor.

    Like resolve_compute but raises if no Compute is available.

    Args:
        tensor: A KPU tensor

    Returns:
        The associated Compute

    Raises:
        RuntimeError: If no Compute context is available
    """
    compute = resolve_compute(tensor)
    if compute is None:
        raise RuntimeError(
            "No Compute context available for KPU tensor operation. "
            "Ensure you are within an 'async with Compute(...):' block."
        )
    return compute
