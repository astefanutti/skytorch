"""
KPU Tensor utilities - Tensor ID and metadata extraction.

This module provides utilities for working with KPU tensors including
computing tensor IDs and extracting metadata.
"""

from __future__ import annotations

import torch

from kpu.torch.client.metadata import TensorMetadata


def get_storage_id(tensor: torch.Tensor) -> int:
    """Get the storage ID from a KPU tensor.

    The storage ID is stored as the data pointer in KPU tensors.

    Args:
        tensor: A KPU tensor

    Returns:
        The storage ID
    """
    return tensor.untyped_storage().data_ptr()


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
