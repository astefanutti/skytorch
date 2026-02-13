"""
SkyTorch ATen Scalar Operations.

This module implements scalar operations that require fetching
values from sky tensors to the local host.
"""

import torch

from skytorch.torch.backend._async import run_async
from skytorch.torch.backend._client import (
    ENABLE_STREAMING,
    _get_tensor_metadata_if_new,
    _register_tensor_locally,
    _require_compute,
    get_scalar,
)
from skytorch.torch.client.request import tensor_metadata_to_proto
from skytorch.torch.client.tensor import get_tensor_id


def _local_scalar_dense(self: torch.Tensor):
    """Get the scalar value from a single-element sky tensor.

    This operation fetches the scalar value from the remote server.
    When streaming is enabled, uses the dedicated GetScalar RPC which
    avoids full tensor serialization overhead.

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

    if ENABLE_STREAMING:
        compute = _require_compute(self)
        tensor_id = get_tensor_id(self)

        meta = _get_tensor_metadata_if_new(self)
        metadata_proto = None
        if meta is not None:
            metadata_proto = tensor_metadata_to_proto(meta)

        result = run_async(get_scalar(compute, tensor_id, metadata_proto)).result()

        if meta is not None:
            _register_tensor_locally(self)

        return result
    else:
        from .copy import _copy_from_device

        cpu_tensor = _copy_from_device(self)
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
