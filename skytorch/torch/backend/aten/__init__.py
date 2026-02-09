"""
SkyTorch ATen Operations Module.

This module registers ATen operation implementations for the sky backend
using PyTorch's library registration system. It provides:

1. A generic fallback mechanism using meta tensor execution for shape inference
2. Explicit implementations for copy and scalar operations

The fallback uses meta tensors to infer output shapes without moving data,
then creates output tensors on the SkyTorch device. This allows most PyTorch
operations to work with sky tensors automatically.
"""

from typing import Any

import torch

from .copy import _copy_from
from .dispatch import _sky_kernel_fallback
from .dropout import _native_dropout
from .dynamic import _masked_select, _masked_select_out
from .scalar import _equal, _local_scalar_dense

# Register fallback for all unspecified operations
# This catches any operation not explicitly registered and uses
# meta tensor execution to determine output shapes
_sky_lib = torch.library.Library("_", "IMPL")
_sky_lib.fallback(_sky_kernel_fallback, dispatch_key="PrivateUse1")


def _sky_autograd_fallback(
    op: torch._ops.OpOverload, *args: Any, **kwargs: Any
) -> Any:
    """Autograd fallback for sky backend.

    Redispatches to the PrivateUse1 implementation while properly
    handling autograd context.
    """
    with torch._C._AutoDispatchBelowAutograd():
        return op(*args, **kwargs)


_sky_lib.fallback(_sky_autograd_fallback, dispatch_key="AutogradPrivateUse1")

# Register specific implementations that need custom handling
_sky_lib_aten = torch.library.Library("aten", "IMPL")

# Copy operations - handle device transfers
_sky_lib_aten.impl("_copy_from", _copy_from, dispatch_key="PrivateUse1")

# Scalar operations - need to fetch values from device
_sky_lib_aten.impl("_local_scalar_dense", _local_scalar_dense, dispatch_key="PrivateUse1")

# Equality comparison - returns Python bool
_sky_lib_aten.impl("equal", _equal, dispatch_key="PrivateUse1")

# Dropout - handle deterministic edge cases client-side
_sky_lib_aten.impl("native_dropout", _native_dropout, dispatch_key="PrivateUse1")

# Masked select - has data-dependent output shape
_sky_lib_aten.impl("masked_select", _masked_select, dispatch_key="PrivateUse1")
_sky_lib_aten.impl("masked_select.out", _masked_select_out, dispatch_key="PrivateUse1")

# Import generated operator registrations
# This registers all core ATen operators with the sky fallback wrapper
from . import ops  # noqa: F401, E402
