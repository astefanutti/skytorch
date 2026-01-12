"""
KPU ATen Operations Module.

This module registers ATen operation implementations for the KPU backend
using PyTorch's library registration system. It provides:

1. A generic fallback mechanism using meta tensor execution for shape inference
2. Explicit implementations for copy and scalar operations

The fallback uses meta tensors to infer output shapes without moving data,
then creates output tensors on the KPU device. This allows most PyTorch
operations to work with KPU tensors automatically.
"""

import torch

from .copy import _copy_from
from .dispatch import _kpu_kernel_fallback
from .scalar import _equal, _local_scalar_dense

# Register fallback for all unspecified operations
# This catches any operation not explicitly registered and uses
# meta tensor execution to determine output shapes
_kpu_lib = torch.library.Library("_", "IMPL")
_kpu_lib.fallback(_kpu_kernel_fallback, dispatch_key="PrivateUse1")

# Register specific implementations that need custom handling
_kpu_lib_aten = torch.library.Library("aten", "IMPL")

# Copy operations - handle device transfers
_kpu_lib_aten.impl("_copy_from", _copy_from, dispatch_key="PrivateUse1")

# Scalar operations - need to fetch values from device
_kpu_lib_aten.impl("_local_scalar_dense", _local_scalar_dense, dispatch_key="PrivateUse1")

# Equality comparison - returns Python bool
_kpu_lib_aten.impl("equal", _equal, dispatch_key="PrivateUse1")

__all__ = [
    "_kpu_kernel_fallback",
    "_copy_from",
    "_local_scalar_dense",
    "_equal",
]
