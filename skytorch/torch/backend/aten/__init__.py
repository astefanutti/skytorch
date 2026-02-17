"""
SkyTorch ATen Operations Module.

This module registers ATen operation implementations for the sky backend.

The dispatch system uses three layers:

1. C++ boxed fallback kernels (RegisterAten.cpp):
   - AutogradPrivateUse1: autograd_fallback_kernel — excludes autograd keys and
     redispatches via op.callBoxed(), staying in C++ dispatch.
   - PrivateUse1: fallback_kernel — handles cache-hit ops entirely in C++ via
     dispatch_cached_aten, falling back to Python _sky_kernel_fallback on miss.

2. C++ boxed op registrations (RegisterAten.cpp):
   Specific PrivateUse1 registrations for ops with CompositeExplicitAutograd (CEA)
   decompositions. These use fallback_kernel as a boxed function, overriding CEA
   to prevent unwanted decomposition (e.g., convolution → convolution_overrideable).

3. Custom Python implementations (this file):
   Ops that need special handling (_copy_from, _local_scalar_dense, equal,
   native_dropout, masked_select).
"""

import torch

from .copy import _copy_from
from .dropout import _native_dropout
from .dynamic import _masked_select, _masked_select_out
from .scalar import _equal, _local_scalar_dense

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

# Import dispatch module to load the C++ extension and register the Python
# fallback callback (_sky_kernel_fallback) for cache misses.
from . import dispatch  # noqa: F401, E402
