"""ATen operator registrations for SkyTorch backend.

Previously, this file contained ~226 Python per-op registrations that wrapped
_sky_kernel_fallback and registered them as specific PrivateUse1 kernels via
torch.library. These registrations existed to override CompositeExplicitAutograd
(CEA) decompositions for ops like convolution, embedding, etc.

These registrations have been moved to C++ in RegisterAten.cpp, where they use
fallback_kernel as a boxed function. This eliminates Python function call overhead
on cache hits (~98.5% of ops), since fallback_kernel now tries dispatch_cached_aten
directly in C++ before falling back to Python _sky_kernel_fallback on cache misses.

This file is kept for backward compatibility with hack/gen-aten-ops.py.
"""
