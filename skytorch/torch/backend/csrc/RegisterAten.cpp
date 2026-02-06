/**
 * SkyTorch PyTorch Backend - ATen Operation Registration
 *
 * This module registers all C++ ATen operation implementations with
 * PyTorch's dispatch system using TORCH_LIBRARY_IMPL.
 *
 * These C++ implementations are critical for operations that would
 * cause infinite recursion if handled by the Python fallback (e.g.,
 * empty, empty_strided), and for operations that need to preserve
 * the custom TensorImpl (e.g., view operations).
 */

#include <torch/library.h>

#include "Resize.h"
#include "Set.h"

namespace skytorch {

// Forward declarations for functions defined in other files
at::Tensor empty_sky(
    at::IntArrayRef size,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<at::MemoryFormat> memory_format);

at::Tensor empty_strided_sky(
    at::IntArrayRef size,
    at::IntArrayRef stride,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory);

at::Tensor as_strided_sky(
    const at::Tensor& self,
    at::IntArrayRef size,
    at::IntArrayRef stride,
    c10::optional<int64_t> storage_offset);

at::Tensor view_sky(const at::Tensor& self, at::IntArrayRef size);
at::Tensor _unsafe_view_sky(const at::Tensor& self, at::IntArrayRef size);
at::Tensor alias_sky(const at::Tensor& self);
at::Tensor _reshape_alias_sky(
    const at::Tensor& self,
    at::IntArrayRef size,
    at::IntArrayRef stride);
at::Tensor _lazy_clone_sky(const at::Tensor& self);

// Register the C++ implementations directly with PyTorch's dispatch system
// These override the Python fallback for these specific operations
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    // Empty tensor creation - these MUST be in C++ to avoid infinite recursion
    // When the Python fallback tries to create output tensors, it would call
    // torch.empty_strided which dispatches back to the fallback, causing recursion.
    m.impl("empty.memory_format", empty_sky);
    m.impl("empty_strided", empty_strided_sky);

    // View operations - implemented in C++ to preserve TensorImpl
    // Without these, view operations would create generic TensorImpl instead
    // of our custom TensorImpl, losing storage ID tracking.
    m.impl("view", view_sky);
    m.impl("as_strided", as_strided_sky);
    m.impl("_unsafe_view", _unsafe_view_sky);
    m.impl("_reshape_alias", _reshape_alias_sky);

    // Set operations for tensor/storage aliasing
    m.impl("set_.source_Tensor", set_source_tensor);
    m.impl("set_.source_Storage", set_source_storage);
    m.impl("set_.source_Storage_storage_offset", set_tensor);

    // Resize with custom hook support
    m.impl("resize_", resize_);

    // Alias and clone operations
    m.impl("alias", alias_sky);
    m.impl("_lazy_clone", _lazy_clone_sky);
}

}  // namespace skytorch
