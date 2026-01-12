/**
 * KPU PyTorch Backend - ATen Operation Registration
 *
 * This module registers all C++ ATen operation implementations with
 * PyTorch's dispatch system using TORCH_LIBRARY_IMPL.
 *
 * These C++ implementations are critical for operations that would
 * cause infinite recursion if handled by the Python fallback (e.g.,
 * empty, empty_strided), and for operations that need to preserve
 * the custom KpuTensorImpl (e.g., view operations).
 */

#include <torch/library.h>

#include "ResizeKpu.h"
#include "SetKpu.h"

namespace kpu {

// Forward declarations for functions defined in other files
at::Tensor empty_kpu(
    at::IntArrayRef size,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<at::MemoryFormat> memory_format);

at::Tensor empty_strided_kpu(
    at::IntArrayRef size,
    at::IntArrayRef stride,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory);

at::Tensor as_strided_kpu(
    const at::Tensor& self,
    at::IntArrayRef size,
    at::IntArrayRef stride,
    c10::optional<int64_t> storage_offset);

at::Tensor view_kpu(const at::Tensor& self, at::IntArrayRef size);
at::Tensor _unsafe_view_kpu(const at::Tensor& self, at::IntArrayRef size);
at::Tensor alias_kpu(const at::Tensor& self);
at::Tensor _reshape_alias_kpu(
    const at::Tensor& self,
    at::IntArrayRef size,
    at::IntArrayRef stride);
at::Tensor _lazy_clone_kpu(const at::Tensor& self);

// Register the C++ implementations directly with PyTorch's dispatch system
// These override the Python fallback for these specific operations
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    // Empty tensor creation - these MUST be in C++ to avoid infinite recursion
    // When the Python fallback tries to create output tensors, it would call
    // torch.empty_strided which dispatches back to the fallback, causing recursion.
    m.impl("empty.memory_format", empty_kpu);
    m.impl("empty_strided", empty_strided_kpu);

    // View operations - implemented in C++ to preserve KpuTensorImpl
    // Without these, view operations would create generic TensorImpl instead
    // of our custom KpuTensorImpl, losing storage ID tracking.
    m.impl("view", view_kpu);
    m.impl("as_strided", as_strided_kpu);
    m.impl("_unsafe_view", _unsafe_view_kpu);
    m.impl("_reshape_alias", _reshape_alias_kpu);

    // Set operations for tensor/storage aliasing
    m.impl("set_.source_Tensor", set_source_tensor_kpu);
    m.impl("set_.source_Storage", set_source_storage_kpu);
    m.impl("set_.source_Storage_storage_offset", set_kpu);

    // Resize with custom hook support
    m.impl("resize_", resize_kpu_);

    // Alias and clone operations
    m.impl("alias", alias_kpu);
    m.impl("_lazy_clone", _lazy_clone_kpu);
}

}  // namespace kpu
