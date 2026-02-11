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
at::Tensor empty(
    at::IntArrayRef size,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<at::MemoryFormat> memory_format);

at::Tensor empty_strided(
    at::IntArrayRef size,
    at::IntArrayRef stride,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory);

at::Tensor as_strided(
    const at::Tensor& self,
    at::IntArrayRef size,
    at::IntArrayRef stride,
    c10::optional<int64_t> storage_offset);

at::Tensor view(const at::Tensor& self, at::IntArrayRef size);
at::Tensor _unsafe_view(const at::Tensor& self, at::IntArrayRef size);
at::Tensor alias(const at::Tensor& self);
at::Tensor _reshape_alias(
    const at::Tensor& self,
    at::IntArrayRef size,
    at::IntArrayRef stride);
at::Tensor _lazy_clone(const at::Tensor& self);

at::Tensor t(const at::Tensor& self);
at::Tensor transpose_int(const at::Tensor& self, int64_t dim0, int64_t dim1);
at::Tensor permute(const at::Tensor& self, at::IntArrayRef dims);
at::Tensor expand(
    const at::Tensor& self,
    at::IntArrayRef sizes,
    bool implicit);
at::Tensor squeeze_dim(const at::Tensor& self, int64_t dim);
at::Tensor squeeze_dims(const at::Tensor& self, at::IntArrayRef dims);
at::Tensor unsqueeze(const at::Tensor& self, int64_t dim);
at::Tensor select_int(const at::Tensor& self, int64_t dim, int64_t index);
at::Tensor slice_tensor(
    const at::Tensor& self,
    int64_t dim,
    c10::optional<int64_t> start,
    c10::optional<int64_t> end,
    int64_t step);

// Register the C++ implementations directly with PyTorch's dispatch system
// These override the Python fallback for these specific operations
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    // Empty tensor creation - these MUST be in C++ to avoid infinite recursion
    // When the Python fallback tries to create output tensors, it would call
    // torch.empty_strided which dispatches back to the fallback, causing recursion.
    m.impl("empty.memory_format", empty);
    m.impl("empty_strided", empty_strided);

    // View operations - implemented in C++ to preserve TensorImpl
    // Without these, view operations would create generic TensorImpl instead
    // of our custom TensorImpl, losing storage ID tracking.
    m.impl("view", view);
    m.impl("as_strided", as_strided);
    m.impl("_unsafe_view", _unsafe_view);
    m.impl("_reshape_alias", _reshape_alias);

    // Set operations for tensor/storage aliasing
    m.impl("set_.source_Tensor", set_source_tensor);
    m.impl("set_.source_Storage", set_source_storage);
    m.impl("set_.source_Storage_storage_offset", set_tensor);

    // Resize with custom hook support
    m.impl("resize_", resize_);

    // Alias and clone operations
    m.impl("alias", alias);
    m.impl("_lazy_clone", _lazy_clone);

    // View/shape operations - purely metadata, no gRPC needed
    m.impl("t", t);
    m.impl("transpose.int", transpose_int);
    m.impl("permute", permute);
    m.impl("expand", expand);
    m.impl("squeeze.dim", squeeze_dim);
    m.impl("squeeze.dims", squeeze_dims);
    m.impl("unsqueeze", unsqueeze);
    m.impl("select.int", select_int);
    m.impl("slice.Tensor", slice_tensor);
}

}  // namespace skytorch
