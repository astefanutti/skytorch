/**
 * SkyTorch PyTorch Backend - Tensor Shape Operations
 *
 * This module implements view and shape operations that preserve
 * the custom TensorImpl. These operations create new views of
 * existing tensors without copying data.
 */

#include <ATen/ATen.h>
#include <ATen/InferSize.h>
#include <ATen/TensorUtils.h>

#include "StorageImpl.h"
#include "TensorImpl.h"

namespace skytorch {

// Forward declarations
at::Tensor empty_sky(
    at::IntArrayRef size,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<at::MemoryFormat> memory_format);

/**
 * Create a strided view of a SkyTorch tensor.
 *
 * This is the core view operation - other view operations delegate to this.
 */
at::Tensor as_strided_sky(
    const at::Tensor& self,
    at::IntArrayRef size,
    at::IntArrayRef stride,
    c10::optional<int64_t> storage_offset) {

    TORCH_CHECK(
        self.device().type() == c10::DeviceType::PrivateUse1,
        "as_strided_sky expects a SkyTorch tensor");

    int64_t offset = storage_offset.value_or(self.storage_offset());

    // Create a new SkyTorch tensor with the same storage but different view
    auto result = at::detail::make_tensor<TensorImpl>(
        self.storage(), self.dtype());

    // Set the new sizes and strides for the view
    auto* impl = result.unsafeGetTensorImpl();
    impl->set_sizes_and_strides(size, stride, offset);
    return result;
}

/**
 * Create a view of a SkyTorch tensor with the specified size.
 */
at::Tensor view_sky(const at::Tensor& self, at::IntArrayRef size) {
    TORCH_CHECK(
        self.device().type() == c10::DeviceType::PrivateUse1,
        "view_sky expects a SkyTorch tensor");

    at::DimVector inferred_size = at::infer_size_dv(size, self.numel());
    auto stride = at::detail::computeStride(
        self.sizes(), self.strides(), inferred_size);
    TORCH_CHECK(
        stride.has_value(),
        "view size is not compatible with input tensor's size and stride "
        "(at least one dimension spans across two contiguous subspaces). "
        "Use .reshape(...) instead.");

    return as_strided_sky(self, inferred_size, *stride, self.storage_offset());
}

/**
 * Unsafe view operation that preserves TensorImpl.
 *
 * Similar to view_sky but used internally by PyTorch.
 */
at::Tensor _unsafe_view_sky(const at::Tensor& self, at::IntArrayRef size) {
    TORCH_CHECK(
        self.device().type() == c10::DeviceType::PrivateUse1,
        "_unsafe_view_sky expects a SkyTorch tensor");

    at::DimVector inferred_size = at::infer_size_dv(size, self.numel());
    auto stride = at::detail::computeStride(
        self.sizes(), self.strides(), inferred_size);
    TORCH_CHECK(
        stride.has_value(),
        "_unsafe_view size is not compatible with input tensor's size and "
        "stride (at least one dimension spans across two contiguous subspaces).");

    return as_strided_sky(self, inferred_size, *stride, self.storage_offset());
}

/**
 * Create an alias of a SkyTorch tensor (same storage, same view).
 */
at::Tensor alias_sky(const at::Tensor& self) {
    TORCH_CHECK(
        self.device().type() == c10::DeviceType::PrivateUse1,
        "alias_sky expects a SkyTorch tensor");

    return as_strided_sky(
        self, self.sizes(), self.strides(), self.storage_offset());
}

/**
 * Reshape alias that preserves TensorImpl.
 *
 * Used when reshape can be implemented as a view (contiguous data).
 */
at::Tensor _reshape_alias_sky(
    const at::Tensor& self,
    at::IntArrayRef size,
    at::IntArrayRef stride) {

    TORCH_CHECK(
        self.device().type() == c10::DeviceType::PrivateUse1,
        "_reshape_alias_sky expects a SkyTorch tensor");

    return as_strided_sky(self, size, stride, self.storage_offset());
}

/**
 * Lazy clone that preserves TensorImpl.
 *
 * Creates a new tensor with its own storage and copies data from self.
 */
at::Tensor _lazy_clone_sky(const at::Tensor& self) {
    TORCH_CHECK(
        self.device().type() == c10::DeviceType::PrivateUse1,
        "_lazy_clone_sky expects a SkyTorch tensor");

    auto scalar_type = c10::typeMetaToScalarType(self.dtype());
    auto result = empty_sky(
        self.sizes(), scalar_type, c10::Layout::Strided,
        self.device(), c10::nullopt, c10::nullopt);

    result.copy_(self);

    return result;
}

}  // namespace skytorch
