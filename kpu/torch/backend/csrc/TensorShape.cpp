/**
 * KPU PyTorch Backend - Tensor Shape Operations
 *
 * This module implements view and shape operations that preserve
 * the custom KpuTensorImpl. These operations create new views of
 * existing tensors without copying data.
 */

#include <ATen/ATen.h>
#include <ATen/InferSize.h>
#include <ATen/TensorUtils.h>

#include "KpuStorageImpl.h"
#include "KpuTensorImpl.h"

namespace kpu {

// Forward declarations
at::Tensor empty_kpu(
    at::IntArrayRef size,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<at::MemoryFormat> memory_format);

/**
 * Create a strided view of a KPU tensor.
 *
 * This is the core view operation - other view operations delegate to this.
 */
at::Tensor as_strided_kpu(
    const at::Tensor& self,
    at::IntArrayRef size,
    at::IntArrayRef stride,
    c10::optional<int64_t> storage_offset) {

    TORCH_CHECK(
        self.device().type() == c10::DeviceType::PrivateUse1,
        "as_strided_kpu expects a KPU tensor");

    int64_t offset = storage_offset.value_or(self.storage_offset());

    // Create a new KPU tensor with the same storage but different view
    auto result = at::detail::make_tensor<KpuTensorImpl>(
        self.storage(), self.dtype());

    // Set the new sizes and strides for the view
    auto* impl = result.unsafeGetTensorImpl();
    impl->set_sizes_and_strides(size, stride, offset);
    return result;
}

/**
 * Create a view of a KPU tensor with the specified size.
 */
at::Tensor view_kpu(const at::Tensor& self, at::IntArrayRef size) {
    TORCH_CHECK(
        self.device().type() == c10::DeviceType::PrivateUse1,
        "view_kpu expects a KPU tensor");

    at::DimVector inferred_size = at::infer_size_dv(size, self.numel());
    auto stride = at::detail::computeStride(
        self.sizes(), self.strides(), inferred_size);
    TORCH_CHECK(
        stride.has_value(),
        "view size is not compatible with input tensor's size and stride "
        "(at least one dimension spans across two contiguous subspaces). "
        "Use .reshape(...) instead.");

    return as_strided_kpu(self, inferred_size, *stride, self.storage_offset());
}

/**
 * Unsafe view operation that preserves KpuTensorImpl.
 *
 * Similar to view_kpu but used internally by PyTorch.
 */
at::Tensor _unsafe_view_kpu(const at::Tensor& self, at::IntArrayRef size) {
    TORCH_CHECK(
        self.device().type() == c10::DeviceType::PrivateUse1,
        "_unsafe_view_kpu expects a KPU tensor");

    at::DimVector inferred_size = at::infer_size_dv(size, self.numel());
    auto stride = at::detail::computeStride(
        self.sizes(), self.strides(), inferred_size);
    TORCH_CHECK(
        stride.has_value(),
        "_unsafe_view size is not compatible with input tensor's size and "
        "stride (at least one dimension spans across two contiguous subspaces).");

    return as_strided_kpu(self, inferred_size, *stride, self.storage_offset());
}

/**
 * Create an alias of a KPU tensor (same storage, same view).
 */
at::Tensor alias_kpu(const at::Tensor& self) {
    TORCH_CHECK(
        self.device().type() == c10::DeviceType::PrivateUse1,
        "alias_kpu expects a KPU tensor");

    return as_strided_kpu(
        self, self.sizes(), self.strides(), self.storage_offset());
}

/**
 * Reshape alias that preserves KpuTensorImpl.
 *
 * Used when reshape can be implemented as a view (contiguous data).
 */
at::Tensor _reshape_alias_kpu(
    const at::Tensor& self,
    at::IntArrayRef size,
    at::IntArrayRef stride) {

    TORCH_CHECK(
        self.device().type() == c10::DeviceType::PrivateUse1,
        "_reshape_alias_kpu expects a KPU tensor");

    return as_strided_kpu(self, size, stride, self.storage_offset());
}

/**
 * Lazy clone that preserves KpuTensorImpl.
 *
 * Creates a new tensor with its own storage and copies data from self.
 */
at::Tensor _lazy_clone_kpu(const at::Tensor& self) {
    TORCH_CHECK(
        self.device().type() == c10::DeviceType::PrivateUse1,
        "_lazy_clone_kpu expects a KPU tensor");

    auto scalar_type = c10::typeMetaToScalarType(self.dtype());
    auto result = empty_kpu(
        self.sizes(), scalar_type, c10::Layout::Strided,
        self.device(), c10::nullopt, c10::nullopt);

    result.copy_(self);

    return result;
}

}  // namespace kpu
