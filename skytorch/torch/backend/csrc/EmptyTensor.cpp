/**
 * SkyTorch PyTorch Backend - Empty Tensor Creation
 *
 * This module implements tensor factory operations (empty, empty_strided)
 * in C++ to avoid infinite recursion in the Python fallback dispatcher.
 *
 * When the Python fallback tries to create output tensors, it calls
 * torch.empty_strided which would dispatch back to the fallback,
 * causing infinite recursion. By implementing these in C++, they
 * bypass the Python fallback entirely.
 */

#include <ATen/ATen.h>
#include <ATen/native/ResizeCommon.h>
#include <c10/core/Allocator.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/TensorOptions.h>

#include "StorageImpl.h"
#include "TensorImpl.h"

namespace skytorch {

/**
 * Create an empty SkyTorch tensor with the specified size and options.
 */
at::Tensor empty(
    at::IntArrayRef size,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<at::MemoryFormat> memory_format) {

    // Require explicit device to avoid masking bugs
    TORCH_CHECK(
        device.has_value(),
        "empty requires explicit device specification");
    c10::Device target_device = *device;
    TORCH_CHECK(
        target_device.type() == c10::DeviceType::PrivateUse1,
        "empty expects PrivateUse1 device, got: ", target_device.type());

    const auto resolved_dtype = c10::dtype_or_default(dtype);
    TORCH_CHECK(
        c10::layout_or_default(layout) == c10::Layout::Strided,
        "Only strided layout is supported");
    TORCH_CHECK(
        !c10::pinned_memory_or_default(pin_memory),
        "Pin memory is not supported on SkyTorch devices");

    const c10::DeviceGuard device_guard(target_device);

    int64_t nelements = c10::multiply_integers(size);
    auto dtype_meta = c10::scalarTypeToTypeMeta(resolved_dtype);
    int64_t size_bytes = nelements * dtype_meta.itemsize();

    // Create custom storage using our StorageImpl
    c10::intrusive_ptr<c10::StorageImpl> storage_impl = make_storage_impl(
        c10::StorageImpl::use_byte_size_t(),
        c10::SymInt(size_bytes),
        c10::DataPtr(),  // Empty DataPtr - factory calls our allocator
        get_allocator(),
        true);

    // Create tensor using custom TensorImpl
    auto tensor = at::detail::make_tensor<TensorImpl>(
        c10::Storage(storage_impl), dtype_meta);

    if (size.size() != 1 || size[0] != 0) {
        tensor.unsafeGetTensorImpl()->set_sizes_and_strides(
            size, c10::contiguous_strides(size));
    }

    return tensor;
}

/**
 * Create an empty strided SkyTorch tensor.
 */
at::Tensor empty_strided(
    at::IntArrayRef size,
    at::IntArrayRef stride,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory) {

    TORCH_CHECK(
        device.has_value(),
        "empty_strided requires explicit device specification");
    c10::Device target_device = *device;
    TORCH_CHECK(
        target_device.type() == c10::DeviceType::PrivateUse1,
        "empty_strided expects PrivateUse1 device, got: ",
        target_device.type());

    TORCH_CHECK(
        size.size() == stride.size(),
        "empty_strided: size and stride must have the same length");

    const auto resolved_dtype = c10::dtype_or_default(dtype);
    TORCH_CHECK(
        c10::layout_or_default(layout) == c10::Layout::Strided,
        "Only strided layout is supported");
    TORCH_CHECK(
        !c10::pinned_memory_or_default(pin_memory),
        "Pin memory is not supported on SkyTorch devices");

    const c10::DeviceGuard device_guard(target_device);

    // Calculate storage size needed for the strided layout
    // storage_size = 1 + sum((size[i] - 1) * stride[i]) for all dimensions
    int64_t storage_size = 1;
    for (size_t i = 0; i < size.size(); i++) {
        if (size[i] == 0) {
            storage_size = 0;
            break;
        }
        storage_size += (size[i] - 1) * stride[i];
    }

    auto dtype_meta = c10::scalarTypeToTypeMeta(resolved_dtype);
    int64_t size_bytes = storage_size * dtype_meta.itemsize();

    // Create custom storage with the correct size for strided layout
    c10::intrusive_ptr<c10::StorageImpl> storage_impl = make_storage_impl(
        c10::StorageImpl::use_byte_size_t(),
        c10::SymInt(size_bytes),
        c10::DataPtr(),
        get_allocator(),
        true);

    // Create tensor using custom TensorImpl
    auto tensor = at::detail::make_tensor<TensorImpl>(
        c10::Storage(storage_impl), dtype_meta);

    // Set the requested size and stride
    tensor.unsafeGetTensorImpl()->set_sizes_and_strides(size, stride);
    return tensor;
}

}  // namespace skytorch
