/**
 * SkyTorch PyTorch Backend - Set Operations
 *
 * Declarations for tensor set operations that allow tensors
 * to share storage or update their metadata.
 */

#pragma once

#include <ATen/Tensor.h>

namespace skytorch {

/**
 * Set tensor metadata from storage with offset.
 */
at::Tensor& set_tensor(
    at::Tensor& result,
    at::Storage storage,
    int64_t storage_offset,
    at::IntArrayRef size,
    at::IntArrayRef stride);

/**
 * Set tensor to share storage with another tensor.
 */
at::Tensor& set_source_tensor(
    at::Tensor& self,
    const at::Tensor& source);

/**
 * Set tensor to use the specified storage.
 */
at::Tensor& set_source_storage(
    at::Tensor& self,
    at::Storage source);

}  // namespace skytorch
