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
at::Tensor empty(
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
at::Tensor as_strided(
    const at::Tensor& self,
    at::IntArrayRef size,
    at::IntArrayRef stride,
    c10::optional<int64_t> storage_offset) {

    TORCH_CHECK(
        self.device().type() == c10::DeviceType::PrivateUse1,
        "as_strided expects a SkyTorch tensor");

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
at::Tensor view(const at::Tensor& self, at::IntArrayRef size) {
    TORCH_CHECK(
        self.device().type() == c10::DeviceType::PrivateUse1,
        "view expects a SkyTorch tensor");

    at::DimVector inferred_size = at::infer_size_dv(size, self.numel());
    auto stride = at::detail::computeStride(
        self.sizes(), self.strides(), inferred_size);
    TORCH_CHECK(
        stride.has_value(),
        "view size is not compatible with input tensor's size and stride "
        "(at least one dimension spans across two contiguous subspaces). "
        "Use .reshape(...) instead.");

    return skytorch::as_strided(self, inferred_size, *stride, self.storage_offset());
}

/**
 * Unsafe view operation that preserves TensorImpl.
 *
 * Similar to view but used internally by PyTorch.
 */
at::Tensor _unsafe_view(const at::Tensor& self, at::IntArrayRef size) {
    TORCH_CHECK(
        self.device().type() == c10::DeviceType::PrivateUse1,
        "_unsafe_view expects a SkyTorch tensor");

    at::DimVector inferred_size = at::infer_size_dv(size, self.numel());
    auto stride = at::detail::computeStride(
        self.sizes(), self.strides(), inferred_size);
    TORCH_CHECK(
        stride.has_value(),
        "_unsafe_view size is not compatible with input tensor's size and "
        "stride (at least one dimension spans across two contiguous subspaces).");

    return skytorch::as_strided(self, inferred_size, *stride, self.storage_offset());
}

/**
 * Create an alias of a SkyTorch tensor (same storage, same view).
 */
at::Tensor alias(const at::Tensor& self) {
    TORCH_CHECK(
        self.device().type() == c10::DeviceType::PrivateUse1,
        "alias expects a SkyTorch tensor");

    return skytorch::as_strided(
        self, self.sizes(), self.strides(), self.storage_offset());
}

/**
 * Reshape alias that preserves TensorImpl.
 *
 * Used when reshape can be implemented as a view (contiguous data).
 */
at::Tensor _reshape_alias(
    const at::Tensor& self,
    at::IntArrayRef size,
    at::IntArrayRef stride) {

    TORCH_CHECK(
        self.device().type() == c10::DeviceType::PrivateUse1,
        "_reshape_alias expects a SkyTorch tensor");

    return skytorch::as_strided(self, size, stride, self.storage_offset());
}

/**
 * Transpose a 2D SkyTorch tensor.
 */
at::Tensor t(const at::Tensor& self) {
    TORCH_CHECK(
        self.device().type() == c10::DeviceType::PrivateUse1,
        "t expects a SkyTorch tensor");

    TORCH_CHECK(
        self.dim() <= 2,
        "t expects a tensor with <= 2 dimensions, but self is ", self.dim(), "D");

    if (self.dim() < 2) {
        return self;
    }

    auto sizes = self.sizes().vec();
    auto strides = self.strides().vec();
    std::swap(sizes[0], sizes[1]);
    std::swap(strides[0], strides[1]);
    return skytorch::as_strided(self, sizes, strides, self.storage_offset());
}

/**
 * Transpose two dimensions of a SkyTorch tensor.
 */
at::Tensor transpose_int(const at::Tensor& self, int64_t dim0, int64_t dim1) {
    TORCH_CHECK(
        self.device().type() == c10::DeviceType::PrivateUse1,
        "transpose_int expects a SkyTorch tensor");

    auto ndim = self.dim();
    if (dim0 < 0) dim0 += ndim;
    if (dim1 < 0) dim1 += ndim;
    TORCH_CHECK(
        dim0 >= 0 && dim0 < ndim && dim1 >= 0 && dim1 < ndim,
        "transpose: dim0 and dim1 out of range");

    if (dim0 == dim1) {
        return self;
    }

    auto sizes = self.sizes().vec();
    auto strides = self.strides().vec();
    std::swap(sizes[dim0], sizes[dim1]);
    std::swap(strides[dim0], strides[dim1]);
    return skytorch::as_strided(self, sizes, strides, self.storage_offset());
}

/**
 * Permute dimensions of a SkyTorch tensor.
 */
at::Tensor permute(const at::Tensor& self, at::IntArrayRef dims) {
    TORCH_CHECK(
        self.device().type() == c10::DeviceType::PrivateUse1,
        "permute expects a SkyTorch tensor");

    auto ndim = self.dim();
    TORCH_CHECK(
        static_cast<int64_t>(dims.size()) == ndim,
        "permute: number of dims doesn't match tensor dimensions");

    auto old_sizes = self.sizes();
    auto old_strides = self.strides();
    std::vector<int64_t> new_sizes(ndim);
    std::vector<int64_t> new_strides(ndim);
    std::vector<bool> seen(ndim, false);

    for (int64_t i = 0; i < ndim; i++) {
        auto d = dims[i];
        if (d < 0) d += ndim;
        TORCH_CHECK(
            d >= 0 && d < ndim,
            "permute: dim ", dims[i], " out of range for tensor of dim ", ndim);
        TORCH_CHECK(
            !seen[d],
            "permute: duplicate dims are not allowed");
        seen[d] = true;
        new_sizes[i] = old_sizes[d];
        new_strides[i] = old_strides[d];
    }

    return skytorch::as_strided(self, new_sizes, new_strides, self.storage_offset());
}

/**
 * Expand a SkyTorch tensor to a larger size.
 */
at::Tensor expand(
    const at::Tensor& self,
    at::IntArrayRef sizes,
    bool implicit) {

    TORCH_CHECK(
        self.device().type() == c10::DeviceType::PrivateUse1,
        "expand expects a SkyTorch tensor");

    auto ndim = static_cast<int64_t>(sizes.size());
    auto self_ndim = self.dim();
    TORCH_CHECK(
        ndim >= self_ndim,
        "expand: the number of sizes provided (", ndim,
        ") must be greater or equal to the number of dimensions in the tensor (", self_ndim, ")");

    auto old_sizes = self.sizes();
    auto old_strides = self.strides();
    std::vector<int64_t> new_sizes(ndim);
    std::vector<int64_t> new_strides(ndim);

    // Align dimensions from the right
    auto diff = ndim - self_ndim;
    for (int64_t i = ndim - 1; i >= 0; i--) {
        auto self_i = i - diff;
        if (self_i >= 0) {
            auto self_size = old_sizes[self_i];
            auto target_size = sizes[i];
            if (target_size == -1) {
                target_size = self_size;
            }
            if (self_size == target_size) {
                new_sizes[i] = self_size;
                new_strides[i] = old_strides[self_i];
            } else {
                TORCH_CHECK(
                    self_size == 1,
                    "expand: the expanded size of the tensor (", target_size,
                    ") must match the existing size (", self_size,
                    ") at non-singleton dimension ", i);
                new_sizes[i] = target_size;
                new_strides[i] = 0;
            }
        } else {
            TORCH_CHECK(
                sizes[i] >= 0,
                "expand: the expanded size at non-existing dimension ", i,
                " must be non-negative, got ", sizes[i]);
            new_sizes[i] = sizes[i];
            new_strides[i] = 0;
        }
    }

    return skytorch::as_strided(self, new_sizes, new_strides, self.storage_offset());
}

/**
 * Squeeze a single dimension of a SkyTorch tensor.
 */
at::Tensor squeeze_dim(const at::Tensor& self, int64_t dim) {
    TORCH_CHECK(
        self.device().type() == c10::DeviceType::PrivateUse1,
        "squeeze_dim expects a SkyTorch tensor");

    auto ndim = self.dim();
    if (dim < 0) dim += ndim;
    TORCH_CHECK(
        dim >= 0 && dim < ndim,
        "squeeze: dim ", dim, " out of range for tensor of dim ", ndim);

    if (self.sizes()[dim] != 1) {
        return skytorch::as_strided(self, self.sizes(), self.strides(), self.storage_offset());
    }

    auto sizes = self.sizes().vec();
    auto strides = self.strides().vec();
    sizes.erase(sizes.begin() + dim);
    strides.erase(strides.begin() + dim);
    return skytorch::as_strided(self, sizes, strides, self.storage_offset());
}

/**
 * Squeeze multiple dimensions of a SkyTorch tensor.
 */
at::Tensor squeeze_dims(const at::Tensor& self, at::IntArrayRef dims) {
    TORCH_CHECK(
        self.device().type() == c10::DeviceType::PrivateUse1,
        "squeeze_dims expects a SkyTorch tensor");

    auto ndim = self.dim();
    auto old_sizes = self.sizes();
    auto old_strides = self.strides();

    // Normalize and validate dims, mark which to squeeze
    std::vector<bool> to_squeeze(ndim, false);
    for (auto d : dims) {
        if (d < 0) d += ndim;
        TORCH_CHECK(
            d >= 0 && d < ndim,
            "squeeze: dim out of range for tensor of dim ", ndim);
        if (old_sizes[d] == 1) {
            to_squeeze[d] = true;
        }
    }

    std::vector<int64_t> new_sizes;
    std::vector<int64_t> new_strides;
    for (int64_t i = 0; i < ndim; i++) {
        if (!to_squeeze[i]) {
            new_sizes.push_back(old_sizes[i]);
            new_strides.push_back(old_strides[i]);
        }
    }

    return skytorch::as_strided(self, new_sizes, new_strides, self.storage_offset());
}

/**
 * Unsqueeze a SkyTorch tensor at the given dimension.
 */
at::Tensor unsqueeze(const at::Tensor& self, int64_t dim) {
    TORCH_CHECK(
        self.device().type() == c10::DeviceType::PrivateUse1,
        "unsqueeze expects a SkyTorch tensor");

    auto ndim = self.dim();
    if (dim < 0) dim += ndim + 1;
    TORCH_CHECK(
        dim >= 0 && dim <= ndim,
        "unsqueeze: dim ", dim, " out of range for tensor of dim ", ndim);

    auto sizes = self.sizes().vec();
    auto strides = self.strides().vec();

    // Compute stride for the new dimension
    int64_t new_stride = 1;
    if (dim < ndim) {
        new_stride = sizes[dim] * strides[dim];
    } else if (ndim > 0) {
        new_stride = 1;
    }

    sizes.insert(sizes.begin() + dim, 1);
    strides.insert(strides.begin() + dim, new_stride);
    return skytorch::as_strided(self, sizes, strides, self.storage_offset());
}

/**
 * Select a single element along a dimension.
 */
at::Tensor select_int(const at::Tensor& self, int64_t dim, int64_t index) {
    TORCH_CHECK(
        self.device().type() == c10::DeviceType::PrivateUse1,
        "select_int expects a SkyTorch tensor");

    auto ndim = self.dim();
    TORCH_CHECK(ndim > 0, "select requires a tensor with at least one dimension");

    if (dim < 0) dim += ndim;
    TORCH_CHECK(
        dim >= 0 && dim < ndim,
        "select: dim ", dim, " out of range for tensor of dim ", ndim);

    auto dim_size = self.sizes()[dim];
    if (index < 0) index += dim_size;
    TORCH_CHECK(
        index >= 0 && index < dim_size,
        "select: index ", index, " out of range for dimension ", dim, " of size ", dim_size);

    auto sizes = self.sizes().vec();
    auto strides = self.strides().vec();
    auto offset = self.storage_offset() + index * strides[dim];
    sizes.erase(sizes.begin() + dim);
    strides.erase(strides.begin() + dim);
    return skytorch::as_strided(self, sizes, strides, offset);
}

/**
 * Slice a SkyTorch tensor along a dimension.
 */
at::Tensor slice_tensor(
    const at::Tensor& self,
    int64_t dim,
    c10::optional<int64_t> start,
    c10::optional<int64_t> end,
    int64_t step) {

    TORCH_CHECK(
        self.device().type() == c10::DeviceType::PrivateUse1,
        "slice_tensor expects a SkyTorch tensor");

    auto ndim = self.dim();
    TORCH_CHECK(ndim > 0, "slice requires a tensor with at least one dimension");

    if (dim < 0) dim += ndim;
    TORCH_CHECK(
        dim >= 0 && dim < ndim,
        "slice: dim ", dim, " out of range for tensor of dim ", ndim);

    TORCH_CHECK(step > 0, "slice: step must be positive, got ", step);

    auto dim_size = self.sizes()[dim];

    // Handle start
    int64_t start_val = start.has_value() ? start.value() : 0;
    if (start_val < 0) start_val += dim_size;
    if (start_val < 0) start_val = 0;
    if (start_val > dim_size) start_val = dim_size;

    // Handle end
    int64_t end_val = end.has_value() ? end.value() : dim_size;
    if (end_val < 0) end_val += dim_size;
    if (end_val < 0) end_val = 0;
    if (end_val > dim_size) end_val = dim_size;

    // Compute new size along dim
    int64_t new_dim_size = 0;
    if (start_val < end_val) {
        new_dim_size = (end_val - start_val + step - 1) / step;
    }

    auto sizes = self.sizes().vec();
    auto strides = self.strides().vec();
    auto offset = self.storage_offset() + start_val * strides[dim];
    sizes[dim] = new_dim_size;
    strides[dim] = strides[dim] * step;
    return skytorch::as_strided(self, sizes, strides, offset);
}

/**
 * Lazy clone that preserves TensorImpl.
 *
 * Creates a new tensor with its own storage and copies data from self.
 */
at::Tensor _lazy_clone(const at::Tensor& self) {
    TORCH_CHECK(
        self.device().type() == c10::DeviceType::PrivateUse1,
        "_lazy_clone expects a SkyTorch tensor");

    auto scalar_type = c10::typeMetaToScalarType(self.dtype());
    auto result = skytorch::empty(
        self.sizes(), scalar_type, c10::Layout::Strided,
        self.device(), c10::nullopt, c10::nullopt);

    result.copy_(self);

    return result;
}

}  // namespace skytorch
