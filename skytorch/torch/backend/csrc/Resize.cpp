/**
 * SkyTorch PyTorch Backend - Resize Operations
 *
 * Implementation of tensor resize operations.
 */

#include "Resize.h"

#include <ATen/detail/PrivateUse1HooksInterface.h>

namespace skytorch {

const at::Tensor& resize_(
    const at::Tensor& self,
    at::IntArrayRef size,
    c10::optional<at::MemoryFormat> memory_format) {

    int64_t new_numel = c10::multiply_integers(size);

    size_t element_size = self.dtype().itemsize();
    size_t required_bytes = new_numel * element_size;
    size_t current_bytes = self.storage().nbytes();

    auto storage = self.storage();

    // If we need more storage, trigger the resize hook
    if (required_bytes > current_bytes) {
        at::detail::getPrivateUse1Hooks().resizePrivateUse1Bytes(
            storage, required_bytes);
    }

    // Compute contiguous strides for the new size
    std::vector<int64_t> new_stride(size.size());
    if (size.size() > 0) {
        new_stride[size.size() - 1] = 1;
        for (int64_t i = size.size() - 2; i >= 0; i--) {
            new_stride[i] = new_stride[i + 1] * size[i + 1];
        }
    }

    // Update the tensor's size and stride
    const_cast<at::Tensor&>(self).set_(
        storage, self.storage_offset(), size, new_stride);

    return self;
}

}  // namespace skytorch
