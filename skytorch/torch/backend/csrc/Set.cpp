/**
 * SkyTorch PyTorch Backend - Set Operations
 *
 * Implementation of tensor set operations that allow tensors
 * to share storage or update their metadata.
 */

#include "Set.h"

namespace skytorch {

at::Tensor& set_tensor(
    at::Tensor& result,
    at::Storage storage,
    int64_t storage_offset,
    at::IntArrayRef size,
    at::IntArrayRef stride) {

    TORCH_CHECK(
        result.device().type() == c10::DeviceType::PrivateUse1,
        "set_tensor expects a SkyTorch tensor");

    auto* impl = result.unsafeGetTensorImpl();
    impl->set_storage_and_dtype(storage, result.dtype());
    impl->set_sizes_and_strides(size, stride, storage_offset);
    return result;
}

at::Tensor& set_source_tensor(at::Tensor& self, const at::Tensor& source) {
    TORCH_CHECK(
        self.device().type() == c10::DeviceType::PrivateUse1,
        "set_source_tensor expects a SkyTorch tensor");
    TORCH_CHECK(
        source.device().type() == c10::DeviceType::PrivateUse1,
        "set_source_tensor expects a SkyTorch source tensor");

    return set_tensor(
        self, source.storage(), source.storage_offset(),
        source.sizes(), source.strides());
}

at::Tensor& set_source_storage(at::Tensor& self, at::Storage source) {
    TORCH_CHECK(
        self.device().type() == c10::DeviceType::PrivateUse1,
        "set_source_storage expects a SkyTorch tensor");

    size_t element_size = self.dtype().itemsize();
    TORCH_CHECK(
        source.nbytes() % element_size == 0,
        "Storage size (", source.nbytes(),
        ") not divisible by element size (", element_size, ")");
    int64_t numel = source.nbytes() / element_size;

    return set_tensor(self, source, 0, {numel}, {1});
}

}  // namespace skytorch
