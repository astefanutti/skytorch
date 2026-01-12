/**
 * KPU PyTorch Backend - Set Operations
 *
 * Implementation of tensor set operations that allow tensors
 * to share storage or update their metadata.
 */

#include "SetKpu.h"

namespace kpu {

at::Tensor& set_kpu(
    at::Tensor& result,
    at::Storage storage,
    int64_t storage_offset,
    at::IntArrayRef size,
    at::IntArrayRef stride) {

    TORCH_CHECK(
        result.device().type() == c10::DeviceType::PrivateUse1,
        "set_kpu expects a KPU tensor");

    auto* impl = result.unsafeGetTensorImpl();
    impl->set_storage_and_dtype(storage, result.dtype());
    impl->set_sizes_and_strides(size, stride, storage_offset);
    return result;
}

at::Tensor& set_source_tensor_kpu(at::Tensor& self, const at::Tensor& source) {
    TORCH_CHECK(
        self.device().type() == c10::DeviceType::PrivateUse1,
        "set_source_tensor_kpu expects a KPU tensor");
    TORCH_CHECK(
        source.device().type() == c10::DeviceType::PrivateUse1,
        "set_source_tensor_kpu expects a KPU source tensor");

    return set_kpu(
        self, source.storage(), source.storage_offset(),
        source.sizes(), source.strides());
}

at::Tensor& set_source_storage_kpu(at::Tensor& self, at::Storage source) {
    TORCH_CHECK(
        self.device().type() == c10::DeviceType::PrivateUse1,
        "set_source_storage_kpu expects a KPU tensor");

    size_t element_size = self.dtype().itemsize();
    TORCH_CHECK(
        source.nbytes() % element_size == 0,
        "Storage size (", source.nbytes(),
        ") not divisible by element size (", element_size, ")");
    int64_t numel = source.nbytes() / element_size;

    return set_kpu(self, source, 0, {numel}, {1});
}

}  // namespace kpu
