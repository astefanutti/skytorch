/**
 * SkyTorch PyTorch Backend - Remote Tensor Creation
 *
 * This module creates sky tensors with server-assigned storage IDs,
 * bypassing the allocator's auto-increment to match the server's
 * storage ID space.
 */

#include <torch/extension.h>
#include <c10/core/Allocator.h>
#include <c10/core/ScalarType.h>
#include <unordered_map>
#include <string>

#include "StorageImpl.h"
#include "TensorImpl.h"

namespace skytorch {

static caffe2::TypeMeta parse_dtype(const std::string& dtype_str) {
    static const std::unordered_map<std::string, c10::ScalarType> dtype_map = {
        {"torch.float16", c10::ScalarType::Half},
        {"torch.float32", c10::ScalarType::Float},
        {"torch.float64", c10::ScalarType::Double},
        {"torch.bfloat16", c10::ScalarType::BFloat16},
        {"torch.int8", c10::ScalarType::Char},
        {"torch.int16", c10::ScalarType::Short},
        {"torch.int32", c10::ScalarType::Int},
        {"torch.int64", c10::ScalarType::Long},
        {"torch.uint8", c10::ScalarType::Byte},
        {"torch.bool", c10::ScalarType::Bool},
        {"torch.complex64", c10::ScalarType::ComplexFloat},
        {"torch.complex128", c10::ScalarType::ComplexDouble},
    };

    auto it = dtype_map.find(dtype_str);
    TORCH_CHECK(it != dtype_map.end(), "Unknown dtype: ", dtype_str);
    return caffe2::TypeMeta::fromScalarType(it->second);
}

/**
 * Create a sky tensor with a server-assigned storage ID.
 *
 * This is used when the server has already created a tensor (e.g., via
 * ExecuteFunction) and we need to create a matching client-side sky tensor
 * that references the same remote storage.
 *
 * @param storage_id Server-assigned storage ID
 * @param shape Tensor shape
 * @param dtype_str Tensor dtype as string (e.g., "torch.float32")
 * @param stride Tensor stride
 * @param storage_offset Storage offset in elements
 * @param nbytes Total storage size in bytes
 * @param device_index Local sky device index
 * @return A sky tensor backed by the server-assigned storage
 */
torch::Tensor create_remote_tensor(
    int64_t storage_id,
    std::vector<int64_t> shape,
    std::string dtype_str,
    std::vector<int64_t> stride,
    int64_t storage_offset,
    int64_t nbytes,
    int64_t device_index) {

    // Advance the client-side storage ID counter past this server-assigned ID
    // to prevent future allocations from colliding
    advance_storage_id_past(storage_id);

    // Build DataPtr with storage_id reinterpreted as void*
    auto* allocator = get_allocator();
    void* data = reinterpret_cast<void*>(storage_id);
    auto device = c10::Device(c10::DeviceType::PrivateUse1, device_index);
    c10::DataPtr data_ptr(data, data, allocator->raw_deleter(), device);

    // Create StorageImpl with the pre-built DataPtr
    auto storage_impl = c10::make_intrusive<StorageImpl>(
        c10::StorageImpl::use_byte_size_t{},
        c10::SymInt(nbytes),
        std::move(data_ptr),
        allocator,
        /*resizable=*/false);

    // Parse dtype string
    auto dtype = parse_dtype(dtype_str);

    // Create TensorImpl with shape/stride/offset
    // set_sizes_and_strides internally calls refresh_numel() and refresh_contiguous()
    c10::Storage storage(std::move(storage_impl));
    auto tensor_impl = c10::make_intrusive<TensorImpl>(storage, dtype);
    tensor_impl->set_sizes_and_strides(shape, stride, storage_offset);

    return torch::Tensor(std::move(tensor_impl));
}

}  // namespace skytorch
