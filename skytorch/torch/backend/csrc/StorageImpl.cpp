/**
 * SkyTorch PyTorch Backend - Custom Storage Implementation
 */

#include "StorageImpl.h"

namespace skytorch {

StorageImpl::StorageImpl(
    c10::StorageImpl::use_byte_size_t use_byte_size,
    c10::SymInt size_bytes,
    c10::DataPtr data_ptr,
    c10::Allocator* allocator,
    bool resizable)
    : c10::StorageImpl(
          use_byte_size,
          size_bytes,
          std::move(data_ptr),
          allocator,
          resizable) {
}

c10::intrusive_ptr<c10::StorageImpl> make_storage_impl(
    c10::StorageImpl::use_byte_size_t use_byte_size,
    c10::SymInt size_bytes,
    c10::DataPtr data_ptr,
    c10::Allocator* allocator,
    bool resizable) {

    // If no data_ptr provided, call allocator to create storage
    // Always call the allocator, even for 0-byte (empty) tensors, so that
    // the DataPtr device is set correctly (otherwise it defaults to CPU).
    if (data_ptr.get() == nullptr) {
        data_ptr = allocator->allocate(size_bytes.as_int_unchecked());
    }

    return c10::make_intrusive<StorageImpl>(
        use_byte_size,
        size_bytes,
        std::move(data_ptr),
        allocator,
        resizable);
}

}  // namespace skytorch
