/**
 * SkyTorch PyTorch Backend - Custom Storage Implementation
 *
 * This module provides a custom StorageImpl for the SkyTorch backend.
 * Storage IDs are used as data pointers, avoiding actual memory allocation.
 */

#pragma once

#include <c10/core/StorageImpl.h>
#include <c10/core/Allocator.h>

namespace skytorch {

// Storage ID type (cast to/from void* for data pointer)
using storage_id_t = int64_t;

/**
 * SkyTorch Storage Implementation
 *
 * Custom StorageImpl that tracks storage IDs for remote storage.
 * The data pointer is actually a storage ID cast to void*.
 */
struct StorageImpl : public c10::StorageImpl {
public:
    explicit StorageImpl(
        c10::StorageImpl::use_byte_size_t use_byte_size,
        c10::SymInt size_bytes,
        c10::DataPtr data_ptr,
        c10::Allocator* allocator,
        bool resizable);

    /**
     * Get the storage ID from the data pointer.
     */
    storage_id_t get_storage_id() const {
        return reinterpret_cast<storage_id_t>(data_ptr().get());
    }
};

/**
 * Factory function to create SkyTorch storage.
 *
 * If data_ptr is empty, the allocator will be called to create storage.
 */
c10::intrusive_ptr<c10::StorageImpl> make_storage_impl(
    c10::StorageImpl::use_byte_size_t use_byte_size,
    c10::SymInt size_bytes,
    c10::DataPtr data_ptr,
    c10::Allocator* allocator,
    bool resizable);

/**
 * Get the SkyTorch allocator.
 */
c10::Allocator* get_allocator();

}  // namespace skytorch
