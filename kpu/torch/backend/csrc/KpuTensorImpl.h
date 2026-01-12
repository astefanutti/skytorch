/**
 * KPU PyTorch Backend - Custom Tensor Implementation
 *
 * This module provides a custom TensorImpl for the KPU backend.
 * It preserves tensor metadata locally while delegating storage to
 * the remote Compute resources.
 */

#pragma once

#include <c10/core/TensorImpl.h>
#include <c10/core/Storage.h>
#include "KpuStorageImpl.h"

namespace kpu {

/**
 * KPU Tensor Implementation
 *
 * Custom TensorImpl that works with KpuStorageImpl to manage
 * tensors backed by remote storage.
 */
class KpuTensorImpl : public c10::TensorImpl {
public:
    /**
     * Create a KPU tensor from storage.
     */
    explicit KpuTensorImpl(
        const c10::Storage& storage,
        const caffe2::TypeMeta& data_type);

    /**
     * Shallow copy from another TensorImpl.
     */
    void shallow_copy_from(
        const c10::intrusive_ptr<c10::TensorImpl>& impl) final;

    /**
     * Shallow copy and detach.
     */
    c10::intrusive_ptr<c10::TensorImpl> shallow_copy_and_detach(
        const c10::VariableVersion& version_counter,
        bool allow_tensor_metadata_change) const final;

    /**
     * Shallow copy and detach with version counter.
     */
    c10::intrusive_ptr<c10::TensorImpl> shallow_copy_and_detach(
        c10::VariableVersion&& version_counter,
        bool allow_tensor_metadata_change) const final;

    /**
     * Get the storage ID from the underlying storage.
     */
    storage_id_t get_storage_id() const;

    /**
     * Compute a hash of the tensor metadata.
     *
     * This is useful for efficient caching and alias detection.
     * The hash includes shape, strides, dtype, offset, and storage ID.
     */
    uint64_t get_metadata_hash() const;
};

}  // namespace kpu
