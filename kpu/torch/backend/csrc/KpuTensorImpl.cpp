/**
 * KPU PyTorch Backend - Custom Tensor Implementation
 */

#include "KpuTensorImpl.h"

namespace kpu {

KpuTensorImpl::KpuTensorImpl(
    const c10::Storage& storage,
    const caffe2::TypeMeta& data_type)
    : c10::TensorImpl(
          c10::Storage(storage),
          c10::DispatchKeySet{
              c10::DispatchKey::PrivateUse1,
              c10::DispatchKey::AutogradPrivateUse1},
          data_type) {
    // KPU tensors may not be non-overlapping and dense
    // since we don't have actual memory to verify
    is_non_overlapping_and_dense_ = false;
}

void KpuTensorImpl::shallow_copy_from(
    const c10::intrusive_ptr<c10::TensorImpl>& impl) {
    // Copy metadata from source tensor implementation
    set_storage_and_dtype(impl->storage(), impl->dtype());
    set_sizes_and_strides(impl->sizes(), impl->strides(), impl->storage_offset());

    refresh_numel();
    refresh_contiguous();
}

c10::intrusive_ptr<c10::TensorImpl> KpuTensorImpl::shallow_copy_and_detach(
    const c10::VariableVersion& version_counter,
    bool allow_tensor_metadata_change) const {

    auto impl = c10::make_intrusive<KpuTensorImpl>(storage(), dtype());

    // Copy metadata from this tensor to the new tensor
    impl->set_storage_and_dtype(storage(), dtype());
    impl->set_sizes_and_strides(sizes(), strides(), storage_offset());

    if (!impl->is_inference()) {
        impl->set_version_counter(version_counter);
    }
    impl->set_allow_tensor_metadata_change(allow_tensor_metadata_change);

    impl->refresh_numel();
    impl->refresh_contiguous();

    return impl;
}

c10::intrusive_ptr<c10::TensorImpl> KpuTensorImpl::shallow_copy_and_detach(
    c10::VariableVersion&& version_counter,
    bool allow_tensor_metadata_change) const {

    auto impl = c10::make_intrusive<KpuTensorImpl>(storage(), dtype());

    // Copy metadata from this tensor to the new tensor
    impl->set_storage_and_dtype(storage(), dtype());
    impl->set_sizes_and_strides(sizes(), strides(), storage_offset());

    if (!impl->is_inference()) {
        impl->set_version_counter(std::move(version_counter));
    }
    impl->set_allow_tensor_metadata_change(allow_tensor_metadata_change);

    impl->refresh_numel();
    impl->refresh_contiguous();

    return impl;
}

storage_id_t KpuTensorImpl::get_storage_id() const {
    auto* kpu_storage = dynamic_cast<KpuStorageImpl*>(
        storage().unsafeGetStorageImpl());
    if (kpu_storage) {
        return kpu_storage->get_storage_id();
    }
    // Fallback: interpret data pointer as storage ID
    return reinterpret_cast<storage_id_t>(storage().data_ptr().get());
}

uint64_t KpuTensorImpl::get_metadata_hash() const {
    // FNV-1a hash
    uint64_t hash = 14695981039346656037ULL;  // FNV offset basis
    const uint64_t prime = 1099511628211ULL;   // FNV prime

    // Hash shape
    for (auto size : sizes()) {
        hash ^= static_cast<uint64_t>(size);
        hash *= prime;
    }

    // Hash strides
    for (auto stride : strides()) {
        hash ^= static_cast<uint64_t>(stride);
        hash *= prime;
    }

    // Hash dtype
    auto dtype_name = dtype().name();
    for (char c : dtype_name) {
        hash ^= static_cast<uint64_t>(c);
        hash *= prime;
    }

    // Hash storage offset
    hash ^= static_cast<uint64_t>(storage_offset());
    hash *= prime;

    // Hash storage ID
    hash ^= static_cast<uint64_t>(get_storage_id());
    hash *= prime;

    return hash;
}

}  // namespace kpu
