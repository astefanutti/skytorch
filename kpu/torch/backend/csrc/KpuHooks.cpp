/**
 * KPU PyTorch Backend - Hooks Implementation
 *
 * This module implements backend hooks for the KPU backend.
 * Hooks provide customization points for backend-specific behavior.
 */

#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <ATen/core/Generator.h>
#include <ATen/CPUGeneratorImpl.h>
#include <c10/core/Device.h>
#include <c10/core/StorageImpl.h>
#include <pybind11/pybind11.h>

#include "KpuStorageImpl.h"

namespace py = pybind11;

namespace kpu {

// Forward declarations
py::object get_method(const std::string& name);
c10::DeviceIndex device_count();
c10::DeviceIndex current_device();

// Generator pool for default generators
static std::vector<at::Generator> default_generators;
static std::mutex generator_mutex;

/**
 * KPU Generator Implementation
 *
 * Extends CPUGeneratorImpl to inherit all random number generation
 * functionality while using the KPU device type.
 */
struct KpuGeneratorImpl : public at::CPUGeneratorImpl {
public:
    explicit KpuGeneratorImpl(c10::DeviceIndex device_index = 0) {
        device_ = c10::Device(c10::DeviceType::PrivateUse1, device_index);
        key_set_ = c10::DispatchKeySet(c10::DispatchKey::PrivateUse1);
    }

    ~KpuGeneratorImpl() override = default;
};

at::Generator make_kpu_generator(c10::DeviceIndex device_index) {
    return at::make_generator<KpuGeneratorImpl>(device_index);
}

/**
 * KPU Hooks Implementation
 *
 * Provides hooks for the PrivateUse1 backend with support for:
 * - Storage resizing
 * - Default generators
 * - Context management
 */
struct KpuHooksImpl : public at::PrivateUse1HooksInterface {
    KpuHooksImpl() = default;

    bool hasPrimaryContext(c10::DeviceIndex device_index) const override {
        py::gil_scoped_acquire acquire;
        try {
            return get_method("has_primary_context")(device_index).cast<bool>();
        } catch (...) {
            return true;  // Default to available
        }
    }

    c10::Allocator* getPinnedMemoryAllocator() const override {
        // No pinned memory for KPU
        return nullptr;
    }

    bool isPinnedPtr(const void* data) const override {
        // KPU doesn't use pinned memory
        return false;
    }

    const at::Generator& getDefaultGenerator(
        c10::DeviceIndex device_index) const override {

        // Initialize generator pool on first access
        static bool initialized = []() {
            std::lock_guard<std::mutex> lock(generator_mutex);
            auto num_devices = device_count();
            default_generators.resize(num_devices);
            for (c10::DeviceIndex i = 0; i < num_devices; i++) {
                default_generators[i] = make_kpu_generator(i);
                default_generators[i].seed();
            }
            return true;
        }();
        (void)initialized;

        c10::DeviceIndex idx = device_index == -1 ? current_device() : device_index;
        TORCH_CHECK(
            idx >= 0 && static_cast<size_t>(idx) < default_generators.size(),
            "Invalid device index for generator: ", idx);

        return default_generators[idx];
    }

    void resizePrivateUse1Bytes(
        const c10::Storage& storage,
        size_t new_bytes) const override {

        py::gil_scoped_acquire acquire;

        size_t old_bytes = storage.nbytes();
        if (new_bytes > old_bytes) {
            // Get storage ID from data pointer
            storage_id_t storage_id =
                reinterpret_cast<storage_id_t>(storage.data_ptr().get());

            // Update local storage size
            const_cast<c10::Storage&>(storage)
                .unsafeGetStorageImpl()
                ->set_nbytes(new_bytes);

            // Notify driver of resize
            get_method("resize_storage")(
                storage_id,
                static_cast<int64_t>(new_bytes));
        }
    }
};

// Static registration of hooks
static bool register_hooks_flag = []() {
    at::RegisterPrivateUse1HooksInterface(new KpuHooksImpl());

    // Register custom storage factory
    c10::SetStorageImplCreate(
        c10::DeviceType::PrivateUse1,
        &make_kpu_storage_impl);

    return true;
}();

// Registration function (kept for compatibility)
void register_kpu_hooks() {
    // Hooks are registered via static initialization
    (void)register_hooks_flag;
}

}  // namespace kpu
