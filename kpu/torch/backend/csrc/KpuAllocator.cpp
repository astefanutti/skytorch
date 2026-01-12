/**
 * KPU PyTorch Backend - Memory Allocator Implementation
 *
 * This module implements the memory allocator for the KPU backend.
 * Storage IDs are used as data pointers, avoiding actual memory allocation.
 * The driver is called to manage remote storage.
 */

#include <c10/core/Allocator.h>
#include <c10/core/DeviceType.h>
#include <pybind11/pybind11.h>
#include <Python.h>

#include "KpuStorageImpl.h"

namespace py = pybind11;

namespace kpu {

// Forward declaration for get_method (defined in Module.cpp)
py::object get_method(const std::string& name);

// Forward declaration for current device
c10::DeviceIndex current_device();

/**
 * KPU Memory Allocator
 *
 * Uses storage IDs as data pointers instead of allocating actual memory.
 * Calls Python driver for storage creation and deletion.
 */
struct KpuAllocator final : public c10::Allocator {
public:
    KpuAllocator() = default;

    c10::DataPtr allocate(size_t nbytes) override {
        py::gil_scoped_acquire acquire;

        auto curr_device_idx = current_device();
        auto curr_device = c10::Device(c10::DeviceType::PrivateUse1, curr_device_idx);

        storage_id_t storage_id = 0;
        if (nbytes > 0) {
            // Create storage on remote side via driver
            storage_id = get_method("create_storage")(
                static_cast<int64_t>(nbytes),
                static_cast<int>(curr_device_idx)
            ).cast<storage_id_t>();
        }

        // Store storage ID as data pointer
        void* data = reinterpret_cast<void*>(storage_id);

        return {
            data,
            data,
            &KpuAllocator::ReportAndDelete,
            curr_device
        };
    }

    static void ReportAndDelete(void* ptr) {
        if (!ptr || !Py_IsInitialized()) {
            return;
        }

        py::gil_scoped_acquire acquire;

        // Stash any existing Python error
        PyObject* type = nullptr;
        PyObject* value = nullptr;
        PyObject* traceback = nullptr;
        PyErr_Fetch(&type, &value, &traceback);

        try {
            storage_id_t storage_id = reinterpret_cast<storage_id_t>(ptr);
            get_method("free_storage")(storage_id);
        } catch (const std::exception& e) {
            // Log but don't throw during deletion
        }

        // Print any new errors without raising
        if (PyErr_Occurred()) {
            PyErr_Print();
        }

        // Restore original error state
        PyErr_Restore(type, value, traceback);
    }

    c10::DeleterFnPtr raw_deleter() const override {
        return &KpuAllocator::ReportAndDelete;
    }

    void copy_data(void* dest, const void* src, std::size_t count) const final {
        // For KPU, copy_data should not be called directly on storage IDs
        // This is here for interface compliance
        TORCH_CHECK(false,
            "KPU allocator copy_data should not be called directly. "
            "Use tensor copy operations instead.");
    }
};

// Global allocator instance
static KpuAllocator g_kpu_allocator;

// Get the KPU allocator
c10::Allocator* get_kpu_allocator() {
    return &g_kpu_allocator;
}

// Registration using REGISTER_ALLOCATOR macro
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &g_kpu_allocator);

// Registration function (kept for backward compatibility)
void register_kpu_allocator() {
    // Allocator is registered via REGISTER_ALLOCATOR macro
    // This function is kept for explicit registration if needed
    c10::SetAllocator(c10::DeviceType::PrivateUse1, &g_kpu_allocator);
}

}  // namespace kpu
