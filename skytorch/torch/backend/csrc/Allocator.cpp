/**
 * SkyTorch PyTorch Backend - Memory Allocator Implementation
 *
 * This module implements the memory allocator for the SkyTorch backend.
 * Storage IDs are used as data pointers, avoiding actual memory allocation.
 *
 * IMPORTANT: The allocator is GIL-free to avoid PyTorch 2.10's kHasPyObject
 * flag being set during tensor creation. Storage registration is deferred
 * to first tensor use in Python code.
 */

#include <c10/core/Allocator.h>
#include <c10/core/DeviceType.h>
#include <pybind11/pybind11.h>
#include <Python.h>
#include <atomic>

#include "StorageImpl.h"

namespace py = pybind11;

namespace skytorch {

// Forward declaration for get_method (defined in Module.cpp)
py::object get_method(const std::string& name);

// Forward declaration for current device (GIL-free in Module.cpp)
c10::DeviceIndex current_device();

// Atomic counter for storage IDs (no Python dependency)
// Starts at 1 to reserve 0 for empty/invalid storage
static std::atomic<storage_id_t> g_next_storage_id{1};

/**
 * SkyTorch Memory Allocator
 *
 * Uses storage IDs as data pointers instead of allocating actual memory.
 * Storage registration is deferred to first use (lazy allocation).
 * This keeps allocate() GIL-free to prevent kHasPyObject issues.
 */
struct Allocator final : public c10::Allocator {
public:
    Allocator() = default;

    c10::DataPtr allocate(size_t nbytes) override {
        // NO GIL ACQUISITION - pure C++ operation
        // This is critical for PyTorch 2.10 compatibility to avoid
        // the kHasPyObject flag being set during tensor creation

        auto curr_device_idx = current_device();
        auto curr_device = c10::Device(c10::DeviceType::PrivateUse1, curr_device_idx);

        storage_id_t storage_id = 0;
        if (nbytes > 0) {
            // Generate storage ID locally with atomic counter
            // Registration with StorageManager is deferred to first tensor use
            storage_id = g_next_storage_id.fetch_add(1);
        }

        // Store storage ID as data pointer
        void* data = reinterpret_cast<void*>(storage_id);

        return {
            data,
            data,
            &Allocator::ReportAndDelete,
            curr_device
        };
    }

    static void ReportAndDelete(void* ptr) {
        // Called when tensor is garbage collected
        // Safe to call Python here since we're in the destructor path
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
            // free_storage handles both registered and unregistered storage IDs
            // If storage was never used (lazy allocation), this is a no-op
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
        return &Allocator::ReportAndDelete;
    }

    void copy_data(void* dest, const void* src, std::size_t count) const final {
        // For SkyTorch, copy_data should not be called directly on storage IDs
        // This is here for interface compliance
        TORCH_CHECK(false,
            "SkyTorch allocator copy_data should not be called directly. "
            "Use tensor copy operations instead.");
    }
};

// Global allocator instance
static Allocator g_allocator;

// Get the SkyTorch allocator
c10::Allocator* get_allocator() {
    return &g_allocator;
}

// Registration using REGISTER_ALLOCATOR macro
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &g_allocator);

// Registration function (kept for backward compatibility)
void register_allocator() {
    // Allocator is registered via REGISTER_ALLOCATOR macro
    // This function is kept for explicit registration if needed
    c10::SetAllocator(c10::DeviceType::PrivateUse1, &g_allocator);
}

}  // namespace skytorch
