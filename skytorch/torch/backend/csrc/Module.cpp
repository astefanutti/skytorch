/**
 * SkyTorch PyTorch Backend - C++ Extension Module
 *
 * This module provides the C++ extension for the SkyTorch PyTorch backend.
 * It registers the SkyTorch device type with PyTorch's C10 library and
 * provides the factory pattern for Python callbacks.
 */

#include <torch/extension.h>
#include <ATen/Context.h>
#include <c10/core/Device.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <pybind11/pybind11.h>
#include <unordered_map>
#include <mutex>

#include "TensorImpl.h"

namespace py = pybind11;

namespace skytorch {

// Forward declarations for registration functions
void register_guard();
void register_allocator();
void register_hooks();

// Method cache for Python callbacks
// Using PyObject* to avoid destructor issues at Python shutdown
static std::unordered_map<std::string, PyObject*> g_method_cache;
static std::mutex g_method_cache_mutex;
static bool g_driver_initialized = false;

/**
 * Clear the method cache.
 *
 * This must be called before Python shuts down to avoid GIL issues
 * when the static destructor runs.
 */
void clear_method_cache() {
    std::lock_guard<std::mutex> lock(g_method_cache_mutex);
    // Decrement reference counts while Python is still alive
    for (auto& pair : g_method_cache) {
        Py_XDECREF(pair.second);
    }
    g_method_cache.clear();
}

/**
 * Get a method from the Python driver.
 *
 * This function provides the factory pattern for C++ to call Python methods.
 * Methods are cached for efficiency.
 *
 * Uses PyObject* internally to avoid destructor issues at Python shutdown.
 */
py::object get_method(const std::string& name) {
    // Check cache first (with lock)
    {
        std::lock_guard<std::mutex> lock(g_method_cache_mutex);
        auto it = g_method_cache.find(name);
        if (it != g_method_cache.end()) {
            // Return borrowed reference wrapped in py::object
            return py::reinterpret_borrow<py::object>(it->second);
        }
    }

    // GIL must be held by caller
    // Import driver and get method
    py::module driver_module = py::module::import("skytorch.torch.backend._driver");
    py::object driver = driver_module.attr("driver");
    py::object method = driver.attr("get_method")(name);

    // Cache the method (increment ref count for cache ownership)
    PyObject* raw_ptr = method.ptr();
    Py_INCREF(raw_ptr);

    {
        std::lock_guard<std::mutex> lock(g_method_cache_mutex);
        // Check if another thread already cached it
        auto it = g_method_cache.find(name);
        if (it != g_method_cache.end()) {
            // Already cached, release our new reference
            Py_DECREF(raw_ptr);
        } else {
            g_method_cache[name] = raw_ptr;
        }
    }

    return method;
}

/**
 * Initialize the driver connection.
 */
void init_driver() {
    if (g_driver_initialized) {
        return;
    }

    py::gil_scoped_acquire acquire;

    // Import driver module to ensure it's initialized
    py::module::import("skytorch.torch.backend._driver");

    g_driver_initialized = true;
}

// Thread-local current device index (avoids GIL acquisition in current_device)
// This is critical for PyTorch 2.10 compatibility - the allocator calls
// current_device() and must be GIL-free to prevent kHasPyObject issues.
static thread_local c10::DeviceIndex g_current_device = 0;

// Device management functions
c10::DeviceIndex device_count() {
    py::gil_scoped_acquire acquire;
    return get_method("device_count")().cast<c10::DeviceIndex>();
}

c10::DeviceIndex current_device() {
    // NO GIL - read from C++ thread-local
    // This is called by the allocator and must be GIL-free
    return g_current_device;
}

void set_device(c10::DeviceIndex device) {
    // Update C++ thread-local first (GIL-free)
    g_current_device = device;
    // Then sync to Python for consistency
    py::gil_scoped_acquire acquire;
    get_method("set_device")(device);
}

void set_device_count(c10::DeviceIndex count) {
    py::gil_scoped_acquire acquire;
    get_method("set_device_count")(count);
}

c10::DeviceIndex exchange_device(c10::DeviceIndex device) {
    // Exchange in C++ thread-local first
    auto old = g_current_device;
    g_current_device = device;
    // Sync to Python
    py::gil_scoped_acquire acquire;
    get_method("exchange_device")(device);
    return old;
}

// Initialization function
void init() {
    // Register backend name with C10
    c10::register_privateuse1_backend("sky");

    // Register components
    register_guard();
    register_allocator();
    register_hooks();

    // Initialize driver connection
    init_driver();
}

/**
 * Get the default generator for a device.
 *
 * This is exposed to Python for RNG state management.
 */
py::object get_default_generator(c10::DeviceIndex device_index) {
    auto generator = at::globalContext().defaultGenerator(
        c10::Device(c10::DeviceType::PrivateUse1, device_index));
    return py::cast(generator);
}

/**
 * Get the metadata hash for a SkyTorch tensor.
 *
 * This is exposed to Python for efficient caching.
 */
py::object get_metadata_hash(py::object tensor_obj) {
    // Extract the tensor from the Python object
    auto tensor = py::cast<at::Tensor>(tensor_obj);

    // Check if tensor is using our custom TensorImpl
    auto* impl_ptr = dynamic_cast<TensorImpl*>(tensor.unsafeGetTensorImpl());
    if (impl_ptr) {
        auto metadata_hash = impl_ptr->get_metadata_hash();
        return py::int_(metadata_hash);
    } else {
        throw std::runtime_error("Tensor is not a SkyTorch tensor with custom TensorImpl");
    }
}

}  // namespace skytorch

// Python module definition
PYBIND11_MODULE(_C, m) {
    m.doc() = "SkyTorch PyTorch Backend C++ Extension";

    // Initialize the backend
    skytorch::init();
    m.def("_init", &skytorch::init, "Initialize SkyTorch backend");

    // RNG and metadata functions
    m.def("_get_default_generator", &skytorch::get_default_generator,
        "Get the default generator for a SkyTorch device");
    m.def("_get_metadata_hash", &skytorch::get_metadata_hash,
        "Get the metadata hash for a SkyTorch tensor");

    // Cleanup function to avoid GIL issues at shutdown
    m.def("_clear_method_cache", &skytorch::clear_method_cache,
        "Clear the method cache (call before shutdown)");

    // Register cleanup with atexit
    py::module atexit = py::module::import("atexit");
    atexit.attr("register")(py::cpp_function(&skytorch::clear_method_cache));
}
