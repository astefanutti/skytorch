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
#include "RequestBuilder.h"

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

// Forward declaration for create_remote_tensor (defined in RemoteTensor.cpp)
torch::Tensor create_remote_tensor(
    int64_t storage_id,
    std::vector<int64_t> shape,
    std::string dtype_str,
    std::vector<int64_t> stride,
    int64_t storage_offset,
    int64_t nbytes,
    int64_t device_index);

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

    // Remote tensor creation
    m.def("_create_remote_tensor", &skytorch::create_remote_tensor,
        "Create a sky tensor with a server-assigned storage ID");

    // Cleanup function to avoid GIL issues at shutdown
    m.def("_clear_method_cache", &skytorch::clear_method_cache,
        "Clear the method cache (call before shutdown)");

    // Binary request builder for fast ATen operation serialization
    m.def("_build_execute_aten_request", &skytorch::build_execute_aten_request,
        "Build a binary-serialized execute_aten request from Python arguments",
        py::arg("op_name"),
        py::arg("args"),
        py::arg("kwargs"),
        py::arg("output_tensors"),
        py::arg("device_index"),
        py::arg("remote_device_type"),
        py::arg("remote_device_index"));

    // Tensor ID registration tracking (sync C++ set with Python storage manager)
    m.def("_register_tensor_id", &skytorch::register_tensor_id,
        "Register a tensor ID as known to the server");
    m.def("_unregister_tensor_id", &skytorch::unregister_tensor_id,
        "Unregister a tensor ID");
    m.def("_clear_registered_tensor_ids", &skytorch::clear_registered_tensor_ids,
        "Clear all registered tensor IDs (for testing/reset)");
    m.def("_register_storage_tensor_mapping", &skytorch::register_storage_tensor_mapping,
        "Register a storage_id to tensor_id mapping for view detection");

    // Dispatch context computation (cache key + tensor collection in one C++ pass)
    m.def("_compute_dispatch_context", &skytorch::compute_dispatch_context,
        "Compute dispatch context: cache key hash, input tensors, and sky device index",
        py::arg("op_name"),
        py::arg("args"),
        py::arg("kwargs"));

    // Fused dispatch: hash + cache lookup + output creation + serialization in one C++ call
    m.def("_dispatch_cached_aten", &skytorch::dispatch_cached_aten,
        "Fused dispatch for cache hits: hash, cache lookup, output creation, and serialization",
        py::arg("op_name"),
        py::arg("args"),
        py::arg("kwargs"));

    // Shape cache management
    m.def("_populate_shape_cache", &skytorch::populate_shape_cache,
        "Populate the C++ shape cache after meta execution",
        py::arg("cache_key"),
        py::arg("output_metas"));
    m.def("_clear_shape_cache", &skytorch::clear_shape_cache,
        "Clear all shape cache entries");

    // Device mapping registry
    m.def("_register_device_mapping", &skytorch::register_device_mapping,
        "Register a local sky device index to remote device mapping",
        py::arg("local_index"),
        py::arg("remote_type"),
        py::arg("remote_index"));
    m.def("_clear_device_mappings", &skytorch::clear_device_mappings,
        "Clear all device mappings");

    // Submit callback for fused C++ dispatch
    m.def("_set_submit_callback", &skytorch::set_submit_callback,
        "Set callback for submitting raw bytes from C++ dispatch",
        py::arg("callback"));
    m.def("_clear_submit_callback", &skytorch::clear_submit_callback,
        "Clear the submit callback");

    // Per-device cached submit methods for fast path (no new tensors)
    m.def("_set_submit_method", &skytorch::set_submit_method,
        "Cache a direct reference to stream_manager.submit_execute_aten_bytes",
        py::arg("dev_idx"),
        py::arg("method"));
    m.def("_clear_submit_methods", &skytorch::clear_submit_methods,
        "Clear all cached submit methods");

    // C++ native raw submit buffer (bypasses Python on fast path)
    m.def("_setup_cpp_submit", &skytorch::setup_cpp_submit,
        "Set up C++ raw submit buffer with event loop integration",
        py::arg("call_soon_threadsafe"),
        py::arg("drain_callback"));
    m.def("_clear_cpp_submit", &skytorch::clear_cpp_submit,
        "Clear C++ raw submit buffer state");
    m.def("_has_cpp_submit", &skytorch::has_cpp_submit,
        "Check whether C++ raw submit path is set up");
    m.def("_cpp_submit_raw_py", &skytorch::cpp_submit_raw_py,
        "Append raw bytes to the C++ submit buffer",
        py::arg("raw_bytes"));
    m.def("_drain_raw_submit_buffer", &skytorch::drain_raw_submit_buffer,
        "Drain all pending raw bytes from the C++ submit buffer");
    // Fire-and-forget ops counter (atomic, GIL-free)
    m.def("_increment_ops_counter", &skytorch::increment_ops_counter,
        "Increment the fire-and-forget ops counter");
    m.def("_get_ops_counter", &skytorch::get_ops_counter,
        "Read the current ops counter value");
    m.def("_reset_ops_counter", &skytorch::reset_ops_counter,
        "Reset the ops counter to zero and return previous value");

    // C++ fast path profiling
    m.def("_set_profiling_enabled", &skytorch::set_profiling_enabled,
        "Enable or disable C++ fast path profiling",
        py::arg("enabled"));
    m.def("_get_cpp_profile_counters", &skytorch::get_cpp_profile_counters,
        "Get C++ fast path profiling counters as {name: (total_ns, count)}");
    m.def("_reset_cpp_profile_counters", &skytorch::reset_cpp_profile_counters,
        "Reset all C++ profiling counters to zero");

    // Python fallback for C++ boxed fallback kernel (cache misses)
    m.def("_set_python_fallback", &skytorch::set_python_fallback,
        "Set Python fallback callback for cache miss ops",
        py::arg("callback"));
    m.def("_clear_python_fallback", &skytorch::clear_python_fallback,
        "Clear the Python fallback callback");

    // Pending fused result for avoiding double dispatch_cached_aten calls
    m.def("_take_pending_fused_result", &skytorch::take_pending_fused_result,
        "Take the pending fused result from the C++ boxed fallback");

    // Register cleanup with atexit
    py::module atexit = py::module::import("atexit");
    atexit.attr("register")(py::cpp_function(&skytorch::clear_method_cache));
}
