/**
 * SkyTorch PyTorch Backend - C++ Boxed Fallback Kernel
 *
 * Registers a boxed fallback for the PrivateUse1 dispatch key that handles
 * cache-hit ops entirely in C++, bypassing the Python wrapper chain.
 *
 * For cache hits (~98.5% of ops), the flow is:
 *   C++ dispatcher → fallback_kernel → dispatch_cached_aten (C++) → submit
 * Only cache misses fall back to Python for meta tensor execution.
 */

#include "RequestBuilder.h"

#include <torch/extension.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <ATen/core/LegacyTypeDispatch.h>
#include <atomic>
#include <chrono>
#include <unordered_map>

namespace skytorch {

// Profiling flag and counters (defined in RequestBuilder.cpp)
extern bool g_profiling_enabled;
extern std::atomic<int64_t> g_prof_fast_path_count;
extern std::atomic<int64_t> g_prof_ivalue_to_py_ns;
extern std::atomic<int64_t> g_prof_dispatch_cached_ns;
extern std::atomic<int64_t> g_prof_rewrite_stack_ns;

// --- Python fallback callback ---

static PyObject* g_python_fallback = nullptr;

void set_python_fallback(py::object callback) {
    Py_XDECREF(g_python_fallback);
    g_python_fallback = callback.ptr();
    Py_INCREF(g_python_fallback);
}

void clear_python_fallback() {
    Py_XDECREF(g_python_fallback);
    g_python_fallback = nullptr;
}

// Pending fused result not needed — see fallback_kernel comments.
void set_pending_fused_result(py::object) {}
py::object take_pending_fused_result() { return py::none(); }

// --- Op name cache (per OperatorHandle address) ---

static std::unordered_map<const void*, std::string> g_op_name_cache;
static std::unordered_map<const void*, PyObject*> g_op_overload_cache;
static std::unordered_map<const void*, size_t> g_pos_args_count_cache;

static const std::string& get_cached_op_name(const c10::OperatorHandle& op) {
    const void* key = &op;
    auto it = g_op_name_cache.find(key);
    if (it != g_op_name_cache.end()) {
        return it->second;
    }
    const auto& schema = op.schema();
    std::string name = schema.name();
    auto pos = name.find("::");
    if (pos != std::string::npos) {
        name.replace(pos, 2, ".");
    }
    const auto& overload = schema.overload_name();
    if (!overload.empty()) {
        name += "." + overload;
    } else {
        name += ".default";
    }
    auto [inserted_it, _] = g_op_name_cache.emplace(key, std::move(name));
    return inserted_it->second;
}

static size_t get_num_positional_args(const c10::OperatorHandle& op) {
    const void* key = &op;
    auto it = g_pos_args_count_cache.find(key);
    if (it != g_pos_args_count_cache.end()) {
        return it->second;
    }
    const auto& arguments = op.schema().arguments();
    size_t n_pos = 0;
    for (const auto& arg : arguments) {
        if (arg.kwarg_only()) break;
        n_pos++;
    }
    g_pos_args_count_cache[key] = n_pos;
    return n_pos;
}

static py::object get_cached_op_overload(const c10::OperatorHandle& op) {
    const void* key = &op;
    auto it = g_op_overload_cache.find(key);
    if (it != g_op_overload_cache.end()) {
        return py::reinterpret_borrow<py::object>(it->second);
    }

    const auto& schema = op.schema();
    std::string full_name = schema.name();
    std::string ns = "aten";
    std::string op_base = full_name;
    auto sep = full_name.find("::");
    if (sep != std::string::npos) {
        ns = full_name.substr(0, sep);
        op_base = full_name.substr(sep + 2);
    }

    py::object torch_ops = py::module::import("torch").attr("ops");
    py::object ns_obj = torch_ops.attr(ns.c_str());
    py::object op_packet = ns_obj.attr(op_base.c_str());

    const auto& overload = schema.overload_name();
    py::object op_overload;
    if (overload.empty()) {
        op_overload = op_packet.attr("default");
    } else {
        op_overload = op_packet.attr(overload.c_str());
    }

    PyObject* raw = op_overload.ptr();
    Py_INCREF(raw);
    g_op_overload_cache[key] = raw;

    return op_overload;
}

// --- Call Python fallback ---

static void call_python_fallback(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack)
{
    TORCH_CHECK(g_python_fallback,
        "SkyTorch: no Python fallback registered for op ", op.schema().name());

    // GIL already held by caller

    py::object op_overload = get_cached_op_overload(op);

    // Split stack into positional args + kwargs based on schema
    const auto& arguments = op.schema().arguments();
    size_t num_stack_args = stack->size();
    size_t n_pos = get_num_positional_args(op);
    size_t actual_pos = std::min(n_pos, num_stack_args);

    // Build call args: _sky_kernel_fallback(op_overload, *pos_args, **kw_args)
    PyObject* cb_args = PyTuple_New(1 + static_cast<Py_ssize_t>(actual_pos));
    Py_INCREF(op_overload.ptr());
    PyTuple_SET_ITEM(cb_args, 0, op_overload.ptr());
    for (size_t i = 0; i < actual_pos; i++) {
        PyObject* item = torch::jit::toPyObject((*stack)[i]).release().ptr();
        PyTuple_SET_ITEM(cb_args, 1 + static_cast<Py_ssize_t>(i), item);
    }

    // Build kwargs dict for keyword-only args
    PyObject* kwargs_dict = nullptr;
    if (actual_pos < num_stack_args) {
        kwargs_dict = PyDict_New();
        for (size_t i = actual_pos; i < num_stack_args && i < arguments.size(); i++) {
            const auto& iv = (*stack)[i];
            if (iv.isNone() && arguments[i].default_value().has_value()) {
                continue;
            }
            PyObject* val = torch::jit::toPyObject(iv).release().ptr();
            PyDict_SetItemString(kwargs_dict, arguments[i].name().c_str(), val);
            Py_DECREF(val);  // PyDict_SetItemString increments ref
        }
        if (PyDict_Size(kwargs_dict) == 0) {
            Py_DECREF(kwargs_dict);
            kwargs_dict = nullptr;
        }
    }

    PyObject* result = PyObject_Call(g_python_fallback, cb_args, kwargs_dict);
    Py_DECREF(cb_args);
    Py_XDECREF(kwargs_dict);

    if (!result) {
        throw py::error_already_set();
    }

    // Rewrite stack with result
    stack->clear();
    const auto& returns = op.schema().returns();

    if (returns.empty()) {
        Py_DECREF(result);
        return;
    }

    if (returns.size() == 1) {
        stack->push_back(torch::jit::toIValue(result, returns[0].type()));
        Py_DECREF(result);
        return;
    }

    if (PyTuple_Check(result)) {
        Py_ssize_t n = PyTuple_GET_SIZE(result);
        for (Py_ssize_t i = 0; i < n && i < static_cast<Py_ssize_t>(returns.size()); i++) {
            stack->push_back(torch::jit::toIValue(
                PyTuple_GET_ITEM(result, i), returns[i].type()));
        }
    } else {
        stack->push_back(torch::jit::toIValue(result, returns[0].type()));
    }
    Py_DECREF(result);
}

// --- Helper: rewrite stack from unpacked Python output ---

static void rewrite_stack_from_output(
    torch::jit::Stack* stack,
    PyObject* unpacked,
    const c10::FunctionSchema& schema)
{
    stack->clear();
    const auto& returns = schema.returns();

    if (returns.empty()) return;

    if (returns.size() == 1) {
        stack->push_back(torch::jit::toIValue(unpacked, returns[0].type()));
        return;
    }

    if (PyTuple_Check(unpacked)) {
        Py_ssize_t n = PyTuple_GET_SIZE(unpacked);
        for (Py_ssize_t i = 0; i < n && i < static_cast<Py_ssize_t>(returns.size()); i++) {
            stack->push_back(torch::jit::toIValue(
                PyTuple_GET_ITEM(unpacked, i), returns[i].type()));
        }
    } else {
        stack->push_back(torch::jit::toIValue(unpacked, returns[0].type()));
    }
}

// --- Autograd Fallback Kernel ---

void autograd_fallback_kernel(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
    // Exclude autograd + ADInplaceOrView keys and redispatch to PrivateUse1.
    // Stays entirely in C++ dispatch — no Python re-entry, no CompositeImplicitAutograd
    // decomposition. This is the pattern used by XLA and other custom backends.
    at::AutoDispatchBelowADInplaceOrView guard;
    op.callBoxed(stack);
}

// --- Boxed Fallback Kernel ---

void fallback_kernel(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
    py::gil_scoped_acquire gil;

    // --- C++ fast path: try dispatch_cached_aten for cache hits ---
    // Only attempt when the submit callback is registered. Without the callback,
    // dispatch_cached_aten returns Tuple(5) on cache hits, which registers tensor
    // IDs as a side effect. If we then fall through to call_python_fallback, the
    // Python path calls dispatch_cached_aten again, creating different output
    // tensors — leaving phantom tensor IDs in the C++ tracking set.
    if (has_submit_callback()) {
        const auto& schema = op.schema();
        const auto& arguments = schema.arguments();
        size_t num_stack_args = stack->size();
        size_t n_pos = get_num_positional_args(op);
        size_t actual_pos = std::min(n_pos, num_stack_args);

        auto t0 = g_profiling_enabled
            ? std::chrono::steady_clock::now()
            : std::chrono::steady_clock::time_point{};

        // Convert IValue stack → Python args/kwargs
        py::tuple py_args(actual_pos);
        for (size_t i = 0; i < actual_pos; i++) {
            py_args[i] = torch::jit::toPyObject((*stack)[i]);
        }

        py::dict py_kwargs;
        for (size_t i = actual_pos; i < num_stack_args && i < arguments.size(); i++) {
            const auto& iv = (*stack)[i];
            if (iv.isNone() && arguments[i].default_value().has_value()) continue;
            py_kwargs[py::str(arguments[i].name().c_str())] = torch::jit::toPyObject(iv);
        }

        auto t1 = g_profiling_enabled
            ? std::chrono::steady_clock::now()
            : std::chrono::steady_clock::time_point{};

        const std::string& op_name = get_cached_op_name(op);
        py::object fused = dispatch_cached_aten(
            py::str(op_name), py_args, py_kwargs);

        if (!fused.is_none()) {
            py::tuple rt = fused.cast<py::tuple>();
            if (rt.size() == 1) {
                auto t2 = g_profiling_enabled
                    ? std::chrono::steady_clock::now()
                    : std::chrono::steady_clock::time_point{};

                // Cache hit with callback — fully handled in C++
                increment_ops_counter();
                rewrite_stack_from_output(stack, rt[0].ptr(), schema);

                if (g_profiling_enabled) {
                    auto t3 = std::chrono::steady_clock::now();
                    g_prof_ivalue_to_py_ns.fetch_add(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            t1 - t0).count(),
                        std::memory_order_relaxed);
                    g_prof_dispatch_cached_ns.fetch_add(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            t2 - t1).count(),
                        std::memory_order_relaxed);
                    g_prof_rewrite_stack_ns.fetch_add(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            t3 - t2).count(),
                        std::memory_order_relaxed);
                    g_prof_fast_path_count.fetch_add(1, std::memory_order_relaxed);
                }
                return;
            }
        }
        // Cache miss (Tuple(3)) or uncacheable (None) — no side effects,
        // safe to fall through to Python.
    }

    // --- Python fallback: cache miss, uncacheable, or no callback yet ---
    call_python_fallback(op, stack);
}

}  // namespace skytorch
