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

// Periodic callback interval: every N cache-hit ops, call back into Python
// to tick the main event loop (metrics, logs, compute events). At ~7µs/op,
// 64 ops ≈ 450µs between callbacks. The Python bytecodes executed by the
// callback also give CPython a chance to switch the GIL to the backend
// event loop thread. Set to 0 to disable.
static const int g_periodic_interval = [] {
    const char* env = std::getenv("SKYTORCH_PERIODIC_INTERVAL");
    return env ? std::atoi(env) : 64;
}();

// Profiling flag and counters (defined in RequestBuilder.cpp)
extern bool g_profiling_enabled;
extern std::atomic<int64_t> g_prof_fast_path_count;
extern std::atomic<int64_t> g_prof_ivalue_to_py_ns;
extern std::atomic<int64_t> g_prof_dispatch_cached_ns;
extern std::atomic<int64_t> g_prof_rewrite_stack_ns;
extern std::atomic<int64_t> g_prof_python_fallback_count;
extern std::atomic<int64_t> g_prof_python_fallback_ns;
extern std::atomic<int64_t> g_prof_last_dispatch_end_ns;
extern std::atomic<int64_t> g_prof_inter_op_gap_total_ns;
extern std::atomic<int64_t> g_prof_inter_op_gap_count;
extern std::atomic<int64_t> g_prof_gil_release_count;
extern std::atomic<int64_t> g_prof_gil_wait_ns;
extern std::atomic<int64_t> g_prof_autograd_overhead_ns;
// Gap sub-decomposition: time OUTSIDE autograd (Python + outer dispatch)
extern std::atomic<int64_t> g_prof_outside_autograd_ns;
extern std::atomic<int64_t> g_prof_outside_autograd_count;
// Gap histogram buckets: <1µs, 1-10µs, 10-100µs, 100µs-1ms, 1-10ms, >10ms
extern std::atomic<int64_t> g_prof_gap_hist[6];
// Large gap (>1ms) breakdown: GIL wait vs other
extern std::atomic<int64_t> g_prof_large_gap_gil_ns;
extern std::atomic<int64_t> g_prof_large_gap_other_ns;
extern std::atomic<int64_t> g_prof_large_gap_count;

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

// --- Periodic callback (ticks the main thread's event loop) ---
// Called every g_periodic_interval cache-hit ops on the main thread.
// Lets the main event loop process pending callbacks (metrics, logs,
// compute events) during the forward pass.

static PyObject* g_periodic_callback = nullptr;

void set_periodic_callback(py::object callback) {
    Py_XDECREF(g_periodic_callback);
    g_periodic_callback = callback.ptr();
    Py_INCREF(g_periodic_callback);
}


// Pending fused result: stores the dispatch_cached_aten result from
// fallback_kernel so that _sky_kernel_fallback can skip the redundant
// _dispatch_cached_aten call. Wrapped in a 1-tuple to distinguish
// "tried, got None" from "not tried" (which returns None).
// GIL is held throughout fallback_kernel → call_python_fallback →
// _sky_kernel_fallback, so no thread-safety concern.
static PyObject* g_pending_fused = nullptr;

void set_pending_fused_result(py::object result) {
    Py_XDECREF(g_pending_fused);
    // Wrap in (result,) tuple so Python can distinguish from "not set"
    g_pending_fused = PyTuple_New(1);
    Py_INCREF(result.ptr());
    PyTuple_SET_ITEM(g_pending_fused, 0, result.ptr());
}

py::object take_pending_fused_result() {
    if (g_pending_fused == nullptr) {
        return py::none();  // not set
    }
    py::object result = py::reinterpret_steal<py::object>(g_pending_fused);
    g_pending_fused = nullptr;
    return result;
}

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

// Thread-local timestamp of last autograd_fallback_kernel exit.
// Used to measure "outside autograd" time: the time between returning
// from autograd and entering the next autograd call. This captures
// Python bytecode + PyTorch outer dispatcher overhead.
static thread_local int64_t g_last_autograd_end_ns = 0;

void autograd_fallback_kernel(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
    // Exclude autograd + ADInplaceOrView keys and redispatch to PrivateUse1.
    // Stays entirely in C++ dispatch — no Python re-entry, no CompositeImplicitAutograd
    // decomposition. This is the pattern used by XLA and other custom backends.
    try {
        auto ag_t0 = g_profiling_enabled
            ? std::chrono::steady_clock::now()
            : std::chrono::steady_clock::time_point{};

        // Measure time OUTSIDE autograd: previous autograd exit → this autograd entry.
        // This captures: Python model code + PyTorch outer dispatcher (Python→C++ boundary,
        // dispatch key selection, Python binding boxing/unboxing).
        if (g_profiling_enabled && g_last_autograd_end_ns > 0) {
            int64_t entry_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                ag_t0.time_since_epoch()).count();
            int64_t outside_ns = entry_ns - g_last_autograd_end_ns;
            if (outside_ns > 0) {
                g_prof_outside_autograd_ns.fetch_add(outside_ns, std::memory_order_relaxed);
                g_prof_outside_autograd_count.fetch_add(1, std::memory_order_relaxed);
            }
        }

        at::AutoDispatchBelowADInplaceOrView guard;
        op.callBoxed(stack);

        if (g_profiling_enabled) {
            auto ag_t1 = std::chrono::steady_clock::now();
            // Total autograd time (includes fallback_kernel inside).
            int64_t ag_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                ag_t1 - ag_t0).count();
            g_prof_autograd_overhead_ns.fetch_add(ag_ns, std::memory_order_relaxed);
            // Record exit timestamp for next iteration's "outside" measurement.
            g_last_autograd_end_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                ag_t1.time_since_epoch()).count();
        }
    } catch (py::error_already_set& e) {
        // During redispatch, device guard callbacks (e.g. exchange_device) call into
        // Python where a pending KeyboardInterrupt can raise py::error_already_set.
        if (e.matches(PyExc_KeyboardInterrupt) ||
            e.matches(PyExc_SystemExit) ||
            e.matches(PyExc_GeneratorExit)) {
            // Cannot re-throw py::error_already_set here — it would propagate
            // through PyTorch's C++ dispatcher which only catches c10::Error,
            // hitting noexcept boundaries and calling std::terminate().
            // Instead, reschedule the signal and convert to c10::Error.
            py::gil_scoped_acquire gil;
            e.restore();
            PyErr_Clear();
            PyErr_SetInterrupt();
            TORCH_CHECK(false, "KeyboardInterrupt");
        }
        // Convert regular exceptions to c10::Error so they propagate safely
        // through PyTorch's C++ dispatcher.
        py::gil_scoped_acquire gil;
        e.restore();
        TORCH_CHECK(false, e.what());
    }
}

// --- Boxed Fallback Kernel ---

void fallback_kernel(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
    // Pre-GIL timestamp to measure GIL acquisition overhead
    auto pre_gil_ts = g_profiling_enabled
        ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};

    py::gil_scoped_acquire gil;

    try {

    // --- Profiling: inter-op gap measurement ---
    auto kernel_t0 = g_profiling_enabled
        ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};
    if (g_profiling_enabled) {
        // GIL acquisition time
        int64_t gil_wait = std::chrono::duration_cast<std::chrono::nanoseconds>(
            kernel_t0 - pre_gil_ts).count();
        if (gil_wait > 0) {
            g_prof_gil_wait_ns.fetch_add(gil_wait, std::memory_order_relaxed);
        }

        int64_t last_end = g_prof_last_dispatch_end_ns.load(std::memory_order_relaxed);
        if (last_end > 0) {
            int64_t pre_gil_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                pre_gil_ts.time_since_epoch()).count();
            int64_t now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                kernel_t0.time_since_epoch()).count();
            int64_t gap = now_ns - last_end;
            if (gap > 0) {
                g_prof_inter_op_gap_total_ns.fetch_add(gap, std::memory_order_relaxed);
                g_prof_inter_op_gap_count.fetch_add(1, std::memory_order_relaxed);
                // Gap histogram
                int bucket;
                if (gap < 1000) bucket = 0;           // <1µs
                else if (gap < 10000) bucket = 1;      // 1-10µs
                else if (gap < 100000) bucket = 2;     // 10-100µs
                else if (gap < 1000000) bucket = 3;    // 100µs-1ms
                else if (gap < 10000000) bucket = 4;   // 1-10ms
                else bucket = 5;                       // >10ms
                g_prof_gap_hist[bucket].fetch_add(1, std::memory_order_relaxed);

                // For large gaps (>1ms), decompose into GIL wait vs other.
                // GIL wait = time waiting for GIL acquisition (event loop holding it).
                // Other = time in C++ before GIL (dispatcher, autograd, etc.)
                //       + time after GIL but before our profiling timestamp.
                if (gap > 1000000) {  // >1ms
                    // pre_gil_ns is BEFORE GIL acquire, so:
                    //   last_end → pre_gil = "not waiting for GIL" (C++ dispatch + Python)
                    //   pre_gil → now = "GIL acquire" (may include event loop work)
                    int64_t gap_before_gil = pre_gil_ns - last_end;
                    int64_t gap_during_gil = now_ns - pre_gil_ns;
                    g_prof_large_gap_other_ns.fetch_add(
                        std::max(int64_t(0), gap_before_gil), std::memory_order_relaxed);
                    g_prof_large_gap_gil_ns.fetch_add(
                        std::max(int64_t(0), gap_during_gil), std::memory_order_relaxed);
                    g_prof_large_gap_count.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
    }

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
                // Note: ops counter already incremented by _submit_and_register callback
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
                    g_prof_last_dispatch_end_ns.store(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            t3.time_since_epoch()).count(),
                        std::memory_order_relaxed);
                }

                // Periodic callback: every N ops, tick the main thread's
                // event loop so it can process async callbacks (metrics,
                // logs, compute events) that would otherwise be starved
                // during C++ dispatch. The Python bytecodes executed by
                // the callback also give CPython a chance to switch the
                // GIL to the backend event loop thread.
                {
                    static thread_local int g_ops_since_callback = 0;

                    if (g_periodic_interval > 0 &&
                        ++g_ops_since_callback >= g_periodic_interval) {
                        g_ops_since_callback = 0;
                        if (g_profiling_enabled) {
                            g_prof_gil_release_count.fetch_add(1, std::memory_order_relaxed);
                        }

                        // Reentrancy guard: if a callback dispatches ATen ops,
                        // the nested fallback_kernel won't trigger another tick.
                        static thread_local bool in_callback = false;
                        if (g_periodic_callback && !in_callback) {
                            in_callback = true;
                            PyObject* result = PyObject_CallNoArgs(
                                g_periodic_callback);
                            if (result == nullptr) {
                                PyErr_Clear();
                            } else {
                                Py_DECREF(result);
                            }
                            in_callback = false;
                        }
                    }
                }

                return;
            }
        }
        // Cache miss (Tuple(3)) or uncacheable (None) — no side effects,
        // safe to fall through to Python.
        // Store the result so Python can skip the redundant _dispatch_cached_aten call.
        set_pending_fused_result(fused);
    }

    // --- Python fallback: cache miss, uncacheable, or no callback yet ---
    {
        auto py_t0 = g_profiling_enabled
            ? std::chrono::steady_clock::now()
            : std::chrono::steady_clock::time_point{};

        call_python_fallback(op, stack);

        if (g_profiling_enabled) {
            auto py_t1 = std::chrono::steady_clock::now();
            g_prof_python_fallback_ns.fetch_add(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    py_t1 - py_t0).count(),
                std::memory_order_relaxed);
            g_prof_python_fallback_count.fetch_add(1, std::memory_order_relaxed);
            g_prof_last_dispatch_end_ns.store(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    py_t1.time_since_epoch()).count(),
                std::memory_order_relaxed);
        }
    }

    } catch (py::error_already_set& e) {
        if (e.matches(PyExc_KeyboardInterrupt) ||
            e.matches(PyExc_SystemExit) ||
            e.matches(PyExc_GeneratorExit)) {
            // Cannot re-throw py::error_already_set here — it would propagate
            // through PyTorch's C++ dispatcher which only catches c10::Error,
            // hitting noexcept boundaries and calling std::terminate().
            // Instead, reschedule the signal and convert to c10::Error.
            // GIL is already held (acquired at top of fallback_kernel).
            e.restore();
            PyErr_Clear();
            PyErr_SetInterrupt();
            TORCH_CHECK(false, "KeyboardInterrupt");
        }
        // Convert regular exceptions to c10::Error so they propagate safely
        // through PyTorch's C++ dispatcher (which doesn't catch pybind11 exceptions).
        // Restore the Python error state so it re-surfaces at the Python/C++ boundary.
        e.restore();
        TORCH_CHECK(false, e.what());
    }
}

}  // namespace skytorch
