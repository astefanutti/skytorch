/**
 * SkyTorch Server - Binary Request Parser
 *
 * C++ implementation of the server-side binary ATen request parser.
 * Replaces _execute_raw_aten_inline and the batched variant with
 * direct memory reads, cached op lookups, and fast tensor dict access.
 *
 * Binary format is defined in RequestBuilder.h (client-side serializer).
 */

#pragma once

#include <pybind11/pybind11.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/ivalue.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/Allocator.h>
#include <c10/util/SmallVector.h>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <optional>
#include <vector>

namespace py = pybind11;

namespace skytorch {
namespace server {

// Arg type tags for binary deserialization (mirrored from RequestBuilder.h:61-73)
enum class ArgType : uint8_t {
    NONE = 0x00,
    TENSOR_ID = 0x01,
    INT64 = 0x02,
    FLOAT64 = 0x03,
    BOOL = 0x04,
    DTYPE = 0x05,
    MEMORY_FORMAT = 0x06,
    LAYOUT = 0x07,
    STRING = 0x08,
    LIST = 0x09,
    TUPLE = 0x0A,
};

/**
 * C++ tensor storage replacing Python dict for GIL-free access.
 *
 * Stores at::Tensor objects in a std::unordered_map<uint64_t, at::Tensor>,
 * enabling tensor lookups, arg parsing, and output registration without
 * holding the GIL. This allows batch-level GIL release (~1.5ms continuous)
 * instead of per-op release (~26μs fragments).
 */
class TensorStore {
    std::unordered_map<uint64_t, at::Tensor> tensors_;
public:
    // GIL-free C++ API (hot path)
    at::Tensor* find(uint64_t id);
    at::Tensor& get(uint64_t id);
    void set(uint64_t id, at::Tensor t);
    bool contains(uint64_t id) const;
    bool erase(uint64_t id);
    void clear();
    size_t size() const;
    void reserve(size_t n);

    // Python API (GIL required, for service.py)
    py::object get_python(uint64_t id);
    void set_python(uint64_t id, py::object t);
};

// --- OpInfo cache (Steps 2-3, 5) ---

// Return pattern tags for specialized output extraction
enum ReturnPattern : uint8_t {
    RETURN_SINGLE_TENSOR = 0,
    RETURN_TUPLE = 1,
    RETURN_GENERIC = 2,
};

/**
 * Cached per-op schema information.
 *
 * Populated on first call, reused on subsequent calls to skip
 * schema()/arguments()/default filling/coercion checks.
 */
struct OpInfo {
    std::optional<c10::OperatorHandle> handle;
    PyObject* py_op = nullptr;       // cached Python callable (for kwargs/fallback)
    size_t num_schema_args = 0;
    std::vector<c10::IValue> default_values;  // pre-resolved defaults
    bool skip_coercion = false;       // true after first successful call without coercion
    bool callboxed_blocked = false;   // true if callBoxed threw an exception
    ReturnPattern return_pattern = RETURN_GENERIC;
    uint8_t expected_return_count = 0;
};

/**
 * Execute a single raw binary execute_aten request inline.
 *
 * Parses the binary data, resolves tensors from the TensorStore,
 * executes the ATen op, and registers output tensors.
 *
 * Args:
 *   data: Binary-serialized execute_aten request (from C++ RequestBuilder)
 *   store: Server tensor manager's TensorStore
 */
void execute_raw_aten_inline(py::bytes data, TensorStore& store);

/**
 * Execute a batch of raw binary execute_aten requests inline.
 *
 * Parses the [uint32 len][op_data]... format and executes each op.
 * Releases the GIL for the entire batch loop, re-acquiring only
 * when kwargs/fallback paths require Python API calls.
 *
 * Returns the number of ops executed (for profiling).
 *
 * Args:
 *   data: Concatenated binary ops, each prefixed with uint32 length
 *   store: Server tensor manager's TensorStore
 */
size_t execute_raw_batched_aten_inline(py::bytes data, TensorStore& store);

/**
 * Scan a raw batched payload for module_forward marker (0xFF).
 *
 * The batch format is [uint32_len][op_data]... — checks the first byte
 * of each op_data segment. Only 0xFF (module forward) triggers mixed batch;
 * 0xFE (copy_tensor) is handled inline in the C++ batch path.
 */
bool batch_has_special_op(py::bytes data);

/**
 * Clear all cached op/attr lookups.
 *
 * Must be called before Python shuts down to avoid GIL issues
 * when static destructors run. Registered with atexit.
 */
void clear_op_cache();

// --- Server profiling (Step 1) ---

void set_server_profiling_enabled(bool enabled);
py::dict get_server_profile_counters();
void reset_server_profile_counters();

}  // namespace server
}  // namespace skytorch
