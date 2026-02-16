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
#include <cstdint>
#include <string>
#include <unordered_map>

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

    // Python API (GIL required, for service.py)
    py::object get_python(uint64_t id);
    void set_python(uint64_t id, py::object t);
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
 * Args:
 *   data: Concatenated binary ops, each prefixed with uint32 length
 *   store: Server tensor manager's TensorStore
 */
void execute_raw_batched_aten_inline(py::bytes data, TensorStore& store);

/**
 * Clear all cached op/attr lookups.
 *
 * Must be called before Python shuts down to avoid GIL issues
 * when static destructors run. Registered with atexit.
 */
void clear_op_cache();

}  // namespace server
}  // namespace skytorch
