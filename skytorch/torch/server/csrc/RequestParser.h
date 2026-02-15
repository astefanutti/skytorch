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
 * Execute a single raw binary execute_aten request inline.
 *
 * Parses the binary data, resolves tensors from tensor_dict,
 * executes the ATen op, and registers output tensors.
 *
 * Args:
 *   data: Binary-serialized execute_aten request (from C++ RequestBuilder)
 *   tensor_dict: Server tensor manager's _tensors dict (int -> Tensor)
 */
void execute_raw_aten_inline(py::bytes data, py::dict tensor_dict);

/**
 * Execute a batch of raw binary execute_aten requests inline.
 *
 * Parses the [uint32 len][op_data]... format and executes each op.
 * Avoids Python bytes slicing overhead.
 *
 * Args:
 *   data: Concatenated binary ops, each prefixed with uint32 length
 *   tensor_dict: Server tensor manager's _tensors dict (int -> Tensor)
 */
void execute_raw_batched_aten_inline(py::bytes data, py::dict tensor_dict);

/**
 * Clear all cached op/attr lookups.
 *
 * Must be called before Python shuts down to avoid GIL issues
 * when static destructors run. Registered with atexit.
 */
void clear_op_cache();

}  // namespace server
}  // namespace skytorch
