/**
 * SkyTorch PyTorch Backend - Binary Request Builder
 *
 * This module provides a C++ implementation of the ATen request builder
 * that serializes operation requests to a compact binary format, avoiding
 * the overhead of Python protobuf object construction.
 *
 * Binary format for ExecuteAtenRequest (v2 - metadata-first):
 *
 * Header (4 bytes):
 *   [num_args: uint8] [num_kwargs: uint8] [num_outputs: uint8] [num_metadata: uint8]
 *
 * Op name (variable):
 *   [op_name_len: uint16] [op_name: bytes]
 *
 * For each tensor metadata (moved before outputs/args for single-pass parsing):
 *   [tensor_id: uint64] [ndim: uint8]
 *   [shape: ndim * int64] [stride: ndim * int64]
 *   [dtype_str_len: uint8] [dtype_str: bytes] [storage_offset: int64] [nbytes: int64]
 *   [device_type_len: uint8] [device_type: bytes] [device_index: int32]
 *   [has_tensor_ref: uint8] [tensor_ref: uint64 if has_tensor_ref]
 *
 * For each output:
 *   [tensor_id: uint64]
 *
 * For each arg:
 *   [type: uint8] [value: varies by type]
 *
 * For each kwarg:
 *   [name_len: uint8] [name: bytes]
 *   [type + value: same as arg]
 *
 * Arg type tags:
 *   0x00 = none
 *   0x01 = tensor_id (uint64)
 *   0x02 = int64
 *   0x03 = float64
 *   0x04 = bool (uint8)
 *   0x05 = dtype string (uint8 len + bytes)
 *   0x06 = memory_format string (uint8 len + bytes)
 *   0x07 = layout string (uint8 len + bytes)
 *   0x08 = string (uint16 len + bytes)
 *   0x09 = list (uint16 count + recursive args)
 *   0x0A = tuple (uint16 count + recursive args)
 */

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/stack.h>
#include <c10/core/ScalarType.h>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

namespace py = pybind11;

namespace skytorch {

// Cached output metadata for a single output tensor
struct OutputMeta {
    std::vector<int64_t> shape;
    std::vector<int64_t> stride;
    c10::ScalarType dtype;
    int64_t storage_offset;
    int alias_input;  // -1 = new storage, -2 = None output, >=0 = input index
};

// Arg type tags for binary serialization
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
 * Build a binary-serialized execute_aten request from Python arguments.
 *
 * This function walks the Python args/kwargs, checks tensor registration
 * status, and serializes everything to a compact binary format. It returns
 * the serialized bytes and a list of new tensor IDs that need local registration.
 *
 * Args:
 *   op_name: ATen operation name (e.g., "aten.add.Tensor")
 *   args: Positional arguments (may contain sky tensors)
 *   kwargs: Keyword arguments
 *   output_tensors: Pre-allocated output tensors (list or None)
 *   device_index: Local sky device index
 *   remote_device_type: Remote device type string (e.g., "cuda")
 *   remote_device_index: Remote device index
 *
 * Returns:
 *   Tuple of (bytes, list[int]) where bytes is the serialized request
 *   and list[int] contains tensor_ids of newly encountered tensors.
 */
py::tuple build_execute_aten_request(
    const std::string& op_name,
    py::tuple args,
    py::dict kwargs,
    py::object output_tensors,
    int64_t device_index,
    const std::string& remote_device_type,
    int64_t remote_device_index);

/**
 * Register a tensor ID as known (already sent to server).
 */
void register_tensor_id(uint64_t tensor_id);

/**
 * Unregister a tensor ID (e.g., after server deletion).
 */
void unregister_tensor_id(uint64_t tensor_id);

/**
 * Clear all registered tensor IDs. Used for testing/reset.
 */
void clear_registered_tensor_ids();

/**
 * Register a storage_id → tensor_id mapping for view detection.
 * Only records the first tensor_id for each storage_id.
 */
void register_storage_tensor_mapping(int64_t storage_id, uint64_t tensor_id);

/**
 * Compute dispatch context for cache key + tensor collection in one C++ pass.
 *
 * Walks args/kwargs once to:
 *   1. Compute a 64-bit hash of the op signature (op_name + all arg shapes/values)
 *   2. Collect unique sky input tensors (by storage pointer)
 *   3. Find the first sky device index
 *
 * Returns: (cache_key_hash, input_tensors_list, sky_device_index)
 *   - cache_key_hash: 64-bit hash (0 = uncacheable args)
 *   - input_tensors_list: list of unique sky tensors
 *   - sky_device_index: first sky device index found (-1 if none)
 */
py::tuple compute_dispatch_context(
    py::str op_name,
    py::tuple args,
    py::dict kwargs);

/**
 * Fused dispatch for cache hits: hash + cache lookup + output creation + serialization.
 *
 * Returns:
 *   None → uncacheable args (Python does full meta execution without caching)
 *   Tuple(3) → cache miss: (cache_hash, input_tensors, sky_device_index)
 *   Tuple(5) → cache hit: (output_tensors_list, raw_bytes, new_tensor_ids, new_storage_ids, sky_device_index)
 */
py::object dispatch_cached_aten(
    py::str op_name,
    py::tuple args,
    py::dict kwargs);

/**
 * Populate the C++ shape cache after a meta execution miss.
 *
 * output_metas: list of (shape, stride, dtype_int, storage_offset, alias_input) tuples,
 *               or None for None outputs.
 */
void populate_shape_cache(uint64_t cache_key, py::list output_metas);

/**
 * Clear all shape cache entries.
 */
void clear_shape_cache();

/**
 * Register a local sky device index → (remote_device_type, remote_device_index) mapping.
 */
void register_device_mapping(int64_t local_index,
                              const std::string& remote_type,
                              int64_t remote_index);

/**
 * Clear all device mappings.
 */
void clear_device_mappings();

/**
 * Set callback for submitting raw bytes to the stream manager.
 * Called from dispatch_cached_aten on cache hits to avoid returning to Python.
 * The callback receives (raw_bytes, dev_idx) and calls
 * stream_manager.submit_execute_aten_bytes(raw_bytes).
 */
void set_submit_callback(py::object callback);

/**
 * Clear the submit callback. Called during reset/shutdown.
 */
void clear_submit_callback();

/**
 * Check if a submit callback is registered.
 * Used by fallback_kernel to decide whether to try the C++ fast path.
 * dispatch_cached_aten returns Tuple(5) when no callback is set, which
 * has tensor registration side effects that prevent safe fallthrough
 * to the Python path.
 */
bool has_submit_callback();

/**
 * Increment the fire-and-forget ops counter (atomic, relaxed ordering).
 * Called from the dispatch path for each submitted op.
 */
void increment_ops_counter();

/**
 * Read the current ops counter value without resetting.
 */
int64_t get_ops_counter();

/**
 * Reset the ops counter to zero and return the previous value.
 */
int64_t reset_ops_counter();

/**
 * C++ boxed fallback kernel for PrivateUse1 dispatch key.
 * Handles cache-hit ops entirely in C++, falling back to Python for misses.
 */
void fallback_kernel(const c10::OperatorHandle& op, torch::jit::Stack* stack);

/**
 * C++ autograd fallback kernel for AutogradPrivateUse1 dispatch key.
 * Excludes autograd keys and redispatches to PrivateUse1 via op.callBoxed(),
 * staying entirely within the C++ dispatcher to avoid CompositeImplicitAutograd
 * decompositions that would occur with Python re-entry.
 */
void autograd_fallback_kernel(const c10::OperatorHandle& op, torch::jit::Stack* stack);

/**
 * Set Python fallback callback for cache misses.
 * The callback should be _sky_kernel_fallback from dispatch.py.
 */
void set_python_fallback(py::object callback);

/**
 * Clear the Python fallback callback.
 */
void clear_python_fallback();

/**
 * Store a pending fused result from fallback_kernel for _sky_kernel_fallback
 * to pick up, avoiding a double dispatch_cached_aten call.
 */
void set_pending_fused_result(py::object result);

/**
 * Take the pending fused result (returns None if not set).
 * Clears the pending result after taking it.
 */
py::object take_pending_fused_result();

}  // namespace skytorch
