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
#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

namespace py = pybind11;

namespace skytorch {

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

}  // namespace skytorch
