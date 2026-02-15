/**
 * SkyTorch Server - Binary Request Parser
 *
 * C++ implementation of the server-side binary ATen request parser.
 * Replaces the Python _execute_raw_aten_inline (service.py:830-857) and
 * the batched variant with direct memory reads, eliminating struct.unpack,
 * bytes.decode(), and Python function call overhead.
 *
 * Key optimizations:
 * - memcpy-based reads (zero struct.unpack overhead)
 * - unordered_map caches for op lookups and torch attr resolution
 * - PyDict_GetItem (borrowed ref) for tensor lookups
 * - Direct PyObject_Call for op execution
 */

#include "RequestParser.h"

#include <torch/extension.h>
#include <torch/csrc/autograd/python_variable.h>
#include <cstring>
#include <cstdio>
#include <vector>

namespace skytorch {
namespace server {

// --- Read helpers (inverse of RequestBuilder.cpp:75-119 write helpers) ---

static inline uint8_t read_uint8(const char* buf, size_t& pos) {
    return static_cast<uint8_t>(buf[pos++]);
}

static inline uint16_t read_uint16(const char* buf, size_t& pos) {
    uint16_t val;
    std::memcpy(&val, buf + pos, 2);
    pos += 2;
    return val;
}

static inline uint32_t read_uint32(const char* buf, size_t& pos) {
    uint32_t val;
    std::memcpy(&val, buf + pos, 4);
    pos += 4;
    return val;
}

static inline int32_t read_int32(const char* buf, size_t& pos) {
    int32_t val;
    std::memcpy(&val, buf + pos, 4);
    pos += 4;
    return val;
}

static inline uint64_t read_uint64(const char* buf, size_t& pos) {
    uint64_t val;
    std::memcpy(&val, buf + pos, 8);
    pos += 8;
    return val;
}

static inline int64_t read_int64(const char* buf, size_t& pos) {
    int64_t val;
    std::memcpy(&val, buf + pos, 8);
    pos += 8;
    return val;
}

static inline double read_float64(const char* buf, size_t& pos) {
    double val;
    std::memcpy(&val, buf + pos, 8);
    pos += 8;
    return val;
}

// --- Static caches (PyObject* pattern from Module.cpp:31-48 for safe shutdown) ---

// Maps op names ("aten.add.Tensor") to callable PyObject*
static std::unordered_map<std::string, PyObject*> g_op_cache;

// Maps torch attribute strings ("torch.float32") to resolved Python objects
static std::unordered_map<std::string, PyObject*> g_torch_attr_cache;

// Cached reference to parse_dtype function from serialization module
static PyObject* g_parse_dtype = nullptr;

// --- Cache helpers ---

/**
 * Get ATen op by name with caching.
 * Replaces _get_aten_op (service.py:598-612).
 *
 * Splits op_name by '.', walks torch.ops, caches result.
 */
static PyObject* get_aten_op(const std::string& op_name) {
    auto it = g_op_cache.find(op_name);
    if (it != g_op_cache.end()) {
        return it->second;
    }

    // Split by '.', walk torch.ops
    py::object obj = py::module::import("torch").attr("ops");
    size_t start = 0;
    while (start < op_name.size()) {
        size_t dot = op_name.find('.', start);
        if (dot == std::string::npos) dot = op_name.size();
        std::string part = op_name.substr(start, dot - start);
        obj = obj.attr(part.c_str());
        start = dot + 1;
    }

    PyObject* raw = obj.ptr();
    Py_INCREF(raw);
    g_op_cache[op_name] = raw;
    return raw;
}

/**
 * Resolve a "torch.X" attribute string to the Python object.
 * Replaces repeated getattr(torch, s[6:]) calls in _parse_raw_arg (service.py:693-708).
 *
 * Handles dtype, memory_format, and layout strings.
 */
static PyObject* resolve_torch_attr(const char* s, size_t len) {
    std::string key(s, len);
    auto it = g_torch_attr_cache.find(key);
    if (it != g_torch_attr_cache.end()) {
        return it->second;
    }

    // "torch.float32" → getattr(torch, "float32")
    if (len <= 6) {
        throw std::runtime_error("Invalid torch attribute string: " + key);
    }
    std::string attr_name(s + 6, len - 6);

    py::module torch_mod = py::module::import("torch");
    py::object attr = torch_mod.attr(attr_name.c_str());

    PyObject* raw = attr.ptr();
    Py_INCREF(raw);
    g_torch_attr_cache[key] = raw;
    return raw;
}

/**
 * Get cached reference to parse_dtype function.
 */
static PyObject* get_parse_dtype() {
    if (g_parse_dtype != nullptr) {
        return g_parse_dtype;
    }

    py::module serialization = py::module::import("skytorch.torch.server.serialization");
    py::object fn = serialization.attr("parse_dtype");

    g_parse_dtype = fn.ptr();
    Py_INCREF(g_parse_dtype);
    return g_parse_dtype;
}

// --- Tensor metadata helpers ---

/**
 * Skip past tensor metadata without parsing.
 * Replaces _skip_raw_tensor_metadata (service.py:732-748).
 * Pure pointer arithmetic — no allocations.
 */
static void skip_tensor_metadata(const char* buf, size_t& pos) {
    pos += 8;  // tensor_id (uint64)
    uint8_t ndim = static_cast<uint8_t>(buf[pos]);
    pos += 1;  // ndim (uint8)
    pos += ndim * 16;  // shape + stride (ndim * 2 * int64)
    uint8_t dtype_len = static_cast<uint8_t>(buf[pos]);
    pos += 1 + dtype_len;  // dtype_str_len + dtype_str
    pos += 16;  // storage_offset + nbytes (2 * int64)
    uint8_t dt_len = static_cast<uint8_t>(buf[pos]);
    pos += 1 + dt_len;  // device_type_len + device_type
    pos += 4;  // device_index (int32)
    uint8_t has_ref = static_cast<uint8_t>(buf[pos]);
    pos += 1;  // has_tensor_ref (uint8)
    if (has_ref) pos += 8;  // tensor_ref (uint64)
}

/**
 * Parse tensor metadata and create tensor via Python API.
 * Replaces _parse_and_create_tensor (service.py:750-828).
 *
 * Only called ~886 times out of 58K ops, so Python API overhead is negligible.
 * Registers the created tensor in tensor_dict.
 */
static void parse_and_create_tensor(
    const char* buf, size_t& pos, PyObject* tensor_dict)
{
    uint64_t tensor_id = read_uint64(buf, pos);
    uint8_t ndim = read_uint8(buf, pos);

    // Read shape and stride
    std::vector<int64_t> shape(ndim);
    for (uint8_t i = 0; i < ndim; i++) {
        shape[i] = read_int64(buf, pos);
    }
    std::vector<int64_t> stride(ndim);
    for (uint8_t i = 0; i < ndim; i++) {
        stride[i] = read_int64(buf, pos);
    }

    // dtype string (uint8 len + bytes)
    uint8_t dtype_len = read_uint8(buf, pos);
    std::string dtype_str(buf + pos, dtype_len);
    pos += dtype_len;

    // storage_offset, nbytes
    int64_t storage_offset = read_int64(buf, pos);
    int64_t nbytes = read_int64(buf, pos);

    // device_type string (uint8 len + bytes)
    uint8_t dt_len = read_uint8(buf, pos);
    std::string device_type(buf + pos, dt_len);
    pos += dt_len;

    // device_index
    int32_t device_index = read_int32(buf, pos);

    // tensor_ref (optional)
    uint8_t has_tensor_ref = read_uint8(buf, pos);
    uint64_t tensor_ref = 0;
    bool has_ref = false;
    if (has_tensor_ref) {
        tensor_ref = read_uint64(buf, pos);
        has_ref = true;
    }

    // Resolve dtype via parse_dtype (uses serialization._DTYPE_MAP)
    PyObject* parse_dtype_fn = get_parse_dtype();
    py::object py_dtype_str = py::str(dtype_str);
    py::object dtype = py::reinterpret_steal<py::object>(
        PyObject_CallOneArg(parse_dtype_fn, py_dtype_str.ptr())
    );
    if (!dtype.ptr()) throw py::error_already_set();

    py::module torch_mod = py::module::import("torch");
    py::object tensor;

    if (has_ref) {
        // Create view from existing tensor's storage
        PyObject* ref_key = PyLong_FromUnsignedLongLong(tensor_ref);
        PyObject* base_obj = PyDict_GetItem(tensor_dict, ref_key);  // borrowed ref
        Py_DECREF(ref_key);
        if (!base_obj) {
            throw std::runtime_error(
                "Base tensor not found for view: " + std::to_string(tensor_ref));
        }

        py::object base = py::reinterpret_borrow<py::object>(base_obj);
        py::object storage = base.attr("untyped_storage")();
        py::object empty = torch_mod.attr("empty")(
            0, py::arg("dtype") = dtype, py::arg("device") = base.attr("device")
        );
        tensor = empty.attr("set_")(
            storage, storage_offset, py::cast(shape), py::cast(stride)
        );
    } else {
        // Create new tensor with fresh storage
        py::object device = torch_mod.attr("device")(device_type, device_index);
        py::object storage = torch_mod.attr("UntypedStorage")(
            nbytes, py::arg("device") = device
        );
        py::object empty = torch_mod.attr("empty")(
            0, py::arg("dtype") = dtype, py::arg("device") = device
        );
        tensor = empty.attr("set_")(
            storage, storage_offset, py::cast(shape), py::cast(stride)
        );
    }

    // Register in tensor_dict
    PyObject* key = PyLong_FromUnsignedLongLong(tensor_id);
    PyDict_SetItem(tensor_dict, key, tensor.ptr());
    Py_DECREF(key);
}

// --- Argument parser ---

/**
 * Parse a single argument from binary format.
 * Replaces _parse_raw_arg (service.py:675-730).
 *
 * Key optimization: TENSOR_ID (0x01) is checked first as the most common type.
 * Uses PyDict_GetItem (borrowed ref) to avoid ref counting overhead.
 */
static py::object parse_arg(const char* buf, size_t& pos, PyObject* tensor_dict) {
    uint8_t arg_type = read_uint8(buf, pos);

    switch (arg_type) {
    case 0x01: {  // TENSOR_ID (most common — checked first)
        uint64_t tid = read_uint64(buf, pos);
        PyObject* key = PyLong_FromUnsignedLongLong(tid);
        PyObject* tensor = PyDict_GetItem(tensor_dict, key);  // borrowed ref
        Py_DECREF(key);
        if (!tensor) {
            throw std::runtime_error("Tensor not found: " + std::to_string(tid));
        }
        return py::reinterpret_borrow<py::object>(tensor);
    }
    case 0x00:  // NONE
        return py::none();
    case 0x02:  // INT64
        return py::int_(read_int64(buf, pos));
    case 0x03:  // FLOAT64
        return py::float_(read_float64(buf, pos));
    case 0x04: {  // BOOL
        bool val = read_uint8(buf, pos) != 0;
        return py::bool_(val);
    }
    case 0x05:  // DTYPE
    case 0x06:  // MEMORY_FORMAT
    case 0x07: {  // LAYOUT
        uint8_t slen = read_uint8(buf, pos);
        PyObject* attr = resolve_torch_attr(buf + pos, slen);
        pos += slen;
        return py::reinterpret_borrow<py::object>(attr);
    }
    case 0x08: {  // STRING
        uint16_t slen = read_uint16(buf, pos);
        py::object s = py::reinterpret_steal<py::object>(
            PyUnicode_DecodeUTF8(buf + pos, slen, nullptr)
        );
        if (!s.ptr()) throw py::error_already_set();
        pos += slen;
        return s;
    }
    case 0x09: {  // LIST
        uint16_t count = read_uint16(buf, pos);
        py::list result(count);
        for (uint16_t i = 0; i < count; i++) {
            result[i] = parse_arg(buf, pos, tensor_dict);
        }
        return result;
    }
    case 0x0A: {  // TUPLE
        uint16_t count = read_uint16(buf, pos);
        py::tuple result(count);
        for (uint16_t i = 0; i < count; i++) {
            result[i] = parse_arg(buf, pos, tensor_dict);
        }
        return result;
    }
    default: {
        char msg[64];
        snprintf(msg, sizeof(msg), "Unknown arg type: 0x%02x", arg_type);
        throw std::runtime_error(msg);
    }
    }
}

// --- Core execution ---

/**
 * Parse and execute one op from a buffer at a given position.
 * Both exported functions delegate to this.
 *
 * Parses header, op_name, metadata, outputs, args, kwargs,
 * then calls the ATen op and registers output tensors.
 */
static void execute_one_op(const char* buf, size_t& pos, PyObject* tensor_dict) {
    // Header (4 bytes)
    uint8_t num_args = read_uint8(buf, pos);
    uint8_t num_kwargs = read_uint8(buf, pos);
    uint8_t num_outputs = read_uint8(buf, pos);
    uint8_t num_metadata = read_uint8(buf, pos);

    // Op name (uint16 len + bytes)
    uint16_t op_name_len = read_uint16(buf, pos);
    std::string op_name(buf + pos, op_name_len);
    pos += op_name_len;

    // Parse metadata and auto-create tensors
    for (uint8_t i = 0; i < num_metadata; i++) {
        // Peek at tensor_id to check if already registered
        uint64_t tensor_id;
        std::memcpy(&tensor_id, buf + pos, 8);

        PyObject* key = PyLong_FromUnsignedLongLong(tensor_id);
        PyObject* existing = PyDict_GetItem(tensor_dict, key);  // borrowed ref
        Py_DECREF(key);

        if (existing) {
            skip_tensor_metadata(buf, pos);
        } else {
            parse_and_create_tensor(buf, pos, tensor_dict);
        }
    }

    // Parse output tensor IDs
    std::vector<uint64_t> output_tensor_ids;
    output_tensor_ids.reserve(num_outputs);
    for (uint8_t i = 0; i < num_outputs; i++) {
        output_tensor_ids.push_back(read_uint64(buf, pos));
    }

    // Parse args
    py::tuple args(num_args);
    for (uint8_t i = 0; i < num_args; i++) {
        args[i] = parse_arg(buf, pos, tensor_dict);
    }

    // Parse kwargs
    py::dict kwargs;
    for (uint8_t i = 0; i < num_kwargs; i++) {
        uint8_t name_len = read_uint8(buf, pos);
        py::str name = py::reinterpret_steal<py::str>(
            PyUnicode_DecodeUTF8(buf + pos, name_len, nullptr)
        );
        if (!name.ptr()) throw py::error_already_set();
        pos += name_len;
        kwargs[name] = parse_arg(buf, pos, tensor_dict);
    }

    // Get and execute the ATen op
    PyObject* op = get_aten_op(op_name);
    py::object result = py::reinterpret_steal<py::object>(
        PyObject_Call(op, args.ptr(), num_kwargs > 0 ? kwargs.ptr() : nullptr)
    );
    if (!result.ptr()) throw py::error_already_set();

    // Register output tensors (matches service.py:847-857)
    if (!output_tensor_ids.empty()) {
        std::vector<PyObject*> result_tensors;

        if (THPVariable_Check(result.ptr())) {
            result_tensors.push_back(result.ptr());
        } else if (PyTuple_Check(result.ptr())) {
            Py_ssize_t n = PyTuple_GET_SIZE(result.ptr());
            for (Py_ssize_t i = 0; i < n; i++) {
                PyObject* item = PyTuple_GET_ITEM(result.ptr(), i);
                if (THPVariable_Check(item)) {
                    result_tensors.push_back(item);
                }
            }
        } else if (PyList_Check(result.ptr())) {
            Py_ssize_t n = PyList_GET_SIZE(result.ptr());
            for (Py_ssize_t i = 0; i < n; i++) {
                PyObject* item = PyList_GET_ITEM(result.ptr(), i);
                if (THPVariable_Check(item)) {
                    result_tensors.push_back(item);
                }
            }
        }

        size_t count = std::min(output_tensor_ids.size(), result_tensors.size());
        for (size_t i = 0; i < count; i++) {
            if (result_tensors[i] != nullptr) {
                PyObject* key = PyLong_FromUnsignedLongLong(output_tensor_ids[i]);
                PyDict_SetItem(tensor_dict, key, result_tensors[i]);
                Py_DECREF(key);
            }
        }
    }
}

// --- Exported functions ---

void execute_raw_aten_inline(py::bytes data, py::dict tensor_dict) {
    char* buf;
    Py_ssize_t len;
    if (PyBytes_AsStringAndSize(data.ptr(), &buf, &len) < 0) {
        throw py::error_already_set();
    }

    size_t pos = 0;
    execute_one_op(buf, pos, tensor_dict.ptr());
}

void execute_raw_batched_aten_inline(py::bytes data, py::dict tensor_dict) {
    char* buf;
    Py_ssize_t len;
    if (PyBytes_AsStringAndSize(data.ptr(), &buf, &len) < 0) {
        throw py::error_already_set();
    }

    size_t pos = 0;
    size_t total = static_cast<size_t>(len);
    while (pos < total) {
        uint32_t op_len = read_uint32(buf, pos);
        size_t op_end = pos + op_len;
        execute_one_op(buf, pos, tensor_dict.ptr());
        // Ensure we advance past exactly this op's data
        pos = op_end;
    }
}

void clear_op_cache() {
    for (auto& pair : g_op_cache) {
        Py_XDECREF(pair.second);
    }
    g_op_cache.clear();

    for (auto& pair : g_torch_attr_cache) {
        Py_XDECREF(pair.second);
    }
    g_torch_attr_cache.clear();

    if (g_parse_dtype) {
        Py_DECREF(g_parse_dtype);
        g_parse_dtype = nullptr;
    }
}

}  // namespace server
}  // namespace skytorch
