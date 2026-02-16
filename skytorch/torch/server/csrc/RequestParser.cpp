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
 * - TensorStore (C++ unordered_map) replaces Python dict for GIL-free access
 * - Static C++ maps for dtype/memory_format/layout resolution
 * - Batch-level GIL release (~1.5ms continuous) instead of per-op release
 */

#include "RequestParser.h"

#include <torch/extension.h>
#include <torch/csrc/autograd/python_variable.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/ivalue.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/MemoryFormat.h>
#include <torch/csrc/Layout.h>
#include <cstring>
#include <cstdio>
#include <optional>
#include <vector>

namespace skytorch {
namespace server {

// --- TensorStore implementation ---

at::Tensor* TensorStore::find(uint64_t id) {
    auto it = tensors_.find(id);
    return (it != tensors_.end()) ? &it->second : nullptr;
}

at::Tensor& TensorStore::get(uint64_t id) {
    auto it = tensors_.find(id);
    if (it == tensors_.end()) {
        throw std::runtime_error("Tensor not found: " + std::to_string(id));
    }
    return it->second;
}

void TensorStore::set(uint64_t id, at::Tensor t) {
    tensors_[id] = std::move(t);
}

bool TensorStore::contains(uint64_t id) const {
    return tensors_.count(id) > 0;
}

bool TensorStore::erase(uint64_t id) {
    return tensors_.erase(id) > 0;
}

void TensorStore::clear() {
    tensors_.clear();
}

size_t TensorStore::size() const {
    return tensors_.size();
}

py::object TensorStore::get_python(uint64_t id) {
    auto it = tensors_.find(id);
    if (it == tensors_.end()) {
        throw std::runtime_error("Tensor not found: " + std::to_string(id));
    }
    // THPVariable_Wrap takes a copy, so we need to copy the tensor
    at::Tensor t = it->second;
    PyObject* py_t = THPVariable_Wrap(std::move(t));
    return py::reinterpret_steal<py::object>(py_t);
}

void TensorStore::set_python(uint64_t id, py::object t) {
    tensors_[id] = THPVariable_Unpack(t.ptr());
}

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

// --- Static C++ maps for GIL-free dtype/format/layout resolution ---

static const std::unordered_map<std::string, at::ScalarType> g_dtype_map = {
    {"torch.float32", at::kFloat},
    {"torch.float64", at::kDouble},
    {"torch.float16", at::kHalf},
    {"torch.bfloat16", at::kBFloat16},
    {"torch.int32", at::kInt},
    {"torch.int64", at::kLong},
    {"torch.int16", at::kShort},
    {"torch.int8", at::kChar},
    {"torch.uint8", at::kByte},
    {"torch.bool", at::kBool},
    {"torch.complex64", at::kComplexFloat},
    {"torch.complex128", at::kComplexDouble},
};

static const std::unordered_map<std::string, at::MemoryFormat> g_memory_format_map = {
    {"torch.contiguous_format", at::MemoryFormat::Contiguous},
    {"torch.channels_last", at::MemoryFormat::ChannelsLast},
    {"torch.channels_last_3d", at::MemoryFormat::ChannelsLast3d},
    {"torch.preserve_format", at::MemoryFormat::Preserve},
};

static const std::unordered_map<std::string, at::Layout> g_layout_map = {
    {"torch.strided", at::Layout::Strided},
    {"torch.sparse_coo", at::Layout::Sparse},
};

/**
 * Resolve dtype string to at::ScalarType using static map (GIL-free).
 * Returns nullopt if not found in the static map.
 */
static std::optional<at::ScalarType> resolve_dtype_scalar(const char* s, size_t len) {
    std::string key(s, len);
    auto it = g_dtype_map.find(key);
    if (it != g_dtype_map.end()) return it->second;
    return std::nullopt;
}

// --- Static caches (PyObject* pattern from Module.cpp:31-48 for safe shutdown) ---

// Maps op names ("aten.add.Tensor") to callable PyObject*
static std::unordered_map<std::string, PyObject*> g_op_cache;

// Maps op names to c10::OperatorHandle for GIL-free callBoxed dispatch
static std::unordered_map<std::string, c10::OperatorHandle> g_op_handle_cache;

// Per-op callBoxed blocklist: ops that have failed callBoxed and must always use Python.
// Once an op fails (exception in callBoxed), it's added here permanently.
// Coercion failures are NOT cached since they depend on per-call argument types.
static std::unordered_map<std::string, bool> g_callboxed_blocked;


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
 * GIL must be held.
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
 * Resolve an op name to a c10::OperatorHandle for GIL-free callBoxed dispatch.
 *
 * Converts "aten.add.Tensor" → OperatorName("aten::add", "Tensor"),
 * then looks up via Dispatcher::findSchemaOrThrow.
 * Cached for fast repeated lookups.
 */
static const c10::OperatorHandle& resolve_op_handle(const std::string& op_name) {
    auto it = g_op_handle_cache.find(op_name);
    if (it != g_op_handle_cache.end()) return it->second;

    // "aten.add.Tensor" → ns_name="aten::add", overload="Tensor"
    // "aten.sum.default" → ns_name="aten::sum", overload="" (default = no overload)
    auto dot1 = op_name.find('.');
    auto dot2 = op_name.find('.', dot1 + 1);
    std::string ns_name = op_name.substr(0, dot1) + "::" +
                          op_name.substr(dot1 + 1, dot2 - dot1 - 1);
    std::string overload_str = (dot2 != std::string::npos) ? op_name.substr(dot2 + 1) : "";
    // "default" in Python naming convention means no overload in C++
    if (overload_str == "default") overload_str.clear();

    auto handle = c10::Dispatcher::singleton().findSchemaOrThrow(
        ns_name.c_str(), overload_str.c_str());
    auto [ins_it, _] = g_op_handle_cache.emplace(op_name, handle);
    return ins_it->second;
}

/**
 * Resolve a "torch.X" attribute string to the Python object.
 * Replaces repeated getattr(torch, s[6:]) calls in _parse_raw_arg (service.py:693-708).
 *
 * Handles dtype, memory_format, and layout strings.
 * GIL must be held.
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
 * GIL must be held.
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
 * Parse tensor metadata and create tensor using ATen C++ API (GIL-free).
 *
 * Uses at::empty + .set_() with proper device options. No Python API calls.
 * Called ~886 times out of 58K ops (1.5%).
 * Registers the created tensor in TensorStore.
 */
static void parse_and_create_tensor_gilfree(
    const char* buf, size_t& pos, TensorStore& store)
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

    // Resolve dtype via static map (GIL-free)
    auto scalar_type_opt = resolve_dtype_scalar(dtype_str.c_str(), dtype_str.size());
    if (!scalar_type_opt.has_value()) {
        throw std::runtime_error("Unknown dtype: " + dtype_str);
    }
    auto scalar_type = *scalar_type_opt;

    // Build device — c10::Device(string) parses "cpu", "cuda:0", etc.
    std::string device_str = device_type;
    if (device_index >= 0) {
        device_str += ":" + std::to_string(device_index);
    }
    c10::Device device(device_str);
    auto options = at::TensorOptions().dtype(scalar_type).device(device);

    at::Tensor tensor;
    if (has_ref) {
        // Create view from existing tensor's storage
        at::Tensor& base = store.get(tensor_ref);
        auto storage = base.storage();
        tensor = at::empty({0}, options).set_(storage, storage_offset, shape, stride);
    } else {
        // Create new tensor with fresh storage
        // Compute storage_numel from nbytes and element size
        int64_t elem_size = c10::elementSize(scalar_type);
        int64_t storage_numel = (nbytes + elem_size - 1) / elem_size;
        auto storage_tensor = at::empty({storage_numel}, options);
        auto storage = storage_tensor.storage();
        tensor = at::empty({0}, options).set_(storage, storage_offset, shape, stride);
    }

    store.set(tensor_id, std::move(tensor));
}

/**
 * Parse tensor metadata and create tensor via Python API (GIL fallback).
 *
 * Used only when GIL is held and we need Python API for unsupported dtypes.
 * Registers the created tensor in TensorStore.
 */
static void parse_and_create_tensor_python(
    const char* buf, size_t& pos, TensorStore& store)
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
    py::object tensor_obj;

    if (has_ref) {
        // Create view from existing tensor's storage
        at::Tensor& base_at = store.get(tensor_ref);
        PyObject* base_pyobj = THPVariable_Wrap(base_at);
        py::object base = py::reinterpret_steal<py::object>(base_pyobj);

        py::object storage = base.attr("untyped_storage")();
        py::object empty = torch_mod.attr("empty")(
            0, py::arg("dtype") = dtype, py::arg("device") = base.attr("device")
        );
        tensor_obj = empty.attr("set_")(
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
        tensor_obj = empty.attr("set_")(
            storage, storage_offset, py::cast(shape), py::cast(stride)
        );
    }

    // Register in TensorStore
    store.set(tensor_id, THPVariable_Unpack(tensor_obj.ptr()));
}

// --- Argument parsers ---

/**
 * Parse a single argument from binary format into py::object.
 * Used for kwargs/fallback paths that require GIL.
 * Reads tensors from TensorStore, wraps via THPVariable_Wrap.
 * GIL must be held.
 */
static py::object parse_arg_from_store(const char* buf, size_t& pos, TensorStore& store) {
    uint8_t arg_type = read_uint8(buf, pos);

    switch (arg_type) {
    case 0x01: {  // TENSOR_ID (most common — checked first)
        uint64_t tid = read_uint64(buf, pos);
        at::Tensor t = store.get(tid);
        PyObject* py_t = THPVariable_Wrap(std::move(t));
        return py::reinterpret_steal<py::object>(py_t);
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
            result[i] = parse_arg_from_store(buf, pos, store);
        }
        return result;
    }
    case 0x0A: {  // TUPLE
        uint16_t count = read_uint16(buf, pos);
        py::tuple result(count);
        for (uint16_t i = 0; i < count; i++) {
            result[i] = parse_arg_from_store(buf, pos, store);
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

/**
 * Parse a single argument from binary format into c10::IValue (GIL-free).
 *
 * Uses TensorStore for tensor lookups and static maps for dtype/memory_format/layout.
 * No Python API calls needed on the hot path.
 */
static c10::IValue parse_arg_to_ivalue_gilfree(
    const char* buf, size_t& pos, TensorStore& store)
{
    uint8_t arg_type = read_uint8(buf, pos);

    switch (arg_type) {
    case 0x01: {  // TENSOR_ID → at::Tensor (GIL-free via TensorStore)
        uint64_t tid = read_uint64(buf, pos);
        return c10::IValue(store.get(tid));
    }
    case 0x00:  // NONE
        return c10::IValue();
    case 0x02:  // INT64
        return c10::IValue(read_int64(buf, pos));
    case 0x03:  // FLOAT64
        return c10::IValue(read_float64(buf, pos));
    case 0x04: {  // BOOL
        bool val = read_uint8(buf, pos) != 0;
        return c10::IValue(val);
    }
    case 0x05: {  // DTYPE → at::ScalarType (GIL-free via static map)
        uint8_t slen = read_uint8(buf, pos);
        std::string key(buf + pos, slen);
        pos += slen;
        auto it = g_dtype_map.find(key);
        if (it == g_dtype_map.end()) {
            throw std::runtime_error("Unknown dtype in IValue path: " + key);
        }
        return c10::IValue(static_cast<int64_t>(it->second));
    }
    case 0x06: {  // MEMORY_FORMAT → at::MemoryFormat (GIL-free via static map)
        uint8_t slen = read_uint8(buf, pos);
        std::string key(buf + pos, slen);
        pos += slen;
        auto it = g_memory_format_map.find(key);
        if (it == g_memory_format_map.end()) {
            throw std::runtime_error("Unknown memory_format in IValue path: " + key);
        }
        return c10::IValue(static_cast<int64_t>(it->second));
    }
    case 0x07: {  // LAYOUT → at::Layout (GIL-free via static map)
        uint8_t slen = read_uint8(buf, pos);
        std::string key(buf + pos, slen);
        pos += slen;
        auto it = g_layout_map.find(key);
        if (it == g_layout_map.end()) {
            throw std::runtime_error("Unknown layout in IValue path: " + key);
        }
        return c10::IValue(static_cast<int64_t>(it->second));
    }
    case 0x08: {  // STRING → std::string
        uint16_t slen = read_uint16(buf, pos);
        std::string s(buf + pos, slen);
        pos += slen;
        return c10::IValue(std::move(s));
    }
    case 0x09: {  // LIST — type-specialized based on element content
        uint16_t count = read_uint16(buf, pos);
        if (count == 0) {
            // Empty list — use GenericList, callBoxed can handle it
            c10::impl::GenericList list(c10::AnyType::get());
            return c10::IValue(std::move(list));
        }
        // Parse all elements first
        std::vector<c10::IValue> elems;
        elems.reserve(count);
        for (uint16_t i = 0; i < count; i++) {
            elems.push_back(parse_arg_to_ivalue_gilfree(buf, pos, store));
        }
        // Specialize based on first element's type
        if (elems[0].isInt()) {
            std::vector<int64_t> int_list;
            int_list.reserve(count);
            for (auto& e : elems) int_list.push_back(e.toInt());
            return c10::IValue(std::move(int_list));
        } else if (elems[0].isDouble()) {
            std::vector<double> double_list;
            double_list.reserve(count);
            for (auto& e : elems) double_list.push_back(e.toDouble());
            return c10::IValue(std::move(double_list));
        } else if (elems[0].isTensor()) {
            std::vector<at::Tensor> tensor_list;
            tensor_list.reserve(count);
            for (auto& e : elems) tensor_list.push_back(e.toTensor());
            return c10::IValue(std::move(tensor_list));
        } else if (elems[0].isBool()) {
            std::vector<bool> bool_list;
            bool_list.reserve(count);
            for (auto& e : elems) bool_list.push_back(e.toBool());
            return c10::IValue(std::move(bool_list));
        } else {
            // Fallback: GenericList for mixed or unknown types
            c10::impl::GenericList list(c10::AnyType::get());
            list.reserve(count);
            for (auto& e : elems) list.push_back(std::move(e));
            return c10::IValue(std::move(list));
        }
    }
    case 0x0A: {  // TUPLE → c10::ivalue::Tuple
        uint16_t count = read_uint16(buf, pos);
        std::vector<c10::IValue> elems;
        elems.reserve(count);
        for (uint16_t i = 0; i < count; i++) {
            elems.push_back(parse_arg_to_ivalue_gilfree(buf, pos, store));
        }
        return c10::IValue(c10::ivalue::Tuple::create(std::move(elems)));
    }
    default: {
        char msg[64];
        snprintf(msg, sizeof(msg), "Unknown arg type: 0x%02x", arg_type);
        throw std::runtime_error(msg);
    }
    }
}

/**
 * Coerce an IValue to match the expected schema type.
 *
 * Handles list type specialization (GenericList → IntList, DoubleList, etc.)
 * and scalar-to-Tensor coercion. Returns true if coercion was possible.
 */
static bool coerce_ivalue(c10::IValue& val, const c10::TypePtr& expected_type) {
    auto kind = expected_type->kind();

    // Schema expects Tensor but we have a scalar → fall back to Python path
    // (Python bindings handle dtype matching automatically; scalar_to_tensor
    // may produce wrong dtype e.g., Double instead of Float)
    if (kind == c10::TypeKind::TensorType && !val.isTensor() && !val.isNone()) {
        return false;
    }

    // GenericList → specialized list coercion
    if (val.isList() && !val.isIntList() && !val.isDoubleList() &&
        !val.isTensorList() && !val.isBoolList()) {
        auto generic_list = val.toList();
        if (kind == c10::TypeKind::ListType) {
            auto elem_kind = expected_type->containedType(0)->kind();
            if (elem_kind == c10::TypeKind::IntType) {
                std::vector<int64_t> typed;
                typed.reserve(generic_list.size());
                for (size_t i = 0; i < generic_list.size(); i++) {
                    typed.push_back(generic_list.get(i).toInt());
                }
                val = c10::IValue(std::move(typed));
                return true;
            } else if (elem_kind == c10::TypeKind::FloatType) {
                std::vector<double> typed;
                typed.reserve(generic_list.size());
                for (size_t i = 0; i < generic_list.size(); i++) {
                    typed.push_back(generic_list.get(i).toDouble());
                }
                val = c10::IValue(std::move(typed));
                return true;
            } else if (elem_kind == c10::TypeKind::TensorType) {
                std::vector<at::Tensor> typed;
                typed.reserve(generic_list.size());
                for (size_t i = 0; i < generic_list.size(); i++) {
                    typed.push_back(generic_list.get(i).toTensor());
                }
                val = c10::IValue(std::move(typed));
                return true;
            } else if (elem_kind == c10::TypeKind::OptionalType) {
                // e.g., Tensor?[] for aten::index
                auto inner_kind = expected_type->containedType(0)->containedType(0)->kind();
                if (inner_kind == c10::TypeKind::TensorType) {
                    c10::List<c10::optional<at::Tensor>> typed;
                    typed.reserve(generic_list.size());
                    for (size_t i = 0; i < generic_list.size(); i++) {
                        const auto& elem = generic_list.get(i);
                        if (elem.isNone()) {
                            typed.push_back(c10::nullopt);
                        } else {
                            typed.push_back(elem.toTensor());
                        }
                    }
                    val = c10::IValue(std::move(typed));
                    return true;
                }
            } else if (elem_kind == c10::TypeKind::BoolType) {
                std::vector<bool> typed;
                typed.reserve(generic_list.size());
                for (size_t i = 0; i < generic_list.size(); i++) {
                    typed.push_back(generic_list.get(i).toBool());
                }
                val = c10::IValue(std::move(typed));
                return true;
            }
        }
    }

    // Optional type — unwrap and try coercing the inner type
    if (kind == c10::TypeKind::OptionalType && !val.isNone()) {
        auto inner_type = expected_type->containedType(0);
        return coerce_ivalue(val, inner_type);
    }

    return true;  // No coercion needed
}

// --- Core execution ---

/**
 * Parse and execute one op from a buffer at a given position.
 *
 * When gil_released=true (batch path), the entire function runs without GIL.
 * GIL is temporarily re-acquired only for kwargs/fallback paths.
 *
 * When gil_released=false (single-op path), behaves like the original:
 * GIL is held throughout, released only during callBoxed.
 */
static void execute_one_op(
    const char* buf, size_t& pos, TensorStore& store, bool gil_released)
{
    // Header (4 bytes)
    uint8_t num_args = read_uint8(buf, pos);
    uint8_t num_kwargs = read_uint8(buf, pos);
    uint8_t num_outputs = read_uint8(buf, pos);
    uint8_t num_metadata = read_uint8(buf, pos);

    // Op name (uint16 len + bytes)
    uint16_t op_name_len = read_uint16(buf, pos);
    std::string op_name(buf + pos, op_name_len);
    pos += op_name_len;

    // Parse metadata and auto-create tensors (GIL-free via TensorStore)
    for (uint8_t i = 0; i < num_metadata; i++) {
        // Peek at tensor_id to check if already registered
        uint64_t tensor_id;
        std::memcpy(&tensor_id, buf + pos, 8);

        if (store.contains(tensor_id)) {
            skip_tensor_metadata(buf, pos);
        } else {
            parse_and_create_tensor_gilfree(buf, pos, store);
        }
    }

    // Parse output tensor IDs
    std::vector<uint64_t> output_tensor_ids;
    output_tensor_ids.reserve(num_outputs);
    for (uint8_t i = 0; i < num_outputs; i++) {
        output_tensor_ids.push_back(read_uint64(buf, pos));
    }

    // ── FALLBACK: kwargs present → must use Python path (requires GIL) ──
    if (num_kwargs > 0) {
        // Acquire GIL if we don't have it (batch path)
        auto acquire = gil_released
            ? std::make_optional<py::gil_scoped_acquire>()
            : std::nullopt;

        py::tuple args(num_args);
        for (uint8_t i = 0; i < num_args; i++) {
            args[i] = parse_arg_from_store(buf, pos, store);
        }

        py::dict kwargs;
        for (uint8_t i = 0; i < num_kwargs; i++) {
            uint8_t name_len = read_uint8(buf, pos);
            py::str name = py::reinterpret_steal<py::str>(
                PyUnicode_DecodeUTF8(buf + pos, name_len, nullptr)
            );
            if (!name.ptr()) throw py::error_already_set();
            pos += name_len;
            kwargs[name] = parse_arg_from_store(buf, pos, store);
        }

        PyObject* op = get_aten_op(op_name);
        py::object result = py::reinterpret_steal<py::object>(
            PyObject_Call(op, args.ptr(), kwargs.ptr())
        );
        if (!result.ptr()) throw py::error_already_set();

        // Register output tensors via TensorStore
        if (!output_tensor_ids.empty()) {
            std::vector<at::Tensor> result_tensors;

            if (THPVariable_Check(result.ptr())) {
                result_tensors.push_back(THPVariable_Unpack(result.ptr()));
            } else if (PyTuple_Check(result.ptr())) {
                Py_ssize_t n = PyTuple_GET_SIZE(result.ptr());
                for (Py_ssize_t i = 0; i < n; i++) {
                    PyObject* item = PyTuple_GET_ITEM(result.ptr(), i);
                    if (THPVariable_Check(item)) {
                        result_tensors.push_back(THPVariable_Unpack(item));
                    }
                }
            } else if (PyList_Check(result.ptr())) {
                Py_ssize_t n = PyList_GET_SIZE(result.ptr());
                for (Py_ssize_t i = 0; i < n; i++) {
                    PyObject* item = PyList_GET_ITEM(result.ptr(), i);
                    if (THPVariable_Check(item)) {
                        result_tensors.push_back(THPVariable_Unpack(item));
                    }
                }
            }

            size_t count = std::min(output_tensor_ids.size(), result_tensors.size());
            for (size_t i = 0; i < count; i++) {
                store.set(output_tensor_ids[i], std::move(result_tensors[i]));
            }
        }
        return;
    }

    // ── Dispatch decision: callBoxed or PyObject_Call? ──
    // Skip callBoxed for ops that previously threw exceptions (permanent blocklist).
    // Coercion failures are checked per-call since they depend on argument types.
    bool blocked = g_callboxed_blocked.count(op_name) > 0;

    if (!blocked) {
        // ── callBoxed path: parse to IValues, dispatch (GIL-free) ──
        size_t saved_pos = pos;

        const auto& op_handle = resolve_op_handle(op_name);
        const auto& schema = op_handle.schema();
        const auto& schema_args = schema.arguments();

        std::vector<c10::IValue> stack;
        stack.reserve(schema_args.size());

        bool viable = true;
        for (uint8_t i = 0; i < num_args; i++) {
            stack.push_back(parse_arg_to_ivalue_gilfree(buf, pos, store));
        }

        // Fill in default values for remaining schema arguments
        for (size_t i = num_args; i < schema_args.size(); i++) {
            const auto& default_val = schema_args[i].default_value();
            if (default_val.has_value()) {
                stack.push_back(*default_val);
            } else {
                stack.push_back(c10::IValue());
            }
        }

        // Coerce IValue types to match schema (per-call, not cached)
        for (size_t i = 0; i < stack.size() && i < schema_args.size(); i++) {
            if (!coerce_ivalue(stack[i], schema_args[i].type())) {
                viable = false;
                break;
            }
        }

        if (viable) {
            try {
                // callBoxed is already GIL-free. In batch path (gil_released=true),
                // we're already without GIL. In single-op path (gil_released=false),
                // we release it here.
                if (gil_released) {
                    op_handle.callBoxed(&stack);
                } else {
                    py::gil_scoped_release release;
                    op_handle.callBoxed(&stack);
                }
            } catch (...) {
                // Permanent failure — blocklist this op
                g_callboxed_blocked[op_name] = true;
                viable = false;
            }
        }

        if (viable) {
            // Register outputs directly in TensorStore (GIL-free)
            if (!output_tensor_ids.empty()) {
                std::vector<at::Tensor> result_tensors;
                for (size_t si = 0; si < stack.size(); si++) {
                    if (stack[si].isTensor()) {
                        at::Tensor t = stack[si].toTensor();
                        if (t.defined()) {
                            result_tensors.push_back(std::move(t));
                        }
                    }
                }

                size_t count = std::min(output_tensor_ids.size(), result_tensors.size());
                for (size_t i = 0; i < count; i++) {
                    store.set(output_tensor_ids[i], std::move(result_tensors[i]));
                }
            }
            return;
        }

        // Coercion failed — rewind and fall through to Python path
        pos = saved_pos;
    }

    // ── Python path: parse to py::objects, dispatch via PyObject_Call ──
    {
        // Acquire GIL if we don't have it (batch path)
        auto acquire = gil_released
            ? std::make_optional<py::gil_scoped_acquire>()
            : std::nullopt;

        py::tuple args(num_args);
        for (uint8_t i = 0; i < num_args; i++) {
            args[i] = parse_arg_from_store(buf, pos, store);
        }

        PyObject* op = get_aten_op(op_name);
        py::object result = py::reinterpret_steal<py::object>(
            PyObject_Call(op, args.ptr(), nullptr)
        );
        if (!result.ptr()) throw py::error_already_set();

        if (!output_tensor_ids.empty()) {
            std::vector<at::Tensor> result_tensors;
            if (THPVariable_Check(result.ptr())) {
                result_tensors.push_back(THPVariable_Unpack(result.ptr()));
            } else if (PyTuple_Check(result.ptr())) {
                Py_ssize_t n = PyTuple_GET_SIZE(result.ptr());
                for (Py_ssize_t i = 0; i < n; i++) {
                    PyObject* item = PyTuple_GET_ITEM(result.ptr(), i);
                    if (THPVariable_Check(item)) {
                        result_tensors.push_back(THPVariable_Unpack(item));
                    }
                }
            } else if (PyList_Check(result.ptr())) {
                Py_ssize_t n = PyList_GET_SIZE(result.ptr());
                for (Py_ssize_t i = 0; i < n; i++) {
                    PyObject* item = PyList_GET_ITEM(result.ptr(), i);
                    if (THPVariable_Check(item)) {
                        result_tensors.push_back(THPVariable_Unpack(item));
                    }
                }
            }

            size_t count = std::min(output_tensor_ids.size(), result_tensors.size());
            for (size_t i = 0; i < count; i++) {
                store.set(output_tensor_ids[i], std::move(result_tensors[i]));
            }
        }
    }
}

// --- Exported functions ---

void execute_raw_aten_inline(py::bytes data, TensorStore& store) {
    char* buf;
    Py_ssize_t len;
    if (PyBytes_AsStringAndSize(data.ptr(), &buf, &len) < 0) {
        throw py::error_already_set();
    }

    size_t pos = 0;
    execute_one_op(buf, pos, store, /*gil_released=*/false);
}

void execute_raw_batched_aten_inline(py::bytes data, TensorStore& store) {
    char* buf;
    Py_ssize_t len;
    if (PyBytes_AsStringAndSize(data.ptr(), &buf, &len) < 0) {
        throw py::error_already_set();
    }

    size_t total = static_cast<size_t>(len);

    {
        py::gil_scoped_release release;  // ~1.5ms continuous GIL-free
        size_t pos = 0;
        while (pos < total) {
            uint32_t op_len = read_uint32(buf, pos);
            size_t op_end = pos + op_len;
            execute_one_op(buf, pos, store, /*gil_released=*/true);
            // Ensure we advance past exactly this op's data
            pos = op_end;
        }
    }
    // GIL re-acquired; py::bytes destructor safe
}

void clear_op_cache() {
    for (auto& pair : g_op_cache) {
        Py_XDECREF(pair.second);
    }
    g_op_cache.clear();

    // OperatorHandle doesn't hold Python refs — no Py_DECREF needed
    g_op_handle_cache.clear();

    g_callboxed_blocked.clear();

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
