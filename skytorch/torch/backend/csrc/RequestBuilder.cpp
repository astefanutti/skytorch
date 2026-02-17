/**
 * SkyTorch PyTorch Backend - Binary Request Builder
 *
 * C++ implementation of the ATen request serializer. Replaces the Python
 * hot path (build_execute_aten_request + to_aten_arg + _get_tensor_metadata_if_new)
 * with direct C++ binary serialization, eliminating Python protobuf overhead.
 */

#include "RequestBuilder.h"
#include "TensorImpl.h"
#include "StorageImpl.h"

#include <torch/extension.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/Layout.h>
#include <torch/csrc/MemoryFormat.h>
#include <torch/csrc/autograd/python_variable.h>
#include <atomic>
#include <cstring>

namespace skytorch {

// Set of tensor IDs already registered with the server.
// Checked in C++ to avoid Python dict lookups per tensor.
static std::unordered_set<uint64_t> g_registered_tensor_ids;
// Map from storage_id to the first tensor_id registered for that storage.
// Used to detect views (tensors sharing storage with a registered base tensor).
static std::unordered_map<int64_t, uint64_t> g_storage_to_tensor_id;

void register_tensor_id(uint64_t tensor_id) {
    g_registered_tensor_ids.insert(tensor_id);
}

void unregister_tensor_id(uint64_t tensor_id) {
    g_registered_tensor_ids.erase(tensor_id);
}

void clear_registered_tensor_ids() {
    g_registered_tensor_ids.clear();
    g_storage_to_tensor_id.clear();
}

void register_storage_tensor_mapping(int64_t storage_id, uint64_t tensor_id) {
    if (g_storage_to_tensor_id.find(storage_id) == g_storage_to_tensor_id.end()) {
        g_storage_to_tensor_id[storage_id] = tensor_id;
    }
}

// Register a tensor ID and its storage mapping
static void register_tensor_with_storage(uint64_t tensor_id, int64_t storage_id) {
    g_registered_tensor_ids.insert(tensor_id);
    if (g_storage_to_tensor_id.find(storage_id) == g_storage_to_tensor_id.end()) {
        g_storage_to_tensor_id[storage_id] = tensor_id;
    }
}

// Check if a tensor_id is registered, or find a registered tensor sharing the same storage
// Returns: (is_registered, tensor_ref_or_0)
// - is_registered=true, tensor_ref=0: tensor itself is registered
// - is_registered=false, tensor_ref>0: tensor is new but shares storage with tensor_ref
// - is_registered=false, tensor_ref=0: tensor is completely new
static std::pair<bool, uint64_t> check_registration(uint64_t tensor_id, int64_t storage_id) {
    if (g_registered_tensor_ids.count(tensor_id)) {
        return {true, 0};
    }
    auto it = g_storage_to_tensor_id.find(storage_id);
    if (it != g_storage_to_tensor_id.end()) {
        return {false, it->second};
    }
    return {false, 0};
}

// --- Binary serialization helpers ---

static inline void write_uint8(std::vector<uint8_t>& buf, uint8_t val) {
    buf.push_back(val);
}

static inline void write_uint16(std::vector<uint8_t>& buf, uint16_t val) {
    buf.push_back(static_cast<uint8_t>(val & 0xFF));
    buf.push_back(static_cast<uint8_t>((val >> 8) & 0xFF));
}

static inline void write_int32(std::vector<uint8_t>& buf, int32_t val) {
    uint32_t u;
    std::memcpy(&u, &val, 4);
    buf.push_back(static_cast<uint8_t>(u & 0xFF));
    buf.push_back(static_cast<uint8_t>((u >> 8) & 0xFF));
    buf.push_back(static_cast<uint8_t>((u >> 16) & 0xFF));
    buf.push_back(static_cast<uint8_t>((u >> 24) & 0xFF));
}

static inline void write_uint64(std::vector<uint8_t>& buf, uint64_t val) {
    for (int i = 0; i < 8; i++) {
        buf.push_back(static_cast<uint8_t>((val >> (i * 8)) & 0xFF));
    }
}

static inline void write_int64(std::vector<uint8_t>& buf, int64_t val) {
    write_uint64(buf, static_cast<uint64_t>(val));
}

static inline void write_float64(std::vector<uint8_t>& buf, double val) {
    uint64_t bits;
    std::memcpy(&bits, &val, 8);
    write_uint64(buf, bits);
}

static inline void write_short_string(std::vector<uint8_t>& buf, const std::string& s) {
    // Prefixed with uint8 length (max 255)
    write_uint8(buf, static_cast<uint8_t>(s.size()));
    buf.insert(buf.end(), s.begin(), s.end());
}

static inline void write_string16(std::vector<uint8_t>& buf, const std::string& s) {
    // Prefixed with uint16 length
    write_uint16(buf, static_cast<uint16_t>(s.size()));
    buf.insert(buf.end(), s.begin(), s.end());
}

// --- Dtype name mapping ---

// Map c10::ScalarType to Python-compatible dtype string (e.g., "torch.float32")
static const char* scalar_type_to_string(c10::ScalarType dtype) {
    switch (dtype) {
        case c10::ScalarType::Float: return "torch.float32";
        case c10::ScalarType::Double: return "torch.float64";
        case c10::ScalarType::Half: return "torch.float16";
        case c10::ScalarType::BFloat16: return "torch.bfloat16";
        case c10::ScalarType::Int: return "torch.int32";
        case c10::ScalarType::Long: return "torch.int64";
        case c10::ScalarType::Short: return "torch.int16";
        case c10::ScalarType::Char: return "torch.int8";
        case c10::ScalarType::Byte: return "torch.uint8";
        case c10::ScalarType::Bool: return "torch.bool";
        case c10::ScalarType::ComplexFloat: return "torch.complex64";
        case c10::ScalarType::ComplexDouble: return "torch.complex128";
        case c10::ScalarType::Float8_e5m2: return "torch.float8_e5m2";
        case c10::ScalarType::Float8_e4m3fn: return "torch.float8_e4m3fn";
        default:
            TORCH_CHECK(false, "Unsupported dtype for binary serialization: ",
                        c10::toString(dtype));
    }
}

// --- Tensor metadata extraction ---

struct TensorInfo {
    uint64_t tensor_id;
    int64_t storage_id;
    std::vector<int64_t> shape;
    std::vector<int64_t> stride;
    int64_t storage_offset;
    int64_t nbytes;
    std::string dtype_str;
    std::string device_type;
    int32_t device_index;
    uint64_t tensor_ref;  // 0 if not a view
    bool is_new;          // needs metadata sent to server
};

static TensorInfo extract_tensor_info(
    const at::Tensor& tensor,
    const std::string& remote_device_type,
    int64_t remote_device_index)
{
    TensorInfo info;

    auto* impl = dynamic_cast<TensorImpl*>(tensor.unsafeGetTensorImpl());
    if (impl) {
        info.storage_id = impl->get_storage_id();
        info.tensor_id = impl->get_metadata_hash();
    } else {
        info.storage_id = reinterpret_cast<int64_t>(tensor.storage().data_ptr().get());
        // Fallback: compute tensor_id from storage pointer (shouldn't happen for sky tensors)
        info.tensor_id = static_cast<uint64_t>(info.storage_id);
    }

    auto [is_registered, tensor_ref] = check_registration(info.tensor_id, info.storage_id);

    if (is_registered) {
        info.is_new = false;
        info.tensor_ref = 0;
    } else {
        info.is_new = true;
        info.tensor_ref = tensor_ref;

        // Collect full metadata for new tensors
        auto sizes = tensor.sizes();
        info.shape.assign(sizes.begin(), sizes.end());
        auto strides = tensor.strides();
        info.stride.assign(strides.begin(), strides.end());
        info.storage_offset = tensor.storage_offset();
        info.nbytes = tensor.storage().nbytes();
        info.dtype_str = scalar_type_to_string(tensor.scalar_type());
        info.device_type = remote_device_type;
        info.device_index = static_cast<int32_t>(remote_device_index);
    }

    return info;
}

// Write tensor metadata to the binary buffer
static void write_tensor_metadata(std::vector<uint8_t>& buf, const TensorInfo& info) {
    write_uint64(buf, info.tensor_id);

    uint8_t ndim = static_cast<uint8_t>(info.shape.size());
    write_uint8(buf, ndim);

    for (auto s : info.shape) {
        write_int64(buf, s);
    }
    for (auto s : info.stride) {
        write_int64(buf, s);
    }

    // dtype as short string
    write_short_string(buf, info.dtype_str);

    write_int64(buf, info.storage_offset);
    write_int64(buf, info.nbytes);

    // device_type as short string
    write_short_string(buf, info.device_type);
    write_int32(buf, info.device_index);

    // tensor_ref (optional)
    if (info.tensor_ref != 0) {
        write_uint8(buf, 1);
        write_uint64(buf, info.tensor_ref);
    } else {
        write_uint8(buf, 0);
    }
}

// --- Argument serialization ---

// Forward declaration
static void serialize_arg(
    std::vector<uint8_t>& buf,
    py::handle obj,
    std::vector<TensorInfo>& new_tensors,
    int64_t device_index,
    const std::string& remote_device_type,
    int64_t remote_device_index);

static void serialize_arg(
    std::vector<uint8_t>& buf,
    py::handle obj,
    std::vector<TensorInfo>& new_tensors,
    int64_t device_index,
    const std::string& remote_device_type,
    int64_t remote_device_index)
{
    // None
    if (obj.is_none()) {
        write_uint8(buf, static_cast<uint8_t>(ArgType::NONE));
        return;
    }

    // Check for torch.Tensor (must check before other types)
    // Use THPVariable_Check for proper detection of all tensor subclasses
    if (THPVariable_Check(obj.ptr())) {
        const auto& tensor = THPVariable_Unpack(obj.ptr());
        auto device_type = tensor.device().type();

        if (device_type == c10::DeviceType::PrivateUse1) {
            // Sky tensor - extract tensor_id and check registration
            auto info = extract_tensor_info(tensor, remote_device_type, remote_device_index);

            write_uint8(buf, static_cast<uint8_t>(ArgType::TENSOR_ID));
            write_uint64(buf, info.tensor_id);

            if (info.is_new) {
                new_tensors.push_back(std::move(info));
            }
            return;
        }

        if (device_type == c10::DeviceType::CPU && tensor.dim() == 0) {
            // CPU scalar tensor - convert to scalar value
            auto dtype = tensor.scalar_type();
            if (dtype == c10::ScalarType::Bool) {
                write_uint8(buf, static_cast<uint8_t>(ArgType::BOOL));
                write_uint8(buf, tensor.item<bool>() ? 1 : 0);
            } else if (c10::isIntegralType(dtype, /*includeBool=*/false)) {
                write_uint8(buf, static_cast<uint8_t>(ArgType::INT64));
                write_int64(buf, tensor.item<int64_t>());
            } else if (c10::isFloatingType(dtype)) {
                write_uint8(buf, static_cast<uint8_t>(ArgType::FLOAT64));
                write_float64(buf, tensor.item<double>());
            } else {
                // Fallback: convert to string
                write_uint8(buf, static_cast<uint8_t>(ArgType::STRING));
                auto s = py::str(obj).cast<std::string>();
                write_string16(buf, s);
            }
            return;
        }

        if (device_type == c10::DeviceType::CPU && tensor.numel() == 0) {
            // Empty CPU tensor - promote to sky device
            // Create empty tensor on sky device
            auto empty = torch::empty(
                tensor.sizes(),
                torch::TensorOptions()
                    .dtype(tensor.dtype())
                    .device(c10::Device(c10::DeviceType::PrivateUse1, device_index)));

            auto info = extract_tensor_info(empty, remote_device_type, remote_device_index);
            write_uint8(buf, static_cast<uint8_t>(ArgType::TENSOR_ID));
            write_uint64(buf, info.tensor_id);

            if (info.is_new) {
                new_tensors.push_back(std::move(info));
            }
            return;
        }

        throw std::runtime_error(
            "Unsupported tensor: " + tensor.device().str() +
            " with dim " + std::to_string(tensor.dim()) +
            ". Only sky tensors and 0-dim cpu scalar tensors are allowed.");
    }

    // Bool (must check before int since PyBool is subclass of PyLong)
    if (py::isinstance<py::bool_>(obj)) {
        write_uint8(buf, static_cast<uint8_t>(ArgType::BOOL));
        write_uint8(buf, obj.cast<bool>() ? 1 : 0);
        return;
    }

    // Int
    if (py::isinstance<py::int_>(obj)) {
        write_uint8(buf, static_cast<uint8_t>(ArgType::INT64));
        write_int64(buf, obj.cast<int64_t>());
        return;
    }

    // Float
    if (py::isinstance<py::float_>(obj)) {
        write_uint8(buf, static_cast<uint8_t>(ArgType::FLOAT64));
        write_float64(buf, obj.cast<double>());
        return;
    }

    // String
    if (py::isinstance<py::str>(obj)) {
        write_uint8(buf, static_cast<uint8_t>(ArgType::STRING));
        write_string16(buf, obj.cast<std::string>());
        return;
    }

    // torch.device — check before generic isinstance calls (common arg type)
    if (THPDevice_Check(obj.ptr())) {
        auto device_obj = py::cast<c10::Device>(obj);
        if (device_obj.type() == c10::DeviceType::PrivateUse1) {
            // Map sky device to remote device
            auto mapped = remote_device_type + ":" + std::to_string(remote_device_index);
            write_uint8(buf, static_cast<uint8_t>(ArgType::STRING));
            write_string16(buf, mapped);
        } else {
            write_uint8(buf, static_cast<uint8_t>(ArgType::STRING));
            write_string16(buf, device_obj.str());
        }
        return;
    }

    // torch.dtype
    if (THPDtype_Check(obj.ptr())) {
        write_uint8(buf, static_cast<uint8_t>(ArgType::DTYPE));
        auto s = py::str(obj).cast<std::string>();
        write_short_string(buf, s);
        return;
    }

    // torch.memory_format
    if (THPMemoryFormat_Check(obj.ptr())) {
        write_uint8(buf, static_cast<uint8_t>(ArgType::MEMORY_FORMAT));
        auto s = py::str(obj).cast<std::string>();
        write_short_string(buf, s);
        return;
    }

    // torch.layout
    if (THPLayout_Check(obj.ptr())) {
        write_uint8(buf, static_cast<uint8_t>(ArgType::LAYOUT));
        auto s = py::str(obj).cast<std::string>();
        write_short_string(buf, s);
        return;
    }

    // List
    if (py::isinstance<py::list>(obj)) {
        auto list = py::cast<py::list>(obj);
        write_uint8(buf, static_cast<uint8_t>(ArgType::LIST));
        write_uint16(buf, static_cast<uint16_t>(list.size()));
        for (auto item : list) {
            serialize_arg(buf, item, new_tensors, device_index,
                         remote_device_type, remote_device_index);
        }
        return;
    }

    // Tuple
    if (py::isinstance<py::tuple>(obj)) {
        auto tup = py::cast<py::tuple>(obj);
        write_uint8(buf, static_cast<uint8_t>(ArgType::TUPLE));
        write_uint16(buf, static_cast<uint16_t>(tup.size()));
        for (auto item : tup) {
            serialize_arg(buf, item, new_tensors, device_index,
                         remote_device_type, remote_device_index);
        }
        return;
    }

    throw std::runtime_error(
        "Unsupported ATen argument type: " + std::string(py::str(py::type::of(obj))));
}

py::tuple build_execute_aten_request(
    const std::string& op_name,
    py::tuple args,
    py::dict kwargs,
    py::object output_tensors,
    int64_t device_index,
    const std::string& remote_device_type,
    int64_t remote_device_index)
{
    // Phase 1: Collect all tensor metadata from args, kwargs, and outputs
    // by serializing into a temporary buffer.
    std::vector<uint8_t> args_buf;
    args_buf.reserve(256);
    std::vector<TensorInfo> new_tensors;

    // Serialize positional args into temporary buffer
    uint8_t num_args = 0;
    for (auto arg : args) {
        serialize_arg(args_buf, arg, new_tensors, device_index,
                     remote_device_type, remote_device_index);
        num_args++;
    }

    // Serialize kwargs into temporary buffer
    std::vector<uint8_t> kwargs_buf;
    uint8_t num_kwargs = 0;
    for (auto item : kwargs) {
        auto key = item.first.cast<std::string>();
        write_short_string(kwargs_buf, key);
        serialize_arg(kwargs_buf, item.second, new_tensors, device_index,
                     remote_device_type, remote_device_index);
        num_kwargs++;
    }

    // Collect output tensor IDs and metadata
    std::vector<uint64_t> output_ids;
    uint8_t num_outputs = 0;
    if (!output_tensors.is_none()) {
        auto outputs = py::cast<py::list>(output_tensors);
        for (auto item : outputs) {
            if (!item.is_none() && THPVariable_Check(item.ptr())) {
                const auto& tensor = THPVariable_Unpack(item.ptr());
                auto* impl = dynamic_cast<TensorImpl*>(tensor.unsafeGetTensorImpl());
                uint64_t tensor_id;
                int64_t storage_id;
                if (impl) {
                    tensor_id = impl->get_metadata_hash();
                    storage_id = impl->get_storage_id();
                } else {
                    storage_id = reinterpret_cast<int64_t>(tensor.storage().data_ptr().get());
                    tensor_id = static_cast<uint64_t>(storage_id);
                }
                output_ids.push_back(tensor_id);
                num_outputs++;

                // Check if output tensor needs metadata
                auto info = extract_tensor_info(tensor, remote_device_type, remote_device_index);
                if (info.is_new) {
                    new_tensors.push_back(std::move(info));
                }
            }
        }
    }

    // Phase 2: Write final buffer in new order:
    // header → op_name → metadata → outputs → args → kwargs
    uint8_t num_metadata = static_cast<uint8_t>(new_tensors.size());

    std::vector<uint8_t> buffer;
    buffer.reserve(4 + 2 + op_name.size() + args_buf.size() + kwargs_buf.size() + 256);

    // Header (4 bytes)
    buffer.push_back(num_args);
    buffer.push_back(num_kwargs);
    buffer.push_back(num_outputs);
    buffer.push_back(num_metadata);

    // Op name
    write_string16(buffer, op_name);

    // Metadata (moved before args/kwargs/outputs for single-pass server parsing)
    for (const auto& info : new_tensors) {
        write_tensor_metadata(buffer, info);
    }

    // Output tensor IDs
    for (auto tid : output_ids) {
        write_uint64(buffer, tid);
    }

    // Args (from pre-serialized buffer)
    buffer.insert(buffer.end(), args_buf.begin(), args_buf.end());

    // Kwargs (from pre-serialized buffer)
    buffer.insert(buffer.end(), kwargs_buf.begin(), kwargs_buf.end());

    // Build list of new tensor IDs for local registration
    // Also register them in C++ tracking set
    py::list new_tensor_ids;
    py::list new_storage_ids;
    for (const auto& info : new_tensors) {
        new_tensor_ids.append(py::int_(info.tensor_id));
        new_storage_ids.append(py::int_(info.storage_id));
        register_tensor_with_storage(info.tensor_id, info.storage_id);
    }

    return py::make_tuple(
        py::bytes(reinterpret_cast<const char*>(buffer.data()), buffer.size()),
        new_tensor_ids,
        new_storage_ids
    );
}

// --- Submit callback for fused dispatch ---

static PyObject* g_submit_callback = nullptr;

void set_submit_callback(py::object callback) {
    Py_XDECREF(g_submit_callback);
    g_submit_callback = callback.ptr();
    Py_INCREF(g_submit_callback);
}

void clear_submit_callback() {
    Py_XDECREF(g_submit_callback);
    g_submit_callback = nullptr;
}

// --- Device mapping registry ---

static std::unordered_map<int64_t, std::pair<std::string, int64_t>> g_device_mappings;
// local sky index → (remote_device_type, remote_device_index)

void register_device_mapping(int64_t local_index,
                              const std::string& remote_type,
                              int64_t remote_index) {
    g_device_mappings[local_index] = {remote_type, remote_index};
}

void clear_device_mappings() {
    g_device_mappings.clear();
}

// --- Shape cache ---

static std::unordered_map<uint64_t, std::vector<OutputMeta>> g_shape_cache;
static constexpr size_t SHAPE_CACHE_MAX = 4096;

void populate_shape_cache(uint64_t cache_key, py::list output_metas) {
    if (g_shape_cache.size() >= SHAPE_CACHE_MAX) return;

    std::vector<OutputMeta> metas;
    metas.reserve(output_metas.size());

    for (auto item : output_metas) {
        if (item.is_none()) {
            OutputMeta m;
            m.dtype = c10::ScalarType::Float;  // unused
            m.storage_offset = 0;
            m.alias_input = -2;  // None output
            metas.push_back(std::move(m));
            continue;
        }

        auto tup = item.cast<py::tuple>();
        auto shape_list = tup[0].cast<std::vector<int64_t>>();
        auto stride_list = tup[1].cast<std::vector<int64_t>>();
        auto dtype_int = tup[2].cast<int>();
        auto storage_offset = tup[3].cast<int64_t>();
        auto alias_input = tup[4].cast<int>();

        OutputMeta m;
        m.shape = std::move(shape_list);
        m.stride = std::move(stride_list);
        m.dtype = static_cast<c10::ScalarType>(dtype_int);
        m.storage_offset = storage_offset;
        m.alias_input = alias_input;
        metas.push_back(std::move(m));
    }

    g_shape_cache[cache_key] = std::move(metas);
}

void clear_shape_cache() {
    g_shape_cache.clear();
}

// --- Dispatch context computation (cache key + tensor collection) ---

static inline uint64_t hash_combine(uint64_t h, uint64_t v) {
    h ^= v + 0x9e3779b97f4a7c15ULL + (h << 12) + (h >> 4);
    return h;
}

static inline uint64_t hash_bytes(const char* data, Py_ssize_t len) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (Py_ssize_t i = 0; i < len; i++) {
        h ^= static_cast<uint64_t>(static_cast<unsigned char>(data[i]));
        h *= 0x100000001b3ULL;
    }
    return h;
}

// Forward declaration
static bool hash_arg(
    py::handle obj,
    uint64_t& h,
    std::vector<py::object>& input_tensors,
    std::unordered_map<int64_t, int>& storage_groups,
    int& next_group,
    std::unordered_set<int64_t>& seen_ptrs,
    int64_t& sky_device_index);

static bool hash_arg(
    py::handle obj,
    uint64_t& h,
    std::vector<py::object>& input_tensors,
    std::unordered_map<int64_t, int>& storage_groups,
    int& next_group,
    std::unordered_set<int64_t>& seen_ptrs,
    int64_t& sky_device_index)
{
    PyObject* ptr = obj.ptr();

    // None
    if (ptr == Py_None) {
        h = hash_combine(h, 0xFFULL);
        return true;
    }

    // Tensor
    if (THPVariable_Check(ptr)) {
        const auto& tensor = THPVariable_Unpack(ptr);
        auto device_type = tensor.device().type();

        // Get storage key for group tracking
        int64_t storage_key;
        auto* impl = dynamic_cast<TensorImpl*>(tensor.unsafeGetTensorImpl());
        if (impl) {
            storage_key = impl->get_storage_id();
        } else {
            storage_key = reinterpret_cast<int64_t>(tensor.storage().data_ptr().get());
        }

        if (device_type == c10::DeviceType::PrivateUse1) {
            // Sky tensor
            if (sky_device_index < 0) {
                sky_device_index = tensor.device().index();
            }

            // Collect unique input tensors by storage
            if (seen_ptrs.find(storage_key) == seen_ptrs.end()) {
                seen_ptrs.insert(storage_key);
                input_tensors.push_back(py::reinterpret_borrow<py::object>(obj));
            }
        }

        // Storage group (for both sky and non-sky tensors)
        int group;
        auto git = storage_groups.find(storage_key);
        if (git == storage_groups.end()) {
            group = next_group++;
            storage_groups[storage_key] = group;
        } else {
            group = git->second;
        }

        // Hash tensor metadata
        h = hash_combine(h, 0x01ULL);  // tensor marker
        auto sizes = tensor.sizes();
        for (auto s : sizes) {
            h = hash_combine(h, static_cast<uint64_t>(s));
        }
        h = hash_combine(h, static_cast<uint64_t>(tensor.scalar_type()));
        auto strides = tensor.strides();
        for (auto s : strides) {
            h = hash_combine(h, static_cast<uint64_t>(s));
        }
        h = hash_combine(h, static_cast<uint64_t>(tensor.storage_offset()));
        h = hash_combine(h, static_cast<uint64_t>(tensor.storage().nbytes()));
        h = hash_combine(h, static_cast<uint64_t>(group));
        return true;
    }

    // Bool (must check before int since PyBool is subclass of PyLong)
    if (PyBool_Check(ptr)) {
        h = hash_combine(h, 0x04ULL);
        h = hash_combine(h, ptr == Py_True ? 1ULL : 0ULL);
        return true;
    }

    // Int
    if (PyLong_Check(ptr)) {
        h = hash_combine(h, 0x02ULL);
        h = hash_combine(h, static_cast<uint64_t>(PyLong_AsLongLong(ptr)));
        return true;
    }

    // Float
    if (PyFloat_Check(ptr)) {
        h = hash_combine(h, 0x03ULL);
        uint64_t bits;
        double val = PyFloat_AsDouble(ptr);
        std::memcpy(&bits, &val, 8);
        h = hash_combine(h, bits);
        return true;
    }

    // String
    if (PyUnicode_Check(ptr)) {
        h = hash_combine(h, 0x08ULL);
        Py_ssize_t len;
        const char* data = PyUnicode_AsUTF8AndSize(ptr, &len);
        h = hash_combine(h, hash_bytes(data, len));
        return true;
    }

    // torch.device
    if (THPDevice_Check(ptr)) {
        const auto& device_obj = reinterpret_cast<THPDevice*>(ptr)->device;
        if (device_obj.type() == c10::DeviceType::PrivateUse1 && sky_device_index < 0) {
            sky_device_index = device_obj.has_index() ? device_obj.index() : 0;
        }
        h = hash_combine(h, 0x0DULL);
        h = hash_combine(h, static_cast<uint64_t>(device_obj.type()));
        h = hash_combine(h, device_obj.has_index()
            ? static_cast<uint64_t>(device_obj.index()) : 0xFFFFFFFFULL);
        return true;
    }

    // torch.dtype — hash enum value directly (no string conversion)
    if (THPDtype_Check(ptr)) {
        h = hash_combine(h, 0x05ULL);
        h = hash_combine(h, static_cast<uint64_t>(
            reinterpret_cast<THPDtype*>(ptr)->scalar_type));
        return true;
    }

    // torch.memory_format — hash enum value directly
    if (THPMemoryFormat_Check(ptr)) {
        h = hash_combine(h, 0x06ULL);
        h = hash_combine(h, static_cast<uint64_t>(
            reinterpret_cast<THPMemoryFormat*>(ptr)->memory_format));
        return true;
    }

    // torch.layout — hash enum value directly
    if (THPLayout_Check(ptr)) {
        h = hash_combine(h, 0x07ULL);
        h = hash_combine(h, static_cast<uint64_t>(
            reinterpret_cast<THPLayout*>(ptr)->layout));
        return true;
    }

    // List
    if (PyList_Check(ptr)) {
        Py_ssize_t n = PyList_GET_SIZE(ptr);
        h = hash_combine(h, 0x09ULL);
        h = hash_combine(h, static_cast<uint64_t>(n));
        for (Py_ssize_t i = 0; i < n; i++) {
            if (!hash_arg(py::handle(PyList_GET_ITEM(ptr, i)), h, input_tensors,
                         storage_groups, next_group, seen_ptrs, sky_device_index)) {
                return false;
            }
        }
        return true;
    }

    // Tuple
    if (PyTuple_Check(ptr)) {
        Py_ssize_t n = PyTuple_GET_SIZE(ptr);
        h = hash_combine(h, 0x0AULL);
        h = hash_combine(h, static_cast<uint64_t>(n));
        for (Py_ssize_t i = 0; i < n; i++) {
            if (!hash_arg(py::handle(PyTuple_GET_ITEM(ptr, i)), h, input_tensors,
                         storage_groups, next_group, seen_ptrs, sky_device_index)) {
                return false;
            }
        }
        return true;
    }

    // Unsupported type — uncacheable
    return false;
}

// Helper to build the (hash, tensors_list, device_index) return tuple via raw C API
static inline py::tuple build_context_result(
    uint64_t h,
    const std::vector<py::object>& input_tensors,
    int64_t sky_device_index)
{
    PyObject* tensors_list = PyList_New(static_cast<Py_ssize_t>(input_tensors.size()));
    for (size_t i = 0; i < input_tensors.size(); i++) {
        PyObject* t = input_tensors[i].ptr();
        Py_INCREF(t);
        PyList_SET_ITEM(tensors_list, static_cast<Py_ssize_t>(i), t);
    }
    PyObject* result = PyTuple_New(3);
    PyTuple_SET_ITEM(result, 0, PyLong_FromUnsignedLongLong(h));
    PyTuple_SET_ITEM(result, 1, tensors_list);
    PyTuple_SET_ITEM(result, 2, PyLong_FromLongLong(sky_device_index));
    return py::reinterpret_steal<py::tuple>(result);
}

py::tuple compute_dispatch_context(
    py::str op_name,
    py::tuple args,
    py::dict kwargs)
{
    uint64_t h = 0;

    // Hash op_name using raw C API (zero-copy)
    Py_ssize_t name_len;
    const char* name_data = PyUnicode_AsUTF8AndSize(op_name.ptr(), &name_len);
    h = hash_combine(h, hash_bytes(name_data, name_len));

    std::vector<py::object> input_tensors;
    std::unordered_map<int64_t, int> storage_groups;
    int next_group = 0;
    std::unordered_set<int64_t> seen_ptrs;
    int64_t sky_device_index = -1;

    // Check kwargs "device" first (matches Python behavior)
    PyObject* dev = PyDict_GetItemString(kwargs.ptr(), "device");
    if (dev && THPDevice_Check(dev)) {
        const auto& device_obj = reinterpret_cast<THPDevice*>(dev)->device;
        if (device_obj.type() == c10::DeviceType::PrivateUse1) {
            sky_device_index = device_obj.has_index() ? device_obj.index() : 0;
        }
    }

    // Hash positional args using raw C API
    Py_ssize_t n_args = PyTuple_GET_SIZE(args.ptr());
    for (Py_ssize_t i = 0; i < n_args; i++) {
        if (!hash_arg(py::handle(PyTuple_GET_ITEM(args.ptr(), i)), h, input_tensors,
                     storage_groups, next_group, seen_ptrs, sky_device_index)) {
            return build_context_result(0, input_tensors, sky_device_index);
        }
    }

    // Hash kwargs (sorted by key for deterministic ordering)
    Py_ssize_t n_kwargs = PyDict_Size(kwargs.ptr());
    if (n_kwargs > 0) {
        // Collect key-value pairs using PyDict_Next (avoids string copies)
        std::vector<std::pair<PyObject*, PyObject*>> sorted_items;
        sorted_items.reserve(n_kwargs);
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(kwargs.ptr(), &pos, &key, &value)) {
            sorted_items.push_back({key, value});
        }
        std::sort(sorted_items.begin(), sorted_items.end(),
            [](const auto& a, const auto& b) {
                return PyUnicode_Compare(a.first, b.first) < 0;
            });

        for (const auto& [k, v] : sorted_items) {
            Py_ssize_t klen;
            const char* kdata = PyUnicode_AsUTF8AndSize(k, &klen);
            h = hash_combine(h, hash_bytes(kdata, klen));
            if (!hash_arg(py::handle(v), h, input_tensors, storage_groups,
                         next_group, seen_ptrs, sky_device_index)) {
                return build_context_result(0, input_tensors, sky_device_index);
            }
        }
    }

    // Ensure hash is never 0 (reserved for uncacheable)
    if (h == 0) h = 1;

    return build_context_result(h, input_tensors, sky_device_index);
}

// --- Fused hash + serialize for dispatch_cached_aten ---

// Forward declaration
static bool hash_and_serialize_arg(
    py::handle obj,
    uint64_t& h,
    std::vector<uint8_t>& buf,
    std::vector<at::Tensor>& input_tensors_at,
    std::vector<py::object>& input_tensors_py,
    std::unordered_map<int64_t, int>& storage_groups,
    int& next_group,
    std::unordered_set<int64_t>& seen_ptrs,
    int64_t& sky_device_index,
    std::vector<TensorInfo>& new_tensors,
    int64_t device_index,
    const std::string& remote_device_type,
    int64_t remote_device_index);

static bool hash_and_serialize_arg(
    py::handle obj,
    uint64_t& h,
    std::vector<uint8_t>& buf,
    std::vector<at::Tensor>& input_tensors_at,
    std::vector<py::object>& input_tensors_py,
    std::unordered_map<int64_t, int>& storage_groups,
    int& next_group,
    std::unordered_set<int64_t>& seen_ptrs,
    int64_t& sky_device_index,
    std::vector<TensorInfo>& new_tensors,
    int64_t device_index,
    const std::string& remote_device_type,
    int64_t remote_device_index)
{
    PyObject* ptr = obj.ptr();

    // None
    if (ptr == Py_None) {
        h = hash_combine(h, 0xFFULL);
        write_uint8(buf, static_cast<uint8_t>(ArgType::NONE));
        return true;
    }

    // Tensor
    if (THPVariable_Check(ptr)) {
        const auto& tensor = THPVariable_Unpack(ptr);
        auto device_type = tensor.device().type();

        // Get storage key for group tracking
        int64_t storage_key;
        auto* impl = dynamic_cast<TensorImpl*>(tensor.unsafeGetTensorImpl());
        if (impl) {
            storage_key = impl->get_storage_id();
        } else {
            storage_key = reinterpret_cast<int64_t>(tensor.storage().data_ptr().get());
        }

        if (device_type == c10::DeviceType::PrivateUse1) {
            // Sky tensor
            if (sky_device_index < 0) {
                sky_device_index = tensor.device().index();
            }

            // Collect unique input tensors by storage
            if (seen_ptrs.find(storage_key) == seen_ptrs.end()) {
                seen_ptrs.insert(storage_key);
                input_tensors_at.push_back(tensor);
                input_tensors_py.push_back(py::reinterpret_borrow<py::object>(obj));
            }

            // Hash tensor metadata
            int group;
            auto git = storage_groups.find(storage_key);
            if (git == storage_groups.end()) {
                group = next_group++;
                storage_groups[storage_key] = group;
            } else {
                group = git->second;
            }

            h = hash_combine(h, 0x01ULL);
            auto sizes = tensor.sizes();
            for (auto s : sizes) {
                h = hash_combine(h, static_cast<uint64_t>(s));
            }
            h = hash_combine(h, static_cast<uint64_t>(tensor.scalar_type()));
            auto strides = tensor.strides();
            for (auto s : strides) {
                h = hash_combine(h, static_cast<uint64_t>(s));
            }
            h = hash_combine(h, static_cast<uint64_t>(tensor.storage_offset()));
            h = hash_combine(h, static_cast<uint64_t>(tensor.storage().nbytes()));
            h = hash_combine(h, static_cast<uint64_t>(group));

            // Serialize: TENSOR_ID tag + tensor_id
            auto info = extract_tensor_info(tensor, remote_device_type, remote_device_index);
            write_uint8(buf, static_cast<uint8_t>(ArgType::TENSOR_ID));
            write_uint64(buf, info.tensor_id);
            if (info.is_new) {
                new_tensors.push_back(std::move(info));
            }
            return true;
        }

        // Non-sky tensor (cpu scalar or empty cpu)
        int group;
        auto git = storage_groups.find(storage_key);
        if (git == storage_groups.end()) {
            group = next_group++;
            storage_groups[storage_key] = group;
        } else {
            group = git->second;
        }

        h = hash_combine(h, 0x01ULL);
        auto sizes = tensor.sizes();
        for (auto s : sizes) {
            h = hash_combine(h, static_cast<uint64_t>(s));
        }
        h = hash_combine(h, static_cast<uint64_t>(tensor.scalar_type()));
        auto strides = tensor.strides();
        for (auto s : strides) {
            h = hash_combine(h, static_cast<uint64_t>(s));
        }
        h = hash_combine(h, static_cast<uint64_t>(tensor.storage_offset()));
        h = hash_combine(h, static_cast<uint64_t>(tensor.storage().nbytes()));
        h = hash_combine(h, static_cast<uint64_t>(group));

        if (device_type == c10::DeviceType::CPU && tensor.dim() == 0) {
            // CPU scalar tensor - serialize as scalar value
            auto dtype = tensor.scalar_type();
            if (dtype == c10::ScalarType::Bool) {
                write_uint8(buf, static_cast<uint8_t>(ArgType::BOOL));
                write_uint8(buf, tensor.item<bool>() ? 1 : 0);
            } else if (c10::isIntegralType(dtype, false)) {
                write_uint8(buf, static_cast<uint8_t>(ArgType::INT64));
                write_int64(buf, tensor.item<int64_t>());
            } else if (c10::isFloatingType(dtype)) {
                write_uint8(buf, static_cast<uint8_t>(ArgType::FLOAT64));
                write_float64(buf, tensor.item<double>());
            } else {
                write_uint8(buf, static_cast<uint8_t>(ArgType::STRING));
                auto s = py::str(obj).cast<std::string>();
                write_string16(buf, s);
            }
            return true;
        }

        if (device_type == c10::DeviceType::CPU && tensor.numel() == 0) {
            // Empty CPU tensor - promote to sky device
            auto empty = torch::empty(
                tensor.sizes(),
                torch::TensorOptions()
                    .dtype(tensor.dtype())
                    .device(c10::Device(c10::DeviceType::PrivateUse1, device_index)));
            auto info = extract_tensor_info(empty, remote_device_type, remote_device_index);
            write_uint8(buf, static_cast<uint8_t>(ArgType::TENSOR_ID));
            write_uint64(buf, info.tensor_id);
            if (info.is_new) {
                new_tensors.push_back(std::move(info));
            }
            return true;
        }

        // Unsupported tensor type — uncacheable
        return false;
    }

    // Bool (must check before int)
    if (PyBool_Check(ptr)) {
        h = hash_combine(h, 0x04ULL);
        h = hash_combine(h, ptr == Py_True ? 1ULL : 0ULL);
        write_uint8(buf, static_cast<uint8_t>(ArgType::BOOL));
        write_uint8(buf, ptr == Py_True ? 1 : 0);
        return true;
    }

    // Int
    if (PyLong_Check(ptr)) {
        int64_t val = PyLong_AsLongLong(ptr);
        h = hash_combine(h, 0x02ULL);
        h = hash_combine(h, static_cast<uint64_t>(val));
        write_uint8(buf, static_cast<uint8_t>(ArgType::INT64));
        write_int64(buf, val);
        return true;
    }

    // Float
    if (PyFloat_Check(ptr)) {
        double val = PyFloat_AsDouble(ptr);
        uint64_t bits;
        std::memcpy(&bits, &val, 8);
        h = hash_combine(h, 0x03ULL);
        h = hash_combine(h, bits);
        write_uint8(buf, static_cast<uint8_t>(ArgType::FLOAT64));
        write_float64(buf, val);
        return true;
    }

    // String
    if (PyUnicode_Check(ptr)) {
        Py_ssize_t len;
        const char* data = PyUnicode_AsUTF8AndSize(ptr, &len);
        h = hash_combine(h, 0x08ULL);
        h = hash_combine(h, hash_bytes(data, len));
        write_uint8(buf, static_cast<uint8_t>(ArgType::STRING));
        write_string16(buf, std::string(data, len));
        return true;
    }

    // torch.device
    if (THPDevice_Check(ptr)) {
        const auto& device_obj = reinterpret_cast<THPDevice*>(ptr)->device;
        if (device_obj.type() == c10::DeviceType::PrivateUse1 && sky_device_index < 0) {
            sky_device_index = device_obj.has_index() ? device_obj.index() : 0;
        }
        h = hash_combine(h, 0x0DULL);
        h = hash_combine(h, static_cast<uint64_t>(device_obj.type()));
        h = hash_combine(h, device_obj.has_index()
            ? static_cast<uint64_t>(device_obj.index()) : 0xFFFFFFFFULL);

        // Serialize: map sky device to remote device string
        if (device_obj.type() == c10::DeviceType::PrivateUse1) {
            auto mapped = remote_device_type + ":" + std::to_string(remote_device_index);
            write_uint8(buf, static_cast<uint8_t>(ArgType::STRING));
            write_string16(buf, mapped);
        } else {
            write_uint8(buf, static_cast<uint8_t>(ArgType::STRING));
            write_string16(buf, device_obj.str());
        }
        return true;
    }

    // torch.dtype
    if (THPDtype_Check(ptr)) {
        h = hash_combine(h, 0x05ULL);
        h = hash_combine(h, static_cast<uint64_t>(
            reinterpret_cast<THPDtype*>(ptr)->scalar_type));
        write_uint8(buf, static_cast<uint8_t>(ArgType::DTYPE));
        auto s = py::str(obj).cast<std::string>();
        write_short_string(buf, s);
        return true;
    }

    // torch.memory_format
    if (THPMemoryFormat_Check(ptr)) {
        h = hash_combine(h, 0x06ULL);
        h = hash_combine(h, static_cast<uint64_t>(
            reinterpret_cast<THPMemoryFormat*>(ptr)->memory_format));
        write_uint8(buf, static_cast<uint8_t>(ArgType::MEMORY_FORMAT));
        auto s = py::str(obj).cast<std::string>();
        write_short_string(buf, s);
        return true;
    }

    // torch.layout
    if (THPLayout_Check(ptr)) {
        h = hash_combine(h, 0x07ULL);
        h = hash_combine(h, static_cast<uint64_t>(
            reinterpret_cast<THPLayout*>(ptr)->layout));
        write_uint8(buf, static_cast<uint8_t>(ArgType::LAYOUT));
        auto s = py::str(obj).cast<std::string>();
        write_short_string(buf, s);
        return true;
    }

    // List
    if (PyList_Check(ptr)) {
        Py_ssize_t n = PyList_GET_SIZE(ptr);
        h = hash_combine(h, 0x09ULL);
        h = hash_combine(h, static_cast<uint64_t>(n));
        write_uint8(buf, static_cast<uint8_t>(ArgType::LIST));
        write_uint16(buf, static_cast<uint16_t>(n));
        for (Py_ssize_t i = 0; i < n; i++) {
            if (!hash_and_serialize_arg(
                    py::handle(PyList_GET_ITEM(ptr, i)), h, buf,
                    input_tensors_at, input_tensors_py,
                    storage_groups, next_group, seen_ptrs, sky_device_index,
                    new_tensors, device_index, remote_device_type, remote_device_index)) {
                return false;
            }
        }
        return true;
    }

    // Tuple
    if (PyTuple_Check(ptr)) {
        Py_ssize_t n = PyTuple_GET_SIZE(ptr);
        h = hash_combine(h, 0x0AULL);
        h = hash_combine(h, static_cast<uint64_t>(n));
        write_uint8(buf, static_cast<uint8_t>(ArgType::TUPLE));
        write_uint16(buf, static_cast<uint16_t>(n));
        for (Py_ssize_t i = 0; i < n; i++) {
            if (!hash_and_serialize_arg(
                    py::handle(PyTuple_GET_ITEM(ptr, i)), h, buf,
                    input_tensors_at, input_tensors_py,
                    storage_groups, next_group, seen_ptrs, sky_device_index,
                    new_tensors, device_index, remote_device_type, remote_device_index)) {
                return false;
            }
        }
        return true;
    }

    // Unsupported type — uncacheable
    return false;
}

// --- dispatch_cached_aten ---

// Forward declaration for empty_strided from EmptyTensor.cpp
at::Tensor empty_strided(
    at::IntArrayRef size,
    at::IntArrayRef stride,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory);

py::object dispatch_cached_aten(
    py::str op_name,
    py::tuple args,
    py::dict kwargs)
{
    // Phase 1: Single walk — hash + serialize args/kwargs simultaneously.
    // Determine remote device mapping from the sky device index.
    uint64_t h = 0;

    Py_ssize_t name_len;
    const char* name_data = PyUnicode_AsUTF8AndSize(op_name.ptr(), &name_len);
    h = hash_combine(h, hash_bytes(name_data, name_len));
    std::string op_name_str(name_data, name_len);

    std::vector<at::Tensor> input_tensors_at;
    std::vector<py::object> input_tensors_py;
    std::unordered_map<int64_t, int> storage_groups;
    int next_group = 0;
    std::unordered_set<int64_t> seen_ptrs;
    int64_t sky_device_index = -1;
    std::vector<TensorInfo> new_tensors;

    // Check kwargs "device" first
    PyObject* dev = PyDict_GetItemString(kwargs.ptr(), "device");
    if (dev && THPDevice_Check(dev)) {
        const auto& device_obj = reinterpret_cast<THPDevice*>(dev)->device;
        if (device_obj.type() == c10::DeviceType::PrivateUse1) {
            sky_device_index = device_obj.has_index() ? device_obj.index() : 0;
        }
    }

    // We need a preliminary scan to find sky_device_index before we can determine
    // remote device mapping. But hash_and_serialize_arg needs the mapping for
    // serialization. So: first check if we can get device from first tensor arg.
    // If not found in first pass, we fall back.
    // Strategy: Do a quick scan for sky device index first.
    if (sky_device_index < 0) {
        Py_ssize_t n_args = PyTuple_GET_SIZE(args.ptr());
        for (Py_ssize_t i = 0; i < n_args && sky_device_index < 0; i++) {
            PyObject* arg = PyTuple_GET_ITEM(args.ptr(), i);
            if (THPVariable_Check(arg)) {
                const auto& tensor = THPVariable_Unpack(arg);
                if (tensor.device().type() == c10::DeviceType::PrivateUse1) {
                    sky_device_index = tensor.device().index();
                }
            }
        }
    }

    // If still no device, we can't proceed with fused path
    if (sky_device_index < 0) {
        // Return None for uncacheable (no sky device found yet)
        return py::none();
    }

    // Look up device mapping
    auto dm_it = g_device_mappings.find(sky_device_index);
    if (dm_it == g_device_mappings.end()) {
        // No mapping registered, fall back to Python path
        return py::none();
    }
    const auto& remote_device_type = dm_it->second.first;
    int64_t remote_device_index = dm_it->second.second;

    // Serialize args buffer
    std::vector<uint8_t> args_buf;
    args_buf.reserve(256);
    uint8_t num_args = 0;

    Py_ssize_t n_args = PyTuple_GET_SIZE(args.ptr());
    for (Py_ssize_t i = 0; i < n_args; i++) {
        if (!hash_and_serialize_arg(
                py::handle(PyTuple_GET_ITEM(args.ptr(), i)), h, args_buf,
                input_tensors_at, input_tensors_py,
                storage_groups, next_group, seen_ptrs, sky_device_index,
                new_tensors, sky_device_index, remote_device_type, remote_device_index)) {
            // Uncacheable
            return py::none();
        }
        num_args++;
    }

    // Serialize kwargs buffer
    std::vector<uint8_t> kwargs_buf;
    uint8_t num_kwargs = 0;
    Py_ssize_t n_kwargs = PyDict_Size(kwargs.ptr());

    if (n_kwargs > 0) {
        // Sort kwargs for deterministic hashing
        std::vector<std::pair<PyObject*, PyObject*>> sorted_items;
        sorted_items.reserve(n_kwargs);
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(kwargs.ptr(), &pos, &key, &value)) {
            sorted_items.push_back({key, value});
        }
        std::sort(sorted_items.begin(), sorted_items.end(),
            [](const auto& a, const auto& b) {
                return PyUnicode_Compare(a.first, b.first) < 0;
            });

        for (const auto& [k, v] : sorted_items) {
            Py_ssize_t klen;
            const char* kdata = PyUnicode_AsUTF8AndSize(k, &klen);
            h = hash_combine(h, hash_bytes(kdata, klen));

            // Write kwarg name to kwargs_buf
            write_short_string(kwargs_buf, std::string(kdata, klen));

            if (!hash_and_serialize_arg(
                    py::handle(v), h, kwargs_buf,
                    input_tensors_at, input_tensors_py,
                    storage_groups, next_group, seen_ptrs, sky_device_index,
                    new_tensors, sky_device_index, remote_device_type, remote_device_index)) {
                return py::none();
            }
            num_kwargs++;
        }
    }

    // Ensure hash is never 0
    if (h == 0) h = 1;

    // Phase 2: Cache lookup
    auto cache_it = g_shape_cache.find(h);
    if (cache_it == g_shape_cache.end()) {
        // Cache miss — return (cache_hash, input_tensors, sky_device_index)
        PyObject* tensors_list = PyList_New(static_cast<Py_ssize_t>(input_tensors_py.size()));
        for (size_t i = 0; i < input_tensors_py.size(); i++) {
            PyObject* t = input_tensors_py[i].ptr();
            Py_INCREF(t);
            PyList_SET_ITEM(tensors_list, static_cast<Py_ssize_t>(i), t);
        }
        PyObject* result = PyTuple_New(3);
        PyTuple_SET_ITEM(result, 0, PyLong_FromUnsignedLongLong(h));
        PyTuple_SET_ITEM(result, 1, tensors_list);
        PyTuple_SET_ITEM(result, 2, PyLong_FromLongLong(sky_device_index));
        return py::reinterpret_steal<py::object>(result);
    }

    // Phase 3: Cache hit — create outputs
    const auto& cached_metas = cache_it->second;
    std::vector<py::object> output_tensors_py;
    output_tensors_py.reserve(cached_metas.size());
    std::vector<uint64_t> output_ids;
    output_ids.reserve(cached_metas.size());

    // New tensors from output creation
    // (new_tensors already contains input new tensors)
    c10::Device sky_device(c10::DeviceType::PrivateUse1, sky_device_index);

    for (const auto& meta : cached_metas) {
        if (meta.alias_input == -2) {
            // None output
            output_tensors_py.push_back(py::none());
            continue;
        }

        at::Tensor output;
        if (meta.alias_input >= 0) {
            // Alias: as_strided on input tensor (direct C++ call, no dispatch)
            if (static_cast<size_t>(meta.alias_input) < input_tensors_at.size()) {
                auto& input = input_tensors_at[meta.alias_input];

                // Resize if input is uninitialized and output has data
                if (input.numel() == 0 && !meta.shape.empty()) {
                    bool all_positive = true;
                    for (auto s : meta.shape) {
                        if (s <= 0) { all_positive = false; break; }
                    }
                    if (all_positive) {
                        input.resize_(meta.shape);
                    }
                }

                output = input.as_strided(meta.shape, meta.stride, meta.storage_offset);
            } else {
                // Invalid alias index, shouldn't happen
                return py::none();
            }
        } else {
            // New storage: call our C++ empty_strided directly
            output = skytorch::empty_strided(
                meta.shape, meta.stride,
                meta.dtype,
                c10::Layout::Strided,
                sky_device,
                false);
        }

        // Get output tensor id for the request
        auto* out_impl = dynamic_cast<TensorImpl*>(output.unsafeGetTensorImpl());
        uint64_t out_tensor_id;
        int64_t out_storage_id;
        if (out_impl) {
            out_tensor_id = out_impl->get_metadata_hash();
            out_storage_id = out_impl->get_storage_id();
        } else {
            out_storage_id = reinterpret_cast<int64_t>(output.storage().data_ptr().get());
            out_tensor_id = static_cast<uint64_t>(out_storage_id);
        }
        output_ids.push_back(out_tensor_id);

        // Check if output tensor needs metadata
        auto info = extract_tensor_info(output, remote_device_type, remote_device_index);
        if (info.is_new) {
            new_tensors.push_back(std::move(info));
        }

        output_tensors_py.push_back(py::cast(output));
    }

    // Phase 4: Assemble binary request
    uint8_t num_outputs = static_cast<uint8_t>(output_ids.size());
    uint8_t num_metadata = static_cast<uint8_t>(new_tensors.size());

    std::vector<uint8_t> buffer;
    buffer.reserve(4 + 2 + op_name_str.size() + args_buf.size() + kwargs_buf.size() + 256);

    // Header
    buffer.push_back(num_args);
    buffer.push_back(num_kwargs);
    buffer.push_back(num_outputs);
    buffer.push_back(num_metadata);

    // Op name
    write_string16(buffer, op_name_str);

    // Metadata
    for (const auto& info : new_tensors) {
        write_tensor_metadata(buffer, info);
    }

    // Output tensor IDs
    for (auto tid : output_ids) {
        write_uint64(buffer, tid);
    }

    // Args + Kwargs
    buffer.insert(buffer.end(), args_buf.begin(), args_buf.end());
    buffer.insert(buffer.end(), kwargs_buf.begin(), kwargs_buf.end());

    // Build new tensor ID / storage ID lists + register
    py::list new_tensor_ids;
    py::list new_storage_ids;
    for (const auto& info : new_tensors) {
        new_tensor_ids.append(py::int_(info.tensor_id));
        new_storage_ids.append(py::int_(info.storage_id));
        register_tensor_with_storage(info.tensor_id, info.storage_id);
    }

    // Phase 5: Submit directly from C++ if callback is registered
    if (g_submit_callback) {
        // Call submit callback with (raw_bytes, new_tensor_ids, new_storage_ids, dev_idx)
        PyObject* raw = PyBytes_FromStringAndSize(
            reinterpret_cast<const char*>(buffer.data()),
            static_cast<Py_ssize_t>(buffer.size()));
        PyObject* cb_args = PyTuple_New(4);
        PyTuple_SET_ITEM(cb_args, 0, raw);
        PyTuple_SET_ITEM(cb_args, 1, new_tensor_ids.release().ptr());
        PyTuple_SET_ITEM(cb_args, 2, new_storage_ids.release().ptr());
        PyTuple_SET_ITEM(cb_args, 3, PyLong_FromLongLong(sky_device_index));
        PyObject* cb_result = PyObject_Call(g_submit_callback, cb_args, nullptr);
        Py_DECREF(cb_args);
        if (cb_result == nullptr) {
            throw py::error_already_set();
        }
        Py_DECREF(cb_result);

        // Build unpacked output directly
        PyObject* unpacked_output;
        if (output_tensors_py.size() > 1) {
            PyObject* tup = PyTuple_New(static_cast<Py_ssize_t>(output_tensors_py.size()));
            for (size_t i = 0; i < output_tensors_py.size(); i++) {
                PyObject* t = output_tensors_py[i].ptr();
                Py_INCREF(t);
                PyTuple_SET_ITEM(tup, static_cast<Py_ssize_t>(i), t);
            }
            unpacked_output = tup;
        } else if (output_tensors_py.size() == 1) {
            unpacked_output = output_tensors_py[0].ptr();
            Py_INCREF(unpacked_output);
        } else {
            unpacked_output = Py_None;
            Py_INCREF(unpacked_output);
        }

        // Return 1-tuple (unpacked_output,) to signal "cache hit, already submitted"
        PyObject* wrapper = PyTuple_New(1);
        PyTuple_SET_ITEM(wrapper, 0, unpacked_output);
        return py::reinterpret_steal<py::object>(wrapper);
    }

    // Fallback: return tuple for Python-side submission
    // Build unpacked output: single tensor, tuple of tensors, or None
    PyObject* unpacked_output;
    if (output_tensors_py.size() > 1) {
        PyObject* tup = PyTuple_New(static_cast<Py_ssize_t>(output_tensors_py.size()));
        for (size_t i = 0; i < output_tensors_py.size(); i++) {
            PyObject* t = output_tensors_py[i].ptr();
            Py_INCREF(t);
            PyTuple_SET_ITEM(tup, static_cast<Py_ssize_t>(i), t);
        }
        unpacked_output = tup;
    } else if (output_tensors_py.size() == 1) {
        unpacked_output = output_tensors_py[0].ptr();
        Py_INCREF(unpacked_output);
    } else {
        unpacked_output = Py_None;
        Py_INCREF(unpacked_output);
    }

    // Return (unpacked_output, raw_bytes, new_tensor_ids, new_storage_ids, sky_device_index)
    PyObject* result = PyTuple_New(5);
    PyTuple_SET_ITEM(result, 0, unpacked_output);
    PyTuple_SET_ITEM(result, 1,
        PyBytes_FromStringAndSize(
            reinterpret_cast<const char*>(buffer.data()),
            static_cast<Py_ssize_t>(buffer.size())));
    PyTuple_SET_ITEM(result, 2, new_tensor_ids.release().ptr());
    PyTuple_SET_ITEM(result, 3, new_storage_ids.release().ptr());
    PyTuple_SET_ITEM(result, 4, PyLong_FromLongLong(sky_device_index));

    return py::reinterpret_steal<py::object>(result);
}

// --- Fire-and-forget ops counter ---

static std::atomic<int64_t> g_ops_since_last_sync{0};

void increment_ops_counter() {
    g_ops_since_last_sync.fetch_add(1, std::memory_order_relaxed);
}

int64_t get_ops_counter() {
    return g_ops_since_last_sync.load(std::memory_order_relaxed);
}

int64_t reset_ops_counter() {
    return g_ops_since_last_sync.exchange(0, std::memory_order_relaxed);
}

}  // namespace skytorch
