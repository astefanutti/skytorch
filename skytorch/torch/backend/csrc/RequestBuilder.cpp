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

}  // namespace skytorch
