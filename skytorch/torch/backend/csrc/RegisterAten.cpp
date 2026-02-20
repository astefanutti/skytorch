/**
 * SkyTorch PyTorch Backend - ATen Operation Registration
 *
 * This module registers all C++ ATen operation implementations with
 * PyTorch's dispatch system using TORCH_LIBRARY_IMPL.
 *
 * There are three categories of registrations:
 *
 * 1. Native C++ implementations (empty, view, etc.): Operations that MUST
 *    be in C++ to avoid infinite recursion or to preserve custom TensorImpl.
 *
 * 2. Boxed fallback registrations: Operations that need specific PrivateUse1
 *    dispatch to override CompositeExplicitAutograd (CEA) decompositions.
 *    These use fallback_kernel as a boxed function, routing through the C++
 *    dispatch_cached_aten fast path (cache hits handled without Python).
 *
 * 3. Backend fallback: Catches all remaining PrivateUse1 ops not specifically
 *    registered above.
 */

#include <torch/library.h>

#include "RequestBuilder.h"
#include "Resize.h"
#include "Set.h"

namespace skytorch {

// Forward declarations for functions defined in other files
at::Tensor empty(
    at::IntArrayRef size,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<at::MemoryFormat> memory_format);

at::Tensor empty_strided(
    at::IntArrayRef size,
    at::IntArrayRef stride,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory);

at::Tensor as_strided(
    const at::Tensor& self,
    at::IntArrayRef size,
    at::IntArrayRef stride,
    c10::optional<int64_t> storage_offset);

at::Tensor view(const at::Tensor& self, at::IntArrayRef size);
at::Tensor _unsafe_view(const at::Tensor& self, at::IntArrayRef size);
at::Tensor alias(const at::Tensor& self);
at::Tensor _reshape_alias(
    const at::Tensor& self,
    at::IntArrayRef size,
    at::IntArrayRef stride);
at::Tensor _lazy_clone(const at::Tensor& self);

at::Tensor t(const at::Tensor& self);
at::Tensor transpose_int(const at::Tensor& self, int64_t dim0, int64_t dim1);
at::Tensor permute(const at::Tensor& self, at::IntArrayRef dims);
at::Tensor expand(
    const at::Tensor& self,
    at::IntArrayRef sizes,
    bool implicit);
at::Tensor squeeze_dim(const at::Tensor& self, int64_t dim);
at::Tensor squeeze_dims(const at::Tensor& self, at::IntArrayRef dims);
at::Tensor unsqueeze(const at::Tensor& self, int64_t dim);
at::Tensor select_int(const at::Tensor& self, int64_t dim, int64_t index);
at::Tensor slice_tensor(
    const at::Tensor& self,
    int64_t dim,
    c10::optional<int64_t> start,
    c10::optional<int64_t> end,
    int64_t step);

// Register the C++ implementations directly with PyTorch's dispatch system
// These override the Python fallback for these specific operations
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    // Empty tensor creation - these MUST be in C++ to avoid infinite recursion
    // When the Python fallback tries to create output tensors, it would call
    // torch.empty_strided which dispatches back to the fallback, causing recursion.
    m.impl("empty.memory_format", empty);
    m.impl("empty_strided", empty_strided);

    // View operations - implemented in C++ to preserve TensorImpl
    // Without these, view operations would create generic TensorImpl instead
    // of our custom TensorImpl, losing storage ID tracking.
    m.impl("view", view);
    m.impl("as_strided", as_strided);
    m.impl("_unsafe_view", _unsafe_view);
    m.impl("_reshape_alias", _reshape_alias);

    // Set operations for tensor/storage aliasing
    m.impl("set_.source_Tensor", set_source_tensor);
    m.impl("set_.source_Storage", set_source_storage);
    m.impl("set_.source_Storage_storage_offset", set_tensor);

    // Resize with custom hook support
    m.impl("resize_", resize_);

    // Alias and clone operations
    m.impl("alias", alias);
    m.impl("_lazy_clone", _lazy_clone);

    // View/shape operations - purely metadata, no gRPC needed
    m.impl("t", t);
    m.impl("transpose.int", transpose_int);
    m.impl("permute", permute);
    m.impl("expand", expand);
    m.impl("squeeze.dim", squeeze_dim);
    m.impl("squeeze.dims", squeeze_dims);
    m.impl("unsqueeze", unsqueeze);
    m.impl("select.int", select_int);
    m.impl("slice.Tensor", slice_tensor);

    // Boxed fallback registrations — route through fallback_kernel which
    // tries dispatch_cached_aten (C++ fast path) before falling back to Python.
    // These specific registrations are needed to override CompositeExplicitAutograd
    // decompositions (CEA has higher priority than fallback in dispatch).
    // Previously registered in Python via ops.py; now fully in C++ to eliminate
    // Python function call overhead on cache hits (~98.5% of ops).
    static const char* boxed_ops[] = {
        "_batch_norm_no_update",
        "_grouped_mm",
        "_safe_softmax",
        "_adaptive_avg_pool2d",
        "_adaptive_avg_pool2d.out",
        "_adaptive_avg_pool2d_backward",
        "_adaptive_avg_pool2d_backward.out",
        "_log_softmax",
        "_log_softmax.out",
        "_log_softmax_backward_data",
        "_log_softmax_backward_data.out",
        "_native_batch_norm_legit",
        "_native_batch_norm_legit.no_stats",
        "_native_batch_norm_legit.no_stats_out",
        "_native_batch_norm_legit.out",
        "_native_batch_norm_legit_no_training",
        "_native_batch_norm_legit_no_training.out",
        "_softmax",
        "_softmax.out",
        "_softmax_backward_data",
        "_softmax_backward_data.out",
        "abs",
        "abs.out",
        "abs_",
        "add.Scalar",
        "add.Tensor",
        "add.out",
        "add_.Scalar",
        "add_.Tensor",
        "addmm",
        "addmm.out",
        "addmm_",
        "arange",
        "arange.out",
        "arange.start",
        "arange.start_out",
        "arange.start_step",
        "argmax",
        "argmax.out",
        "avg_pool2d",
        "avg_pool2d.out",
        "avg_pool2d_backward",
        "avg_pool2d_backward.grad_input",
        "bmm",
        "bmm.out",
        "cat",
        "cat.out",
        "clone",
        "convolution",
        "convolution.out",
        "convolution_backward",
        "convolution_backward.out",
        "cos",
        "cos.out",
        "cumsum",
        "cumsum.out",
        "div.Scalar",
        "div.Scalar_mode",
        "div.Tensor",
        "div.Tensor_mode",
        "div.out",
        "div_.Scalar",
        "div_.Tensor",
        "embedding",
        "embedding_dense_backward",
        "eq.Scalar",
        "eq.Scalar_out",
        "eq.Tensor",
        "eq.Tensor_out",
        "exp",
        "exp.out",
        "exp_",
        "fill_.Scalar",
        "fill_.Tensor",
        "full",
        "full.out",
        "full_like",
        "gather",
        "gather.out",
        "ge.Scalar",
        "ge.Tensor",
        "gelu",
        "gelu.out",
        "gelu_",
        "gelu_backward",
        "gelu_backward.grad_input",
        "gt.Scalar",
        "gt.Tensor",
        "index.Tensor",
        "index.Tensor_out",
        "index_put",
        "index_put_",
        "isinf",
        "isinf.out",
        "le.Scalar",
        "le.Tensor",
        "log",
        "log.out",
        "log_",
        "logical_and",
        "logical_and.out",
        "logical_and_",
        "logical_not",
        "logical_not.out",
        "logical_not_",
        "logical_or",
        "logical_or.out",
        "logical_or_",
        "lt.Scalar",
        "lt.Tensor",
        "masked_fill.Scalar",
        "masked_fill.Tensor",
        "masked_fill_.Scalar",
        "masked_fill_.Tensor",
        "max",
        "max.dim",
        "max.dim_max",
        "max_pool2d_with_indices",
        "max_pool2d_with_indices.out",
        "max_pool2d_with_indices_backward",
        "max_pool2d_with_indices_backward.grad_input",
        "mean",
        "mean.dim",
        "mean.out",
        "min",
        "min.dim",
        "min.dim_min",
        "mm",
        "mm.out",
        "mul.Scalar",
        "mul.Tensor",
        "mul.out",
        "mul_.Scalar",
        "mul_.Tensor",
        "multinomial",
        "multinomial.out",
        "native_batch_norm_backward",
        "native_group_norm",
        "native_layer_norm",
        "native_dropout.out",
        "native_dropout_backward",
        "native_dropout_backward.out",
        "ne.Scalar",
        "ne.Tensor",
        "neg",
        "neg.out",
        "neg_",
        "nll_loss2d_backward",
        "nll_loss2d_backward.grad_input",
        "nll_loss2d_forward",
        "nll_loss2d_forward.output",
        "nll_loss_backward",
        "nll_loss_backward.grad_input",
        "nll_loss_forward",
        "nll_loss_forward.output",
        "ones",
        "ones.out",
        "ones_like",
        "pow.Scalar",
        "pow.Scalar_out",
        "pow.Tensor_Scalar",
        "pow.Tensor_Scalar_out",
        "pow.Tensor_Tensor",
        "pow.Tensor_Tensor_out",
        "pow_.Scalar",
        "prod",
        "prod.dim_int",
        "prod.int_out",
        "relu",
        "relu.out",
        "relu_",
        "repeat_interleave.self_int",
        "rsqrt",
        "rsqrt.out",
        "rsqrt_",
        "scalar_tensor",
        "scalar_tensor.out",
        "scatter.src",
        "scatter.value",
        "scatter_.src",
        "scatter_.value",
        "scatter_add",
        "scatter_add.out",
        "scatter_add_",
        "sigmoid",
        "sigmoid.out",
        "sigmoid_",
        "sigmoid_backward",
        "sigmoid_backward.grad_input",
        "silu",
        "silu.out",
        "silu_",
        "silu_backward",
        "silu_backward.grad_input",
        "sin",
        "sin.out",
        "sort",
        "sort.values",
        "sort.values_stable",
        "sqrt",
        "sqrt.out",
        "sqrt_",
        "stack",
        "stack.out",
        "sub.Scalar",
        "sub.Tensor",
        "sub.out",
        "sub_.Scalar",
        "sub_.Tensor",
        "sum",
        "sum.dim_IntList",
        "sum.out",
        "tanh",
        "tanh.out",
        "tanh_",
        "tanh_backward",
        "tanh_backward.grad_input",
        "threshold_backward",
        "threshold_backward.grad_input",
        "topk",
        "topk.values",
        "tril",
        "tril.out",
        "triu",
        "triu.out",
        "where.self",
        "where.self_out",
        "zero_",
        "zeros",
        "zeros.out",
        "zeros_like",
    };
    for (const char* op : boxed_ops) {
        m.impl(op, torch::CppFunction::makeFromBoxedFunction<&fallback_kernel>());
    }
}

}  // namespace skytorch

// C++ autograd fallback — redispatches below autograd to PrivateUse1.
// Uses op.callBoxed() to stay in C++ dispatch and avoid CompositeImplicitAutograd
// decompositions (e.g., convolution → convolution_overrideable).
TORCH_LIBRARY_IMPL(_, AutogradPrivateUse1, m) {
    m.fallback(torch::CppFunction::makeFromBoxedFunction<&skytorch::autograd_fallback_kernel>());
}

// C++ backend fallback — handles all PrivateUse1 ops not specifically registered.
// Cache hits are handled entirely in C++; misses fall through to Python.
TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
    m.fallback(torch::CppFunction::makeFromBoxedFunction<&skytorch::fallback_kernel>());
}
