/**
 * KPU PyTorch Backend - Resize Operations
 *
 * Declarations for tensor resize operations.
 */

#pragma once

#include <ATen/Tensor.h>
#include <c10/util/Optional.h>

namespace kpu {

/**
 * Resize a KPU tensor, calling storage resize hooks as needed.
 *
 * If the new size requires more storage than currently available,
 * this will trigger the resizePrivateUse1Bytes hook to expand storage.
 */
const at::Tensor& resize_kpu_(
    const at::Tensor& self,
    at::IntArrayRef size,
    c10::optional<at::MemoryFormat> memory_format);

}  // namespace kpu
