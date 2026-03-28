//
// Created by UshioHayase on 2026-03-22.
//

#include "kernel/cpu/matmul_cpu.h"

#include "utils/log_macro.h"

namespace ushionn::cpu
{
void matmul_kernel(Tensor& result, const Tensor& a, const Tensor& b)
{
    Tensor result_contiguous =
        result.is_contiguous() ? result : result.contiguous();
    Tensor a_contiguous = a.is_contiguous() ? a : a.contiguous();
    Tensor b_contiguous = b.is_contiguous() ? b : b.contiguous();

    const DType type = result_contiguous.dtype();
    const uint32_t a_dim = a_contiguous.dim();
    const uint32_t b_dim = b_contiguous.dim();

    const auto& shape = result_contiguous.shape();

    switch (type)
    {
    case DType::FP64: {
    }
    }
}
} // namespace ushionn::cpu
