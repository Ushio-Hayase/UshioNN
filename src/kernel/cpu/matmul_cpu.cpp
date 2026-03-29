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

    const auto& a_shape = a_contiguous.shape();
    const auto& b_shape = b_contiguous.shape();
    const auto& result_shape = result_contiguous.shape();

    switch (type)
    {
    case DType::FP64: {
        ei double* result_data = result_contiguous.data_ptr<double>();
        double* a_data = a_contiguous.data_ptr<double>();
        double* b_data = b_contiguous.data_ptr<double>();

        for (uint32_t i = 0; i < a_dim; ++i)
            for (uint64_t j = 0; j < a_shape[i]; ++j)
                for (uint64_t k = 0; k < b_shape[b_dim - 1]; ++k)
                    result_data[j]
    }
    }
}
} // namespace ushionn::cpu
