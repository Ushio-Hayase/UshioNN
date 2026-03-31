//
// Created by UshioHayase on 2026-03-22.
//

#include "kernel/cpu/matmul_cpu.h"

#include "utils/log_macro.h"

namespace ushionn::cpu
{
void matmul_kernel(Tensor& result, const Tensor& a, const Tensor& b)
{
    Tensor a_contiguous = a.is_contiguous() ? a : a.contiguous();
    Tensor b_contiguous = b.is_contiguous() ? b : b.contiguous();

    const DType type = result.dtype();
    const uint32_t a_dim = a_contiguous.dim();
    const uint32_t b_dim = b_contiguous.dim();
    const uint32_t result_dim = result.dim();

    const auto& a_shape = a_contiguous.shape();
    const auto& b_shape = b_contiguous.shape();
    const auto& result_shape = result.shape();

    const uint64_t k_size = a_shape.back();
    const uint64_t i_size = a_shape[a_dim - 2];
    const uint64_t j_size = b_shape[b_dim - 1];

    const auto& a_strides = a_contiguous.strides();
    const auto& b_strides = b_contiguous.strides();
    const auto& result_strides = result.strides();

    result.zero();

    // TODO : 현재 SISD를 SIMD로 구현 필요
    switch (type)
    {
    case DType::FP64: {

        double* result_data = result.data_ptr<double>();
        double* a_data = a_contiguous.data_ptr<double>();
        double* b_data = b_contiguous.data_ptr<double>();

        uint64_t batch_offset = 0;
        uint64_t batch_strides = 1;

        for (int batch = result_dim - 2; batch >= 0; --batch)
        {
            batch_strides *= result_shape[batch];
            for (int b = 0; b < result_shape[batch]; ++b)
            {
                batch_offset += batch_strides;
                for (int i = 0; i < i_size; ++i)
                {
                    for (int k = 0; k < k_size; ++k)
                    {
                        const double r =
                            a_data[batch_offset + (i * k_size + k)];
                        for (int64_t j = 0; j < j_size; ++j)
                        {

                            result_data[batch_offset + (i * j_size + j)] +=
                                r * b_data[batch_offset + (k * i_size + j)];
                        }
                    }
                }
            }
        }
    }
    case DType::FP32: {

        float* result_data = result.data_ptr<float>();
        float* a_data = a_contiguous.data_ptr<float>();
        float* b_data = b_contiguous.data_ptr<float>();

        uint64_t batch_offset = 0;
        uint64_t batch_strides = 1;

        for (int batch = result_dim - 2; batch >= 0; --batch)
        {
            batch_strides *= result_shape[batch];
            for (uint64_t b = 0; b < result_shape[batch]; ++b)
            {
                batch_offset += batch_strides;
                for (int64_t i = 0; i < i_size; ++i)
                {
                    for (int64_t k = 0; k < k_size; ++k)

                    {
                        const double r =
                            a_data[batch_offset + (i * k_size + k)];
                        for (int64_t j = 0; j < j_size; ++j)
                        {

                            result_data[batch_offset + (i * j_size + j)] +=
                                r * b_data[batch_offset + (k * i_size + j)];
                        }
                    }
                }
            }
        }
    }
        // TODO: 텐서 행렬곱 FP4 ~ FP16 구현 필요
    default:
        LOG_ERROR("{} is a data type that is not yet supported.",
                  dtype_to_string(result.dtype()));
    }
}

} // namespace ushionn::cpu
