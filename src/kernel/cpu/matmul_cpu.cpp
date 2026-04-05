//
// Created by UshioHayase on 2026-03-22.
//

#include "kernel/cpu/matmul_cpu.h"

#include "utils/log_macro.h"

namespace ushionn::cpu
{
void matmul_kernel(Tensor& result, const Tensor& a, const Tensor& b)
{
    const Tensor& a_contiguous = a.contiguous();
    const Tensor& b_contiguous = b.contiguous();

    const DType type = result.dtype();
    const uint32_t a_dim = a_contiguous.dim();
    const uint32_t b_dim = b_contiguous.dim();
    const uint32_t result_dim = result.dim();

    const auto& a_shape = a_contiguous.shape();
    const auto& b_shape = b_contiguous.shape();
    const auto& result_shape = result.shape();

    const uint64_t K = a_shape[a_dim - 1];
    const uint64_t M = a_shape[a_dim - 2];
    const uint64_t N = b_shape[b_dim - 1];

    const auto& a_strides = a_contiguous.strides();
    const auto& b_strides = b_contiguous.strides();
    const auto& result_strides = result.strides();

    uint64_t total_batch_size = 1;
    for (int i = 0; i < result_dim - 2; ++i)
    {
        total_batch_size *= result_shape[i];
    }

    // TODO : 현재 SISD를 SIMD로 구현 필요
    switch (type)
    {
    case DType::FP64: {

        double* result_data = result.data_ptr<double>();
        double* a_data = a_contiguous.data_ptr<double>();
        double* b_data = b_contiguous.data_ptr<double>();

        for (int64_t batch = 0; batch < total_batch_size; ++batch)
        {
            for (int64_t i = 0; i < M; ++i)
            {
                for (int64_t k = 0; k < K; ++k)
                {
                    const double r = a_data[batch * a_strides[a_dim - 3] +
                                            i * a_strides[a_dim - 2] +
                                            k * a_strides[a_dim - 1]];
                    for (int64_t j = 0; j < N; ++j)
                    {
                        const uint64_t result_idx =
                            batch * result_strides[result_dim - 3] +
                            i * result_strides[result_dim - 2] +
                            j * result_strides[result_dim - 1];
                        const uint64_t b_idx = batch * b_strides[b_dim - 3] +
                                               k * b_strides[b_dim - 2] +
                                               j * b_strides[b_dim - 1];
                        result_data[result_idx] += r * b_data[b_idx];
                    }
                }
            }
        }
        break;
    }
    case DType::FP32: {

        float* result_data = result.data_ptr<float>();
        float* a_data = a_contiguous.data_ptr<float>();
        float* b_data = b_contiguous.data_ptr<float>();

        for (int64_t batch = 0; batch < total_batch_size; ++batch)
        {
            for (int64_t i = 0; i < M; ++i)
            {
                for (int64_t k = 0; k < K; ++k)
                {
                    const float r = a_data[batch * a_strides[a_dim - 3] +

                                           (i * a_strides[a_dim - 2] +
                                            k * a_strides[a_dim - 1])];
                    for (int64_t j = 0; j < N; ++j)
                    {

                        result_data[batch * result_strides[result_dim - 3] +
                                    (i * result_strides[result_dim - 2]) +
                                    (j * result_strides[result_dim - 1])] +=
                            r * b_data[batch * b_strides[b_dim - 3] +
                                       (k * b_strides[b_dim - 2]) +
                                       (j * b_strides[b_dim - 1])];
                    }
                }
            }
        }
        break;
    }
        // TODO: 텐서 행렬곱 FP4 ~ FP16 구현 필요
    default:
        LOG_ERROR("{} is a data type that is not yet supported.",
                  dtype_to_string(result.dtype()));
    }
}

} // namespace ushionn::cpu
