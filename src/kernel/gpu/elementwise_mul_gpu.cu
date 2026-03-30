//
// Created by UshioHayase on 2026-03-22.
//

#include "core/tensor.h"
#include "core/type.h"
#include "kernel/gpu/elementwise_mul_gpu.h"
#include "utils/constant.h"
#include "utils/log_macro.h"

#include "cuda/cuda_utils.cuh"

namespace ushionn::gpu
{
template <ScalarType T>
static __global__ void elementwise_mul(T* src, const T* a, const T* b,
                                       const uint64_t total_elements)
{
    const size_t idx = cuda::utils::get_global_idx();
    const size_t strides = cuda::utils::get_strides();

    for (size_t i = idx; i < total_elements; i += strides)
        src[idx] = a[idx] * b[idx];
}

void elementwise_mul_kernel(Tensor& result, const Tensor& a, const Tensor& b)
{
    ASSERT_MESSAGE(result.device().type != Device::DeviceType::NONE,
                   "Tensor not assigned.");
    ASSERT_MESSAGE(a.device().type != Device::DeviceType::NONE,
                   "Tensor not assigned.");
    ASSERT_MESSAGE(b.device().type != Device::DeviceType::NONE,
                   "Tensor not assigned.");
    ASSERT_MESSAGE(result.dtype() == a.dtype(),
                   "The two tensors are of different types.");
    ASSERT_MESSAGE(a.dtype() == b.dtype(),
                   "The two tensors are of different types.");
    ASSERT_MESSAGE(result.shape() == a.shape(),
                   "Two tensors have different sizes.");
    ASSERT_MESSAGE(a.shape() == b.shape(), "Two tensors have different sizes.");
    ASSERT_MESSAGE(result.device().type == a.device().type,
                   "Both tensors must be in the same device.");
    ASSERT_MESSAGE(a.device().type == b.device().type,
                   "Both tensors must be in the same device.");

    const size_t grid_size = (result.numel() * BLOCK_SIZE - 1) / BLOCK_SIZE;
    const size_t total_elem = result.numel();

    switch (result.dtype())
    {
    case DType::FP64: {
        elementwise_mul(result.data_ptr<double>(), a.data_ptr<double>(),
                        b.data_ptr<double>(), total_elem);
        break;
    }
    case DType::FP32: {
        elementwise_mul<float><<<grid_size, BLOCK_SIZE>>>(
            result.data_ptr<float>(), a.data_ptr<float>(), b.data_ptr<float>(),
            total_elem);
        break;
    }
    case DType::FP16: {
        elementwise_mul<fp16_t><<<grid_size, BLOCK_SIZE>>>(
            result.data_ptr<fp16_t>(), a.data_ptr<fp16_t>(),
            b.data_ptr<fp16_t>(), total_elem);
        break;
    }
    case DType::BF16: {
        elementwise_mul<bf16_t><<<grid_size, BLOCK_SIZE>>>(
            result.data_ptr<bf16_t>(), a.data_ptr<bf16_t>(),
            b.data_ptr<bf16_t>(), total_elem);
        break;
    }
    case DType::FP8_e4m3: {
        elementwise_mul<fp8_e4m3_t><<<grid_size, BLOCK_SIZE>>>(
            result.data_ptr<fp8_e4m3_t>(), a.data_ptr<fp8_e4m3_t>(),
            b.data_ptr<fp8_e4m3_t>(), total_elem);
        break;
    }
    case DType::FP8_e5m2: {
        elementwise_mul<fp8_e5m2_t><<<grid_size, BLOCK_SIZE>>>(
            result.data_ptr<fp8_e5m2_t>(), a.data_ptr<fp8_e5m2_t>(),
            b.data_ptr<fp8_e5m2_t>(), total_elem);
        break;
    }
    case DType::FP4: {
        elementwise_mul<fp4_t><<<grid_size, BLOCK_SIZE>>>(
            result.data_ptr<fp4_t>(), a.data_ptr<fp4_t>(), b.data_ptr<fp4_t>(),
            total_elem);
        break;
    }
    }
}

} // namespace ushionn::gpu
