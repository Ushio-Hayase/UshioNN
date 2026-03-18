//
// Created by UshioHayase on 2026-03-18.
//
#include "kernel/gpu/mul_gpu.h"
#include "utils/constant.h"
#include "utils/log_macro.h"

#include "cuda/cuda_utils.cuh"

namespace ushionn::gpu
{

template <ScalarType T>
static void __global__ scalar_mul(T* dst, const T* src, const float scalar,
                                  const size_t total_elements)
{
    const size_t idx = cuda::utils::get_global_idx();
    const size_t strides = blockDim.x * blockDim.y * blockDim.z * gridDim.x *
                           gridDim.y * gridDim.z;

    for (size_t i = idx; i < total_elements; i += strides)
        dst[idx] = src[idx] * scalar;
}

void scalar_mul_kernel(Tensor& result, const Tensor& src, const float scalar)
{
    ASSERT_MESSAGE(result.device().type != Device::DeviceType::NONE,
                   "Tensor not assigned.");
    ASSERT_MESSAGE(src.device().type != Device::DeviceType::NONE,
                   "Tensor not assigned.");
    ASSERT_MESSAGE(result.dtype() == src.dtype(),
                   "The two tensors are of different types.");
    ASSERT_MESSAGE(result.shape() == src.shape(),
                   "Two tensors have different sizes.");
    ASSERT_MESSAGE(result.device().type == src.device().type,
                   "Both tensors must be in the same device.");

    const size_t grid_size = (result.numel() * BLOCK_SIZE - 1) / BLOCK_SIZE;
    const size_t total_elem = result.numel();

    switch (result.dtype())
    {
    case DType::FP64: {
        scalar_mul<double><<<grid_size, BLOCK_SIZE>>>(result.data_ptr<double>(),
                                                      src.data_ptr<double>(),
                                                      scalar, total_elem);
        break;
    }
    case DType::FP32: {
        scalar_mul<float><<<grid_size, BLOCK_SIZE>>>(result.data_ptr<float>(),
                                                     src.data_ptr<float>(),
                                                     scalar, total_elem);
        break;
    }
    case DType::FP16: {
        scalar_mul<fp16_t><<<grid_size, BLOCK_SIZE>>>(result.data_ptr<fp16_t>(),
                                                      src.data_ptr<fp16_t>(),
                                                      scalar, total_elem);
        break;
    }
    case DType::BF16: {
        scalar_mul<bf16_t><<<grid_size, BLOCK_SIZE>>>(result.data_ptr<bf16_t>(),
                                                      src.data_ptr<bf16_t>(),
                                                      scalar, total_elem);
        break;
    }
    case DType::FP8_e4m3: {
        scalar_mul<fp8_e4m3_t><<<grid_size, BLOCK_SIZE>>>(
            result.data_ptr<fp8_e4m3_t>(), src.data_ptr<fp8_e4m3_t>(), scalar,
            total_elem);
        break;
    }
    case DType::FP8_e5m2: {
        scalar_mul<fp8_e5m2_t><<<grid_size, BLOCK_SIZE>>>(
            result.data_ptr<fp8_e5m2_t>(), src.data_ptr<fp8_e5m2_t>(), scalar,
            total_elem);
        break;
    }
    case DType::FP4: {
        scalar_mul<fp4_t><<<grid_size, BLOCK_SIZE>>>(result.data_ptr<fp4_t>(),
                                                     src.data_ptr<fp4_t>(),
                                                     scalar, total_elem);
        break;
    }
    }
}

} // namespace ushionn::gpu