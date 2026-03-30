#include "core/type.h"
#include "kernel/gpu/add_gpu.h"
#include "utils/constant.h"
#include "utils/log_macro.h"

#include "cuda/cuda_utils.cuh"

namespace ushionn::gpu
{
template <ushionn::ScalarType T>
static __global__ void add(T* src, const T* tensor1, const T* tensor2,
                           const uint64_t total_elements)
{

    const size_t idx = cuda::utils::get_global_idx();
    const size_t strides = cuda::utils::get_strides();

    for (size_t i = idx; i < total_elements; i += strides)
        src[idx] = tensor1[idx] + tensor2[idx];
}

void add_kernel(Tensor& result, const Tensor& tensor1, const Tensor& tensor2)
{
    ASSERT_MESSAGE(result.device().type != Device::DeviceType::NONE,
                   "Tensor not assigned.");
    ASSERT_MESSAGE(tensor1.device().type != Device::DeviceType::NONE,
                   "Tensor not assigned.");
    ASSERT_MESSAGE(tensor2.device().type != Device::DeviceType::NONE,
                   "Tensor not assigned.");
    ASSERT_MESSAGE(result.dtype() == tensor1.dtype(),
                   "The two tensors are of different types.");
    ASSERT_MESSAGE(tensor1.dtype() == tensor2.dtype(),
                   "The two tensors are of different types.");
    ASSERT_MESSAGE(result.shape() == tensor1.shape(),
                   "Two tensors have different sizes.");
    ASSERT_MESSAGE(tensor1.shape() == tensor2.shape(),
                   "Two tensors have different sizes.");
    ASSERT_MESSAGE(result.device().type == tensor1.device().type,
                   "Both tensors must be in the same device.");
    ASSERT_MESSAGE(tensor1.device().type == tensor2.device().type,
                   "Both tensors must be in the same device.");

    const uint64_t grid_size = (result.numel() * BLOCK_SIZE - 1) / BLOCK_SIZE;
    const uint64_t total_elem = result.numel();

    switch (result.dtype())
    {
    case DType::FP64: {
        add<double><<<grid_size, BLOCK_SIZE>>>(
            result.data_ptr<double>(), tensor1.data_ptr<double>(),
            tensor2.data_ptr<double>(), total_elem);
        break;
    }
    case DType::FP32: {
        add<float><<<grid_size, BLOCK_SIZE>>>(
            result.data_ptr<float>(), tensor1.data_ptr<float>(),
            tensor2.data_ptr<float>(), total_elem);
        break;
    }
    case DType::FP16: {
        add<fp16_t><<<grid_size, BLOCK_SIZE>>>(
            result.data_ptr<fp16_t>(), tensor1.data_ptr<fp16_t>(),
            tensor2.data_ptr<fp16_t>(), total_elem);
        break;
    }
    case DType::BF16: {
        add<bf16_t><<<grid_size, BLOCK_SIZE>>>(
            result.data_ptr<bf16_t>(), tensor1.data_ptr<bf16_t>(),
            tensor2.data_ptr<bf16_t>(), total_elem);
        break;
    }
    case DType::FP8_e4m3: {
        add<fp8_e4m3_t><<<grid_size, BLOCK_SIZE>>>(
            result.data_ptr<fp8_e4m3_t>(), tensor1.data_ptr<fp8_e4m3_t>(),
            tensor2.data_ptr<fp8_e4m3_t>(), total_elem);
        break;
    }
    case DType::FP8_e5m2: {
        add<fp8_e5m2_t><<<grid_size, BLOCK_SIZE>>>(
            result.data_ptr<fp8_e5m2_t>(), tensor1.data_ptr<fp8_e5m2_t>(),
            tensor2.data_ptr<fp8_e5m2_t>(), total_elem);
        break;
    }
    case DType::FP4: {
        add<fp4_t><<<grid_size, BLOCK_SIZE>>>(
            result.data_ptr<fp4_t>(), tensor1.data_ptr<fp4_t>(),
            tensor2.data_ptr<fp4_t>(), total_elem);
        break;
    }
    }

    cudaDeviceSynchronize();
}
} // namespace ushionn::gpu
