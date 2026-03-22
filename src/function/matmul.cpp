//
// Created by UshioHayase on 2026-03-22.
//

#include "function/matmul.h"

#include "kernel/cpu/matmul_cpu.h"
#include "kernel/gpu/matmul_gpu.h"
#include "utils/log_macro.h"

namespace ushionn::function
{
Tensor Matmul::forward(const Tensor& a, const Tensor& b)
{
    ASSERT_MESSAGE(a.device().type != Device::DeviceType::NONE,
                   "Tensor not assigned.");
    ASSERT_MESSAGE(b.device().type != Device::DeviceType::NONE,
                   "Tensor not assigned.");
    ASSERT_MESSAGE(a.dtype() == b.dtype(),
                   "The two tensors are of different types.");
    ASSERT_MESSAGE(a.shape() == b.shape(), "Two tensors have different sizes.");
    ASSERT_MESSAGE(a.device().type == b.device().type,
                   "Both tensors must be in the same device.");

    const Device& device = a.device();

    Tensor result(a.shape(), a.dtype(), device);

    if (device.type == Device::DeviceType::HOST)
    {
        cpu::matmul_kernel(result, a, b);
    }
    else
    {
        gpu::matmul_kernel(result, a, b);
    }

    return result;
}

} // namespace ushionn::function
