//
// Created by UshioHayase on 2026-03-18.
//

#include "function/elementwise_mul.h"

#include "kernel/cpu/elementwise_mul_cpu.h"
#include "kernel/gpu/elementwise_mul_gpu.h"
#include "utils/log_macro.h"

#include <chrono>

namespace ushionn
{
Tensor function::ElementWiseMul::forward(const Tensor& a, const Tensor& b)
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
        cpu::elementwise_mul_kernel(result, a, b);
    }
    else
    {
        gpu::elementwise_mul_kernel(result, a, b);
    }

    return result;
}

} // namespace ushionn
