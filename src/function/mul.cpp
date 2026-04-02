//
// Created by UshioHayase on 2026-04-02.
//
//
// Created by UshioHayase on 2026-03-18.
//
#include "function/mul.h"

#include "kernel/cpu/mul_cpu.h"
#include "kernel/gpu/mul_gpu.h"
#include "utils/log_macro.h"

ushionn::Tensor ushionn::function::Mul::forward(const Tensor& a, float b)
{
    ASSERT_MESSAGE(a.device().type != Device::DeviceType::NONE,
                   "The two tensors are of different types.");

    const Device& device = a.device();

    Tensor result(a.shape(), device, a.dtype());

    if (device.type == Device::DeviceType::HOST)
    {
        cpu::scalar_mul_kernel(result, a, b);
    }
    else
    {
        gpu::scalar_mul_kernel(result, a, b);
    }

    return result;
}

void ushionn::function::Mul::forward(Tensor& result, const Tensor& a, float b)
{

    ASSERT_MESSAGE(a.device().type != Device::DeviceType::NONE,
                   "Tensor not assigned.");

    const Device& device = a.device();

    if (device.type == Device::DeviceType::HOST)
    {
        cpu::scalar_mul_kernel(result, a, b);
    }
    else
    {
        gpu::scalar_mul_kernel(result, a, b);
    }
}