//
// Created by UshioHayase on 2026-03-18.
//
#include "function/add.h"

#include "kernel/cpu/add_cpu.h"
#include "kernel/gpu/add_gpu.h"
#include "utils/log_macro.h"

ushionn::Tensor ushionn::function::Add::forward(const Tensor& a,
                                                const Tensor& b)
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

    Tensor result(a.shape(), device, a.dtype());

    if (device.type == Device::DeviceType::HOST)
    {
        cpu::add_kernel(result, a, b);
    }
    else
    {
        gpu::add_kernel(result, a, b);
    }

    return result;
}

void ushionn::function::Add::forward(Tensor& result, const Tensor& a,
                                     const Tensor& b)
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

    if (device.type == Device::DeviceType::HOST)
    {
        cpu::add_kernel(result, a, b);
    }
    else
    {
        gpu::add_kernel(result, a, b);
    }
}
