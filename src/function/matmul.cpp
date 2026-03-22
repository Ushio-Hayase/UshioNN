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
    const uint32_t a_dim = a.dim();
    const uint32_t b_dim = b.dim();

    const auto& a_shape = a.shape();
    const auto& b_shape = b.shape();

    if (a_dim == 1 && b_dim == 1)
    {
        Tensor result({1}, a.dtype(), device);
        if (device.type == Device::DeviceType::HOST)
            cpu::matmul_kernel(result, a, b);
        else
            gpu::matmul_kernel(result, a, b);
        return result;
    }
    else if (a_dim == 1 && b_dim == 2)
    {

        ASSERT_MESSAGE(a_shape[0] == b_shape[0],
                       "Tensor a and Tensor b shapes cannot be multiplied ({} "
                       "x {} and {} x {})",
                       1, a_shape[0], b_shape[0], b_shape[1]);

        Tensor result({1, b_shape[1]}, a.dtype(), device);
        if (device.type == Device::DeviceType::HOST)
            cpu::matmul_kernel(result, a, b);
        else
            gpu::matmul_kernel(result, a, b);
        return result;
    }
    else if (a_dim == 1 && b_dim == 3)
    {
        ASSERT_MESSAGE(a_shape[1] == b_shape[1],
                       "Tensor a and Tensor b shapes cannot be multiplied ({} "
                       "x {} x {} and {} x {} x {})",
                       b_shape[0], 1, a_shape[0], b_shape[0], b_shape[1],
                       b_shape[2]);
        Tensor result({b_shape[0], b_shape[2]}, a.dtype(), device);
        if (device.type == Device::DeviceType::HOST)
            cpu::matmul_kernel(result, a, b);
        else
            gpu::matmul_kernel(result, a, b);
        return result;
    }
    else if (a_dim == 2 && b_dim == 1)
    {
        ASSERT_MESSAGE(a_shape[1] == b_shape[0],
                       "Tensor a and Tensor b shapes cannot be multiplied ({} "
                       "x {} and {} x {})",
                       a_shape[0], a_shape[1], b_shape[0], 1);
        Tensor result({a_shape[0], 1}, a.dtype(), device);
        if (device.type == Device::DeviceType::HOST)
            cpu::matmul_kernel(result, a, b);
        else
            gpu::matmul_kernel(result, a, b);
        return result;
    }
    else if (a_dim == 2 && b_dim == 2)
    {
        ASSERT_MESSAGE(a_shape[1] == b_shape[0],
                       "Tensor a and Tensor b shapes cannot be multiplied ({} "
                       "x {} and {} x {})",
                       a_shape[0], a_shape[1], b_shape[0], b_shape[1]);
        Tensor result({a_shape[0], b_shape[1]}, a.dtype(), device);
        if (device.type == Device::DeviceType::HOST)
            cpu::matmul_kernel(result, a, b);
        else
            gpu::matmul_kernel(result, a, b);
        return result;
    }
    else if (a_dim == 2 && b_dim == 3)
    {
        ASSERT_MESSAGE(a_shape[1] == b_shape[1],
                       "Tensor a and Tensor b shapes cannot be multiplied ({} "
                       "x {} x {} and {} x {} x {})",
                       1, a_shape[0], a_shape[1], b_shape[0], b_shape[1],
                       b_shape[2]);
        Tensor result({b_shape[0], a_shape[0], b_shape[2]}, a.dtype(), device);
        if (device.type == Device::DeviceType::HOST)
            cpu::matmul_kernel(result, a, b);
        else
            gpu::matmul_kernel(result, a, b);
        return result;
    }
    else if (a_dim == 3 && b_dim == 2)
    {
        ASSERT_MESSAGE(a_shape[1] == b_shape[0],
                       "Tensor a and Tensor b shapes cannot be multiplied ({} "
                       "x {} x {} and {} x {} x {})",
                       a_shape[0], a_shape[1], a_shape[2], 1, b_shape[0],
                       b_shape[1]);
        Tensor result({a_shape[0], a_shape[1], b_shape[1]}, a.dtype(), device);
        if (device.type == Device::DeviceType::HOST)
            cpu::matmul_kernel(result, a, b);
        else
            gpu::matmul_kernel(result, a, b);
        return result;
    }
    
    return Tensor();
}

} // namespace ushionn::function
