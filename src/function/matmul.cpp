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

        Tensor result({b_shape[1]}, a.dtype(), device);
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
        Tensor result({a_shape[0]}, a.dtype(), device);
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
    else if (a_dim > b_dim)
    {
        std::vector<uint64_t> expand_b_dim(a_dim);
        for (int i = 0; i < a_dim - b_dim; ++i)
            expand_b_dim[i] = 1;
        expand_b_dim.insert(b_shape.begin(), b_dim);

        for (int i = 0; i < a_dim - 2; ++i)
        {
            if (a_shape[i] == expand_b_dim[i] || expand_b_dim[i] == 1)
            {
                std::string error_msg =
                    "Tensor a and Tensor b shapess cannot be multiplied (";
                for (int i = 0; i < a_dim; ++i)
                {
                    error_msg += std::to_string(a_shape[i]);
                    if (i != a_dim - 1)
                        error_msg += " x ";
                }
                error_msg += " and ";
                for (int i = 0; i < a_dim; ++i)
                {
                    error_msg += std::to_string(a_shape[i]);
                    if (i != a_dim - 1)
                        error_msg += " x ";
                }
                error_msg += ")";
                LOG_ERROR(error_msg);
            }
        }

        std::vector<uint64_t> result_shape(a_dim);
        result_shape[a_dim - 1] = expand_b_dim.back();
        Tensor result(result_shape, a.dtype(), device);

        Tensor expand_b(b, expand_b_dim, )

            if (device.type == Device::DeviceType::HOST)
                cpu::matmul_kernel(result, a, b);
    }
    else if (a_dim < b_dim)
    {
    }
}

} // namespace ushionn::function
