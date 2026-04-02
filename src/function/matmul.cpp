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
    const auto& a_strides = a.strides();
    const auto& b_strides = b.strides();

    std::vector<uint64_t> a_shape_expand;
    std::vector<uint64_t> b_shape_expand;
    std::vector<uint64_t> a_strides_expand;
    std::vector<uint64_t> b_strides_expand;

    if (a_dim == 1 && b_dim == 1)
    {
        a_shape_expand = {1, 1, 1};
        b_shape_expand = {1, 1, 1};
        a_strides_expand = {0, 0, 1};
        b_strides_expand = {0, 0, 1};
    }
    else if (b_dim == 1)
    {
    }
    else if (a_dim < b_dim)
    {
        b_shape_expand = b_shape;

        a_shape_expand.resize(b_dim - a_dim, 1);
        a_shape_expand.insert(a_shape_expand.end(), a_shape.begin(),
                              a_shape.end());
        a_strides_expand.resize(b_dim - a_dim, 0);
        a_strides_expand.insert(a_strides_expand.end(), a_strides.begin(),
                                a_strides.end());
    }
    else if (a_dim > b_dim)
    {
        a_shape_expand = a_shape;

        b_shape_expand.resize(a_dim - b_dim, 1);
        b_shape_expand.insert(b_shape_expand.end(), b_shape.begin(),
                              b_shape.end());

        b_strides_expand.resize(b_dim - a_dim, 0);
        b_strides_expand.insert(b_strides_expand.end(), b_strides.begin(),
                                b_strides.end());
    }

    Tensor a_expand(a);
    a_expand = a_expand.reshape(a_shape_expand);
    Tensor b_expand(b);
    b_expand = b_expand.reshape(b_shape_expand);

    const std::vector<uint64_t> result_shape =
        calculate_matmul_size(a_shape_expand, b_shape_expand);
    Tensor result(result_shape, device, a.dtype());

    if (device.type == Device::DeviceType::HOST)
        cpu::matmul_kernel(result, a_expand, b_expand);
    else
        gpu::matmul_kernel(result, a_expand, b_expand);

    return result;
}

void Matmul::forward(Tensor& result, const Tensor& a, const Tensor& b)
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
    const auto& a_strides = a.strides();
    const auto& b_strides = b.strides();

    std::vector<uint64_t> a_shape_expand;
    std::vector<uint64_t> b_shape_expand;
    std::vector<uint64_t> a_strides_expand;
    std::vector<uint64_t> b_strides_expand;

    if (a_dim == 1 && b_dim == 1)
    {
        a_shape_expand = {1, 1, 1};
        b_shape_expand = {1, 1, 1};
        a_strides_expand = {0, 0, 1};
        b_strides_expand = {0, 0, 1};
    }
    else if (b_dim == 1)
    {
    }
    else if (a_dim < b_dim)
    {
        b_shape_expand = b_shape;

        a_shape_expand.resize(b_dim - a_dim, 1);
        a_shape_expand.insert(a_shape_expand.end(), a_shape.begin(),
                              a_shape.end());
        a_strides_expand.resize(b_dim - a_dim, 0);
        a_strides_expand.insert(a_strides_expand.end(), a_strides.begin(),
                                a_strides.end());
    }
    else if (a_dim > b_dim)
    {
        a_shape_expand = a_shape;

        b_shape_expand.resize(a_dim - b_dim, 1);
        b_shape_expand.insert(b_shape_expand.end(), b_shape.begin(),
                              b_shape.end());

        b_strides_expand.resize(b_dim - a_dim, 0);
        b_strides_expand.insert(b_strides_expand.end(), b_strides.begin(),
                                b_strides.end());
    }

    Tensor a_expand(a);
    a_expand = a_expand.reshape(a_shape_expand);
    Tensor b_expand(b);
    b_expand = b_expand.reshape(b_shape_expand);

    const std::vector<uint64_t> result_shape =
        calculate_matmul_size(a_shape_expand, b_shape_expand);

    if (device.type == Device::DeviceType::HOST)
        cpu::matmul_kernel(result, a_expand, b_expand);
    else
        gpu::matmul_kernel(result, a_expand, b_expand);
}

std::vector<uint64_t> Matmul::calculate_matmul_size(
    const std::vector<uint64_t>& a, const std::vector<uint64_t>& b)
{
    const auto& a_dim = a.size();
    const auto& b_dim = b.size();

    ASSERT_MESSAGE(
        a_dim == b_dim,
        "The two input tensors must be ranked the same, (a: {}, b: {})", a_dim,
        b_dim);

    bool can_matmul = true;

    if (a_dim == 1 && b_dim == 1)
    {
        return {1};
    }
    else
    {
        if (a[a.size() - 1] == b[b.size() - 2])
        {
            for (int i = 0; i < a_dim - 2; ++i)
            {
                if (a[i] == b[i])
                {
                    can_matmul = false;
                    break;
                };
            }
        }
    }
    if (!can_matmul) [[unlikely]]
    {
        std::string error_msg =
            "Tensor a and Tensor b shapes cannot be multiplied (";
        for (int i = 0; i < a_dim; ++i)
        {
            error_msg += std::to_string(a[i]);
            if (i != a_dim - 1)
                error_msg += " x ";
        }
        for (int i = 0; i < b_dim; ++i)
        {
            error_msg += std::to_string(b[i]);
            if (i != b_dim - 1)
                error_msg += " x ";
        }
        error_msg += ")";
        LOG_ERROR(error_msg);
        assert(0);
    }

    std::vector<uint64_t> result(a);
    result[result.size() - 1] = b.back();

    return result;
}

} // namespace ushionn::function
