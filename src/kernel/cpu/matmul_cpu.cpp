//
// Created by UshioHayase on 2026-03-22.
//

#include "kernel/cpu/matmul_cpu.h"

#include "utils/log_macro.h"

namespace ushionn::cpu
{
void matmul_kernel(Tensor& result, const Tensor& a, const Tensor& b)
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
    ASSERT_MESSAGE(result.device().type == a.device().type,
                   "Both tensors must be in the same device.");
    ASSERT_MESSAGE(a.device().type == b.device().type,
                   "Both tensors must be in the same device.");
    Tensor result_contiguous =
        result.is_contiguous() ? result : result.contiguous();
    Tensor a_contiguous = a.is_contiguous() ? a : a.contiguous();
    Tensor b_contiguous = b.is_contiguous() ? b : b.contiguous();

    DType type = result.dtype();
    int dim = a.dim();

    if (dim == 1)
    {
    }
}
} // namespace ushionn::cpu
