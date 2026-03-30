//
// Created by UshioHayase on 2026-03-22.
//

#include "core/type.h"
#include "kernel/gpu/matmul_gpu.h"

namespace ushionn::gpu
{
template <ScalarType T>
static void __global__ matmul(T* dst, const T* a, const T* b,
                              const uint64_t total_elements)
{
}

void matmul_kernel(Tensor& result, const Tensor& a, const Tensor& b) {}
} // namespace ushionn::gpu
