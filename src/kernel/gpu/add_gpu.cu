#include "core/type.h"
#include "kernel/gpu/add_gpu.h"

namespace ushionn
{
namespace gpu
{
template <ushionn::ScalarType T>
__global__ void add(T* src, const T* tensor1, const T* tensor2)
{
}

void add_kernel(Tensor& result, const Tensor& tensor1, const Tensor& tensor2) {}
} // namespace gpu
} // namespace ushionn
