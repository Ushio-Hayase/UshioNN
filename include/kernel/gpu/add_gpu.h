#pragma once
#include "core/tensor.h"

namespace ushionn::gpu
{
void add_kernel(Tensor& result, const Tensor& tensor1, const Tensor& tensor2);
}