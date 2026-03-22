//
// Created by UshioHayase on 2026-03-22.
//

#pragma once
#include "core/tensor.h"

namespace ushionn::gpu
{
void matmul_kernel(Tensor& result, const Tensor& a, const Tensor& b);
}
