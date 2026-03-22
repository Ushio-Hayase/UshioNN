//
// Created by UshioHayase on 2026-03-18.
//

#pragma once
#include "core/tensor.h"

namespace ushionn::cpu
{
void elementwise_mul_kernel(Tensor& result, const Tensor& a, const Tensor& b);
}
