//
// Created by UshioHayase on 3/8/2026.
//
#pragma once

#include "core/tensor.h"

namespace ushionn::cpu
{
void scalar_mul_kernel(Tensor& result, const Tensor& src, const float scalar);
} // namespace ushionn::cpu
