//
// Created by UshioHayase on 2026-03-18.
//

#pragma once

#include "core/tensor.h"

namespace ushionn::gpu
{
void scalar_mul_kernel(Tensor& result, const Tensor& src, float scalar);
} // namespace ushionn::gpu
