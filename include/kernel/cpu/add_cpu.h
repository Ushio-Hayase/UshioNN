//
// Created by UshioHayase on 3/8/2026.
//
#pragma once
#include "core/tensor.h"

namespace nunet::cpu
{
void add_kernel(Tensor& result, const Tensor& tensor1, const Tensor& tensor2);
}
