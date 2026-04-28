//
// Created by UshioHayase on 2026-03-14.
//

#pragma once
#include "operation.h"

#include "core/tensor.h"

namespace ushionn
{
namespace function
{
class ReLU : Operation
{
  public:
    static Tensor forward(const Tensor& a);
};
} // namespace function
} // namespace ushionn
