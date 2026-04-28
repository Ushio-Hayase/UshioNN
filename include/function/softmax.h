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
class Softmax : Operation
{
  public:
    static Tensor forward(const Tensor& a, int dim);
};
} // namespace function
} // namespace ushionn
