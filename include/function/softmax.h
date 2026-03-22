//
// Created by UshioHayase on 2026-03-14.
//

#pragma once
#include "core/tensor.h"

namespace ushionn
{
namespace function
{
class Softmax
{
  public:
    static Tensor forward(const Tensor& a, int dim);
};
} // namespace function
} // namespace ushionn
