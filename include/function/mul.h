//
// Created by UshioHayase on 2026-04-02.
//

#pragma once
#include "core/tensor.h"

namespace ushionn::function
{
class Mul
{
  public:
    static Tensor forward(const Tensor& a, float b);
    static void forward(Tensor& result, const Tensor& a, float b);
};
} // namespace ushionn::function
