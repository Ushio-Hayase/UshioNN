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
    Mul() = delete;
    ~Mul() = delete;

    static Tensor forward(const Tensor& a, float b);
    static void forward(Tensor& result, const Tensor& a, float b);
};

inline Tensor mul(const Tensor& a, float b) { return Mul::forward(a, b); }

} // namespace ushionn::function
