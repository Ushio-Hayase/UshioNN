//
// Created by UshioHayase on 2026-03-14.
//

#pragma once
#include "core/tensor.h"

namespace ushionn
{
namespace function
{
class ElementWiseMul
{
  public:
    ElementWiseMul() = delete;
    ~ElementWiseMul() = delete;

    static Tensor forward(const Tensor& a, const Tensor& b);
    static void forward(Tensor& result, const Tensor& a, const Tensor& b);
};

inline Tensor elementwise_mul(const Tensor& a, const Tensor& b)
{
    return ElementWiseMul::forward(a, b);
}

} // namespace function
} // namespace ushionn
