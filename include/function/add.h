//
// Created by UshioHayase on 2026-03-14.
//

#pragma once
#include "core/tensor.h"

namespace ushionn
{
namespace function
{
class Add
{
  public:
    Add() = delete;
    ~Add() = delete;

    static Tensor forward(const Tensor& a, const Tensor& b);
    static void forward(Tensor& result, const Tensor& a, const Tensor& b);
};

inline Tensor add(const Tensor& a, const Tensor& b)
{
    return Add::forward(a, b);
}

} // namespace function
} // namespace ushionn
