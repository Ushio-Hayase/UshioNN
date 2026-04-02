//
// Created by UshioHayase on 2026-03-14.
//

#pragma once
#include "core/tensor.h"

namespace ushionn
{
namespace function
{
class Matmul
{
  public:
    static Tensor forward(const Tensor& a, const Tensor& b);
    static void forward(Tensor& result, const Tensor& a, const Tensor& b);
    static std::vector<uint64_t> calculate_matmul_size(
        const std::vector<uint64_t>& a, const std::vector<uint64_t>& b);
};
} // namespace function
} // namespace ushionn