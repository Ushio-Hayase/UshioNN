//
// Created by UshioHayase on 2026-03-14.
//

#pragma once
#include "shape.h"

#include <vector>

namespace ushionn
{
class Strides
{
  public:
    void transpose(size_t dim1, size_t dim2);
    void view(const std::vector<size_t>& shape);
    void broadcat(const Shape& other_shape);
    const std::vector<size_t>& strides();

  private:
    std::vector<size_t> strides_;
};

} // namespace ushionn
