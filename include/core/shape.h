//
// Created by UshioHayase on 2026-03-14.
//

#pragma once
#include <vector>

namespace ushionn
{
class Shape
{
  public:
    void reshape(const std::vector<size_t>& shape);
    const std::vector<size_t>& shape();

  private:
    std::vector<size_t> shape_;
};
} // namespace ushionn
