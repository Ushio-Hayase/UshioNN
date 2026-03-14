//
// Created by UshioHayase on 2026-03-14.
//

#pragma once
#include "core/tensor.h"

#include <unordered_map>

namespace ushionn
{
namespace utils
{
class StateDict
{
  public:
    Tensor find(std::string str);
    void append(std::string str, const Tensor& tensor);

  private:
    std::unordered_map<std::string, Tensor> map_;
};
} // namespace utils
} // namespace ushionn
