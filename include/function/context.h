#pragma once

#include "core/tensor.h"

#include <cstdint>

namespace nunet
{

class Context
{
  public:
    virtual void save() = 0;
    virtual Tensor& get() = 0;
};
} // namespace nunet
