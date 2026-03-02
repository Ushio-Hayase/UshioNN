#pragma once

#include "core/tensor.h"

#include <cstdint>

namespace ushionn
{

class Context
{
  public:
    virtual void save() = 0;
    virtual Tensor& get() = 0;
};
} // namespace ushionn
