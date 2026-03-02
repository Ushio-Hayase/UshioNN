#pragma once

#include "core/tensor.h"

namespace ushionn
{

static int gsp_variable_id = 500;

class Function
{
  public:
    virtual Tensor forward() = 0;
    virtual std::vector<Tensor> backward() = 0;
};

} // namespace ushionn
