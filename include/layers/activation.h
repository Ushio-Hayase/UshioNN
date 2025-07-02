#pragma once

#include <memory>

#include "core/variable.h"

namespace ushionn
{
void relu(const Tensor& x, Tensor& result);
void relu_d(const Tensor& x, Tensor& result);
}  // namespace ushionn
