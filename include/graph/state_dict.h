//
// Created by UshioHayase on 2026-03-14.
//

#pragma once
#include "core/tensor.h"

namespace ushionn
{
namespace graph
{
struct StateDict
{
    std::string name;
    Tensor content;
};
} // namespace graph
} // namespace ushionn
