//
// Created by UshioHayase on 2026-03-14.
//

#pragma once
#include "layers/layers.h"

namespace ushionn
{
namespace running
{
class Sequential
{
  private:
    std::vector<std::shared_ptr<layer::ILayer>> order_;
};
} // namespace running
} // namespace ushionn
