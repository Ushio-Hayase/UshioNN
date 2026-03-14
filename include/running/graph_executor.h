//
// Created by UshioHayase on 2026-03-14.
//

#pragma once
#include "core/tensor.h"

namespace ushionn
{
namespace running
{
class GraphExecutor
{
  public:
    ~GraphExecutor() = default;

    GraphExecutor& instance();

    Tensor run();

  private:
    GraphExecutor() = default;
    GraphExecutor(const GraphExecutor&) = delete;
    GraphExecutor& operator=(const GraphExecutor&) = delete;
    GraphExecutor(GraphExecutor&&) = delete;
    GraphExecutor& operator=(GraphExecutor&&) = delete;

    std::vector<std::string> order_;
};
} // namespace running
} // namespace ushionn
