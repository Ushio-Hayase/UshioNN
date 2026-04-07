//
// Created by UshioHayase on 2026-03-14.
//

#pragma once
#include "layers.h"

#include "graph/state_dict.h"

namespace ushionn
{
namespace layer
{
class Linear final : public ILayer
{
  public:
    Tensor forward(const Tensor& t) override;
    void to(const Device& device) override;
    void load_weights(
        const std::unordered_map<std::string, Tensor>& state_dict) override;
    const std::string& name() override;

  private:
    graph::StateDict data_;
    using operation_t = Tensor (*)(Tensor, Tensor);
    operation_t operation_;
};
} // namespace layer
} // namespace ushionn
