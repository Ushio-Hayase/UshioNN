//
// Created by UshioHayase on 2026-03-14.
//

#pragma once
#include "layers.h"

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
};
} // namespace layer
} // namespace ushionn
