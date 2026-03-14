//
// Created by UshioHayase on 2026-03-14.
//

#pragma once
#include "layers.h"

#include "core/tensor.h"

#include <unordered_map>

namespace ushionn
{
namespace layer
{
class Conv1D final : public ILayer
{
  public:
    Tensor forward(const Tensor& a) override;
    void to(const Device& device) override;
    void load_weights(
        const std::unordered_map<std::string, Tensor>& stat_dict) override;
    const std::string& name() override;

  private:
    Tensor weight_;
    Tensor bias_;
};
} // namespace layer
} // namespace ushionn