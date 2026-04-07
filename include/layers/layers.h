#pragma once

#include "core/tensor.h"

#include <unordered_map>

namespace ushionn
{
namespace layer
{

class ILayer
{
  public:
    virtual ~ILayer();
    virtual Tensor forward(const Tensor& t) = 0;
    virtual void to(const Device& device) = 0;
    virtual void load_weights(
        const std::unordered_map<std::string, Tensor>& state_dict) = 0;
    virtual const std::string& name() = 0;
};
} // namespace layer
} // namespace ushionn