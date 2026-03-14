//
// Created by UshioHayase on 2026-03-14.
//

#pragma once
#include "state_dict.h"

namespace ushionn
{
namespace utils
{
class Parser
{
  public:
    virtual ~Parser() = default;
    virtual StateDict parse(std::string name) = 0;
};
} // namespace utils
} // namespace ushionn
