//
// Created by UshioHayase on 2026-03-14.
//

#pragma once
#include "parser.h"

namespace ushionn
{
namespace utils
{
class SafeTensorParser : public Parser
{
  public:
    StateDict parse(std::string name) override;
};
} // namespace utils
} // namespace ushionn
