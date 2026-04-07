#pragma once

#include "graph/state_dict.h"

#include <unordered_map>

namespace ushionn::graph
{
struct OperationNode
{
};
struct DataNode
{
    StateDict content;

    OperationNode* next = nullptr;
};

class Graph
{
  public:
  private:
    std::unordered_map<uint32_t, std::string> idx_to_str_map_;
    std::unordered_map<std::string, uint32_t> str_to_idx_map_;
    std::vector<StateDict> arr_;
};

} // namespace ushionn::graph