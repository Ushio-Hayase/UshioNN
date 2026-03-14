//
// Created by UshioHayase on 2026-03-14.
//

#pragma once
#include <vector_types.h>

namespace ushionn
{
namespace accelarator
{
class KernelConfigurator
{
    static dim3 get_grid_size(size_t total_elements);
    static dim3 get_block_size();
};
} // namespace accelarator
} // namespace ushionn
