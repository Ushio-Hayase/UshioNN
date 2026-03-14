//
// Created by UshioHayase on 2026-03-13.
//

#pragma once

#include "memory/allocator.hpp"

namespace ushionn
{
namespace memory
{
class CPUAllocator : public BaseAllocator<CPUAllocator>
{
  public:
    void* allocate(size_t size) override;
    void deallocate(void* ptr) override;
};
} // namespace memory
} // namespace ushionn