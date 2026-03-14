//
// Created by UshioHayase on 2026-03-14.
//

#pragma once
#include "allocator.hpp"

namespace ushionn
{
namespace memory
{
class CUDAAllocator : BaseAllocator<CUDAAllocator>
{
  public:
    void* allocate(size_t size) override;
    void deallocate(void* ptr) override;
};
} // namespace memory
} // namespace ushionn