//
// Created by UshioHayase on 2026-03-14.
//

#pragma once
#include "allocator.h"

namespace ushionn
{
namespace memory
{
class CUDAAllocator final : public IAllocator
{
  public:
    void* allocate(size_t size) override;
    void deallocate(void* ptr) override;
};
} // namespace memory
} // namespace ushionn