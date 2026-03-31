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
    void* allocate(uint64_t size) override;
    static void deallocate(void* ptr);
};
} // namespace memory
} // namespace ushionn