//
// Created by UshioHayase on 2026-03-13.
//

#pragma once

#include "memory/allocator.h"

namespace ushionn::memory
{
class CPUAllocator final : public IAllocator
{
  public:
    void* allocate(uint64_t size) override;
    static void deallocate(void* ptr);
};
} // namespace ushionn::memory
