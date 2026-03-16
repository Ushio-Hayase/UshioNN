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
    void* allocate(size_t size) override;
    void deallocate(void* ptr) override;
};
} // namespace ushionn::memory
