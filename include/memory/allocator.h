//
// Created by UshioHayase on 2026-03-13.
//

#pragma once
#include <memory>

namespace ushionn
{

namespace memory
{

class IAllocator
{
  public:
    virtual void* allocate(uint64_t size) = 0;
};

} // namespace memory
} // namespace ushionn
