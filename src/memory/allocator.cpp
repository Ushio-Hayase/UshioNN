//
// Created by UshioHayase on 2026-03-15.
//

#include "memory/allocator.h"

namespace ushionn::memory
{
AllocatorAdapter::AllocatorAdapter(IAllocator* allocator)
    : allocator_(allocator)
{
}

AllocatorAdapter::~AllocatorAdapter() { delete allocator_; }

void* AllocatorAdapter::allocate(size_t size) const
{
    return allocator_->allocate(size);
}

} // namespace ushionn::memory