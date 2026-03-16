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
    virtual ~IAllocator() = default;
    virtual void* allocate(size_t size) = 0;
    virtual void deallocate(void* ptr) = 0;
};

class AllocatorAdapter
{
  public:
    AllocatorAdapter() = default;
    AllocatorAdapter(IAllocator* allocator);
    ~AllocatorAdapter();
    [[nodiscard]] void* allocate(size_t size) const;
    void operator()(void* ptr) const { allocator_->deallocate(ptr); }

  private:
    IAllocator* allocator_ = nullptr;
};

} // namespace memory
} // namespace ushionn
