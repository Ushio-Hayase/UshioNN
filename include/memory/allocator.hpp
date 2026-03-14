//
// Created by UshioHayase on 2026-03-13.
//

#pragma once

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

template <typename T> class BaseAllocator : public IAllocator
{
  public:
    void* allocate(size_t size) override
    {
        return static_cast<const T*>(this)->allocate(size);
    };
    void deallocate(void* ptr) override
    {
        static_cast<const T*>(this)->deallocate(ptr);
    };
};
} // namespace memory
} // namespace ushionn