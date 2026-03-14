//
// Created by UshioHayase on 3/8/2026.
//

#pragma once
#include "device.h"
#include "type.h"

#include <memory>

#include "memory/allocator.hpp"

namespace ushionn
{
class StorageImpl
{
  public:
    StorageImpl(size_t total_bytes);
    template <typename T>
    StorageImpl(void* ptr, size_t total_bytes, Device location,
                BaseAllocator<T> deleter);
    ~StorageImpl() = default;

    void* data() const;
    Device location() const;
    size_t nbytes() const;

  private:
    std::unique_ptr<void, BaseAllocator<T>> data_;

    size_t total_bytes_;
    Device location_;
};
} // namespace ushionn
