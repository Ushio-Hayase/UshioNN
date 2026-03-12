//
// Created by UshioHayase on 3/8/2026.
//

#pragma once
#include "allocator.h"
#include "type.h"

#include <memory>

namespace ushionn
{
class Storage
{
  public:
    Storage(size_t total_bytes);
    Storage(void* ptr, size_t total_bytes, Device location,
            MemoryDeleter deleter);
    ~Storage() = default;

    void* data() const;
    Device location() const;
    size_t nbytes() const;

  private:
    std::unique_ptr<void, MemoryDeleter> data_;

    size_t total_bytes_;
    Device location_;
};
} // namespace ushionn
