//
// Created by UshioHayase on 3/8/2026.
//

#include "core/storage.h"

#include "core/allocator.h"

namespace ushionn
{
Storage::Storage(size_t total_bytes)
    : total_bytes_(total_bytes), location_(Device::HOST),
      data_(alignedMalloc(total_bytes, 64), HostDeleter)
{
}

Storage::Storage(void* ptr, size_t total_bytes, Device location,
                 MemoryDeleter deleter)
    : total_bytes_(total_bytes), location_(location), data_(ptr, deleter)
{
}

void* Storage::data() const { return data_.get(); }
Device Storage::location() const { return location_; }
size_t Storage::nbytes() const { return total_bytes_; }

} // namespace nunet
