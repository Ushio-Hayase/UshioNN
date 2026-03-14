//
// Created by UshioHayase on 3/8/2026.
//

#include "core/allocator.h"
#include "core/storage_impl.h"

namespace ushionn
{
StorageImpl::StorageImpl(size_t total_bytes)
    : total_bytes_(total_bytes), location_(Device::HOST),
      data_(alignedMalloc(total_bytes, 64), HostDeleter)
{
}

StorageImpl::StorageImpl(void* ptr, size_t total_bytes, Device location,
                 MemoryDeleter deleter)
    : total_bytes_(total_bytes), location_(location), data_(ptr, deleter)
{
}

void* StorageImpl::data() const { return data_.get(); }
Device StorageImpl::location() const { return location_; }
size_t StorageImpl::nbytes() const { return total_bytes_; }

} // namespace nunet
