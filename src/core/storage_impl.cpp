//
// Created by UshioHayase on 3/8/2026.
//

#include "core/storage_impl.h"

#include "memory/cpu_allocator.h"
#include "memory/cuda_allocator.h"
#include "utils/log_macro.h"

namespace ushionn
{
StorageImpl::StorageImpl(size_t total_bytes)
    : total_bytes_(total_bytes), allocator_(new memory::CPUAllocator()),
      data_(allocator_.allocate(total_bytes), allocator_)
{
}

StorageImpl::StorageImpl(size_t total_bytes, Device device)
    : total_bytes_(total_bytes), device_(device)
{
    ASSERT_MESSAGE(device_.type == Device::DeviceType::NONE,
                   "Memory not assigned");

    if (device_.type == Device::DeviceType::HOST)
        allocator_ = {new memory::CPUAllocator()};
    else if (device_.type == Device::DeviceType::DEVICE)
        allocator_ = {new memory::CUDAAllocator()};

    data_ = {allocator_.allocate(total_bytes_), allocator_};
}

StorageImpl::StorageImpl(const StorageImpl& impl, size_t total_bytes,
                         Device device)
    : total_bytes_(total_bytes), device_(device)
{
    if (device.type == Device::DeviceType::DEVICE)
        allocator_ = new memory::CUDAAllocator();
    else if (device.type == Device::DeviceType::HOST)
        allocator_ = new memory::CPUAllocator();

    data_ = std::unique_ptr<void, memory::AllocatorAdapter>(
        allocator_.allocate(total_bytes), allocator_);

    copy(impl);
}

void* StorageImpl::data() const { return data_.get(); }
Device StorageImpl::device() const { return device_; }
size_t StorageImpl::nbytes() const { return total_bytes_; }

} // namespace ushionn
