//
// Created by UshioHayase on 3/8/2026.
//

#include "core/storage.h"

#include "memory/cpu_allocator.h"
#include "memory/cuda_allocator.h"
#include "utils/log_macro.h"

#include <memory>

namespace ushionn
{
Storage::Storage(size_t total_bytes)
    : total_bytes_(total_bytes), allocator_(new memory::CPUAllocator()),
      data_(allocator_->allocate(total_bytes),
            &memory::CPUAllocator::deallocate)
{
}

Storage::Storage(size_t total_bytes, Device device)
    : total_bytes_(total_bytes), device_(device)
{
    ASSERT_MESSAGE(device_.type != Device::DeviceType::NONE,
                   "Memory not assigned");

    if (device_.type == Device::DeviceType::HOST)
    {
        allocator_ = std::make_unique<memory::CPUAllocator>();
        data_ = std::unique_ptr<void, void (*)(void*)>{
            allocator_->allocate(total_bytes_),
            &memory::CPUAllocator::deallocate};
    }
    else if (device_.type == Device::DeviceType::DEVICE)
    {
        allocator_ = std::make_unique<memory::CUDAAllocator>();
        data_ = std::unique_ptr<void, void (*)(void*)>{
            allocator_->allocate(total_bytes_),
            &memory::CUDAAllocator::deallocate};
    }
}

Storage::Storage(const Storage& impl, size_t total_bytes, Device device)
    : total_bytes_(total_bytes), device_(device)
{

    if (device_.type == Device::DeviceType::HOST)
    {
        allocator_ = std::make_unique<memory::CPUAllocator>();
        data_ = std::unique_ptr<void, void (*)(void*)>(
            allocator_->allocate(total_bytes_),
            &memory::CPUAllocator::deallocate);
    }
    else if (device_.type == Device::DeviceType::DEVICE)
    {
        allocator_ = std::make_unique<memory::CUDAAllocator>();
        data_ = std::unique_ptr<void, void (*)(void*)>{
            allocator_->allocate(total_bytes),
            &memory::CUDAAllocator::deallocate};
    }

    copy(impl);
}

void* Storage::data() const { return data_.get(); }
Device Storage::device() const { return device_; }
size_t Storage::nbytes() const { return total_bytes_; }

} // namespace ushionn
