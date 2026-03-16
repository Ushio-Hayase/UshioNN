//
// Created by UshioHayase on 2026-03-16.
//
#include "core/storage_impl.h"

#include <memory>

namespace ushionn
{
void StorageImpl::copy(const StorageImpl& impl)
{
    if (impl.device_.type == Device::DeviceType::HOST &&
        this->device_.type == Device::DeviceType::HOST)
    {
        std::copy_n(impl.data(), impl.nbytes(), this->data());
    }
    else if (impl.device_.type == Device::DeviceType::DEVICE &&
             this->device_.type == Device::DeviceType::HOST)
    {
        cudaMemcpy(this->data(), impl.data(), impl.nbytes(),
                   cudaMemcpyDeviceToHost);
    }
    else if (impl.device_.type == Device::DeviceType::HOST &&
             this->device_.type == Device::DeviceType::DEVICE)
    {
        cudaMemcpy(this->data(), impl.data(), impl.nbytes(),
                   cudaMemcpyHostToDevice);
    }
    else if (impl.device_.type == Device::DeviceType::DEVICE &&
             this->device_.type == Device::DeviceType::DEVICE)
    {
        cudaMemcpy(this->data(), impl.data(), impl.nbytes(),
                   cudaMemcpyDeviceToDevice);
    }
}

} // namespace ushionn
