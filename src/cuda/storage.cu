//
// Created by UshioHayase on 2026-03-16.
//
#include "core/storage.h"

#include <memory>

namespace ushionn
{
void Storage::copy(const Storage& impl)
{
    if (impl.device_.type == Device::DeviceType::HOST &&
        this->device_.type == Device::DeviceType::HOST)
    {
        std::memcpy(this->data(), impl.data(), impl.nbytes());
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
