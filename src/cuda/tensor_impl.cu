//
// Created by UshioHayase on 2026-04-05.
//
#include "core/tensor_impl.h"

#include <cstring>

namespace ushionn
{
void TensorImpl::zero() noexcept
{
    const auto& st = storage();
    if (device().type == Device::DeviceType::HOST)
        std::memset(st->data(), 0, st->nbytes());
    else if (device().type == Device::DeviceType::DEVICE)
        cudaMemset(st->data(), 0, st->nbytes());
}
} // namespace ushionn