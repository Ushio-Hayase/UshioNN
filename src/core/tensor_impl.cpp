//
// Created by UshioHayase on 3/8/2026.
//

#include "core/tensor_impl.h"

#include <utility>
namespace ushionn
{
TensorImpl::TensorImpl(std::vector<uint64_t> shape, DType type, Device device)
    : shape_(shape), type_(type), total_elements_(1)
{
    strides_ = calculate_default_strides(shape_);

    for (const auto& elem : shape_)
        total_elements_ *= elem;

    storage_ = std::make_shared<Storage>(total_elements_ * get_elem_size(),
                                             device);
}

TensorImpl::TensorImpl(std::shared_ptr<Storage> storage,
                       std::vector<uint64_t> shape,
                       std::vector<uint64_t> strides, uint64_t offset,
                       DType type)
    : storage_(std::move(storage)), shape_(std::move(shape)),
      strides_(std::move(strides)), storage_offset_(offset), type_(type),
      total_elements_(1)
{
    for (const auto& elem : shape_)
        total_elements_ *= elem;
}

const std::vector<uint64_t>& TensorImpl::shape() const noexcept
{
    return shape_;
}
const std::vector<uint64_t>& TensorImpl::strides() const noexcept
{
    return strides_;
}
uint64_t TensorImpl::dim() const noexcept { return shape_.size(); }
uint64_t TensorImpl::numel() const noexcept { return total_elements_; }
uint64_t TensorImpl::storage_offset() const noexcept { return storage_offset_; }
DType TensorImpl::dtype() const noexcept { return type_; }
Device TensorImpl::device() const noexcept { return storage_->device(); }
uint64_t TensorImpl::get_elem_size() const noexcept
{
    switch (type_)
    {
    case DType::FP64:
        return 8;
    case DType::FP32:
        return 4;
    case DType::FP16:
        return 2;
    case DType::BF16:
        return 2;
    case DType::FP8_e4m3:
        return 1;
    case DType::FP8_e5m2:
        return 1;
    case DType::FP4:
        return 1;
    default:
        return 0;
    }
}

void TensorImpl::zero() noexcept
{
    switch (type_)
    {
    case DType::FP64:
        std::memset(data_ptr<double>(), 0, storage_->nbytes());
    case DType::FP32:
        std::memset(data_ptr<float>(), 0, storage_->nbytes());
    case DType::FP16:
        std::memset(data_ptr<fp16_t>(), 0, storage_->nbytes());
    case DType::BF16:
        std::memset(data_ptr<bf16_t>(), 0, storage_->nbytes());
    case DType::FP8_e4m3:
        std::memset(data_ptr<fp8_e4m3_t>(), 0, storage_->nbytes());
    case DType::FP8_e5m2:
        std::memset(data_ptr<fp8_e5m2_t>(), 0, storage_->nbytes());
    case DType::FP4:
        std::memset(data_ptr<fp4_t>(), 0, storage_->nbytes());
    }
}

bool TensorImpl::is_contiguous() const
{
    if (total_elements_ == 0 || shape_.size() == 0)
        return true;

    uint64_t expected_stride = 1;

    for (int i = shape_.size() - 1; i >= 0; --i)
    {
        if (shape_[i] != 1)
            if (strides_[i] != expected_stride)
                return false;
        expected_stride *= shape_[i];
    }

    return true;
}

std::shared_ptr<Storage> TensorImpl::storage() const { return storage_; }

std::vector<uint64_t> TensorImpl::calculate_default_strides(
    const std::vector<uint64_t>& shape)
{

    const int ndim = shape.size();
    std::vector<uint64_t> strides(ndim);

    if (ndim == 0)
    {
        return strides;
    }

    strides[ndim - 1] = 1;

    for (int i = ndim - 2; i >= 0; --i)
    {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    return strides;
}

} // namespace ushionn