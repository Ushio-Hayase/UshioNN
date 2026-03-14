//
// Created by UshioHayase on 3/8/2026.
//

#include "core/tensor_impl.h"
namespace ushionn
{
TensorImpl::TensorImpl(std::vector<size_t> shape, DType type,
                       Device location)
    : shape_(shape), type_(type)
{
    strides_ = calculate_default_strides(shape_);

    for (const auto& elem : shape_)

        switch (type_)
        {
        case DType::FP64:
            storage_ = std::make_shared<StorageImpl>();
        }
}

const std::vector<size_t>& TensorImpl::shape() const { return shape_; }
const std::vector<size_t>& TensorImpl::strides() const { return strides_; }
size_t TensorImpl::dim() const { return shape_.size(); }
size_t TensorImpl::numel() const { return total_elements_; }
size_t TensorImpl::storage_offset() const { return storage_offset_; }
DType TensorImpl::dtype() const { return type_; }
Device TensorImpl::device() const { return storage_->location(); }

bool TensorImpl::is_contiguous() const
{
    if (total_elements_ == 0 || shape_.empty())
        return true;

    size_t expected_stride = 1;
    for (int i = shape_.size() - 1; i >= 0; --i)
    {
        if (shape_[i] != 1)
            if (strides_[i] != expected_stride)
                return false;
        expected_stride *= shape_[i];
    }

    return true;
}

} // namespace ushionn