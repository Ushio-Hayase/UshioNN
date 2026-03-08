//
// Created by UshioHayase on 3/8/2026.
//

#include "core/tensor_impl.h"
namespace nunet
{
TensorImpl::TensorImpl(std::vector<size_t> shape, DType type,
                       std::shared_ptr<Storage> storage)
    : shape_(shape), type_(type), storage_(storage)
{
}

const std::vector<size_t>& TensorImpl::shape() const { return shape_; }
const std::vector<size_t>& TensorImpl::strides() const { return strides_; }
size_t TensorImpl::dims() const { return shape_.size(); }
size_t TensorImpl::numel() const { return total_elements_; }
size_t TensorImpl::storageOffset() const { return storage_offset_; }
DType TensorImpl::dtype() const { return type_; }
DataLocation TensorImpl::device() const { return storage_->location(); }

} // namespace nunet