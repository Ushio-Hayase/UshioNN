#include "core/tensor.h"

#include "core/simd.h"
#include "memory/cuda_allocator.h"
#include "utils/common.h"
#include "utils/constant.h"
#include "utils/log_macro.h"

#include <emmintrin.h>

#include <cmath>

namespace ushionn
{

Tensor::Tensor(std::vector<size_t> shape, DType type, Device location)
{
    impl_ = std::make_shared<TensorImpl>(shape, type, location);
}

template <ScalarType T>
Tensor::Tensor(const std::vector<size_t>& shape, const T* ptr, Device location)
{
    DType type;
    if constexpr (std::is_same_v<T, double>)
        type = DType::FP64;
    else if constexpr (std::is_same_v<T, float>)
        type = DType::FP32;
    else if constexpr (std::is_same_v<T, fp16_t>)
        type = DType::FP16;
    else if constexpr (std::is_same_v<T, bf16_t>)
        type = DType::BF16;
    else if constexpr (std::is_same_v<T, fp8_e4m3_t>)
        type = DType::FP8_e4m3;
    else if constexpr (std::is_same_v<T, fp8_e5m2_t>)
        type = DType::FP8_e5m2;
    else if constexpr (std::is_same_v<T, fp4_t>)
        type = DType::FP4;
    impl_ = std::make_shared<TensorImpl>(shape, type, location);
    std::copy(ptr, ptr + numel(), data_ptr<T>());
}

// 명시적 인스턴스화
template Tensor::Tensor(const std::vector<size_t>&, const double*, Device);
template Tensor::Tensor(const std::vector<size_t>&, const float*, Device);
template Tensor::Tensor(const std::vector<size_t>&, const fp16_t*, Device);
template Tensor::Tensor(const std::vector<size_t>&, const bf16_t*, Device);
template Tensor::Tensor(const std::vector<size_t>&, const fp8_e4m3_t*, Device);
template Tensor::Tensor(const std::vector<size_t>&, const fp8_e5m2_t*, Device);
template Tensor::Tensor(const std::vector<size_t>&, const fp4_t*, Device);

Tensor::Tensor(std::shared_ptr<StorageImpl> storage, std::vector<size_t> shape,
               std::vector<size_t> strides, size_t offset, DType type)
    : impl_(std::make_shared<TensorImpl>(storage, std::move(shape),
                                         std::move(strides), offset, type))
{
}

Tensor Tensor::transpose(size_t dim1, size_t dim2) const
{
    ASSERT_MESSAGE(device().type != Device::DeviceType::NONE,
                   "Tensor not assigned")
    ASSERT_MESSAGE(dim1 < dim(),
                   "Ranks of the passed parameter are not the same as those of "
                   "the tensor.");
    ASSERT_MESSAGE(dim2 < dim(),
                   "Ranks of the passed parameter are not the same as those of "
                   "the tensor.");

    std::vector<size_t> new_shape = this->shape();
    std::vector<size_t> new_strides = this->strides();

    std::swap(new_shape[dim1], new_shape[dim2]);
    std::swap(new_strides[dim1], new_strides[dim2]);

    auto new_impl = std::make_shared<TensorImpl>(
        this->impl_->storage(), new_shape, new_strides,
        this->impl_->storage_offset(), this->dtype());

    return Tensor(new_impl);
}

Tensor Tensor::permute(const std::vector<size_t>& order) const
{
    ASSERT_MESSAGE(device().type != Device::DeviceType::NONE,
                   "Tensor not assigned")
    ASSERT_MESSAGE(order.size() == dim(),
                   "Ranks of the passed parameter are not the same as those of "
                   "the tensor.");

    size_t rank = dim();

    const std::vector<size_t>& old_strides = strides();
    const std::vector<size_t>& old_shape = shape();

    std::vector<size_t> new_strides(rank);
    std::vector<size_t> new_shape(rank);

    for (size_t i = 0; i < rank; ++i)
    {
        ASSERT_MESSAGE(
            order[i] < rank,
            "The dimension of the input vector is greater than the rank.");

        new_strides[i] = old_strides[order[i]];
        new_shape[i] = old_shape[order[i]];
    }

    auto new_impl = std::make_shared<TensorImpl>(
        this->impl_->storage(), new_shape, new_strides,
        this->impl_->storage_offset(), this->dtype());

    return Tensor(new_impl);
    ;
}

Tensor Tensor::view(const std::vector<size_t>& shape) const
{
    ASSERT_MESSAGE(device().type != Device::DeviceType::NONE,
                   "Tensor not assigned")
    ASSERT_MESSAGE(is_contiguous(), "Tensor is not continuous");

    size_t new_total_elems = 1;
    for (const auto& elem : shape)
        new_total_elems *= elem;

    ASSERT_MESSAGE(numel() == new_total_elems, "Shape sequence size mismatch.");

    const std::vector<size_t> new_strides =
        TensorImpl::calculate_default_strides(shape);

    return Tensor(impl_->storage(), shape, new_strides, impl_->storage_offset(),
                  impl_->dtype());
}

Tensor Tensor::reshape(const std::vector<size_t>& shape) const
{
    if (is_contiguous())
        return this->view(shape);

    Tensor t = contiguous();
    return t.view(shape);
}

Tensor Tensor::clone() const
{

    if (device().type == Device::DeviceType::HOST)
    {
        return clone_cpu();
    }
    else if (device().type == Device::DeviceType::DEVICE)
    {
        return clone_gpu();
    }
    LOG_ERROR("Unsupported device type.");
}

Tensor Tensor::contiguous() const
{
    if (is_contiguous())
        return *this;
    return clone();
}

Tensor Tensor::to(Device d) const
{
    if (d.type == device().type)
        return *this;

    std::shared_ptr<StorageImpl> s = std::make_shared<StorageImpl>(
        *this->impl_->storage(), impl_->storage()->nbytes(), d);
    return Tensor(s, shape(), strides(), impl_->storage_offset(), dtype());
}

Tensor Tensor::to(DType t) const
{
    if (dtype() == t)
        return *this;

    std::shared_ptr<StorageImpl> s = std::make_shared<StorageImpl>(
        *this->impl_->storage(), impl_->storage()->nbytes(), device());
    return Tensor(s, shape(), strides(), impl_->storage_offset(), t);
}

Tensor Tensor::cpu() const
{
    const Device d = {Device::DeviceType::HOST, 0};
    return to(d);
}

Tensor Tensor::cuda() const
{
    const Device d = {Device::DeviceType::DEVICE, 0};
    return to(d);
}

Tensor& Tensor::operator+=(const Tensor& other) {}

Tensor Tensor::clone_cpu() const
{
    const auto& _shape = shape();
    Tensor result(_shape, dtype(), device());
    int total_elements = result.numel();
    int ndim = _shape.size();
    std::vector<size_t> dst_strides =
        TensorImpl::calculate_default_strides(_shape);
    std::vector<size_t> coords(ndim);

    const auto& strides = impl_->strides();

    // CPU 루프: 순차적 처리
    for (int i = 0; i < total_elements; ++i)
    {
        int remaining = i;
        int src_physical_offset = impl_->storage_offset();
        for (int d = 0; d < ndim; ++d)
        {
            coords[d] = remaining / dst_strides[d];
            remaining %= dst_strides[d];
            src_physical_offset += coords[d] * strides[d];
        }
        switch (dtype())
        {
        case DType::FP64: {
            result.data_ptr<double>()[i] =
                data_ptr<double>()[src_physical_offset];
            break;
        }
        case DType::FP32: {
            result.data_ptr<float>()[i] =
                data_ptr<float>()[src_physical_offset];
            break;
        }
        case DType::FP16: {
            result.data_ptr<fp16_t>()[i] =
                data_ptr<fp16_t>()[src_physical_offset];
            break;
        }
        case DType::BF16: {
            result.data_ptr<bf16_t>()[i] =
                data_ptr<bf16_t>()[src_physical_offset];
            break;
        }
        case DType::FP8_e4m3: {
            result.data_ptr<fp8_e4m3_t>()[i] =
                data_ptr<fp8_e4m3_t>()[src_physical_offset];
            break;
        }
        case DType::FP8_e5m2: {
            result.data_ptr<fp8_e5m2_t>()[i] =
                data_ptr<fp8_e5m2_t>()[src_physical_offset];
            break;
        }
        case DType::FP4: {
            result.data_ptr<fp4_t>()[i] =
                data_ptr<fp4_t>()[src_physical_offset];
            break;
        }
        }
    }
    return result;
}
} // namespace ushionn