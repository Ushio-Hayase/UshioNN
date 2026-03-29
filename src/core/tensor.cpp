#include "core/tensor.h"

#include "../../include/utils/simd.h"
#include "kernel/cpu/add_cpu.h"
#include "kernel/cpu/mul_cpu.h"
#include "kernel/gpu/add_gpu.h"
#include "kernel/gpu/mul_gpu.h"
#include "utils/log_macro.h"

#include <cmath>

namespace ushionn
{

Tensor::Tensor(const std::vector<uint64_t> shape, Device location, DType type)
{
    impl_ = std::make_shared<TensorImpl>(shape, type, location);
}

template <ScalarType T>
Tensor::Tensor(const std::vector<uint64_t>& shape, const T* ptr,
               Device location)
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
template Tensor::Tensor(const std::vector<uint64_t>&, const double*, Device);
template Tensor::Tensor(const std::vector<uint64_t>&, const float*, Device);
template Tensor::Tensor(const std::vector<uint64_t>&, const fp16_t*, Device);
template Tensor::Tensor(const std::vector<uint64_t>&, const bf16_t*, Device);
template Tensor::Tensor(const std::vector<uint64_t>&, const fp8_e4m3_t*,
                        Device);
template Tensor::Tensor(const std::vector<uint64_t>&, const fp8_e5m2_t*,
                        Device);
template Tensor::Tensor(const std::vector<uint64_t>&, const fp4_t*, Device);

Tensor::Tensor(const Tensor& origin, std::vector<uint64_t> shape,
               std::vector<uint64_t> strides, uint64_t offset, DType type)
    : impl_(std::make_shared<TensorImpl>(origin.impl_->storage(),
                                         std::move(shape), std::move(strides),
                                         offset, type))
{
}

Tensor Tensor::transpose(uint64_t dim1, uint64_t dim2) const
{
    ASSERT_MESSAGE(device().type != Device::DeviceType::NONE,
                   "Tensor not assigned")
    ASSERT_MESSAGE(dim1 < dim(),
                   "Ranks of the passed parameter are not the same as those of "
                   "the tensor.");
    ASSERT_MESSAGE(dim2 < dim(),
                   "Ranks of the passed parameter are not the same as those of "
                   "the tensor.");

    std::vector<uint64_t> new_shape = this->shape();
    std::vector<uint64_t> new_strides = this->strides();

    std::swap(new_shape[dim1], new_shape[dim2]);
    std::swap(new_strides[dim1], new_strides[dim2]);

    auto new_impl = std::make_shared<TensorImpl>(
        this->impl_->storage(), new_shape, new_strides,
        this->impl_->storage_offset(), this->dtype());

    return Tensor(new_impl);
}

Tensor Tensor::permute(const std::vector<uint64_t>& order) const
{
    ASSERT_MESSAGE(device().type != Device::DeviceType::NONE,
                   "Tensor not assigned")
    ASSERT_MESSAGE(order.size() == dim(),
                   "Ranks of the passed parameter are not the same as those of "
                   "the tensor.");

    uint64_t rank = dim();

    const std::vector<uint64_t>& old_strides = strides();
    const std::vector<uint64_t>& old_shape = shape();

    std::vector<uint64_t> new_strides(rank);
    std::vector<uint64_t> new_shape(rank);

    for (uint64_t i = 0; i < rank; ++i)
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

Tensor Tensor::view(const std::vector<uint64_t>& shape) const
{
    ASSERT_MESSAGE(device().type != Device::DeviceType::NONE,
                   "Tensor not assigned")
    ASSERT_MESSAGE(is_contiguous(), "Tensor is not continuous");

    uint64_t new_total_elems = 1;
    for (const auto& elem : shape)
        new_total_elems *= elem;

    ASSERT_MESSAGE(numel() == new_total_elems, "Shape sequence size mismatch.");

    const std::vector<uint64_t> new_strides =
        TensorImpl::calculate_default_strides(shape);

    return Tensor(*this, shape, new_strides, impl_->storage_offset(),
                  impl_->dtype());
}

Tensor Tensor::reshape(const std::vector<uint64_t>& shape) const
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

    return Tensor();
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

    return Tensor(shape(), device(), dtype());
}

Tensor Tensor::to(DType t) const
{
    if (dtype() == t)
        return *this;

    return Tensor(shape(), device(), dtype());
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

Tensor& Tensor::operator+=(const Tensor& other)
{
    ASSERT_MESSAGE(this->device().type != Device::DeviceType::NONE,
                   "Tensor not assigned");
    ASSERT_MESSAGE(other.device().type != Device::DeviceType::NONE,
                   "Tensor not assigned");
    ASSERT_MESSAGE(this->device().type == other.device().type,
                   "Both tensors must be on the same deivce");
    ASSERT_MESSAGE(this->dtype() == other.dtype(),
                   "Both tensors must have the same data type");
    ASSERT_MESSAGE(this->shape() == other.shape(),
                   "Both tensors must have the same shape");

    if (this->device().type == Device::DeviceType::HOST)
    {
        cpu::add_kernel(*this, *this, other);
    }
    else if (this->device().type == Device::DeviceType::DEVICE)
    {
        gpu::add_kernel(*this, *this, other);
    }
    return *this;
}

Tensor& Tensor::operator*=(const float scalar)
{
    ASSERT_MESSAGE(this->device().type != Device::DeviceType::NONE,
                   "Tensor not assigned");

    if (this->device().type == Device::DeviceType::HOST)
    {
        cpu::scalar_mul_kernel(*this, *this, scalar);
    }
    else if (this->device().type == Device::DeviceType::DEVICE)
    {
        gpu::scalar_mul_kernel(*this, *this, scalar);
    }
    return *this;
}

const std::vector<uint64_t>& Tensor::shape() const noexcept
{
    return impl_->shape();
}
const std::vector<uint64_t>& Tensor::strides() const noexcept
{
    return impl_->strides();
}
uint64_t Tensor::dim() const noexcept { return impl_->dim(); }
uint64_t Tensor::numel() const noexcept { return impl_->numel(); }
DType Tensor::dtype() const noexcept { return impl_->dtype(); }
Device Tensor::device() const noexcept { return impl_->device(); }
bool Tensor::is_contiguous() const noexcept { return impl_->is_contiguous(); }
uint64_t Tensor::get_elem_size() const noexcept
{
    return impl_->get_elem_size();
}

void Tensor::zero() noexcept { impl_->zero(); }

void* Tensor::data() const { return impl_->storage()->data(); }

Tensor Tensor::clone_cpu() const
{
    const auto& _shape = shape();
    Tensor result(_shape, device(), dtype());
    int total_elements = result.numel();
    int ndim = _shape.size();
    std::vector<uint64_t> dst_strides =
        TensorImpl::calculate_default_strides(_shape);
    std::vector<uint64_t> coords(ndim);

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