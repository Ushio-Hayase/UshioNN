#include "core/tensor.h"

#include "function/add.h"
#include "function/elementwise_mul.h"
#include "function/matmul.h"
#include "function/mul.h"
#include "utils/log_macro.h"

#include <cmath>

namespace ushionn
{

Tensor::Tensor(const std::vector<uint64_t>& shape, Device location, DType type)
{
    impl_ = std::make_unique<TensorImpl>(shape, type, location);
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
    if (location.type == Device::DeviceType::HOST)
        std::memcpy(data_ptr<T>(), ptr, numel() * get_elem_size());
    else if (location.type == Device::DeviceType::DEVICE)
        cudaMemcpy(data_ptr<T>(), ptr, numel() * get_elem_size(),
                   cudaMemcpyDeviceToDevice);
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

Tensor::Tensor(const Tensor& other) { *this = other.clone(); }

Tensor::Tensor(const Tensor& other, const std::vector<uint64_t>& shape,
               const std::vector<uint64_t>& strides, uint64_t offset)
    : impl_(std::make_unique<TensorImpl>(other.impl_->storage(), shape, strides,
                                         offset, other.dtype()))
{
}

Tensor& Tensor::operator=(const Tensor& other) { return *this = other.clone(); }

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

    auto new_impl = std::make_unique<TensorImpl>(
        this->impl_->storage(), new_shape, new_strides,
        this->impl_->storage_offset(), this->dtype());

    return Tensor(std::move(new_impl));
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

    auto new_impl = std::make_unique<TensorImpl>(
        this->impl_->storage(), new_shape, new_strides,
        this->impl_->storage_offset(), this->dtype());

    return Tensor(std::move(new_impl));
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

    return Tensor(*this);
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

    function::Add::forward(*this, *this, other);
    return *this;
}

Tensor& Tensor::operator*=(const float scalar)
{
    ASSERT_MESSAGE(this->device().type != Device::DeviceType::NONE,
                   "Tensor not assigned");

    function::Mul::forward(*this, *this, scalar);

    return *this;
}

Tensor operator+(const Tensor& lhs, const Tensor& rhs)
{
    return function::Add::forward(lhs, rhs);
}

Tensor operator*(const Tensor& lhs, const Tensor& rhs)
{
    return function::ElementWiseMul::forward(lhs, rhs);
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

void* Tensor::data() const
{
    return static_cast<unsigned char*>(impl_->storage()->data()) +
           impl_->storage_offset() * get_elem_size();
}

template <typename T>
static void apply_contiguous(const void* src_ptr_raw, void* dst_ptr_raw,
                             uint64_t total_elements, int rank,
                             const std::vector<uint64_t>& shape,
                             const std::vector<uint64_t>& strides);

Tensor Tensor::clone_cpu() const
{
    Tensor result(shape(), device(), dtype());

    const void* src_ptr = data();
    void* dst_ptr = result.data();

    switch (this->dtype())
    {
    case DType::FP64:
        apply_contiguous<double>(src_ptr, dst_ptr, numel(), dim(), shape(),
                                 strides());
        break;
    case DType::FP32:
        apply_contiguous<float>(src_ptr, dst_ptr, numel(), dim(), shape(),
                                strides());
        break;
    case DType::FP16:
        apply_contiguous<fp16_t>(src_ptr, dst_ptr, numel(), dim(), shape(),
                                 strides());
        break;
    case DType::BF16:
        apply_contiguous<bf16_t>(src_ptr, dst_ptr, numel(), dim(), shape(),
                                 strides());
        break;

    case DType::FP8_e4m3:
        apply_contiguous<fp8_e4m3_t>(src_ptr, dst_ptr, numel(), dim(), shape(),
                                     strides());
        break;
    case DType::FP8_e5m2:
        apply_contiguous<fp8_e5m2_t>(src_ptr, dst_ptr, numel(), dim(), shape(),
                                     strides());
        break;
    default:
        LOG_ERROR("{} is a data type that is not yet supported.",
                  dtype_to_string(result.dtype()));
    }
    return result;
}

template <typename T>
void apply_contiguous(const void* src_ptr_raw, void* dst_ptr_raw,
                      uint64_t total_elements, int rank,
                      const std::vector<uint64_t>& shape,
                      const std::vector<uint64_t>& strides)
{
    const T* src = static_cast<const T*>(src_ptr_raw);
    T* dst = static_cast<T*>(dst_ptr_raw);

    // 스칼라(0차원) 처리
    if (rank == 0)
    {
        dst[0] = src[0];
        return;
    }

    std::vector<uint64_t> coords(rank, 0);
    uint64_t src_offset = 0;

    for (uint64_t i = 0; i < total_elements; ++i)
    {
        // 데이터 복사
        dst[i] = src[src_offset];

        // N차원 인덱스 전진 (Division/Modulo 없이 덧셈/뺄셈만 사용)
        for (int d = rank - 1; d >= 0; --d)
        {
            coords[d]++;
            if (coords[d] < shape[d])
            {
                src_offset += strides[d];
                break; // 캐리(Carry)가 발생하지 않으면 하위 차원 전진 멈춤
            }
            else
            {
                coords[d] = 0; // 초기화
                // 해당 차원을 리셋하면서, 더해졌던 스트라이드만큼 다시 빼줌
                src_offset -= (shape[d] - 1) * strides[d];
            }
        }
    }
}

} // namespace ushionn