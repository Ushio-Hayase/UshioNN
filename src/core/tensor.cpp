#include "core/tensor.h"

#include "core/simd.h"
#include "utils/common.h"
#include "utils/constant.h"
#include "utils/log_macro.h"

#include <emmintrin.h>

#include <cmath>

namespace ushionn
{

Tensor::Tensor(std::vector<size_t> shape, DType type, DataLocation location)
{
    impl_ = std::make_shared<TensorImpl>(shape, type, location);
}

template <ScalarType T>
Tensor::Tensor(const std::vector<size_t>& shape, const T* ptr,
               DataLocation location)
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
template Tensor::Tensor(const std::vector<size_t>&, const double*,
                        DataLocation);
template Tensor::Tensor(const std::vector<size_t>&, const float*, DataLocation);
template Tensor::Tensor(const std::vector<size_t>&, const fp16_t*,
                        DataLocation);
template Tensor::Tensor(const std::vector<size_t>&, const bf16_t*,
                        DataLocation);
template Tensor::Tensor(const std::vector<size_t>&, const fp8_e4m3_t*,
                        DataLocation);
template Tensor::Tensor(const std::vector<size_t>&, const fp8_e5m2_t*,
                        DataLocation);
template Tensor::Tensor(const std::vector<size_t>&, const fp4_t*, DataLocation);

Tensor Tensor::transpose(size_t dim1, size_t dim2) const
{
    ASSERT_MESSAGE(device() != DataLocation::NONE, "Tensor not assigned")
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
    ASSERT_MESSAGE(device() != DataLocation::NONE, "Tensor not assigned")
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
    ASSERT_MESSAGE(device() != DataLocation::NONE, "Tensor not assigned")
    ASSERT_MESSAGE(is_contiguous(), "Tensor is not continuous");

    std::vector<size_t> new_strides = impl_->
}

} // namespace ushionn