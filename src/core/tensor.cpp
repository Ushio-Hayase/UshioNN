#include "core/tensor.h"

#include "core/simd.h"
#include "utils/log_macro.h"

#include <cmath>

namespace nunet
{
Tensor::Tensor() : total_bytes_(0), shape_size_(0), type_(DType::FP32) {}

Tensor::Tensor(::std::vector<uint64_t> shape, DType type)
    : shape_(shape), type_(type), location_(DataLocation::HOST), total_bytes_(1)
{
    for (const auto& size : shape_)
        total_bytes_ *= size;

    switch (type_)
    {
    case DType::FP64:
        total_bytes_ *= 8;
        break;
    case DType::FP32:
        total_bytes_ *= 4;
        break;

    case DType::FP16:
        total_bytes_ *= 2;
        break;
    case DType::BF16:
        total_bytes_ *= 2;
        break;
    case DType::FP8_e4m3:
        total_bytes_ *= 1;
        break;
    case DType::FP8_e5m2:
        total_bytes_ *= 1;
        break;
    case DType::FP4:
        total_bytes_ = std::ceil(total_bytes_ / 2.0f);
        break;
    }
    shape_size_ = shape_.size();

    cpu_data_ptr_.reset(std::malloc(total_bytes_));

    strides_ = calculateStrides();
}

template <ScalarType T>
Tensor::Tensor(const std::vector<uint64_t>& shape, const T* ptr)
    : shape_(shape), location_(DataLocation::HOST), total_bytes_(1)
{
    for (const auto& size : shape_)
        total_bytes_ *= size;

    if constexpr (std::is_same_v<T, double>)
    {
        type_ = DType::FP64;
        total_bytes_ *= 8;
    }
    else if constexpr (std::is_same_v<T, float>)
    {
        type_ = DType::FP32;
        total_bytes_ *= 4;
    }
    else if constexpr (std::is_same_v<T, fp16_t>)
    {
        type_ = DType::FP16;
        total_bytes_ *= 2;
    }
    else if constexpr (std::is_same_v<T, bf16_t>)
    {
        type_ = DType::BF16;
        total_bytes_ *= 2;
    }
    else if constexpr (std::is_same_v<T, fp8_e5m2_t>)
    {
        type_ = DType::FP8_e5m2;
    }
    else if constexpr (std::is_same_v<T, fp8_e4m3_t>)
    {
        type_ = DType::FP8_e4m3;
    }
    else if constexpr (std::is_same_v<T, fp4_t>)
    {
        type_ = DType::FP4;
        total_bytes_ = std::ceil(total_bytes_ / 2.0f);
    }

    shape_size_ = shape_.size();

    strides_ = calculateStrides();

    cpu_data_ptr_.reset(malloc(total_bytes_));

    std::copy(ptr, ptr + (total_bytes_ / sizeof(T)), (T*)cpu_data_ptr_.get());
}

Tensor::Tensor(const std::vector<uint64_t>& shape, void* gpu_ptr, DType type)
    : shape_(shape), type_(type), location_(DataLocation::DEVICE),
      total_bytes_(1)
{
    for (const auto& size : shape_)
        total_bytes_ *= size;

    switch (type_)
    {
    case DType::FP64:
        total_bytes_ *= 8;
        break;
    case DType::FP32:
        total_bytes_ *= 4;
        break;

    case DType::FP16:
        total_bytes_ *= 2;
        break;
    case DType::BF16:
        total_bytes_ *= 2;
        break;
    case DType::FP8_e4m3:
        total_bytes_ *= 1;
        break;
    case DType::FP8_e5m2:
        total_bytes_ *= 1;
        break;
    case DType::FP4:
        total_bytes_ = std::ceil(total_bytes_ / 2.0f);
        break;
    }

    shape_size_ = shape_.size();

    gpu_data_ptr_.reset(gpu_ptr);
}

// 명시적 인스턴스화
template Tensor::Tensor(const std::vector<uint64_t>& shape, const double* ptr);
template Tensor::Tensor(const std::vector<uint64_t>& shape, const float* ptr);
template Tensor::Tensor(const std::vector<uint64_t>& shape, const fp16_t* ptr);
template Tensor::Tensor(const std::vector<uint64_t>& shape, const bf16_t* ptr);
template Tensor::Tensor(const std::vector<uint64_t>& shape,
                        const fp8_e5m2_t* ptr);
template Tensor::Tensor(const std::vector<uint64_t>& shape,
                        const fp8_e4m3_t* ptr);
template Tensor::Tensor(const std::vector<uint64_t>& shape, const fp4_t* ptr);

Tensor& Tensor::operator+=(const Tensor& other)
{
    ASSERT_MESSAGE(this->location_ != DataLocation::NONE,
                   "Tensor not assigned.");
    ASSERT_MESSAGE(this->type_ != other.type_,
                   "The two tensors are of different types.");
    ASSERT_MESSAGE(this->shape_ != other.shape_,
                   "Two tensors have different sizes.");

    if (this->location_ == DataLocation::HOST)
    {

        switch (type_)
        {
        case DType::FP64: {
            double* this_cpu_ptr = static_cast<double*>(cpu_data_ptr_.get());
            double* other_cpu_ptr =
                static_cast<double*>(other.cpu_data_ptr_.get());
#if SIMD_LEVEL == 4
            struct alignas(64) SimdData
            {
                union {
                    __m512d value;
                    double d[8];
                };
            };

            if (total_bytes_ / 8 <= 8)
            {

                for (int i = 0; i < total_bytes_ / elementSize(type_); ++i)
                    *(this_cpu_ptr + i) += *(other_cpu_ptr + i);
            }
            else
            {
                SimdData data;
                data.value = _mm512_load_ps(this_cpu_ptr);
            }

#elif SIMD_LEVEL == 3
#elif SIMD_LEVEL == 2
#elif SIMD_LEVEL == 1
#elif SIMD_LEVEL == 0
#endif

            break;
        }
        case DType::FP32: {
            float* this_cpu_ptr = static_cast<float*>(cpu_data_ptr_.get());
            float* other_cpu_ptr =
                static_cast<float*>(other.cpu_data_ptr_.get());
            for (int i = 0; i < total_bytes_ / elementSize(type_); ++i)
                *(this_cpu_ptr + i) += *(other_cpu_ptr + i);
            break;
        }
        case DType::FP16: {
            fp16_t* this_cpu_ptr = static_cast<fp16_t*>(cpu_data_ptr_.get());
            fp16_t* other_cpu_ptr =
                static_cast<fp16_t*>(other.cpu_data_ptr_.get());
            for (int i = 0; i < total_bytes_ / elementSize(type_); ++i)
                *(this_cpu_ptr + i) += *(other_cpu_ptr + i);
            break;
        }
        case DType::BF16: {
            bf16_t* this_cpu_ptr = static_cast<bf16_t*>(cpu_data_ptr_.get());
            bf16_t* other_cpu_ptr =
                static_cast<bf16_t*>(other.cpu_data_ptr_.get());

            break;
        }
        }
    }

    return *this;
}

uint64_t Tensor::elementSize(DType type)
{
    switch (type)
    {
    case DType::FP64:
        return sizeof(double);
    case DType::FP32:
        return sizeof(float);
    case DType::FP16:
        return 2;
    case DType::BF16:
        return 2;
    case DType::FP8_e5m2:
        return 1;
    case DType::FP8_e4m3:
        return 1;
    case DType::FP4:
        return 1; // 4 bits but 1 byte of storage
    default:
        return 0;
    }
}

} // namespace nunet