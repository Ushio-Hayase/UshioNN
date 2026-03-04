#include "core/tensor.h"

#include <cmath>

namespace nunet
{
Tensor::Tensor() : total_bytes_(0), shape_size_(0), type_(DataType::FP32) {}

Tensor::Tensor(::std::vector<uint64_t> shape, DataType type)
    : shape_(shape), type_(type), location_(DataLocation::HOST), total_bytes_(1)
{
    for (const auto& size : shape_)
        total_bytes_ *= size;

    switch (type_)
    {
    case DataType::FP64:
        total_bytes_ *= 8;
        break;
    case DataType::FP32:
        total_bytes_ *= 4;
        break;

    case DataType::FP16:
        total_bytes_ *= 2;
        break;
    case DataType::BF16:
        total_bytes_ *= 2;
        break;
    case DataType::FP8_e4m3:
        total_bytes_ *= 1;
        break;
    case DataType::FP8_e5m2:
        total_bytes_ *= 1;
        break;
    case DataType::FP4:
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
        type_ = DataType::FP64;
        total_bytes_ *= 8;
    }
    else if constexpr (std::is_same_v<T, float>)
    {
        type_ = DataType::FP32;
        total_bytes_ *= 4;
    }
    else if constexpr (std::is_same_v<T, fp16_t>)
    {
        type_ = DataType::FP16;
        total_bytes_ *= 2;
    }
    else if constexpr (std::is_same_v<T, bf16_t>)
    {
        type_ = DataType::BF16;
        total_bytes_ *= 2;
    }
    else if constexpr (std::is_same_v<T, fp8_e5m2_t>)
    {
        type_ = DataType::FP8_e5m2;
    }
    else if constexpr (std::is_same_v<T, fp8_e4m3_t>)
    {
        type_ = DataType::FP8_e4m3;
    }
    else if constexpr (std::is_same_v<T, fp4_t>)
    {
        type_ = DataType::FP4;
        total_bytes_ = std::ceil(total_bytes_ / 2.0f);
    }

    shape_size_ = shape_.size();

    strides_ = calculateStrides();

    cpu_data_ptr_.reset(malloc(total_bytes_));

    std::copy(ptr, ptr + (total_bytes_ / sizeof(T)), (T*)cpu_data_ptr_.get());
}

Tensor::Tensor(const std::vector<uint64_t>& shape, void* gpu_ptr, DataType type)
    : shape_(shape), type_(type), location_(DataLocation::DEVICE),
      total_bytes_(1)
{
    for (const auto& size : shape_)
        total_bytes_ *= size;

    switch (type_)
    {
    case DataType::FP64:
        total_bytes_ *= 8;
        break;
    case DataType::FP32:
        total_bytes_ *= 4;
        break;

    case DataType::FP16:
        total_bytes_ *= 2;
        break;
    case DataType::BF16:
        total_bytes_ *= 2;
        break;
    case DataType::FP8_e4m3:
        total_bytes_ *= 1;
        break;
    case DataType::FP8_e5m2:
        total_bytes_ *= 1;
        break;
    case DataType::FP4:
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

} // namespace nunet