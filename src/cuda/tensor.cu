#include "core/tensor.h"

#include "cuda/cuda_utils.cuh"

template <nunet::ScalarType T>
__global__ void add(T* src, const T* other, size_t total_elements)
{
    size_t current_idx = nunet::cuda::utils::getGlobalIdx();

    if (current_idx < total_elements)
        src[current_idx] += other[current_idx];
}

template <>
__global__ void add<nunet::fp8_e4m3_t>(nunet::fp8_e4m3_t* src,
                                       const nunet::fp8_e4m3_t* other,
                                       size_t total_elements)
{
    size_t current_idx = nunet::cuda::utils::getGlobalIdx();

    if (current_idx < total_elements)
    {
        // TODO: FP16으로 캐스팅하여 성능 향상 필요
        const float src_elem = static_cast<float>(src[current_idx]);
        const float other_elem = static_cast<float>(other[current_idx]);
        src[current_idx] = nunet::fp8_e4m3_t(src_elem + other_elem);
    }
}

template <>
__global__ void add<nunet::fp8_e5m2_t>(nunet::fp8_e5m2_t* src,
                                       const nunet::fp8_e5m2_t* other,
                                       size_t total_elements)
{
    size_t current_idx = nunet::cuda::utils::getGlobalIdx();

    if (current_idx < total_elements)
    {
        // TODO: FP16으로 캐스팅하여 성능 향상 필요
        const float src_elem = static_cast<float>(src[current_idx]);
        const float other_elem = static_cast<float>(other[current_idx]);
        src[current_idx] = nunet::fp8_e5m2_t(src_elem + other_elem);
    }
}

void nunet::Tensor::addAssignGpu(const nunet::Tensor& other, DType type)
{
    dim3 GridDims(256);
    dim3 BlockDims(256);

    // TODO: FP8 이하 텐서코어 사용 로직 필요

    switch (type)
    {
    case DType::FP64: {
        add<double><<<GridDims, BlockDims>>>(
            static_cast<double*>(this->gpu_data_ptr_.get()),
            static_cast<double*>(other.gpu_data_ptr_.get()),
            this->total_bytes_ / sizeof(double));
        break;
    }
    case DType::FP32: {
        add<float><<<GridDims, BlockDims>>>(
            static_cast<float*>(this->gpu_data_ptr_.get()),
            static_cast<float*>(other.gpu_data_ptr_.get()),
            this->total_bytes_ / sizeof(double));
        break;
    }
    case DType::FP16: {
        add<fp16_t><<<GridDims, BlockDims>>>(
            static_cast<fp16_t*>(this->gpu_data_ptr_.get()),
            static_cast<fp16_t*>(other.gpu_data_ptr_.get()),
            this->total_bytes_ / sizeof(fp16_t));
        break;
    }
    case DType::BF16: {
        add<bf16_t><<<GridDims, BlockDims>>>(
            static_cast<bf16_t*>(this->gpu_data_ptr_.get()),
            static_cast<bf16_t*>(other.gpu_data_ptr_.get()),
            this->total_bytes_ / sizeof(bf16_t));
        break;
    }
    case DType::FP8_e4m3: {
        add<fp8_e4m3_t><<<GridDims, BlockDims>>>(
            static_cast<fp8_e4m3_t*>(this->gpu_data_ptr_.get()),
            static_cast<fp8_e4m3_t*>(other.gpu_data_ptr_.get()),
            this->total_bytes_ / sizeof(fp8_e4m3_t));
        break;
    }
    case DType::FP8_e5m2: {
        add<fp8_e5m2_t><<<GridDims, BlockDims>>>(
            static_cast<fp8_e5m2_t*>(this->gpu_data_ptr_.get()),
            static_cast<fp8_e5m2_t*>(other.gpu_data_ptr_.get()),
            this->total_bytes_ / sizeof(fp8_e5m2_t));
        break;
    }
    case DType::FP4: {
        // TODO: FP4일경우 연산 추가 필요
    }
    }
}

template <nunet::ScalarType T>
__global__ void mul(T* src, const float scalar, size_t total_elements)
{
    size_t current_idx = nunet::cuda::utils::getGlobalIdx();

    if (current_idx < total_elements)
        src[current_idx] *= scalar;
}

template <>
__global__ void mul<nunet::fp8_e4m3_t>(nunet::fp8_e4m3_t* src,
                                       const float scalar,
                                       size_t total_elements)
{
    size_t current_idx = nunet::cuda::utils::getGlobalIdx();

    if (current_idx < total_elements)
    {
        // TODO: FP16으로 캐스팅하여 성능 향상 필요
        const float src_elem = static_cast<float>(src[current_idx]);
        src[current_idx] = nunet::fp8_e4m3_t(src_elem * scalar);
    }
}

template <>
__global__ void mul<nunet::fp8_e5m2_t>(nunet::fp8_e5m2_t* src,
                                       const float scalar,
                                       size_t total_elements)
{
    size_t current_idx = nunet::cuda::utils::getGlobalIdx();

    if (current_idx < total_elements)
    {
        // TODO: FP16으로 캐스팅하여 성능 향상 필요
        const float src_elem = static_cast<float>(src[current_idx]);
        src[current_idx] = nunet::fp8_e5m2_t(src_elem * scalar);
    }
}

void nunet::Tensor::mulAssignGpu(const float& scalar)
{
    dim3 GridDims(256);
    dim3 BlockDims(256);

    switch (type_)
    {
    case DType::FP64: {
        mul<double><<<GridDims, BlockDims>>>(
            static_cast<double*>(this->gpu_data_ptr_.get()), scalar,
            this->total_bytes_ / sizeof(double));
        break;
    }
    case DType::FP32: {
        mul<float><<<GridDims, BlockDims>>>(
            static_cast<float*>(this->gpu_data_ptr_.get()), scalar,
            this->total_bytes_ / sizeof(double));
        break;
    }
    case DType::FP16: {
        mul<fp16_t><<<GridDims, BlockDims>>>(
            static_cast<fp16_t*>(this->gpu_data_ptr_.get()), scalar,
            this->total_bytes_ / sizeof(fp16_t));
        break;
    }
    case DType::BF16: {
        mul<bf16_t><<<GridDims, BlockDims>>>(
            static_cast<bf16_t*>(this->gpu_data_ptr_.get()), scalar,
            this->total_bytes_ / sizeof(bf16_t));
        break;
    }
    case DType::FP8_e4m3: {
        mul<fp8_e4m3_t><<<GridDims, BlockDims>>>(
            static_cast<fp8_e4m3_t*>(this->gpu_data_ptr_.get()), scalar,
            this->total_bytes_ / sizeof(fp8_e4m3_t));
        break;
    }
    case DType::FP8_e5m2: {
        mul<fp8_e5m2_t><<<GridDims, BlockDims>>>(
            static_cast<fp8_e5m2_t*>(this->gpu_data_ptr_.get()), scalar,
            this->total_bytes_ / sizeof(fp8_e5m2_t));
        break;
    }
    case DType::FP4: {
        // TODO: FP4일경우 연산 추가 필요
    }
    }
}
