#include "core/tensor.h"

#include "cuda/cuda_utils.cuh"

template <typename T>
__global__ static void clone_kernel(
    const T* src_data,         // 원본 비연속 텐서의 VRAM 데이터 포인터
    T* dst_data,               // 대상 연속 텐서의 VRAM 데이터 포인터
    const size_t* shape,       // VRAM에 복사된 차원 크기 배열
    const size_t* src_strides, // VRAM에 복사된 원본 Stride 배열
    const size_t* dst_strides, // VRAM에 복사된 대상 연속 Stride 배열
    size_t ndim,               // 텐서의 차원 수
    size_t total_elements      // 텐서의 총 요소 개수
)
{
    // 1. 1D 스레드 인덱스 계산 (이 값이 연속된 메모리의 Flat Index가 됨)
    uint64_t idx = ushionn::cuda::utils::get_global_idx();

    // 할당된 총 스레드 수가 총 요소 개수보다 많을 수 있으므로 경계 검사
    if (idx >= total_elements)
        return;

    int remaining = idx;
    int src_physical_offset = 0;

    // 2. Flat Index -> ND-Index 변환 및 물리 오프셋 누적 (각 스레드가 병렬로
    // 수행)
    for (int d = 0; d < ndim; ++d)
    {
        int coord = remaining / dst_strides[d];
        remaining %= dst_strides[d];
        src_physical_offset += coord * src_strides[d];
    }

    // 3. 글로벌 메모리(VRAM) 읽기 및 쓰기
    // Global Memory 병합(Coalescing) 접근이 성립하여 VRAM 대역폭을 효율적으로
    // 사용함
    dst_data[idx] = src_data[src_physical_offset];
}

ushionn::Tensor ushionn::Tensor::clone_gpu() const
{
    const auto& _shape = shape();
    const auto& _strides = strides();
    const auto& type = dtype();

    Tensor result(_shape, device(), type);
    int total_elements = result.numel();
    int ndim = shape().size();
    std::vector<size_t> dst_strides =
        TensorImpl::calculate_default_strides(_shape);

    // 메타데이터를 Device(VRAM)로 전송하기 위한 임시 메모리 할당
    size_t* d_shape;
    size_t* d_src_strides;
    size_t* d_dst_strides;
    cudaMalloc(&d_shape, ndim * sizeof(size_t));
    cudaMalloc(&d_src_strides, ndim * sizeof(size_t));
    cudaMalloc(&d_dst_strides, ndim * sizeof(size_t));

    cudaMemcpy(d_shape, _shape.data(), ndim * sizeof(size_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_src_strides, _strides.data(), ndim * sizeof(size_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst_strides, dst_strides.data(), ndim * sizeof(size_t),
               cudaMemcpyHostToDevice);

    // CUDA 그리드/블록 차원 설정
    int blockSize = 256;
    int gridSize = (total_elements + blockSize - 1) / blockSize;

    switch (type)
    {
    case DType::FP64: {
        clone_kernel<double><<<gridSize, blockSize>>>(
            data_ptr<double>(), result.data_ptr<double>(), d_shape,
            d_src_strides, d_dst_strides, ndim, total_elements);
        break;
    }
    case DType::FP32: {
        clone_kernel<float><<<gridSize, blockSize>>>(
            data_ptr<float>(), result.data_ptr<float>(), d_shape, d_src_strides,
            d_dst_strides, ndim, total_elements);
        break;
    }
    case DType::FP16: {
        clone_kernel<fp16_t><<<gridSize, blockSize>>>(
            data_ptr<fp16_t>(), result.data_ptr<fp16_t>(), d_shape,
            d_src_strides, d_dst_strides, ndim, total_elements);
        break;
    }
    case DType::BF16: {
        clone_kernel<bf16_t><<<gridSize, blockSize>>>(
            data_ptr<bf16_t>(), result.data_ptr<bf16_t>(), d_shape,
            d_src_strides, d_dst_strides, ndim, total_elements);
        break;
    }
    case DType::FP8_e4m3: {
        clone_kernel<fp8_e4m3_t><<<gridSize, blockSize>>>(
            data_ptr<fp8_e4m3_t>(), result.data_ptr<fp8_e4m3_t>(), d_shape,
            d_src_strides, d_dst_strides, ndim, total_elements);
        break;
    }
    case DType::FP8_e5m2: {
        clone_kernel<fp8_e5m2_t><<<gridSize, blockSize>>>(
            data_ptr<fp8_e5m2_t>(), result.data_ptr<fp8_e5m2_t>(), d_shape,
            d_src_strides, d_dst_strides, ndim, total_elements);
        break;
    }
    case DType::FP4: {
        clone_kernel<fp4_t><<<gridSize, blockSize>>>(
            data_ptr<fp4_t>(), result.data_ptr<fp4_t>(), d_shape, d_src_strides,
            d_dst_strides, ndim, total_elements);
        break;
    }
    }
    cudaDeviceSynchronize();

    cudaFree(d_shape);
    cudaFree(d_src_strides);
    cudaFree(d_dst_strides);

    return result;
}
