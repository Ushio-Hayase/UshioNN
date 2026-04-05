//
// Created by UshioHayase on 2026-03-22.
//

#include "core/type.h"
#include "kernel/gpu/matmul_gpu.h"
#include "utils/constant.h"
#include "utils/log_macro.h"

namespace ushionn::gpu
{
template <ScalarType T>
static void __global__ matmul(T* dst, const T* a, const T* b,
                              uint32_t batch_size, uint32_t M, uint32_t K,
                              uint32_t N, uint32_t a_batch_stride,
                              uint32_t b_batch_stride)
{
    const uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;

    // 3D 그리드의 Z축을 활용한 배치 인덱스 획득
    const uint32_t batch_idx = blockIdx.z;

    // 경계 검사
    if (batch_idx < batch_size && row < M && col < N)
    {

        // 배치 차원에 따른 메모리 포인터 오프셋 계산
        const T* a_batch_ptr = a + (batch_idx * a_batch_stride);
        const T* b_batch_ptr = b + (batch_idx * b_batch_stride);
        T* dst_batch_ptr = dst + (batch_idx * M * N);

        // 내적
        T sum = 0;
        for (uint32_t k = 0; k < K; ++k)
        {
            sum += a_batch_ptr[row * K + k] * b_batch_ptr[k * N + col];
        }
        // 저장
        dst_batch_ptr[row * N + col] = sum;
    }
}

void matmul_kernel(Tensor& result, const Tensor& a, const Tensor& b)
{
    dim3 block_dim(16, 16, 1);

    const Tensor& a_contiguous = a.contiguous();
    const Tensor& b_contiguous = b.contiguous();

    const uint32_t M = a_contiguous.shape()[a_contiguous.dim() - 2];
    const uint32_t N = b_contiguous.shape().back();
    const uint32_t K = a_contiguous.shape().back();
    const uint32_t batch_size = result.numel() / (M * N);

    // 2. 그리드 차원 (Grid Dimension) 설정
    // x축: N(열)을 커버하기 위한 블록 수
    // y축: M(행)을 커버하기 위한 블록 수
    // z축: 배치축
    dim3 grid_dim((N + block_dim.x - 1) / block_dim.x,
                  (M + block_dim.y - 1) / block_dim.y, batch_size);

    // 배치 차원이 1이면 스트라이드는 0, 아니면 M*K, K*N
    const uint32_t a_batch_stride =
        (a_contiguous.shape()[a_contiguous.dim() - 3] == 1) ? 0 : M * K;
    const uint32_t b_batch_stride =
        (b_contiguous.shape()[b_contiguous.dim() - 3] == 1) ? 0 : K * N;

    const DType type = result.dtype();

    switch (type)
    {
    case DType::FP64: {
        const auto* a_data = a_contiguous.data_ptr<double>();
        const auto* b_data = b_contiguous.data_ptr<double>();
        auto* result_data = result.data_ptr<double>();
        matmul<double><<<grid_dim, block_dim>>>(result_data, a_data, b_data,
                                                batch_size, M, K, N,
                                                a_batch_stride, b_batch_stride);
        break;
    }
    case DType::FP32: {
        const auto* a_data = a_contiguous.data_ptr<float>();
        const auto* b_data = b_contiguous.data_ptr<float>();
        auto* result_data = result.data_ptr<float>();

        matmul<float><<<grid_dim, block_dim>>>(result_data, a_data, b_data,
                                               batch_size, M, K, N,
                                               a_batch_stride, b_batch_stride);
        break;
    }
    default:
        LOG_ERROR("{} is a data type that is not yet supported.",
                  dtype_to_string(result.dtype()));
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        LOG_ERROR("CUDA Kernel Launch Error: {}", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();
}
} // namespace ushionn::gpu
