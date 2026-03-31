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
static void __global__ matmul(T* dst, const T* a, const T* b, int batch_size,
                              int M, int K, int N)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // 3D 그리드의 Z축을 활용한 배치 인덱스 획득
    int batch_idx = blockIdx.z;

    // 경계 검사
    if (batch_idx < batch_size && row < M && col < N)
    {

        // 배치 차원에 따른 메모리 포인터 오프셋 계산
        const T* a_batch_ptr = a + (batch_idx * M * K);
        const T* b_batch_ptr = b + (batch_idx * K * N);
        T* dst_batch_ptr = dst + (batch_idx * M * N);

        // 내적
        T sum = 0;
        for (int k = 0; k < K; ++k)
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

    const int M = a.shape()[a.dim() - 2];
    const int N = b.shape()[b.dim() - 1];
    const int K = a.shape().back();
    const uint64_t batch_size = result.numel() / (M * N);

    // 2. 그리드 차원 (Grid Dimension) 설정
    // x축: N(열)을 커버하기 위한 블록 수
    // y축: M(행)을 커버하기 위한 블록 수
    // z축: 배치축
    dim3 grid_dim((N + block_dim.x - 1) / block_dim.x,
                  (M + block_dim.y - 1) / block_dim.y, batch_size);

    DType type = result.dtype();

    switch (type)
    {
    case DType::FP64: {
        double* a_data = a.data_ptr<double>();
        double* b_data = b.data_ptr<double>();
        double* result_data = result.data_ptr<double>();
        matmul<double><<<grid_dim, block_dim>>>(result_data, a_data, b_data,
                                                batch_size, M, K, N);
    }
    case DType::FP32: {
        float* a_data = a.data_ptr<float>();
        float* b_data = b.data_ptr<float>();
        float* result_data = result.data_ptr<float>();

        matmul<float><<<grid_dim, block_dim>>>(result_data, a_data, b_data,
                                               batch_size, M, K, N);
    }
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        LOG_ERROR("CUDA Kernel Launch Error: {}", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();
}
} // namespace ushionn::gpu
