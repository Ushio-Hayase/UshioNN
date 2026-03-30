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
        float sum = 0.0f;
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
    dim3 blockDim(16, 16, 1);

    // 2. 그리드 차원 (Grid Dimension) 설정
    // x축: N(열)을 커버하기 위한 블록 수
    // y축: M(행)을 커버하기 위한 블록 수
    // z축: 배치축
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y, batch_size);


    matmul<><<<gridDim, blockDim>>>(d_A, d_B, d_C, batch_size, M, K, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        LOG_ERROR("CUDA Kernel Launch Error: {}", cudaGetErrorString(err));
    }
    
    cudaDeviceSynchronize();
}
} // namespace ushionn::gpu
