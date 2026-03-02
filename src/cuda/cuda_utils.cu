// src/cuda/cuda_utils.cu
#include <iostream>  // 여기서 직접 std::cerr 사용 안함 (handleErrorInternal이 처리)
#include <sstream>  // 여기서 직접 std::ostringstream 사용 안함 (handleErrorInternal이 처리)

#include "cuda/cuda_utils.h"  // 선언부
#include "utils/common.h"  // ushionn::internal::handleErrorInternal 사용 위함
#include "utils/log_macro.h"

namespace ushionn
{
namespace cuda
{
namespace utils
{

void handleCudaError(cudaError_t err_code, const char* file, int line,
                     const char* func_name)
{
    if (err_code != cudaSuccess)
    {
        LOG_ERROR("CUDA API error at {}:{} in {}: {} ({})", file, line,
                  func_name, cudaGetErrorName(err_code),
                  cudaGetErrorString(err_code));
    }
}

void checkCudaKernelError(const char* message_prefix, const char* file,
                          int line, const char* func_name)
{
    // 커널 실행은 비동기이므로, 이전 모든 작업 완료를 기다리거나,
    // cudaGetLastError()를 사용하여 마지막 비동기 에러를 가져옴.
    // cudaDeviceSynchronize(); // 필요시 활성화 (성능 영향 고려)
    cudaError_t err_code = cudaGetLastError();
    if (err_code != cudaSuccess)
    {
        LOG_ERROR("CUDA Kernel Execution error ({}) : {}({}))", message_prefix,
                  cudaGetErrorName(err_code), cudaGetErrorString(err_code));
    }
}

void printGpuMemoryUsageImpl(const std::string& tag)
{
    size_t free_bytes, total_bytes;
    cudaError_t err =
        cudaMemGetInfo(&free_bytes, &total_bytes);  // CUDA API 직접 호출
    if (err == cudaSuccess)
    {
        // common.h의 formatBytes 사용 (utils 네임스페이스 명시)
        LOG_INFO("[{}] - Free: {} / Total: {}: (Used:{})",
                 (tag.empty() ? "GPU Memory" : tag),
                 ushionn::utils::formatBytes(free_bytes),
                 ushionn::utils::formatBytes(total_bytes),
                 ushionn::utils::formatBytes(total_bytes - free_bytes));
    }
    else
    {
        LOG_INFO("[{}] - Failed to get GPU memory info: {}",
                 (tag.empty() ? "GPU Memory" : tag), cudaGetErrorString(err));
    }
}

}  // namespace utils
}  // namespace cuda
}  // namespace ushionn