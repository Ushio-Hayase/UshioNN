//
// Created by UshioHayase on 2026-03-31.
//
#include "memory/cuda_allocator.h"
#include "utils/log_macro.h"

#include <cuda_runtime_api.h>

namespace ushionn::memory
{
void* CUDAAllocator::allocate(size_t size)
{
    void* ptr = nullptr;
    auto err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess)
        LOG_ERROR("CUDA Kernel Launch Error: {}", cudaGetErrorString(err));
    return ptr;
}
void CUDAAllocator::deallocate(void* ptr) { cudaFree(ptr); }

} // namespace ushionn::memory
