//
// Created by UshioHayase on 2026-03-15.
//

#include "memory/cpu_allocator.h"

#include "utils/constant.h"
#include "utils/log_macro.h"

#include <cassert>

namespace ushionn::memory
{
void* CPUAllocator::allocate(uint64_t size)
{
    if (size % MEMORY_ALIGNMENT != 0)
    {
        LOG_WARN("The amount of memory to allocate must be multiples of 64, "
                 "Forced to assign in multiples of 64.");
        size += MEMORY_ALIGNMENT - (size % MEMORY_ALIGNMENT);
    }
#if defined(_WIN32)
    return _aligned_malloc(size, MEMORY_ALIGNMENT);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, MEMORY_ALIGNMENT, size) != 0)
        return nullptr;
    return ptr;
#endif
}

void CPUAllocator::deallocate(void* ptr)
{
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

} // namespace ushionn::memory
