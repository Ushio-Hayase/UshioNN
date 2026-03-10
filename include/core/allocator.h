//
// Created by UshioHayase on 3/8/2026.
//

#pragma once

namespace ushionn
{

void* alignedMalloc(size_t size, size_t alignment);

using MemoryDeleter = void (*)(void*);

inline void HostDeleter(void* ptr)
{
    if (ptr)
#defined(_MSC_VER)
        _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

void cudaDeleter(void* ptr);

} // namespace nunet