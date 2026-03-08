//
// Created by UshioHayase on 3/8/2026.
//
namespace nunet
{
void* alignedMalloc(size_t size, size_t alignment)
{
#if defined(_MSC_VER)
    return _aligned_malloc(size, alignment);
#elif defined(__APPLE__) || defined(__linux__)
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0)
        return nullptr;
    return ptr;
#else
    return std::aligned_malloc(alignment, size);
#endif
}
} // namespace nunet