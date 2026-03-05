#include "utils/common.h"

#include <cstdlib>
#include <iostream>

namespace nunet
{
namespace utils
{

void* aligned_malloc(size_t size, size_t alignment)
{
#if defined(_MSC_VER)
    return _aligned_malloc(size, alignment);
#elif defined(__APPLE__) || defined(__linux__)
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0)
        return nullptr;
    return ptr;
#else
    return aligned_malloc(alignment, size);
#endif
}

std::string formatBytes(size_t bytes)
{
    const char* suffixes[] = {"B", "KB", "MB", "GB", "TB"};
    int suffix_idx = 0;
    double count = static_cast<double>(bytes);
    while (count >= 1024 && suffix_idx < 4)
    {
        count /= 1024;
        suffix_idx++;
    }
    std::ostringstream oss;
    oss.precision(2);
    oss << std::fixed << count << " " << suffixes[suffix_idx];
    return oss.str();
}

} // namespace utils
} // namespace nunet