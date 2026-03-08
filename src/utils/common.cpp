#include "utils/common.h"

#include <cstdlib>
#include <iostream>

namespace nunet
{
namespace utils
{

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