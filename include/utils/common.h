#pragma once

#include "core/type.h"
#include "utils/log_macro.h"

#include <string>

namespace nunet
{

namespace utils
{ // 순수 C++ 유틸리티 함수 (선언)

// 바이트 크기를 읽기 쉬운 문자열로 변환 (구현은 common.cpp에)
std::string formatBytes(size_t bytes);

} // namespace utils
} // namespace nunet
