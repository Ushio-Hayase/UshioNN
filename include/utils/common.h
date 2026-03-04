#pragma once

#include "core/common.h"
#include "utils/log_macro.h"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace nunet
{

namespace utils
{ // 순수 C++ 유틸리티 함수 (선언)

// 바이트 크기를 읽기 쉬운 문자열로 변환 (구현은 common.cpp에)
std::string formatBytes(size_t bytes);

template <typename T> nunet::DataType primitiveTypeToDataType()
{
    if constexpr (std::is_same_v<T, float>)
    {
        return ::nunet::DataType::FLOAT32;
    }
    else if constexpr (std::is_same_v<T, double>)
    {
        return ::nunet::DataType::FLOAT64;
    }
    else
    {
        LOG_ERROR("Unknown type received");
    }
}

} // namespace utils
} // namespace nunet
