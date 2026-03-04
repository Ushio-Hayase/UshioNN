#pragma once

namespace nunet
{
enum class DataType
{
    FLOAT32,
    FLOAT64,
};

enum class DataLocation
{
    NONE,   // 데이터 없음 (메모리 할당 전)
    HOST,   // CPU 메모리에만 유효한 데이터 존재
    DEVICE, // GPU 메모리에만 유효한 데이터 존재
};
} // namespace nunet