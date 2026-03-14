//
// Created by UshioHayase on 2026-03-13.
//

#pragma once

namespace ushionn
{
struct Device
{
    enum class DeviceType
    {
        NONE,   // 데이터 없음 (메모리 할당 전)
        HOST,   // CPU 메모리에만 유효한 데이터 존재
        DEVICE, // GPU 메모리에만 유효한 데이터 존재
    };
    int index_;
};
} // namespace ushionn