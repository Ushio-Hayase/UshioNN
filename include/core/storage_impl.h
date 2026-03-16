//
// Created by UshioHayase on 3/8/2026.
//

#pragma once
#include "device.h"

#include "memory/allocator.h"

#include <memory>

namespace ushionn
{
class StorageImpl
{
  public:
    /// @brief 전달된 바이트 수만큼 HOST에 메모리를 할당합니다.
    /// @param total_bytes 할당할 바이트 수
    StorageImpl(size_t total_bytes);

    /// @brief 전달된 Device에 전달된 바이트만큼 메모리를 할당합니다.
    /// @param total_bytes 할당할 바이트 수
    /// @param device 할당할 Device
    StorageImpl(size_t total_bytes, Device device);

    /// @brief 전달된 impl에 있는 데이터를 device에 복사해 StorageImpl
    /// 인스턴스를 생성합니다.
    /// @param impl 데이터를 복사할 StorageImpl 클래스
    /// @param total_bytes 복사할 바이트 수
    /// @param device 복사할 위치
    StorageImpl(const StorageImpl& impl, size_t total_bytes, Device device);
    ~StorageImpl() = default;

    [[nodiscard]] void* data() const;
    [[nodiscard]] Device device() const;
    size_t nbytes() const;

  private:
    void copy(const StorageImpl& impl);

    memory::AllocatorAdapter allocator_;
    std::unique_ptr<void, memory::AllocatorAdapter> data_;

    size_t total_bytes_;
    Device device_;
};
} // namespace ushionn
