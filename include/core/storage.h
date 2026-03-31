//
// Created by UshioHayase on 3/8/2026.
//

#pragma once
#include "device.h"

#include "memory/allocator.h"

#include <memory>

namespace ushionn
{
class Storage
{
  public:
    /// @brief 전달된 바이트 수만큼 HOST에 메모리를 할당합니다.
    /// @param total_bytes 할당할 바이트 수
    Storage(size_t total_bytes);

    /// @brief 전달된 Device에 전달된 바이트만큼 메모리를 할당합니다.
    /// @param total_bytes 할당할 바이트 수
    /// @param device 할당할 Device
    Storage(size_t total_bytes, Device device);

    /// @brief 전달된 impl에 있는 데이터를 device에 복사해 StorageImpl
    /// 인스턴스를 생성합니다.
    /// @param impl 데이터를 복사할 StorageImpl 클래스
    /// @param total_bytes 복사할 바이트 수
    /// @param device 복사할 위치
    Storage(const Storage& impl, size_t total_bytes, Device device);
    ~Storage() = default;

    [[nodiscard]] void* data() const;
    [[nodiscard]] Device device() const;
    size_t nbytes() const;

  private:
    void copy(const Storage& impl);

    std::unique_ptr<memory::IAllocator> allocator_;

    std::unique_ptr<void, void (*)(void*)> data_{nullptr, [](void*) {}};

    size_t total_bytes_;
    Device device_;
};
} // namespace ushionn
