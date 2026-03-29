//
// Created by UshioHayase on 3/8/2026.
//
#pragma once
#include "storage_impl.h"
#include "type.h"

#include <memory>
#include <vector>

namespace ushionn
{
class TensorImpl
{
  public:
    /// @brief 새로운 Storage를 할당하며 생성
    /// @param shape 생성될 텐서의 차원
    /// @param type 생성될 텐서의 타입
    /// @param location 생성될 텐서의 장치
    TensorImpl(std::vector<uint64_t> shape, DType type, Device location);

    /// @brief 기존 Storage를 공유하는 텐서 생성
    /// @param storage 공유할 Storage 포인터
    /// @param shape 생성될 텐서의 차원
    /// @param strides 생성될 텐서의 strides
    /// @param offset 생성될 텐서의 offset
    /// @param type 생성될 텐서의 타입
    TensorImpl(std::shared_ptr<StorageImpl> storage,
               std::vector<uint64_t> shape, std::vector<uint64_t> strides,
               uint64_t offset, DType type);

    ~TensorImpl() = default;

    [[nodiscard]] const std::vector<uint64_t>& shape() const noexcept;
    [[nodiscard]] const std::vector<uint64_t>& strides() const noexcept;
    [[nodiscard]] uint64_t dim() const noexcept;
    [[nodiscard]] uint64_t numel() const noexcept;
    [[nodiscard]] uint64_t storage_offset() const noexcept;
    [[nodiscard]] DType dtype() const noexcept;
    [[nodiscard]] Device device() const noexcept;
    [[nodiscard]] uint64_t get_elem_size() const noexcept;
    [[nodiscard]] bool is_contiguous() const;
    void zero() noexcept;

    template <ScalarType T> T* data_ptr() const
    {
        return static_cast<T*>(storage_->data()) + storage_offset_;
    }

    [[nodiscard]] std::shared_ptr<StorageImpl> storage() const;

    static std::vector<uint64_t> calculate_default_strides(
        const std::vector<uint64_t>& shape);

  private:
    std::vector<uint64_t> shape_;
    std::vector<uint64_t> strides_;
    uint64_t total_elements_;
    uint64_t storage_offset_;

    DType type_;
    std::shared_ptr<StorageImpl> storage_;
};
} // namespace ushionn
