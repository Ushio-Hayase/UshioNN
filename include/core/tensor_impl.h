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
    TensorImpl(std::vector<size_t> shape, DType type, Device location);

    /// @brief 기존 Storage를 공유하는 텐서 생성
    /// @param storage 공유할 Storage 포인터
    /// @param shape 생성될 텐서의 차원
    /// @param strides 생성될 텐서의 strides
    /// @param offset 생성될 텐서의 offset
    /// @param type 생성될 텐서의 타입
    TensorImpl(std::shared_ptr<StorageImpl> storage, std::vector<size_t> shape,
               std::vector<size_t> strides, size_t offset, DType type);

    ~TensorImpl() = default;

    const std::vector<size_t>& shape() const;
    const std::vector<size_t>& strides() const;
    size_t dim() const;
    size_t numel() const;
    size_t storage_offset() const;
    DType dtype() const;
    Device device() const;

    bool is_contiguous() const;

    void* data() const;

    template <ScalarType T> T* data_ptr() const
    {
        return static_cast<T*>(data());
    }

    std::shared_ptr<StorageImpl> storage() const { return storage_; }

  private:
    std::vector<size_t> calculate_default_strides(
        const std::vector<size_t>& shape) const;

    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    size_t total_elements_;
    size_t storage_offset_;

    DType type_;
    std::shared_ptr<StorageImpl> storage_;
};
} // namespace ushionn
