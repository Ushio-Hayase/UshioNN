//
// Created by UshioHayase on 3/8/2026.
//
#pragma once
#include "storage.h"
#include "type.h"

#include <memory>
#include <vector>

namespace ushionn
{
class TensorImpl
{
  public:
    TensorImpl(std::vector<size_t> shape, DType type, DataLocation location);
    TensorImpl(std::shared_ptr<Storage> storage, std::vector<size_t> shape,
               std::vector<size_t> strides, size_t offset, DType type);

    ~TensorImpl() = default;

    const std::vector<size_t>& shape() const;
    const std::vector<size_t>& strides() const;
    size_t dims() const;
    size_t numel() const;
    size_t storage_offset() const;
    DType dtype() const;
    DataLocation device() const;

    bool is_contiguous() const;

    void* data() const;

    template <ScalarType T> T* data_ptr() const
    {
        return static_cast<T*>(data());
    }

    std::shared_ptr<Storage> storage() const { return storage_; }

  private:
    std::vector<size_t> calculate_default_strides(
        const std::vector<size_t>& shape) const;

    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    size_t total_elements_;
    size_t storage_offset_;

    DType type_;
    std::shared_ptr<Storage> storage_;
};
} // namespace ushionn
