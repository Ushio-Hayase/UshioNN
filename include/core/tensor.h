// include/nunet/core/tensor.h
#pragma once

#include "tensor_impl.h"

#include "core/type.h"

#include <atomic>
#include <memory> // for std::unique_ptr
#include <vector>

namespace nunet
{

static ::std::atomic<uint64_t> tensor_uid_counter = 1000;

class Tensor
{
  public:
    Tensor() = default;

    Tensor(std::vector<size_t> shape, DType type = DType::FP32,
           DataLocation location = DataLocation::HOST);

    template <ScalarType T>
    Tensor(const std::vector<size_t>& shape, const T* ptr,
           DataLocation location);

    explicit Tensor(std::shared_ptr<TensorImpl> impl) : impl_(impl) {}

    ~Tensor() = default;

    Tensor(const Tensor&) = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor(Tensor&&) = default;
    Tensor& operator=(Tensor&&) = default;

    Tensor transpose(size_t dim1, size_t dim2) const;
    Tensor permute(const std::vector<size_t>& order) const;
    Tensor view(
        const std::vector<size_t>& shape) const; // 메모리가 연속일 때만 성공
    Tensor reshape(const std::vector<size_t>& shape)
        const; // 연속적이면 view, 아니면 clone후 view

    Tensor clone() const; // 깊은 복사
    Tensor contiguous()
        const; // 텐서가 비연속적이면 복사본, 연속적이면 자신 반환

    Tensor to(DataLocation location) const;
    Tensor to(DType type) const;
    Tensor cpu() const;
    Tensor cuda() const;

    Tensor& operator+=(const Tensor& other);
    Tensor& operator*=(const float scalar);

    friend Tensor operator+(const Tensor& lhs, const Tensor& rhs);
    friend Tensor operator*(const Tensor& lhs, const Tensor& rhs);

    const std::vector<size_t>& shape() const;
    const std::vector<size_t>& strides() const;
    size_t dim() const;
    size_t numel() const;
    DType dtype() const;
    DataLocation device() const;
    bool is_contiguous() const;

    template <ScalarType T> T* data_ptr() const
    {
        return impl_->data_ptr<T>();
    };

  private:
    std::shared_ptr<TensorImpl> impl_;
};

} // namespace nunet
