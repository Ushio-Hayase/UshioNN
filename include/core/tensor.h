// include/nunet/core/tensor.h
#pragma once

#include "device.h"
#include "tensor_impl.h"

#include "core/type.h"

#include <atomic>
#include <memory> // for std::unique_ptr
#include <utility>
#include <vector>

namespace ushionn
{
class Tensor
{
  public:
    Tensor() = default;

    /// @brief 새로운 텐서를 생성합니다.
    /// @param shape 생성할 텐서 shape
    /// @param type 생성할 텐서 type
    /// @param device 생성하
    explicit Tensor(std::vector<size_t> shape, DType type = DType::FP32,
                    Device device = {});

    template <ScalarType T>
    Tensor(const std::vector<size_t>& shape, const T* ptr, Device device);

    Tensor(std::shared_ptr<TensorImpl> impl) : impl_(std::move(impl)) {}
    Tensor(std::shared_ptr<StorageImpl> impl, std::vector<size_t> shape,
           std::vector<size_t> strides, size_t offset, DType type);

    ~Tensor() = default;

    Tensor(const Tensor&) = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor(Tensor&&) = default;
    Tensor& operator=(Tensor&&) = default;

    [[nodiscard]] Tensor transpose(size_t dim1, size_t dim2) const;
    [[nodiscard]] Tensor permute(const std::vector<size_t>& order) const;
    [[nodiscard]] Tensor view(
        const std::vector<size_t>& shape) const; // 메모리가 연속일 때만 성공
    [[nodiscard]] Tensor reshape(const std::vector<size_t>& shape)
        const; // 연속적이면 view, 아니면 clone후 view

    [[nodiscard]] Tensor clone() const; // 깊은 복사
    [[nodiscard]] Tensor contiguous()
        const; // 텐서가 비연속적이면 복사본, 연속적이면 자신 반환

    [[nodiscard]] Tensor to(Device d) const;
    [[nodiscard]] Tensor to(DType type) const;
    [[nodiscard]] Tensor cpu() const;
    [[nodiscard]] Tensor cuda() const;

    Tensor& operator+=(const Tensor& other);
    Tensor& operator*=(const float scalar);

    friend Tensor operator+(const Tensor& lhs, const Tensor& rhs);
    friend Tensor operator*(const Tensor& lhs, const Tensor& rhs);

    inline const std::vector<size_t>& shape() const;
    inline const std::vector<size_t>& strides() const;
    inline size_t dim() const;
    inline size_t numel() const;
    inline DType dtype() const;
    inline Device device() const;
    inline bool is_contiguous() const;
    inline size_t get_elem_size() const;

    void* data() const;

    template <ScalarType T> T* data_ptr() const
    {
        return impl_->data_ptr<T>() + impl_->storage_offset();
    };

  private:
    Tensor clone_cpu() const;
    Tensor clone_gpu() const;

    std::shared_ptr<TensorImpl> impl_;
};

} // namespace ushionn
