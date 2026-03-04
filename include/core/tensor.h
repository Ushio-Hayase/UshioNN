// include/nunet/core/tensor.h
#pragma once

#include "core/type.h"

#include <cstdlib>
#include <memory> // for std::unique_ptr
#include <vector>

namespace nunet
{

static ::std::atomic<uint64_t> tensor_uid_counter = 1000;

class Tensor
{
  public:
    // --- 생성자 및 소멸자 ---
    Tensor();

    Tensor(::std::vector<uint64_t> shape, DType type = DType::FP32);

    /// @brief HOST에 데이터를 복사하며 텐서를 생성합니다.
    /// @tparam T 자료형(FP4, FP8_e5m2, FP8_e4m3, BF16, FP16, FP32, FP64 허용)
    /// @param shape 차원
    /// @param ptr 복사할 데이터의 포인터
    template <ScalarType T>
    Tensor(const ::std::vector<uint64_t>& shape, const T* ptr);

    /// @brief DEVICE에 데이터를 참조하며 텐서를 생성합니다.
    /// @param shape 차원
    /// @param gpu_ptr 참조할 DEVICE 포인터
    /// @param type 자료형
    Tensor(const ::std::vector<uint64_t>& shape, void* gpu_ptr, DType type);

    Tensor(const Tensor& other) = delete;
    Tensor operator=(const Tensor& other) = delete;
    Tensor(Tensor&& other) = default;
    Tensor operator=(Tensor&& other);

    ~Tensor() = default;

    Tensor& operator+=(const Tensor& other);

    template <ScalarType T>
    friend Tensor operator*(const T& scalar, const Tensor& tensor);

    Tensor& operator*=(const Tensor& other);

    template <ScalarType T> Tensor& operator*=(const T& scalar);

    void allocateGpuMem(size_t total_bytes);
    void allocateCpuMem(size_t total_bytes);

    void to(DataLocation location);
    void to(DType type);

    std::vector<size_t> getShape() const;
    DataLocation getDevice() const;
    DType getType() const;
    size_t getTotalBytes() const;
    size_t getShapeSize() const;

    const void* const getCpuPtr() const;
    const void* const getGpuPtr() const;
    void* const getCpuPtrMutable();
    void* const getGpuPtrMutable();

    std::vector<size_t> calculateDotResultShape(const Tensor& b) const;

    /// @brief 텐서를 전치합니다 (마지막 2차원을 교환)
    /// @return 전치된 새로운 텐서
    Tensor transpose() const;

    /// @brief 자기 자신을 전치합니다 (in-place)
    void transpose();

    /// @brief 지정된 두 차원을 교환합니다
    /// @param dim1 첫 번째 차원
    /// @param dim2 두 번째 차원
    /// @return 차원이 교환된 새로운 텐서
    Tensor& permute(size_t dim1, size_t dim2);

  private:
    struct CudaDeleter
    {
        void operator()(void* ptr) const
        {
            if (ptr)
            {
            }
        }
    };

    /// @brief 메모리 해제를 위한 커스텀 Deleter,
    struct HostDeleter
    {
        void operator()(void* ptr) const
        {
            if (ptr)
                std::free(ptr);
        }
    };

    std::vector<uint64_t> calculateStrides();

    void matrixMultiplyCpu(const float* a, size_t a_rows, size_t a_cols,
                           const float* b, size_t b_rows, size_t b_cols,
                           float* c, size_t c_rows, size_t c_cols) const;

    void matrixMultiplyCpu(const double* a, size_t a_rows, size_t a_cols,
                           const double* b, size_t b_rows, size_t b_cols,
                           double* c, size_t c_rows, size_t c_cols) const;

    std::vector<size_t> calculateTransposeShape() const;

    void permuteCpuGeneral(size_t dim1, size_t dim2, Tensor& result);
    void permuteGpuGeneral(size_t dim1, size_t dim2, Tensor& result);

    std::unique_ptr<void, HostDeleter> cpu_data_ptr_ = nullptr;
    std::unique_ptr<void, CudaDeleter> gpu_data_ptr_ = nullptr;

    size_t total_bytes_;

    std::vector<size_t> shape_;
    size_t shape_size_;

    std::vector<size_t> strides_;

    DataLocation location_ = DataLocation::NONE;
    DType type_;
};

Tensor operator+(const Tensor& lhs, const Tensor& rhs);
Tensor operator*(const Tensor& lhs, const Tensor& rhs);

/**
 * @brief Take two one-dimensional tensors and perform an inner product.
 *
 * @param lhs The front one-dimensional tensor to perform the inner product
 * @param rhs The backward one-dimensional tensor to perform the inner product
 * @return Tensor Resulting tensor
 */
Tensor dot(const Tensor& lhs, const Tensor& rhs);

/**
 * @brief Performs elementwise multiplication. Returns an exception if the
 * dimensions are not appropriate.
 *
 * @param lhs The front tensor that will perform the elementwise product
 * @param rhs The back tensor that will perform the elementwise product
 * @return Tensor Resulting tensor
 */
Tensor multiply(const Tensor& lhs, const Tensor& rhs); // Product by Element

/**
 * @brief Perform a matrix product. The last dimension size of the left tensor
 * and the last-to-second tensor dimension size of the right tensor must match.
 *
 * @param lhs Left tensor to perform matrix product
 * @param rhs Right tensor to perform matrix product
 * @return Tensor Computation Result Tensor
 */
Tensor matmul(const Tensor& lhs, const Tensor& rhs); // Matrix Product

} // namespace nunet
