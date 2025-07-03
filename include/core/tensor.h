// include/ushionn/core/tensor.h
#pragma once

#include <functional>
#include <memory>  // for std::unique_ptr
#include <numeric>
#include <stdexcept>
#include <vector>

#include "core/common.h"      // USHIONN_ASSERT 등
#include "cuda/cuda_utils.h"  // CUDA_CHECK 등 (common.h를 통해 포함됨)

template <typename T>
concept ScalarType =
    std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, int> || std::is_same_v<T, long>;

namespace ushionn
{
static std::atomic<int64_t> tensor_uid_counter = 1000;

class Tensor
{
   public:
    // --- 생성자 및 소멸자 ---
    Tensor();

    Tensor(std::vector<size_t> shape, DataType type = DataType::FLOAT32);

    /// @brief HOST에 데이터를 채우며 텐서를 생성합니다.
    /// @tparam T 자료형
    /// @param shape 차원
    /// @param ptr 복사할 데이터의 포인터
    template <typename T>
    Tensor(std::vector<size_t> shape, const T* ptr);

    /// @brief DEVICE에 데이터를 참조하며 텐서를 생성합니다.
    /// @param shape 차원
    /// @param gpu_ptr 참조할 DEVICE 포인터
    /// @param type 자료형
    Tensor(std::vector<size_t> shape, void* gpu_ptr, DataType type);

    Tensor(const Tensor& other) = delete;

    Tensor(Tensor&& other);

    ~Tensor() = default;  // 스마트 포인터가 메모리 관리

    // 텐서 덧셈
    Tensor operator+(const Tensor& other);

    Tensor& operator+=(const Tensor& other);

    // 원소별 텐서 간 곱셈
    Tensor operator*(const Tensor& other);

    // 스칼라 배
    template <ScalarType T>
    Tensor operator*(const T& scalar);

    template <ScalarType T>
    friend Tensor operator*(const T& scalar, const Tensor& tensor);

    Tensor& operator*=(const Tensor& other);

    template <ScalarType T>
    Tensor& operator*=(const T& scalar);

    Tensor operator=(const Tensor& other) = delete;

    Tensor operator=(Tensor&& other);

    /// @brief 두 텐서를 더합니다.
    /// @param b 더할 텐서
    /// @param r 결과 텐서
    void add(const Tensor& b, Tensor& r);

    /// @brief 두 텐서의 각 원소끼리 곱합니다.
    /// @param b 곱할 텐서
    /// @param r 결과 텐서
    void multiply(const Tensor& b, Tensor& r);

    template <typename T>
    void multiply(const T& b, Tensor& r);

    Tensor dot(const Tensor& b);

    void dot(const Tensor& b, Tensor& r) const;

    void allocate_gpu_mem(size_t total_bytes);
    void allocate_cpu_mem(size_t total_bytes);

    void to(DataLocation location);

    std::vector<size_t> get_shape() const;
    DataLocation get_device() const;
    DataType get_type() const;
    size_t get_total_bytes() const;
    size_t get_shape_size() const;

    const void* const get_cpu_ptr() const;
    const void* const get_gpu_ptr() const;
    void* const get_cpu_ptr_mutable();
    void* const get_gpu_ptr_mutable();

    std::vector<size_t> calculate_dot_result_shape(const Tensor& b) const;

    /// @brief 텐서를 전치합니다 (마지막 2차원을 교환)
    /// @return 전치된 새로운 텐서
    Tensor transpose() const;

    /// @brief 텐서를 전치합니다 (마지막 2차원을 교환)
    /// @param r 결과 텐서
    void transpose(Tensor& r);

    /// @brief 자기 자신을 전치합니다 (in-place)
    void transpose_();

    /// @brief 지정된 두 차원을 교환합니다
    /// @param dim1 첫 번째 차원
    /// @param dim2 두 번째 차원
    /// @return 차원이 교환된 새로운 텐서
    Tensor permute(size_t dim1, size_t dim2);

   private:
    struct CudaDeleter
    {
        void operator()(void* ptr) const
        {
            if (ptr)
            {
                CUDA_CHECK(cudaFree(ptr));
            }
        }
    };

    /// @brief 메모리 해제를 위한 커스텀 Deleter,
    struct HostDeleter
    {
        std::function<void(void*)> deleter_func;

        HostDeleter() = default;

        HostDeleter(DataType type)
        {
            switch (type)
            {
                case DataType::FLOAT32:
                    deleter_func = [](void* ptr) { delete[] static_cast<float*>(ptr); };
                    break;
                case DataType::FLOAT64:
                    deleter_func = [](void* ptr) { delete[] static_cast<double*>(ptr); };
                    break;
            }
        }

        void operator()(void* ptr) const
        {
            if (ptr && deleter_func)
            {
                deleter_func(ptr);
            }
        }
    };

    void calculate_strides();

    void add_cpu(const Tensor& b, Tensor& r);

    template <ScalarType T>
    void scalar_multiply_cpu(const T& b, Tensor& r);

    void gemm_2d(const Tensor& b, Tensor& r) const;

    void gemm_strided_batched(const Tensor& b, Tensor& r) const;

    void dot_cpu_batched(const Tensor& b, Tensor& r) const;

    void matrix_multiply_cpu(const float* a, size_t a_rows, size_t a_cols, const float* b, size_t b_rows, size_t b_cols,
                             float* c, size_t c_rows, size_t c_cols) const;

    void matrix_multiply_cpu(const double* a, size_t a_rows, size_t a_cols, const double* b, size_t b_rows,
                             size_t b_cols, double* c, size_t c_rows, size_t c_cols) const;

    void transpose_cpu(Tensor& r) const;
    void transpose_gpu(Tensor& r) const;

    std::vector<size_t> calculate_transpose_shape() const;

    void permute_cpu_general(size_t dim1, size_t dim2, Tensor& result);
    void permute_gpu_general(size_t dim1, size_t dim2, Tensor& result);

    cublasHandle_t cublas_handle_;

    std::unique_ptr<void, HostDeleter> cpu_data_ptr_ = nullptr;
    std::unique_ptr<void, CudaDeleter> gpu_data_ptr_ = nullptr;

    size_t total_bytes_;

    std::vector<size_t> shape_;
    size_t shape_size_;

    std::vector<size_t> strides_;

    DataLocation location_ = DataLocation::NONE;
    DataType type_;
};

}  // namespace ushionn