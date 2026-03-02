#include "core/tensor.h"

#include <cuda_runtime.h>  // for cudaMemcpy, cudaMemset etc.

#include <algorithm>
#include <iostream>  // for print_meta_info

#include "utils/log_macro.h"

#ifdef min
#undef min
#endif

#ifdef max
#undef max
#endif

namespace ushionn
{

// --- 생성자 구현 ---
Tensor::Tensor() { cublasCreate(&cublas_handle_); }

Tensor::Tensor(std::vector<size_t> shape, DataType type)
    : cpu_data_ptr_(nullptr, type), shape_(shape)
{
    cublasCreate(&cublas_handle_);
    cudaThreadSynchronize();

    shape_size_ = shape_.size();
    type_ = type;

    size_t total_elements = 1;
    for (size_t dim : shape)
    {
        total_elements *= dim;
    }
    if (type_ == DataType::FLOAT32)
        total_bytes_ = total_elements * sizeof(float);
    else if (type_ == DataType::FLOAT64)
        total_bytes_ = total_elements * sizeof(double);
}

template <typename T>
Tensor::Tensor(std::vector<size_t> shape, const T* ptr)
{
    cublasCreate(&cublas_handle_);
    cudaThreadSynchronize();

    std::copy(shape.begin(), shape.end(), shape_.begin());

    shape_size_ = shape_.size();
    total_bytes_ = std::accumulate(shape.begin(), shape.end(), 0) * sizeof(T);

    cpu_data_ptr_.reset(new T[total_bytes_ / sizeof(T)]);

    memcpy(cpu_data_ptr_.get(), ptr, total_bytes_);

    type_ = utils::primitiveTypeToDataType<T>();
    location_ = DataLocation::HOST;
}

Tensor::Tensor(std::vector<size_t> shape, void* gpu_ptr, DataType type)
    : cpu_data_ptr_(nullptr, type), shape_(shape)
{
    cublasCreate(&cublas_handle_);
    cudaThreadSynchronize();

    shape_size_ = shape_.size();
    type_ = type;

    if (type == DataType::FLOAT32)
        total_bytes_ =
            std::accumulate(shape.begin(), shape.end(), 0ULL) * sizeof(float);
    else if (type == DataType::FLOAT64)
        total_bytes_ =
            std::accumulate(shape.begin(), shape.end(), 0ULL) * sizeof(double);

    location_ = DataLocation::DEVICE;

    gpu_data_ptr_.reset(gpu_ptr);
}

Tensor::Tensor(Tensor&& other)
    : cpu_data_ptr_(std::move(other.cpu_data_ptr_)),
      gpu_data_ptr_(std::move(other.gpu_data_ptr_))
{
    cublas_handle_ = other.cublas_handle_;
    total_bytes_ = other.total_bytes_;
    shape_ = std::move(other.shape_);
    shape_size_ = other.shape_size_;
    strides_ = std::move(other.strides_);
    location_ = other.location_;
    type_ = other.type_;
}

Tensor Tensor::operator+(const Tensor& other)
{
    ASSERT_MESSAGE(location_ != DataLocation::NONE,
                   "텐서가 할당되지 않았습니다.");
    ASSERT_MESSAGE(other.location_ != DataLocation::NONE,
                   "텐서가 할당되지 않았습니다.");

    Tensor result(this->shape_, type_);
    result.allocate_cpu_mem(total_bytes_);
    add(other, result);
    return result;
}

Tensor& Tensor::operator+=(const Tensor& other)
{
    ASSERT_MESSAGE(location_ != DataLocation::NONE,
                   "텐서가 할당되지 않았습니다.");
    ASSERT_MESSAGE(other.location_ != DataLocation::NONE,
                   "텐서가 할당되지 않았습니다.");

    add(other, *this);

    return *this;
}

Tensor Tensor::operator*(const Tensor& other)
{
    ASSERT_MESSAGE(location_ != DataLocation::NONE,
                   "텐서가 할당되지 않았습니다.");
    ASSERT_MESSAGE(other.location_ != DataLocation::NONE,
                   "텐서가 할당되지 않았습니다.");

    Tensor result(this->shape_, type_);
    result.allocate_cpu_mem(total_bytes_);

    multiply(other, result);

    return result;
}

template <ScalarType T>
Tensor Tensor::operator*(const T& scalar)
{
    static_assert(std::is_arithmetic_v<T>, "스칼라는 숫자 타입이여야 합니다.");

    ASSERT_MESSAGE(location_ != DataLocation::NONE,
                   "텐서가 할당되지 않았습니다.");

    Tensor result(this->shape_, type_);

    // 위치에 따른 메모리 할당
    if (location_ == DataLocation::DEVICE)
    {
        result.allocate_gpu_mem(total_bytes_);
        result.location_ = DataLocation::DEVICE;
    }
    else
    {
        result.allocate_cpu_mem(total_bytes_);
        result.location_ = DataLocation::HOST;
    }

    multiply(scalar, result);
    return result;
}

template <typename T>
Tensor operator*(const T& scalar, const Tensor& tensor)
{
    return tensor * scalar;
}

Tensor Tensor::operator=(Tensor&& other)
{
    Tensor tensor(std::move(other));
    return tensor;
}

void Tensor::allocate_gpu_mem(size_t total_bytes)
{
    total_bytes_ = total_bytes;

    if (!gpu_data_ptr_)
        LOG_WARN("Gpu memory is already allocated. Ignore the command");

    if (!gpu_data_ptr_)
    {
        if (type_ == DataType::FLOAT32)
        {
            void* tmp_ptr;
            cudaMalloc(&tmp_ptr, total_bytes_);
            gpu_data_ptr_.reset(tmp_ptr);
        }
        else if (type_ == DataType::FLOAT64)
        {
            void* tmp_ptr;
            cudaMalloc(&tmp_ptr, total_bytes_);
            gpu_data_ptr_.reset(tmp_ptr);
        }
    }
}

void Tensor::allocate_cpu_mem(size_t total_bytes)
{
    total_bytes_ = total_bytes;

    if (!cpu_data_ptr_)
        LOG_WARN("Cpu memory is already allocated. Ignore the command");

    if (!cpu_data_ptr_)
    {
        if (type_ == DataType::FLOAT32)
            cpu_data_ptr_.reset(new float[total_bytes_ / sizeof(float)]);
        else if (type_ == DataType::FLOAT64)
            cpu_data_ptr_.reset(new double[total_bytes_ / sizeof(double)]);
    }
}

void Tensor::to(DataLocation location)
{
    if (location == DataLocation::HOST && location_ != DataLocation::HOST)
    {
        if (type_ == DataType::FLOAT32)
            cpu_data_ptr_.reset(new float[total_bytes_ / sizeof(float)]);
        else if (type_ == DataType::FLOAT64)
            cpu_data_ptr_.reset(new double[total_bytes_ / sizeof(double)]);

        auto status = cudaMemcpy(cpu_data_ptr_.get(), gpu_data_ptr_.get(),
                                 total_bytes_, cudaMemcpyDeviceToHost);

        ASSERT_MESSAGE(status == cudaSuccess,
                       "cudaMemcpy 오류, code : " + status);

        gpu_data_ptr_.reset(nullptr);
        location_ = DataLocation::HOST;
    }
    else if (location == DataLocation::DEVICE &&
             location_ != DataLocation::DEVICE)
    {
        void* tmp_ptr = nullptr;
        auto status1 = cudaMalloc(&tmp_ptr, total_bytes_);
        gpu_data_ptr_.reset(tmp_ptr);

        ASSERT_MESSAGE(status1 == cudaSuccess,
                       "cudaMalloc 오류, code : " + status1);

        auto status2 = cudaMemcpy(gpu_data_ptr_.get(), cpu_data_ptr_.get(),
                                  total_bytes_, cudaMemcpyHostToDevice);

        ASSERT_MESSAGE(status2 == cudaSuccess,
                       "cudaMemcpy 오류, code : " + status2);

        cpu_data_ptr_.reset(nullptr);
        location_ = DataLocation::DEVICE;
    }
}

std::vector<size_t> Tensor::get_shape() const { return shape_; }

DataLocation Tensor::get_device() const { return location_; }

DataType Tensor::get_type() const { return type_; }

size_t Tensor::get_total_bytes() const { return total_bytes_; }

const void* const Tensor::get_cpu_ptr() const { return cpu_data_ptr_.get(); }

const void* const Tensor::get_gpu_ptr() const { return gpu_data_ptr_.get(); }

void* const Tensor::get_cpu_ptr_mutable() { return cpu_data_ptr_.get(); }

void* const Tensor::get_gpu_ptr_mutable() { return gpu_data_ptr_.get(); }

void Tensor::add(const Tensor& b, Tensor& r)
{
    ASSERT_MESSAGE(shape_ == b.shape_,
                   "The dimension of the tensor calculating does not match");
    ASSERT_MESSAGE(shape_ == r.shape_,
                   "The dimension of the tensor calculating does not match");

    ASSERT_MESSAGE(location_ == b.location_,
                   "The location of the data exists must be the same.");
    ASSERT_MESSAGE(location_ == r.location_,
                   "The location of the data exists must be the same.");

    if (location_ == DataLocation::DEVICE && type_ == DataType::FLOAT32)
    {
        float alpha = 1.f;
        float beta = 1.f;

        size_t row = 1;
        if (shape_size_ >= 2)
            for (size_t i = 0; i < shape_size_ - 1; ++i) row *= shape_[i];
        else
            row = 1;
        size_t col = shape_[shape_size_ - 1];
        auto state = cublasSgeam(
            r.cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N, row, col, &alpha,
            static_cast<const float*>(gpu_data_ptr_.get()), row, &beta,
            static_cast<const float*>(b.gpu_data_ptr_.get()), row,
            static_cast<float*>(r.gpu_data_ptr_.get()), row);

        if (state != CUBLAS_STATUS_SUCCESS)
            std::cerr << "There was a problem adding tensor, Error state : "
                      << state << std::endl;
    }
    else if (location_ == DataLocation::DEVICE && type_ == DataType::FLOAT64)
    {
        double alpha = 1.f;
        double beta = 1.f;

        size_t row = 1;
        if (shape_size_ >= 2)
            for (size_t i = 0; i < shape_size_ - 1; ++i) row *= shape_[i];
        else
            row = 1;
        size_t col = shape_[shape_size_ - 1];
        auto state = cublasDgeam(
            r.cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N, row, col, &alpha,
            static_cast<const double*>(gpu_data_ptr_.get()), row, &beta,
            static_cast<const double*>(b.gpu_data_ptr_.get()), row,
            static_cast<double*>(r.gpu_data_ptr_.get()), row);

        if (state != CUBLAS_STATUS_SUCCESS)
            std::cerr << "There was a problem adding tensor, Error state : "
                      << state << std::endl;
    }
    else if (location_ == DataLocation::HOST)
    {
        add_cpu(b, r);
    }
}

template <typename T>
void Tensor::multiply(const T& b, Tensor& r)
{
    USHIONN_ASSERT(shape_ == r.shape_,
                   "The dimension of the tensor calculating does not match");
    USHIONN_ASSERT(location_ == r.location_,
                   "The location of the data exists must be the same.");

    if (location_ == DataLocation::DEVICE && type_ == DataType::FLOAT32)
    {
        size_t total_elements = 1;
        for (size_t dim : shape_)
        {
            total_elements *= dim;
        }

        // 타입 안전한 변환
        float scalar_f = static_cast<float>(b);
        auto state = cublasSscal(r.cublas_handle_, total_elements, &scalar_f,
                                 static_cast<float*>(gpu_data_ptr_.get()), 1);

        if (state != CUBLAS_STATUS_SUCCESS)
            std::cerr
                << "There was a problem multiplying tensor, Error state : "
                << state << std::endl;
    }
    else if (location_ == DataLocation::DEVICE && type_ == DataType::FLOAT64)
    {
        size_t total_elements = 1;
        for (size_t dim : shape_)
        {
            total_elements *= dim;
        }

        // 타입 안전한 변환
        double scalar_d = static_cast<double>(b);
        auto state = cublasDscal(r.cublas_handle_, total_elements, &scalar_d,
                                 static_cast<double*>(gpu_data_ptr_.get()), 1);

        if (state != CUBLAS_STATUS_SUCCESS)
            std::cerr
                << "There was a problem multiplying tensor, Error state : "
                << state << std::endl;
    }
    else if (location_ == DataLocation::HOST)
    {
        scalar_multiply_cpu(b, r);
    }
}

void Tensor::add_cpu(const Tensor& b, Tensor& r)
{
    // 전체 원소 개수 계산
    size_t total_elements = 1;
    for (size_t dim : shape_)
    {
        total_elements *= dim;
    }

    // 타입별로 계산 수행
    if (type_ == DataType::FLOAT32)
    {
        const float* a_data = static_cast<const float*>(cpu_data_ptr_.get());
        const float* b_data = static_cast<const float*>(b.cpu_data_ptr_.get());
        float* r_data = static_cast<float*>(r.cpu_data_ptr_.get());

        for (size_t i = 0; i < total_elements; ++i)
        {
            r_data[i] = a_data[i] + b_data[i];
        }
    }
    else if (type_ == DataType::FLOAT64)
    {
        const double* a_data = static_cast<const double*>(cpu_data_ptr_.get());
        const double* b_data =
            static_cast<const double*>(b.cpu_data_ptr_.get());
        double* r_data = static_cast<double*>(r.cpu_data_ptr_.get());

        for (size_t i = 0; i < total_elements; ++i)
        {
            r_data[i] = a_data[i] + b_data[i];
        }
    }
}

template <ScalarType T>
void Tensor::scalar_multiply_cpu(const T& b, Tensor& r)
{
    // 전체 원소수 계산
    size_t total_elements = 1;
    for (size_t dim : shape_)
    {
        total_elements *= dim;
    }

    // 타입별로 계산 수행
    if (type_ == DataType::FLOAT32)
    {
        const float* a_data = static_cast<const float*>(cpu_data_ptr_.get());
        float* r_data = static_cast<float*>(r.cpu_data_ptr_.get());

        const float alpha = static_cast<float>(b);

        for (size_t i = 0; i < total_elements; ++i)
        {
            r_data[i] = a_data[i] * alpha;
        }
    }
    else if (type_ == DataType::FLOAT64)
    {
        const double* a_data = static_cast<const double*>(cpu_data_ptr_.get());
        double* r_data = static_cast<double*>(r.cpu_data_ptr_.get());

        const double alpha = static_cast<double>(b);

        for (size_t i = 0; i < total_elements; ++i)
        {
            r_data[i] = a_data[i] * alpha;
        }
    }
}

void Tensor::calculate_strides()
{
    strides_.resize(shape_.size());

    strides_[0] = 1;

    size_t stride = 1;

    for (int i = 1; i < shape_size_; ++i)
    {
        stride *= shape_[i];
        strides_[i] = stride;
    }
}

void Tensor::transpose_cpu(Tensor& r) const
{
    ASSERT_MESSAGE(shape_size_ >= 2, "최소 2차원 텐서여야 합니다");

    // 배치 크기 계산
    size_t batch_size = 1;
    for (size_t i = 0; i < shape_size_ - 2; ++i)
    {
        batch_size *= shape_[i];
    }

    size_t rows = shape_[shape_size_ - 2];
    size_t cols = shape_[shape_size_ - 1];

    if (type_ == DataType::FLOAT32)
    {
        const float* input = static_cast<const float*>(cpu_data_ptr_.get());
        float* output = static_cast<float*>(r.cpu_data_ptr_.get());

        for (size_t batch = 0; batch < batch_size; ++batch)
        {
            size_t batch_offset_input = batch * rows * cols;
            size_t batch_offset_output = batch * rows * cols;

            for (size_t i = 0; i < rows; ++i)
            {
                for (size_t j = 0; j < cols; ++j)
                {
                    // (i, j) -> (j, i)
                    output[batch_offset_output + j * rows + i] =
                        input[batch_offset_input + i * cols + j];
                }
            }
        }
    }
    else if (type_ == DataType::FLOAT64)
    {
        const double* input = static_cast<const double*>(cpu_data_ptr_.get());
        double* output = static_cast<double*>(r.cpu_data_ptr_.get());

        for (size_t batch = 0; batch < batch_size; ++batch)
        {
            size_t batch_offset_input = batch * rows * cols;
            size_t batch_offset_output = batch * rows * cols;

            for (size_t i = 0; i < rows; ++i)
            {
                for (size_t j = 0; j < cols; ++j)
                {
                    // (i, j) -> (j, i)
                    output[batch_offset_output + j * rows + i] =
                        input[batch_offset_input + i * cols + j];
                }
            }
        }
    }
}

Tensor Tensor::permute(size_t dim1, size_t dim2)
{
    ASSERT_MESSAGE(dim1 < shape_size_ && dim2 < shape_size_,
                   "차원 인덱스가 유효하지 않습니다");

    if (dim1 == dim2)
    {
        // 같은 차원이면 복사본 반환
        Tensor result(shape_, type_);
        if (location_ == DataLocation::DEVICE)
        {
            result.allocate_gpu_mem(total_bytes_);
            result.location_ = DataLocation::DEVICE;
            cudaMemcpy(result.gpu_data_ptr_.get(), gpu_data_ptr_.get(),
                       total_bytes_, cudaMemcpyDeviceToDevice);
        }
        else
        {
            result.allocate_cpu_mem(total_bytes_);
            result.location_ = DataLocation::HOST;
            std::memcpy(result.cpu_data_ptr_.get(), cpu_data_ptr_.get(),
                        total_bytes_);
        }
        return result;
    }

    // 결과 차원 계산
    std::vector<size_t> result_shape = shape_;
    std::swap(result_shape[dim1], result_shape[dim2]);

    Tensor result(result_shape, type_);

    if (location_ == DataLocation::DEVICE)
    {
        result.allocate_gpu_mem(result.get_total_bytes());
        result.location_ = DataLocation::DEVICE;
    }
    else
    {
        result.allocate_cpu_mem(result.get_total_bytes());
        result.location_ = DataLocation::HOST;
    }

    // 마지막 2차원 교환인 경우 최적화된 transpose 사용
    if ((dim1 == shape_size_ - 2 && dim2 == shape_size_ - 1) ||
        (dim1 == shape_size_ - 1 && dim2 == shape_size_ - 2))
    {
        if (location_ == DataLocation::DEVICE)
        {
            transpose_gpu(result);
        }
        else
        {
            transpose_cpu(result);
        }
    }
    else
    {
        // 일반적인 차원 교환
        if (location_ == DataLocation::DEVICE)
        {
            permute_gpu_general(dim1, dim2, result);
        }
        else
        {
            permute_cpu_general(dim1, dim2, result);
        }
    }

    return result;
}

// 일반적인 차원 교환을 위한 CPU 구현
void Tensor::permute_cpu_general(size_t dim1, size_t dim2, Tensor& result)
{
    // 전체 원소 개수
    size_t total_elements = 1;
    for (size_t dim : shape_)
    {
        total_elements *= dim;
    }

    // Stride 계산 (원본)
    std::vector<size_t> strides(shape_size_);
    strides[shape_size_ - 1] = 1;
    for (int i = shape_size_ - 2; i >= 0; --i)
    {
        strides[i] = strides[i + 1] * shape_[i + 1];
    }

    // Stride 계산 (결과)
    std::vector<size_t> result_strides(shape_size_);
    result_strides[shape_size_ - 1] = 1;
    for (int i = shape_size_ - 2; i >= 0; --i)
    {
        result_strides[i] = result_strides[i + 1] * result.shape_[i + 1];
    }

    if (type_ == DataType::FLOAT32)
    {
        const float* input = static_cast<const float*>(cpu_data_ptr_.get());
        float* output = static_cast<float*>(result.cpu_data_ptr_.get());

        for (size_t linear_idx = 0; linear_idx < total_elements; ++linear_idx)
        {
            // 1차원 인덱스를 다차원 인덱스로 변환
            std::vector<size_t> multi_idx(shape_size_);
            size_t temp_idx = linear_idx;

            for (size_t i = 0; i < shape_size_; ++i)
            {
                multi_idx[i] = temp_idx / strides[i];
                temp_idx %= strides[i];
            }

            // 차원 교환
            std::swap(multi_idx[dim1], multi_idx[dim2]);

            // 다차원 인덱스를 결과의 1차원 인덱스로 변환
            size_t result_linear_idx = 0;
            for (size_t i = 0; i < shape_size_; ++i)
            {
                result_linear_idx += multi_idx[i] * result_strides[i];
            }

            output[result_linear_idx] = input[linear_idx];
        }
    }
    else if (type_ == DataType::FLOAT64)
    {
        const double* input = static_cast<const double*>(cpu_data_ptr_.get());
        double* output = static_cast<double*>(result.cpu_data_ptr_.get());

        for (size_t linear_idx = 0; linear_idx < total_elements; ++linear_idx)
        {
            // 1차원 인덱스를 다차원 인덱스로 변환
            std::vector<size_t> multi_idx(shape_size_);
            size_t temp_idx = linear_idx;

            for (size_t i = 0; i < shape_size_; ++i)
            {
                multi_idx[i] = temp_idx / strides[i];
                temp_idx %= strides[i];
            }

            // 차원 교환
            std::swap(multi_idx[dim1], multi_idx[dim2]);

            // 다차원 인덱스를 결과의 1차원 인덱스로 변환
            size_t result_linear_idx = 0;
            for (size_t i = 0; i < shape_size_; ++i)
            {
                result_linear_idx += multi_idx[i] * result_strides[i];
            }

            output[result_linear_idx] = input[linear_idx];
        }
    }
}

std::vector<size_t> Tensor::calculate_transpose_shape() const
{
    std::vector<size_t> result_shape = shape_;

    if (shape_size_ >= 2)
    {
        // 마지막 2차원을 교환
        std::swap(result_shape[shape_size_ - 2], result_shape[shape_size_ - 1]);
    }

    return result_shape;
}

Tensor Tensor::transpose() const
{
    std::vector<size_t> result_shape = calculate_transpose_shape();
    Tensor result(result_shape, type_);

    if (location_ == DataLocation::DEVICE)
    {
        result.allocate_gpu_mem(result.get_total_bytes());
        result.location_ = DataLocation::DEVICE;
        transpose_gpu(result);
    }
    else
    {
        result.allocate_cpu_mem(result.get_total_bytes());
        result.location_ = DataLocation::HOST;
        transpose_cpu(result);
    }

    return result;
}

void Tensor::transpose(Tensor& r)
{
    ASSERT_MESSAGE(location_ == r.location_, "데이터 위치가 동일해야 합니다");

    std::vector<size_t> expected_shape = calculate_transpose_shape();
    ASSERT_MESSAGE(r.shape_ == expected_shape,
                   "결과 텐서의 차원이 전치 결과와 맞지 않습니다");

    if (location_ == DataLocation::DEVICE)
    {
        transpose_gpu(r);
    }
    else
    {
        transpose_cpu(r);
    }
}

void Tensor::transpose_()
{
    if (shape_size_ < 2)
    {
        return;  // 1차원 이하는 전치할 필요 없음
    }

    // 임시 텐서 생성
    Tensor temp = transpose();

    // 자기 자신으로 이동
    *this = std::move(temp);
}

size_t Tensor::get_shape_size() const { return shape_size_; }

void Tensor::matrix_multiply_cpu(const float* a, size_t a_rows, size_t a_cols,
                                 const float* b, size_t b_rows, size_t b_cols,
                                 float* c, size_t c_rows, size_t c_cols) const
{
    ASSERT_MESSAGE(a_cols == b_rows, "행렬 곱셈 차원이 맞지 않습니다");
    ASSERT_MESSAGE(a_rows == c_rows && b_cols == c_cols,
                   "결과 행렬 차원이 맞지 않습니다");

    // 결과 행렬 초기화
    std::fill(c, c + c_rows * c_cols, 0.0f);

    // 블록 크기 (캐시 최적화)
    const size_t BLOCK_SIZE = 64;

    // 블록별로 계산 (더 캐시 친화적)
    for (size_t ii = 0; ii < a_rows; ii += BLOCK_SIZE)
    {
        for (size_t jj = 0; jj < b_cols; jj += BLOCK_SIZE)
        {
            for (size_t kk = 0; kk < a_cols; kk += BLOCK_SIZE)
            {
                // 실제 블록 범위 계산
                size_t i_end = ::std::min(ii + BLOCK_SIZE, a_rows);
                size_t j_end = ::std::min(jj + BLOCK_SIZE, b_cols);
                size_t k_end = ::std::min(kk + BLOCK_SIZE, a_cols);

                // 블록 내 계산
                for (size_t i = ii; i < i_end; ++i)
                {
                    for (size_t k = kk; k < k_end; ++k)
                    {
                        float a_ik = a[i * a_cols + k];
                        for (size_t j = jj; j < j_end; ++j)
                        {
                            c[i * c_cols + j] += a_ik * b[k * b_cols + j];
                        }
                    }
                }
            }
        }
    }
}

void Tensor::matrix_multiply_cpu(const double* a, size_t a_rows, size_t a_cols,
                                 const double* b, size_t b_rows, size_t b_cols,
                                 double* c, size_t c_rows, size_t c_cols) const
{
    ASSERT_MESSAGE(a_cols == b_rows, "행렬 곱셈 차원이 맞지 않습니다");
    ASSERT_MESSAGE(a_rows == c_rows && b_cols == c_cols,
                   "결과 행렬 차원이 맞지 않습니다");

    // 결과 행렬 초기화
    std::fill(c, c + c_rows * c_cols, 0.0f);

    // 블록 크기 (캐시 최적화)
    const size_t BLOCK_SIZE = 64;

    // 블록별로 계산 (더 캐시 친화적)
    for (size_t ii = 0; ii < a_rows; ii += BLOCK_SIZE)
    {
        for (size_t jj = 0; jj < b_cols; jj += BLOCK_SIZE)
        {
            for (size_t kk = 0; kk < a_cols; kk += BLOCK_SIZE)
            {
                // 실제 블록 범위 계산
                size_t i_end = ::std::min(ii + BLOCK_SIZE, a_rows);
                size_t j_end = ::std::min(jj + BLOCK_SIZE, b_cols);
                size_t k_end = ::std::min(kk + BLOCK_SIZE, a_cols);

                // 블록 내 계산
                for (size_t i = ii; i < i_end; ++i)
                {
                    for (size_t k = kk; k < k_end; ++k)
                    {
                        float a_ik = a[i * a_cols + k];
                        for (size_t j = jj; j < j_end; ++j)
                        {
                            c[i * c_cols + j] += a_ik * b[k * b_cols + j];
                        }
                    }
                }
            }
        }
    }
}

void Tensor::dot_cpu_batched(const Tensor& b, Tensor& r) const
{
    ASSERT_MESSAGE(location_ == DataLocation::HOST,
                   "CPU 배치 연산은 HOST 데이터여야 합니다");
    ASSERT_MESSAGE(b.location_ == DataLocation::HOST,
                   "CPU 배치 연산은 HOST 데이터여야 합니다");
    ASSERT_MESSAGE(r.location_ == DataLocation::HOST,
                   "CPU 배치 연산은 HOST 데이터여야 합니다");

    // 배치 크기 계산
    size_t batch_size = 1;
    for (size_t i = 0; i < shape_size_ - 2; ++i)
    {
        batch_size *= shape_[i];
    }

    // 행렬 차원
    size_t m = shape_[shape_size_ - 2];      // A의 행
    size_t k = shape_[shape_size_ - 1];      // A의 열 = B의 행
    size_t n = b.shape_[b.shape_size_ - 1];  // B의 열

    if (type_ == DataType::FLOAT32)
    {
        const float* a_data = static_cast<const float*>(cpu_data_ptr_.get());
        const float* b_data = static_cast<const float*>(b.cpu_data_ptr_.get());
        float* r_data = static_cast<float*>(r.cpu_data_ptr_.get());

        // 각 배치에 대해 행렬 곱셈 수행
        for (size_t batch = 0; batch < batch_size; ++batch)
        {
            size_t offset_a = batch * m * k;
            size_t offset_b = batch * k * n;
            size_t offset_c = batch * m * n;

            matrix_multiply_cpu(a_data + offset_a, m, k, b_data + offset_b, k,
                                n, r_data + offset_c, m, n);
        }
    }
    else if (type_ == DataType::FLOAT64)
    {
        const double* a_data = static_cast<const double*>(cpu_data_ptr_.get());
        const double* b_data =
            static_cast<const double*>(b.cpu_data_ptr_.get());
        double* r_data = static_cast<double*>(r.cpu_data_ptr_.get());

        for (size_t batch = 0; batch < batch_size; ++batch)
        {
            size_t offset_a = batch * m * k;
            size_t offset_b = batch * k * n;
            size_t offset_c = batch * m * n;

            matrix_multiply_cpu(a_data + offset_a, m, k, b_data + offset_b, k,
                                n, r_data + offset_c, m, n);
        }
    }
}

std::vector<size_t> Tensor::calculate_dot_result_shape(const Tensor& b) const
{
    std::vector<size_t> result_shape;

    if (shape_size_ == 2 && b.shape_size_ == 2)
    {
        // 2D × 2D = 2D
        result_shape = {shape_[0], b.shape_[1]};
    }
    else
    {
        // 배치 차원들 계산
        size_t batch_dims_a = shape_size_ - 2;
        size_t batch_dims_b = b.shape_size_ - 2;
        size_t max_batch_dims = std::max(batch_dims_a, batch_dims_b);

        // 브로드캐스팅 규칙 적용
        for (size_t i = 0; i < max_batch_dims; ++i)
        {
            size_t dim_a = (i < batch_dims_a) ? shape_[i] : 1;
            size_t dim_b = (i < batch_dims_b) ? b.shape_[i] : 1;

            if (dim_a == 1)
            {
                result_shape.push_back(dim_b);
            }
            else if (dim_b == 1)
            {
                result_shape.push_back(dim_a);
            }
            else if (dim_a == dim_b)
            {
                result_shape.push_back(dim_a);
            }
            else
            {
                ASSERT_MESSAGE(false, "브로드캐스팅 차원이 맞지 않습니다");
            }
        }

        // 마지막 2차원 추가
        result_shape.push_back(shape_[shape_size_ - 2]);      // A의 행
        result_shape.push_back(b.shape_[b.shape_size_ - 1]);  // B의 열
    }

    return result_shape;
}

Tensor Tensor::dot(const Tensor& b)
{
    ASSERT_MESSAGE(shape_size_ >= 2 && b.shape_size_ >= 2,
                   "최소 2차원 텐서여야 합니다");
    ASSERT_MESSAGE(shape_[shape_size_ - 1] == b.shape_[b.shape_size_ - 2],
                   "행렬 곱셈 차원이 맞지 않습니다");
    ASSERT_MESSAGE(location_ == b.location_, "데이터 위치가 동일해야 합니다");

    // 결과 차원 계산
    std::vector<size_t> result_shape = calculate_dot_result_shape(b);

    // 임시 텐서 생성
    Tensor temp_result(result_shape, type_);

    if (location_ == DataLocation::DEVICE)
    {
        temp_result.allocate_gpu_mem(temp_result.get_total_bytes());
        temp_result.location_ = DataLocation::DEVICE;
    }
    else
    {
        temp_result.allocate_cpu_mem(temp_result.get_total_bytes());
        temp_result.location_ = DataLocation::HOST;
    }

    // 행렬곱 수행
    dot(b, temp_result);

    return temp_result;
}
void Tensor::dot(const Tensor& b, Tensor& r) const
{
    ASSERT_MESSAGE(shape_size_ >= 2 && b.shape_size_ >= 2,
                   "최소 2차원 텐서여야 합니다");
    ASSERT_MESSAGE(shape_[shape_size_ - 1] == b.shape_[b.shape_size_ - 2],
                   "행렬 곱셈 차원이 맞지 않습니다");

    if (shape_size_ == 2 && b.shape_size_ == 2)
    {
        // 일반 2D 행렬 곱셈
        gemm_2d(b, r);
    }
    else if (shape_size_ >= 3 || b.shape_size_ >= 3)
    {
        // 배치 행렬 곱셈
        if (location_ == DataLocation::DEVICE)
        {
            gemm_strided_batched(b, r);
        }
        else
        {
            dot_cpu_batched(b, r);
        }
    }
}

}  // namespace ushionn