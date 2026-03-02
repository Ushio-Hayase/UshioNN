#include "core/tensor.h"

/*

__global__ void multiply_kernel_float(const float* a, const float* b, float*
result, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        result[idx] = a[idx] * b[idx];
    }
}

__global__ void multiply_kernel_double(const double* a, const double* b, double*
result, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        result[idx] = a[idx] * b[idx];
    }
}

__global__ void transpose_kernel_float(const float* input, float* output, size_t
rows, size_t cols, size_t batch_size)
{
    size_t batch = blockIdx.z;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    if (batch < batch_size && row < rows && col < cols)
    {
        size_t input_offset = batch * rows * cols;
        size_t output_offset = batch * rows * cols;

        // (row, col) -> (col, row)
        output[output_offset + col * rows + row] = input[input_offset + row *
cols + col];
    }
}

__global__ void transpose_kernel_double(const double* input, double* output,
size_t rows, size_t cols, size_t batch_size)
{
    size_t batch = blockIdx.z;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    if (batch < batch_size && row < rows && col < cols)
    {
        size_t input_offset = batch * rows * cols;
        size_t output_offset = batch * rows * cols;

        // (row, col) -> (col, row)
        output[output_offset + col * rows + row] = input[input_offset + row *
cols + col];
    }
}

// 최적화된 공유 메모리 버전 (2D 행렬용)
template <int TILE_DIM>
__global__ void transpose_shared_kernel_float(const float* input, float* output,
size_t rows, size_t cols)
{
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // +1로 뱅크 충돌 방지

    size_t x = blockIdx.x * TILE_DIM + threadIdx.x;
    size_t y = blockIdx.y * TILE_DIM + threadIdx.y;

    // 타일을 공유 메모리로 로드
    if (x < cols && y < rows)
    {
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }

    __syncthreads();

    // 전치된 위치에 쓰기
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    if (x < rows && y < cols)
    {
        output[y * rows + x] = tile[threadIdx.x][threadIdx.y];
    }
}

__global__ void permute_kernel_float(const float* input, float* output, size_t*
input_strides, size_t* output_strides, size_t* shape, size_t shape_size, size_t
dim1, size_t dim2, size_t total_elements)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements)
    {
        // 1차원 인덱스를 다차원 인덱스로 변환
        size_t multi_idx[8];  // 최대 8차원까지 지원
        size_t temp_idx = idx;

        for (size_t i = 0; i < shape_size; ++i)
        {
            multi_idx[i] = temp_idx / input_strides[i];
            temp_idx %= input_strides[i];
        }

        // 차원 교환
        size_t temp = multi_idx[dim1];
        multi_idx[dim1] = multi_idx[dim2];
        multi_idx[dim2] = temp;

        // 다차원 인덱스를 결과의 1차원 인덱스로 변환
        size_t output_idx = 0;
        for (size_t i = 0; i < shape_size; ++i)
        {
            output_idx += multi_idx[i] * output_strides[i];
        }

        output[output_idx] = input[idx];
    }
}

__global__ void permute_kernel_double(const double* input, double* output,
size_t* input_strides, size_t* output_strides, size_t* shape, size_t shape_size,
size_t dim1, size_t dim2, size_t total_elements)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements)
    {
        // 1차원 인덱스를 다차원 인덱스로 변환
        size_t multi_idx[8];  // 최대 8차원까지 지원
        size_t temp_idx = idx;

        for (size_t i = 0; i < shape_size; ++i)
        {
            multi_idx[i] = temp_idx / input_strides[i];
            temp_idx %= input_strides[i];
        }

        // 차원 교환
        size_t temp = multi_idx[dim1];
        multi_idx[dim1] = multi_idx[dim2];
        multi_idx[dim2] = temp;

        // 다차원 인덱스를 결과의 1차원 인덱스로 변환
        size_t output_idx = 0;
        for (size_t i = 0; i < shape_size; ++i)
        {
            output_idx += multi_idx[i] * output_strides[i];
        }

        output[output_idx] = input[idx];
    }
}

namespace ushionn
{
void Tensor::multiply(const Tensor& b, Tensor& r)
{
    USHIONN_ASSERT(shape_ == b.shape_, "계산을 수행하는 두 텐서의 차원이
일치하지 않습니다."); USHIONN_ASSERT(shape_ == r.shape_, "계산을 수행하는 두
텐서의 차원이 일치하지 않습니다.");

    USHIONN_ASSERT(location_ == b.location_, "데이터는 동일한 위치에
존재해야합니다."); USHIONN_ASSERT(location_ == r.location_, "데이터는 동일한
위치에 존재해야합니다.");

    if (location_ == DataLocation::DEVICE && type_ == DataType::FLOAT32)
    {
        size_t total_elements = 1;
        for (size_t dim : shape_)
        {
            total_elements *= dim;
        }

        const int block_size = 256;
        const int grid_size = (total_elements + block_size - 1) / block_size;

        multiply_kernel_float<<<grid_size, block_size>>>(static_cast<const
float*>(gpu_data_ptr_.get()), static_cast<const float*>(b.gpu_data_ptr_.get()),
                                                         static_cast<float*>(r.gpu_data_ptr_.get()),
total_elements);
    }
    else if (location_ == DataLocation::DEVICE && type_ == DataType::FLOAT64)
    {
        size_t total_elements = 1;
        for (size_t dim : shape_)
        {
            total_elements *= dim;
        }

        const int block_size = 256;
        const int grid_size = (total_elements + block_size - 1) / block_size;

        multiply_kernel_double<<<grid_size, block_size>>>(static_cast<const
double*>(gpu_data_ptr_.get()), static_cast<const
double*>(b.gpu_data_ptr_.get()), static_cast<double*>(r.gpu_data_ptr_.get()),
total_elements);
    }
    else if (location_ == DataLocation::HOST)
    {
        size_t total_elements = 1;
        for (size_t dim : shape_)
        {
            total_elements *= dim;
        }

        if (type_ == DataType::FLOAT32)
        {
            const float* a_data = static_cast<const
float*>(cpu_data_ptr_.get()); const float* b_data = static_cast<const
float*>(b.cpu_data_ptr_.get()); float* r_data =
static_cast<float*>(r.cpu_data_ptr_.get());

            for (size_t i = 0; i < total_elements; ++i)
            {
                r_data[i] = a_data[i] * b_data[i];
            }
        }
        else if (type_ == DataType::FLOAT64)
        {
            const double* a_data = static_cast<const
double*>(cpu_data_ptr_.get()); const double* b_data = static_cast<const
double*>(b.cpu_data_ptr_.get()); double* r_data =
static_cast<double*>(r.cpu_data_ptr_.get());

            for (size_t i = 0; i < total_elements; ++i)
            {
                r_data[i] = a_data[i] * b_data[i];
            }
        }
    }
}

void Tensor::gemm_2d(const Tensor& b, Tensor& r) const
{
    size_t m = shape_[0];
    size_t k = shape_[1];
    size_t n = b.shape_[1];
    if (type_ == DataType::FLOAT32)
    {
        float alpha = 1.0f, beta = 0.0f;

        cublasSgemm(r.cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
                    static_cast<const float*>(gpu_data_ptr_.get()), m,
static_cast<const float*>(b.gpu_data_ptr_.get()), k, &beta,
static_cast<float*>(r.gpu_data_ptr_.get()), m);
    }
    else if (type_ == DataType::FLOAT64)
    {
        double alpha = 1.0f, beta = 0.0f;
        cublasDgemm(r.cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
                    static_cast<const double*>(gpu_data_ptr_.get()), m,
                    static_cast<const double*>(b.gpu_data_ptr_.get()), k, &beta,
                    static_cast<double*>(r.gpu_data_ptr_.get()), m);
    }
}

void Tensor::gemm_strided_batched(const Tensor& b, Tensor& r) const
{
    size_t batch_size = 1;
    for (size_t i = 0; i < shape_size_ - 2; ++i)
    {
        batch_size *= shape_[i];
    }

    size_t m = shape_[shape_size_ - 2];
    size_t k = shape_[shape_size_ - 1];
    size_t n = b.shape_[b.shape_size_ - 1];

    // Stride 계산
    long long stride_a = m * k;
    long long stride_b = k * n;
    long long stride_c = m * n;

    if (type_ == DataType::FLOAT32)
    {
        float alpha = 1.0f, beta = 0.0f;

        // Strided Batched GEMM 수행
        cublasSgemmStridedBatched(r.cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N, m,
n, k, &alpha, static_cast<const float*>(gpu_data_ptr_.get()), m, stride_a,
                                  static_cast<const
float*>(b.gpu_data_ptr_.get()), k, stride_b, &beta,
                                  static_cast<float*>(r.gpu_data_ptr_.get()), m,
stride_c, batch_size);
    }
    else if (type_ == DataType::FLOAT64)
    {
        double alpha = 1.0f, beta = 0.0f;
        cublasDgemmStridedBatched(r.cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N, m,
n, k, &alpha, static_cast<const double*>(gpu_data_ptr_.get()), m, stride_a,
                                  static_cast<const
double*>(b.gpu_data_ptr_.get()), k, stride_b, &beta,
                                  static_cast<double*>(r.gpu_data_ptr_.get()),
m, stride_c, batch_size);
    }
}

void Tensor::transpose_gpu(Tensor& r) const
{
    USHIONN_ASSERT(shape_size_ >= 2, "최소 2차원 텐서여야 합니다");

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
        if (batch_size == 1 && rows >= 32 && cols >= 32)
        {
            // 큰 2D 행렬은 최적화된 공유 메모리 버전 사용
            const int TILE_DIM = 32;
            dim3 block(TILE_DIM, TILE_DIM);
            dim3 grid((cols + TILE_DIM - 1) / TILE_DIM, (rows + TILE_DIM - 1) /
TILE_DIM);

            transpose_shared_kernel_float<TILE_DIM><<<grid, block>>>(
                static_cast<const float*>(gpu_data_ptr_.get()),
static_cast<float*>(r.gpu_data_ptr_.get()), rows, cols);
        }
        else
        {
            // 일반적인 배치 처리
            dim3 block(16, 16);
            dim3 grid((cols + 15) / 16, (rows + 15) / 16, batch_size);

            transpose_kernel_float<<<grid, block>>>(static_cast<const
float*>(gpu_data_ptr_.get()), static_cast<float*>(r.gpu_data_ptr_.get()), rows,
cols, batch_size);
        }
    }
    else if (type_ == DataType::FLOAT64)
    {
        dim3 block(16, 16);
        dim3 grid((cols + 15) / 16, (rows + 15) / 16, batch_size);

        transpose_kernel_double<<<grid, block>>>(static_cast<const
double*>(gpu_data_ptr_.get()), static_cast<double*>(r.gpu_data_ptr_.get()),
rows, cols, batch_size);
    }

    cudaDeviceSynchronize();
}

void Tensor::permute_gpu_general(size_t dim1, size_t dim2, Tensor& result)
{
    // 전체 원소 개수
    size_t total_elements = 1;
    for (size_t dim : shape_)
    {
        total_elements *= dim;
    }

    // Stride 계산 (원본)
    std::vector<size_t> input_strides(shape_size_);
    input_strides[shape_size_ - 1] = 1;
    for (int i = shape_size_ - 2; i >= 0; --i)
    {
        input_strides[i] = input_strides[i + 1] * shape_[i + 1];
    }

    // Stride 계산 (결과)
    std::vector<size_t> output_strides(shape_size_);
    output_strides[shape_size_ - 1] = 1;
    for (int i = shape_size_ - 2; i >= 0; --i)
    {
        output_strides[i] = output_strides[i + 1] * result.shape_[i + 1];
    }

    // GPU 메모리에 stride와 shape 정보 복사
    size_t* d_input_strides;
    size_t* d_output_strides;
    size_t* d_shape;

    cudaMalloc(&d_input_strides, shape_size_ * sizeof(size_t));
    cudaMalloc(&d_output_strides, shape_size_ * sizeof(size_t));
    cudaMalloc(&d_shape, shape_size_ * sizeof(size_t));

    cudaMemcpy(d_input_strides, input_strides.data(), shape_size_ *
sizeof(size_t), cudaMemcpyHostToDevice); cudaMemcpy(d_output_strides,
output_strides.data(), shape_size_ * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_shape, shape_.data(), shape_size_ * sizeof(size_t),
cudaMemcpyHostToDevice);

    // 커널 실행
    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;

    if (type_ == DataType::FLOAT32)
    {
        permute_kernel_float<<<grid_size, block_size>>>(
            static_cast<const float*>(gpu_data_ptr_.get()),
static_cast<float*>(result.gpu_data_ptr_.get()), d_input_strides,
d_output_strides, d_shape, shape_size_, dim1, dim2, total_elements);
    }
    else if (type_ == DataType::FLOAT64)
    {
        permute_kernel_double<<<grid_size, block_size>>>(
            static_cast<const double*>(gpu_data_ptr_.get()),
static_cast<double*>(result.gpu_data_ptr_.get()), d_input_strides,
d_output_strides, d_shape, shape_size_, dim1, dim2, total_elements);
    }

    cudaDeviceSynchronize();

    // GPU 메모리 해제
    cudaFree(d_input_strides);
    cudaFree(d_output_strides);
    cudaFree(d_shape);
}
}  // namespace ushionn

*/