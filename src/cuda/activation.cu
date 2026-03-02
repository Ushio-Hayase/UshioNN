#include "layers/activation.h"

/*

namespace ushionn
{

__global__ void relu_gpu_float(const float* x, float* result, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        result[idx] = x[idx] >= 0 ? x[idx] : 0;
    }
}

__global__ void relu_gpu_double(const double* x, double* result, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        result[idx] = x[idx] >= 0 ? x[idx] : 0;
    }
}

__global__ void relu_d_gpu_float(const float* x, float* result, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        result[idx] = x[idx] > 0 ? 1 : 0;
    }
}

__global__ void relu_d_gpu_double(const double* x, double* result, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        result[idx] = x[idx] > 0 ? 1 : 0;
    }
}

void relu(const Tensor& x, Tensor& result)
{
    USHIONN_ASSERT(x.get_shape() == result.get_shape(), "계산을 수행하는 두
텐서의 차원이 일치하지 않습니다.");

    USHIONN_ASSERT(x.get_device() == result.get_device(), "데이터는 동일한
위치에 존재해야합니다.");

    if (x.get_device() == DataLocation::DEVICE && x.get_type() ==
DataType::FLOAT32)
    {
        size_t total_elements = 1;
        for (size_t dim : x.get_shape())
        {
            total_elements *= dim;
        }

        const int block_size = 256;
        const int grid_size = (total_elements + block_size - 1) / block_size;

        relu_gpu_float<<<grid_size, block_size>>>(static_cast<const float*
const>(x.get_gpu_ptr()), static_cast<float*>(result.get_gpu_ptr_mutable()),
                                                  x.get_total_bytes() /
sizeof(float));
    }
    else if (x.get_device() == DataLocation::DEVICE && x.get_type() ==
DataType::FLOAT64)
    {
        size_t total_elements = 1;
        for (size_t dim : x.get_shape())
        {
            total_elements *= dim;
        }

        const int block_size = 256;
        const int grid_size = (total_elements + block_size - 1) / block_size;
        relu_gpu_double<<<grid_size, block_size>>>(static_cast<const double*
const>(x.get_gpu_ptr()), static_cast<double*>(result.get_gpu_ptr_mutable()),
                                                   x.get_total_bytes() /
sizeof(double));
    }
    else if (x.get_device() == DataLocation::HOST && x.get_type() ==
DataType::FLOAT32)
    {
        size_t total_elements = 1;
        for (size_t dim : x.get_shape())
        {
            total_elements *= dim;
        }

        const float* const x_ptr = static_cast<const float*
const>(x.get_cpu_ptr()); float* const result_ptr = static_cast<float*
const>(result.get_cpu_ptr_mutable());

        for (int i = 0; i < total_elements; ++i)
        {
            result_ptr[i] = x_ptr[i] >= 0 ? x_ptr[i] : 0;
        }
    }
    else if (x.get_device() == DataLocation::HOST && x.get_type() ==
DataType::FLOAT64)
    {
        size_t total_elements = 1;
        for (size_t dim : x.get_shape())
        {
            total_elements *= dim;
        }

        const double* const x_ptr = static_cast<const double*
const>(x.get_cpu_ptr()); double* const result_ptr = static_cast<double*
const>(result.get_cpu_ptr_mutable());

        for (int i = 0; i < total_elements; ++i)
        {
            result_ptr[i] = x_ptr[i] >= 0 ? x_ptr[i] : 0;
        }
    }
}

void relu_d(const Tensor& x, Tensor& result)
{
    USHIONN_ASSERT(x.get_shape() == result.get_shape(), "계산을 수행하는 두
텐서의 차원이 일치하지 않습니다.");

    USHIONN_ASSERT(x.get_device() == result.get_device(), "데이터는 동일한
위치에 존재해야합니다.");

    if (x.get_device() == DataLocation::DEVICE && x.get_type() ==
DataType::FLOAT32)
    {
        size_t total_elements = 1;
        for (size_t dim : x.get_shape())
        {
            total_elements *= dim;
        }

        const int block_size = 256;
        const int grid_size = (total_elements + block_size - 1) / block_size;

        relu_d_gpu_float<<<grid_size, block_size>>>(static_cast<const float*
const>(x.get_gpu_ptr()), static_cast<float*>(result.get_gpu_ptr_mutable()),
                                                    x.get_total_bytes() /
sizeof(float));
    }
    else if (x.get_device() == DataLocation::DEVICE && x.get_type() ==
DataType::FLOAT64)
    {
        size_t total_elements = 1;
        for (size_t dim : x.get_shape())
        {
            total_elements *= dim;
        }

        const int block_size = 256;
        const int grid_size = (total_elements + block_size - 1) / block_size;
        relu_d_gpu_double<<<grid_size, block_size>>>(static_cast<const double*
const>(x.get_gpu_ptr()), static_cast<double*>(result.get_gpu_ptr_mutable()),
                                                     x.get_total_bytes() /
sizeof(double));
    }
    else if (x.get_device() == DataLocation::HOST && x.get_type() ==
DataType::FLOAT32)
    {
        size_t total_elements = 1;
        for (size_t dim : x.get_shape())
        {
            total_elements *= dim;
        }

        const float* const x_ptr = static_cast<const float*
const>(x.get_cpu_ptr()); float* const result_ptr = static_cast<float*
const>(result.get_cpu_ptr_mutable());

        for (int i = 0; i < total_elements; ++i)
        {
            result_ptr[i] = x_ptr[i] > 0 ? 1 : 0;
        }
    }
    else if (x.get_device() == DataLocation::HOST && x.get_type() ==
DataType::FLOAT64)
    {
        size_t total_elements = 1;
        for (size_t dim : x.get_shape())
        {
            total_elements *= dim;
        }

        const double* const x_ptr = static_cast<const double*
const>(x.get_cpu_ptr()); double* const result_ptr = static_cast<double*
const>(result.get_cpu_ptr_mutable());

        for (int i = 0; i < total_elements; ++i)
        {
            result_ptr[i] = x_ptr[i] > 0 ? 1 : 0;
        }
    }
}
}  // namespace ushionn

*/