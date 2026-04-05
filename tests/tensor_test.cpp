#include "function/add.h"
#include "function/matmul.h"

#include <core/tensor.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

TEST(TensorConstructorTest, ConstructorWithCPUTest)
{
    const std::vector<uint64_t> shape = {3, 4, 5};
    const ushionn::Device device = {ushionn::Device::DeviceType::HOST, 0};
    ushionn::Tensor t(shape, device, ushionn::DType::FP64);

    EXPECT_EQ(t.shape(), shape);
    EXPECT_EQ(t.device().type, device.type);
    EXPECT_EQ(t.dtype(), ushionn::DType::FP64);

    EXPECT_EQ(t.data(), t.data_ptr<double>());
    EXPECT_EQ(t.numel(), 3 * 4 * 5);

    ushionn::Tensor t_copy(t);

    EXPECT_EQ(t_copy.shape(), shape);
    EXPECT_EQ(t_copy.device().type, device.type);
    EXPECT_EQ(t_copy.dtype(), ushionn::DType::FP64);

    EXPECT_NE(t_copy.data(), t.data_ptr<double>());
    EXPECT_EQ(t_copy.numel(), 3 * 4 * 5);

    EXPECT_NE(t.data_ptr<double>(), t_copy.data_ptr<double>());

    double* data = new double[60];
    std::memset(data, 0, sizeof(double) * 60);

    ushionn::Tensor t1(shape, data, device);
    EXPECT_EQ(t1.data_ptr<double>()[0], 0);

    delete[] data;
}

TEST(TensorOperationTest, HOSTAddTest)
{
    const std::vector<uint64_t> shape = {4, 5, 6};
    const ushionn::Device device = {ushionn::Device::DeviceType::HOST, 0};
    double* data = new double[120];
    std::fill_n(data, 120, 1.0);

    ushionn::Tensor t1(shape, data, device);

    ushionn::Tensor t2(shape, data, device);

    ushionn::Tensor result = t1 + t2;

    double* result_data = result.data_ptr<double>();

    for (int i = 0; i < 120; ++i)
        EXPECT_EQ(result_data[i], 2.0);

    delete[] data;
}

TEST(TensorOperationTest, HOSTElemwiseMulTest)
{

    const std::vector<uint64_t> shape = {4, 5, 6};
    const ushionn::Device device = {ushionn::Device::DeviceType::HOST, 0};
    double* data = new double[120];
    std::fill_n(data, 120, 1.0);

    ushionn::Tensor t1(shape, data, device);

    ushionn::Tensor t2(shape, data, device);

    ushionn::Tensor result = t1 * t2;

    double* result_data = result.data_ptr<double>();

    for (int i = 0; i < 20; ++i)
        EXPECT_EQ(result_data[i], 1.0);

    delete[] data;
}

TEST(TensorOperationTest, HOSTMatmulTest)
{
    const ushionn::Device device = {ushionn::Device::DeviceType::HOST, 0};

    double* data_a = new double[4];
    double* data_b = new double[4];
    double* data_result = new double[4];

    for (int i = 1; i <= 4; ++i)
    {
        data_a[i - 1] = i;
        data_b[i - 1] = 4 + i;
    }

    data_result[0] = 19;
    data_result[1] = 22;
    data_result[2] = 43;
    data_result[3] = 50;

    ushionn::Tensor t1({1, 2, 2}, data_a, device);
    ushionn::Tensor t2({1, 2, 2}, data_b, device);
    ushionn::Tensor result = ushionn::function::matmul(t1, t2);

    auto result_ptr = result.data_ptr<double>();

    for (int i = 0; i < 4; ++i)
        EXPECT_EQ(result_ptr[i], data_result[i]);

    delete[] data_a;
    delete[] data_b;
    delete[] data_result;
}

#if defined(USE_CUDA)
TEST(TensorConstructorTest, ConsturctorWithGPUTest)
{
    const std::vector<uint64_t> shape = {3, 4, 5};
    const ushionn::Device device = {ushionn::Device::DeviceType::DEVICE, 0};
    ushionn::Tensor t(shape, device, ushionn::DType::FP64);

    EXPECT_EQ(t.shape(), shape);
    EXPECT_EQ(t.device().type, device.type);
    EXPECT_EQ(t.dtype(), ushionn::DType::FP64);

    EXPECT_EQ(t.data(), t.data_ptr<double>());
    EXPECT_EQ(t.numel(), 3 * 4 * 5);

    ushionn::Tensor t_copy(t);

    EXPECT_EQ(t_copy.shape(), shape);
    EXPECT_EQ(t_copy.device().type, device.type);
    EXPECT_EQ(t_copy.dtype(), ushionn::DType::FP64);

    EXPECT_NE(t_copy.data_ptr<double>(), t.data_ptr<double>());
    EXPECT_EQ(t_copy.numel(), 3 * 4 * 5);

    EXPECT_NE(t.data_ptr<double>(), t_copy.data_ptr<double>());
    double* data = nullptr;
    cudaMalloc((void**)&data, sizeof(double) * 60);
    cudaMemset(data, 0, sizeof(double) * 60);

    ushionn::Tensor t1(shape, data, device);

    double* ptr = new double[60];
    cudaMemcpy(ptr, t1.data_ptr<double>(), sizeof(double) * 60,
               cudaMemcpyDeviceToHost);

    EXPECT_EQ(ptr[0], 0);

    delete[] ptr;
    cudaFree(data);
}

TEST(TensorOperationTest, DEVICEAddTest)
{

    const std::vector<uint64_t> shape = {4, 5, 6};
    const ushionn::Device device = {ushionn::Device::DeviceType::DEVICE, 0};
    double* cpu_data = new double[120];
    std::fill_n(cpu_data, 120, 1.0);

    double* data = nullptr;
    cudaMalloc((void**)&data, sizeof(double) * 120);
    cudaMemcpy(data, cpu_data, sizeof(double) * 120, cudaMemcpyHostToDevice);

    delete[] cpu_data;

    ushionn::Tensor t1(shape, data, device);

    ushionn::Tensor t2(shape, data, device);
    cudaFree(data);
    ushionn::Tensor result = t1 + t2;

    double* result_data = result.data_ptr<double>();
    double* result_cpu_data = new double[120];

    cudaMemcpy(result_cpu_data, result_data, sizeof(double) * 120,
               cudaMemcpyDeviceToHost);

    for (int i = 0; i < 120; ++i)
        EXPECT_EQ(result_cpu_data[i], 2.0);

    delete[] result_cpu_data;
}
TEST(TensorOperationTest, DEVICEElemwiseMulTest)
{
    const std::vector<uint64_t> shape = {4, 5, 6};
    const ushionn::Device device = {ushionn::Device::DeviceType::DEVICE, 0};
    double* cpu_data = new double[120];
    std::fill_n(cpu_data, 120, 1.0);

    double* data = nullptr;
    cudaMalloc((void**)&data, sizeof(double) * 120);
    cudaMemcpy(data, cpu_data, sizeof(double) * 120, cudaMemcpyHostToDevice);

    delete[] cpu_data;

    ushionn::Tensor t1(shape, data, device);

    ushionn::Tensor t2(shape, data, device);
    cudaFree(data);
    ushionn::Tensor result = t1 * t2;

    double* result_data = result.data_ptr<double>();
    double* result_cpu_data = new double[120];

    cudaMemcpy(result_cpu_data, result_data, sizeof(double) * 120,
               cudaMemcpyDeviceToHost);

    for (int i = 0; i < 120; ++i)
        EXPECT_EQ(result_cpu_data[i], 1.0);

    delete[] result_cpu_data;
}
#endif