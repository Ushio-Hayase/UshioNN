#include <core/tensor.h>
#include <gtest/gtest.h>

TEST(TensorConstructorTest, ConstructorWithValidCPUPointerTest)
{
    const std::vector<uint64_t> shape = {3, 4, 5};
    const ushionn::Device device = {ushionn::Device::DeviceType::HOST, 0};
    ushionn::Tensor t(shape, device, ushionn::DType::FP64);

    EXPECT_EQ(t.shape(), shape);
    EXPECT_EQ(t.device().type, device.type);
}

TEST(TensorTest, ConsturctorWithNoDataTest)
{
    ushionn::Tensor tensor{};

    EXPECT_EQ(tensor.elementSize(), 4);
    EXPECT_EQ(tensor.getCpuPtr(), nullptr);
    EXPECT_EQ(tensor.getGpuPtr(), nullptr);

    EXPECT_EQ(tensor.getTotalBytes(), 0);
    EXPECT_EQ(tensor.getType(), ushionn::DType::FP32);
}
//
// TEST(TensorTest, AddAssignTest) {}