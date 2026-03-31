#include <core/tensor.h>
#include <gtest/gtest.h>

TEST(TensorConstructorTest, ConstructorWithValidCPUPointerTest)
{
    const std::vector<uint64_t> shape = {3, 4, 5};
    const ushionn::Device device = {ushionn::Device::DeviceType::HOST, 0};
    ushionn::Tensor t(shape, device, ushionn::DType::FP64);

    EXPECT_EQ(t.shape(), shape);
    EXPECT_EQ(t.device().type, device.type);
    EXPECT_EQ(t.dtype(), ushionn::DType::FP64);

    EXPECT_EQ(t.data(), t.data_ptr<double>());
    EXPECT_EQ(t.numel(), 3 * 4 * 5);
}

TEST(TensorTest, ConsturctorWithNoDataTest) { ; }
//
// TEST(TensorTest, AddAssignTest) {}