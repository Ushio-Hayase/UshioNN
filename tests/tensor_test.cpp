#include <core/tensor.h>
#include <gtest/gtest.h>

TEST(TensorConstructorTest, ConstructorWithValidCPUPointerTest)
{
    double* src = new double[24];
    std::vector<size_t> shape = {2, 3, 4};

    nunet::Tensor tensor(shape, src);

    EXPECT_EQ(tensor.elementSize(), 8);
    EXPECT_EQ(tensor.getTotalBytes(), 24 * 8);

    EXPECT_EQ(tensor.getShape(), shape);
    EXPECT_EQ(tensor.getShapeSize(), 3);
    EXPECT_EQ(tensor.getType(), nunet::DType::FP64);

    EXPECT_NE(tensor.getCpuPtr(), nullptr);
    EXPECT_EQ(tensor.getGpuPtr(), nullptr);
}

TEST(TensorTest, ConsturctorWithNoDataTest)
{
    nunet::Tensor tensor{};

    EXPECT_EQ(tensor.elementSize(), 4);
    EXPECT_EQ(tensor.getCpuPtr(), nullptr);
    EXPECT_EQ(tensor.getGpuPtr(), nullptr);

    EXPECT_EQ(tensor.getTotalBytes(), 0);
    EXPECT_EQ(tensor.getType(), nunet::DType::FP32);
}
//
// TEST(TensorTest, AddAssignTest) {}