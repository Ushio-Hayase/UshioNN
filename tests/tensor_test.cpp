#include <core/tensor.h>
#include <gtest/gtest.h>

TEST(TensorConstructorTest, ConstructorWithValidCPUPointerTest)
{
    double* src = new double[24];
    std::vector<size_t> shape = {2, 3, 4};

    nunet::Tensor tensor(shape, src);

    EXPECT_EQ(tensor.elementSize(), 8);
    EXPECT_EQ(tensor.getTotalBytes(), 24 * 8);

    EXPECT_NE(tensor.getCpuPtr(), nullptr);
    EXPECT_EQ(tensor.getGpuPtr(), nullptr);
}

TEST(TensorTest, GetValidElemSize) {}
//
// TEST(TensorTest, AddAssignTest) {}