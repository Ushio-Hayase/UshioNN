#include "kernel/cpu/mul_cpu.h"

#include "core/simd.h"
#include "utils/constant.h"
#include "utils/log_macro.h"

#include <thread>

namespace ushionn
{
void scalr_mul_kernel(Tensor& result, const Tensor& src, float scalar)
{
    ASSERT_MESSAGE(result.device() != Device::NONE,
                   "Tensor not assigned.");
    ASSERT_MESSAGE(src.device() != Device::NONE, "Tensor not assigned.");
    ASSERT_MESSAGE(result.dtype() == src.dtype(),
                   "Tensors have different type.");
    ASSERT_MESSAGE(result.device() == src.device(),
                   "Both tensors must be in the same device.");
    ASSERT_MESSAGE(result.shape() == src.shape(),
                   "Both tensors must have same shape");

    uint32_t num_of_threads = std::thread::hardware_concurrency();

    if (num_of_threads == 0)
        num_of_threads = 1;

    DType type = result.dtype();

    switch (type)
    {
    case DType::FP64: {
        double* src_data = src.data_ptr<double>();
        double* result_data = src.data_ptr<double>();

        size_t total_elements = result.numel();

        int aligned_step = 1;

#if SIMD_LEVEL == 4
        aligned_step = 8;
#elif SIMD_LEVEL == 3 || SIMD_LEVEL == 2
        aligned_step = 4;
#elif SIMD_LEVEL == 1
        aligned_step = 2;
#endif

        if (total_elements < MULTI_THREAD_BASELINE)
            num_of_threads = 1;

        size_t chunk_size = total_elements / num_of_threads;
        chunk_size &= ~(aligned_step - 1);

        auto worker = []() {}
    }
    }
}
} // namespace nunet
