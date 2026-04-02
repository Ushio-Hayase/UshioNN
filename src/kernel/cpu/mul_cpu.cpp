#include "kernel/cpu/mul_cpu.h"

#include "../../../include/utils/simd.h"
#include "utils/constant.h"
#include "utils/log_macro.h"

#include <thread>

namespace ushionn::cpu
{
void scalar_mul_kernel(Tensor& result, const Tensor& src, float scalar)
{
    ASSERT_MESSAGE(result.device().type != Device::DeviceType::NONE,
                   "Tensor not assigned.");
    ASSERT_MESSAGE(src.device().type != Device::DeviceType::NONE,
                   "Tensor not assigned.");
    ASSERT_MESSAGE(result.dtype() == src.dtype(),
                   "Tensors have different type.");
    ASSERT_MESSAGE(result.device().type == src.device().type,
                   "Both tensors must be in the same device.");
    ASSERT_MESSAGE(result.shape() == src.shape(),
                   "Both tensors must have same shape");

    Tensor _src = src.is_contiguous() ? src : src.contiguous();

    uint32_t number_of_thread = std::thread::hardware_concurrency();

    if (number_of_thread == 0)
        number_of_thread = 1;

    DType type = result.dtype();

    switch (type)
    {
    case DType::FP64: {
        double* src_data = _src.data_ptr<double>();
        double* result_data = result.data_ptr<double>();
        size_t total_elements = result.numel();

        size_t align_step = 1;
#if SIMD_LEVEL == 4
        align_step = 8;
#elif SIMD_LEVEL == 3 || SIMD_LEVEL == 2
        align_step = 4;
#elif SIMD_LEVEL == 1
        align_step = 2;
#endif

        if (total_elements < MULTI_THREAD_BASELINE)
            number_of_thread = 1;

        size_t chunk_size = total_elements / number_of_thread;
        chunk_size &= ~(align_step - 1);

        auto worker = [](double* result, const double* src, const float scalar,
                         const size_t start, const size_t end) {
#if SIMD_LEVEL == 4
            size_t i = start;
            const size_t limit = end - ((end - start) & ~7ULL);
            const __m512d data2(_mm512_set1_pd(scalar));

            for (; i < limit; i += 8)
            {
                const __m512d data1(_mm512_load_pd(src + i));
                _mm512_store_pd(result + i, _mm512_mul_pd(data1, data2));
            }

            for (; i < end; ++i)
            {
                result[i] = src[i] + scalar;
            }

#elif SIMD_LEVEL == 3 || SIMD_LEVEL == 2
            size_t i = start;
            const size_t limit = end - ((end - start) & ~3ULL);
            const __m256d data2(_mm256_set1_pd(scalar));

            for (; i < limit; i += 4)
            {
                const __m256d data1(_mm256_load_pd(src + i));
                _mm256_store_pd(result + i, _mm256_mul_pd(data1, data2));
            }

            for (; i < end; ++i)
            {
                result[i] = src[i] + scalar;
            }

#elif SIMD_LEVEL == 1
            size_t i = start;
            const size_t limit = end - ((end - start) & ~1ULL);
            const __m128d data2(_mm_set1_pd(scalar));

            for (; i < limit; i += 2)
            {
                const __m128d data1(_mm_load_pd(src + i));
                _mm_store_pd(src + i, _mm_add_pd(data1, data2));
            }

            for (; i < end; ++i)
            {
                result[i] = src[i] + scalar;
            }

#elif SIMD_LEVEL == 0
            for (size_t i = start; i < end; ++i)
                result[i] = src[i] + scalar;
#endif
        };

        std::vector<std::thread> threads;
        if (number_of_thread <= 1)
        {
            worker(result_data, src_data, scalar, 0, total_elements);
            break;
        }
        threads.reserve(number_of_thread);
        size_t current_start = 0;

        for (size_t i = 0; i < number_of_thread - 1; ++i)
        {
            size_t current_end = current_start + chunk_size;
            if (current_end > total_elements)
                current_end = total_elements;
            threads.emplace_back(worker, result_data, src_data, scalar,
                                 current_start, current_end);
            current_start = current_end;
        }

        threads.emplace_back(worker, result_data, src_data, scalar,
                             current_start, total_elements);
        for (auto& t : threads)
            t.join();

        break;
    }
    case DType::FP32: {
        float* src_data = _src.data_ptr<float>();
        float* result_data = result.data_ptr<float>();
        size_t total_elements = result.numel();

        size_t align_step = 1;
#if SIMD_LEVEL == 4
        align_step = 16;
#elif SIMD_LEVEL == 3 || SIMD_LEVEL == 2
        align_step = 8;
#elif SIMD_LEVEL == 1
        align_step = 4;
#endif

        if (total_elements < MULTI_THREAD_BASELINE)
            number_of_thread = 1;

        size_t chunk_size = total_elements / number_of_thread;
        chunk_size &= ~(align_step - 1);

        auto worker = [](float* result, const float* src, const float scalar,
                         const size_t start, const size_t end) {
#if SIMD_LEVEL == 4
            size_t i = start;
            const size_t limit = end - ((end - start) & ~15ULL);
            const __m512 data2(_mm512_set1_ps(scalar));

            for (; i < limit; i += 16)
            {
                const __m512 data1(_mm512_load_ps(src + i));
                _mm512_store_ps(result + i, _mm512_mul_ps(data1, data2));
            }

            for (; i < end; ++i)
            {
                result[i] = src[i] + scalar;
            }

#elif SIMD_LEVEL == 3 || SIMD_LEVEL == 2
            size_t i = start;
            const size_t limit = end - ((end - start) & ~7ULL);
            const __m256 data2(_mm256_set1_ps(scalar));

            for (; i < limit; i += 4)
            {
                const __m256 data1(_mm256_load_ps(src + i));
                _mm256_store_ps(result + i, _mm256_mul_ps(data1, data2));
            }

            for (; i < end; ++i)
            {
                result[i] = src[i] + scalar;
            }

#elif SIMD_LEVEL == 1
            size_t i = start;
            const size_t limit = end - ((end - start) & ~3ULL);
            const __m128 data2(_mm_set1_pd(scalar));

            for (; i < limit; i += 2)
            {
                const __m128 data1(_mm_load_ps(src + i));
                _mm_store_ps(src + i, _mm_add_ps(data1, data2));
            }

            for (; i < end; ++i)
            {
                result[i] = src[i] + scalar;
            }

#elif SIMD_LEVEL == 0
            for (size_t i = start; i < end; ++i)
                result[i] = src[i] + scalar;
#endif
        };

        std::vector<std::thread> threads;
        if (number_of_thread <= 1)
        {
            worker(result_data, src_data, scalar, 0, total_elements);
            break;
        }
        threads.reserve(number_of_thread);
        size_t current_start = 0;

        for (size_t i = 0; i < number_of_thread - 1; ++i)
        {
            size_t current_end = current_start + chunk_size;
            if (current_end > total_elements)
                current_end = total_elements;
            threads.emplace_back(worker, result_data, src_data, scalar,
                                 current_start, current_end);
            current_start = current_end;
        }

        threads.emplace_back(worker, result_data, src_data, scalar,
                             current_start, total_elements);
        for (auto& t : threads)
            t.join();

        break;
    }
        // TODO: 텐서 스칼라배 FP4 ~ FP16 구현 필요
    default: {
        LOG_ERROR("{} is a data type that is not yet supported.",
                  dtype_to_string(result.dtype()));
    }
    }
}
} // namespace ushionn::cpu
