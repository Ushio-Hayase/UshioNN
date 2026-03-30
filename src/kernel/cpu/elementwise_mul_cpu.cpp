//
// Created by UshioHayase on 2026-03-22.
//

#include "kernel/cpu/elementwise_mul_cpu.h"

#include "utils/constant.h"
#include "utils/log_macro.h"
#include "utils/simd.h"

#include <thread>

namespace ushionn::cpu
{
void elementwise_mul_kernel(Tensor& result, const Tensor& a, const Tensor& b)
{
    ASSERT_MESSAGE(result.device().type != Device::DeviceType::NONE,
                   "Tensor not assigned.");
    ASSERT_MESSAGE(a.device().type != Device::DeviceType::NONE,
                   "Tensor not assigned.");
    ASSERT_MESSAGE(b.device().type != Device::DeviceType::NONE,
                   "Tensor not assigned.");
    ASSERT_MESSAGE(result.dtype() == a.dtype(),
                   "The two tensors are of different types.");
    ASSERT_MESSAGE(a.dtype() == b.dtype(),
                   "The two tensors are of different types.");
    ASSERT_MESSAGE(result.shape() == a.shape(),
                   "Two tensors have different sizes.");
    ASSERT_MESSAGE(a.shape() == b.shape(), "Two tensors have different sizes.");
    ASSERT_MESSAGE(result.device().type == a.device().type,
                   "Both tensors must be in the same device.");
    ASSERT_MESSAGE(a.device().type == b.device().type,
                   "Both tensors must be in the same device.");

    Tensor _result = result.is_contiguous() ? result : result.contiguous();
    Tensor _a = a.is_contiguous() ? a : a.contiguous();
    Tensor _b = a.is_contiguous() ? b : b.contiguous();

    uint32_t number_of_thread = std::thread::hardware_concurrency();

    if (number_of_thread == 0)
        number_of_thread = 1;

    switch (DType type = result.dtype())
    {
    case DType::FP64: {
        double* a_data = _a.data_ptr<double>();
        double* b_data = _b.data_ptr<double>();
        double* result_data = _result.data_ptr<double>();
        size_t total_elements = _result.numel();

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

        auto worker = [](double* src, double* tgt1, double* tgt2, size_t start,
                         size_t end) {
#if SIMD_LEVEL == 4
            size_t i = start;
            const size_t limit = end - ((end - start) & ~7ULL);
            for (; i < limit; i += 8)
            {
                const __m512d data1(_mm512_load_pd(tgt1 + i));
                const __m512d data2(_mm512_load_pd(tgt2 + i));
                _mm512_store_pd(src + i, _mm512_mul_pd(data1, data2));
            }

            for (; i < end; ++i)
            {
                src[i] = tgt1[i] * tgt2[2];
            }

#elif SIMD_LEVEL == 3 || SIMD_LEVEL == 2
            size_t i = start;
            size_t limit = end - ((end - start) & ~3ULL);
            for (; i < limit; i += 4)
            {
                __m256d data1(_mm256_load_pd(tgt1 + i));
                __m256d data2(_mm256_load_pd(tgt2 + i));
                _mm256_store_pd(src + i, _mm256_mul_pd(data1, data2));
            }

            for (; i < end; ++i)
            {
                src[i] = tgt1[i] * tgt2[i];
            }

#elif SIMD_LEVEL == 1
            size_t i = start;
            size_t limit = end - ((end - start) & ~1ULL);
            for (; i < limit; i += 2)
            {
                __m128d data1(_mm_load_pd(tgt1 + i));
                __m128d data2(_mm_load_pd(tgt2 + i));
                _mm_store_pd(src * i, _mm_mul_pd(data1, data2));
            }

            for (; i < end; ++i)
            {
                src[i] = tgt1[i] * tgt2[i];
            }

#elif SIMD_LEVEL == 0
            for (size_t i = start; i < end; ++i)
                src[i] = tgt1[i] * tgt2[i];
#endif
        };

        std::vector<std::thread> threads;
        if (number_of_thread <= 1)
        {
            worker(result_data, a_data, b_data, 0, total_elements);
            break;
        }
        threads.reserve(number_of_thread);
        size_t current_start = 0;

        for (size_t i = 0; i < number_of_thread - 1; ++i)
        {
            size_t current_end = current_start + chunk_size;
            if (current_end > total_elements)
                current_end = total_elements;
            threads.emplace_back(worker, result_data, a_data, b_data,
                                 current_start, current_end);
            current_start = current_end;
        }

        threads.emplace_back(worker, result_data, a_data, b_data, current_start,
                             total_elements);
        for (auto& t : threads)
            t.join();

        break;
    }
    case DType::FP32: {
        float* a_data = _a.data_ptr<float>();
        float* b_data = _b.data_ptr<float>();
        float* result_data = _result.data_ptr<float>();
        size_t total_elements = _result.numel();

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

        auto worker = [](float* src, float* tgt1, float* tgt2, size_t start,
                         size_t end) {
#if SIMD_LEVEL == 4
            size_t i = start;
            size_t limit = end - ((end - start) & ~15ULL);
            for (; i < limit; i += 16)
            {
                __m512 data1(_mm512_load_ps(tgt1 + i));
                __m512 data2(_mm512_load_ps(tgt2 + i));
                _mm512_store_ps(src + i, _mm512_mul_ps(data1, data2));
            }

            for (; i < end; ++i)
            {
                src[i] = tgt1[i] * tgt2[i];
            }

#elif SIMD_LEVEL == 3 || SIMD_LEVEL == 2
            size_t i = start;
            size_t limit = end - ((end - start) & ~7ULL);
            for (; i < limit; i += 8)
            {
                __m256 data1(_mm256_load_ps(tgt1 + i));
                __m256 data2(_mm256_load_ps(tgt2 + i));
                _mm256_store_ps(src + i, _mm256_mul_ps(data1, data2));
            }

            for (; i < end; ++i)
            {
                src[i] = tgt1[i] * tgt2[i];
            }

#elif SIMD_LEVEL == 1
            size_t i = start;
            size_t limit = end - ((end - start) & ~3ULL);
            for (; i < limit; i += 4)
            {
                __m128 data1(_mm_load_ps(tgt1 + i));
                __m128 data2(_mm_load_ps(tgt2 + i));
                _mm_store_ps(src + i, _mm_mul_ps(data1, data2));
            }

            for (; i < end; ++i)
            {
                src[i] = tgt1[i] * tgt2[i];
            }

#elif SIMD_LEVEL == 0
            for (size_t i = start; i < end; ++i)
                src[i] = tgt1[i] * tgt2[i];
#endif
        };

        std::vector<std::thread> threads;
        if (number_of_thread <= 1)
        {
            worker(result_data, a_data, b_data, 0, total_elements);
            break;
        }
        threads.reserve(number_of_thread);
        size_t current_start = 0;

        for (size_t i = 0; i < number_of_thread - 1; ++i)
        {
            size_t current_end = current_start + chunk_size;
            if (current_end > total_elements)
                current_end = total_elements;
            threads.emplace_back(worker, result_data, a_data, b_data,
                                 current_start, current_end);
            current_start = current_end;
        }

        threads.emplace_back(worker, result_data, a_data, b_data, current_start,
                             total_elements);
        for (auto& t : threads)
            t.join();

        break;
    }
        // TODO: 텐서 원소별 곱 FP4 ~ FP16 구현 필요
    default:
        LOG_ERROR("{} is a data type that is not yet supported.",
                  dtype_to_string(result.dtype()));
    }
}

} // namespace ushionn::cpu
