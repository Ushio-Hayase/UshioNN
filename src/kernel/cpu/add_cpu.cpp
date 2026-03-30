//
// Created by UshioHayase on 3/8/2026.
//
#include "kernel/cpu/add_cpu.h"

#include "../../../include/utils/simd.h"
#include "utils/constant.h"
#include "utils/log_macro.h"

#include <thread>

namespace ushionn
{
void cpu::add_kernel(Tensor& result, const Tensor& tensor1,
                     const Tensor& tensor2)
{
    ASSERT_MESSAGE(result.device().type != Device::DeviceType::NONE,
                   "Tensor not assigned.");
    ASSERT_MESSAGE(tensor1.device().type != Device::DeviceType::NONE,
                   "Tensor not assigned.");
    ASSERT_MESSAGE(tensor2.device().type != Device::DeviceType::NONE,
                   "Tensor not assigned.");
    ASSERT_MESSAGE(result.dtype() == tensor1.dtype(),
                   "The two tensors are of different types.");
    ASSERT_MESSAGE(tensor1.dtype() == tensor2.dtype(),
                   "The two tensors are of different types.");
    ASSERT_MESSAGE(result.shape() == tensor1.shape(),
                   "Two tensors have different sizes.");
    ASSERT_MESSAGE(tensor1.shape() == tensor2.shape(),
                   "Two tensors have different sizes.");
    ASSERT_MESSAGE(result.device().type == tensor1.device().type,
                   "Both tensors must be in the same device.");
    ASSERT_MESSAGE(tensor1.device().type == tensor2.device().type,
                   "Both tensors must be in the same device.");

    Tensor _result = result.is_contiguous() ? result : result.contiguous();
    Tensor _tensor1 = tensor1.is_contiguous() ? tensor1 : tensor1.contiguous();
    Tensor _tensor2 = tensor1.is_contiguous() ? tensor2 : tensor2.contiguous();

    uint32_t number_of_thread = std::thread::hardware_concurrency();

    if (number_of_thread == 0)
        number_of_thread = 1;

    DType type = result.dtype();

    switch (type)
    {
    case DType::FP64: {
        double* tensor1_data = _tensor1.data_ptr<double>();
        double* tensor2_data = _tensor2.data_ptr<double>();
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
                _mm512_store_pd(src + i, _mm512_add_pd(data1, data2));
            }

            for (; i < end; ++i)
            {
                src[i] = tgt1[i] + tgt2[2];
            }

#elif SIMD_LEVEL == 3 || SIMD_LEVEL == 2
            size_t i = start;
            size_t limit = end - ((end - start) & ~3ULL);
            for (; i < limit; i += 4)
            {
                __m256d data1(_mm256_load_pd(tgt1 + i));
                __m256d data2(_mm256_load_pd(tgt2 + i));
                _mm256_store_pd(src + i, _mm256_add_pd(data1, data2));
            }

            for (; i < end; ++i)
            {
                src[i] = tgt1[i] + tgt2[i];
            }

#elif SIMD_LEVEL == 1
            size_t i = start;
            size_t limit = end - ((end - start) & ~1ULL);
            for (; i < limit; i += 2)
            {
                __m128d data1(_mm_load_pd(tgt1 + i));
                __m128d data2(_mm_load_pd(tgt2 + i));
                _mm_store_pd(src + i, _mm_add_pd(data1, data2));
            }

            for (; i < end; ++i)
            {
                src[i] = tgt1[i] + tgt2[i];
            }

#elif SIMD_LEVEL == 0
            for (size_t i = start; i < end; ++i)
                src[i] = tgt1[i] + tgt2[i];
#endif
        };

        std::vector<std::thread> threads;
        if (number_of_thread <= 1)
        {
            worker(result_data, tensor1_data, tensor2_data, 0, total_elements);
            break;
        }
        threads.reserve(number_of_thread);
        size_t current_start = 0;

        for (size_t i = 0; i < number_of_thread - 1; ++i)
        {
            size_t current_end = current_start + chunk_size;
            if (current_end > total_elements)
                current_end = total_elements;
            threads.emplace_back(worker, result_data, tensor1_data,
                                 tensor2_data, current_start, current_end);
            current_start = current_end;
        }

        threads.emplace_back(worker, result_data, tensor1_data, tensor2_data,
                             current_start, total_elements);
        for (auto& t : threads)
            t.join();

        break;
    }
    case DType::FP32: {
        float* tensor1_data = _tensor1.data_ptr<float>();
        float* tensor2_data = _tensor2.data_ptr<float>();
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
                _mm512_store_ps(src + i, _mm512_add_ps(data1, data2));
            }

            for (; i < end; ++i)
            {
                src[i] = tgt1[i] + tgt2[i];
            }

#elif SIMD_LEVEL == 3 || SIMD_LEVEL == 2
            size_t i = start;
            size_t limit = end - ((end - start) & ~7ULL);
            for (; i < limit; i += 8)
            {
                __m256 data1(_mm256_load_ps(tgt1 + i));
                __m256 data2(_mm256_load_ps(tgt2 + i));
                _mm256_store_ps(src + i, _mm256_add_ps(data1, data2));
            }

            for (; i < end; ++i)
            {
                src[i] = tgt1[i] + tgt2[i];
            }

#elif SIMD_LEVEL == 1
            size_t i = start;
            size_t limit = end - ((end - start) & ~3ULL);
            for (; i < limit; i += 4)
            {
                __m128 data1(_mm_load_ps(tgt1 + i));
                __m128 data2(_mm_load_ps(tgt2 + i));
                _mm_store_ps(src + i, _mm_add_ps(data1, data2));
            }

            for (; i < end; ++i)
            {
                src[i] = tgt1[i] + tgt2[i];
            }

#elif SIMD_LEVEL == 0
            for (size_t i = start; i < end; ++i)
                src[i] = tgt1[i] + tgt2[i];
#endif
        };

        std::vector<std::thread> threads;
        if (number_of_thread <= 1)
        {
            worker(result_data, tensor1_data, tensor2_data, 0, total_elements);
            break;
        }
        threads.reserve(number_of_thread);
        size_t current_start = 0;

        for (size_t i = 0; i < number_of_thread - 1; ++i)
        {
            size_t current_end = current_start + chunk_size;
            if (current_end > total_elements)
                current_end = total_elements;
            threads.emplace_back(worker, result_data, tensor1_data,
                                 tensor2_data, current_start, current_end);
            current_start = current_end;
        }

        threads.emplace_back(worker, result_data, tensor1_data, tensor2_data,
                             current_start, total_elements);
        for (auto& t : threads)
            t.join();

        break;
    }
        // TODO: 텐서 원소별합 FP4 ~ FP16 구현 필요
    default:
        LOG_ERROR("{} is a data type that is not yet supported.",
                  dtype_to_string(result.dtype()));
    }
}

} // namespace ushionn