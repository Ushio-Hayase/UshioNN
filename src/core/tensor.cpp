#include "core/tensor.h"

#include "core/simd.h"
#include "utils/common.h"
#include "utils/constant.h"
#include "utils/log_macro.h"

#include <emmintrin.h>

#include <cmath>
#include <thread>

namespace nunet
{

Tensor::Tensor(::std::vector<uint64_t> shape, DType type)
    : shape_(shape), type_(type), location_(DataLocation::HOST), total_bytes_(1)
{
    for (const auto& size : shape_)
        total_bytes_ *= size;

    switch (type_)
    {
    case DType::FP64:
        total_bytes_ *= 8;
        break;
    case DType::FP32:
        total_bytes_ *= 4;
        break;

    case DType::FP16:
        total_bytes_ *= 2;
        break;
    case DType::BF16:
        total_bytes_ *= 2;
        break;
    case DType::FP8_e4m3:
        total_bytes_ *= 1;
        break;
    case DType::FP8_e5m2:
        total_bytes_ *= 1;
        break;
    case DType::FP4:
        total_bytes_ = std::ceil(total_bytes_ / 2.0f);
        break;
    }
    shape_size_ = shape_.size();

#if SIMD_LEVEL == 4
    cpu_data_ptr_.reset(utils::alignedMalloc(total_bytes_, 8));
#elif SIMD_LEVEL == 3 || SIMD_LEVEL == 2
    cpu_data_ptr_.reset(utils::alignedMalloc(total_bytes_, 4));
#elif SIMD_LEVEL == 1
    cpu_data_ptr_.reset(utils::alignedMalloc(total_bytes_, 2));
#elif SIMD_LEVEL == 0
    cpu_data_ptr_.reset(std::malloc(total_bytes_));
#endif

    strides_ = calculateStrides();
}

template <ScalarType T>
Tensor::Tensor(const std::vector<uint64_t>& shape, const T* ptr)
    : shape_(shape), location_(DataLocation::HOST), total_bytes_(1)
{
    for (const auto& size : shape_)
        total_bytes_ *= size;

    if constexpr (std::is_same_v<T, double>)
    {
        type_ = DType::FP64;
        total_bytes_ *= 8;
    }
    else if constexpr (std::is_same_v<T, float>)
    {
        type_ = DType::FP32;
        total_bytes_ *= 4;
    }
    else if constexpr (std::is_same_v<T, fp16_t>)
    {
        type_ = DType::FP16;
        total_bytes_ *= 2;
    }
    else if constexpr (std::is_same_v<T, bf16_t>)
    {
        type_ = DType::BF16;
        total_bytes_ *= 2;
    }
    else if constexpr (std::is_same_v<T, fp8_e5m2_t>)
    {
        type_ = DType::FP8_e5m2;
    }
    else if constexpr (std::is_same_v<T, fp8_e4m3_t>)
    {
        type_ = DType::FP8_e4m3;
    }
    else if constexpr (std::is_same_v<T, fp4_t>)
    {
        type_ = DType::FP4;
        total_bytes_ = std::ceil(total_bytes_ / 2.0f);
    }

    shape_size_ = shape_.size();

    strides_ = calculateStrides();

#if SIMD_LEVEL == 4
    cpu_data_ptr_.reset(utils::alignedMalloc(total_bytes_, 8));
#elif SIMD_LEVEL == 3 || SIMD_LEVEL == 2
    cpu_data_ptr_.reset(utils::alignedMalloc(total_bytes_, 4));
#elif SIMD_LEVEL == 1
    cpu_data_ptr_.reset(utils::alignedMalloc(total_bytes_, 2));
#elif SIMD_LEVEL == 0
    cpu_data_ptr_.reset(std::malloc(total_bytes_));
#endif

    std::copy(ptr, ptr + (total_bytes_ / sizeof(T)),
              static_cast<T*>(cpu_data_ptr_.get()));
}

Tensor::Tensor(const std::vector<uint64_t>& shape, void* gpu_ptr, DType type)
    : shape_(shape), type_(type), location_(DataLocation::DEVICE),
      total_bytes_(1)
{
    for (const auto& size : shape_)
        total_bytes_ *= size;

    switch (type_)
    {
    case DType::FP64:
        total_bytes_ *= 8;
        break;
    case DType::FP32:
        total_bytes_ *= 4;
        break;

    case DType::FP16:
        total_bytes_ *= 2;
        break;
    case DType::BF16:
        total_bytes_ *= 2;
        break;
    case DType::FP8_e4m3:
        total_bytes_ *= 1;
        break;
    case DType::FP8_e5m2:
        total_bytes_ *= 1;
        break;
    case DType::FP4:
        total_bytes_ = std::ceil(total_bytes_ / 2.0f);
        break;
    }

    shape_size_ = shape_.size();

    gpu_data_ptr_.reset(gpu_ptr);
}

// 명시적 인스턴스화
template Tensor::Tensor(const std::vector<uint64_t>& shape, const double* ptr);
template Tensor::Tensor(const std::vector<uint64_t>& shape, const float* ptr);
template Tensor::Tensor(const std::vector<uint64_t>& shape, const fp16_t* ptr);
template Tensor::Tensor(const std::vector<uint64_t>& shape, const bf16_t* ptr);
template Tensor::Tensor(const std::vector<uint64_t>& shape,
                        const fp8_e5m2_t* ptr);
template Tensor::Tensor(const std::vector<uint64_t>& shape,
                        const fp8_e4m3_t* ptr);
template Tensor::Tensor(const std::vector<uint64_t>& shape, const fp4_t* ptr);

Tensor& Tensor::operator+=(const Tensor& other)
{
    ASSERT_MESSAGE(this->location_ != DataLocation::NONE,
                   "Tensor not assigned.");
    ASSERT_MESSAGE(this->type_ != other.type_,
                   "The two tensors are of different types.");
    ASSERT_MESSAGE(this->shape_ != other.shape_,
                   "Two tensors have different sizes.");
    ASSERT_MESSAGE(this->location_ != other.location_,
                   "Both tensors must be in the same position.");

    if (this->location_ == DataLocation::HOST)
    {
        uint32_t number_of_thread = std::thread::hardware_concurrency();

        if (number_of_thread == 0)
            number_of_thread = 1;

        switch (type_)
        {
        case DType::FP64: {
            double* this_cpu_ptr = static_cast<double*>(cpu_data_ptr_.get());
            double* other_cpu_ptr =
                static_cast<double*>(other.cpu_data_ptr_.get());
            size_t total_elements = total_bytes_ / 8;

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

            auto worker = [](double* src, double* tgt, size_t start,
                             size_t end) {
#if SIMD_LEVEL == 4
                size_t i = start;
                size_t limit = end - ((end - start) & ~7ULL);
                for (; i < limit; i += 8)
                {
                    __m512d data_origin(_mm512_load_pd(src + i));
                    __m512d data_other(_mm512_load_pd(tgt + i));
                    _mm512_store_pd(src + i,
                                    _mm512_add_pd(data_origin, data_other));
                }

                for (; i < end; ++i)
                {
                    src[i] += tgt[i];
                }

#elif SIMD_LEVEL == 3 || SIMD_LEVEL == 2
                size_t i = start;
                size_t limit = end - ((end - start) & ~3ULL);
                for (; i < limit; i += 4)
                {
                    __m256d data_origin(_mm256_load_pd(src + i));
                    __m256d data_other(_mm256_load_pd(tgt + i));
                    _mm256_store_pd(src + i,
                                    _mm256_add_pd(data_origin, data_other));
                }

                for (; i < end; ++i)
                {
                    src[i] += tgt[i];
                }

#elif SIMD_LEVEL == 1
                size_t i = start;
                size_t limit = end - ((end - start) & ~1ULL);
                for (; i < limit; i += 2)
                {
                    __m128d data_origin(_mm_load_pd(src + i));
                    __m128d data_other(_mm_load_pd(tgt + i));
                    _mm_store_pd(src + i, _mm_add_pd(data_origin, data_other));
                }

                for (; i < end; ++i)
                {
                    src[i] += tgt[i];
                }

#elif SIMD_LEVEL == 0
                for (size_t i = start; i < end; ++i)
                    src[i] += tgt[i];
#endif
            };

            std::vector<std::thread> threads;
            if (number_of_thread <= 1)
            {
                worker(this_cpu_ptr, other_cpu_ptr, 0, total_elements);
                break;
            }
            threads.reserve(number_of_thread);
            size_t current_start = 0;

            for (size_t i = 0; i < number_of_thread - 1; ++i)
            {
                size_t current_end = current_start + chunk_size;
                if (current_end > total_elements)
                    current_end = total_elements;
                threads.emplace_back(worker, this_cpu_ptr, other_cpu_ptr,
                                     current_start, current_end);
                current_start = current_end;
            }

            threads.emplace_back(worker, this_cpu_ptr, other_cpu_ptr,
                                 current_start, total_elements);
            for (auto& t : threads)
                t.join();

            break;
        }
        case DType::FP32: {
            float* this_cpu_ptr = static_cast<float*>(cpu_data_ptr_.get());
            float* other_cpu_ptr =
                static_cast<float*>(other.cpu_data_ptr_.get());
            size_t total_elements = total_bytes_ / sizeof(float);

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

            auto worker = [](float* src, float* tgt, size_t start, size_t end) {
#if SIMD_LEVEL == 4
                size_t i = start;
                size_t limit = end - ((end - start) & ~15ULL);
                for (; i < limit; i += 16)
                {
                    __m512 data_origin(_mm512_load_ps(src + i));
                    __m512 data_other(_mm512_load_ps(tgt + i));
                    _mm512_store_ps(src + i,
                                    _mm512_add_ps(data_origin, data_other));
                }

                for (; i < end; ++i)
                {
                    src[i] += tgt[i];
                }

#elif SIMD_LEVEL == 3 || SIMD_LEVEL == 2
                size_t i = start;
                size_t limit = end - ((end - start) & ~7ULL);
                for (; i < limit; i += 8)
                {
                    __m256 data_origin(_mm256_load_ps(src + i));
                    __m256 data_other(_mm256_load_ps(tgt + i));
                    _mm256_store_ps(src + i,
                                    _mm256_add_ps(data_origin, data_other));
                }

                for (; i < end; ++i)
                {
                    src[i] += tgt[i];
                }

#elif SIMD_LEVEL == 1
                size_t i = start;
                size_t limit = end - ((end - start) & ~3ULL);
                for (; i < limit; i += 4)
                {
                    __m128 data_origin(_mm_load_ps(src + i));
                    __m128 data_other(_mm_load_ps(tgt + i));
                    _mm_store_ps(src + i, _mm_add_ps(data_origin, data_other));
                }

                for (; i < end; ++i)
                {
                    src[i] += tgt[i];
                }

#elif SIMD_LEVEL == 0
                for (size_t i = start; i < end; ++i)
                    src[i] += tgt[i];
#endif
            };

            std::vector<std::thread> threads;
            if (number_of_thread <= 1)
            {
                worker(this_cpu_ptr, other_cpu_ptr, 0, total_elements);
                break;
            }
            threads.reserve(number_of_thread);
            size_t current_start = 0;

            for (size_t i = 0; i < number_of_thread - 1; ++i)
            {
                size_t current_end = current_start + chunk_size;
                if (current_end > total_elements)
                    current_end = total_elements;
                threads.emplace_back(worker, this_cpu_ptr, other_cpu_ptr,
                                     current_start, current_end);
                current_start = current_end;
            }

            threads.emplace_back(worker, this_cpu_ptr, other_cpu_ptr,
                                 current_start, total_elements);
            for (auto& t : threads)
                t.join();

            break;
        }
        case DType::FP16: {
            fp16_t* this_cpu_ptr = static_cast<fp16_t*>(cpu_data_ptr_.get());
            fp16_t* other_cpu_ptr =
                static_cast<fp16_t*>(other.cpu_data_ptr_.get());
            for (int i = 0; i < total_bytes_ / elementSize(); ++i)
                *(this_cpu_ptr + i) += *(other_cpu_ptr + i);
            break;
        }
        case DType::BF16: {
            bf16_t* this_cpu_ptr = static_cast<bf16_t*>(cpu_data_ptr_.get());
            bf16_t* other_cpu_ptr =
                static_cast<bf16_t*>(other.cpu_data_ptr_.get());

            break;
        }
        case DType::FP8_e5m2: {
            break;
        }
        case DType::FP8_e4m3: {
            break;
        }
        case DType::FP4: {
            break;
        }
        }
    }
    else if (location_ == DataLocation::DEVICE)
    {
        addAssignGpu(other, type_);
    }

    return *this;
}

Tensor& Tensor::operator*=(const Tensor& other)
{
    ASSERT_MESSAGE(this->location_ != DataLocation::NONE,
                   "Tensor not assigned.");
    ASSERT_MESSAGE(this->type_ != other.type_,
                   "The two tensors are of different types.");
    ASSERT_MESSAGE(this->shape_ != other.shape_,
                   "Two tensors have different sizes.");
    ASSERT_MESSAGE(this->location_ != other.location_,
                   "Both tensors must be in the same position.");

    if (this->location_ == DataLocation::HOST)
    {
        uint32_t number_of_thread = std::thread::hardware_concurrency();

        if (number_of_thread == 0)
            number_of_thread = 1;

        switch (type_)
        {
        case DType::FP64: {
            double* this_cpu_ptr = static_cast<double*>(cpu_data_ptr_.get());
            double* other_cpu_ptr =
                static_cast<double*>(other.cpu_data_ptr_.get());
            size_t total_elements = total_bytes_ / 8;

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

            auto worker = [](double* src, double* tgt, size_t start,
                             size_t end) {
#if SIMD_LEVEL == 4
                size_t i = start;
                size_t limit = end - ((end - start) & ~7ULL);
                for (; i < limit; i += 8)
                {
                    __m512d data_origin(_mm512_load_pd(src + i));
                    __m512d data_other(_mm512_load_pd(tgt + i));
                    _mm512_store_pd(src + i,
                                    _mm512_mul_pd(data_origin, data_other));
                }

                for (; i < end; ++i)
                {
                    src[i] *= tgt[i];
                }

#elif SIMD_LEVEL == 3 || SIMD_LEVEL == 2
                size_t i = start;
                size_t limit = end - ((end - start) & ~3ULL);
                for (; i < limit; i += 4)
                {
                    __m256d data_origin(_mm256_load_pd(src + i));
                    __m256d data_other(_mm256_load_pd(tgt + i));
                    _mm256_store_pd(src + i,
                                    _mm256_mul_pd(data_origin, data_other));
                }

                for (; i < end; ++i)
                {
                    src[i] *= tgt[i];
                }

#elif SIMD_LEVEL == 1
                size_t i = start;
                size_t limit = end - ((end - start) & ~1ULL);
                for (; i < limit; i += 2)
                {
                    __m128d data_origin(_mm_load_pd(src + i));
                    __m128d data_other(_mm_load_pd(tgt + i));
                    _mm_store_pd(src + i, _mm_mul_pd(data_origin, data_other));
                }

                for (; i < end; ++i)
                {
                    src[i] *= tgt[i];
                }

#elif SIMD_LEVEL == 0
                for (size_t i = start; i < end; ++i)
                    src[i] *= tgt[i];
#endif
            };

            std::vector<std::thread> threads;
            if (number_of_thread <= 1)
            {
                worker(this_cpu_ptr, other_cpu_ptr, 0, total_elements);
                break;
            }
            threads.reserve(number_of_thread);
            size_t current_start = 0;

            for (size_t i = 0; i < number_of_thread - 1; ++i)
            {
                size_t current_end = current_start + chunk_size;
                if (current_end > total_elements)
                    current_end = total_elements;
                threads.emplace_back(worker, this_cpu_ptr, other_cpu_ptr,
                                     current_start, current_end);
                current_start = current_end;
            }

            threads.emplace_back(worker, this_cpu_ptr, other_cpu_ptr,
                                 current_start, total_elements);
            for (auto& t : threads)
                t.join();

            break;
        }
        case DType::FP32: {
            float* this_cpu_ptr = static_cast<float*>(cpu_data_ptr_.get());
            float* other_cpu_ptr =
                static_cast<float*>(other.cpu_data_ptr_.get());
            size_t total_elements = total_bytes_ / 8;

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

            auto worker = [](float* src, float* tgt, size_t start, size_t end) {
#if SIMD_LEVEL == 4
                size_t i = start;
                size_t limit = end - ((end - start) & ~15ULL);
                for (; i < limit; i += 16)
                {
                    __m512 data_origin(_mm512_load_ps(src + i));
                    __m512 data_other(_mm512_load_ps(tgt + i));
                    _mm512_store_ps(src + i,
                                    _mm512_mul_ps(data_origin, data_other));
                }

                for (; i < end; ++i)
                {
                    src[i] *= tgt[i];
                }

#elif SIMD_LEVEL == 3 || SIMD_LEVEL == 2
                size_t i = start;
                size_t limit = end - ((end - start) & ~7ULL);
                for (; i < limit; i += 8)
                {
                    __m256 data_origin(_mm256_load_ps(src + i));
                    __m256 data_other(_mm256_load_ps(tgt + i));
                    _mm256_store_ps(src + i,
                                    _mm256_mul_ps(data_origin, data_other));
                }

                for (; i < end; ++i)
                {
                    src[i] *= tgt[i];
                }

#elif SIMD_LEVEL == 1
                size_t i = start;
                size_t limit = end - ((end - start) & ~3ULL);
                for (; i < limit; i += 4)
                {
                    __m128 data_origin(_mm_load_ps(src + i));
                    __m128 data_other(_mm_load_ps(tgt + i));
                    _mm_store_ps(src + i, _mm_mul_ps(data_origin, data_other));
                }

                for (; i < end; ++i)
                {
                    src[i] *= tgt[i];
                }

#elif SIMD_LEVEL == 0
                for (size_t i = start; i < end; ++i)
                    src[i] *= tgt[i];
#endif
            };

            std::vector<std::thread> threads;
            if (number_of_thread <= 1)
            {
                worker(this_cpu_ptr, other_cpu_ptr, 0, total_elements);
                break;
            }
            threads.reserve(number_of_thread);
            size_t current_start = 0;

            for (size_t i = 0; i < number_of_thread - 1; ++i)
            {
                size_t current_end = current_start + chunk_size;
                if (current_end > total_elements)
                    current_end = total_elements;
                threads.emplace_back(worker, this_cpu_ptr, other_cpu_ptr,
                                     current_start, current_end);
                current_start = current_end;
            }

            threads.emplace_back(worker, this_cpu_ptr, other_cpu_ptr,
                                 current_start, total_elements);
            for (auto& t : threads)
                t.join();

            break;
        }
        case DType::FP16: {
            fp16_t* this_cpu_ptr = static_cast<fp16_t*>(cpu_data_ptr_.get());
            fp16_t* other_cpu_ptr =
                static_cast<fp16_t*>(other.cpu_data_ptr_.get());
            for (int i = 0; i < total_bytes_ / elementSize(); ++i)
                *(this_cpu_ptr + i) += *(other_cpu_ptr + i);
            break;
        }
        case DType::BF16: {
            bf16_t* this_cpu_ptr = static_cast<bf16_t*>(cpu_data_ptr_.get());
            bf16_t* other_cpu_ptr =
                static_cast<bf16_t*>(other.cpu_data_ptr_.get());

            break;
        }
        case DType::FP8_e5m2: {
            break;
        }
        case DType::FP8_e4m3: {
            break;
        }
        case DType::FP4: {
            break;
        }
        }
    }
    else if (location_ == DataLocation::DEVICE)
    {
        addAssignGpu(other, type_);
    }

    return *this;
}

Tensor& Tensor::operator*=(const float scalar)
{
    ASSERT_MESSAGE(this->location_ != DataLocation::NONE,
                   "Tensor not assigned.");

    if (this->location_ == DataLocation::HOST)
    {
        uint32_t number_of_thread = std::thread::hardware_concurrency();

        if (number_of_thread == 0)
            number_of_thread = 1;

        switch (type_)
        {
        case DType::FP64: {
            double* this_cpu_ptr = static_cast<double*>(cpu_data_ptr_.get());
            size_t total_elements = total_bytes_ / 8;

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

            auto worker = [](double* src, const float& scalar, size_t start,
                             size_t end) {
#if SIMD_LEVEL == 4
                size_t i = start;
                size_t limit = end - ((end - start) & ~7ULL);
                for (; i < limit; i += 8)
                {
                    const __m512d data_origin(_mm512_load_pd(src + i));
                    const __m512d data_other(_mm512_set1_pd(scalar));
                    _mm512_store_pd(src + i,
                                    _mm512_mul_pd(data_origin, data_other));
                }

                for (; i < end; ++i)
                {
                    src[i] *= scalar;
                }

#elif SIMD_LEVEL == 3 || SIMD_LEVEL == 2
                size_t i = start;
                size_t limit = end - ((end - start) & ~3ULL);
                for (; i < limit; i += 4)
                {
                    const __m256d data_origin(_mm256_load_pd(src + i));
                    const __m256d data_other(_mm256_set1_pd(scalar));
                    _mm256_store_pd(src + i,
                                    _mm256_mul_pd(data_origin, data_other));
                }

                for (; i < end; ++i)
                {
                    src[i] *= scalar;
                }

#elif SIMD_LEVEL == 1
                size_t i = start;
                size_t limit = end - ((end - start) & ~1ULL);
                for (; i < limit; i += 2)
                {
                    const __m128d data_origin(_mm_load_pd(src + i));
                    const __m128d data_other(_mm_set1_pd(scalar));
                    _mm_store_pd(src + i, _mm_mul_pd(data_origin, data_other));
                }

                for (; i < end; ++i)
                {
                    src[i] *= scalar;
                }

#elif SIMD_LEVEL == 0
                for (size_t i = start; i < end; ++i)
                    src[i] *= scalar;
#endif
            };

            std::vector<std::thread> threads;
            if (number_of_thread <= 1)
            {
                worker(this_cpu_ptr, scalar, 0, total_elements);
                break;
            }
            threads.reserve(number_of_thread);
            size_t current_start = 0;

            for (size_t i = 0; i < number_of_thread - 1; ++i)
            {
                size_t current_end = current_start + chunk_size;
                if (current_end > total_elements)
                    current_end = total_elements;
                threads.emplace_back(worker, this_cpu_ptr, scalar,
                                     current_start, current_end);
                current_start = current_end;
            }

            threads.emplace_back(worker, this_cpu_ptr, scalar, current_start,
                                 total_elements);
            for (auto& t : threads)
                t.join();

            break;
        }
        case DType::FP32: {
            float* this_cpu_ptr = static_cast<float*>(cpu_data_ptr_.get());
            size_t total_elements = total_bytes_ / 4;

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

            auto worker = [](float* src, const float& scalar, size_t start,
                             size_t end) {
#if SIMD_LEVEL == 4
                size_t i = start;
                size_t limit = end - ((end - start) & ~15ULL);
                for (; i < limit; i += 16)
                {
                    const __m512 data_origin(_mm512_load_ps(src + i));
                    const __m512 data_other(_mm512_set1_ps(scalar));
                    _mm512_store_ps(src + i,
                                    _mm512_mul_ps(data_origin, data_other));
                }

                for (; i < end; ++i)
                {
                    src[i] *= scalar;
                }

#elif SIMD_LEVEL == 3 || SIMD_LEVEL == 2
                size_t i = start;
                size_t limit = end - ((end - start) & ~7ULL);
                for (; i < limit; i += 8)
                {
                    const __m256 data_origin(_mm256_load_ps(src + i));
                    const __m256 data_other(_mm256_set1_ps(scalar));
                    _mm256_store_ps(src + i,
                                    _mm256_mul_ps(data_origin, data_other));
                }

                for (; i < end; ++i)
                {
                    src[i] *= scalar;
                }

#elif SIMD_LEVEL == 1
                size_t i = start;
                size_t limit = end - ((end - start) & ~3ULL);
                for (; i < limit; i += 4)
                {
                    const __m128 data_other(_mm_set1_ps(scalar));
                    const __m128 data_origin(_mm_load_ps(src + i));
                    _mm_store_ps(src + i, _mm_mul_ps(data_origin, data_other));
                }

                for (; i < end; ++i)
                {
                    src[i] *= scalar;
                }

#elif SIMD_LEVEL == 0
                for (size_t i = start; i < end; ++i)
                    src[i] *= scalar;

#endif
            };

            std::vector<std::thread> threads;
            if (number_of_thread <= 1)
            {
                worker(this_cpu_ptr, scalar, 0, total_elements);
                break;
            }
            threads.reserve(number_of_thread);
            size_t current_start = 0;

            for (size_t i = 0; i < number_of_thread - 1; ++i)
            {
                size_t current_end = current_start + chunk_size;
                if (current_end > total_elements)
                    current_end = total_elements;
                threads.emplace_back(worker, this_cpu_ptr, scalar,
                                     current_start, current_end);
                current_start = current_end;
            }

            threads.emplace_back(worker, this_cpu_ptr, scalar, current_start,
                                 total_elements);
            for (auto& t : threads)
                t.join();
            break;
        }
        case DType::FP16: {
            fp16_t* this_cpu_ptr = static_cast<fp16_t*>(cpu_data_ptr_.get());
            size_t total_elements = total_bytes_ / 2;

            size_t align_step = 1;
#if SIMD_LEVEL == 4
            align_step = 32;
#elif SIMD_LEVEL == 3 || SIMD_LEVEL == 2
            align_step = 16;
#elif SIMD_LEVEL == 1
            align_step = 8;
#endif

            if (total_elements < MULTI_THREAD_BASELINE)
                number_of_thread = 1;

            size_t chunk_size = total_elements / number_of_thread;
            chunk_size &= ~(align_step - 1);

            auto worker = [](fp16_t* src, const float& scalar, size_t start,
                             size_t end) {
#if SIMD_LEVEL == 4
                size_t i = start;
                size_t limit = end - ((end - start) & ~31ULL);
                for (; i < limit; i += 32)
                {
                    const __m512h data_origin(_mm512_load_ph(src + i));
#if defined(_MSC_VER)

                    const __m128 f_vec = _mm_set_ss(scalar);
                    const __m128i h_vec =
                        _mm_cvtps_ph(f_vec, _MM_FROUND_TO_NEAREST_INT);
                    const uint16_t h_val = static_cast<unsigned short>(
                        _mm_extract_epi16(h_vec, 0));
                    __m512i vec_i = _mm512_set1_epi16(h_val);
                    const __m512h data_other(_mm512_castsi512_ph(vec_i));
                    _mm512_store_ph(src + i,
                                    _mm512_mul_ph(data_origin, data_other));
                }
#else
                    const _Float16 scalar_copy = static_cast<_Float16>(scalar);
                    const __m512h data_other(_mm512_set1_ph(scalar_copy));
                    _mm512_store_ph(src,
                                    _mm512_mul_ph(data_origin, data_other));
                }

#endif
                for (; i < end; ++i)
                {
                    src[i] *= scalar;
                }

#elif SIMD_LEVEL == 3 || SIMD_LEVEL == 2
                size_t i = start;
                size_t limit = end - ((end - start) & ~15ULL);
                for (; i < limit; i += 16)
                {
                    const _Float16 scalar_copy = static_cast<_Float16>(scalar);
                    const __m256 data_origin(_mm256_load_ph(src + i));
                    const __m256 data_other(_mm256_set1_ph(scalar_copy));
                    _mm256_store_ph(src + i,
                                    _mm256_mul_ph(data_origin, data_other));
                }

                for (; i < end; ++i)
                {
                    src[i] *= scalar;
                }

#elif SIMD_LEVEL == 1
                size_t i = start;
                size_t limit = end - ((end - start) & ~7ULL);
                for (; i < limit; i += 8)
                {
                    const _Float16 scalar_copy = static_cast<_Float16>(scalar);
                    const __m128 data_other(_mm_set1_ph(scalar_copy));
                    const __m128 data_origin(_mm_load_ps(src + i));
                    _mm_store_ph(src + i, _mm_mul_ph(data_origin, data_other));
                }

                for (; i < end; ++i)
                {
                    src[i] *= scalar;
                }

#elif SIMD_LEVEL == 0
                for (size_t i = start; i < end; ++i)
                    src[i] *= scalar;
#endif
            };
            std::vector<std::thread> threads;
            if (number_of_thread <= 1)
            {
                worker(this_cpu_ptr, scalar, 0, total_elements);
                break;
            }
            threads.reserve(number_of_thread);
            size_t current_start = 0;

            for (size_t i = 0; i < number_of_thread - 1; ++i)
            {
                size_t current_end = current_start + chunk_size;
                if (current_end > total_elements)
                    current_end = total_elements;
                threads.emplace_back(worker, this_cpu_ptr, scalar,
                                     current_start, current_end);
                current_start = current_end;
            }

            threads.emplace_back(worker, this_cpu_ptr, scalar, current_start,
                                 total_elements);
            for (auto& t : threads)
                t.join();
            break;
        }

        case DType::BF16: {
            bf16_t* this_cpu_ptr = static_cast<bf16_t*>(cpu_data_ptr_.get());
            // TODO: BF16형 텐서 *= 연산자 구현 필요
            break;
        }
        case DType::FP8_e5m2: {
            // TODO: FP8_e5m2형 텐서 *= 연산자 구현 필요
            break;
        }
        case DType::FP8_e4m3: {
            // TODO: FP8_e4m3형 텐서 *= 연산자 구현 필요
            break;
        }
        case DType::FP4: {
            // TODO: FP4형 텐서 *= 연산자 구현 필요
            break;
        }
        }
    }
    else if (location_ == DataLocation::DEVICE)
    {
        mulAssignGpu(scalar);
    }

    return *this;
}

std::vector<size_t> Tensor::getShape() const { return shape_; }
DataLocation Tensor::getDevice() const { return location_; }
DType Tensor::getType() const { return type_; }
size_t Tensor::getTotalBytes() const { return total_bytes_; }
size_t Tensor::getShapeSize() const { return shape_size_; }

const void* Tensor::getCpuPtr() const { return cpu_data_ptr_.get(); }
const void* Tensor::getGpuPtr() const { return gpu_data_ptr_.get(); }
void* Tensor::getCpuPtrMutable() { return cpu_data_ptr_.get(); }
void* Tensor::getGpuPtrMutable() { return gpu_data_ptr_.get(); }

uint64_t Tensor::elementSize()
{
    switch (type_)
    {
    case DType::FP64:
        return sizeof(double);
    case DType::FP32:
        return sizeof(float);
    case DType::FP16:
        return 2;
    case DType::BF16:
        return 2;
    case DType::FP8_e4m3:
        return 1;
    case DType::FP8_e5m2:
        return 1;
    case DType::FP4:
        return 1; // 4 bits but 1 byte of storage
    default:
        return 0;
    }
}

std::vector<size_t> Tensor::calculateStrides() const
{
    std::vector<size_t> strides(shape_size_);

    size_t stride_cnt = 1;

    for (int i = shape_size_ - 1; i >= 0; --i)
    {
        strides[i] = stride_cnt;
        stride_cnt *= shape_[i];
    }

    return strides;
}

} // namespace nunet