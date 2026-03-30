#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#if defined(USE_CUDA)
#include <cuda_fp16.h>
#if defined(TARGET_CUDA_ARCH) && TARGET_CUDA_ARCH >= 80
#include <cuda_bf16.h>
#endif
#if defined(TARGET_CUDA_ARCH) && TARGET_CUDA_ARCH >= 89
#include <cuda_fp8.h>
#endif

#elif defined(USE_ROCM) || defined(USE_HIP)
#include <hip/hip_fp16.h>
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
#include <hip/hip_bfloat16.h>
#endif

#else
#if defined(__cpp_lib_stdfloat) && __cplusplus >= 202302L
#include <stdfloat>
#endif
#endif

namespace ushionn
{

struct FallbackBf16
{
    uint16_t bits;
};
struct FallbackFp8E4m3
{
    uint8_t bits;
};
struct FallbackFp8E5m2
{
    uint8_t bits;
};

#if defined(USE_CUDA)
using fp16_t = __half;
// TODO: half2로 변경하여 GPU SIMD 가능, 하지만 4바이트 메모리 정렬 필요

#if defined(TARGET_CUDA_ARCH) && TARGET_CUDA_ARCH >= 80
using bf16_t = __nv_bfloat16;
#else
using bf16_t = FallbackBf16;
#endif

#if defined(TARGET_CUDA_ARCH) && TARGET_CUDA_ARCH >= 89
using fp8_e4m3_t = __nv_fp8_e4m3;
using fp8_e5m2_t = __nv_fp8_e5m2;
#else
using fp8_e4m3_t = FallbackFp8E4m3;
using fp8_e5m2_t = FallbackFp8E5m2;
#endif

using fp4_t = uint8_t; // packs two 4-bit values into one byte (uint8_t) with a
                       // bit operation packed into a byte (int8_t)

#elif defined(USE_ROCM)
using fp16_t = __half;

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
using bf16_t = hip_bfloat16;
#else
using bf16_t = fallback_bf16;
#endif

using fp8_e4m3_t = Fallback_fp8_e4m3;
using fp8_e5m2_t = Fallback_fp8_e5m2;

using fp4_t = uint8_t; // packs two 4-bit values into one byte (uint8_t) with a
                       // bit operation packed into a byte (int8_t)

#else

#if defined(__cpp_lib_stdfloat) && __cplusplus >= 202302L
using fp16_t = std::float16_t;
using bf16_t = std::bfloat16_t;
#else
// For unsupported older compilers, replace with 16-bit integer to preserve
// memory layout
using fp16_t = uint16_t;
using bf16_t = FallbackBf16;
#endif

using fp8_e4m3_t = FallbackFp8E4m3;
using fp8_e5m2_t = FallbackFp8E5m2;
using fp4_t = uint8_t;

#endif

#if __cplusplus >= 202002L || (defined(_MSVC_LANG) && _MSVC_LANG >= 202002L)

template <typename T>
concept ScalarType = std::is_same_v<T, float> || std::is_same_v<T, double> ||
                     std::is_same_v<T, fp16_t> || std::is_same_v<T, bf16_t> ||
                     std::is_same_v<T, fp8_e4m3_t> ||
                     std::is_same_v<T, fp8_e5m2_t> || std::is_same_v<T, fp4_t>;

#else

// Type Traits for Scalar Type Validation
template <typename T> struct IsScalarType
{
    static constexpr bool value =
        std::is_same<T, float>::value || std::is_same<T, double>::value ||
        std::is_same<T, fp16_t>::value || std::is_same<T, bf16_t>::value ||
        std::is_same<T, fp8_e4m3_t>::value ||
        std::is_same<T, fp8_e5m2_t>::value || std::is_same<T, fp4_t>::value;
};
#endif

enum class DType
{
    FP4,      // fp4_t
    FP8_e5m2, // fp8_e5m2_t
    FP8_e4m3, // fp8_e4m3_t
    BF16,     // bf16_t
    FP16,     // fp16_t
    FP32,     // float
    FP64,     // double
};

inline std::string dtype_to_string(DType type)
{
    switch (type)
    {
    case DType::FP64:
        return "FP64";
    case DType::FP32:
        return "FP32";
    case DType::FP16:
        return "FP16";
    case DType::BF16:
        return "BF16";
    case DType::FP8_e4m3:
        return "FP8_e4m3";
    case DType::FP8_e5m2:
        return "FP8_e5m2";
    case DType::FP4:
        return "FP4";
    default:
        return "undefined";
    }
}

/**
 * @brief Get the Dtype object
 *
 * @tparam T Data type to enumerate
 * @return constexpr DType Changed enumeration type data type
 */
template <ScalarType T> constexpr DType get_dtype()
{
    if constexpr (std::is_same_v<T, float>)
        return DType::FP32;
    else if constexpr (std::is_same_v<T, double>)
        return DType::FP64;
    else if constexpr (std::is_same_v<T, fp16_t>)
        return DType::FP16;
    else if constexpr (std::is_same_v<T, bf16_t>)
        return DType::BF16;
    else if constexpr (std::is_same_v<T, fp8_e4m3_t>)
        return DType::FP8_e4m3;
    else if constexpr (std::is_same_v<T, fp8_e5m2_t>)
        return DType::FP8_e5m2;
    else if constexpr (std::is_same_v<T, fp4_t>)
        return DType::FP4;
    else
    {
        static_assert(sizeof(T) == 0, "Unsupported type mapping!");
    }
}

} // namespace ushionn