#include <immintrin.h>

#if USE_SIMD && USE_AVX512 && defined(__AVX512F__) && defined(__AVX512BW__) && \
    defined(__AVX512DQ__) && defined(__AVX512VL__)
#define SIMD_LEVEL 4
#elif USE_SIMD && USE_AVX2 && defined(__AVX2__)
#define SIMD_LEVEL 3
#elif USE_SIMD && USE_AVX && defined(__AVX__)
#define SIMD_LEVEL 2
#elif USE_SIMD && defined(__SSE2__)
#define SIMD_LEVEL 1
#else
#define SIMD_LEVEL 0
#endif