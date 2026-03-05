#include <immintrin.h>

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512DQ__) &&  \
    defined(__AVX512VL__)
#define SIMD_LEVEL 4
#elif
#endif