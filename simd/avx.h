#ifndef HF_SIMD_SIMD_H
#define HF_SIMD_SIMD_H

#include <immintrin.h>
#include "simd/simd.h"

namespace hf {
namespace simd_detail {

#ifdef __AVX__
namespace avx {
struct double4 {
    using scalar_type = double;
    using vector_type = __m256d;

    constexpr static std::size_t width() { return 4; }

    static vector_type broadcast(double v) {
        return _mm256_set1_pd(v);
    }

    static vector_type immediate(double v0, double v1, double v2, double v3) {
        return _mm256_setr_pd(v0, v1, v2, v3);
    }

    static void copy_to(vector_type v, scalar_type* p) {
        _mm256_storeu_pd(p, v);
    }

    static vector_type copy_from(const scalar_type* p) {
        return _mm256_loadu_pd(p);
    }

    static vector_type add(vector_type a, vector_type b) {
        return _mm256_add_pd(a, b);
    }

    static vector_type sub(vector_type a, vector_type b) {
        return _mm256_sub_pd(a, b);
    }

    static vector_type mul(vector_type a, vector_type b) {
        return _mm256_mul_pd(a, b);
    }

    static vector_type div(vector_type a, vector_type b) {
        return _mm256_div_pd(a, b);
    }

    static vector_type fma(vector_type a, vector_type b, vector_type c) {
        return a*b+c;
    }
};
} // namespace avx

#ifdef __AVX2__
namespace avx2 {
struct double4: avx::double4 {
    static vector_type fma(vector_type a, vector_type b, vector_type c) {
        return _mm256_fmadd_pd(a, b, c);
    }
};
} // namespace avx2

template <>
struct native<double, 4> {
    using type = avx2::double4;
};

#else

template <>
struct native<double, 4> {
    using type = avx::double4;
};

#endif // def __AVX2__
#endif // def __AVX__

} // namespace simd_detail
} // namespace hf

#endif // ndef HF_SIMD_SIMD_H
