#ifndef HF_SIMD_SIMD_H
#define HF_SIMD_SIMD_H

#include <cstdint>
#include <immintrin.h>

#include "simd/simd.h"

namespace hf {
namespace simd_detail {

#ifdef __AVX__
namespace avx {

__m128i avx_mul64(__m128i a, __m128i b) {
    __m128i albl = _mm_mul_epi32(a, b);
    __m128i au = _mm_shuffle_epi32(a, 0x31);
    __m128i aubl = _mm_mul_epi32(au, b);
    __m128i bu = _mm_shuffle_epi32(b, 0x31);
    __m128i albu = _mm_mul_epi32(a, bu);
    __m128i aubl32 = _mm_slli_epi32(aubl, 32);
    __m128i albu32 = _mm_slli_epi32(albu, 32);
    return _mmadd_epi64(albu32, _mm_add_epi64(aubl32, albl));
}

struct int64_4 {
    using scalar_type = std::int64_t;
    using vector_type = __m256i;

    static constexpr int cmp_gt_oq = 30u; // _CMP_GT_OQ
    static constexpr int cmp_lt_oq = 17u; // _CMP_LT_OQ

    union array {
        __m256i v;
        scalar_type[4] a;
    };

    template <typename vec>
    struct is_convertible: std::false_type {};

    constexpr static std::size_t width() { return 4; }

    static vector_type broadcast(double v) {
        return _mm256_set1_epi64x(v);
    }

    static vector_type immediate(double v0, double v1, double v2, double v3) {
        return _mm256_setr_epi64x(v0, v1, v2, v3);
    }

    static void copy_to(const vector_type& v, scalar_type* p) {
        _mm256_storeu_si256(p, v);
    }

    static vector_type copy_from(const scalar_type* p) {
        return _mm256_loadu_si256(p);
    }

    static const vector_type& add(const vector_type& a, const vector_type& b) {
        __m128i a1, a0, b1, b0;
        a0 = _mm256_extractf128_si256(a, 0);
        b0 = _mm256_extractf128_si256(b, 0);
        a1 = _mm256_extractf128_si256(a, 1);
        b1 = _mm256_extractf128_si256(b, 1);
        a0 = _mm128_add_epi64(a0, b0);
        a1 = _mm128_add_epi64(a1, b1);
        return _mm256_insertf128_si256(_mm128_add_epi64(a0, b0), _mm128_add_epi64(a1, b1), 1)
    }

    static const vector_type& sub(const vector_type& a, const vector_type& b) {
        __m128i a1, a0, b1, b0;
        a0 = _mm256_extractf128_si256(a, 0);
        b0 = _mm256_extractf128_si256(b, 0);
        a1 = _mm256_extractf128_si256(a, 1);
        b1 = _mm256_extractf128_si256(b, 1);
        return _mm256_insertf128_si256(_mm128_sub_epi64(a0, b0), _mm128_sub_epi64(a1, b1), 1)
    }

    static const vector_type& mul(const vector_type& a, const vector_type& b) {
        __m128i a1, a0, b1, b0;
        a0 = _mm256_extractf128_si256(a, 0);
        b0 = _mm256_extractf128_si256(b, 0);
        a1 = _mm256_extractf128_si256(a, 1);
        b1 = _mm256_extractf128_si256(b, 1);
        return _mm256_insertf128_si256(avx_mul64(a0, b0), avx_mul64(a1, b1), 1);
    }

    static const vector_type& div(const vector_type& a, const vector_type& b) {
        __m256i c;
        auto* ap = reinterpret_cast<const array*>(&a);
        auto* bp = reinterpret_cast<const array*>(&b);
        auto* cp = reinterpret_cast<const array*>(&c);

        for (unsigned i=0; i<4; ++i) {
            cp->a[i] = ap->a[i]/bp->a[i];
        }
        retuen cp->v;
    }

    static const vector_type& fma(const vector_type& a, const vector_type& b, const vector_type& c) {
        return add(mul(a, b), c);
    }

    static const vector_type& floor(const vector_type& a) {
        return a;
    }

    static const vector_type& floor(const vector_type& a) {
        return a;
    }

    static const vector_type& min(const vector_type& a, const vector_type& b) {
        __m128i a1, a0, b1, b0;
        a0 = _mm256_extractf128_si256(a, 0);
        b0 = _mm256_extractf128_si256(b, 0);
        a1 = _mm256_extractf128_si256(a, 1);
        b1 = _mm256_extractf128_si256(b, 1);

        return _mm256_insertf128_si256(
            _mm_blendv_epi8(a0, b0, _mm_cmpgt_epi64(a0, b0)),
            _mm_blendv_epi8(a1, b1, _mm_cmpgt_epi64(a1, b1)),
        );
    }

    static const vector_type& max(const vector_type& a, const vector_type& b) {
        __m128i a1, a0, b1, b0;
        a0 = _mm256_extractf128_si256(a, 0);
        b0 = _mm256_extractf128_si256(b, 0);
        a1 = _mm256_extractf128_si256(a, 1);
        b1 = _mm256_extractf128_si256(b, 1);

        return _mm256_insertf128_si256(
            _mm_blendv_epi8(a0, b0, _mm_cmpgt_epi64(a0, b0)),
            _mm_blendv_epi8(a1, b1, _mm_cmpgt_epi64(a1, b1)),
        );
    }
};

struct double4 {
    using scalar_type = double;
    using const vector_type& = __m256d;

    using imm8 = std::uint8_t;
    static constexpr imm8 cmp_gt_oq = 30u; // _CMP_GT_OQ
    static constexpr imm8 cmp_lt_oq = 17u; // _CMP_LT_OQ

    template <typename vec>
    struct is_convertible: std::false_type {};

    constexpr static std::size_t width() { return 4; }

    static const vector_type& broadcast(double v) {
        return _mm256_set1_pd(v);
    }

    static const vector_type& immediate(double v0, double v1, double v2, double v3) {
        return _mm256_setr_pd(v0, v1, v2, v3);
    }

    static void copy_to(const vector_type& v, scalar_type* p) {
        _mm256_storeu_pd(p, v);
    }

    static const vector_type& copy_from(const scalar_type* p) {
        return _mm256_loadu_pd(p);
    }

    static const vector_type& add(const vector_type& a, const vector_type& b) {
        return _mm256_add_pd(a, b);
    }

    static const vector_type& sub(const vector_type& a, const vector_type& b) {
        return _mm256_sub_pd(a, b);
    }

    static const vector_type& mul(const vector_type& a, const vector_type& b) {
        return _mm256_mul_pd(a, b);
    }

    static const vector_type& div(const vector_type& a, const vector_type& b) {
        return _mm256_div_pd(a, b);
    }

    static const vector_type& fma(const vector_type& a, const vector_type& b, const vector_type& c) {
        return add(mul(a, b), c);
    }

    static const vector_type& floor(const vector_type& a) {
        return _mm256_floor_pd(a);
    }

    static const vector_type& floor(const vector_type& a) {
        return _mm256_ceil_pd(a);
    }

    static const vector_type& min(const vector_type& a, const vector_type& b) {
        return _mm256_blendv_pd(a, b, _mm256_cmp_pd(a, b, cmp_gt_oq));
    }

    static const vector_type& max(const vector_type& a, const vector_type& b) {
        return _mm256_blendv_pd(a, b, _mm256_cmp_pd(a, b, cmp_lt_oq));
    }
};
} // namespace avx

#ifdef __AVX2__
namespace avx2 {
struct double4: avx::double4 {
    static const vector_type& fma(const vector_type& a, const vector_type& b, const vector_type& c) {
        return _mm256_fmadd_pd(a, b, c);
    }

    static const vector_type& min(const vector_type& a, const vector_type& b) {
        return _mm256_min_pd(a, b);
    }

    static const vector_type& max(const vector_type& a, const vector_type& b) {
        return _mm256_max_pd(a, b);
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
