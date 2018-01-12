#ifndef HF_SIMD_GENERIC_H
#define HF_SIMD_GENERIC_H

#include <array>
#include <cmath>
#include <cstring>

#include "simd/simd.h"

namespace hf {
namespace simd_detail {

namespace generic {
template <typename T, unsigned N>
struct simd {
    using scalar_type = T;
    using vector_type = std::array<T, N>;

    constexpr static std::size_t width() { return N; }

    static vector_type broadcast(double v) {
        vector_type result;
        for (auto& x: result) x = v;
        return result;
    }

    template <typename... V>
    static vector_type immediate(V... vs) {
        vector_type result({static_cast<scalar_type>(vs)...});
        return result;
    }

    static void copy_to(vector_type v, scalar_type* p) {
        std::memcpy(p, v.data(), sizeof(v));
    }

    static vector_type copy_from(const scalar_type* p) {
        vector_type result;
        std::memcpy(result.data(), p, sizeof(result));
        return result;
    }

    static vector_type add(vector_type a, vector_type b) {
        vector_type result;
        for (unsigned i = 0; i<N; ++i) result[i] = a[i]+b[i];
        return result;
    }

    static vector_type sub(vector_type a, vector_type b) {
        vector_type result;
        for (unsigned i = 0; i<N; ++i) result[i] = a[i]-b[i];
        return result;
    }

    static vector_type mul(vector_type a, vector_type b) {
        vector_type result;
        for (unsigned i = 0; i<N; ++i) result[i] = a[i]*b[i];
        return result;
    }

    static vector_type div(vector_type a, vector_type b) {
        vector_type result;
        for (unsigned i = 0; i<N; ++i) result[i] = a[i]/b[i];
        return result;
    }

    static vector_type fma(vector_type a, vector_type b, vector_type c) {
        vector_type result;
        for (unsigned i = 0; i<N; ++i) result[i] = std::fma(a[i], b[i], c[i]);
        return result;
    }
};
} // namespace generic
} // namespace simd_detail

template <typename T, int N>
using simd_generic = simd_base<simd_detail::generic::simd<T, N>>;

} // namespace hf

#endif // ndef HF_SIMD_GENERIC_H
