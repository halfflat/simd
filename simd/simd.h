#ifndef HF_SIMD_AVX_H
#define HF_SIMD_AVX_H

#include <cstring>
#include <type_traits>

namespace hf {

namespace simd_detail {
    template <typename T, unsigned N>
    struct native;
}

template <typename Impl>
struct simd_base {
    using scalar_type = typename Impl::scalar_type;
    using vector_type = typename Impl::vector_type;

    vector_type value_;

    static constexpr std::size_t width() {
        return Impl::width();
    }

    simd_base(vector_type v) {
        std::memcpy(&value_, &v, sizeof(vector_type));
    }

    simd_base(scalar_type v) {
        value_ = Impl::broadcast(v);
    }

    simd_base& operator=(simd_base x) {
        std::memcpy(&value_, &x.value_, sizeof(vector_type));
    }

    template <typename... Args,
        typename = typename std::enable_if<1+sizeof...(Args)==Impl::width()>::type>
    simd_base(scalar_type v0, Args... rest) {
        value_ = Impl::immediate(v0, rest...);
    }

    void copy_to(scalar_type* p) const {
        Impl::copy_to(value_, p);
    }

    void copy_from(const scalar_type* p) {
        value_ = Impl::copy_from(p);
    }

    friend simd_base operator+(simd_base a, simd_base b) {
        return simd_base(Impl::add(a.value_, b.value_));
    }

    simd_base& operator+=(simd_base a) {
        return (*this) = (*this)+a;
    }

    friend simd_base operator-(simd_base a, simd_base b) {
        return simd_base(Impl::sub(a.value_, b.value_));
    }

    simd_base& operator-=(simd_base a) {
        return (*this) = (*this)-a;
    }

    friend simd_base operator*(simd_base a, simd_base b) {
        return simd_base(Impl::mul(a.value_, b.value_));
    }

    simd_base& operator*=(simd_base a) {
        return (*this) = (*this)*a;
    }

    friend simd_base operator/(simd_base a, simd_base b) {
        return simd_base(Impl::div(a.value_, b.value_));
    }

    simd_base& operator/=(simd_base a) {
        return (*this) = (*this)/a;
    }

    friend simd_base fma(simd_base a, simd_base b, simd_base c) {
        return simd_base(Impl::fma(a.value_, b.value_, c.value_));
    }

};

template <typename T, unsigned N>
using simd = simd_base<typename simd_detail::native<T, N>::type>;

} // namespace hf

#endif // ndfef HF_SIMD_AVX_H
