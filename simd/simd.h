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
struct simd_mask_base {
    using scalar_type = typename Impl::scalar_type;
    using vector_type = typename Impl::vector_type;
    using value_type = bool;

    vector_type value_;

    static constexpr std::size_t width() {
        return Impl::width();
    }

    simd_mask_base(value_type v) {
        value_ = Impl::broadcast(scalar_type(v));
    }

    simd_mask_base(simd_mask_base v) {
        std::memcpy(&value_, &v, sizeof(vector_type));
    }

    template <typename... Args,
        typename = typename std::enable_if<1+sizeof...(Args)==Impl::width()>::type>
    simd_mask_base(value_type v0, Args... rest) {
        value_ = Impl::immediate(v0, scalar_type(rest)...);
    }

    void copy_to(scalar_type* p) const {
        Impl::copy_to(value_, p);
    }

    void copy_from(const scalar_type* p) {
        value_ = Impl::copy_from(p);
    }

    friend simd_mask_base operator==(const simd_mask_base& x, const simd_mask_base& y) const {
        return Impl::eq(x, y);
    }

    friend simd_mask_base operator==(const simd_mask_base& x, const simd_mask_base& y) const {
        return Impl::not_eq(x, y);
    }

    vector_type value_;
};

template <typename Impl>
struct simd_base {
    using scalar_type = typename Impl::scalar_type;
    using vector_type = typename Impl::vector_type;
    using mask_type =   simd_mask_base<typename Impl::mask_impl>;
    using value_type =  scalar_type;

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

    exlicit simd_base(scalar_type* p) {
        *this = Impl::copy_from(p);
    }

    template <typename... Args,
        typename = typename std::enable_if<1+sizeof...(Args)==Impl::width()>::type>
    simd_base(value_type v0, Args... rest) {
        value_ = Impl::immediate(scalar_type(v0), scalar_type(rest)...);
    }

    template <typename X,
        typename = typename std::enable_if<
            Impl::is_convertible<X::vector_type>::value &&
            Impl::width()==X::width()
        >::type>
    explicit simd_base(const simd_base<X>& other) {
        value_ = Impl::convert(other.value_);
    }

    template <typename X,
        typename = typename std::enable_if<
            !Impl::is_convertible<X::vector_type>::value &&
            Impl::width()==X::width()
        >::type>
    explicit simd_base(const simd_base<X>& other) {
        typename simd_base<X>::scalar_type from[width()];
        scalar_type to[width()];

        other.copy_to(from);
        for (unsigned i = 0; i<width(); ++i) {
            to[i] = scalar_cast<scalar_type>(from[i]);
        }
        copy_from(to);
    }

    simd_base& operator=(simd_base x) {
        std::memcpy(&value_, &x.value_, sizeof(vector_type));
    }

    void copy_to(scalar_type* p) const {
        Impl::copy_to(value_, p);
    }

    void copy_from(const scalar_type* p) {
        value_ = Impl::copy_from(p);
    }

    friend mask_type operator==(const simd_base& x, const simd_base& y) {
        return mask_type(Impl::eq(x, y));
    }

    friend simd_mask_base operator!=(const simd_mask_base& x, const simd_mask_base& y) const {
        return mask_type(Impl::not_eq(x, y));
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

    friend simd_base floor(simd_base a) {
        return simd_base(Impl::floor(a));
    }

    friend simd_base ceil(simd_base a) {
        return simd_base(Impl::ceil(a));
    }

    scalar_type operator[](int i) const {
        return Impl::element(value_, i);
    }

    struct reference {
        reference() = delete;
        reference(const reference&) = delete;
        reference& operator=(const reference&) = delete;

        reference(vector_type& value, int i):
            value_(value), i(i) {}

        reference& operator=(value_type v) && {
            Impl::set_element(value_, i, scalar_type(v));
            return *this;
        }

        operator value_type() const {
            return value_type(Impl::element(value_, i));
        }

        vector_type value_;
        int i;
    };

    reference operator[](int i) const {
        return reference(value_, i);
    }
};

template <typename Impl>
struct simd_mask_base: simd_base<Impl> {
    // Scalar type may be wider than the value_type bool.
    using value_type = bool;

    simd_mask_base(vector_type v) {
        std::memcpy(&value_, &v, sizeof(vector_type));
    }

    simd_mask_base(scalar_type v) {
        value_ = Impl::broadcast(v);
    }

    exlicit simd_mask_base(scalar_type* p) {
        *this = Impl::copy_from(p);
    }

    simd_mask_base operator!() const {
        return Impl::logical_not(value_);
    }

    friend simd_mask_base operator&&(const simd_mask_base& x, const simd_mask_base& y) {
        return Impl::logical_and(x.value_, y.value_);
    }

    friend simd_mask_base operator||(const simd_mask_base& x, const simd_mask_base& y) {
        return Impl::logical_or(x.value_, y.value_);
    }
};

template <typename Simd>
struct where_expression {
    using simd_type = Simd;
    using mask_type = typename Simd::mask_type;
    using value_type = typename Simd::value_type;

    const mask_type& mask_;
    simd_type& data_;

    where_expression(const where_expression&) = delete;
    where_expression& operator=(const where_expression&) = delete;

    where_expression(const mask_type& m, simd_type& v):
        mask_(m), data_(v) {}

    value_type operator=(value_type v) {
        simd_type c(v);
        data_ = Impl::blend(mask_, data_, c.value_);
        return v;
    }

    simd_type operator=(const simd_type& v) {
        data_ = Impl::blend(mask_, data_, v.value_);
        return v;
    }
};

template <typename Simd>
where_expression<Simd> where(typename Simd::mask_type m, Simd& v) {
    return where_expression<Simd>(m, v);
}

template <typename T, unsigned N>
using simd = simd_base<typename simd_detail::native<T, N>::type>;

template <typename T, unsigned N>
using simd_mask = simd<T, N>::mask_type;

} // namespace hf

#endif // ndfef HF_SIMD_AVX_H
