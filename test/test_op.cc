#include <array>
#include <initializer_list>
#include <vector>
#include <iostream>
#include <iterator>

#include "simd/simd.h"
#include "simd/avx.h"
#include "simd/generic.h"

#include "gtest/gtest.h"

using namespace hf;

template <typename T>
struct sequence {
    std::vector<T> data;
    sequence(std::initializer_list<T> l): data(l) {}

    template <typename X>
    bool operator==(const X& x) const {
        // (for C++14 could just use std::equal)
        auto bi = std::begin(data);
        auto ei = std::end(data);
        auto bj = std::begin(x);
        auto ej = std::end(x);
        while (bi!=ei && bj!=ej) {
            if (*bi++!=*bj++) return false;
        }
        return bi==ei && bj==ej;
    }

    template <typename X>
    bool operator!=(const X& x) const {
        return !(*this==x);
    }
};

template <typename... T>
sequence<typename std::common_type<T...>::type> seq(T... xs) {
    using U = typename std::common_type<T...>::type;
    return sequence<U>({U(xs)...});
}

template <typename T>
std::ostream& std::operator<<(std::ostream& o, const sequence<T>& s) {
    o << '{';
    for (std::size_t i = 0; i<s.data.size(); ++i) {
        if (i) o << ", ";
        o << s.data[i];
    }
    o << '}';
    return o;
}

template <typename Simd>
std::array<typename Simd::scalar_type, Simd::width()> unpack(const Simd& packed) {
    std::array<typename Simd::scalar_type, Simd::width()> a;
    packed.copy_to(a.data());
    return a;
}

TEST(simd, init) {
    using double4 = simd<double, 4>;

    double4 a(2.);
    EXPECT_EQ(seq(2, 2, 2, 2), unpack(a));

    double4 b(3, 5, 7, 9);
    EXPECT_EQ(seq(3, 5, 7, 9), unpack(b));
}

TEST(simd, arithmetic) {
    using double4 = simd<double, 4>;

    double4 a(0.5,  1, 0.25, -1);
    double4 b(  3,  5,    7,  9);
    double4 c(  2, -2,    3, -3);

    EXPECT_EQ(seq( 3.5,  6,  7.25,   8), unpack(a+b));
    EXPECT_EQ(seq(-2.5, -4, -6.75, -10), unpack(a-b));
    EXPECT_EQ(seq( 1.5,  5,  1.75,  -9), unpack(a*b));
    EXPECT_EQ(seq( 3.5,  3,  4.75, -12), unpack(fma(a, b, c)));
    EXPECT_EQ(seq(  6.,  5,    28,  -9), unpack(b/a));
}

TEST(simd, generic_arithmetic) {
    using double4 = simd_generic<double, 4>;

    double4 a(0.5,  1, 0.25, -1);
    double4 b(  3,  5,    7,  9);
    double4 c(  2, -2,    3, -3);

    EXPECT_EQ(seq( 3.5,  6,  7.25,   8), unpack(a+b));
    EXPECT_EQ(seq(-2.5, -4, -6.75, -10), unpack(a-b));
    EXPECT_EQ(seq( 1.5,  5,  1.75,  -9), unpack(a*b));
    EXPECT_EQ(seq( 3.5,  3,  4.75, -12), unpack(fma(a, b, c)));
    EXPECT_EQ(seq(  6.,  5,    28,  -9), unpack(b/a));
}
