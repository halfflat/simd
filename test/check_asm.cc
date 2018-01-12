#include "simd/simd.h"
#include "simd/avx.h"
#include "simd/generic.h"

using namespace hf;

using native_d4 = simd<double, 4>;
using generic_d4 = simd_generic<double, 4>;

native_d4 mean3_native(native_d4 a, native_d4 b, native_d4 c) {
    return (a+b+c)/3;
}

generic_d4 mean3_generic(generic_d4 a, generic_d4 b, generic_d4 c) {
    return (a+b+c)/3;
}
