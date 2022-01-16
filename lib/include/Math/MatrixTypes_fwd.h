#ifndef __CUPBR_MATH_MATRIXTYPES_FWD
#define __CUPBR_MATH_MATRIXTYPES_FWD

#include <cstdint>

namespace cupbr
{
    template<typename T>
    struct Matrix2x2;

    template<typename T>
    struct Matrix3x3;

    template<typename T>
    struct Matrix4x4;

    using Matrix2x2int8_t  = Matrix2x2<int8_t>;
    using Matrix2x2int16_t = Matrix2x2<int16_t>;
    using Matrix2x2int32_t = Matrix2x2<int32_t>;
    using Matrix2x2int64_t = Matrix2x2<int64_t>;

    using Matrix2x2uint8_t  = Matrix2x2<uint8_t>;
    using Matrix2x2uint16_t = Matrix2x2<uint16_t>;
    using Matrix2x2uint32_t = Matrix2x2<uint32_t>;
    using Matrix2x2uint64_t = Matrix2x2<uint64_t>;

    using Matrix2x2float  = Matrix2x2<float>;
    using Matrix2x2double = Matrix2x2<double>;

    using Matrix3x3int8_t  = Matrix3x3<int8_t>;
    using Matrix3x3int16_t = Matrix3x3<int16_t>;
    using Matrix3x3int32_t = Matrix3x3<int32_t>;
    using Matrix3x3int64_t = Matrix3x3<int64_t>;

    using Matrix3x3uint8_t  = Matrix3x3<uint8_t>;
    using Matrix3x3uint16_t = Matrix3x3<uint16_t>;
    using Matrix3x3uint32_t = Matrix3x3<uint32_t>;
    using Matrix3x3uint64_t = Matrix3x3<uint64_t>;

    using Matrix3x3float  = Matrix3x3<float>;
    using Matrix3x3double = Matrix3x3<double>;

    using Matrix4x4int8_t  = Matrix4x4<int8_t>;
    using Matrix4x4int16_t = Matrix4x4<int16_t>;
    using Matrix4x4int32_t = Matrix4x4<int32_t>;
    using Matrix4x4int64_t = Matrix4x4<int64_t>;

    using Matrix4x4uint8_t  = Matrix4x4<uint8_t>;
    using Matrix4x4uint16_t = Matrix4x4<uint16_t>;
    using Matrix4x4uint32_t = Matrix4x4<uint32_t>;
    using Matrix4x4uint64_t = Matrix4x4<uint64_t>;

    using Matrix4x4float  = Matrix4x4<float>;
    using Matrix4x4double = Matrix4x4<double>;

} //namespace cupbr

#endif