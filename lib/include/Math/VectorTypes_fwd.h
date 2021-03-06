#ifndef __CUPBR_MATH_VECTORTYPES_FWD
#define __CUPBR_MATH_VECTORTYPES_FWD

#include <cstdint>

namespace cupbr
{
    template<typename T>
    struct Vector2;

    template<typename T>
    struct Vector3;

    template<typename T>
    struct Vector4;

    using Vector2int8_t = Vector2<int8_t>;
    using Vector2int16_t = Vector2<int16_t>;
    using Vector2int32_t = Vector2<int32_t>;
    using Vector2int64_t = Vector2<int64_t>;

    using Vector2uint8_t = Vector2<uint8_t>;
    using Vector2uint16_t = Vector2<uint16_t>;
    using Vector2uint32_t = Vector2<uint32_t>;
    using Vector2uint64_t = Vector2<uint64_t>;

    using Vector2float = Vector2<float>;
    using Vector2double = Vector2<double>;

    using Vector3int8_t = Vector3<int8_t>;
    using Vector3int16_t = Vector3<int16_t>;
    using Vector3int32_t = Vector3<int32_t>;
    using Vector3int64_t = Vector3<int64_t>;

    using Vector3uint8_t = Vector3<uint8_t>;
    using Vector3uint16_t = Vector3<uint16_t>;
    using Vector3uint32_t = Vector3<uint32_t>;
    using Vector3uint64_t = Vector3<uint64_t>;

    using Vector3float = Vector3<float>;
    using Vector3double = Vector3<double>;

    using Vector4int8_t = Vector4<int8_t>;
    using Vector4int16_t = Vector4<int16_t>;
    using Vector4int32_t = Vector4<int32_t>;
    using Vector4int64_t = Vector4<int64_t>;

    using Vector4uint8_t = Vector4<uint8_t>;
    using Vector4uint16_t = Vector4<uint16_t>;
    using Vector4uint32_t = Vector4<uint32_t>;
    using Vector4uint64_t = Vector4<uint64_t>;

    using Vector4float = Vector4<float>;
    using Vector4double = Vector4<double>;

} //namespace cupbr

#endif