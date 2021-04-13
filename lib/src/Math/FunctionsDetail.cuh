#ifndef __CUPBR_MATH_FUNCTIONSDETAIL_CUH
#define __CUPBR_MATH_FUNCTIONSDETAIL_CUH

#include <Core/CUDA.cuh>

__host__ __device__
inline bool
Math::safeFloatEqual(const float& lhs, const float& rhs, const float& eps)
{
    return fabsf(lhs-rhs) < eps;
}

template<typename T>
T
Math::clamp(const T& x, const T& mini, const T& maxi)
{
    return min(max(x, mini),maxi);
}

__host__ __device__
inline float
Math::delta(const float& inp)
{
    return safeFloatEqual(inp, 0.0f) ? 1.0f : 0.0f;
}

__host__ __device__
inline Vector3float
Math::reflect(const Vector3float& inc_dir, const Vector3float& normal)
{
    return normalize(2.0f*dot(inc_dir,normal)*normal-inc_dir);
}

#endif