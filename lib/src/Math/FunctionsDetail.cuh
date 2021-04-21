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

__host__ __device__
inline Vector3float
Math::refract(const float& eta, const Vector3float& inc_dir, const Vector3float& normal)
{
    float NdotI = dot(inc_dir, normal);
    float k = 1.0f - eta * eta * (1.0f - NdotI*NdotI);
    if(k < 0)
    {
        return 0;
    }
    else
    {
        return -eta * inc_dir - (-eta * NdotI + sqrtf(k))*normal;
    }
}

__host__ __device__
inline float
Math::fresnel_schlick(const float& F0, const float& VdotH)
{
    return F0 + (1.0f - F0) * powf(1.0f - VdotH, 5.0f);
}

#endif