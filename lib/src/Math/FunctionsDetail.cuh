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

#endif