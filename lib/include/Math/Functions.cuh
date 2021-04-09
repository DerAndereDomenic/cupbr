#ifndef __CUPBR_MATH_FUNCTIONS_CUH
#define __CUPBR_MATH_FUNCTIONS_CUH

#define EPSILON 1e-5f

namespace Math
{
    __host__ __device__
    bool
    safeFloatEqual(const float& lhs, const float& rhs, const float& eps = EPSILON);
}

#include "../../src/Math/FunctionsDetail.cuh"

#endif