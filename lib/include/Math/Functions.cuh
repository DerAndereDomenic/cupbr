#ifndef __CUPBR_MATH_FUNCTIONS_CUH
#define __CUPBR_MATH_FUNCTIONS_CUH

#define EPSILON 1e-5f

namespace Math
{
    /**
    *   @brief Test if two floats are equal up to an epsilon
    *   @param[in] lhs The left hand side
    *   @param[in] rhs The right hand side
    *   @param[in] eps The epsilon (defaul=1e-5)
    *   @return True if |lhs-rhs|<eps 
    */
    __host__ __device__
    bool
    safeFloatEqual(const float& lhs, const float& rhs, const float& eps = EPSILON);

    /**
    *   @brief Clamp a value
    *   @tparam The type
    *   @param[in] x The value to clamp
    *   @param[in] mini The left side of the interval
    *   @param[in] maxi The right side of the interval
    *   @return The clamped value  
    */
    template<typename T>
    __host__ __device__
    T
    clamp(const T& x, const T& mini, const T& maxi);

}

#include "../../src/Math/FunctionsDetail.cuh"

#endif