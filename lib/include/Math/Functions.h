#ifndef __CUPBR_MATH_FUNCTIONS_H
#define __CUPBR_MATH_FUNCTIONS_H

#define EPSILON 1e-5f

#include <Core/CUDA.h>

namespace cupbr
{
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
        bool safeFloatEqual(const float& lhs, const float& rhs, const float& eps = EPSILON);

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
        T clamp(const T& x, const T& mini, const T& maxi);

        /**
        *   @brief Delta distribution
        *   @param[in] inp
        *   @return One if inp equals zero, zero else
        */
        __host__ __device__
        float delta(const float& inp);

        /**
        *   @brief Reflect the incoming direction on the normal
        *   @param[in] inc_dir The incoming direction
        *   @param[in] normal The surface normal
        *   @return The reflected direction
        *   @pre dot(inc_dir,normal) > 0
        *   @post dot(out_dir,normal) > 0
        */
        __host__ __device__
        Vector3float reflect(const Vector3float& inc_dir, const Vector3float& normal);

        /**
        *   @brief Refract the incoming direction on the normal
        *   @param[in] eta Index of refraction
        *   @param[in] inc_dir The incoming direction
        *   @param[in] normal The surface normal
        *   @return The refracted direction
        */
        __host__ __device__
        Vector3float refract(const float& eta, const Vector3float& inc_dir, const Vector3float& normal);

        /**
        *   @brief Fresnel schlick
        *   @param[in] F0 The F0 term
        *   @param[in] VdotH cos of the angle between incoming direction and normal
        *   @return The fresnel schlick term
        */
        __host__ __device__
        float fresnel_schlick(const float& F0, const float& VdotH);

        /**
        *   @brief Fresnel schlick
        *   @param[in] F0 The F0 term
        *   @param[in] VdotH cos of the angle between incoming direction and normal
        *   @return The fresnel schlick term
        */
        __host__ __device__
        Vector3float fresnel_schlick(const Vector3float& F0, const float& VdotH);

        /**
        *   @brief Create a seed for the rng
        *   @param[in] val0 Thread id
        *   @param[in] val1 Subframe index
        *   @return The seed
        */
        template<uint32_t N>
        __host__ __device__
        uint32_t tea(uint32_t val0, uint32_t val1);

        /**
        *   @brief Create a random float between 0 and 1
        *   @param[in] seed The seed
        *   @return The random number
        */
        __host__ __device__
        float rnd(uint32_t& prev);

        /**
        *   @brief Convert a direction into a local coordinate frame
        *   @param[in] N The surface normal
        *   @param[in] direction The direction to convert
        *   @return The direction in local coordinates
        */
        __host__ __device__
        Vector3float toLocalFrame(const Vector3float& N, const Vector3float& direction);

    } //namespace Math
} //namespace cupbr

#include "../../src/Math/FunctionsDetail.h"

#endif