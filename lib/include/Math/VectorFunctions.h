#ifndef __CUPBR_MATH_VECTORFUNCTIONS_H
#define __CUPBR_MATH_VECTORFUNCTIONS_H

#include <Math/VectorTypes_fwd.h>
#include <Core/CUDA.h>

namespace cupbr
{
    namespace Math
    {
        /**
        *	@brief Get the norm of a vector
        *	@tparam T The vector data type
        *	@param v The vector
        *	@return The L2 norm
        */
        template<typename T>
        __host__ __device__
        T norm(const Vector2<T>& v);

        /**
        *	@brief Calculates the dot product
        *	@tparam T The vector data type
        *	@param v The first vector
        *	@param w The second vector
        *	@return The dot product
        */
        template<typename T>
        __host__ __device__
        T dot(const Vector2<T>& v, const Vector2<T>& w);

        /**
        *	@brief Normalizes the given vector
        *	@tparam T The vector data type
        *	@param v The vector to normalize
        *	@return The normalized vector
        */
        template<typename T>
        __host__ __device__
        Vector2<T> normalize(const Vector2<T>& v);

        /**
        *   @brief Compute component-wise exponential function of a vector
        *   @param[in] v The vector
        *   @return exp(v) of the vector
        */
        template<typename T>
        __host__ __device__
        Vector2<T> exp(const Vector2<T>& v);

        /**
        *	@brief Get the norm of a vector
        *	@tparam T The vector data type
        *	@param v The vector
        *	@return The L2 norm
        */
        template<typename T>
        __host__ __device__
        T norm(const Vector3<T>& v);

        /**
        *	@brief Calculates the dot product
        *	@tparam T The vector data type
        *	@param v The first vector
        *	@param w The second vector
        *	@return The dot product
        */
        template<typename T>
        __host__ __device__
        T dot(const Vector3<T>& v, const Vector3<T>& w);

        /**
        *	@brief Normalizes the given vector
        *	@tparam T The vector data type
        *	@param v The vector to normalize
        *	@return The normalized vector
        */
        template<typename T>
        __host__ __device__
        Vector3<T> normalize(const Vector3<T>& v);

        /**
        *   @brief Compute component-wise exponential function of a vector
        *   @param[in] v The vector
        *   @return exp(v) of the vector
        */
        template<typename T>
        __host__ __device__
        Vector3<T> exp(const Vector3<T>& v);

        /**
        *	@brief Computes the cross product between two vectors
        *	@tparam T The vector data type
        *	@param[in] v The first vector
        *	@param[in] w The second vector
        *	@return The cross product
        */
        template<typename T>
        __host__ __device__
        Vector3<T> cross(const Vector3<T>& v, const Vector3<T>& w);

        /**
        *	@brief Get the norm of a vector
        *	@tparam T The vector data type
        *	@param v The vector
        *	@return The L2 norm
        */
        template<typename T>
        __host__ __device__
        T norm(const Vector4<T>& v);

        /**
        *	@brief Calculates the dot product
        *	@tparam T The vector data type
        *	@param v The first vector
        *	@param w The second vector
        *	@return The dot product
        */
        template<typename T>
        __host__ __device__
        T dot(const Vector4<T>& v, const Vector4<T>& w);

        /**
        *	@brief Normalizes the given vector
        *	@tparam T The vector data type
        *	@param v The vector to normalize
        *	@return The normalized vector
        */
        template<typename T>
        __host__ __device__
        Vector4<T> normalize(const Vector4<T>& v);

        /**
        *   @brief Compute component-wise exponential function of a vector
        *   @param[in] v The vector
        *   @return exp(v) of the vector
        */
        template<typename T>
        __host__ __device__
        Vector4<T> exp(const Vector4<T>& v);

    } //namespace Math
} //namespace cupbr

#include "../../src/Math/VectorFunctionsDetail.h"

#endif