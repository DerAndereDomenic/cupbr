#ifndef __CUPBR_MATH_VECTOROPERATIONS_H
#define __CUPBR_MATH_VECTOROPERATIONS_H

#include <Core/CUDA.h>
#include "VectorTypes_fwd.h"

namespace cupbr
{
    /**
    *   @brief Add two vectors
    *   @param[in] lhs The first vector
    *   @param[in] rhs The second vector
    *   @return The elementwise sum of these two vectors
    */
    template<typename T>
    __host__ __device__
    Vector2<T> operator+(const Vector2<T>& lhs, const Vector2<T>& rhs);

    /**
    *   @brief Subtract two vectors
    *   @param[in] lhs The first vector
    *   @param[in] rhs The second vector
    *   @return The elementwise subtraction of these two vectors
    */
    template<typename T>
    __host__ __device__
    Vector2<T> operator-(const Vector2<T>& lhs, const Vector2<T>& rhs);

    /**
    *   @brief Multiply a vector with a scalar
    *   @param[in] lhs The vector
    *   @param[in] rhs The scalar
    *   @return The elementwise product of the vector with the scalar
    */
    template<typename T>
    __host__ __device__
    Vector2<T> operator*(const Vector2<T>& lhs, const T& rhs);

    /**
    *   @brief Multiply a vector with a scalar
    *   @param[in] lhs The scalar
    *   @param[in] rhs The vector
    *   @return The elementwise product of the vector with the scalar
    */
    template<typename T>
    __host__ __device__
    Vector2<T> operator*(const T& lhs, const Vector2<T>& rhs);

    /**
    *   @brief Multiply two vectors
    *   @param[in] lhs The first vector
    *   @param[in] rhs The second vector
    *   @return Elementwise product between these two vectors
    */
    template<typename T>
    __host__ __device__
    Vector2<T> operator*(const Vector2<T>& lhs, const Vector2<T>& rhs);

    /**
    *   @brief Divide a vector by a scalar
    *   @param[in] lhs The vector
    *   @param[in] rhs The scalar
    *   @return The vector where each component was divided by the scalar
    */
    template<typename T>
    __host__ __device__
    Vector2<T> operator/(const Vector2<T>& lhs, const T& rhs);

    /**
    *   @brief Divide a vector by a vector (component-wise)
    *   @param[in] lhs The first vector
    *   @param[in] rhs The second
    *   @return The vectors divided component wise
    */
    template<typename T>
    __host__ __device__
    Vector2<T> operator/(const Vector2<T>& lhs, const Vector2<T>& rhs);

    /**
    *   @brief Adds the rhs to the lhs
    *   @param[in] lhs The first vector
    *   @param[in] rhs The second vector
    */
    template<typename T>
    __host__ __device__
    void operator+=(Vector2<T>& lhs, const Vector2<T>& rhs);

    /**
    *   @brief Substracts the rhs from the lhs
    *   @param[in] lhs The first vector
    *   @param[in] rhs The second vector
    */
    template<typename T>
    __host__ __device__
    void operator-=(Vector2<T>& lhs, const Vector2<T>& rhs);

    /**
    *   @brief Multiplies a vector with a scalar
    *   @param[in] lhs The vector
    *   @param[in] rhs The scalar
    */
    template<typename T>
    __host__ __device__
    void operator*=(Vector2<T>& lhs, const T& rhs);

    /**
    *   @brief Divides a vector component wise with a scalar
    *   @param[in] lhs The vector
    *   @param[in] rhs The scalar
    */
    template<typename T>
    __host__ __device__
    void operator/=(Vector2<T>& lhs, const T& rhs);

    /**
    *   @brief Add two vectors
    *   @param[in] lhs The first vector
    *   @param[in] rhs The second vector
    *   @return The elementwise sum of these two vectors
    */
    template<typename T>
    __host__ __device__
    Vector3<T> operator+(const Vector3<T>& lhs, const Vector3<T>& rhs);

    /**
    *   @brief Subtract two vectors
    *   @param[in] lhs The first vector
    *   @param[in] rhs The second vector
    *   @return The elementwise subtraction of these two vectors
    */
    template<typename T>
    __host__ __device__
    Vector3<T> operator-(const Vector3<T>& lhs, const Vector3<T>& rhs);

    /**
    *   @brief Multiply a vector with a scalar
    *   @param[in] lhs The vector
    *   @param[in] rhs The scalar
    *   @return The elementwise product of the vector with the scalar
    */
    template<typename T>
    __host__ __device__
    Vector3<T> operator*(const Vector3<T>& lhs, const T& rhs);

    /**
    *   @brief Multiply a vector with a scalar
    *   @param[in] lhs The scalar
    *   @param[in] rhs The vector
    *   @return The elementwise product of the vector with the scalar
    */
    template<typename T>
    __host__ __device__
    Vector3<T> operator*(const T& lhs, const Vector3<T>& rhs);

    /**
    *   @brief Multiply two vectors
    *   @param[in] lhs The first vector
    *   @param[in] rhs The second vector
    *   @return Elementwise product between these two vectors
    */
    template<typename T>
    __host__ __device__
    Vector3<T> operator*(const Vector3<T>& lhs, const Vector3<T>& rhs);

    /**
    *   @brief Divide a vector by a scalar
    *   @param[in] lhs The vector
    *   @param[in] rhs The scalar
    *   @return The vector where each component was divided by the scalar
    */
    template<typename T>
    __host__ __device__
    Vector3<T> operator/(const Vector3<T>& lhs, const T& rhs);

    /**
    *   @brief Divide a vector by a vector (component-wise)
    *   @param[in] lhs The first vector
    *   @param[in] rhs The second
    *   @return The vectors divided component wise
    */
    template<typename T>
    __host__ __device__
    Vector3<T> operator/(const Vector3<T>& lhs, const Vector3<T>& rhs);

    /**
    *   @brief Adds the rhs to the lhs
    *   @param[in] lhs The first vector
    *   @param[in] rhs The second vector
    */
    template<typename T>
    __host__ __device__
    void operator+=(Vector3<T>& lhs, const Vector3<T>& rhs);

    /**
    *   @brief Substracts the rhs from the lhs
    *   @param[in] lhs The first vector
    *   @param[in] rhs The second vector
    */
    template<typename T>
    __host__ __device__
    void operator-=(Vector3<T>& lhs, const Vector3<T>& rhs);

    /**
    *   @brief Multiplies a vector with a scalar
    *   @param[in] lhs The vector
    *   @param[in] rhs The scalar
    */
    template<typename T>
    __host__ __device__
    void operator*=(Vector3<T>& lhs, const T& rhs);

    /**
    *   @brief Divides a vector component wise with a scalar
    *   @param[in] lhs The vector
    *   @param[in] rhs The scalar
    */
    template<typename T>
    __host__ __device__
    void operator/=(Vector3<T>& lhs, const T& rhs);

    /**
    *   @brief Add two vectors
    *   @param[in] lhs The first vector
    *   @param[in] rhs The second vector
    *   @return The elementwise sum of these two vectors
    */
    template<typename T>
    __host__ __device__
    Vector4<T> operator+(const Vector4<T>& lhs, const Vector4<T>& rhs);

    /**
    *   @brief Subtract two vectors
    *   @param[in] lhs The first vector
    *   @param[in] rhs The second vector
    *   @return The elementwise subtraction of these two vectors
    */
    template<typename T>
    __host__ __device__
    Vector4<T> operator-(const Vector4<T>& lhs, const Vector4<T>& rhs);

    /**
    *   @brief Multiply a vector with a scalar
    *   @param[in] lhs The vector
    *   @param[in] rhs The scalar
    *   @return The elementwise product of the vector with the scalar
    */
    template<typename T>
    __host__ __device__
    Vector4<T> operator*(const Vector4<T>& lhs, const T& rhs);

    /**
    *   @brief Multiply a vector with a scalar
    *   @param[in] lhs The scalar
    *   @param[in] rhs The vector
    *   @return The elementwise product of the vector with the scalar
    */
    template<typename T>
    __host__ __device__
    Vector4<T> operator*(const T& lhs, const Vector4<T>& rhs);

    /**
    *   @brief Multiply two vectors
    *   @param[in] lhs The first vector
    *   @param[in] rhs The second vector
    *   @return Elementwise product between these two vectors
    */
    template<typename T>
    __host__ __device__
    Vector4<T> operator*(const Vector4<T>& lhs, const Vector4<T>& rhs);

    /**
    *   @brief Divide a vector by a scalar
    *   @param[in] lhs The vector
    *   @param[in] rhs The scalar
    *   @return The vector where each component was divided by the scalar
    */
    template<typename T>
    __host__ __device__
    Vector4<T> operator/(const Vector4<T>& lhs, const T& rhs);

    /**
    *   @brief Divide a vector by a vector (component-wise)
    *   @param[in] lhs The first vector
    *   @param[in] rhs The second
    *   @return The vectors divided component wise
    */
    template<typename T>
    __host__ __device__
    Vector4<T> operator/(const Vector4<T>& lhs, const Vector4<T>& rhs);

    /**
    *   @brief Adds the rhs to the lhs
    *   @param[in] lhs The first vector
    *   @param[in] rhs The second vector
    */
    template<typename T>
    __host__ __device__
    void operator+=(Vector4<T>& lhs, const Vector4<T>& rhs);

    /**
    *   @brief Substracts the rhs from the lhs
    *   @param[in] lhs The first vector
    *   @param[in] rhs The second vector
    */
    template<typename T>
    __host__ __device__
    void operator-=(Vector4<T>& lhs, const Vector4<T>& rhs);

    /**
    *   @brief Multiplies a vector with a scalar
    *   @param[in] lhs The vector
    *   @param[in] rhs The scalar
    */
    template<typename T>
    __host__ __device__
    void operator*=(Vector4<T>& lhs, const T& rhs);

    /**
    *   @brief Divides a vector component wise with a scalar
    *   @param[in] lhs The vector
    *   @param[in] rhs The scalar
    */
    template<typename T>
    __host__ __device__
    void operator/=(Vector4<T>& lhs, const T& rhs);

} //namespace cupbr

#include "../../src/Math/VectorOperationsDetail.h"

#endif