#ifndef __CUPBR_MATH_MATRIXOPERATIONS_H
#define __CUPBR_MATH_MATRIXOPERATIONS_H

#include <Core/CUDA.h>
#include "MatrixTypes_fwd.h"
#include "VectorTypes_fwd.h"

namespace cupbr
{
    /**
    *   @brief Add two matrices
    *   @param[in] lhs The first matrix
    *   @param[in] rhs The second matrix
    *   @return The elementwise sum of these two matrices
    */
    template<typename T>
    __host__ __device__
    Matrix2x2<T> operator+(Matrix2x2<T>& lhs, Matrix2x2<T>& rhs);

    /**
    *   @brief Subtract two matrices
    *   @param[in] lhs The first matrix
    *   @param[in] rhs The second matrix
    *   @return The elementwise subtraction of these two matrices
    */
    template<typename T>
    __host__ __device__
    Matrix2x2<T> operator-(Matrix2x2<T>& lhs, Matrix2x2<T>& rhs);

    /**
    *   @brief Multiply a matrix with a scalar
    *   @param[in] lhs The matrix
    *   @param[in] rhs The scalar
    *   @return The elementwise product of the matrix with the scalar
    */
    template<typename T>
    __host__ __device__
    Matrix2x2<T> operator*(Matrix2x2<T>& lhs, const T& rhs);

    /**
    *   @brief Multiply a matrix with a scalar
    *   @param[in] lhs The scalar
    *   @param[in] rhs The matrix
    *   @return The elementwise product of the matrix with the scalar
    */
    template<typename T>
    __host__ __device__
    Matrix2x2<T> operator*(const T& lhs, Matrix2x2<T>& rhs);

    /**
    *   @brief Multiply two matrices
    *   @param[in] lhs The first matrix
    *   @param[in] rhs The second matrix
    *   @return Matrix multiplication between two matrices
    */
    template<typename T>
    __host__ __device__
    Matrix2x2<T> operator*(Matrix2x2<T>& lhs, Matrix2x2<T>& rhs);

    /**
    *   @brief Multiply a matrix with a vector
    *   @param[in] lhs The matrix
    *   @param[in] rhs The vector
    *   @return The matrix vector product
    */
    template<typename T>
    __host__ __device__
    Vector2<T> operator*(Matrix2x2<T>& lhs, const Vector2<T>& rhs);

    /**
    *   @brief Add two matrices
    *   @param[in] lhs The first matrix
    *   @param[in] rhs The second matrix
    *   @return The elementwise sum of these two matrices
    */
    template<typename T>
    __host__ __device__
    Matrix3x3<T> operator+(Matrix3x3<T>& lhs, Matrix3x3<T>& rhs);

    /**
    *   @brief Subtract two matrices
    *   @param[in] lhs The first matrix
    *   @param[in] rhs The second matrix
    *   @return The elementwise subtraction of these two matrices
    */
    template<typename T>
    __host__ __device__
    Matrix3x3<T> operator-(Matrix3x3<T>& lhs, Matrix3x3<T>& rhs);

    /**
    *   @brief Multiply a matrix with a scalar
    *   @param[in] lhs The matrix
    *   @param[in] rhs The scalar
    *   @return The elementwise product of the matrix with the scalar
    */
    template<typename T>
    __host__ __device__
    Matrix3x3<T> operator*(Matrix3x3<T>& lhs, const T& rhs);

    /**
    *   @brief Multiply a matrix with a scalar
    *   @param[in] lhs The scalar
    *   @param[in] rhs The matrix
    *   @return The elementwise product of the matrix with the scalar
    */
    template<typename T>
    __host__ __device__
    Matrix3x3<T> operator*(const T& lhs, Matrix3x3<T>& rhs);

    /**
    *   @brief Multiply two matrices
    *   @param[in] lhs The first matrix
    *   @param[in] rhs The second matrix
    *   @return Matrix multiplication between two matrices
    */
    template<typename T>
    __host__ __device__
    Matrix3x3<T> operator*(Matrix3x3<T>& lhs, Matrix3x3<T>& rhs);

    /**
    *   @brief Multiply a matrix with a vector
    *   @param[in] lhs The matrix
    *   @param[in] rhs The vector
    *   @return The matrix vector product
    */
    template<typename T>
    __host__ __device__
    Vector3<T> operator*(Matrix3x3<T>& lhs, const Vector3<T>& rhs);

    /**
    *   @brief Add two matrices
    *   @param[in] lhs The first matrix
    *   @param[in] rhs The second matrix
    *   @return The elementwise sum of these two matrices
    */
    template<typename T>
    __host__ __device__
    Matrix4x4<T> operator+(Matrix4x4<T>& lhs, Matrix4x4<T>& rhs);

    /**
    *   @brief Subtract two matrices
    *   @param[in] lhs The first matrix
    *   @param[in] rhs The second matrix
    *   @return The elementwise subtraction of these two matrices
    */
    template<typename T>
    __host__ __device__
    Matrix2x2<T> operator-(Matrix4x4<T>& lhs, Matrix4x4<T>& rhs);

    /**
    *   @brief Multiply a matrix with a scalar
    *   @param[in] lhs The matrix
    *   @param[in] rhs The scalar
    *   @return The elementwise product of the matrix with the scalar
    */
    template<typename T>
    __host__ __device__
    Matrix4x4<T> operator*(Matrix4x4<T>& lhs, const T& rhs);

    /**
    *   @brief Multiply a matrix with a scalar
    *   @param[in] lhs The scalar
    *   @param[in] rhs The matrix
    *   @return The elementwise product of the matrix with the scalar
    */
    template<typename T>
    __host__ __device__
    Matrix4x4<T> operator*(const T& lhs, Matrix4x4<T>& rhs);

    /**
    *   @brief Multiply two matrices
    *   @param[in] lhs The first matrix
    *   @param[in] rhs The second matrix
    *   @return Matrix multiplication between two matrices
    */
    template<typename T>
    __host__ __device__
    Matrix4x4<T> operator*(Matrix4x4<T>& lhs, Matrix4x4<T>& rhs);

    /**
    *   @brief Multiply a matrix with a vector
    *   @param[in] lhs The matrix
    *   @param[in] rhs The vector
    *   @return The matrix vector product
    */
    template<typename T>
    __host__ __device__
    Vector4<T> operator*(Matrix4x4<T>& lhs, const Vector4<T>& rhs);

} //namespace cupbr

#include "../../src/Math/MatrixOperationsDetail.h"

#endif