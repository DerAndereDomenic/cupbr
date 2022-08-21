#ifndef __CUPBR_MATH_MATRIXFUNCTIONS_H
#define __CUPBR_MATH_MATRIXFUNCTIONS_H

#include <Math/MatrixTypes_fwd.h>
#include <Core/CUDA.h>

namespace cupbr
{
    namespace Math
    {
        /**
        *   @brief Transposes a matrix
        *   @return The transposed matrix
        */
        template<typename T>
        CUPBR_HOST_DEVICE
        Matrix2x2<T> transpose(Matrix2x2<T>& M);

        /**
        *   @brief Inverts a matrix
        *   @return The inverted matrix
        */
        template<typename T>
        CUPBR_HOST_DEVICE
        Matrix2x2<T> inverse(Matrix2x2<T>& M);

        /**
        *   @brief Computes the determinant of a matrix
        *   @return The determinant
        */
        template<typename T>
        CUPBR_HOST_DEVICE
        T det(Matrix2x2<T>& M);

        /**
        *   @brief Transposes a matrix
        *   @return The transposed matrix
        */
        template<typename T>
        CUPBR_HOST_DEVICE
        Matrix3x3<T> transpose(Matrix3x3<T>& M);

        /**
        *   @brief Inverts a matrix
        *   @return The inverted matrix
        */
        template<typename T>
        CUPBR_HOST_DEVICE
        Matrix3x3<T> inverse(Matrix3x3<T>& M);

        /**
        *   @brief Computes the determinant of a matrix
        *   @return The determinant
        */
        template<typename T>
        CUPBR_HOST_DEVICE
        T det(Matrix3x3<T>& M);

        /**
        *   @brief Transposes a matrix
        *   @return The transposed matrix
        */
        template<typename T>
        CUPBR_HOST_DEVICE
        Matrix4x4<T> transpose(Matrix4x4<T>& M);

        /**
        *   @brief Inverts a matrix
        *   @return The inverted matrix
        */
        //template<typename T>
        //CUPBR_HOST_DEVCE
        //Matrix4x4<T> inverse(const Matrix4x4<T>& M);

        /**
        *   @brief Computes the determinant of a matrix
        *   @return The determinant
        */
        template<typename T>
        CUPBR_HOST_DEVICE
        T det(Matrix4x4<T>& M);

    } //namespace Math
} //namespace cupbr

#include "../../src/Math/MatrixFunctionsDetail.h"

#endif