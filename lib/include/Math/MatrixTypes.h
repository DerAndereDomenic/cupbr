#ifndef __CUPBR_MATH_MATRIXTYPES_H
#define __CUPBR_MATH_MATRIXTYPES_H

#include <Math/MatrixTypes_fwd.h>
#include <Math/VectorTypes.h>
#include <Core/CUDA.h>

namespace cupbr
{
    /**
    *	@brief A 2x2 matrix
    */
    template<typename T>
    struct Matrix2x2
    {
        public:
        /**
        *	@brief Default constructor.
        */
        Matrix2x2() = default;

        /**
        *	@brief Creates a 2x2 matrix.
        *	@param[in] col1 The first column
        *	@param[in] col2 The second column
        */
        CUPBR_HOST_DEVICE
        Matrix2x2(const Vector2<T>& col1, const Vector2<T>& col2);

        /**
        *   @brief Get an element of the matrix
        *   @param[in] row The row index
        *   @param[in] col The col index
        */
        CUPBR_HOST_DEVICE
        inline T&
        operator()(const uint32_t& row, const uint32_t& col)
        {
            return data[row][col];
        }

        T data[2][2];
        
    };

    /**
    *	@brief A 3x3 matrix
    */
    template<typename T>
    struct Matrix3x3
    {
        public:
        /**
        *	@brief Default constructor.
        */
        Matrix3x3() = default;

        /**
        *	@brief Creates a 3x3 matrix.
        *	@param[in] col1 The first column
        *	@param[in] col2 The second column
        *   @param[in] col3 The third column
        * 
        */
        CUPBR_HOST_DEVICE
        Matrix3x3(const Vector3<T>& col1, const Vector3<T>& col2, const Vector3<T>& col3);

        /**
        *   @brief Get an element of the matrix
        *   @param[in] row The row index
        *   @param[in] col The col index
        */
        CUPBR_HOST_DEVICE
        inline T&
        operator()(const uint32_t& row, const uint32_t& col)
        {
            return data[row][col];
        }

        T data[3][3];
        
    };

    /**
    *	@brief A 4x4 matrix
    */
    template<typename T>
    struct Matrix4x4
    {
        public:
        /**
        *	@brief Default constructor.
        */
        Matrix4x4() = default;

        /**
        *	@brief Creates a 2x2 matrix.
        *	@param[in] col1 The first column
        *	@param[in] col2 The second column
        *   @param[in] col3 The third column
        *   @param[in] col4 The fourth column
        */
        CUPBR_HOST_DEVICE
        Matrix4x4(const Vector4<T>& col1, const Vector4<T>& col2, const Vector4<T>& col3, const Vector4<T>& col4);

        /**
        *   @brief Get an element of the matrix
        *   @param[in] row The row index
        *   @param[in] col The col index
        */
        CUPBR_HOST_DEVICE
        inline T&
        operator()(const uint32_t& row, const uint32_t& col)
        {
            return data[row][col];
        }

        T data[4][4];
        
    };

} //namespace cupbr

#include "../../src/Math/MatrixTypesDetail.h"

#endif