#ifndef __CUPBR_MATH_MATRIXOPERATIONSDETAIL_H
#define __CUPBR_MATH_MATRIXOPERATIONSDETAIL_H


namespace cupbr
{
    template<typename T>
    __host__ __device__
    Matrix2x2<T> operator+(Matrix2x2<T>& lhs, Matrix2x2<T>& rhs)
    {
        Matrix2x2<T> result;

        #define ADD(i,j) result(i,j)  = lhs(i,j) + rhs(i,j)

        ADD(0, 0);
        ADD(0, 1);
        ADD(1, 0);
        ADD(1, 1);

        #undef ADD

        return result;
    }

    template<typename T>
    __host__ __device__
    Matrix2x2<T> operator-(Matrix2x2<T>& lhs, Matrix2x2<T>& rhs)
    {
        Matrix2x2<T> result;

        #define SUB(i,j) result(i,j)  = lhs(i,j) - rhs(i,j)

        SUB(0, 0);
        SUB(0, 1);
        SUB(1, 0);
        SUB(1, 1);

        #undef ADD

        return result;
    }

    template<typename T>
    __host__ __device__
    Matrix2x2<T> operator*(Matrix2x2<T>& lhs, const T& rhs)
    {
        Matrix2x2<T> result;

        #define MUL(i,j) result(i,j) = lhs(i,j) * rhs

        MUL(0, 0);
        MUL(0, 1);
        MUL(1, 0);
        MUL(1, 1);

        #undef MUL

        return result;
    }

    template<typename T>
    __host__ __device__
    Matrix2x2<T> operator*(const T& lhs, Matrix2x2<T>& rhs)
    {
        return operator*(rhs, lhs);
    }

    template<typename T>
    __host__ __device__
    Matrix2x2<T> operator*(Matrix2x2<T>& lhs, const Matrix2x2<T>& rhs)
    {
        Matrix2x2<T> result;

        result(0,0) = lhs(0,0) * rhs(0,0) + lhs(0,1) * rhs(1,0);
        result(0,1) = lhs(0,0) * rhs(0,1) + lhs(0,1) * rhs(1,1);
        result(1,0) = lhs(1,0) * rhs(0,0) + lhs(1,1) * rhs(1,0);
        result(1,1) = lhs(1,0) * rhs(0,1) + lhs(1,1) * rhs(1,1);

        return result;
    }

    template<typename T>
    __host__ __device__
    Vector2<T> operator*(Matrix2x2<T>& lhs, const Vector2<T>& rhs)
    {
        Vector2<T> result;
        result.x = lhs(0,0) * rhs.x + lhs(0,1) * rhs.y;
        result.y = lhs(1,0) * rhs.x + lhs(1,1) * rhs.y;
    }

    template<typename T>
    __host__ __device__
    Matrix3x3<T> operator+(Matrix3x3<T>& lhs, Matrix3x3<T>& rhs)
    {
        Matrix3x3<T> result;

        #define ADD(i,j) result(i,j)  = lhs(i,j) + rhs(i,j)

        ADD(0, 0);
        ADD(0, 1);
        ADD(0, 2);
        ADD(1, 0);
        ADD(1, 1);
        ADD(1, 2);
        ADD(2, 0);
        ADD(2, 1);
        ADD(2, 2);

        #undef ADD

        return result;
    }

    template<typename T>
    __host__ __device__
    Matrix3x3<T> operator-(Matrix3x3<T>& lhs, Matrix3x3<T>& rhs)
    {
        Matrix3x3<T> result;

        #define SUB(i,j) result(i,j)  = lhs(i,j) - rhs(i,j)

        SUB(0, 0);
        SUB(0, 1);
        SUB(0, 2);
        SUB(1, 0);
        SUB(1, 1);
        SUB(1, 2);
        SUB(2, 0);
        SUB(2, 1);
        SUB(2, 2);

        #undef SUB

        return result;
    }

    template<typename T>
    __host__ __device__
    Matrix3x3<T> operator*(Matrix3x3<T>& lhs, const T& rhs)
    {
        Matrix3x3<T> result;

        #define MUL(i,j) result(i,j) = lhs(i,j) * rhs

        MUL(0, 0);
        MUL(0, 1);
        MUL(0, 2);
        MUL(1, 0);
        MUL(1, 1);
        MUL(1, 2);
        MUL(2, 0);
        MUL(2, 1);
        MUL(2, 2);

        #undef MUL

        return result;
    }

    template<typename T>
    __host__ __device__
    Matrix3x3<T> operator*(const T& lhs, Matrix3x3<T>& rhs)
    {
        return operator*(rhs, lhs);
    }

    template<typename T>
    __host__ __device__
    Matrix3x3<T> operator*(Matrix3x3<T>& lhs, Matrix3x3<T>& rhs)
    {
        Matrix3x3<T> result;

        result(0,0) = lhs(0,0) * rhs(0,0) + lhs(0,1) * rhs(1,0) + lhs(0,2) * rhs(2,0);
        result(0,1) = lhs(0,0) * rhs(0,1) + lhs(0,1) * rhs(1,1) + lhs(0,2) * rhs(2,1);
        result(0,2) = lhs(0,0) * rhs(0,2) + lhs(0,1) * rhs(1,2) + lhs(0,2) * rhs(2,2);
        result(1,0) = lhs(1,0) * rhs(0,0) + lhs(1,1) * rhs(1,0) + lhs(1,2) * rhs(2,0);
        result(1,1) = lhs(1,0) * rhs(0,1) + lhs(1,1) * rhs(1,1) + lhs(1,2) * rhs(2,1);
        result(1,2) = lhs(1,0) * rhs(0,2) + lhs(1,1) * rhs(1,2) + lhs(1,2) * rhs(2,2);
        result(2,0) = lhs(2,0) * rhs(0,0) + lhs(2,1) * rhs(1,0) + lhs(2,2) * rhs(2,0);
        result(2,1) = lhs(2,0) * rhs(0,1) + lhs(2,1) * rhs(1,1) + lhs(2,2) * rhs(2,1);
        result(2,2) = lhs(2,0) * rhs(0,2) + lhs(2,1) * rhs(1,2) + lhs(2,2) * rhs(2,2);

        return result;
    }

    template<typename T>
    __host__ __device__
    Vector3<T> operator*(Matrix3x3<T>& lhs, const Vector3<T>& rhs)
    {
        Vector3<T> result;

        result.x = lhs(0,0) * rhs.x + lhs(0,1) * rhs.y + lhs(0,2) * rhs.z;
        result.y = lhs(1,0) * rhs.x + lhs(1,1) * rhs.y + lhs(1,2) * rhs.z;
        result.z = lhs(2,0) * rhs.x + lhs(2,1) * rhs.y + lhs(2,2) * rhs.z;

        return result;
    }

    template<typename T>
    __host__ __device__
    Matrix4x4<T> operator+(Matrix4x4<T>& lhs, Matrix4x4<T>& rhs)
    {
        Matrix4x4<T> result;

        #define ADD(i,j) result(i,j)  = lhs(i,j) + rhs(i,j)

        ADD(0, 0);
        ADD(0, 1);
        ADD(0, 2);
        ADD(0, 3);
        ADD(1, 0);
        ADD(1, 1);
        ADD(1, 2);
        ADD(1, 3);
        ADD(2, 0);
        ADD(2, 1);
        ADD(2, 2);
        ADD(2, 3);
        ADD(3, 0);
        ADD(3, 1);
        ADD(3, 2);
        ADD(3, 3);

        #undef ADD

        return result;
    }

    template<typename T>
    __host__ __device__
    Matrix2x2<T> operator-(Matrix4x4<T>& lhs, Matrix4x4<T>& rhs)
    {
        Matrix4x4<T> result;

        #define SUB(i,j) result(i,j)  = lhs(i,j) - rhs(i,j)

        SUB(0, 0);
        SUB(0, 1);
        SUB(0, 2);
        SUB(0, 3);
        SUB(1, 0);
        SUB(1, 1);
        SUB(1, 2);
        SUB(1, 3);
        SUB(2, 0);
        SUB(2, 1);
        SUB(2, 2);
        SUB(2, 3);
        SUB(3, 0);
        SUB(3, 1);
        SUB(3, 2);
        SUB(3, 3);

        #undef SUB

        return result;
    }

    template<typename T>
    __host__ __device__
    Matrix4x4<T> operator*(Matrix4x4<T>& lhs, const T& rhs)
    {
        Matrix4x4<T> result;

        #define MUL(i,j) result(i,j)  = lhs(i,j) * rhs
        MUL(0, 0);
        MUL(0, 1);
        MUL(0, 2);
        MUL(0, 3);
        MUL(1, 0);
        MUL(1, 1);
        MUL(1, 2);
        MUL(1, 3);
        MUL(2, 0);
        MUL(2, 1);
        MUL(2, 2);
        MUL(2, 3);
        MUL(3, 0);
        MUL(3, 1);
        MUL(3, 2);
        MUL(3, 3);

        #undef MUL

        return result;
    }

    template<typename T>
    __host__ __device__
    Matrix4x4<T> operator*(const T& lhs, Matrix4x4<T>& rhs)
    {
        return operator*(rhs, lhs);
    }

    template<typename T>
    __host__ __device__
    Matrix4x4<T> operator*(Matrix4x4<T>& lhs, Matrix4x4<T>& rhs)
    {
        Matrix4x4<T> result;

        result(0,0) = lhs(0,0) * rhs(0,0) + lhs(0,1) * rhs(1,0) + lhs(0,2) * rhs(2,0) + lhs(0,3) * rhs(3,0);
        result(0,1) = lhs(0,0) * rhs(0,1) + lhs(0,1) * rhs(1,1) + lhs(0,2) * rhs(2,1) + lhs(0,3) * rhs(3,1);
        result(0,2) = lhs(0,0) * rhs(0,2) + lhs(0,1) * rhs(1,2) + lhs(0,2) * rhs(2,2) + lhs(0,3) * rhs(3,2);
        result(0,3) = lhs(0,0) * rhs(0,3) + lhs(0,1) * rhs(1,3) + lhs(0,2) * rhs(2,3) + lhs(0,3) * rhs(3,3);
        result(1,0) = lhs(1,0) * rhs(0,0) + lhs(1,1) * rhs(1,0) + lhs(1,2) * rhs(2,0) + lhs(1,3) * rhs(3,0);
        result(1,1) = lhs(1,0) * rhs(0,1) + lhs(1,1) * rhs(1,1) + lhs(1,2) * rhs(2,1) + lhs(1,3) * rhs(3,1);
        result(1,2) = lhs(1,0) * rhs(0,2) + lhs(1,1) * rhs(1,2) + lhs(1,2) * rhs(2,2) + lhs(1,3) * rhs(3,2);
        result(1,3) = lhs(1,0) * rhs(0,3) + lhs(1,1) * rhs(1,3) + lhs(1,2) * rhs(2,3) + lhs(1,3) * rhs(3,3);
        result(2,0) = lhs(2,0) * rhs(0,0) + lhs(2,1) * rhs(1,0) + lhs(2,2) * rhs(2,0) + lhs(2,3) * rhs(3,0);
        result(2,1) = lhs(2,0) * rhs(0,1) + lhs(2,1) * rhs(1,1) + lhs(2,2) * rhs(2,1) + lhs(2,3) * rhs(3,1);
        result(2,2) = lhs(2,0) * rhs(0,2) + lhs(2,1) * rhs(1,2) + lhs(2,2) * rhs(2,2) + lhs(2,3) * rhs(3,2);
        result(2,3) = lhs(2,0) * rhs(0,3) + lhs(2,1) * rhs(1,3) + lhs(2,2) * rhs(2,3) + lhs(2,3) * rhs(3,3);
        result(3,0) = lhs(3,0) * rhs(0,0) + lhs(3,1) * rhs(1,0) + lhs(3,2) * rhs(2,0) + lhs(3,3) * rhs(3,0);
        result(3,1) = lhs(3,0) * rhs(0,1) + lhs(3,1) * rhs(1,1) + lhs(3,2) * rhs(2,1) + lhs(3,3) * rhs(3,1);
        result(3,2) = lhs(3,0) * rhs(0,2) + lhs(3,1) * rhs(1,2) + lhs(3,2) * rhs(2,2) + lhs(3,3) * rhs(3,2);
        result(3,3) = lhs(3,0) * rhs(0,3) + lhs(3,1) * rhs(1,3) + lhs(3,2) * rhs(2,3) + lhs(3,3) * rhs(3,3);

        return result;
    }

    template<typename T>
    __host__ __device__
    Vector4<T> operator*(Matrix4x4<T>& lhs, const Vector4<T>& rhs)
    {
        Vector4<T> result;

        result.x = lhs(0,0) * rhs.x + lhs(0,1) * rhs.y + lhs(0,2) * rhs.z + lhs(0,3) * rhs.w;
        result.y = lhs(1,0) * rhs.x + lhs(1,1) * rhs.y + lhs(1,2) * rhs.z + lhs(1,3) * rhs.w;
        result.z = lhs(2,0) * rhs.x + lhs(2,1) * rhs.y + lhs(2,2) * rhs.z + lhs(2,3) * rhs.w;
        result.w = lhs(3,0) * rhs.x + lhs(3,1) * rhs.y + lhs(3,2) * rhs.z + lhs(3,3) * rhs.w;

        return result;
    }

} //namespace cupbr

#endif