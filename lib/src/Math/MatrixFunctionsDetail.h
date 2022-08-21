#ifndef __CUPBR_MATH_MATRIXFUNCTIONSDETAIL_H
#define __CUPBR_MATH_MATRIXFUNCTIONSDETAIL_H

namespace cupbr
{
    namespace Math
    {

        template<typename T>
        CUPBR_HOST_DEVICE
        Matrix2x2<T> transpose(Matrix2x2<T>& M)
        {
            Matrix2x2<T> result;

            #define TRANSPOSE(i,j) result(i,j) = M(j,i);

            TRANSPOSE(0, 0);
            TRANSPOSE(0, 1);
            TRANSPOSE(1, 0);
            TRANSPOSE(1, 1);

            #undef TRANSPOSE

            return result;
        }

        template<typename T>
        CUPBR_HOST_DEVICE
        Matrix2x2<T> inverse(Matrix2x2<T>& M)
        {
            Matrix2x2<T> result;

            T invDet = 1.0f / det(M);

            result(0,0) =  M(1,1) * invDet;
            result(1,1) =  M(0,0) * invDet;
            result(0,1) = -M(0,1) * invDet;
            result(1,0) = -M(1,0) * invDet;

            return result;
        }

        template<typename T>
        CUPBR_HOST_DEVICE
        T det(const Matrix2x2<T>& M)
        {
            T result = M(0,0) * M(1,1) - M(0,1) * M(1,0);

            return result;
        }

        template<typename T>
        CUPBR_HOST_DEVICE
        Matrix3x3<T> transpose(Matrix3x3<T>& M)
        {
            Matrix3x3<T> result;

            #define TRANSPOSE(i,j) result(i,j) = M(j,i);

            TRANSPOSE(0, 0);
            TRANSPOSE(0, 1);
            TRANSPOSE(0, 2);
            TRANSPOSE(1, 0);
            TRANSPOSE(1, 1);
            TRANSPOSE(1, 2);
            TRANSPOSE(2, 0);
            TRANSPOSE(2, 1);
            TRANSPOSE(2, 2);

            #undef TRANSPOSE

            return result;
        }

        template<typename T>
        CUPBR_HOST_DEVICE
        Matrix3x3<T> inverse(Matrix3x3<T>& M)
        {
            Matrix3x3<T> result;
            T invDet = 1.0f / det(M);

            result(0, 0) = (M(1, 1) * M(2, 2) - M(2, 1) * M(1, 2)) * invDet;
            result(0, 1) = (M(0, 2) * M(2, 1) - M(0, 1) * M(2, 2)) * invDet;
            result(0, 2) = (M(0, 1) * M(1, 2) - M(0, 2) * M(1, 1)) * invDet;
            result(1, 0) = (M(1, 2) * M(2, 0) - M(1, 0) * M(2, 2)) * invDet;
            result(1, 1) = (M(0, 0) * M(2, 2) - M(0, 2) * M(2, 0)) * invDet;
            result(1, 2) = (M(1, 0) * M(0, 2) - M(0, 0) * M(1, 2)) * invDet;
            result(2, 0) = (M(1, 0) * M(2, 1) - M(2, 0) * M(1, 1)) * invDet;
            result(2, 1) = (M(2, 0) * M(0, 1) - M(0, 0) * M(2, 1)) * invDet;
            result(2, 2) = (M(0, 0) * M(1, 1) - M(1, 0) * M(0, 1)) * invDet;

            return result;
        }

        template<typename T>
        CUPBR_HOST_DEVICE
        T det(Matrix3x3<T>& M)
        {
            T result =    M(0, 0) * M(1, 1) * M(2, 2)
                        + M(0, 1) * M(1, 2) * M(2, 0)
                        + M(0, 2) * M(1, 0) * M(2, 1)
                        - M(0, 2) * M(1, 1) * M(2, 0)
                        - M(0, 1) * M(1, 0) * M(2, 2)
                        - M(0, 0) * M(1, 2) * M(2, 1);

            return result;
        }

        template<typename T>
        CUPBR_HOST_DEVICE
        Matrix4x4<T> transpose(Matrix4x4<T>& M)
        {
            Matrix4x4<T> result;

            #define TRANSPOSE(i,j) result(i,j) = M(j,i);

            TRANSPOSE(0, 0);
            TRANSPOSE(0, 1);
            TRANSPOSE(0, 2);
            TRANSPOSE(0, 3);
            TRANSPOSE(1, 0);
            TRANSPOSE(1, 1);
            TRANSPOSE(1, 2);
            TRANSPOSE(1, 3);
            TRANSPOSE(2, 0);
            TRANSPOSE(2, 1);
            TRANSPOSE(2, 2);
            TRANSPOSE(2, 3);
            TRANSPOSE(3, 0);
            TRANSPOSE(3, 1);
            TRANSPOSE(3, 2);
            TRANSPOSE(3, 3);

            #undef TRANSPOSE

            return result;
        }

        template<typename T>
        CUPBR_HOST_DEVICE
        T det(Matrix4x4<T>& M)
        {
            float result = M(0, 3) * M(1, 2) * M(2, 1) * M(3, 0)
                         - M(0, 2) * M(1, 3) * M(2, 1) * M(3, 0)
                         - M(0, 3) * M(1, 1) * M(2, 2) * M(3, 0)
                         + M(0, 1) * M(1, 3) * M(2, 2) * M(3, 0)
                         + M(0, 2) * M(1, 1) * M(2, 3) * M(3, 0)
                         - M(0, 1) * M(1, 2) * M(2, 3) * M(3, 0)
                         - M(0, 3) * M(1, 2) * M(2, 0) * M(3, 1)
                         + M(0, 2) * M(1, 3) * M(2, 0) * M(3, 1)
                         + M(0, 3) * M(1, 0) * M(2, 2) * M(3, 1)
                         - M(0, 0) * M(1, 3) * M(2, 2) * M(3, 1)
                         - M(0, 2) * M(1, 0) * M(2, 3) * M(3, 1)
                         + M(0, 0) * M(1, 2) * M(2, 3) * M(3, 1)
                         + M(0, 3) * M(1, 1) * M(2, 0) * M(3, 2)
                         - M(0, 1) * M(1, 3) * M(2, 0) * M(3, 2)
                         - M(0, 3) * M(1, 0) * M(2, 1) * M(3, 2)
                         + M(0, 0) * M(1, 3) * M(2, 1) * M(3, 2)
                         + M(0, 1) * M(1, 0) * M(2, 3) * M(3, 2)
                         - M(0, 0) * M(1, 1) * M(2, 3) * M(3, 2)
                         - M(0, 2) * M(1, 1) * M(2, 0) * M(3, 3)
                         + M(0, 1) * M(1, 2) * M(2, 0) * M(3, 3)
                         + M(0, 2) * M(1, 0) * M(2, 1) * M(3, 3)
                         - M(0, 0) * M(1, 2) * M(2, 1) * M(3, 3)
                         - M(0, 1) * M(1, 0) * M(2, 2) * M(3, 3)
                         + M(0, 0) * M(1, 1) * M(2, 2) * M(3, 3);

            return result;
        }

    } //namespace Math
} //namespace cupbr

#endif