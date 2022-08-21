#ifndef __CUPBR_MATH_MATRIXTYPESDETAIL_H
#define __CUPBR_MATH_MATRIXTYPESDETAIL_H


namespace cupbr
{
    template<typename T>
    CUPBR_HOST_DEVICE
    Matrix2x2<T>::Matrix2x2(const Vector2<T>& col1, const Vector2<T>& col2)
    {
        data[0][0] = col1.x;
        data[1][0] = col1.y;
        data[0][1] = col2.x;
        data[1][1] = col2.y;
    }

    template<typename T>
    CUPBR_HOST_DEVICE
    Matrix3x3<T>::Matrix3x3(const Vector3<T>& col1, const Vector3<T>& col2, const Vector3<T>& col3)
    {
        data[0][0] = col1.x;
        data[1][0] = col1.y;
        data[2][0] = col1.z;
        data[0][1] = col2.x;
        data[1][1] = col2.y;
        data[2][1] = col2.z;
        data[0][2] = col3.x;
        data[1][2] = col3.y;
        data[2][2] = col3.z;
    }

    template<typename T>
    CUPBR_HOST_DEVICE
    Matrix4x4<T>::Matrix4x4(const Vector4<T>& col1, const Vector4<T>& col2, const Vector4<T>& col3, const Vector4<T>& col4)
    {
        data[0][0] = col1.x;
        data[1][0] = col1.y;
        data[2][0] = col1.z;
        data[3][0] = col1.w;
        data[0][1] = col2.x;
        data[1][1] = col2.y;
        data[2][1] = col2.z;
        data[3][1] = col2.w;
        data[0][2] = col3.x;
        data[1][2] = col3.y;
        data[2][2] = col3.z;
        data[3][2] = col3.w;
        data[0][3] = col4.x;
        data[1][3] = col4.y;
        data[2][3] = col4.z;
        data[3][3] = col4.w;
    }

} //namespace cupbr

#endif