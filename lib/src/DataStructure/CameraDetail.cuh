#ifndef __CUPBR_DATASTRUCTURE_CAMERADETAIL_CUH
#define __CUPBR_DATASTRUCTURE_CAMERADETAIL_CUH

__host__ __device__
inline Vector3float
Camera::position() const
{
    return _position;
}

__host__ __device__
inline Vector3float
Camera::xAxis() const
{
    return _xAxis;
}

__host__ __device__
inline Vector3float
Camera::yAxis() const
{
    return _yAxis;
}

__host__ __device__
inline Vector3float
Camera::zAxis() const
{
    return _zAxis;
}

#endif