#ifndef __CUPBR_DATASTRUCTURE_CAMERADETAIL_H
#define __CUPBR_DATASTRUCTURE_CAMERADETAIL_H

namespace cupbr
{
    CUPBR_HOST_DEVICE
    inline Vector3float
    Camera::position() const
    {
        return _position;
    }

    CUPBR_HOST_DEVICE
    inline Vector3float
    Camera::xAxis() const
    {
        return _xAxis;
    }

    CUPBR_HOST_DEVICE
    inline Vector3float
    Camera::yAxis() const
    {
        return _yAxis;
    }

    CUPBR_HOST_DEVICE
    inline Vector3float
    Camera::zAxis() const
    {
        return _zAxis;
    }

    CUPBR_HOST_DEVICE
    inline bool
    Camera::moved() const
    {
        return _moved;
    }

    CUPBR_HOST_DEVICE
    inline float 
    Camera::aspect_ratio() const
    {
        return _aspect_ratio;
    }
} //namespace cupbr

#endif