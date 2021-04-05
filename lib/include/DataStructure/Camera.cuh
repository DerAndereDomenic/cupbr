#ifndef __CUPBR_DATASTRUCTURE_CAMERA_CUH
#define __CUPBR_DATASTRUCTURE_CAMERA_CUH

#include <Core/CUDA.cuh>
#include <Math/Vector.h>

class Camera
{
    public:
        Camera() = default;

        __host__ __device__
        Vector3float
        position() const;

        __host__ __device__
        Vector3float
        xAxis() const;

        __host__ __device__
        Vector3float
        yAxis() const;

        __host__ __device__
        Vector3float
        zAxis() const;

    private:
        Vector3float _position = Vector3float(0,0,0);
        Vector3float _xAxis = Vector3float(1,0,0);
        Vector3float _yAxis = Vector3float(0,1,0);
        Vector3float _zAxis = Vector3float(0,0,1);
};

#include "../../src/DataStructure/CameraDetail.cuh"

#endif