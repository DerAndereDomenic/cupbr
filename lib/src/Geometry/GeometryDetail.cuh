#ifndef __CUPBR_GEOMETRY_GEOMETRYDETAIL_CUH
#define __CUPBR_GEOMETRY_GEOMETRYDETAIL_CUH

namespace cupbr
{
    __host__ __device__
    inline Vector4float
    Geometry::computeRayIntersection(const Ray& ray)
    {
        return Vector4float(INFINITY);
    }

    __host__ __device__
    inline Vector3float
    Geometry::getNormal(const Vector3float& x)
    {
        return Vector3float(INFINITY);
    }
} //namespace cupbr

#endif