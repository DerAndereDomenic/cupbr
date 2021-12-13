#ifndef __CUPBR_GEOMETRY_GEOMETRYDETAIL_H
#define __CUPBR_GEOMETRY_GEOMETRYDETAIL_H

namespace cupbr
{
    __host__ __device__
    inline LocalGeometry
    Geometry::computeRayIntersection(const Ray& ray)
    {
        return LocalGeometry();
    }

    __host__ __device__
    inline Vector3float
    Geometry::getNormal(const Vector3float& x)
    {
        return Vector3float(INFINITY);
    }

    __host__ __device__
    inline uint32_t
    Geometry::id() const
    {
        return _id;
    }

} //namespace cupbr

#endif