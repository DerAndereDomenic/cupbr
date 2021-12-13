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
    inline uint32_t
    Geometry::id() const
    {
        return _id;
    }

    __host__ __device__
    inline AABB
    Geometry::aabb() const
    {
        return _aabb;
    }

} //namespace cupbr

#endif