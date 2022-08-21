#ifndef __CUPBR_GEOMETRY_GEOMETRYDETAIL_H
#define __CUPBR_GEOMETRY_GEOMETRYDETAIL_H

namespace cupbr
{
    CUPBR_HOST_DEVICE
    inline LocalGeometry
    Geometry::computeRayIntersection(const Ray& ray)
    {
        return LocalGeometry();
    }

    CUPBR_HOST_DEVICE
    inline uint32_t
    Geometry::id() const
    {
        return _id;
    }

    CUPBR_HOST_DEVICE
    inline AABB
    Geometry::aabb() const
    {
        return _aabb;
    }

} //namespace cupbr

#endif