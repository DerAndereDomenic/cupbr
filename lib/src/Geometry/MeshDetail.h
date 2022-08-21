#ifndef __CUPBR_GEOMETRY_MESHDETAIL_H
#define __CUPBR_GEOMETRY_MESHDETAIL_H

#include <Core/CUDA.h>
#include <Math/Functions.h>

namespace cupbr
{
    CUPBR_HOST_DEVICE
    inline
    Mesh::Mesh(Triangle* triangle_buffer, const uint32_t& num_triangles, const Vector3float& minimum, const Vector3float& maximum)
        :Geometry(),
        _triangles(triangle_buffer),
        _num_triangles(num_triangles)
    {
        type = GeometryType::MESH;
        _aabb = AABB(minimum, maximum);
    }

    CUPBR_HOST_DEVICE
    inline LocalGeometry
    Mesh::computeRayIntersection(const Ray& ray)
    {
        LocalGeometry geom;

        for (uint32_t i = 0; i < _num_triangles; ++i)
        {
            LocalGeometry intersection_triangle = _triangles[i].computeRayIntersection(ray);
            if (intersection_triangle.depth < geom.depth)
            {
                geom = intersection_triangle;
            }
        }
        geom.type = GeometryType::MESH;
        geom.material = material;
        geom.scene_index = _id;
        return geom;
    }

    CUPBR_HOST_DEVICE
    inline uint32_t
    Mesh::num_triangles()
    {
        return _num_triangles;
    }

    CUPBR_HOST_DEVICE
    inline Triangle*
    Mesh::triangles()
    {
        return _triangles;
    }
} //namespace cupbr

#endif