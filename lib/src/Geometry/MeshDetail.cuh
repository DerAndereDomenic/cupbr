#ifndef __CUPBR_GEOMETRY_MESHDETAIL_CUH
#define __CUPBR_GEOMETRY_MESHDETAIL_CUH

#include <Core/CUDA.cuh>
#include <Math/Functions.cuh>

namespace cupbr
{
    __host__ __device__
    inline
    Mesh::Mesh(Triangle* triangle_buffer, const uint32_t& num_triangles)
        :_triangles(triangle_buffer),
        _normal(0),
        _num_triangles(num_triangles)
    {
        type = GeometryType::MESH;
    }

    __host__ __device__
    inline Vector4float
    Mesh::computeRayIntersection(const Ray& ray)
    {
        Vector4float intersection(INFINITY);
        Vector3float normal = 0;
        for (uint32_t i = 0; i < _num_triangles; ++i)
        {
            Vector4float intersection_triangle = _triangles[i].computeRayIntersection(ray);
            if (intersection_triangle.w < intersection.w)
            {
                intersection = intersection_triangle;
                normal = _triangles[i].getNormal(Vector3float(intersection));
            }
        }

        //TODO: HERE IS A RACE CONDITION
        _normal = Vector3float(normal);
        return intersection;
    }

    __host__ __device__
    inline Vector3float
    Mesh::getNormal(const Vector3float& x)
    {
        return _normal;
    }

    __host__ __device__
    inline uint32_t
    Mesh::num_triangles()
    {
        return _num_triangles;
    }

    __host__ __device__
    inline Triangle*
    Mesh::triangles()
    {
        return _triangles;
    }
} //namespace cupbr

#endif