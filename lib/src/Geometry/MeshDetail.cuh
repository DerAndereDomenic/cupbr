#ifndef __CUPBR_GEOMETRY_MESHDETAIL_CUH
#define __CUPBR_GEOMETRY_MESHDETAIL_CUH

#include <Core/CUDA.cuh>
#include <Math/Functions.cuh>

__host__ __device__
inline 
Mesh::Mesh(Triangle* triangle_buffer, const uint32_t& num_triangles)
    :_triangles(triangle_buffer),
     _num_triangles(num_triangles)
{
    type = GeometryType::MESH;
}

__host__ __device__
inline Vector4float
Mesh::computeRayIntersection(const Ray& ray)
{
    return Vector4float(INFINITY);
}

__host__ __device__
inline Vector3float
Mesh::getNormal(const Vector3float& x)
{
    return _normal;
}

#endif