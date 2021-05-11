#ifndef __CUPBR_GEOMETRY_TRIANGLEDETAIL_CUH
#define __CUPBR_GEOMETRY_TRIANGLEDETAIL_CUH

#include <Core/CUDA.cuh>
#include <Math/Functions.cuh>

__host__ __device__
inline 
Triangle::Triangle(const Vector3float& vertex1, const Vector3float& vertex2, const Vector3float& vertex3)
    :_vertex1(vertex1),
     _vertex2(vertex2),
     _vertex3(vertex3),
     _normal(Math::normalize(Math::cross(vertex2 - vertex1, vertex3 - vertex1)))
{
    type = GeometryType::TRIANGLE;
}

__host__ __device__
inline Vector4float
Triangle::computeRayIntersection(const Ray& ray)
{
    float PdotN = Math::dot(_vertex1, _normal);
    float OdotN = Math::dot(ray.origin(), _normal);
    float DdotN = Math::dot(ray.direction(), _normal);

    if(Math::safeFloatEqual(DdotN, 0.0f))
    {
        return Vector4float(INFINITY);
    }

    float t = (PdotN - OdotN)/DdotN;

    if(t < 0)return Vector4float(INFINITY);

    return Vector4float(ray.origin() + t*ray.direction(), t);
}

__host__ __device__
inline Vector3float
Triangle::getNormal(const Vector3float& x)
{
    return _normal;
}

#endif