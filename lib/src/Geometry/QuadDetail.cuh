#ifndef __CUPBR_GEOMETRY_QUADDETAIL_CUH
#define __CUPBR_GEOMETRY_QUADDETAIL_CUH

#include <Core/CUDA.cuh>
#include <Math/Functions.cuh>

__host__ __device__
inline 
Quad::Quad(const Vector3float& position, const Vector3float& normal, const Vector3float& extend1, const Vector3float& extend2)
    :_position(position),
     _normal(Math::normalize(normal)),
     _extend1(extend1),
     _extend2(extend2)
{
    type = GeometryType::QUAD;
}

__host__ __device__
inline Vector4float
Quad::computeRayIntersection(const Ray& ray)
{
    float PdotN = Math::dot(_position, _normal);
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
Quad::getNormal(const Vector3float& x)
{
    return _normal;
}

__host__ __device__
inline Vector3float
Quad::position()
{
    return _position;
}

#endif