#ifndef __CUPBR_GEOMETRY_PLANEDETAIL_CUH
#define __CUPBR_GEOMETRY_PLANEDETAIL_CUH

#include <Math/Functions.cuh>

__host__ __device__
inline 
Plane::Plane(const Vector3float& position, const Vector3float& normal)
    :_position(position),
     _normal(Math::normalize(normal))
{

}

__host__ __device__
inline Vector4float
Plane::computeRayIntersection(const Ray& ray)
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
Plane::getNormal(const Vector3float& x)
{
    return _normal;
}

__host__ __device__
inline Vector3float
Plane::position()
{
    return _position;
}

#endif